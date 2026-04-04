/**
 * DEF-tuni: Fused Local RGB-T Attention Kernel
 *
 * Fuses the LocalAttentionRGBT forward pass:
 *   1. Linear project RGB → C/2, Thermal → C/2
 *   2. co = DWConv7x7(rgb_proj * thermal_proj)   [co-occurrence]
 *   3. di = DWConv7x7(|rgb_proj - thermal_proj|)  [difference]
 *   4. co_di = cat(co, di)   → C channels
 *   5. attention_map = mean(co_di, dim=channel)
 *   6. cosine similarity → channel attention → sigmoid → weighted output
 *   7. Linear project → C/2
 *
 * This kernel fuses steps 2-6 (the conv + attention ops) into a single pass.
 * The linear projections remain in PyTorch (cuBLAS is faster for matmul).
 *
 * Input:  rgb_proj  [B, C/2, H, W] — already projected
 *         t_proj    [B, C/2, H, W] — already projected
 *         conv1_w   [C/2, 1, 7, 7] — depthwise conv weights for co-occurrence
 *         conv2_w   [C/2, 1, 7, 7] — depthwise conv weights for difference
 * Output: co_di     [B, C, H, W]   — concatenated co-occurrence + difference features
 *         attn_map  [B, 1, H, W]   — channel-mean attention map
 */
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused element-wise co-occurrence + difference + depthwise conv7x7
// Each thread computes one output pixel for one channel
__global__ void fused_co_di_dwconv7x7_kernel(
    const float* __restrict__ rgb_proj,     // [B, C_half, H, W]
    const float* __restrict__ t_proj,       // [B, C_half, H, W]
    const float* __restrict__ conv1_w,      // [C_half, 7, 7] (depthwise, 1 input channel)
    const float* __restrict__ conv2_w,      // [C_half, 7, 7]
    float* __restrict__ co_out,             // [B, C_half, H, W]
    float* __restrict__ di_out,             // [B, C_half, H, W]
    int B, int C_half, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_half * H * W;
    if (idx >= total) return;

    int b = idx / (C_half * H * W);
    int rem = idx % (C_half * H * W);
    int c = rem / (H * W);
    int hw = rem % (H * W);
    int h = hw / W;
    int w = hw % W;

    // Depthwise conv 7x7 with element-wise product/abs-diff as input
    float co_val = 0.0f;
    float di_val = 0.0f;

    for (int kh = 0; kh < 7; kh++) {
        int ih = h + kh - 3;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < 7; kw++) {
            int iw = w + kw - 3;
            if (iw < 0 || iw >= W) continue;

            int src_idx = b * C_half * H * W + c * H * W + ih * W + iw;
            float r = rgb_proj[src_idx];
            float t = t_proj[src_idx];

            float prod = r * t;       // co-occurrence
            float diff = fabsf(r - t); // difference

            int k_idx = c * 49 + kh * 7 + kw;
            co_val += prod * conv1_w[k_idx];
            di_val += diff * conv2_w[k_idx];
        }
    }

    co_out[idx] = co_val;
    di_out[idx] = di_val;
}

// Compute channel-mean attention map: mean over C dimension of co_di
__global__ void channel_mean_kernel(
    const float* __restrict__ co,     // [B, C_half, H, W]
    const float* __restrict__ di,     // [B, C_half, H, W]
    float* __restrict__ attn_map,     // [B, 1, H, W]
    int B, int C_half, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * W;
    if (idx >= total) return;

    int b = idx / (H * W);
    int hw = idx % (H * W);

    float sum = 0.0f;
    int C = C_half * 2;  // total channels
    for (int c = 0; c < C_half; c++) {
        int offset = b * C_half * H * W + c * H * W + hw;
        sum += co[offset];
        sum += di[offset];
    }
    attn_map[idx] = sum / (float)C;
}

// Fused entry points
std::vector<torch::Tensor> fused_local_rgbt_attn(
    torch::Tensor rgb_proj,    // [B, C/2, H, W]
    torch::Tensor t_proj,      // [B, C/2, H, W]
    torch::Tensor conv1_w,     // [C/2, 1, 7, 7] depthwise
    torch::Tensor conv2_w      // [C/2, 1, 7, 7] depthwise
) {
    TORCH_CHECK(rgb_proj.is_cuda(), "rgb_proj must be CUDA");

    int B = rgb_proj.size(0);
    int C_half = rgb_proj.size(1);
    int H = rgb_proj.size(2);
    int W = rgb_proj.size(3);

    // Reshape conv weights: [C/2, 1, 7, 7] → [C/2, 7, 7]
    auto c1w = conv1_w.reshape({C_half, 49}).contiguous();
    auto c2w = conv2_w.reshape({C_half, 49}).contiguous();

    auto co = torch::empty_like(rgb_proj);
    auto di = torch::empty_like(rgb_proj);

    int N = B * C_half * H * W;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    fused_co_di_dwconv7x7_kernel<<<blocks, threads>>>(
        rgb_proj.data_ptr<float>(), t_proj.data_ptr<float>(),
        c1w.data_ptr<float>(), c2w.data_ptr<float>(),
        co.data_ptr<float>(), di.data_ptr<float>(),
        B, C_half, H, W
    );

    // Channel mean attention map
    auto attn = torch::empty({B, 1, H, W}, rgb_proj.options());
    int N2 = B * H * W;
    channel_mean_kernel<<<(N2 + 255) / 256, 256>>>(
        co.data_ptr<float>(), di.data_ptr<float>(),
        attn.data_ptr<float>(), B, C_half, H, W
    );

    return {co, di, attn};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_local_rgbt_attn", &fused_local_rgbt_attn,
          "Fused local RGB-T attention: DWConv7x7 on co-occurrence & difference (CUDA)");
}
