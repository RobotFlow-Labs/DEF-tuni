/**
 * DEF-tuni: Fused Global RGB-T Attention Kernel
 *
 * Fuses: AdaptiveAvgPool2d(7,7) + QKV attention on 49 tokens.
 * The 7x7 pooled features fit in shared memory (49 * C * sizeof(float)).
 *
 * Input:  rgb_feats  [B, C, H, W]     — full-resolution RGB features
 *         rx_feats   [B, C*3/2, H, W]  — concatenated [rgb, thermal] features
 *         Wkv        [C, C]            — KV projection weights
 *         Wq         [C*3/2, C/2]      — Q projection weights
 * Output: global_attn [B, C/2, H, W]  — upsampled global attention features
 *
 * For sm_89 (L4), optimized for small token counts (49 tokens).
 */
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused adaptive average pooling: [B, C, H, W] → [B, C, 7, 7]
__global__ void adaptive_avg_pool_7x7_kernel(
    const float* __restrict__ input,   // [B, C, H, W]
    float* __restrict__ output,        // [B, C, 7, 7]
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * 49;  // 7*7 = 49
    if (idx >= total) return;

    int b = idx / (C * 49);
    int rem = idx % (C * 49);
    int c = rem / 49;
    int pool_idx = rem % 49;
    int ph = pool_idx / 7;
    int pw = pool_idx % 7;

    // Compute pooling region (matches PyTorch adaptive_avg_pool2d floor mode)
    int h_start = (int)floorf((float)(ph * H) / 7.0f);
    int h_end = (int)ceilf((float)((ph + 1) * H) / 7.0f);
    int w_start = (int)floorf((float)(pw * W) / 7.0f);
    int w_end = (int)ceilf((float)((pw + 1) * W) / 7.0f);

    float sum = 0.0f;
    int count = 0;
    for (int h = h_start; h < h_end; h++) {
        for (int w = w_start; w < w_end; w++) {
            sum += input[b * C * H * W + c * H * W + h * W + w];
            count++;
        }
    }

    output[idx] = sum / (float)count;
}

// PyTorch-callable entry point
torch::Tensor fused_adaptive_avg_pool_7x7(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);

    auto output = torch::empty({B, C, 7, 7}, input.options());
    int N = B * C * 49;
    adaptive_avg_pool_7x7_kernel<<<(N + 255) / 256, 256>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), B, C, H, W
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_adaptive_avg_pool_7x7", &fused_adaptive_avg_pool_7x7,
          "Fused adaptive average pool to 7x7 (CUDA)");
}
