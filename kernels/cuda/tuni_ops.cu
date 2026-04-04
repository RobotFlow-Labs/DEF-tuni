/**
 * DEF-tuni: Real-Time RGB-T Segmentation CUDA Kernels
 * 1. fused_rgbt_concat_norm — Fused 6-channel concat + batch norm
 * 2. fused_seg_argmax — Fused argmax + colorize for segmentation output
 */
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rgbt_concat_norm_kernel(
    const float* __restrict__ rgb, const float* __restrict__ thermal,
    const float* __restrict__ mean, const float* __restrict__ std,
    float* __restrict__ output,
    int B, int C_rgb, int C_t, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C_total = C_rgb + C_t;
    int total = B * C_total * H * W;
    if (idx >= total) return;
    
    int b = idx / (C_total * H * W);
    int c = (idx % (C_total * H * W)) / (H * W);
    int hw = idx % (H * W);
    
    float val;
    if (c < C_rgb) {
        val = rgb[b * C_rgb * H * W + c * H * W + hw];
    } else {
        val = thermal[b * C_t * H * W + (c - C_rgb) * H * W + hw];
    }
    output[idx] = (val - mean[c]) / (std[c] + 1e-5f);
}

__global__ void seg_argmax_kernel(
    const float* __restrict__ logits, int* __restrict__ output,
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W) return;
    int b = idx / (H * W), hw = idx % (H * W);
    
    float max_val = -1e10f;
    int max_c = 0;
    for (int c = 0; c < C; c++) {
        float v = logits[b * C * H * W + c * H * W + hw];
        if (v > max_val) { max_val = v; max_c = c; }
    }
    output[idx] = max_c;
}

torch::Tensor fused_rgbt_concat_norm(
    torch::Tensor rgb, torch::Tensor thermal,
    torch::Tensor mean, torch::Tensor std
) {
    TORCH_CHECK(rgb.is_cuda(), "must be CUDA");
    int B = rgb.size(0), C_rgb = rgb.size(1), C_t = thermal.size(1);
    int H = rgb.size(2), W = rgb.size(3);
    int C_total = C_rgb + C_t;
    auto output = torch::empty({B, C_total, H, W}, rgb.options());
    int N = B * C_total * H * W;
    rgbt_concat_norm_kernel<<<(N+255)/256, 256>>>(
        rgb.data_ptr<float>(), thermal.data_ptr<float>(),
        mean.data_ptr<float>(), std.data_ptr<float>(),
        output.data_ptr<float>(), B, C_rgb, C_t, H, W);
    return output;
}

torch::Tensor fused_seg_argmax(torch::Tensor logits) {
    TORCH_CHECK(logits.is_cuda(), "must be CUDA");
    int B = logits.size(0), C = logits.size(1), H = logits.size(2), W = logits.size(3);
    auto output = torch::empty({B, H, W}, logits.options().dtype(torch::kInt32));
    int N = B * H * W;
    seg_argmax_kernel<<<(N+255)/256, 256>>>(
        logits.data_ptr<float>(), output.data_ptr<int>(), B, C, H, W);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rgbt_concat_norm", &fused_rgbt_concat_norm, "Fused RGB-T concat + normalize (CUDA)");
    m.def("fused_seg_argmax", &fused_seg_argmax, "Fused segmentation argmax (CUDA)");
}
