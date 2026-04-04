// DEF-tuni scaffold: fused LocalAttentionRGBT kernel (PRD-006-A)
// Reference math from repositories/TUNI/backbone_model/TUNI.py::LocalAttentionRGBT

#include <cstdint>

extern "C" __global__ void local_rgbt_attn_forward_stub(const float* rgb,
                                                        const float* thermal,
                                                        float* out,
                                                        int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Placeholder pass-through for scaffold wiring.
    out[idx] = rgb[idx] + thermal[idx];
  }
}

