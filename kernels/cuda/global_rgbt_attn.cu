// DEF-tuni scaffold: fused pooled global RGB-T attention kernel (PRD-006-B)

#include <cstdint>

extern "C" __global__ void global_rgbt_attn_forward_stub(const float* rgb,
                                                         const float* thermal,
                                                         float* out,
                                                         int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = 0.5f * (rgb[idx] + thermal[idx]);
  }
}

