// DEF-tuni scaffold: fused TUNI block kernel (PRD-006-C)

#include <cstdint>

extern "C" __global__ void tuni_block_forward_stub(const float* in_ptr,
                                                   float* out_ptr,
                                                   int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out_ptr[idx] = in_ptr[idx];
  }
}

