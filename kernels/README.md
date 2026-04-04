# Custom Kernels — DEF-tuni
# TUNI: Real-time RGB-T Semantic Segmentation CUDA/MLX Kernels
# Architecture: Unified encoder with 3 attention types per block
# Following /anima-optimize-cuda-pipeline Phase 3

## Architecture-Specific Kernel Targets

NOTE: This repo is pure PyTorch (no C++/CUDA extensions in the repo).
All optimization is about FUSING the many small PyTorch ops in the
unified encoder's triple-attention blocks into efficient kernels.

### Kernel 1: Fused LocalAttentionRGBT (`local_rgbt_attn.cu`)
**Bottleneck**: LocalAttentionRGBT runs ~15 separate CUDA ops per block, 10 blocks total = 150+ kernel launches
**Current**: 2× Linear project → permute → element multiply + abs diff → 2× DWConv7x7 → concat → mean → cosine similarity → channel attention → FC → sigmoid → weighted output
**Target**: 3-5x speedup per LocalAttentionRGBT call

```
Input: rgb_feat (B×C×H×W), thermal_feat (B×C/2×H×W),
       W_rgb_proj (C×C, Linear), W_thermal_proj (C/2×C/2, Linear),
       DWConv7x7_co_weight (C×1×7×7), DWConv7x7_di_weight (C×1×7×7),
       FC_channel_attn (2C×C)
Output: local_rgbt_out (B×C×H×W)

Method: Single fused kernel per block
  1. Linear project: rgb_proj = W_rgb @ permute(rgb), thermal_proj = W_thermal @ permute(thermal)
  2. Co-occurrence: co_raw = rgb_proj * thermal_proj [element-wise, shared memory]
  3. Difference: di_raw = |rgb_proj - thermal_proj| [element-wise, same pass]
  4. DWConv7x7 on co_raw → co [depthwise conv in shared memory, 7×7 tile]
  5. DWConv7x7 on di_raw → di [same kernel, different weights]
  6. Concat: cat_feat = [co, di] → (B, 2C, H, W)
  7. Mean attention map: attn = mean(cat_feat, dim=1) → (B, 1, H, W) [warp reduction]
  8. Cosine similarity: cos_sim[c] = dot(co[:,c,:,:], di[:,c,:,:]) / (||co[:,c]|| * ||di[:,c]||)
     [per-channel, warp-level dot product + norm]
  9. Channel attention: gate = sigmoid(FC(cos_sim)) → (B, C, 1, 1)
  10. Output: out = gate * cat_feat[:, :C, :, :] [broadcast multiply]

  All steps in single kernel — no intermediate B×C×H×W allocations
  Steps 2-3 share same memory access pattern → fused into single pass
  Steps 4-5 share same tile pattern → 2 DWConvs in single kernel call
  Steps 7-9 are reductions → single warp reduction pass

Backward: gradient through sigmoid, FC, cosine similarity, DWConv, multiply all fused
```

**Python wrapper**: `local_rgbt_attn(rgb_feat, thermal_feat, W_rgb, W_thermal, dwconv_co, dwconv_di, fc_attn)` → out
**Save to**: `/mnt/forge-data/shared_infra/cuda_extensions/local_rgbt_attn/`
**REUSABLE by**: Any model with element-wise cross-modal attention + DWConv gating pattern

### Kernel 2: Fused RGB-T Global Attention (`global_rgbt_attn.cu`)
**Bottleneck**: AdaptiveAvgPool(7×7) → concat → QKV → attention → upsample — 8+ ops per block
**Current**: Separate pool, reshape, matmul, softmax, matmul, upsample kernels
**Target**: 2-3x speedup per global attention call

```
Input: rgb_feat (B×C×H×W), thermal_feat (B×C/2×H×W),
       W_q (3C/2×d_k), W_k (C×d_k), W_v (C×d_v),
       W_proj (d_v×C)
Output: global_out (B×C×H×W)

Method: Fused pool → attention → upsample
  1. AdaptiveAvgPool2d(7×7):
     - rgb_pooled (B, C, 7, 7) — block-level average pooling in shared memory
     - thermal_pooled (B, C/2, 7, 7)
  2. Concat: q_input = cat(rgb_pooled, thermal_pooled) → (B, 3C/2, 7, 7)
  3. Flatten: q_input → (B, 49, 3C/2), kv_input = rgb_pooled → (B, 49, C)
  4. QKV: Q = q_input @ W_q, K = kv_input @ W_k, V = kv_input @ W_v
     [3 GEMMs, can be batched as single GEMM with split]
  5. Attention: A = softmax(Q @ K^T / sqrt(d_k)) @ V [flash attention on 49×49 — fits in shared memory!]
  6. Project: out_49 = A @ W_proj → (B, 49, C)
  7. Reshape: out_7x7 = reshape(out_49) → (B, C, 7, 7)
  8. Bilinear upsample: out = upsample(out_7x7, (H, W)) [custom bilinear with precomputed weights]

  Key insight: 49 tokens (7×7) is TINY — entire attention fits in shared memory
  No need for flash attention tiling — single warp handles full attention
  Pool + attention: shared memory holds both pooled features + attention matrix
  Upsample: precompute bilinear weights for H×W → store as constant memory

Backward: gradient through bilinear upsample + attention + pool — all fused
```

**Python wrapper**: `global_rgbt_attn(rgb_feat, thermal_feat, W_q, W_k, W_v, W_proj, output_size)` → out
**Save to**: `/mnt/forge-data/shared_infra/cuda_extensions/global_rgbt_attn/`
**REUSABLE by**: Any model using pooled cross-attention with spatial upsample

### Kernel 3: Fused TUNI Block (`tuni_block.cu`)
**Bottleneck**: Each block has 3 attentions + concat + project + MLP + layer scale + DropPath = 20+ ops
**Current**: Sequential PyTorch ops with many intermediate allocations
**Target**: 1.5-2x speedup per block (beyond individual kernel fusion)

```
Input: rgb_feat (B×C×H×W), thermal_feat (B×C/2×H×W), all_weights, training (bool)
Output: rgb_out (B×C×H×W)

Method: Mega-kernel combining Kernels 1+2 + RGB-RGB local + MLP
  1. RGB-RGB Local: self_attn = x * DWConv7x7(x) [element-wise gating]
  2. RGB-T Local: cross_local = Kernel_1(rgb, thermal) [call fused kernel]
  3. RGB-T Global: cross_global = Kernel_2(rgb, thermal) [call fused kernel]
  4. Concat: features = cat[self_attn, cross_local, cross_global] → (B, 3C, H, W)
  5. Project: projected = Conv1x1(features) → (B, C, H, W)
  6. Layer scale: scaled = layer_scale * projected [broadcast scalar]
  7. Residual: rgb_mid = rgb_feat + DropPath(scaled)
  8. MLP: LayerNorm → Linear → 3×3 DWConv → GELU → Linear
  9. Layer scale + DropPath + residual → rgb_out

  This is a compositor kernel that calls Kernel 1 and 2 as sub-kernels
  Main savings: eliminates intermediate concat allocation + project alloc
  Also fuses MLP sub-operations (LN+Linear+DWConv+GELU+Linear)

Block counts per stage: [2, 2, 4, 2] = 10 total blocks
Memory savings: ~10 × 3 × (B×C×H×W) intermediate tensors eliminated
```

**Python wrapper**: `tuni_block_forward(rgb, thermal, block_weights, stage_idx, block_idx, training)` → rgb_out
**Save to**: `/mnt/forge-data/shared_infra/cuda_extensions/tuni_block/`

### Kernel 4: TensorRT Custom Plugins (`tuni_trt_plugin.cu`)
**Bottleneck**: Standard ONNX→TRT cannot fuse the custom attention patterns
**Target**: 40+ FPS on Jetson Orin NX (up from 27 FPS paper claim)

```
Plugin 1: LocalAttentionRGBT_TRT (IPluginV2DynamicExt)
  - Wraps Kernel 1 as TRT custom plugin
  - Supports FP16 and INT8 (with per-channel quantization for DWConv)
  - Dynamic batch size support

Plugin 2: GlobalAttention_TRT (IPluginV2DynamicExt)
  - Wraps Kernel 2 as TRT custom plugin
  - FP16: attention in FP16, softmax in FP32 (accumulator precision)
  - INT8: quantize Q/K/V, dequantize for attention

Plugin 3: Full TUNI Block as TRT subgraph
  - Register entire block as single TRT plugin
  - Eliminates all intermediate tensor materializations

Export pipeline:
  1. PyTorch → ONNX with custom op registration
  2. ONNX → TRT with plugin library (.so)
  3. Calibration with 500 training samples for INT8
  4. Benchmark: FP16 and INT8 on Jetson Orin NX
```

**Save to**: `/mnt/forge-data/shared_infra/cuda_extensions/tuni_trt/`
**DEMO CRITICAL**: This is the edge deployment kernel for Shenzhen partner demo

## MLX Metal Equivalents

All kernels have MLX equivalents using native MLX operations:

1. **`local_rgbt_attn_mlx.py`** — MLX fused LocalAttentionRGBT
   - `mlx.nn.Conv2d(groups=C)` for DWConv7x7
   - Element-wise multiply/diff: native MLX array ops
   - Cosine similarity: manual dot product + norm
   - No custom Metal kernels needed — MLX fusion handles it via lazy evaluation

2. **`global_rgbt_attn_mlx.py`** — MLX fused global attention
   - AdaptiveAvgPool: reshape + `mx.mean()` over spatial regions
   - Attention on 49 tokens: `mx.fast.scaled_dot_product_attention`
   - Bilinear upsample: custom implementation using `mx.interp` or manual

3. **`tuni_block_mlx.py`** — MLX full block
   - Compose local_rgbt + global_rgbt + RGB self-attention
   - MLX's lazy evaluation naturally fuses operations
   - DropPath: `mx.random.bernoulli` gating

4. **`tuni_model_mlx.py`** — Full model in MLX
   - Weight conversion: PyTorch state_dict → MLX safetensors/npz
   - Inference-only (training on CUDA)

## Benchmark Targets

| Kernel | Baseline (ms) | Target (ms) | Speedup |
|--------|--------------|-------------|---------|
| LocalAttentionRGBT per block (B=1, C=96, 120×160) | ~2.5 | ~0.6 | 4x |
| LocalAttentionRGBT per block (B=1, C=384, 30×40) | ~1.5 | ~0.4 | 3.5x |
| GlobalAttention per block (B=1, C=96, 120×160) | ~1.8 | ~0.7 | 2.5x |
| GlobalAttention per block (B=1, C=384, 30×40) | ~0.8 | ~0.4 | 2x |
| TUNI Block (stage 3, C=192, 60×80) | ~8.0 | ~4.0 | 2x |
| TUNI Block (stage 4, C=384, 30×40) | ~5.0 | ~3.0 | 1.7x |
| **Full forward pass (backbone_384_2242, 480×640)** | **~25** | **~13** | **1.9x** |
| **Full forward + decoder** | **~30** | **~16** | **1.9x** |
| **TRT FP16 on Jetson (480×640)** | **~37** | **~25** | **1.5x** |
| **TRT INT8 on Jetson (480×640)** | **N/A** | **~20** | **N/A** |

## Shared Memory Analysis (Key for Performance)

| Stage | Dims (C) | Feature Size | LocalAttentionRGBT Shared Mem | GlobalAttention Shared Mem |
|-------|----------|-------------|-------------------------------|---------------------------|
| 1 | 48 | 120×160 | ~7×7 tile × 48ch = 2.3KB | 49×48 = 2.3KB (fits easily) |
| 2 | 96 | 60×80 | ~7×7 tile × 96ch = 4.7KB | 49×96 = 4.7KB (fits) |
| 3 | 192 | 30×40 | ~7×7 tile × 192ch = 9.4KB | 49×192 = 9.4KB (fits) |
| 4 | 384 | 15×20 | ~7×7 tile × 384ch = 18.8KB | 49×384 = 18.8KB (fits) |

All stages fit comfortably in 48KB shared memory per SM.
7×7 pooled global attention is extremely cache-friendly (49 tokens total).

## IP Notes

- **local_rgbt_attn.cu** is the most novel — element-wise cross-modal attention with DWConv gating is TUNI's key innovation. Reusable by any cross-modal architecture using co-occurrence + difference features.
- **global_rgbt_attn.cu** leverages the 7×7 pooled attention being tiny enough to fit entirely in shared memory — a pattern applicable to any pooled cross-attention module.
- **tuni_trt_plugin.cu** is the DEMO priority — enables real-time inference on Jetson for the Shenzhen partner demonstration.
- All kernels stored at `/mnt/forge-data/shared_infra/cuda_extensions/`.

---
*Updated 2026-04-04 by ANIMA Research Agent*
