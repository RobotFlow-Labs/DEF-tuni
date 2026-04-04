# Benchmarks — DEF-tuni
# TUNI: Real-time RGB-T Semantic Segmentation
# Paper: ICRA 2026 | ArXiv: 2509.10005 | Journal: IEEE TCSVT (TUNI-v2)
# Config: backbone_384_2242 (dims [48,96,192,384], depths [2,2,4,2])
# Input: 480×640 (H×W), RGB 3ch + Thermal 1ch (converted internally)

## Paper Baseline Results (to reproduce)

### FMB Dataset (14 classes) — mIoU (%)
| Model Variant | mIoU | Params (M) | GFLOPs | FPS (GPU) |
|---------------|------|-----------|--------|-----------|
| Tiny [32,64,128,256] | TBD | TBD | TBD | TBD |
| backbone_384_2242 | TBD | TBD | TBD | TBD |
| backbone_512_2242 | TBD | TBD | TBD | TBD |
| backbone_320_2262 | TBD | TBD | TBD | TBD |

### PST900 Dataset (5 classes) — mIoU (%)
| Model Variant | mIoU | Params (M) | GFLOPs | FPS (GPU) |
|---------------|------|-----------|--------|-----------|
| backbone_384_2242 | TBD | TBD | TBD | TBD |

### CART Dataset (12 classes, aerial) — mIoU (%)
| Model Variant | mIoU | Params (M) | GFLOPs | FPS (GPU) |
|---------------|------|-----------|--------|-----------|
| backbone_384_2242 | TBD | TBD | TBD | TBD |

### Per-Class IoU — FMB (14 classes, backbone_384_2242)
| Class | IoU (%) |
|-------|---------|
| (14 rows) | TBD |

### Per-Class IoU — CART (12 classes, backbone_384_2242)
| Class | IoU (%) |
|-------|---------|
| (12 rows) | TBD |

## Real-Time Performance (KEY METRIC)

### FPS on Different Hardware
| Hardware | FP32 FPS | FP16 FPS | INT8 FPS | Latency FP16 (ms) |
|----------|---------|---------|---------|-------------------|
| RTX 6000 Pro Blackwell | TBD | TBD | TBD | TBD |
| Jetson Orin NX | TBD | TBD (27 claimed) | TBD | TBD (~37) |
| Jetson AGX Orin | TBD | TBD | TBD | TBD |
| Mac Studio M-series (MLX) | TBD | TBD | N/A | TBD |

### FPS vs Batch Size (RTX 6000 Pro, 480×640)
| Batch Size | FPS | Latency (ms) | GPU Memory (GB) |
|-----------|-----|-------------|----------------|
| 1 | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD |

### FPS vs Resolution (RTX 6000 Pro, batch_size=1)
| Resolution (H×W) | FPS | Latency (ms) | GFLOPs |
|------------------|-----|-------------|--------|
| 240×320 | TBD | TBD | TBD |
| 480×640 | TBD | TBD | TBD |
| 720×960 | TBD | TBD | TBD |
| 1080×1920 | TBD | TBD | TBD |

## Attention Type Breakdown (per-block profiling)

### Time Per Block — Stage 3 (C=192, 60×80), backbone_384_2242
| Component | Time (ms) | % of Block | Kernel Launches |
|-----------|----------|-----------|----------------|
| RGB-RGB Local (DWConv7x7 gate) | TBD | TBD | TBD |
| RGB-T Local (LocalAttentionRGBT) | TBD | TBD (~15 ops) | TBD |
| RGB-T Global (pool+attn+upsample) | TBD | TBD (~8 ops) | TBD |
| Concat + Project (1×1 Conv) | TBD | TBD | TBD |
| Layer Scale + DropPath + Residual | TBD | TBD | TBD |
| MLP (LN+Linear+DWConv+GELU+Linear) | TBD | TBD | TBD |
| **Block Total** | **TBD** | **100%** | **TBD (~20+)** |

### Time Per Stage — Full Forward Pass
| Stage | Blocks | Dims (C) | Feature Size | Time (ms) | % of Forward |
|-------|--------|----------|-------------|----------|-------------|
| Stem (RGB + Thermal) | — | 3→48, 1→24 | 120×160 | TBD | TBD |
| Stage 1 | 2 | 48 | 120×160 | TBD | TBD |
| Stage 2 | 2 | 96 | 60×80 | TBD | TBD |
| Stage 3 | 4 | 192 | 30×40 | TBD | TBD |
| Stage 4 | 2 | 384 | 15×20 | TBD | TBD |
| Decoder MLP | — | 256 | 120×160 out | TBD | TBD |
| **Total** | **10** | | | **TBD** | **100%** |

## Ablation Studies

### Attention Type Ablation — FMB (backbone_384_2242)
| Config | mIoU (%) | Params (M) | FPS | Delta vs Full |
|--------|---------|-----------|-----|--------------|
| Full model (all 3 attentions) | TBD | TBD | TBD | baseline |
| RGB-RGB Local only | TBD | TBD | TBD | TBD |
| RGB-T Local only | TBD | TBD | TBD | TBD |
| RGB-T Global only | TBD | TBD | TBD | TBD |
| RGB-RGB + RGB-T Local (no global) | TBD | TBD | TBD | TBD |
| RGB-RGB + RGB-T Global (no local) | TBD | TBD | TBD | TBD |

### Thermal Channel Dimension Ablation — FMB
| Thermal Channels | mIoU (%) | Params (M) | GFLOPs | FPS |
|-----------------|---------|-----------|--------|-----|
| C/4 (quarter) | TBD | TBD | TBD | TBD |
| C/2 (default) | TBD | TBD | TBD | TBD |
| C (full, same as RGB) | TBD | TBD | TBD | TBD |

### Pre-training Ablation — FMB
| Pre-training | mIoU (%) | Convergence (iters to 90% peak) |
|-------------|---------|-------------------------------|
| sRGB-TIR (default) | TBD | TBD |
| ImageNet RGB-only | TBD | TBD |
| Random init | TBD | TBD |

## Kernel Optimization Impact

| Kernel | Baseline (ms) | Optimized (ms) | Speedup |
|--------|--------------|----------------|---------|
| LocalAttentionRGBT (stage 2, C=96) | TBD | TBD | TBD (target 4x) |
| LocalAttentionRGBT (stage 4, C=384) | TBD | TBD | TBD (target 3.5x) |
| GlobalAttention (stage 2, C=96) | TBD | TBD | TBD (target 2.5x) |
| GlobalAttention (stage 4, C=384) | TBD | TBD | TBD (target 2x) |
| TUNI Block (stage 3, C=192) | TBD | TBD | TBD (target 2x) |
| **Full forward pass** | **TBD** | **TBD** | **TBD (target 1.9x)** |
| **TRT FP16 on Jetson** | **~37** | **~25** | **1.5x** |
| **TRT INT8 on Jetson** | **N/A** | **~20** | **N/A** |

## Dual-Compute Validation

| Backend | Dataset | mIoU | FPS | Inference (ms) |
|---------|---------|------|-----|---------------|
| CUDA (RTX 6000 Pro) | FMB | TBD | reference | TBD |
| CUDA (RTX 6000 Pro) | PST900 | TBD | reference | TBD |
| MLX (Mac Studio) | FMB | TBD (within 0.5%) | TBD | TBD |
| MLX (Mac Studio) | PST900 | TBD (within 0.5%) | TBD | TBD |
| TRT FP16 (Jetson Orin NX) | FMB | TBD (within 1%) | TBD (27+ target) | TBD |
| TRT INT8 (Jetson Orin NX) | FMB | TBD (within 2%) | TBD (40+ target) | TBD |

## Cross-Module RGB-T Comparison (shared datasets)

| Module | Method | FMB mIoU | PST900 mIoU | Params (M) | GFLOPs | FPS (GPU) | FPS (Jetson) |
|--------|--------|---------|------------|-----------|--------|-----------|-------------|
| DEF-tuni | TUNI (384_2242) | TBD | TBD | TBD | TBD | TBD | 27 (claimed) |
| DEF-rtfdnet | RTFDNet (MiT-B4) | TBD | TBD | TBD | TBD | TBD | TBD |
| DEF-cmssm | CMSSM (Mamba) | TBD | TBD | TBD | TBD | TBD | TBD |
| DEF-hypsam | HyPSAM | TBD | TBD | TBD | TBD | TBD | TBD |

### Accuracy-Speed Pareto Analysis
```
mIoU (%) ↑
  |   * CMSSM (highest accuracy, slower)
  |  * RTFDNet (robust, moderate speed)
  | * HyPSAM (SAM-based, high params)
  |* TUNI (real-time champion ← HERE)
  |________________________→ FPS ↑
```

## Model Variant Comparison

| Variant | Dims | Depths | Params (M) | GFLOPs | FMB mIoU | FPS (GPU) | FPS (Jetson) |
|---------|------|--------|-----------|--------|---------|-----------|-------------|
| Tiny | [32,64,128,256] | [3,4,4,6] | TBD | TBD | TBD | TBD | TBD |
| 320_2262 | [40,80,160,320] | [2,2,6,2] | TBD | TBD | TBD | TBD | TBD |
| 384_2242 (default) | [48,96,192,384] | [2,2,4,2] | TBD | TBD | TBD | TBD | TBD |
| 512_2242 | [64,128,256,512] | [2,2,4,2] | TBD | TBD | TBD | TBD | TBD |

## Hardware & Methodology
- RTX 6000 Pro Blackwell (training + evaluation)
- Mac Studio M-series (MLX local dev)
- Jetson Orin NX (edge deployment — DEMO target)
- Input: 480×640, RGB (3ch) + Thermal (1ch internally)
- FPS measurement: 100 warmup + 1000 iterations, mean ± std
- GFLOPs: fvcore FlopCountAnalysis + thop verification
- Results stored as JSON: `results_*.json`

---
*Updated 2026-04-04 by ANIMA Research Agent*
