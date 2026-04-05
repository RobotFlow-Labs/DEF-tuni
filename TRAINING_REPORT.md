# TRAINING REPORT — DEF-tuni (TUNI RGB-T Segmentation)
## Date: 2026-04-05
## Module: DEF-tuni v0.2.0

---

## Model Architecture
- **Name**: TUNI (Real-time RGB-T Semantic Segmentation)
- **Paper**: ICRA 2026, ArXiv 2509.10005
- **Variant**: 384_2242 (dims=[48,96,192,384], depths=[2,2,4,2])
- **Parameters**: 10.6M
- **Backbone**: Custom ConvNet + 3-type attention (RGB-RGB local, RGB-T local, RGB-T global)
- **Decoder**: MLP (256-dim embed)
- **Input**: RGB (3×H×W) + Thermal (3×H×W, single-channel extracted internally)

## Training Configuration
| Parameter | FMB | PST900 | CART |
|-----------|-----|--------|------|
| Batch size | 4 | 4 | 8 |
| Learning rate | 1e-4 | 1e-4 | 1e-4 |
| Optimizer | AdamW | AdamW | AdamW |
| Weight decay | 5e-4 | 5e-4 | 5e-4 |
| Precision | FP16 | FP16 | FP16 |
| Scheduler | Warmup + Cosine | Warmup + Cosine | Warmup + Cosine |
| Warmup ratio | 5% | 5% | 5% |
| Max epochs | 80 | 200 | 300 |
| Early stopping | patience=20 | patience=20 | patience=20 |
| Crop size | 600×800 | 640×1280 | 512×512 |
| Pretrained | FMB.pth | PST900.pth | CART.pth |

## Results

### FMB (Freiburg Multi-modal Benchmark) — 15 classes
| Metric | Value |
|--------|-------|
| **mIoU** | **61.87%** |
| Pixel Accuracy | 92.02% |
| Epochs trained | 73/80 (early stop) |
| Best epoch | ~53 |
| Training time | ~2h on L4 |
| GPU | NVIDIA L4 (GPU 5) |

### PST900 (Penn State Thermal) — 5 classes
| Metric | Value |
|--------|-------|
| **mIoU** | **84.55%** |
| Pixel Accuracy | 99.51% |
| Epochs trained | 22/200 (early stop) |
| Best epoch | ~2 |
| Training time | ~30min on L4 |
| GPU | NVIDIA L4 (GPU 6) |

### CART (Caltech Aerial RGB-T) — 12 classes
| Metric | Value |
|--------|-------|
| **mIoU** | **69.67%** |
| Pixel Accuracy | 93.20% |
| Epochs trained | 21/300 (early stop) |
| Best epoch | ~1 |
| Training time | ~25min on L4 |
| GPU | NVIDIA L4 (GPU 6) |

## Hardware
- **GPU**: NVIDIA L4 (23GB VRAM each)
- **VRAM Usage**: ~6.5GB per training run (28% of 23GB)
- **Throughput**: ~18 img/s (FMB), ~35 img/s (CART)
- **Server**: 8×L4, CUDA 12.8, PyTorch 2.11

## Exports
| Format | Size | Status |
|--------|------|--------|
| PyTorch (.pth) | 42.8MB × 3 | ✅ All datasets |
| Safetensors | 42.6MB × 3 | ✅ All datasets |
| ONNX (opset 18) | 3.6MB + 41MB data | ✅ FMB |
| TensorRT FP32 | 50MB | ✅ FMB |
| TensorRT FP16 | 29MB | ✅ FMB |

## CUDA Kernels (Custom)
| Kernel | Op | Location |
|--------|----|----------|
| fused_rgbt_concat_norm | RGB-T concat + batch norm | tuni_ops.cu |
| fused_seg_argmax | Argmax + class index | tuni_ops.cu |
| fused_local_rgbt_attn | DWConv7x7 co-occurrence + difference | local_rgbt_attn.cu |
| fused_adaptive_avg_pool_7x7 | Global attention pooling | global_rgbt_attn.cu |

Saved to: `/mnt/forge-data/shared_infra/cuda_extensions/tuni_rgbt_attention/`

## Artifacts
- Checkpoints: `/mnt/artifacts-datai/checkpoints/DEF-tuni{,-pst900,-cart}/best.pth`
- Exports: `/mnt/artifacts-datai/exports/DEF-tuni/`
- Logs: `/mnt/artifacts-datai/logs/DEF-tuni/`
- HuggingFace: `ilessio-aiflowlab/DEF-tuni`

## Notes
- Model started from pretrained weights (paper-provided), fine-tuned on each dataset
- Early stopping triggered on all 3 datasets — convergence was fast from pretrained initialization
- PST900 achieved highest mIoU (84.55%) due to fewer classes (5) and strong pretrained baseline
- mmcv dependency removed — replaced with inline `build_norm_layer` and `DropPath`
- No mmcv compilation needed (saved ~1h of build debugging)

---
*Generated 2026-04-05 by ANIMA Autopilot*
