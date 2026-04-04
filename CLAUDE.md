# DEF-tuni — TUNI: Real-time RGB-T Semantic Segmentation with Unified Multi-Modal Feature Extraction
# Wave 8 Defense Module
# Paper: "TUNI: Real-time RGB-T Semantic Segmentation with Unified Multi-Modal Feature Extraction and Cross-Modal Feature Fusion"
# Authors: Xiaodong Guo et al.
# Published: ICRA 2026 | ArXiv: 2509.10005
# Journal version: IEEE TCSVT (submitted as TUNI-v2)
# Repo: https://github.com/xiaodonguo/TUNI
# Domain: Real-Time RGB-Thermal Semantic Segmentation
# Product Stack: ORACLE (through-wall safety) / ATLAS (fleet AV night ops) / NEMESIS (terrain nav)
# Related: DEF-rtfdnet, DEF-cmssm (same author), DEF-hypsam, DEF-rgbtcc (RGB-T family)

## Status: ⬜ Not Started

## Paper Summary
TUNI is a **real-time** RGB-T semantic segmentation architecture designed for edge deployment (27 FPS on Jetson Orin NX). Unlike dual-stream approaches that process RGB and thermal independently then fuse, TUNI uses a **unified encoder** with three types of attention in each block: (1) **RGB-RGB Local Attention** — self-attention within RGB features using element-wise gating with 7×7 depthwise conv, (2) **RGB-T Local Attention** (`LocalAttentionRGBT`) — cross-modal attention computing co-occurrence (rgb*thermal) and difference (|rgb-thermal|) with 7×7 depthwise conv + channel attention via cosine similarity, (3) **RGB-T Global Attention** — pooled cross-modal attention with 7×7 adaptive average pooling for long-range dependencies. The thermal stream runs at **half channel dimension** (C/2 vs C for RGB) for efficiency. Pre-training uses sRGB-TIR translation to generate pseudo-thermal from ImageNet. Accepted at ICRA 2026 with journal extension at IEEE TCSVT.

## Architecture
- **Backbone**: `Model1_Backbone` (custom ConvNet + attention hybrid)
  - 4 stages with dims [48, 96, 192, 384] (default config `backbone_384_2242`)
  - Depths: [2, 2, 4, 2] blocks per stage
  - RGB stem: Conv 3→24→48 (stride 4 total)
  - Thermal stem: Conv 1→12→24 (stride 4 total, **single channel input**)
  - Each stage: downsample → N × Block → output features
- **Block** (per stage, key innovation):
  - `Attention` module with 3 sub-attentions:
    1. **RGB-RGB Local**: `x * conv7x7(x)` — element-wise gated self-attention
    2. **RGB-T Local** (`LocalAttentionRGBT`):
       - co = conv7x7(rgb_proj * thermal_proj) — co-occurrence features
       - di = conv7x7(|rgb_proj - thermal_proj|) — difference features
       - Channel attention via cosine similarity → sigmoid → weighted output
    3. **RGB-T Global**: AdaptiveAvgPool(7×7) → cross-attention (Q from concat, KV from RGB) → bilinear upsample
  - Output: concat [local_rr, local_rx, global_rx] → project
  - Layer scale (1e-6 init) + DropPath
  - MLP: LayerNorm → Linear → 3×3 DWConv (positional) → GELU → Linear
- **Decoder**: `Decoder_MLP` — lightweight MLP head
  - Input channels: [48, 96, 192, 384]
  - Embed dim: 256
  - 4× bilinear upsample at output
- **Thermal input**: Single channel (grayscale), NOT 3-channel like RTFDNet
- **Model variants**:
  - Tiny: dims [32,64,128,256], depths [3,4,4,6]
  - backbone_384_2242: dims [48,96,192,384], depths [2,2,4,2] (default)
  - backbone_512_2242: dims [64,128,256,512], depths [2,2,4,2]
  - backbone_320_2262: dims [40,80,160,320], depths [2,2,6,2]
- **Pre-training**: RGB-T pre-training using sRGB-TIR translated pseudo-thermal from ImageNet

## Datasets Used
- **FMB** (Freiburg Multi-modal Benchmark) — 14 classes, RGB-T urban driving
- **PST900** (Penn State Thermal) — 5 classes, RGB-T off-road/indoor
- **CART** (Caltech Aerial RGB-T) — 12 classes, aerial RGB-T
- Input resolution: 480×640 (H×W)
- Shared with DEF-rtfdnet (FMB, PST900), DEF-cmssm (same author)

## Key Results (to reproduce)
- FMB mIoU: TBD (expect competitive with SOTA at real-time speed)
- PST900 mIoU: TBD
- CART mIoU: TBD
- **27 FPS on Jetson Orin NX** — real-time edge deployment (key selling point)
- GFLOPs: TBD (measured with fvcore + thop in fps.py)
- Parameters: TBD M

## Dependencies
- Python 3.9+ (we use 3.10)
- PyTorch 2.0+ + CUDA (cu128)
- mmcv 2.2.0 + mmengine
- fvcore, thop (for FLOPs/params measurement)
- Pre-trained weights: available at GitHub releases (TUNI.zip)

## Repo Structure
```
TUNI/
├── README.md
├── __init__.py
├── model1.py                     — Full model (Encoder + Decoder), 3.3KB
├── fps.py                        — FPS benchmarking script, 2.3KB
├── evaluate_FMB.py               — FMB evaluation script
├── evaluate_pst900.py            — PST900 evaluation script
├── evaluate_CART.py              — CART evaluation script
├── backbone_model/
│   └── TUNI.py                   — Core backbone (15.7KB, main architecture)
├── decoder/
│   ├── MLP.py                    — Decoder_MLP (lightweight, default)
│   ├── Hamburger.py              — Decoder_Ham (alternative)
│   ├── DeepLabV3.py              — DeepLabV3 decoder option
│   ├── FSN.py                    — FSN decoder option
│   ├── decode_head.py            — Base decode head
│   ├── mask_decoder.py           — Mask decoder variant
│   ├── promt_decoder.py          — Prompt decoder variant
│   └── transformer.py            — Transformer decoder components
├── configs/
│   ├── base_cfg.py               — Base training config
│   ├── FMB.json                  — FMB dataset config
│   ├── PST900.json               — PST900 dataset config
│   └── CART.json                 — CART dataset config
├── toolbox/                      — Training/eval utilities
└── Fig/                          — Paper figures
```

## Build Requirements
- [ ] Clone repo: `git clone https://github.com/xiaodonguo/TUNI.git`
- [ ] Create uv env: `uv venv .venv --python 3.10`
- [ ] Install PyTorch cu128: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- [ ] Install mmcv + mmengine: `uv pip install mmcv==2.2.0 mmengine`
- [ ] Install measurement tools: `uv pip install fvcore thop ptflops`
- [ ] Download pre-trained weights from GitHub releases (TUNI.zip)
- [ ] Download RGB-T pre-training weights (sRGB-TIR based)
- [ ] Download datasets (FMB, PST900, CART)
- [ ] Fix absolute paths in model1.py (pretrained weight paths)
- [ ] Run evaluation on all 3 datasets
- [ ] Measure FPS on target hardware
- [ ] Profile training and inference pipeline
- [ ] Build custom CUDA kernels
- [ ] Port to MLX
- [ ] Dual-compute validation

## CUDA Kernel Targets
1. **Fused RGB-T Local Attention** — `LocalAttentionRGBT` does: 2× Linear project → permute → element multiply + abs diff → 2× DWConv7x7 → concat → mean attention map → cosine similarity → channel attention → FC → sigmoid → weighted output. ~15 ops per call, 4 stages. Fuse into single kernel.
2. **Fused RGB-T Global Attention** — AdaptiveAvgPool(7×7) → QKV projection → attention → bilinear upsample. Currently separate ops. Fuse pool+attention+upsample.
3. **Fused Block Forward** — Each Block has 3 attentions + 2 MLPs + layer scale + DropPath. Significant kernel launch overhead from 20+ small ops per block.
4. **Optimized Dual-Stream Downsampling** — RGB (C) and thermal (C/2) downsample layers are independent Conv2d calls. Fuse into single batched conv.

## Defense Marketplace Value
TUNI is the **real-time champion** of RGB-T segmentation — 27 FPS on Jetson Orin NX makes it deployable on actual edge devices (drones, patrol robots, autonomous vehicles). The pre-training on pseudo-thermal data means it generalizes beyond specific thermal cameras. Direct application to: UAV surveillance in darkness, autonomous convoy night operations, border patrol with thermal cameras, fire-and-rescue perception, night-time pedestrian detection. The ICRA 2026 publication + TCSVT journal extension demonstrates peer-reviewed quality. Same author as DEF-cmssm — complementary approaches (TUNI = real-time, CMSSM = state-space accuracy).

## Package Manager: uv (NEVER pip)
## Python: >= 3.10
## Torch: cu128 index
## Git prefix: [DEF-tuni]
