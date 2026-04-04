# NEXT_STEPS — DEF-tuni
## Last Updated: 2026-04-04
## Status: ENRICHED — Ready for build
## MVP Readiness: 0%
## Total PRDs: 10 (62 hours estimated)
## Critical Path: PRD-001 → PRD-002 → PRD-003 → PRD-005 → PRD-006 → PRD-008

---

### Immediate Next Actions
1. Clone repo: `git clone https://github.com/xiaodonguo/TUNI.git`
2. Create uv env: `uv venv .venv --python 3.11`
3. Install PyTorch cu128 + mmcv 2.2.0 + mmengine
4. Install measurement tools: fvcore, thop, ptflops
5. Download pre-trained weights from GitHub releases (TUNI.zip)
6. Download sRGB-TIR pre-training weights
7. Download 3 RGB-T datasets (FMB, PST900, CART)
8. Fix absolute paths in model1.py
9. Run evaluation on all 3 datasets

### What This Module Does
TUNI is a real-time RGB-T semantic segmentation architecture designed for edge deployment (27 FPS on Jetson Orin NX). Uses a unified encoder with three attention types per block: RGB-RGB Local (element-wise gating), RGB-T Local (co-occurrence + difference with DWConv + cosine similarity channel attention), and RGB-T Global (pooled cross-attention with bilinear upsample). Thermal stream runs at half channel dimension for efficiency. Published at ICRA 2026 with journal extension at IEEE TCSVT. Same author as DEF-cmssm (complementary approaches).

### Key Results to Reproduce
- mIoU on 3 RGB-T datasets (FMB 14-class, PST900 5-class, CART 12-class)
- 27 FPS on Jetson Orin NX (THE key selling point)
- Model variants: Tiny, 384_2242 (default), 512_2242, 320_2262
- Ablation: attention types, thermal channels, pre-training

### TODO (by PRD)
- [ ] **PRD-001**: Environment setup — uv + mmcv + weights download (5h)
- [ ] **PRD-002**: Dataset download — FMB + PST900 + CART (5h)
- [ ] **PRD-003**: Evaluation baseline — all 3 datasets, all model variants (6h)
- [ ] **PRD-004**: Real-time FPS benchmarking — resolution/batch sweeps (4h)
- [ ] **PRD-005**: CUDA profiling — per-block breakdown of 3 attention types (5h)
- [ ] **PRD-006**: Custom CUDA kernels — 4 kernels (16h)
- [ ] **PRD-007**: MLX port — unified encoder on Mac Studio (8h)
- [ ] **PRD-008**: Edge deployment — TensorRT on Jetson Orin NX, DEMO PRIORITY (8h)
- [ ] **PRD-009**: RGB-T family integration — compare with RTFDNet/CMSSM/HyPSAM (3h)
- [ ] **PRD-010**: Ablation studies — attention types, thermal channels, pre-training (2h)

### Blockers
- None — repo has real code, weights on GitHub releases, datasets freely available

### Datasets/Models Needed
- FMB (~2GB) — FREE, shared with DEF-rtfdnet, DEF-cmssm
- PST900 (~1GB) — FREE, shared with DEF-rtfdnet, DEF-cmssm
- CART (~2GB) — FREE, specific to TUNI (aerial RGB-T)
- Pre-trained weights: TUNI.zip from GitHub releases (~50MB)
- sRGB-TIR pre-training weights (ImageNet pseudo-thermal)
- Total: ~5-6GB datasets + ~100MB models

### Kernel IP Targets (shared across ANIMA)
1. **Fused LocalAttentionRGBT** → novel cross-modal co-occurrence + difference attention (3-5x speedup)
2. **Fused Global RGB-T Attention** → pooled cross-attention with 49 tokens fits in shared memory (2-3x)
3. **Fused TUNI Block** → mega-kernel for entire block forward pass (1.5-2x)
4. **TensorRT Plugins** → DEMO CRITICAL for Jetson edge deployment (40+ FPS target)

### Related Modules
- **DEF-rtfdnet** — RGB-T dual-stream SegFormer, robustness focus, shares FMB/PST900
- **DEF-cmssm** — RGB-T Mamba state-space, SAME AUTHOR (Xiaodong Guo), shares datasets
- **DEF-hypsam** — RGB-T SOD + SAM, shares thermal processing
- **DEF-rgbtcc** — RGB-T crowd counting, shares thermal sensors

### DEMO Priority
TUNI is the real-time champion — 27 FPS on Jetson makes it the most deployable RGB-T module. Edge deployment (PRD-008) is CRITICAL for the Shenzhen partner demo on April 23-24.

---
*Updated 2026-04-04 by ANIMA Research Agent*
