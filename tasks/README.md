# Tasks — DEF-tuni
# TUNI: Real-time RGB-T Semantic Segmentation
# Paper: ICRA 2026 | ArXiv: 2509.10005 | Journal: IEEE TCSVT (TUNI-v2)
# Author: Xiaodong Guo et al. (same as DEF-cmssm)
# Repo: https://github.com/xiaodonguo/TUNI
# Total: 10 PRDs | 62 hours estimated
# Critical Path: PRD-001 → PRD-002 → PRD-003 → PRD-004/005 → PRD-006

---

## PRD-001: Environment Setup (5h) ⬜
**Priority**: P0 — blocking everything
**Dependencies**: None

### Steps
```bash
# 1. Clone repo
cd /mnt/forge-data/shared_infra/repos/
git clone https://github.com/xiaodonguo/TUNI.git
cd TUNI

# 2. Create uv env
uv venv .venv --python 3.10
source .venv/bin/activate

# 3. Install PyTorch cu128
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install mmcv + mmengine (required by eval scripts)
uv pip install mmcv==2.2.0 mmengine

# 5. Install measurement tools
uv pip install fvcore thop ptflops

# 6. Install additional deps
uv pip install timm einops scipy

# 7. Download pre-trained weights from GitHub releases
# TUNI.zip contains checkpoint for backbone_384_2242 config
wget https://github.com/xiaodonguo/TUNI/releases/download/v1.0/TUNI.zip
unzip TUNI.zip -d ./pretrained/

# 8. Download RGB-T pre-training weights (sRGB-TIR based)
# These are the ImageNet pretrained weights with pseudo-thermal translation
# Check releases page for pretrained_backbone.pth

# 9. Fix absolute paths in model1.py
# model1.py has hardcoded pretrained weight paths — update to local paths
# backbone_path = './pretrained/TUNI_backbone.pth'
```

### Acceptance Criteria
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] `python -c "import mmcv; print(mmcv.__version__)"` → 2.2.0
- [ ] `python -c "from backbone_model.TUNI import Model1_Backbone; print('OK')"` → OK
- [ ] `python -c "from model1 import Model1; print('OK')"` → OK
- [ ] `python -c "from decoder.MLP import Decoder_MLP; print('OK')"` → OK
- [ ] Pre-trained weights exist in `./pretrained/`
- [ ] `fvcore` and `thop` importable for FLOPs measurement

---

## PRD-002: Dataset Download & Preparation (5h) ⬜
**Priority**: P0 — blocking evaluation
**Dependencies**: PRD-001

### Datasets (3 total, shared with RGB-T family)
```bash
DATASET_ROOT=/mnt/forge-data/datasets/rgbt/

# 1. FMB (Freiburg Multi-modal Benchmark) — 14 classes, urban driving
# ~2GB, RGB-T pairs at 480×640
mkdir -p $DATASET_ROOT/FMB
# Download from official source (check paper/repo for URL)
# Expected structure: FMB/{train,test}/{rgb,thermal,labels}/

# 2. PST900 (Penn State Thermal) — 5 classes, off-road/indoor
# ~1GB, RGB-T pairs
mkdir -p $DATASET_ROOT/PST900
# Download from: https://github.com/ShreyasSkandanS/PST900_thermal_rgb
# Expected structure: PST900/{train,test}/{rgb,thermal,labels}/

# 3. CART (Caltech Aerial RGB-T) — 12 classes, aerial
# ~2GB, drone-captured RGB-T
mkdir -p $DATASET_ROOT/CART
# Download from official source (check paper for URL)
# Expected structure: CART/{train,test}/{rgb,thermal,labels}/
```

### Config Updates
```bash
# Update dataset paths in configs/
# configs/FMB.json → set data_root to $DATASET_ROOT/FMB
# configs/PST900.json → set data_root to $DATASET_ROOT/PST900
# configs/CART.json → set data_root to $DATASET_ROOT/CART
```

### IMPORTANT: Thermal Input Format
- TUNI takes **single-channel thermal** input (unlike RTFDNet which uses 3-channel)
- The backbone converts 3ch→1ch internally: `x_e = x_e[:,0,:,:].unsqueeze(1)`
- But input must still be loaded as image (possibly 3-channel, backbone handles conversion)
- Thermal stem: Conv2d(1, 12, 3, 2, 1) → Conv2d(12, 24, 3, 2, 1)

### Acceptance Criteria
- [ ] FMB dataset accessible at configured path, correct train/test split
- [ ] PST900 dataset accessible at configured path
- [ ] CART dataset accessible at configured path
- [ ] All config JSON files updated with correct data_root paths
- [ ] Sample images load correctly: RGB (H×W×3), Thermal (H×W×1 or H×W×3)
- [ ] Datasets shared with DEF-rtfdnet, DEF-cmssm symlinked

---

## PRD-003: Evaluation Baseline — All 3 Datasets (6h) ⬜
**Priority**: P0 — establishes ground truth
**Dependencies**: PRD-001, PRD-002

### Steps
```bash
# Run evaluation on each dataset with pre-trained weights
# Default config: backbone_384_2242 (dims [48,96,192,384], depths [2,2,4,2])

# 1. FMB evaluation (14 classes)
python evaluate_FMB.py \
    --config configs/FMB.json \
    --checkpoint ./pretrained/TUNI_FMB.pth \
    --gpu 0

# 2. PST900 evaluation (5 classes)
python evaluate_pst900.py \
    --config configs/PST900.json \
    --checkpoint ./pretrained/TUNI_PST900.pth \
    --gpu 0

# 3. CART evaluation (12 classes)
python evaluate_CART.py \
    --config configs/CART.json \
    --checkpoint ./pretrained/TUNI_CART.pth \
    --gpu 0

# 4. Record all results
# Save per-class IoU + mIoU for each dataset
# Compare with paper-reported numbers
```

### Key Metrics to Record
- mIoU per dataset (FMB 14-class, PST900 5-class, CART 12-class)
- Per-class IoU for all classes
- Pixel accuracy
- Model parameters (should be ~5-10M given dims [48,96,192,384])
- GFLOPs at 480×640 input

### Acceptance Criteria
- [ ] FMB mIoU recorded, within 1% of paper-reported value
- [ ] PST900 mIoU recorded, within 1% of paper-reported value
- [ ] CART mIoU recorded, within 1% of paper-reported value
- [ ] Per-class IoU tables saved to benchmarks/
- [ ] GFLOPs measured using fvcore: `FlopCountAnalysis(model, (rgb_input, thermal_input))`
- [ ] Parameter count verified with `sum(p.numel() for p in model.parameters())`

---

## PRD-004: Real-Time FPS Benchmarking (4h) ⬜
**Priority**: P0 — KEY selling point is 27 FPS on Jetson Orin NX
**Dependencies**: PRD-001

### Steps
```bash
# 1. GPU FPS benchmark (RTX 6000 Pro)
python fps.py \
    --input_size 480 640 \
    --warmup 100 \
    --iterations 1000

# fps.py uses fvcore + thop internally
# Measures: FPS, GFLOPs, parameters, latency (ms)

# 2. Batch size sweep
for BS in 1 2 4 8 16; do
    python fps.py --batch_size $BS --input_size 480 640
done

# 3. Resolution sweep (TUNI should work at various resolutions)
for RES in "240 320" "480 640" "720 960" "1080 1920"; do
    python fps.py --input_size $RES --batch_size 1
done

# 4. Model variant comparison
# Tiny: dims [32,64,128,256], depths [3,4,4,6]
# 384_2242: dims [48,96,192,384], depths [2,2,4,2] (default)
# 512_2242: dims [64,128,256,512], depths [2,2,4,2]
# 320_2262: dims [40,80,160,320], depths [2,2,6,2]
# Benchmark all 4 variants

# 5. Compare with RTFDNet (DEF-rtfdnet) on same hardware
# RTFDNet uses SegFormer dual-stream — likely slower
```

### Key Targets
- 27 FPS on Jetson Orin NX (paper claim)
- 60+ FPS on RTX 6000 Pro expected
- Latency < 37ms per frame (for 27 FPS)

### Acceptance Criteria
- [ ] FPS measured on RTX 6000 Pro at 480×640 (batch_size=1)
- [ ] Latency (ms) measured with warmup and std deviation
- [ ] GFLOPs measured with fvcore FlopCountAnalysis
- [ ] All 4 model variants benchmarked
- [ ] Batch size scaling curve plotted
- [ ] Resolution scaling curve plotted
- [ ] Results saved to benchmarks/ as JSON

---

## PRD-005: CUDA Profiling — Per-Block Breakdown (5h) ⬜
**Priority**: P1 — identifies kernel optimization targets
**Dependencies**: PRD-001, PRD-003

### Profiling Strategy
```bash
# 1. Nsight Systems full trace
nsys profile --trace=cuda,nvtx --output=tuni_profile \
    python fps.py --input_size 480 640 --iterations 100

# 2. PyTorch profiler with per-op breakdown
python -c "
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from model1 import Model1

model = Model1(backbone='backbone_384_2242', num_classes=14).cuda().eval()
rgb = torch.randn(1, 3, 480, 640).cuda()
thermal = torch.randn(1, 3, 480, 640).cuda()

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function('full_forward'):
        out = model(rgb, thermal)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=30))
prof.export_chrome_trace('tuni_trace.json')
"

# 3. Per-block annotation with NVTX markers
# Add markers to backbone_model/TUNI.py:
# - torch.cuda.nvtx.range_push('Block_stage{i}_rgb_local')
# - torch.cuda.nvtx.range_push('Block_stage{i}_rgbt_local')
# - torch.cuda.nvtx.range_push('Block_stage{i}_rgbt_global')
# - torch.cuda.nvtx.range_push('Decoder_MLP')
```

### Expected Bottlenecks (per architecture analysis)
1. **LocalAttentionRGBT** — ~15 ops per call × 10 blocks = 150 kernel launches
   - 2× Linear project → permute → element multiply + abs diff
   - 2× DWConv7x7 → concat → mean → cosine similarity → sigmoid → FC → weighted output
2. **RGB-T Global Attention** — AdaptiveAvgPool(7×7) + QKV + attention + upsample per block
3. **Dual-stream stems** — RGB (3→24→48) + thermal (1→12→24) separate Conv2d calls
4. **Layer scale + DropPath** — small ops with per-block kernel launch overhead

### Acceptance Criteria
- [ ] Nsight Systems trace generated and analyzed
- [ ] Per-block time breakdown (3 attention types × 10 blocks)
- [ ] Top 20 CUDA ops by time listed
- [ ] Memory bandwidth utilization measured
- [ ] Kernel launch overhead quantified (expected: significant due to many small ops)
- [ ] Bottleneck ranking: which attention type dominates per stage
- [ ] Profile saved to benchmarks/profiling/

---

## PRD-006: Custom CUDA Kernels — 4 Targets (16h) ⬜
**Priority**: P1 — core IP generation
**Dependencies**: PRD-005

### Kernel 1: Fused LocalAttentionRGBT (6h)
```
File: kernels/cuda/local_rgbt_attn.cu
Save to: /mnt/forge-data/shared_infra/cuda_extensions/local_rgbt_attn/

Input: rgb_feat (B×C×H×W), thermal_feat (B×C/2×H×W),
       W_rgb (C×C, Linear), W_thermal (C/2×C/2, Linear),
       DWConv7x7_co, DWConv7x7_di, FC_attn
Output: fused_output (B×C×H×W)

Method: Fuse ~15 separate ops into single kernel per block:
  1. Linear project: rgb_proj = W_rgb @ rgb, thermal_proj = W_thermal @ thermal
  2. Co-occurrence: co = DWConv7x7(rgb_proj * thermal_proj)
  3. Difference: di = DWConv7x7(|rgb_proj - thermal_proj|)
  4. Concat: cat = [co, di] along channel
  5. Mean attention: attn_map = mean(cat, dim=1)
  6. Cosine similarity: cos_sim = (co · di) / (||co|| * ||di||)
  7. Channel attention: gate = sigmoid(FC(cos_sim))
  8. Output: out = gate * cat

Target: 3-5x speedup per LocalAttentionRGBT call
Reason: ~150 kernel launches across 10 blocks → ~15 fused launches

REUSABLE by: Any model with element-wise cross-modal attention + DWConv gating
```

### Kernel 2: Fused RGB-T Global Attention (4h)
```
File: kernels/cuda/global_rgbt_attn.cu
Save to: /mnt/forge-data/shared_infra/cuda_extensions/global_rgbt_attn/

Input: rgb_feat (B×C×H×W), thermal_feat (B×C/2×H×W),
       QKV weights, output projection weights
Output: global_attn_out (B×C×H×W)

Method: Fuse pool+attention+upsample:
  1. AdaptiveAvgPool2d(7×7): rgb (B,C,7,7), thermal (B,C/2,7,7) [shared memory reduction]
  2. Concat: (B, C+C/2, 7, 7) → flatten → (B, 49, 3C/2)
  3. QKV from concat, KV from RGB: Q (from concat), K,V (from rgb_pooled)
  4. Scaled dot-product attention: softmax(QK^T / sqrt(d)) @ V
  5. Output projection: Linear
  6. Bilinear upsample: 7×7 → H×W

Target: 2-3x speedup per global attention call
Key: Pool + attention in shared memory avoids materialization of 7×7 intermediate
```

### Kernel 3: Fused TUNI Block Forward (4h)
```
File: kernels/cuda/tuni_block.cu
Save to: /mnt/forge-data/shared_infra/cuda_extensions/tuni_block/

Input: rgb_feat (B×C×H×W), thermal_feat (B×C/2×H×W), all block weights
Output: rgb_out (B×C×H×W), thermal_out (B×C/2×H×W)

Method: Fuse entire block's 3 attentions + MLP:
  1. RGB-RGB Local: x * DWConv7x7(x) [element-wise gated self-attention]
  2. RGB-T Local: call Kernel 1 (fused LocalAttentionRGBT)
  3. RGB-T Global: call Kernel 2 (fused global attention)
  4. Concat [local_rr, local_rx, global_rx] → 1×1 Conv project
  5. Layer scale (1e-6 init) + residual
  6. LayerNorm → Linear → 3×3 DWConv (positional) → GELU → Linear (MLP)
  7. Layer scale + residual + DropPath

Target: 1.5-2x speedup per block (20+ ops → single mega-kernel)
Eliminates: kernel launch overhead from ~20 separate ops per block
```

### Kernel 4: TensorRT Plugin for Edge (2h)
```
File: kernels/trt/tuni_trt_plugin.cu
Save to: /mnt/forge-data/shared_infra/cuda_extensions/tuni_trt/

Method: Convert fused kernels to TensorRT custom plugins for Jetson deployment
  1. LocalAttentionRGBT as TRT plugin (IPluginV2DynamicExt)
  2. GlobalAttention as TRT plugin
  3. Full model export: ONNX → TRT with custom plugins
  4. FP16/INT8 quantization calibration

Target: 40+ FPS on Jetson Orin NX (up from 27 FPS paper claim)
This is the DEMO priority — real-time on actual edge hardware
```

### Acceptance Criteria
- [ ] All 4 kernels compile with nvcc and pass unit tests
- [ ] Kernel 1 achieves 3x+ speedup on LocalAttentionRGBT
- [ ] Kernel 2 achieves 2x+ speedup on global attention
- [ ] Kernel 3 achieves 1.5x+ speedup on full block
- [ ] Kernel 4 generates TRT engine that runs on Jetson Orin NX
- [ ] All kernels have Python wrappers via torch.utils.cpp_extension
- [ ] Gradients verified against PyTorch reference (atol=1e-4)
- [ ] Kernels stored in shared_infra for cross-module reuse

---

## PRD-007: MLX Port (8h) ⬜
**Priority**: P1 — dual-compute mandatory
**Dependencies**: PRD-003

### Steps
```bash
# 1. Create model_mlx.py
# Port Model1_Backbone → MLX using mlx.nn
# Key mappings:
#   Conv2d → mlx.nn.Conv2d
#   LayerNorm → mlx.nn.LayerNorm
#   Linear → mlx.nn.Linear
#   GELU → mlx.nn.GELU
#   AdaptiveAvgPool2d → custom (mlx doesn't have native adaptive pool)

# 2. Port attention modules
# RGB-RGB Local: element-wise multiply + DWConv → mlx.nn.Conv2d(groups=C)
# RGB-T Local: same pattern but cross-modal → element-wise + group conv
# RGB-T Global: AdaptiveAvgPool → mlx reshape + mean → attention → upsample

# 3. Port decoder
# Decoder_MLP is simple: 4× Linear + bilinear upsample
# mlx has no bilinear upsample → implement manually or use nearest + conv

# 4. Weight conversion
# PyTorch state_dict → MLX npz weights
python convert_weights.py --src pretrained/TUNI_FMB.pth --dst pretrained/TUNI_FMB.npz

# 5. Verify output match
# Compare MLX vs CUDA outputs on same input
# Tolerance: atol=1e-4 for fp32, atol=1e-2 for fp16
```

### MLX-Specific Challenges
- **AdaptiveAvgPool2d(7×7)**: Not native in MLX → implement as reshape + mean over pooled regions
- **DWConv (groups=C)**: MLX supports group convolutions — verify performance
- **Cosine similarity**: Manual implementation: `dot(a,b) / (norm(a) * norm(b))`
- **DropPath**: Stochastic depth → implement with mlx.random

### Acceptance Criteria
- [ ] Full model runs on MLX (Mac Studio M-series)
- [ ] FMB mIoU matches CUDA within 0.5%
- [ ] Weight conversion script works both directions
- [ ] device.py abstraction layer functional
- [ ] `ANIMA_BACKEND=mlx python eval.py` works end-to-end
- [ ] MLX inference FPS measured and recorded

---

## PRD-008: Edge Deployment — Jetson Orin NX (8h) ⬜
**Priority**: P0 — DEMO PRIORITY for Shenzhen partner
**Dependencies**: PRD-003, PRD-006 (Kernel 4)

### Steps
```bash
# 1. ONNX export
python export_onnx.py \
    --checkpoint pretrained/TUNI_FMB.pth \
    --config backbone_384_2242 \
    --input_size 480 640 \
    --output tuni_fmb.onnx

# 2. TensorRT conversion on Jetson
trtexec --onnx=tuni_fmb.onnx \
    --saveEngine=tuni_fmb_fp16.trt \
    --fp16 \
    --workspace=2048

# 3. INT8 calibration
python calibrate_int8.py \
    --onnx tuni_fmb.onnx \
    --calibration_data $DATASET_ROOT/FMB/train/ \
    --num_samples 500

trtexec --onnx=tuni_fmb.onnx \
    --saveEngine=tuni_fmb_int8.trt \
    --int8 \
    --calib=calibration_cache.bin

# 4. Custom TRT plugins for fused kernels (from PRD-006 Kernel 4)
# Register LocalAttentionRGBT and GlobalAttention as TRT plugins

# 5. Real-time demo pipeline
# Camera input → preprocess → TRT inference → segmentation overlay → display
# Target: 27+ FPS sustained with visualization

# 6. Thermal camera integration
# Connect actual thermal camera (or use recorded thermal data)
# Verify real-time RGB-T fusion on live streams
```

### FPS Targets on Jetson Orin NX
| Precision | Paper Claim | Target (with kernels) |
|-----------|------------|----------------------|
| FP32 | ~15 FPS | ~20 FPS |
| FP16 | 27 FPS | 35+ FPS |
| INT8 | N/A | 45+ FPS |

### Acceptance Criteria
- [ ] ONNX export successful, validated with onnxruntime
- [ ] TRT FP16 engine runs at 27+ FPS on Jetson Orin NX
- [ ] TRT INT8 engine runs at 40+ FPS with <1% mIoU drop
- [ ] FMB mIoU with TRT FP16 within 1% of PyTorch reference
- [ ] Real-time demo pipeline with visualization working
- [ ] Power consumption measured (Jetson has power monitoring)
- [ ] Memory footprint < 4GB (Jetson Orin NX has 8/16GB)

---

## PRD-009: RGB-T Family Integration & Comparison (3h) ⬜
**Priority**: P2 — cross-module analysis
**Dependencies**: PRD-003

### Cross-Module Comparison
```bash
# Compare TUNI with other RGB-T modules on shared datasets:
# 1. DEF-rtfdnet (RTFDNet — SegFormer dual-stream, robustness focus)
# 2. DEF-cmssm (CMSSM — Mamba state-space, same author as TUNI)
# 3. DEF-hypsam (HyPSAM — RGB-T SOD + SAM)
# 4. DEF-rgbtcc (RGB-T Crowd Counting)

# Shared datasets: FMB, PST900
# Compare: mIoU, FPS, params, GFLOPs

# TUNI's unique advantage: real-time (27 FPS) + unified encoder
# RTFDNet's advantage: modality degradation robustness
# CMSSM's advantage: linear attention via Mamba (same author, complementary)
```

### Shared Kernel Opportunities
- eaef_fused.cu (from DEF-rtfdnet) ↔ local_rgbt_attn.cu (from DEF-tuni): different fusion but similar cross-modal attention
- Both share thermal preprocessing pipelines
- Dataset loading can be shared (FMB, PST900 configs identical)

### Acceptance Criteria
- [ ] Comparison table: TUNI vs RTFDNet vs CMSSM on FMB + PST900
- [ ] Speed comparison: FPS on same hardware (RTX 6000 Pro)
- [ ] Accuracy-speed tradeoff plot (Pareto frontier)
- [ ] Shared kernel reuse documented
- [ ] Integration with DEF-cmssm (same author) validated

---

## PRD-010: Ablation Studies & Attention Analysis (2h) ⬜
**Priority**: P2 — understanding architecture
**Dependencies**: PRD-003

### Ablation Experiments
```bash
# 1. Attention type ablation
# a) RGB-RGB Local only (disable cross-modal)
# b) RGB-T Local only (disable global)
# c) RGB-T Global only (disable local)
# d) All three (full model)

# 2. Thermal channel dimension ablation
# Default: thermal at C/2
# Test: thermal at C/4, C/3, C (full), 0 (RGB-only)

# 3. Pre-training ablation
# a) With sRGB-TIR pre-training
# b) Without pre-training (random init)
# c) ImageNet RGB-only pre-training (no pseudo-thermal)

# 4. Model variant comparison
# Tiny [32,64,128,256] vs 384_2242 [48,96,192,384] vs 512_2242 vs 320_2262
```

### Acceptance Criteria
- [ ] Attention ablation: mIoU for each combination on FMB
- [ ] Thermal channel ablation: mIoU vs channel ratio
- [ ] Pre-training ablation: mIoU with/without sRGB-TIR pre-training
- [ ] Model variant comparison: accuracy vs speed across all 4 configs
- [ ] Results saved to benchmarks/ablations/

---

## Build Plan Summary

| PRD | Task | Hours | Status | Depends On |
|-----|------|-------|--------|-----------|
| 001 | Environment Setup | 5h | ⬜ | — |
| 002 | Dataset Download | 5h | ⬜ | 001 |
| 003 | Evaluation Baseline | 6h | ⬜ | 001, 002 |
| 004 | FPS Benchmarking | 4h | ⬜ | 001 |
| 005 | CUDA Profiling | 5h | ⬜ | 001, 003 |
| 006 | Custom CUDA Kernels | 16h | ⬜ | 005 |
| 007 | MLX Port | 8h | ⬜ | 003 |
| 008 | Edge Deployment Jetson | 8h | ⬜ | 003, 006 |
| 009 | RGB-T Family Integration | 3h | ⬜ | 003 |
| 010 | Ablation Studies | 2h | ⬜ | 003 |
| **TOTAL** | | **62h** | | |

## Critical Path
```
PRD-001 (env) → PRD-002 (data) → PRD-003 (baseline) → PRD-005 (profile) → PRD-006 (kernels) → PRD-008 (Jetson demo)
                                                       ↘ PRD-004 (FPS) — can run parallel with 005
                                                       ↘ PRD-007 (MLX) — can run parallel with 005/006
                                                       ↘ PRD-009 (integration) — can run after 003
                                                       ↘ PRD-010 (ablation) — can run after 003
```

---
*Updated 2026-04-04 by ANIMA Research Agent*
