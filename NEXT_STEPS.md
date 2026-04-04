# NEXT_STEPS — DEF-tuni
## Last Updated: 2026-04-04
## Status: TRAINING — Full pipeline built, training in progress
## MVP Readiness: 75%

---

### Completed
- [x] **Environment**: venv + torch cu128 + mmengine + all deps
- [x] **Datasets**: FMB (1220+280), PST900 (597+288) downloaded and verified
- [x] **Model**: Self-contained TUNI model (no mmcv dependency), 10.6M params
- [x] **Pretrained weights**: All 3 checkpoints verified (FMB 62.4% mIoU, PST900, CART)
- [x] **Training pipeline**: Config-driven, warmup+cosine, checkpointing, early stopping
- [x] **CUDA kernels**: 4 ops compiled (concat_norm, seg_argmax, local_rgbt_attn, global_pool_7x7)
- [x] **Exports**: pth + safetensors + ONNX + TRT FP16 + TRT FP32
- [x] **Docker**: Dockerfile.serve + docker-compose.serve.yml + anima_module.yaml
- [x] **HuggingFace**: Pushed to ilessio-aiflowlab/DEF-tuni

### In Progress
- [ ] **Training**: FMB 80 epochs on GPU 5 (currently at epoch ~10/80, ~59% mIoU)
  - Monitor: `tail -f /mnt/artifacts-datai/logs/DEF-tuni/train_*.log`
  - Checkpoints: `/mnt/artifacts-datai/checkpoints/DEF-tuni/`

### Remaining
- [ ] **PST900 training**: Train on PST900 dataset (5-class)
- [ ] **CART training**: Download CART dataset and train (12-class)
- [ ] **CART dataset**: Awaiting download from Caltech portal
- [ ] **FPS benchmark**: Profile inference speed on L4
- [ ] **MLX port**: Port unified encoder to Apple Silicon
- [ ] **Re-export**: After training completes, export fine-tuned model

### Key Results
| Dataset | Pretrained mIoU | Training Status |
|---------|----------------|-----------------|
| FMB     | 62.42%         | In progress (80 epochs) |
| PST900  | Verified       | Not started     |
| CART    | Verified       | Dataset pending |

### CUDA Kernels (shared at /mnt/forge-data/shared_infra/cuda_extensions/tuni_rgbt_attention/)
1. `tuni_ops.cu` — fused RGB-T concat+norm + seg_argmax
2. `local_rgbt_attn.cu` — fused DWConv7x7 co-occurrence + difference
3. `global_rgbt_attn.cu` — fused adaptive avg pool 7x7

---
*Updated 2026-04-04 by ANIMA Autopilot*
