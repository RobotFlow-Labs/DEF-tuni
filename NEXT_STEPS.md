# NEXT_STEPS — DEF-tuni
## Last Updated: 2026-04-05
## Status: COMPLETE — All 3 datasets trained, exported, pushed to HF
## MVP Readiness: 95%

---

### Completed
- [x] **Environment**: venv + torch 2.11 cu128 + all deps
- [x] **Datasets**: FMB (1500), PST900 (885), CART (2282) — all verified
- [x] **Model**: Self-contained TUNI (no mmcv), 10.6M params, 4 backbone variants
- [x] **Pretrained weights**: All 3 checkpoints loaded (587/587 keys)
- [x] **Training — FMB**: 73 epochs (early stop), **mIoU=61.87%**, pixel_acc=92.02%
- [x] **Training — PST900**: 22 epochs (early stop), **mIoU=84.55%**, pixel_acc=99.51%
- [x] **Training — CART**: 21 epochs (early stop), **mIoU=69.67%**, pixel_acc=93.20%
- [x] **CUDA kernels**: 4 ops (concat_norm, seg_argmax, local_rgbt_attn, global_pool_7x7)
- [x] **Exports**: pth + safetensors (all 3 datasets) + ONNX + TRT FP16 + TRT FP32 (FMB)
- [x] **Docker**: Dockerfile.serve + docker-compose.serve.yml + anima_module.yaml
- [x] **HuggingFace**: ilessio-aiflowlab/DEF-tuni (all exports pushed)
- [x] **Shared infra**: Kernels at shared_infra/cuda_extensions/tuni_rgbt_attention/

### Key Results

| Dataset | Classes | Best mIoU | Pixel Acc | Epochs |
|---------|---------|-----------|-----------|--------|
| FMB     | 15      | 61.87%    | 92.02%    | 73     |
| PST900  | 5       | 84.55%    | 99.51%    | 22     |
| CART    | 12      | 69.67%    | 93.20%    | 21     |

### Remaining (nice-to-have)
- [ ] ONNX + TRT export for PST900 and CART models
- [ ] FPS benchmark on L4 (target: 27+ FPS at 480×640)
- [ ] MLX port for Apple Silicon
- [ ] Edge deployment profiling (Jetson Orin NX target)

### Artifacts
- Checkpoints: `/mnt/artifacts-datai/checkpoints/DEF-tuni{,-pst900,-cart}/best.pth`
- Exports: `/mnt/artifacts-datai/exports/DEF-tuni/`
- Logs: `/mnt/artifacts-datai/logs/DEF-tuni/`
- CUDA kernels: `/mnt/forge-data/shared_infra/cuda_extensions/tuni_rgbt_attention/`

---
*Updated 2026-04-05 by ANIMA Autopilot*
