# DEF-tuni Autopilot Task Slices

## Phase 1 — Foundation (P0)
- [x] `PRD-001` Bootstrap environment with `uv` and build metadata.
- [ ] `PRD-002` Normalize dataset roots and checkpoint contracts.
- [ ] `PRD-003` Baseline eval wrappers for FMB/PST900/CART.

## Phase 2 — Performance Baseline (P0/P1)
- [ ] `PRD-004` FPS benchmark matrix (batch, resolution, variant).
- [ ] `PRD-005` CUDA profiling with per-block attention timing.

## Phase 3 — Kernel IP (P1)
- [ ] `PRD-006-A` Fused `LocalAttentionRGBT` kernel.
- [ ] `PRD-006-B` Fused global RGB-T attention kernel.
- [ ] `PRD-006-C` Fused TUNI block kernel.
- [ ] `PRD-006-D` TensorRT plugin packaging path.

## Phase 4 — Dual Backend and Edge (P1/P0)
- [ ] `PRD-007` MLX parity path for core inference.
- [ ] `PRD-008` Jetson/TensorRT edge deployment validation.

## Phase 5 — Productization (P2)
- [ ] `PRD-009` Cross-module comparison report (TUNI, RTFDNet, CMSSM).
- [ ] `PRD-010` Ablation report and release benchmark sheet.

## Immediate Execution Order
1. `PRD-001`
2. `PRD-002`
3. `PRD-003`
4. `PRD-004`
5. `PRD-005`
6. `PRD-006`
7. `PRD-007`
8. `PRD-008`
9. `PRD-009`
10. `PRD-010`
