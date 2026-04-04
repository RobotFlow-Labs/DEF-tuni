# PRD — DEF-tuni (Wave 8)

## 1. Product Definition
- Module: `DEF-tuni`
- Scope: Real-time RGB-T semantic segmentation productization around TUNI (`repositories/TUNI`)
- Primary objective: Turn paper code into production-oriented ANIMA module with dual compute (`ANIMA_BACKEND=mlx|cuda`) and kernel IP pipeline
- Reference paper: `papers/2509.10005.pdf` (arXiv: `2509.10005`)

## 2. Problem and Value
- Problem: Existing TUNI reference code is research-grade and tightly coupled to hardcoded paths, one-off scripts, and CUDA-only assumptions.
- Value: Deployable RGB-T segmentation with edge-grade latency (Jetson path) and reproducible module workflow for ORACLE/ATLAS/NEMESIS stacks.

## 3. Goals
1. Reproducible baseline evaluation on FMB, PST900, CART.
2. Unified execution interface for CUDA and MLX backends.
3. Kernel optimization track with explicit stubs and benchmark protocol.
4. Deployment track toward TensorRT edge runtime.
5. Artifact discipline: PRD, sliced tasks, benchmarks, and kernel source ownership.

## 4. Non-Goals (Current Sprint)
1. Full retraining from scratch.
2. Completing all custom kernel implementations in one pass.
3. Shipping full MLX parity implementation in this scaffold pass.

## 5. Current Code Reality (Read from `repositories/TUNI`)
1. Core model pathing is hardcoded under `proposed.*` import namespace and absolute checkpoint paths.
2. Eval scripts read JSON from a `logdir` folder and expect local `.pth` paths.
3. Config roots point to `/home/ubuntu/dataset/...`.
4. Backend handling is implicit CUDA, with no explicit MLX abstraction.
5. Architecture bottlenecks align with local/global RGB-T attention and multi-op block execution.

## 6. Functional Requirements
1. Provide module-level runner for eval/fps orchestration.
2. Support backend selection via `ANIMA_BACKEND`.
3. Normalize dataset path management.
4. Keep upstream paper repo untouched; integrate through wrappers.
5. Provide benchmark output location and schema.
6. Provide kernel source placeholders in `kernels/` for ownership and iteration.

## 7. Non-Functional Requirements
1. Python `>=3.11`.
2. Package/build via `uv` and `hatchling`.
3. Reproducible command-line interface and deterministic file layout.
4. Separation of concerns: upstream reference code vs ANIMA module wrapper.

## 8. Architecture (Scaffold Target)
1. `src/def_tuni/device.py`: backend selection and validation.
2. `src/def_tuni/config.py`: module paths and dataset roots.
3. `src/def_tuni/runner.py`: single entrypoint for eval/fps commands.
4. `scripts/bootstrap_uv.sh`: environment bootstrap for local setup.
5. `benchmarks/run_benchmark_matrix.py`: matrix benchmark orchestrator.
6. `kernels/cuda/*`, `kernels/trt/*`, `kernels/mlx/*`: optimization stubs and ownership points.

## 9. Delivery Plan (PRD Slices)
1. PRD-001 Environment bootstrap and dependency lock.
2. PRD-002 Dataset root normalization and validation hooks.
3. PRD-003 Baseline eval wrappers for FMB/PST900/CART.
4. PRD-004 FPS benchmark matrix (resolution and batch sweeps).
5. PRD-005 CUDA profile protocol (Nsight + torch profiler).
6. PRD-006 Kernel implementation iteration (local/global/block fusion).
7. PRD-007 MLX parity implementation for encoder/decoder path.
8. PRD-008 TensorRT edge export and runtime validation.
9. PRD-009 Cross-module comparison with DEF-rtfdnet/DEF-cmssm.
10. PRD-010 Ablation protocol and report.

## 10. Acceptance Criteria
1. `def-tuni` CLI can run dry-run eval/fps for each dataset and backend.
2. Dataset/checkpoint roots configurable without editing upstream repo.
3. Benchmark script emits JSON into `benchmarks/`.
4. Kernel and TRT stubs exist with clear IO signatures.
5. Task backlog is explicit, ordered, and dependency-aware.

## 11. Risks and Mitigations
1. Upstream import mismatch (`proposed.*`) blocks execution.
   - Mitigation: wrapper-level path injection and staged compatibility patch in PRD-003.
2. Missing datasets/checkpoints prevents runtime validation.
   - Mitigation: dry-run mode and dataset contract checks before execution.
3. MLX feature mismatch vs PyTorch ops.
   - Mitigation: staged parity path with explicit MLX stubs and progressive operator replacement.

## 12. Definition of Done (Scaffold Phase)
1. Project-level PRD is complete.
2. Tasks are sliced and tracked.
3. Dual-backend module scaffolding exists and is executable in dry-run mode.
4. Kernel and benchmark starting points are in place for immediate implementation.
