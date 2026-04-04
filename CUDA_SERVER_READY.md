# CUDA Server Readiness — DEF-tuni

## Status
Code scaffold is CUDA-server ready for execution, with the remaining runtime prerequisites being:
1. CUDA host with Python 3.11.
2. Dataset roots exported:
   - `ANIMA_DATASET_ROOT_FMB`
   - `ANIMA_DATASET_ROOT_PST900`
   - `ANIMA_DATASET_ROOT_CART`
3. Checkpoint paths exported:
   - `ANIMA_CKPT_FMB`
   - `ANIMA_CKPT_PST900`
   - `ANIMA_CKPT_CART`

## What was fixed for server execution
1. Removed broken `proposed.*` dependency path from active TUNI execution route.
2. Fixed `fps.py` to use `mode='TUNI'` and configurable CUDA device.
3. Made eval scripts use `ANIMA_CUDA_DEVICE` and robustly handle tuple outputs.
4. Added project runner (`def-tuni`) and CUDA autopilot script.

## One-command execution
```bash
cd /path/to/DEF-tuni
bash scripts/cuda_server_autopilot.sh
```

