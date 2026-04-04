#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -d "repositories/TUNI" ]]; then
  echo "Missing repositories/TUNI"
  exit 1
fi

bash scripts/bootstrap_uv.sh
source .venv/bin/activate

export ANIMA_BACKEND=cuda
export ANIMA_CUDA_DEVICE="${ANIMA_CUDA_DEVICE:-0}"

if [[ -z "${ANIMA_DATASET_ROOT_FMB:-}" || -z "${ANIMA_DATASET_ROOT_PST900:-}" || -z "${ANIMA_DATASET_ROOT_CART:-}" ]]; then
  echo "Set ANIMA_DATASET_ROOT_FMB, ANIMA_DATASET_ROOT_PST900, ANIMA_DATASET_ROOT_CART"
  exit 1
fi

if [[ -z "${ANIMA_CKPT_FMB:-}" || -z "${ANIMA_CKPT_PST900:-}" || -z "${ANIMA_CKPT_CART:-}" ]]; then
  echo "Set ANIMA_CKPT_FMB, ANIMA_CKPT_PST900, ANIMA_CKPT_CART"
  exit 1
fi

def-tuni eval --dataset FMB --checkpoint "${ANIMA_CKPT_FMB}"
def-tuni eval --dataset PST900 --checkpoint "${ANIMA_CKPT_PST900}"
def-tuni eval --dataset CART --checkpoint "${ANIMA_CKPT_CART}"

def-tuni fps
python benchmarks/run_benchmark_matrix.py --checkpoint "${ANIMA_CKPT_FMB}"

echo "CUDA autopilot completed."

