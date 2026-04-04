#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed."
  exit 1
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

uv venv .venv --python "${PYTHON_VERSION}"
source .venv/bin/activate

uv pip install -e ".[cuda]"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install mmcv==2.2.0 mmengine fvcore thop ptflops timm einops scipy
uv pip install pillow tqdm scikit-learn matplotlib opencv-python

echo "Bootstrap complete."
echo "Use:"
echo "  source .venv/bin/activate"
echo "  ANIMA_BACKEND=cuda def-tuni eval --dataset FMB --checkpoint /path/to/model.pth --dry-run"
echo "  ANIMA_BACKEND=mlx  def-tuni eval --dataset FMB --dry-run"
