from __future__ import annotations

import os
from enum import Enum


class Backend(str, Enum):
    CUDA = "cuda"
    MLX = "mlx"


def detect_backend() -> Backend:
    raw = os.getenv("ANIMA_BACKEND", Backend.CUDA.value).strip().lower()
    if raw not in {Backend.CUDA.value, Backend.MLX.value}:
        raise ValueError(
            f"Unsupported ANIMA_BACKEND={raw!r}. Expected 'cuda' or 'mlx'."
        )
    return Backend(raw)


def describe_backend(backend: Backend) -> str:
    if backend is Backend.CUDA:
        return "CUDA backend (GPU server / PyTorch runtime)"
    return "MLX backend (Apple Silicon / MLX runtime)"

