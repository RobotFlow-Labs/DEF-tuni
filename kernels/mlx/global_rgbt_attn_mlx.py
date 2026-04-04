"""DEF-tuni MLX scaffold for global RGB-T attention (PRD-007)."""

from __future__ import annotations


def global_rgbt_attn(rgb, thermal):
    return (rgb + thermal) * 0.5

