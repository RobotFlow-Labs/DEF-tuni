from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModuleConfig:
    root: Path
    upstream_repo: Path
    run_config_root: Path
    benchmark_root: Path

    @staticmethod
    def discover(start: Path | None = None) -> "ModuleConfig":
        root = (start or Path.cwd()).resolve()
        return ModuleConfig(
            root=root,
            upstream_repo=root / "repositories" / "TUNI",
            run_config_root=root / ".anima" / "run_configs",
            benchmark_root=root / "benchmarks",
        )

