from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import ModuleConfig
from .device import Backend, describe_backend, detect_backend

DATASET_TO_CFG = {
    "FMB": "FMB.json",
    "PST900": "PST900.json",
    "CART": "CART.json",
}

EVAL_SCRIPT = {
    "FMB": "evaluate_FMB.py",
    "PST900": "evaluate_pst900.py",
    "CART": "evaluate_CART.py",
}

WEIGHT_ARG = {
    "FMB": "--model_weight",
    "PST900": "--model_pth",
    "CART": "--load_pth",
}


def _dataset_root_from_env(dataset: str) -> str | None:
    return os.getenv(f"ANIMA_DATASET_ROOT_{dataset}")


def _materialize_config(
    cfg: ModuleConfig, dataset: str, dataset_root: str | None
) -> Path:
    cfg.run_config_root.mkdir(parents=True, exist_ok=True)
    out_dir = cfg.run_config_root / dataset.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    src = cfg.upstream_repo / "configs" / DATASET_TO_CFG[dataset]
    dst = out_dir / DATASET_TO_CFG[dataset]
    shutil.copy2(src, dst)

    if dataset_root:
        payload = json.loads(dst.read_text())
        payload["root"] = dataset_root
        dst.write_text(json.dumps(payload, indent=2) + "\n")
    return out_dir


def _build_cuda_eval_cmd(
    cfg: ModuleConfig, dataset: str, checkpoint: str, dataset_root: str | None
) -> list[str]:
    logdir = _materialize_config(cfg, dataset, dataset_root)
    script = cfg.upstream_repo / EVAL_SCRIPT[dataset]
    return [
        sys.executable,
        str(script),
        "--logdir",
        str(logdir),
        WEIGHT_ARG[dataset],
        checkpoint,
        "-s",
        "False",
    ]


def _build_cuda_fps_cmd(cfg: ModuleConfig) -> list[str]:
    script = cfg.upstream_repo / "fps.py"
    return [sys.executable, str(script)]


def _build_mlx_stub_cmd(action: str, dataset: str | None = None) -> list[str]:
    cmd = [sys.executable, "scripts/mlx_eval_stub.py", "--action", action]
    if dataset:
        cmd.extend(["--dataset", dataset])
    return cmd


def _run(cmd: list[str], dry_run: bool, cwd: Path) -> int:
    print("backend command:", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd), check=False).returncode


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = ModuleConfig.discover()
    backend = detect_backend()
    dataset = args.dataset.upper()
    if dataset not in DATASET_TO_CFG:
        raise ValueError(f"Unsupported dataset: {dataset}")

    dataset_root = args.dataset_root or _dataset_root_from_env(dataset)
    print(f"backend: {backend.value} ({describe_backend(backend)})")
    if backend is Backend.CUDA:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for CUDA evaluation")
        cmd = _build_cuda_eval_cmd(cfg, dataset, args.checkpoint, dataset_root)
        return _run(cmd, args.dry_run, cfg.root)
    cmd = _build_mlx_stub_cmd(action="eval", dataset=dataset)
    return _run(cmd, args.dry_run, cfg.root)


def cmd_fps(args: argparse.Namespace) -> int:
    cfg = ModuleConfig.discover()
    backend = detect_backend()
    print(f"backend: {backend.value} ({describe_backend(backend)})")
    if backend is Backend.CUDA:
        cmd = _build_cuda_fps_cmd(cfg)
        return _run(cmd, args.dry_run, cfg.root)
    cmd = _build_mlx_stub_cmd(action="fps")
    return _run(cmd, args.dry_run, cfg.root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="def-tuni")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Run dataset evaluation wrapper.")
    p_eval.add_argument("--dataset", required=True, help="FMB | PST900 | CART")
    p_eval.add_argument("--checkpoint", help="Checkpoint path for CUDA backend.")
    p_eval.add_argument(
        "--dataset-root",
        help="Override dataset root path (else ANIMA_DATASET_ROOT_<DATASET>).",
    )
    p_eval.add_argument("--dry-run", action="store_true", help="Print command only.")
    p_eval.set_defaults(func=cmd_eval)

    p_fps = sub.add_parser("fps", help="Run FPS benchmark entrypoint.")
    p_fps.add_argument("--dry-run", action="store_true", help="Print command only.")
    p_fps.set_defaults(func=cmd_fps)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

