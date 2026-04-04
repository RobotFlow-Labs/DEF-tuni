#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

DATASETS = ["FMB", "PST900", "CART"]
RESOLUTIONS = ["240x320", "480x640", "720x960", "1080x1920"]
BATCH_SIZES = [1, 2, 4, 8]


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> dict:
    started = time.time()
    if dry_run:
        return {"cmd": cmd, "returncode": 0, "elapsed_s": 0.0, "dry_run": True}
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="benchmarks/benchmark_matrix.json")
    args = parser.parse_args()

    root = Path.cwd()
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "datasets": {},
        "fps": [],
    }

    for dataset in DATASETS:
        cmd = ["def-tuni", "eval", "--dataset", dataset, "--dry-run"]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        results["datasets"][dataset] = run_cmd(cmd, root, args.dry_run)

    for batch in BATCH_SIZES:
        for res in RESOLUTIONS:
            cmd = [
                "def-tuni",
                "fps",
                "--dry-run",
            ]
            row = run_cmd(cmd, root, args.dry_run)
            row.update({"batch_size": batch, "resolution": res})
            results["fps"].append(row)

    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote benchmark matrix scaffold to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

