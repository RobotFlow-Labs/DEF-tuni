#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=["eval", "fps"])
    parser.add_argument("--dataset")
    args = parser.parse_args()

    if args.action == "eval":
        print(
            f"[MLX scaffold] Eval request accepted for dataset={args.dataset}. "
            "Implement PRD-007 MLX parity path next."
        )
        return 0

    print("[MLX scaffold] FPS request accepted. Implement MLX profiling path in PRD-007.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

