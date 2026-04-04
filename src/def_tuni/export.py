"""Export TUNI model: pth → safetensors → ONNX → TRT FP16/FP32.

Usage:
    python -m def_tuni.export --checkpoint best.pth --variant 384_2242 --n-classes 15
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from .model import TUNIModel


def export_safetensors(state_dict: dict, output_path: Path):
    from safetensors.torch import save_file
    save_file(state_dict, str(output_path))
    print(f"[EXPORT] safetensors → {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def export_onnx(model: TUNIModel, output_path: Path, input_h: int = 480, input_w: int = 640):
    model.eval()
    device = next(model.parameters()).device
    rgb = torch.randn(1, 3, input_h, input_w, device=device)
    thermal = torch.randn(1, 3, input_h, input_w, device=device)

    torch.onnx.export(
        model, (rgb, thermal), str(output_path),
        input_names=["rgb", "thermal"],
        output_names=["segmentation"],
        dynamic_axes={
            "rgb": {0: "batch"},
            "thermal": {0: "batch"},
            "segmentation": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"[EXPORT] ONNX → {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def export_trt(onnx_path: Path, output_dir: Path, input_h: int = 480, input_w: int = 640):
    """Export TRT using shared toolkit."""
    trt_script = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
    if not trt_script.exists():
        print("[WARN] TRT toolkit not found, attempting trtexec directly")
        import subprocess
        for precision in ["fp16", "fp32"]:
            out = output_dir / f"tuni_{precision}.engine"
            flag = "--fp16" if precision == "fp16" else ""
            cmd = (
                f"trtexec --onnx={onnx_path} --saveEngine={out} {flag} "
                f"--minShapes=rgb:1x3x{input_h}x{input_w},thermal:1x3x{input_h}x{input_w} "
                f"--optShapes=rgb:1x3x{input_h}x{input_w},thermal:1x3x{input_h}x{input_w} "
                f"--maxShapes=rgb:4x3x{input_h}x{input_w},thermal:4x3x{input_h}x{input_w}"
            )
            print(f"[TRT] Building {precision}...")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[EXPORT] TRT {precision} → {out}")
            else:
                print(f"[WARN] TRT {precision} build failed: {result.stderr[-200:]}")
        return

    import subprocess
    for precision in ["fp16", "fp32"]:
        out = output_dir / f"tuni_{precision}.engine"
        cmd = [
            "python", str(trt_script),
            "--onnx", str(onnx_path),
            "--output", str(out),
            "--precision", precision,
        ]
        print(f"[TRT] Building {precision}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[EXPORT] TRT {precision} → {out}")
        else:
            print(f"[WARN] TRT {precision} build failed: {result.stderr[-200:]}")


def main():
    parser = argparse.ArgumentParser(description="TUNI Model Export")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--variant", default="384_2242")
    parser.add_argument("--n-classes", type=int, default=15)
    parser.add_argument("--output-dir", default="/mnt/artifacts-datai/exports/DEF-tuni")
    parser.add_argument("--input-h", type=int, default=480)
    parser.add_argument("--input-w", type=int, default=640)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TUNIModel(variant=args.variant, n_classes=args.n_classes).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get('model', ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[LOAD] Model loaded from {args.checkpoint}")

    # 1. Copy pth
    pth_out = output_dir / "tuni_best.pth"
    shutil.copy2(args.checkpoint, pth_out)
    print(f"[EXPORT] pth → {pth_out}")

    # 2. Safetensors
    st_out = output_dir / "tuni_best.safetensors"
    export_safetensors(model.state_dict(), st_out)

    # 3. ONNX
    onnx_out = output_dir / "tuni.onnx"
    export_onnx(model, onnx_out, args.input_h, args.input_w)

    # 4. TRT FP16 + FP32
    export_trt(onnx_out, output_dir, args.input_h, args.input_w)

    print(f"\n[DONE] All exports saved to {output_dir}")


if __name__ == "__main__":
    main()
