"""TUNI Serving Node — AnimaNode subclass for RGB-T segmentation."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .model import TUNIModel


class TUNINode:
    """TUNI inference server for RGB-T semantic segmentation."""

    def __init__(self):
        self.model = None
        self.device = None
        self.n_classes = 15  # FMB default

    def setup_inference(self):
        device_str = os.environ.get("ANIMA_DEVICE", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.model = TUNIModel(variant="384_2242", n_classes=self.n_classes)

        weight_dir = Path(os.environ.get("ANIMA_WEIGHT_DIR", "/data/weights"))
        weight_format = os.environ.get("ANIMA_WEIGHT_FORMAT", "auto")

        # Try loading in priority order: safetensors > pth
        safetensors_path = weight_dir / "tuni_fmb.safetensors"
        pth_path = weight_dir / "tuni_fmb.pth"

        if safetensors_path.exists():
            from safetensors.torch import load_file
            sd = load_file(str(safetensors_path))
            self.model.load_state_dict(sd)
            print(f"[TUNI] Loaded weights from {safetensors_path}")
        elif pth_path.exists():
            sd = torch.load(pth_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(sd)
            print(f"[TUNI] Loaded weights from {pth_path}")
        else:
            print("[TUNI] WARNING: No weights found, using random initialization")

        self.model.to(self.device).eval()
        print(f"[TUNI] Model ready on {self.device}")

    @torch.no_grad()
    def process(self, rgb: np.ndarray, thermal: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            rgb: RGB image [H, W, 3] uint8
            thermal: Thermal image [H, W, 3] uint8 (or [H, W, 1])

        Returns:
            Segmentation mask [H, W] int32
        """
        from torchvision import transforms

        rgb_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(rgb).unsqueeze(0).to(self.device)

        thermal_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])(thermal).unsqueeze(0).to(self.device)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            pred = self.model(rgb_t, thermal_t)

        return pred.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    def get_status(self) -> dict:
        return {
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "n_classes": self.n_classes,
            "variant": "384_2242",
        }
