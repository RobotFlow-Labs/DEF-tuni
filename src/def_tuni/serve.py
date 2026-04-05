"""TUNI Serving Node — AnimaNode subclass for RGB-T segmentation.

Works in both ROS2 and API-only modes via anima_serve infrastructure.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from anima_serve.node import AnimaNode
    from anima_serve.health import NodeStatus
    HAS_ANIMA_SERVE = True
except ImportError:
    HAS_ANIMA_SERVE = False

from .model import TUNIModel


class TUNINode(AnimaNode if HAS_ANIMA_SERVE else object):
    """TUNI inference server for RGB-T semantic segmentation.

    Implements AnimaNode interface:
      - setup_inference(): load model + weights
      - process(input): run segmentation
      - get_status(): return health info
    """

    def setup_inference(self) -> None:
        """Load TUNI model and weights from weight_manager."""
        from torchvision import transforms

        n_classes = int(getattr(self.config, "n_classes", 15))
        self.n_classes = n_classes
        self.model = TUNIModel(variant="384_2242", n_classes=n_classes)

        # Download/select weights via anima_serve weight manager
        weight_path = self.weight_manager.select_checkpoint()
        if weight_path:
            sd = torch.load(weight_path, map_location=self._device, weights_only=True)
            if "model" in sd:
                sd = sd["model"]
            self.model.load_state_dict(sd, strict=True)
            logger.info("[TUNI] Loaded weights from %s", weight_path)
        else:
            logger.warning("[TUNI] No weights found — random initialization")

        self.model.to(self._device).eval()
        self.health.set_active_backends(["pytorch"])

        # Pre-build transforms
        self._rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._thermal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        logger.info("[TUNI] Model ready on %s (%d classes)", self._device, n_classes)

    @torch.no_grad()
    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run RGB-T segmentation.

        Args:
            input_data: dict with 'rgb' (H,W,3 uint8) and 'thermal' (H,W,3 uint8)

        Returns:
            dict with 'segmentation' (H,W int32), 'n_classes' (int)
        """
        with self.health.measure_latency():
            rgb = input_data["rgb"]  # np.ndarray [H, W, 3]
            thermal = input_data.get("thermal", rgb)  # fallback to RGB if no thermal

            # Handle single-channel thermal
            if thermal.ndim == 2:
                thermal = np.stack([thermal] * 3, axis=-1)
            elif thermal.shape[-1] == 1:
                thermal = np.concatenate([thermal] * 3, axis=-1)

            rgb_t = self._rgb_transform(rgb).unsqueeze(0).to(self._device)
            thermal_t = self._thermal_transform(thermal).unsqueeze(0).to(self._device)

            with torch.amp.autocast("cuda", dtype=torch.float16,
                                     enabled=self._device.type == "cuda"):
                pred = self.model(rgb_t, thermal_t)

            mask = pred.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

        return {
            "segmentation": mask,
            "n_classes": self.n_classes,
            "shape": list(mask.shape),
        }

    def get_status(self) -> "NodeStatus":
        """Return current node status for health endpoint."""
        snapshot = self.health.snapshot()
        snapshot.extra.update({
            "model_loaded": self.model is not None,
            "variant": "384_2242",
            "n_classes": self.n_classes,
            "device": str(self._device),
        })
        return snapshot.status


# Standalone fallback (no anima_serve installed)
if not HAS_ANIMA_SERVE:
    class TUNINodeStandalone:
        """Minimal serve node when anima_serve is not available."""

        def __init__(self):
            self.model = None
            self.device = None
            self.n_classes = 15

        def setup_inference(self):
            import os
            from pathlib import Path

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = TUNIModel(variant="384_2242", n_classes=self.n_classes).to(self.device)

            weight_dir = Path(os.environ.get("ANIMA_WEIGHT_DIR", "/data/weights"))
            for name in ["tuni_fmb.safetensors", "tuni_fmb.pth", "best.pth"]:
                path = weight_dir / name
                if path.exists():
                    if name.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        self.model.load_state_dict(load_file(str(path)))
                    else:
                        sd = torch.load(path, map_location=self.device, weights_only=True)
                        self.model.load_state_dict(sd.get("model", sd))
                    break

            self.model.eval()
            logger.info("[TUNI-standalone] Ready on %s", self.device)

    TUNINode = TUNINodeStandalone  # type: ignore[misc]
