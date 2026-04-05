"""TUNI ROS2 Node — RGB-T Semantic Segmentation.

Subscribes to RGB + Thermal image topics, publishes segmentation masks.
Runs as AnimaNode subclass with FastAPI health endpoints.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

from .model import TUNIModel


class TUNISegmentationNode:
    """ROS2 node for real-time RGB-T semantic segmentation."""

    NODE_NAME = "tuni_segmentation"

    def __init__(self):
        self.model = None
        self.device = None
        self.bridge = CvBridge() if HAS_ROS2 else None
        self.rgb_msg = None
        self.thermal_msg = None
        self.n_classes = 15
        self._setup_complete = False

    def setup_inference(self):
        device_str = os.environ.get("ANIMA_DEVICE", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        n_classes = int(os.environ.get("ANIMA_N_CLASSES", "15"))
        self.n_classes = n_classes

        self.model = TUNIModel(variant="384_2242", n_classes=n_classes).to(self.device)

        weight_dir = Path(os.environ.get("ANIMA_WEIGHT_DIR", "/data/weights"))

        # Try safetensors first, then pth
        for fmt, loader in [
            ("safetensors", self._load_safetensors),
            ("pth", self._load_pth),
        ]:
            for name in ["tuni_fmb", "tuni_pst900", "tuni_cart", "best"]:
                path = weight_dir / f"{name}.{fmt}"
                if path.exists():
                    loader(path)
                    print(f"[TUNI-ROS2] Loaded {path}")
                    self._setup_complete = True
                    return

        print("[TUNI-ROS2] WARNING: No weights found")
        self._setup_complete = True

    def _load_safetensors(self, path: Path):
        from safetensors.torch import load_file
        self.model.load_state_dict(load_file(str(path)))
        self.model.eval()

    def _load_pth(self, path: Path):
        sd = torch.load(path, map_location=self.device, weights_only=True)
        if "model" in sd:
            sd = sd["model"]
        self.model.load_state_dict(sd)
        self.model.eval()

    @torch.no_grad()
    def process(self, rgb: np.ndarray, thermal: np.ndarray) -> np.ndarray:
        """Run segmentation inference.

        Args:
            rgb: [H, W, 3] uint8 BGR (ROS2 convention) or RGB
            thermal: [H, W, 3] uint8 or [H, W, 1] uint8

        Returns:
            [H, W] int32 class indices
        """
        from torchvision import transforms

        # Normalize
        rgb_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(rgb).unsqueeze(0).to(self.device)

        if thermal.ndim == 2:
            thermal = np.stack([thermal] * 3, axis=-1)
        elif thermal.shape[2] == 1:
            thermal = np.concatenate([thermal] * 3, axis=-1)

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
            "setup_complete": self._setup_complete,
            "device": str(self.device) if self.device else "none",
            "n_classes": self.n_classes,
        }


def create_ros2_node():
    """Create and return a ROS2-wrapped TUNI node."""
    if not HAS_ROS2:
        raise RuntimeError("rclpy not available — install ros-jazzy-rclpy")

    rclpy.init()

    class _TUNIRos2Node(Node):
        def __init__(self):
            super().__init__(TUNISegmentationNode.NODE_NAME)
            self.tuni = TUNISegmentationNode()
            self.tuni.setup_inference()
            self.bridge = CvBridge()

            self.rgb_sub = self.create_subscription(
                Image, "/camera/rgb/image_raw", self._rgb_cb, 10)
            self.thermal_sub = self.create_subscription(
                Image, "/camera/thermal/image_raw", self._thermal_cb, 10)
            self.seg_pub = self.create_publisher(
                Image, "/tuni/segmentation", 10)

            self._rgb = None
            self._thermal = None
            self.get_logger().info("[TUNI] ROS2 node ready")

        def _rgb_cb(self, msg):
            self._rgb = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            self._try_inference()

        def _thermal_cb(self, msg):
            self._thermal = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self._try_inference()

        def _try_inference(self):
            if self._rgb is None or self._thermal is None:
                return
            rgb, thermal = self._rgb, self._thermal
            self._rgb = self._thermal = None

            mask = self.tuni.process(rgb, np.stack([thermal] * 3, axis=-1))
            msg = self.bridge.cv2_to_imgmsg(mask.astype(np.uint8), "mono8")
            self.seg_pub.publish(msg)

    return _TUNIRos2Node()
