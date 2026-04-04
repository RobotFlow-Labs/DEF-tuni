"""DEF-tuni: TUNI Real-Time RGB-T Semantic Segmentation."""

from .config import ModuleConfig
from .device import Backend, detect_backend
from .model import TUNIModel, BACKBONE_CONFIGS

__all__ = ["Backend", "ModuleConfig", "detect_backend", "TUNIModel", "BACKBONE_CONFIGS"]
