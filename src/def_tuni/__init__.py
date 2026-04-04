"""DEF-tuni package scaffolding."""

from .config import ModuleConfig
from .device import Backend, detect_backend

__all__ = ["Backend", "ModuleConfig", "detect_backend"]

