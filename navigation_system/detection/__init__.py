"""Detection package with lazy model import."""

from importlib import import_module
from typing import Any

__all__ = ["GroundedSAM"]


def __getattr__(name: str) -> Any:
    if name == "GroundedSAM":
        return getattr(import_module("navigation_system.detection.grounded_sam"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
