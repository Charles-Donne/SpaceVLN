"""Visualization package with lazy exports and no eager heavy imports."""

from importlib import import_module
from typing import Any

__all__ = ["MapVisualizer", "PanoramaGenerator"]


def __getattr__(name: str) -> Any:
    if name == "MapVisualizer":
        return getattr(import_module("navigation_system.visualization.visualizer"), name)
    if name == "PanoramaGenerator":
        return getattr(import_module("navigation_system.visualization.panorama_generator"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
