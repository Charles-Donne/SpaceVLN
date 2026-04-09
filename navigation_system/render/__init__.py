"""Rendering subsystem for maps, model views, and episode visualization."""

from importlib import import_module
from typing import Any

__all__ = ["MapVisualizer", "PanoramaGenerator"]


def __getattr__(name: str) -> Any:
    if name == "MapVisualizer":
        return getattr(import_module("navigation_system.render.map.map_visualizer"), name)
    if name == "PanoramaGenerator":
        return getattr(import_module("navigation_system.render.views.panorama_generator"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
