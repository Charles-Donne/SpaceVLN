"""Navigation controllers and controller-local state."""

from importlib import import_module
from typing import Any

__all__ = [
    "BaseNavigationController",
    "EpisodeTimingTracker",
    "VLMControllerOptions",
    "VLMNavigationController",
]


def __getattr__(name: str) -> Any:
    if name in {"BaseNavigationController", "VLMNavigationController"}:
        module_name = (
            "navigation_system.controller.base_controller"
            if name == "BaseNavigationController"
            else "navigation_system.controller.navigation_controller"
        )
        return getattr(import_module(module_name), name)
    if name in {"EpisodeTimingTracker", "VLMControllerOptions"}:
        return getattr(import_module("navigation_system.controller.state"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
