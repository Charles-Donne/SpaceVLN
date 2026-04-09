"""Configuration package for SpaceVLN.

Layout:
- `system/`: external deployment defaults such as GPU, model paths, task paths, and camera/sensor setup
- `runtime/`: structured panels plus synchronization into Habitat/runtime-derived fields
- `core/`: static constants, categories, and algorithm/behavior default parameters
- `experiments/`: experiment YAML control panels
- `vlm/`: canonical VLM API config templates
"""

from importlib import import_module
from typing import Any

__all__ = [
    "CategoryConfig",
    "ConfigHelper",
    "create_category_config",
    "get_config",
]


def __getattr__(name: str) -> Any:
    if name == "get_config":
        return getattr(import_module("navigation_system.config.runtime.default"), name)
    if name in {"ConfigHelper", "CategoryConfig", "create_category_config"}:
        return getattr(import_module("navigation_system.config.core"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
