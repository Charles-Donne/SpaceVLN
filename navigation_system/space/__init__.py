"""Space subsystem: map, topology, landmarks, geometry, and prompt-facing descriptions."""

from importlib import import_module
from typing import Any

__all__ = [
    "LandmarkMemoryPool",
    "SemanticMapper",
    "SemanticProcessor",
    "normalize_space_type",
    "strip_space_type_variant_suffixes",
]


def __getattr__(name: str) -> Any:
    if name == "SemanticMapper":
        return getattr(import_module("navigation_system.space.map.semantic_mapper"), name)
    if name == "SemanticProcessor":
        return getattr(import_module("navigation_system.space.map.semantic_processor"), name)
    if name == "LandmarkMemoryPool":
        return getattr(import_module("navigation_system.space.landmarks"), name)
    if name in {"normalize_space_type", "strip_space_type_variant_suffixes"}:
        return getattr(import_module("navigation_system.space.topology"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
