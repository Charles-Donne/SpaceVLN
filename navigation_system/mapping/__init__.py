"""Mapping package with lazy exports to avoid import-time side effects."""

from importlib import import_module
from typing import Any

__all__ = [
    "Semantic_Mapping",
    "SemanticMapper",
    "WaypointManager",
    "SpaceAreaManager",
    "SemanticProcessor",
]


def __getattr__(name: str) -> Any:
    if name == "Semantic_Mapping":
        return getattr(import_module("navigation_system.mapping.semantic_mapping"), name)
    if name == "SemanticMapper":
        return getattr(import_module("navigation_system.mapping.mapper"), name)
    if name == "SemanticProcessor":
        return getattr(import_module("navigation_system.mapping.processor"), name)
    if name == "SpaceAreaManager":
        return getattr(import_module("navigation_system.mapping.space_area_manager"), name)
    if name == "WaypointManager":
        return getattr(import_module("navigation_system.mapping.waypoint_manager"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
