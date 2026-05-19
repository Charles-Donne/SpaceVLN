"""Space structure: area types, connectivity, and waypoints."""

from navigation_system.space.structure.space_types import (
    canonical_space_types_text,
    infer_space_type_from_texts,
    normalize_space_type,
    strip_space_type_variant_suffixes,
)
from navigation_system.space.structure.region_manager import RegionManager
from navigation_system.space.structure.waypoint_manager import WaypointManager

__all__ = [
    "RegionManager",
    "WaypointManager",
    "canonical_space_types_text",
    "infer_space_type_from_texts",
    "normalize_space_type",
    "strip_space_type_variant_suffixes",
]
