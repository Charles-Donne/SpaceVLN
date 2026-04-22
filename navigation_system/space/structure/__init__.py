"""Space structure: area types, connectivity, and waypoints."""

from navigation_system.space.structure.space_types import (
    normalize_space_type,
    strip_space_type_variant_suffixes,
)
from navigation_system.space.structure.space_area_manager import SpaceAreaManager
from navigation_system.space.structure.waypoint_manager import WaypointManager

__all__ = [
    "SpaceAreaManager",
    "WaypointManager",
    "normalize_space_type",
    "strip_space_type_variant_suffixes",
]
