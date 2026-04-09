"""Space topology: area types, connectivity, and waypoints."""

from navigation_system.space.topology.space_types import (
    normalize_space_type,
    strip_space_type_variant_suffixes,
)
from navigation_system.space.topology.space_area_manager import SpaceAreaManager
from navigation_system.space.topology.waypoint_manager import WaypointManager

__all__ = [
    "SpaceAreaManager",
    "WaypointManager",
    "normalize_space_type",
    "strip_space_type_variant_suffixes",
]
