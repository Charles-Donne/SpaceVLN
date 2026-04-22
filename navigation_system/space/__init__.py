"""Space subsystem: structure, landmark memory, map state, geometry, and descriptions."""

from navigation_system.space.landmarks import LandmarkMemory
from navigation_system.space.map import SemanticMapper, SemanticProcessor
from navigation_system.space.structure import (
    normalize_space_type,
    strip_space_type_variant_suffixes,
)

__all__ = [
    "LandmarkMemory",
    "SemanticMapper",
    "SemanticProcessor",
    "normalize_space_type",
    "strip_space_type_variant_suffixes",
]
