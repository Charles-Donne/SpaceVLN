"""Space subsystem: structure, landmark memory, map state, geometry, and descriptions."""

from navigation_system.space.landmarks import LandmarkMemory
from navigation_system.space.map import SemanticMapper, SemanticProcessor
from navigation_system.space.structure import (
    canonical_space_types_text,
    infer_space_type_from_texts,
    normalize_space_type,
    strip_space_type_variant_suffixes,
)

__all__ = [
    "LandmarkMemory",
    "SemanticMapper",
    "SemanticProcessor",
    "canonical_space_types_text",
    "infer_space_type_from_texts",
    "normalize_space_type",
    "strip_space_type_variant_suffixes",
]
