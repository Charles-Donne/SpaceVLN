"""Space subsystem: structure, landmark memory, map state, geometry, and descriptions."""

from navigation_system.space.landmarks import LandmarkMemory
from navigation_system.space.structure import (
    canonical_space_types_text,
    infer_space_type_from_texts,
    normalize_space_type,
    strip_space_type_variant_suffixes,
)


def __getattr__(name):
    if name in {"SemanticMapper", "SemanticProcessor"}:
        from navigation_system.space.map import SemanticMapper, SemanticProcessor

        values = {
            "SemanticMapper": SemanticMapper,
            "SemanticProcessor": SemanticProcessor,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LandmarkMemory",
    "SemanticMapper",
    "SemanticProcessor",
    "canonical_space_types_text",
    "infer_space_type_from_texts",
    "normalize_space_type",
    "strip_space_type_variant_suffixes",
]
