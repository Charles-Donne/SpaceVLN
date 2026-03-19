"""
Canonical room / space type normalization shared by map-side modules.
"""

import re
from typing import List, Optional, Sequence, Tuple


COMMON_SPACE_TYPE_LIBRARY: List[Tuple[str, Sequence[str]]] = [
    ("bedroom", ("bedroom", "bed room", "master bedroom", "guest bedroom", "kids bedroom", "nursery")),
    ("bathroom", ("bathroom", "restroom", "washroom", "toilet room")),
    ("kitchen", ("kitchen", "kitchenette")),
    ("living room", ("living room", "living area", "lounge", "family room", "sitting room")),
    ("dining room", ("dining room", "dining area", "breakfast room")),
    ("office", ("office", "study", "workspace", "work room")),
    ("laundry room", ("laundry room", "laundry", "utility room")),
    ("entryway", ("entryway", "entry", "entrance", "foyer", "entry hall")),
    ("stairs", ("stairs", "stair", "stairway", "staircase", "landing")),
    ("hallway", ("hallway", "hall", "corridor", "passage", "passageway")),
    ("closet", ("closet", "wardrobe")),
    ("pantry", ("pantry",)),
    ("mudroom", ("mudroom", "mud room")),
    ("garage", ("garage",)),
    ("balcony", ("balcony",)),
    ("patio", ("patio", "terrace", "deck")),
    ("lobby", ("lobby", "reception")),
    ("gym", ("gym", "exercise room", "fitness room")),
    ("library", ("library", "reading room")),
    ("playroom", ("playroom", "kids playroom")),
    ("media room", ("media room", "tv room", "home theater", "theater room", "game room")),
    ("sunroom", ("sunroom", "sun room", "conservatory")),
    ("studio", ("studio", "studio apartment")),
    ("basement", ("basement", "cellar")),
    ("attic", ("attic",)),
    ("loft", ("loft",)),
    ("conference room", ("conference room", "meeting room", "board room")),
    ("dressing room", ("dressing room", "changing room")),
    ("storage", ("storage", "storage room", "storeroom", "pantry")),
]


def canonical_space_types_text() -> str:
    """Short prompt-side text listing the allowed canonical room/space types."""
    return ", ".join(name for name, _aliases in COMMON_SPACE_TYPE_LIBRARY)


def strip_space_type_variant_suffixes(text: Optional[str]) -> Optional[str]:
    """Remove space-area variant numbers such as bedroom1 / hallway2 from free-form text."""
    if text is None:
        return None

    cleaned = str(text)
    alias_to_canonical = {
        alias: canonical_name
        for canonical_name, aliases in COMMON_SPACE_TYPE_LIBRARY
        for alias in aliases
    }
    names = sorted(alias_to_canonical.keys(), key=len, reverse=True)

    for alias in names:
        canonical_name = alias_to_canonical[alias]
        pattern = re.compile(
            rf"(?i)\b({re.escape(alias)})(?:\s*[-_ ]?\s*(\d+))\b"
        )
        cleaned = pattern.sub(
            lambda match: _match_case(match.group(1), canonical_name),
            cleaned,
        )

    return cleaned


def normalize_space_type(text: Optional[str]) -> str:
    """Map free-form room text to one canonical common room / space type."""
    normalized = _normalize_text(text)
    if not normalized:
        return "Unknown"

    exact_match = _match_alias(normalized, exact_only=True)
    if exact_match:
        return exact_match

    fuzzy_match = _match_alias(normalized, exact_only=False)
    if fuzzy_match:
        return fuzzy_match

    return "Unknown"


def _match_alias(normalized: str, exact_only: bool) -> Optional[str]:
    padded = f" {normalized} "
    for canonical_name, aliases in COMMON_SPACE_TYPE_LIBRARY:
        for alias in aliases:
            alias_norm = _normalize_text(alias)
            if not alias_norm:
                continue
            if exact_only:
                if normalized == alias_norm:
                    return canonical_name
                continue
            if f" {alias_norm} " in padded:
                return canonical_name
    return None


def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def _match_case(template: str, canonical_name: str) -> str:
    if template.isupper():
        return canonical_name.upper()
    if template[:1].isupper():
        return " ".join(word.capitalize() for word in canonical_name.split())
    return canonical_name
