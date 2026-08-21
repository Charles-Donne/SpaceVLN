"""Common landmark vocabulary and canonical name helpers."""

import re
from typing import Dict, List, Optional, Sequence, Tuple


COMMON_LANDMARK_LIBRARY: List[Tuple[str, Sequence[str]]] = [
    ("bed", ("bed", "beds", "mattress", "bunk bed", "queen bed", "king bed")),
    ("sofa", ("sofa", "sofas", "couch", "couches", "settee", "loveseat", "sectional", "sectional sofa", "sectional couch")),
    ("chair", ("chair", "chairs", "armchair", "arm chair", "dining chair", "office chair", "seat", "seats")),
    ("table", ("table", "tables", "coffee table", "dining table", "side table", "end table", "nightstand", "night stand", "bedside table")),
    ("desk", ("desk", "desks", "office desk", "computer desk", "work desk")),
    ("cabinet", ("cabinet", "cabinets", "cupboard", "cupboards", "kitchen cabinet", "storage cabinet")),
    ("bookshelf", ("bookshelf", "bookshelves", "book shelf", "book shelves", "bookcase", "book case", "bookcases")),
    ("tv", ("tv", "t v", "television", "television set", "tv set", "tv screen", "flat screen tv")),
    ("monitor", ("monitor", "computer monitor", "display monitor")),
    ("sink", ("sink", "sinks", "basin", "wash basin", "washbasin")),
    ("toilet", ("toilet", "toilets", "commode", "wc", "water closet")),
    ("bathtub", ("bathtub", "bath tub", "tub")),
    ("shower", ("shower", "shower stall", "shower cabin")),
    ("refrigerator", ("refrigerator", "fridge", "freezer", "refrigerator freezer")),
    ("stove", ("stove", "cooktop", "hob", "range", "kitchen range")),
    ("oven", ("oven", "wall oven")),
    ("microwave", ("microwave", "microwave oven")),
    ("plant", ("plant", "plants", "potted plant", "houseplant", "indoor plant", "flower pot", "plant pot")),
    ("painting", ("painting", "paintings", "picture", "pictures", "wall art", "artwork", "framed picture", "framed art")),
    ("window", ("window", "windows")),
    ("door", ("door", "doors", "doorway", "door way")),
    ("stairs", ("stairs", "stair", "stairway", "stair case", "staircase", "steps")),
    ("mirror", ("mirror", "mirrors")),
    ("lamp", ("lamp", "lamps", "floor lamp", "table lamp", "light fixture")),
    ("rug", ("rug", "rugs", "carpet", "mat", "floor mat")),
    ("curtain", ("curtain", "curtains", "drape", "drapes", "blinds", "window blinds")),
    ("counter", ("counter", "countertop", "counter top", "kitchen counter", "worktop")),
    ("island", ("island", "kitchen island")),
    ("washer", ("washer", "washing machine", "laundry machine")),
    ("dryer", ("dryer", "clothes dryer")),
    ("trash can", ("trash can", "trash bin", "garbage can", "garbage bin", "bin")),
]

_LEADING_DETERMINERS = re.compile(r"^(?:a|an|the|this|that|these|those)\s+")
_TRAILING_GENERIC_WORDS = re.compile(r"\s+(?:object|item|thing|landmark|area)$")


def _basic_normalize(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    cleaned = str(text).strip().lower()
    if not cleaned:
        return None

    cleaned = cleaned.replace("’", "'").replace("`", "'")
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None

    while True:
        updated = _LEADING_DETERMINERS.sub("", cleaned).strip()
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = _TRAILING_GENERIC_WORDS.sub("", cleaned).strip()
    return cleaned or None


def _build_alias_map() -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for canonical_name, aliases in COMMON_LANDMARK_LIBRARY:
        canonical_norm = _basic_normalize(canonical_name)
        if not canonical_norm:
            continue
        alias_map[canonical_norm] = canonical_norm
        for alias in aliases:
            alias_norm = _basic_normalize(alias)
            if alias_norm:
                alias_map[alias_norm] = canonical_norm
    return alias_map


_ALIAS_TO_CANONICAL = _build_alias_map()


def normalize_landmark_text(text: Optional[str]) -> Optional[str]:
    """Normalize free-form landmark text and collapse common aliases."""
    normalized = _basic_normalize(text)
    if not normalized:
        return None
    return _ALIAS_TO_CANONICAL.get(normalized, normalized)


def canonical_landmark_from_known_alias(text: Optional[str]) -> Optional[str]:
    """Return a canonical name only when the text is in the common vocabulary."""
    normalized = _basic_normalize(text)
    if not normalized:
        return None
    return _ALIAS_TO_CANONICAL.get(normalized)


def canonical_landmark_names_text() -> str:
    """Return canonical common landmark names for prompt/debug display."""
    return ", ".join(canonical_name for canonical_name, _aliases in COMMON_LANDMARK_LIBRARY)


def common_landmark_detection_classes() -> List[str]:
    """Return canonical class names used for common-object detection."""
    return [canonical_name for canonical_name, _aliases in COMMON_LANDMARK_LIBRARY]


__all__ = [
    "COMMON_LANDMARK_LIBRARY",
    "canonical_landmark_from_known_alias",
    "canonical_landmark_names_text",
    "common_landmark_detection_classes",
    "normalize_landmark_text",
]
