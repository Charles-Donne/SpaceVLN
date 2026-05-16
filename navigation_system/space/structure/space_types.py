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
    ("stairs", ("stairs", "stair", "stairway", "staircase", "landing")),
    ("hallway", (
        "hallway", "hall", "corridor", "passage", "passageway",
        "entryway", "entry", "entrance", "doorway", "door way", "foyer", "entry hall",
    )),
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


COMMON_SPACE_TYPE_CUE_LIBRARY: List[Tuple[str, Sequence[str]]] = [
    ("bedroom", (
        "bed", "headboard", "mattress", "pillow", "nightstand", "bedside table",
        "dresser", "vanity table", "clothes drawer",
    )),
    ("bathroom", (
        "sink", "toilet", "bathtub", "bath tub", "shower", "vanity", "bath mat",
        "towel rack", "medicine cabinet",
    )),
    ("kitchen", (
        "refrigerator", "fridge", "freezer", "stove", "oven", "microwave",
        "dishwasher", "countertop", "counter top", "kitchen island", "sink cabinet",
        "range hood", "cooktop",
    )),
    ("living room", (
        "sofa", "couch", "sectional", "coffee table", "television", "tv",
        "fireplace", "armchair", "lounge chair", "media console",
    )),
    ("dining room", (
        "dining table", "dining chair", "chairs around table", "place setting",
        "breakfast table", "buffet table", "sideboard",
    )),
    ("office", (
        "desk", "office chair", "computer", "monitor", "keyboard", "printer",
        "bookshelf", "bookcase", "filing cabinet",
    )),
    ("laundry room", (
        "washer", "dryer", "washer dryer", "washing machine", "laundry machine",
        "laundry basket", "detergent", "utility sink",
    )),
    ("stairs", (
        "stairs", "stair", "stairway", "staircase", "steps", "landing", "railing",
        "banister",
    )),
    ("hallway", (
        "hallway", "corridor", "passage", "passageway", "connector", "junction",
        "foyer", "entry hall", "entryway",
    )),
    ("closet", ("closet", "coat rack", "hanging clothes", "shelf closet")),
    ("pantry", ("pantry", "food shelf", "storage jars")),
    ("mudroom", ("mudroom", "mud room", "shoe rack", "coat bench")),
    ("garage", ("garage", "garage door", "car bay")),
    ("balcony", ("balcony", "balcony railing")),
    ("patio", ("patio", "terrace", "deck", "outdoor table")),
    ("gym", ("treadmill", "exercise bike", "weights", "fitness equipment")),
    ("library", ("library", "reading chair", "book shelves")),
    ("media room", ("home theater", "theater seating", "game console", "projector")),
    ("storage", ("storage shelf", "storage shelves", "boxes", "storage boxes")),
]


COMMON_SPACE_TYPE_INFERENCE_ALIAS_LIBRARY: List[Tuple[str, Sequence[str]]] = [
    ("bedroom", ("bedroom", "bed room", "master bedroom", "guest bedroom", "kids bedroom", "nursery")),
    ("bathroom", ("bathroom", "restroom", "washroom", "toilet room")),
    ("kitchen", ("kitchen", "kitchenette")),
    ("living room", ("living room", "living area", "lounge", "family room", "sitting room")),
    ("dining room", ("dining room", "dining area", "breakfast room")),
    ("office", ("office", "study", "workspace", "work room")),
    ("laundry room", ("laundry room", "laundry", "utility room")),
    ("stairs", ("stairs", "stair", "stairway", "staircase")),
    ("hallway", ("hallway", "hall", "corridor", "passageway", "foyer", "entry hall")),
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
    ("storage", ("storage", "storage room", "storeroom")),
]


def canonical_space_types_text() -> str:
    """Short prompt-side text listing the allowed canonical room/space types."""
    return ", ".join(name for name, _aliases in COMMON_SPACE_TYPE_LIBRARY)


def strip_space_type_variant_suffixes(text: Optional[str]) -> Optional[str]:
    """Remove region variant numbers such as bedroom1 / hallway2 from free-form text."""
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

    # Also clean unknown/custom space labels like `Poolarea11` when the suffixed
    # token is used as a space name prefix or area reference.
    generic_pattern = re.compile(
        r"(?i)\b([a-z][a-z_-]*[a-z])(?:[-_]?)(\d+)(?=(?:'s\b)|\s*[-|,:)\]]|$)"
    )
    cleaned = generic_pattern.sub(lambda match: match.group(1), cleaned)

    return cleaned


def strip_space_type_label_variant_suffix(text: Optional[str]) -> Optional[str]:
    """Remove a trailing variant id from an isolated room/space label."""
    if text is None:
        return None

    cleaned = strip_space_type_variant_suffixes(text)
    if cleaned is None:
        return None

    full_label_pattern = re.compile(
        r"(?i)^\s*([a-z][a-z ]*[a-z])(?:\s*[-_ ]\s*|\s+)(\d+)\s*$"
    )
    match = full_label_pattern.match(cleaned)
    if match:
        return " ".join(match.group(1).split())
    return " ".join(str(cleaned).split())


def normalize_space_type(text: Optional[str]) -> str:
    """Map free-form room text to one canonical common room / space type."""
    normalized = _normalize_text(text)
    if not normalized:
        return "Unknown"

    exact_match = _match_alias(normalized, exact_only=True)
    if exact_match:
        return exact_match

    wrapped_match = _match_alias_with_generic_wrapper(normalized)
    if wrapped_match:
        return wrapped_match

    return "Unknown"


def infer_space_type_from_texts(texts: Sequence[Optional[str]]) -> str:
    """Infer a canonical common room / space type from local landmark or layout cues."""
    normalized_parts = [
        _normalize_text(strip_space_type_variant_suffixes(text) or text)
        for text in list(texts or [])
        if str(text or "").strip()
    ]
    normalized_text = " ".join(part for part in normalized_parts if part)
    if not normalized_text:
        return "Unknown"

    scores = {}
    _score_space_terms(
        normalized_text=normalized_text,
        cue_library=COMMON_SPACE_TYPE_INFERENCE_ALIAS_LIBRARY,
        scores=scores,
        base_score=4,
    )
    _score_space_terms(
        normalized_text=normalized_text,
        cue_library=COMMON_SPACE_TYPE_CUE_LIBRARY,
        scores=scores,
        base_score=1,
    )

    if not scores:
        return "Unknown"

    ordered_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered_scores) > 1 and ordered_scores[0][1] == ordered_scores[1][1]:
        return "Unknown"
    return str(ordered_scores[0][0])


def _score_space_terms(
    *,
    normalized_text: str,
    cue_library: Sequence[Tuple[str, Sequence[str]]],
    scores: dict,
    base_score: int,
) -> None:
    padded_text = f" {normalized_text} "
    for canonical_name, cues in cue_library:
        for cue in cues:
            cue_norm = _normalize_text(cue)
            if not cue_norm:
                continue
            if f" {cue_norm} " not in padded_text:
                continue
            scores[canonical_name] = int(scores.get(canonical_name, 0)) + int(base_score) + len(cue_norm.split())


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


def _match_alias_with_generic_wrapper(normalized: str) -> Optional[str]:
    """Allow only light wrappers like `bedroom area`, but preserve custom names like `large hall`."""
    generic_prefixes = ("area ", "space ", "zone ", "section ")
    generic_suffixes = (" area", " space", " zone", " section")

    for generic_prefix in generic_prefixes:
        if normalized.startswith(generic_prefix):
            candidate = normalized[len(generic_prefix):].strip()
            exact_match = _match_alias(candidate, exact_only=True)
            if exact_match:
                return exact_match

    for generic_suffix in generic_suffixes:
        if normalized.endswith(generic_suffix):
            candidate = normalized[:-len(generic_suffix)].strip()
            exact_match = _match_alias(candidate, exact_only=True)
            if exact_match:
                return exact_match

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
