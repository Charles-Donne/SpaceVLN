"""Helpers for OVON object-goal metadata and soft semantic priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


DEFAULT_SEARCH_SPACES: Sequence[str] = (
    "kitchen",
    "living room",
    "bedroom",
    "bathroom",
    "office",
    "dining room",
    "hallway",
)

SPACE_LANDMARK_PRIORS = {
    "kitchen": ["sink", "cabinet", "counter", "refrigerator", "table", "doorway"],
    "pantry": ["shelf", "cabinet", "storage rack", "doorway"],
    "living room": ["sofa", "tv", "coffee table", "window", "doorway"],
    "bedroom": ["bed", "dresser", "nightstand", "lamp", "doorway"],
    "bathroom": ["sink", "toilet", "mirror", "bathtub", "doorway"],
    "office": ["desk", "chair", "bookshelf", "computer", "doorway"],
    "study": ["desk", "chair", "bookshelf", "lamp", "doorway"],
    "dining room": ["table", "chair", "cabinet", "doorway"],
    "laundry room": ["washer", "dryer", "cabinet", "sink", "doorway"],
    "utility room": ["shelf", "cabinet", "sink", "doorway"],
    "hallway": ["doorway", "opening", "corner", "wall picture"],
    "garage": ["shelf", "cabinet", "tool rack", "doorway"],
}

OBJECT_SPACE_PRIORS = {
    "freezer": ["kitchen", "pantry", "utility room"],
    "refrigerator": ["kitchen", "pantry"],
    "microwave": ["kitchen", "dining room"],
    "oven": ["kitchen"],
    "stove": ["kitchen"],
    "sink": ["kitchen", "bathroom", "laundry room"],
    "toilet": ["bathroom"],
    "bathtub": ["bathroom"],
    "bed": ["bedroom"],
    "dresser": ["bedroom"],
    "nightstand": ["bedroom"],
    "sofa": ["living room", "lounge"],
    "tv": ["living room", "bedroom"],
    "coffee table": ["living room"],
    "desk": ["office", "study", "bedroom"],
    "chair": ["dining room", "office", "living room", "bedroom"],
    "table": ["dining room", "kitchen", "living room", "office"],
    "cabinet": ["kitchen", "bathroom", "bedroom", "dining room"],
    "bookshelf": ["office", "study", "living room", "bedroom"],
    "plant": ["living room", "hallway", "bedroom"],
    "painting": ["living room", "hallway", "bedroom"],
    "washer": ["laundry room", "bathroom", "utility room"],
    "dryer": ["laundry room", "utility room"],
}


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().replace("_", " ").split())


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        normalized = _normalize_text(item)
        if not normalized or normalized in seen:
            continue
        result.append(normalized)
        seen.add(normalized)
    return result


@dataclass(frozen=True)
class ObjectGoalPlan:
    object_category: str
    child_categories: Sequence[str]
    likely_spaces: Sequence[str]
    proxy_landmarks: Sequence[str]


def build_object_goal_plan(
    object_category: str,
    child_categories: Sequence[str] | None = None,
) -> ObjectGoalPlan:
    category = _normalize_text(object_category)
    children = _dedupe_keep_order(child_categories or [])

    likely_spaces = list(OBJECT_SPACE_PRIORS.get(category, []))
    if not likely_spaces:
        likely_spaces = list(DEFAULT_SEARCH_SPACES[:4])

    proxy_landmarks: List[str] = [category]
    for space_name in likely_spaces:
        proxy_landmarks.extend(SPACE_LANDMARK_PRIORS.get(space_name, []))
    proxy_landmarks.extend(children)

    return ObjectGoalPlan(
        object_category=category,
        child_categories=tuple(children),
        likely_spaces=tuple(_dedupe_keep_order(likely_spaces)),
        proxy_landmarks=tuple(_dedupe_keep_order(proxy_landmarks)[:8]),
    )


def build_objectnav_instruction(
    object_category: str,
    child_categories: Sequence[str] | None = None,
) -> str:
    plan = build_object_goal_plan(
        object_category=object_category,
        child_categories=child_categories,
    )
    likely_spaces_text = ", then ".join(plan.likely_spaces)
    proxy_landmarks_text = ", ".join(plan.proxy_landmarks)

    child_text = ""
    if plan.child_categories:
        child_text = (
            f" Child-category matches that may also count: "
            f"{', '.join(plan.child_categories)}."
        )

    return (
        f"Goal object: {plan.object_category}. "
        f"Treat this as staged object-directed navigation rather than instruction following. "
        f"Stage 1: search the current local area for the {plan.object_category}. "
        f"Stage 2: if the {plan.object_category} is not already visible nearby, move through the connector, doorway, or opening that best leads toward the most likely next target space in this priority order: {likely_spaces_text}. "
        f"Stage 3: after entering a likely target space, approach the {plan.object_category} itself and stop only when the {plan.object_category} is reached. "
        f"Useful proxy landmarks and room cues include: {proxy_landmarks_text}.{child_text}"
    ).strip()


def build_raw_object_goal_instruction(
    object_category: str,
    child_categories: Sequence[str] | None = None,
) -> str:
    category = _normalize_text(object_category)
    children = _dedupe_keep_order(child_categories or [])

    lines = [f"Object goal: {category}"]
    if children:
        lines.append(f"Goal aliases: {', '.join(children)}")
    return "\n".join(lines).strip()


def parse_object_goal_instruction(instruction: str) -> tuple[str, tuple[str, ...]]:
    goal = ""
    aliases: List[str] = []
    for raw_line in str(instruction or "").splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("object goal:"):
            goal = _normalize_text(line.split(":", 1)[1])
        elif lower.startswith("goal aliases:"):
            alias_text = line.split(":", 1)[1]
            aliases.extend(_dedupe_keep_order(alias_text.split(",")))

    if not goal:
        goal = _normalize_text(instruction)
    return goal, tuple(_dedupe_keep_order(aliases))
