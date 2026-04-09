"""Shared navigation constants and canonical planner/action payload helpers."""

from typing import Any, Dict, Optional, Sequence


DIRECTION_CONFIG = [
    {"step": 12, "angle": 0, "name": "IMAGE 1: Front 0deg"},
    {"step": 1, "angle": 30, "name": "IMAGE 2: Left 30deg"},
    {"step": 2, "angle": 60, "name": "IMAGE 3: Left 60deg"},
    {"step": 3, "angle": 90, "name": "IMAGE 4: Left 90deg"},
    {"step": 4, "angle": 120, "name": "IMAGE 5: Left 120deg"},
    {"step": 5, "angle": 150, "name": "IMAGE 6: Left 150deg"},
    {"step": 6, "angle": 180, "name": "IMAGE 7: Back 180deg"},
    {"step": 7, "angle": 210, "name": "IMAGE 8: Right 150deg"},
    {"step": 8, "angle": 240, "name": "IMAGE 9: Right 120deg"},
    {"step": 9, "angle": 270, "name": "IMAGE 10: Right 90deg"},
    {"step": 10, "angle": 300, "name": "IMAGE 11: Right 60deg"},
    {"step": 11, "angle": 330, "name": "IMAGE 12: Right 30deg"},
]

ACTION_MAPPING = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
}

REQUIRED_SUBTASK_FIELDS: Sequence[str] = (
    "current_waypoint",
    "next_waypoint_direction",
    "next_waypoint",
    "subtask_instruction",
)


def normalize_subtask_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize planner output to the canonical field set used across the runtime."""
    if not payload:
        return payload

    normalized = dict(payload)
    if normalized.get("subtask_landmark") is None:
        normalized["subtask_landmark"] = ""
    if normalized.get("global_task_finish") is None:
        normalized["global_task_finish"] = False
    return normalized


def get_next_waypoint(payload: Optional[Dict[str, Any]]) -> str:
    if not payload:
        return ""
    return str(payload.get("next_waypoint") or "").strip()


def get_subtask_landmark(payload: Optional[Dict[str, Any]]) -> str:
    if not payload:
        return ""
    return str(payload.get("subtask_landmark") or "").strip()


__all__ = [
    "ACTION_MAPPING",
    "DIRECTION_CONFIG",
    "REQUIRED_SUBTASK_FIELDS",
    "get_next_waypoint",
    "get_subtask_landmark",
    "normalize_subtask_payload",
]
