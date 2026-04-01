"""Canonical planner/action subtask payload helpers."""

from typing import Any, Dict, Optional, Sequence


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
