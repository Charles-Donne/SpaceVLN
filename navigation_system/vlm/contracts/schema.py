"""Shared navigation constants and canonical planner/executor payload helpers."""

from typing import Any, Dict, List, Optional, Sequence


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


def _direction_name_for_angle(angle: float) -> str:
    angle_int = int(round(float(angle))) % 360
    if angle_int == 0:
        return "Front 0deg"
    if angle_int == 180:
        return "Back 180deg"
    if 0 < angle_int < 180:
        return f"Left {angle_int}deg"
    return f"Right {360 - angle_int}deg"


def build_direction_config(sample_count: int = 12, angle_step_deg: float = 30.0) -> List[Dict[str, Any]]:
    """Build the IMAGE-label mapping for a stopped or continuous lookaround scan.

    The scan captures after left turns. To keep IMAGE 1 as the final front-facing
    view, the front frame is the last captured sample, matching the historical
    12x30deg ordering.
    """
    try:
        count = max(1, int(sample_count))
    except (TypeError, ValueError):
        count = 12
    try:
        step_deg = float(angle_step_deg)
    except (TypeError, ValueError):
        step_deg = 30.0
    if count == 12 and abs(step_deg - 30.0) < 1e-6:
        return [dict(item) for item in DIRECTION_CONFIG]

    configs: List[Dict[str, Any]] = []
    configs.append({
        "step": count,
        "angle": 0,
        "name": "IMAGE 1: Front 0deg",
    })
    for step_idx in range(1, count):
        angle = (float(step_idx) * step_deg) % 360.0
        angle_int = int(round(angle)) % 360
        configs.append({
            "step": step_idx,
            "angle": angle_int,
            "name": f"IMAGE {step_idx + 1}: {_direction_name_for_angle(angle_int)}",
        })
    return configs

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


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    return bool(value)


def normalize_subtask_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize VLNCE planner output without adding OVON-only stop fields."""
    if not payload:
        return payload

    normalized = dict(payload)
    if normalized.get("subtask_landmark") is None:
        normalized["subtask_landmark"] = ""
    normalized["global_task_finish"] = _coerce_bool(
        normalized.get("global_task_finish"),
        default=False,
    )
    normalized.pop("global_landmark_arrival", None)
    return normalized


def normalize_objectnav_subtask_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize OVON planner output while keeping only the OVON-facing stop flag."""
    if not payload:
        return payload

    normalized = dict(payload)
    if not normalized:
        return normalized

    if normalized.get("subtask_landmark") is None:
        normalized["subtask_landmark"] = ""
    normalized["global_landmark_arrival"] = _coerce_bool(
        normalized.get("global_landmark_arrival"),
        default=False,
    )
    normalized.pop("global_task_finish", None)
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
    "build_direction_config",
    "get_next_waypoint",
    "get_subtask_landmark",
    "normalize_objectnav_subtask_payload",
    "normalize_subtask_payload",
]
