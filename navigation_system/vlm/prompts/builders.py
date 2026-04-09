"""Prompt builders backed by external markdown templates."""

import re

from navigation_system.config.core.params.actions import (
    ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M,
    ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M,
)
from navigation_system.config.core.params.thresholds import (
    ARRIVAL_NEAR_M,
    OBS_BLOCKED_M,
    OBS_OPEN_M,
    OBS_RISKY_M,
)
from navigation_system.vlm.prompts.common import load_prompt_template


INITIAL_PLANNING_PROMPT = load_prompt_template("planning_initial.prompt.md")
VERIFICATION_REPLANNING_PROMPT = load_prompt_template("planning_verify.prompt.md")
ACTION_EXECUTION_PROMPT = load_prompt_template("action_execution.prompt.md")
DEFAULT_ALLOWED_ACTION_NAMES = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")


def _fmt_threshold_m(value: float) -> str:
    text = f"{float(value):.2f}"
    if text.endswith("00"):
        return f"{float(value):.1f}"
    return text.rstrip("0").rstrip(".")


def get_initial_planning_prompt(instruction: str, action_space: str) -> str:
    """Render the initial planning prompt."""
    return INITIAL_PLANNING_PROMPT.format(
        instruction=instruction,
        action_space=action_space,
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(ARRIVAL_NEAR_M),
    )


def _get_verify_view_count(direction_names=None):
    provided_direction_names = [
        str(name).strip()
        for name in list(direction_names or [])
        if str(name or "").strip()
    ]
    view_count = len(provided_direction_names)
    return view_count if 0 < view_count < 12 else 12


def get_verification_replanning_prompt(
    instruction: str,
    subtask_destination: str,
    subtask_instruction: str,
    action_space: str,
    detected_landmarks: str = None,
    waypoint_summary: str = None,
    previous_subtask_landmark_summary: str = None,
    verify_replan_prompt_notice: str = None,
    direction_names: list = None,
) -> str:
    """Render the verification/replanning prompt."""
    if not waypoint_summary:
        waypoint_summary = "Unavailable"
    if verify_replan_prompt_notice:
        verify_replan_prompt_notice_block = (
            f"\n**Stuck Notice**: {verify_replan_prompt_notice.strip()}"
        )
    else:
        verify_replan_prompt_notice_block = ""

    previous_subtask_landmark_summary = str(previous_subtask_landmark_summary or "").strip()
    previous_subtask_landmark_block = (
        f"- {previous_subtask_landmark_summary}"
        if previous_subtask_landmark_summary
        else ""
    )
    verify_view_count = _get_verify_view_count(direction_names)

    return VERIFICATION_REPLANNING_PROMPT.format(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        action_space=action_space,
        detected_landmarks=detected_landmarks,
        waypoint_summary=waypoint_summary,
        previous_subtask_landmark_summary=previous_subtask_landmark_summary,
        previous_subtask_landmark_block=previous_subtask_landmark_block,
        verify_replan_prompt_notice_block=verify_replan_prompt_notice_block,
        verify_view_count=verify_view_count,
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(ARRIVAL_NEAR_M),
    )


def _parse_distance_m(distance_text) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)", str(distance_text or ""))
    if not match:
        return -1.0
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return -1.0


def _format_obstacle_state(distance_text) -> str:
    distance_m = _parse_distance_m(distance_text)
    if distance_m < 0.0:
        return ""
    if distance_m < float(OBS_BLOCKED_M):
        return "warning"
    if distance_m > float(OBS_OPEN_M):
        return "open"
    return ""


def _build_obstacle_perception_summary(obstacle_distances=None) -> str:
    distances = dict(obstacle_distances or {})
    items = []
    for label, key in (
        ("FRONT", "front"),
        ("Left 30deg", "left_30"),
        ("Right 30deg", "right_30"),
    ):
        distance_text = distances.get(key, "Unknown")
        state_text = _format_obstacle_state(distance_text)
        lower_text = str(distance_text or "").strip().lower()
        has_state_text = any(token in lower_text for token in ("warning", "open"))
        if state_text and not has_state_text:
            items.append(f"{label} {distance_text} {state_text}")
        else:
            items.append(f"{label} {distance_text}")
    return " | ".join(items)


def _build_landmark_perception_summary(
    detected_landmarks=None,
    landmark_map_info=None,
) -> str:
    lines = []
    landmark_map_text = str(landmark_map_info or "").strip()
    detected_text = str(detected_landmarks or "").strip()
    if landmark_map_text:
        for line in landmark_map_text.splitlines():
            clean_line = str(line).rstrip()
            if clean_line:
                lines.append(clean_line)
    elif detected_text and not detected_text.lower().startswith("no "):
        lines.append(f"- raw detected landmarks: {detected_text}")

    if not lines:
        return "- no valid visible landmark entries"
    return "\n".join(lines)


def _normalize_allowed_action_names(allowed_action_names=None):
    if not allowed_action_names:
        return list(DEFAULT_ALLOWED_ACTION_NAMES)

    allowed = {
        str(name or "").strip().upper()
        for name in allowed_action_names
        if str(name or "").strip()
    }
    ordered = [name for name in DEFAULT_ALLOWED_ACTION_NAMES if name in allowed]
    return ordered or list(DEFAULT_ALLOWED_ACTION_NAMES)


def _build_allowed_action_output(allowed_action_names=None) -> str:
    ordered = _normalize_allowed_action_names(allowed_action_names)
    choices = []
    if "MOVE_FORWARD" in ordered:
        choices.extend(
            [
                "MOVE_FORWARD 0.25m",
                "MOVE_FORWARD 0.5m",
                "MOVE_FORWARD 0.75m",
                "MOVE_FORWARD 1.0m",
                "MOVE_FORWARD 1.25m",
            ]
        )
    if "TURN_LEFT" in ordered:
        choices.append("TURN_LEFT 30deg")
    if "TURN_RIGHT" in ordered:
        choices.append("TURN_RIGHT 30deg")
    if "STOP" in ordered:
        choices.append("STOP")
    return " | ".join(choices)


def _build_allowed_action_bullets(allowed_action_names=None) -> str:
    ordered = _normalize_allowed_action_names(allowed_action_names)
    lines = []
    if "MOVE_FORWARD" in ordered:
        lines.append("- `MOVE_FORWARD {0.25m, 0.5m, 0.75m, 1.0m, 1.25m}`")
    turn_parts = []
    if "TURN_LEFT" in ordered:
        turn_parts.append("`TURN_LEFT 30deg`")
    if "TURN_RIGHT" in ordered:
        turn_parts.append("`TURN_RIGHT 30deg`")
    if turn_parts:
        lines.append("- " + " | ".join(turn_parts))
    if "STOP" in ordered:
        lines.append("- `STOP`")
    return "\n".join(lines)


def get_action_execution_prompt(
    next_waypoint: str,
    subtask_instruction: str,
    progress_summary: str = "",
    waypoint_summary: str = "",
    detected_landmarks: str = None,
    previous_action_reason: str = "",
    obstacle_distances=None,
    landmark_map_info: str = None,
    allowed_action_names=None,
    move_distance: float = 0.25,
    turn_angle: int = 30,
) -> str:
    """Render the action-execution prompt."""
    if not progress_summary:
        progress_summary = "Just started"
    if not waypoint_summary:
        waypoint_summary = "No space structure recorded yet."

    return ACTION_EXECUTION_PROMPT.format(
        subtask_destination=next_waypoint,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary,
        waypoint_summary=waypoint_summary,
        previous_action_reason=previous_action_reason or "N/A (first step)",
        detected_landmarks=detected_landmarks or "none",
        obstacle_perception_summary=_build_obstacle_perception_summary(obstacle_distances),
        landmark_perception_summary=_build_landmark_perception_summary(
            detected_landmarks=detected_landmarks,
            landmark_map_info=landmark_map_info,
        ),
        allowed_action_output=_build_allowed_action_output(allowed_action_names),
        allowed_action_bullets=_build_allowed_action_bullets(allowed_action_names),
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        open_autocomplete_m=_fmt_threshold_m(ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M),
        solid_autocomplete_m=_fmt_threshold_m(ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M),
        move_distance=move_distance,
        turn_angle=turn_angle,
    )


__all__ = [
    "ACTION_EXECUTION_PROMPT",
    "DEFAULT_ALLOWED_ACTION_NAMES",
    "INITIAL_PLANNING_PROMPT",
    "VERIFICATION_REPLANNING_PROMPT",
    "_build_allowed_action_bullets",
    "_build_allowed_action_output",
    "_build_landmark_perception_summary",
    "_build_obstacle_perception_summary",
    "_fmt_threshold_m",
    "_get_verify_view_count",
    "get_action_execution_prompt",
    "get_initial_planning_prompt",
    "get_verification_replanning_prompt",
]
