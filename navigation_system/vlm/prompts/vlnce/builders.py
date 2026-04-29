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


def _normalize_anchor_notation_text(prompt: str) -> str:
    normalized = str(prompt or "")
    literal_replacements = (
        ("[space] - [landmark1 / landmark2 / landmark3]", "space - landmark1 / landmark2 / landmark3"),
        ("[space] - [landmark / landmark / landmark]", "space - landmark / landmark / landmark"),
        ("[space]'s [landmark]", "space's landmark"),
    )
    for old, new in literal_replacements:
        normalized = normalized.replace(old, new)
    normalized = re.sub(
        r"\[([A-Za-z][^\[\]]*?)\]'s\s+\[([^\[\]]+?)\]",
        lambda match: f"{match.group(1).strip()}'s {match.group(2).strip()}",
        normalized,
    )
    normalized = re.sub(
        r"\[space\]\s*-\s*\[([^\[\]]+?)\]",
        lambda match: f"space - {match.group(1).strip()}",
        normalized,
    )
    return normalized


def _build_previous_subtask_landmark_block(previous_subtask_landmark_summary: str) -> str:
    summary_text = str(previous_subtask_landmark_summary or "").strip()
    if not summary_text:
        return ""
    return (
        f"- {summary_text}\n"
        "- Use the previous subtask landmark only as supporting evidence to check whether that landmark was reached."
    )


def get_initial_planning_prompt(instruction: str, action_space: str) -> str:
    """Render the initial planning prompt."""
    return _normalize_anchor_notation_text(INITIAL_PLANNING_PROMPT.format(
        instruction=instruction,
        action_space=action_space,
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(ARRIVAL_NEAR_M),
    ))


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

    previous_subtask_landmark_summary = str(previous_subtask_landmark_summary or "").strip()
    previous_subtask_landmark_block = _build_previous_subtask_landmark_block(
        previous_subtask_landmark_summary
    )
    verify_notice_block = str(verify_replan_prompt_notice or "").strip()
    verify_view_count = _get_verify_view_count(direction_names)

    return _normalize_anchor_notation_text(VERIFICATION_REPLANNING_PROMPT.format(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        action_space=action_space,
        detected_landmarks=detected_landmarks,
        waypoint_summary=waypoint_summary,
        previous_subtask_landmark_summary=previous_subtask_landmark_summary,
        previous_subtask_landmark_block=previous_subtask_landmark_block,
        verify_replan_prompt_notice_block=verify_notice_block,
        verify_view_count=verify_view_count,
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(ARRIVAL_NEAR_M),
    ))


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
        choices.extend(["TURN_LEFT_AVOID 30deg", "TURN_LEFT_ALIGN 30deg"])
    if "TURN_RIGHT" in ordered:
        choices.extend(["TURN_RIGHT_AVOID 30deg", "TURN_RIGHT_ALIGN 30deg"])
    if "STOP" in ordered:
        choices.append("STOP")
    return " | ".join(choices)


def _build_allowed_action_bullets(allowed_action_names=None) -> str:
    ordered = _normalize_allowed_action_names(allowed_action_names)
    lines = []
    if "MOVE_FORWARD" in ordered:
        lines.append(
            "- `MOVE_FORWARD {0.25m, 0.5m, 0.75m, 1.0m, 1.25m}`: move forward by the selected distance"
        )
    if "TURN_LEFT" in ordered:
        lines.append(
            f"- `TURN_LEFT_AVOID 30deg` to avoid obstacle only when FRONT <{_fmt_threshold_m(OBS_BLOCKED_M)}m or the current FRONT route is unusable | "
            "`TURN_LEFT_ALIGN 30deg` to align destination landmark"
        )
    if "TURN_RIGHT" in ordered:
        lines.append(
            f"- `TURN_RIGHT_AVOID 30deg` to avoid obstacle only when FRONT <{_fmt_threshold_m(OBS_BLOCKED_M)}m or the current FRONT route is unusable | "
            "`TURN_RIGHT_ALIGN 30deg` to align destination landmark"
        )
    if "STOP" in ordered:
        lines.append("- `STOP`: stop only when the current destination is reached")
    return "\n".join(lines)


def _build_action_space_constraint_notice(allowed_action_names=None) -> str:
    ordered = _normalize_allowed_action_names(allowed_action_names)
    forbidden = [name for name in DEFAULT_ALLOWED_ACTION_NAMES if name not in ordered]
    if not forbidden:
        return ""

    reason_by_action = {
        "MOVE_FORWARD": "front-route recovery: forward is temporarily blocked",
        "TURN_LEFT": "controller-side constraint: left turn is temporarily unavailable",
        "TURN_RIGHT": "controller-side constraint: right turn is temporarily unavailable",
        "STOP": "arrival is not available for this retry",
    }
    reasons = "; ".join(reason_by_action.get(name, name) for name in forbidden)
    return (
        "**Temporary action constraint**: choose only from the Action space above; "
        f"do not output omitted actions. Reason: {reasons}."
    )


def _normalize_action_prompt_text(prompt: str) -> str:
    normalized = str(prompt or "")
    literal_replacements = (
        (
            "use nearby landmarks, valid detections, `Subtask Progress`, `Previous Step Analysis`, and current image content",
            "use nearby landmarks, valid detections, `Subtask Progress`, and current image content",
        ),
        (
            "use `Subtask Progress`, `Previous Step Analysis`, and current image content",
            "use `Subtask Progress` and current image content",
        ),
        (
            "Use `Subtask Progress` and `Previous Step Analysis` to avoid repeating a finished turn:",
            "Use `Subtask Progress` to avoid repeating a finished turn:",
        ),
        (
            "If `Subtask Progress` contains `(warning: front route blocked; forced stop)` or `Previous Step Analysis` says the last forward step was blocked, do not push into that same FRONT route on this call; choose `STOP` only if arrival is already satisfied, otherwise choose a side turn.",
            "If `Subtask Progress` contains `(warning: front route blocked; forced stop)`, do not push into that same FRONT route on this call; choose `STOP` only if arrival is already satisfied, otherwise choose a side turn.",
        ),
        (
            "On that retry, only `TURN_LEFT 30deg`, `TURN_RIGHT 30deg`, or valid `STOP` are allowed.",
            "On that retry, choose only a side turn from the action space or valid `STOP`; use an `*_AVOID` turn for obstacle clearing and an `*_ALIGN` turn for destination re-alignment.",
        ),
        (
            "A close valid side landmark/destination usually beats weak generic avoidance: if it is clearly on the left, prefer `TURN_LEFT 30deg`; if clearly on the right, prefer `TURN_RIGHT 30deg`.",
            "A close valid side landmark/destination usually beats weak generic avoidance: if it is clearly on the left, prefer `TURN_LEFT_ALIGN 30deg`; if clearly on the right, prefer `TURN_RIGHT_ALIGN 30deg`.",
        ),
        (
            "A clearly side-offset destination usually beats weak generic avoidance: if it is clearly on the left, prefer `TURN_LEFT 30deg`; if clearly on the right, prefer `TURN_RIGHT 30deg`.",
            "A clearly side-offset destination usually beats weak generic avoidance: if it is clearly on the left, prefer `TURN_LEFT_ALIGN 30deg`; if clearly on the right, prefer `TURN_RIGHT_ALIGN 30deg`.",
        ),
        (
            "If `Subtask Progress` is empty / `Just started` and `Previous Step Analysis` is empty / `N/A (first step)`, treat the current facing as aligned;",
            "If `Subtask Progress` is empty / `Just started`, treat the current facing as aligned;",
        ),
        (
            "If `Previous Step Analysis` shows the last action already turned toward the destination or already turned to avoid an obstacle and re-align the route, treat that reorientation as finished.",
            "If `Subtask Progress` already records that the last step turned for destination alignment or obstacle avoidance, treat that reorientation as finished.",
        ),
        (
            "If `Controller Notice` says the current FRONT retry is blocked or `Subtask Progress` carries the blocked-front warning, do not answer with another forward into that same blocked FRONT route on this retry unless current evidence clearly shows it has reopened or it is the correct stair run.",
            "If `Subtask Progress` carries the blocked-front warning, do not answer with another forward into that same FRONT route on this retry unless current evidence clearly shows it has reopened or it is the correct stair run.",
        ),
        (
            "Always read `Subtask Progress` and `Previous Step Analysis` to judge stage completion, route relation, and whether the last turn already aligned the agent.",
            "Use `Subtask Progress` as last-step memory to judge stage completion, route relation, and whether the previous action already finished the needed turn.",
        ),
        (
            "Use `Subtask Progress` and `Previous Step Analysis` only as route-state hints; if they say the front route was blocked on the last call, do not push into that same blocked FRONT route again immediately.",
            "Use `Subtask Progress` only as last-step memory; if it says the front route was blocked on the last call, do not push into that same FRONT route again immediately.",
        ),
        (
            "compare Left 30deg and Right 30deg",
            "compare left-turn and right-turn options",
        ),
        (
            '"action": "TURN_LEFT 30deg"',
            '"action": "TURN_LEFT_AVOID 30deg"',
        ),
        (
            "Output `action` only from the fixed action space: `TURN_LEFT 30deg` / `TURN_RIGHT 30deg` / `MOVE_FORWARD {{0.25m, 0.5m, 0.75m, 1.0m, 1.25m}}` / `STOP`.",
            "Output `action` only from the fixed action space: `TURN_LEFT_AVOID 30deg` / `TURN_LEFT_ALIGN 30deg` / `TURN_RIGHT_AVOID 30deg` / `TURN_RIGHT_ALIGN 30deg` / `MOVE_FORWARD {{0.25m, 0.5m, 0.75m, 1.0m, 1.25m}}` / `STOP`.",
        ),
        (
            "Output `action` only from the fixed action space: `TURN_LEFT 30deg` / `TURN_RIGHT 30deg` / `MOVE_FORWARD {0.25m, 0.5m, 0.75m, 1.0m, 1.25m}` / `STOP`.",
            "Output `action` only from the fixed action space: `TURN_LEFT_AVOID 30deg` / `TURN_LEFT_ALIGN 30deg` / `TURN_RIGHT_AVOID 30deg` / `TURN_RIGHT_ALIGN 30deg` / `MOVE_FORWARD {0.25m, 0.5m, 0.75m, 1.0m, 1.25m}` / `STOP`.",
        ),
        (
            "toward the current-stage destination",
            "toward the current destination",
        ),
    )
    for old, new in literal_replacements:
        normalized = normalized.replace(old, new)

    normalized = re.sub(
        r"(?m)^- \*\*Current-stage only\*\*:.*$",
        "- **Focus**: rely on the current `Instruction`, current `Destination`, visible landmark/route cues, obstacle layout, and `Subtask Progress`.",
        normalized,
    )
    normalized = re.sub(
        r"(?m)^(\s*)a\. \*\*Destination-first stage following\*\*:.*$",
        r"\1a. **Current cues first**: focus on the current `Instruction`, current `Destination`, visible landmark/route cues, obstacle layout, and `Subtask Progress`. Choose the route that really leads to the destination rather than the nearest open side or most obvious reference cue.",
        normalized,
    )
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def get_action_execution_prompt(
    next_waypoint: str,
    subtask_instruction: str,
    subtask_landmark: str = "",
    progress_summary: str = "",
    waypoint_summary: str = "",
    detected_landmarks: str = None,
    obstacle_distances=None,
    landmark_map_info: str = None,
    allowed_action_names=None,
    move_distance: float = 0.25,
    turn_angle: int = 30,
) -> str:
    """Render the action-execution prompt."""
    del waypoint_summary
    if not progress_summary:
        progress_summary = "Just started"

    return _normalize_action_prompt_text(ACTION_EXECUTION_PROMPT.format(
        subtask_destination=next_waypoint,
        subtask_landmark=str(subtask_landmark or "").strip() or "none",
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary,
        detected_landmarks=detected_landmarks or "none",
        obstacle_perception_summary=_build_obstacle_perception_summary(obstacle_distances),
        landmark_perception_summary=_build_landmark_perception_summary(
            detected_landmarks=detected_landmarks,
            landmark_map_info=landmark_map_info,
        ),
        allowed_action_output=_build_allowed_action_output(allowed_action_names),
        allowed_action_bullets=_build_allowed_action_bullets(allowed_action_names),
        action_space_constraint_notice=_build_action_space_constraint_notice(allowed_action_names),
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        open_autocomplete_m=_fmt_threshold_m(ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M),
        solid_autocomplete_m=_fmt_threshold_m(ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M),
        move_distance=move_distance,
        turn_angle=turn_angle,
    ))


__all__ = [
    "ACTION_EXECUTION_PROMPT",
    "DEFAULT_ALLOWED_ACTION_NAMES",
    "INITIAL_PLANNING_PROMPT",
    "VERIFICATION_REPLANNING_PROMPT",
    "_build_allowed_action_bullets",
    "_build_allowed_action_output",
    "_build_action_space_constraint_notice",
    "_build_landmark_perception_summary",
    "_normalize_action_prompt_text",
    "_build_obstacle_perception_summary",
    "_fmt_threshold_m",
    "_get_verify_view_count",
    "get_action_execution_prompt",
    "get_initial_planning_prompt",
    "get_verification_replanning_prompt",
]
