"""Prompt builders backed by external markdown templates."""

import os
import re

from navigation_system.config.core.params.actions import (
    ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M,
    ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M,
    VALID_MOVE_METERS,
)
from navigation_system.config.core.params.thresholds import (
    ARRIVAL_NEAR_M,
    OBS_BLOCKED_M,
    OBS_OPEN_M,
    OBS_RISKY_M,
)
from navigation_system.vlm.prompts.common import (
    PromptBundle,
    compose_full_prompt,
    load_prompt_template,
)


INITIAL_PLANNING_SYSTEM_PROMPT = load_prompt_template("planning_initial.system.prompt.md")
INITIAL_PLANNING_USER_PROMPT = load_prompt_template("planning_initial.user.prompt.md")
VERIFY_PLANNING_SYSTEM_PROMPT = load_prompt_template("planning_verify.system.prompt.md")
VERIFY_PLANNING_USER_PROMPT = load_prompt_template("planning_verify.user.prompt.md")
EXECUTOR_SYSTEM_PROMPT = load_prompt_template("executor.system.prompt.md")
EXECUTOR_USER_PROMPT = load_prompt_template("executor.user.prompt.md")
FAST_INITIAL_PLANNING_SYSTEM_PROMPT = load_prompt_template(
    "fast/planning_initial.system.prompt.md"
)
FAST_INITIAL_PLANNING_USER_PROMPT = load_prompt_template(
    "fast/planning_initial.user.prompt.md"
)
FAST_VERIFY_PLANNING_SYSTEM_PROMPT = load_prompt_template(
    "fast/planning_verify.system.prompt.md"
)
FAST_VERIFY_PLANNING_USER_PROMPT = load_prompt_template(
    "fast/planning_verify.user.prompt.md"
)
FAST_EXECUTOR_SYSTEM_PROMPT = load_prompt_template("fast/executor.system.prompt.md")
FAST_EXECUTOR_USER_PROMPT = load_prompt_template("fast/executor.user.prompt.md")
DEFAULT_ALLOWED_ACTION_NAMES = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")


def _normalize_prompt_profile(value=None) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _use_fast_prompt_profile(model_name=None, prompt_profile=None) -> bool:
    del model_name
    profile = _normalize_prompt_profile(
        prompt_profile or os.getenv("SPACEVLN_PROMPT_PROFILE", "")
    )
    return profile in {"fast", "compressed", "compact"}


def _read_positive_int_env(name: str, default: int) -> int:
    raw_value = str(os.getenv(name, "") or "").strip()
    if not raw_value:
        return int(default)
    try:
        return max(1, int(float(raw_value)))
    except (TypeError, ValueError):
        return int(default)


def _read_positive_float_env(name: str, default: float) -> float:
    raw_value = str(os.getenv(name, "") or "").strip()
    if not raw_value:
        return float(default)
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed > 0.0 else float(default)


def _fmt_degrees(value: float) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 30.0
    if abs(numeric - round(numeric)) < 1e-6:
        return str(int(round(numeric)))
    return f"{numeric:g}"


def _apply_lookaround_prompt_overrides(prompt: str) -> str:
    view_count = _read_positive_int_env("SPACEVLN_LOOKAROUND_VIEW_COUNT", 12)
    step_deg = _read_positive_float_env("SPACEVLN_LOOKAROUND_STEP_DEG", 30.0)
    allow_missing_map = str(os.getenv("SPACEVLN_ALLOW_MISSING_GLOBAL_MAP", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    allow_generic_waypoints = str(os.getenv("SPACEVLN_ALLOW_GENERIC_WAYPOINT_LABELS", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if (
        view_count == 12
        and abs(step_deg - 30.0) < 1e-6
        and not allow_missing_map
        and not allow_generic_waypoints
    ):
        return prompt

    text = str(prompt or "")
    step_text = _fmt_degrees(step_deg)
    replacements = (
        ("`12 Views`", f"`{view_count} Views`"),
        ("12 Views", f"{view_count} Views"),
        ("12 views", f"{view_count} views"),
        ("12-View", f"{view_count}-View"),
        ("12-view", f"{view_count}-view"),
        ("12-IMAGE", f"{view_count}-IMAGE"),
        ("12 IMAGE", f"{view_count} IMAGE"),
        ("IMAGE 1-12", f"IMAGE 1-{view_count}"),
        ("IMAGE1-12", f"IMAGE1-{view_count}"),
        ("IMAGE 12", f"IMAGE {view_count}"),
        ("IMAGE12", f"IMAGE{view_count}"),
        ("all 12 views", f"all {view_count} views"),
        ("all 12 Views", f"all {view_count} Views"),
        ("current 12-view", f"current {view_count}-view"),
        ("real 12-view", f"real {view_count}-view"),
        ("sampled every 30°", f"sampled every {step_text}°"),
        ("sampled every 30deg", f"sampled every {step_text}deg"),
    )
    for old, new in replacements:
        text = text.replace(old, new)

    if view_count == 8 and abs(step_deg - 45.0) < 1e-6:
        label_replacements = (
            ("IMAGE 2 (Left 30°)", "IMAGE 2 (Left 45°)"),
            ("IMAGE 2 (Left 30deg)", "IMAGE 2 (Left 45deg)"),
            ("IMAGE 3 (Left 60°)", "IMAGE 2 (Left 45°)"),
            ("IMAGE 3 (Left 60deg)", "IMAGE 2 (Left 45deg)"),
            ("IMAGE 4 (Left 90°)", "IMAGE 3 (Left 90°)"),
            ("IMAGE 4 (Left 90deg)", "IMAGE 3 (Left 90deg)"),
            ("IMAGE 5 (Left 120°)", "IMAGE 4 (Left 135°)"),
            ("IMAGE 5 (Left 120deg)", "IMAGE 4 (Left 135deg)"),
            ("IMAGE 6 (Left 150°)", "IMAGE 4 (Left 135°)"),
            ("IMAGE 6 (Left 150deg)", "IMAGE 4 (Left 135deg)"),
            ("IMAGE 7 (Back 180°)", "IMAGE 5 (Back 180°)"),
            ("IMAGE 7 (Back 180deg)", "IMAGE 5 (Back 180deg)"),
            ("IMAGE 8 (Right 150°)", "IMAGE 6 (Right 135°)"),
            ("IMAGE 8 (Right 150deg)", "IMAGE 6 (Right 135deg)"),
        )
        for old, new in label_replacements:
            text = text.replace(old, new)

    if allow_missing_map and "Real-robot depth-map note" not in text:
        text += (
            "\n\n**Real-robot depth-map note**: if the Global Map image states that "
            "depth mapping is disabled, do not infer route geometry from that map. "
            "Use the stopped RGB views, their IMAGE labels, and per-view obstacle distances."
        )
    if allow_generic_waypoints and "Real-robot waypoint label note" not in text:
        text += (
            "\n\n**Real-robot waypoint label note**: prefer concrete space/landmark names, "
            "but if the live scene is ambiguous, output the best useful area/room/space label "
            "instead of failing or retrying only to avoid a generic word."
        )
    return text


def _select_prompt_template(default_template: str, fast_template: str, *, model_name=None, prompt_profile=None) -> str:
    if _use_fast_prompt_profile(model_name=model_name, prompt_profile=prompt_profile):
        return fast_template
    return default_template


def _fmt_threshold_m(value: float) -> str:
    text = f"{float(value):.2f}"
    if text.endswith("00"):
        return f"{float(value):.1f}"
    return text.rstrip("0").rstrip(".")


def _fmt_action_meters(value: float) -> str:
    text = f"{float(value):g}"
    return f"{text}m"


def _resolve_move_values(move_distance: float = 0.25) -> tuple:
    try:
        base_distance = float(move_distance)
    except (TypeError, ValueError):
        base_distance = 0.25
    if base_distance >= 0.5:
        return (0.5, 0.75, 1.0, 1.25, 1.5)
    return tuple(float(value) for value in VALID_MOVE_METERS)


def _move_action_choices(move_distance: float = 0.25) -> list:
    return [
        f"MOVE_FORWARD {_fmt_action_meters(value)}"
        for value in _resolve_move_values(move_distance)
    ]


def _move_action_set_text(move_distance: float = 0.25) -> str:
    return ", ".join(
        _fmt_action_meters(value)
        for value in _resolve_move_values(move_distance)
    )


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


def _render_initial_planning_system_prompt(*, model_name=None, prompt_profile=None) -> str:
    template = _select_prompt_template(
        INITIAL_PLANNING_SYSTEM_PROMPT,
        FAST_INITIAL_PLANNING_SYSTEM_PROMPT,
        model_name=model_name,
        prompt_profile=prompt_profile,
    )
    return _apply_lookaround_prompt_overrides(_normalize_anchor_notation_text(template.format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(ARRIVAL_NEAR_M),
    )))


def build_initial_planner_prompt_bundle(
    *,
    instruction: str,
    action_space: str,
    model_name: str = None,
    prompt_profile: str = None,
) -> PromptBundle:
    """Render the initial-planning system/user prompt bundle."""
    del action_space
    system_prompt = _render_initial_planning_system_prompt(
        model_name=model_name,
        prompt_profile=prompt_profile,
    )
    user_template = _select_prompt_template(
        INITIAL_PLANNING_USER_PROMPT,
        FAST_INITIAL_PLANNING_USER_PROMPT,
        model_name=model_name,
        prompt_profile=prompt_profile,
    )
    user_prompt = user_template.format(instruction=instruction)
    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=compose_full_prompt(system_prompt, user_prompt),
    )


def get_initial_planning_prompt(
    instruction: str,
    action_space: str,
    *,
    model_name: str = None,
    prompt_profile: str = None,
) -> str:
    """Compatibility helper returning the combined initial-planning prompt."""
    return build_initial_planner_prompt_bundle(
        instruction=instruction,
        action_space=action_space,
        model_name=model_name,
        prompt_profile=prompt_profile,
    ).full_prompt


def _get_verify_view_count(direction_names=None):
    provided_direction_names = [
        str(name).strip()
        for name in list(direction_names or [])
        if str(name or "").strip()
    ]
    view_count = len(provided_direction_names)
    return view_count if 0 < view_count < 12 else 12


def _render_verify_planning_system_prompt(*, model_name=None, prompt_profile=None) -> str:
    template = _select_prompt_template(
        VERIFY_PLANNING_SYSTEM_PROMPT,
        FAST_VERIFY_PLANNING_SYSTEM_PROMPT,
        model_name=model_name,
        prompt_profile=prompt_profile,
    )
    return _apply_lookaround_prompt_overrides(_normalize_anchor_notation_text(template.format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(ARRIVAL_NEAR_M),
    )))


def build_verify_planner_prompt_bundle(
    *,
    instruction: str,
    subtask_destination: str,
    subtask_instruction: str,
    action_space: str,
    detected_landmarks: str = None,
    waypoint_summary: str = None,
    previous_subtask_landmark_summary: str = None,
    verify_replan_prompt_notice: str = None,
    direction_names: list = None,
    model_name: str = None,
    prompt_profile: str = None,
) -> PromptBundle:
    """Render the verification/replanning system/user prompt bundle."""
    del action_space
    del detected_landmarks
    del direction_names
    if not waypoint_summary:
        waypoint_summary = "Unavailable"

    previous_subtask_landmark_summary = str(previous_subtask_landmark_summary or "").strip()
    previous_subtask_landmark_block = _build_previous_subtask_landmark_block(
        previous_subtask_landmark_summary
    )
    verify_notice_block = str(verify_replan_prompt_notice or "").strip()

    system_prompt = _render_verify_planning_system_prompt(
        model_name=model_name,
        prompt_profile=prompt_profile,
    )
    user_template = _select_prompt_template(
        VERIFY_PLANNING_USER_PROMPT,
        FAST_VERIFY_PLANNING_USER_PROMPT,
        model_name=model_name,
        prompt_profile=prompt_profile,
    )
    user_prompt = user_template.format(
        verify_replan_prompt_notice_block=verify_notice_block,
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        waypoint_summary=waypoint_summary,
        previous_subtask_landmark_block=previous_subtask_landmark_block,
    )
    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=compose_full_prompt(system_prompt, user_prompt),
    )


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
    model_name: str = None,
    prompt_profile: str = None,
) -> str:
    """Compatibility helper returning the combined verification/replanning prompt."""
    return build_verify_planner_prompt_bundle(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        action_space=action_space,
        detected_landmarks=detected_landmarks,
        waypoint_summary=waypoint_summary,
        previous_subtask_landmark_summary=previous_subtask_landmark_summary,
        verify_replan_prompt_notice=verify_replan_prompt_notice,
        direction_names=direction_names,
        model_name=model_name,
        prompt_profile=prompt_profile,
    ).full_prompt


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


def _build_obstacle_perception_summary(obstacle_distances=None, turn_angle: float = 30) -> str:
    distances = dict(obstacle_distances or {})
    side_angle = _fmt_degrees(turn_angle)
    items = []
    for label, key in (
        ("FRONT", "front"),
        (f"Left {side_angle}deg", "left_30"),
        (f"Right {side_angle}deg", "right_30"),
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


def _build_allowed_action_output(allowed_action_names=None, move_distance: float = 0.25, turn_angle: float = 30) -> str:
    ordered = _normalize_allowed_action_names(allowed_action_names)
    turn_text = _fmt_degrees(turn_angle)
    choices = []
    if "MOVE_FORWARD" in ordered:
        choices.extend(_move_action_choices(move_distance))
    if "TURN_LEFT" in ordered:
        choices.extend([f"TURN_LEFT_AVOID {turn_text}deg", f"TURN_LEFT_ALIGN {turn_text}deg"])
    if "TURN_RIGHT" in ordered:
        choices.extend([f"TURN_RIGHT_AVOID {turn_text}deg", f"TURN_RIGHT_ALIGN {turn_text}deg"])
    if "STOP" in ordered:
        choices.append("STOP")
    return " | ".join(choices)


def _build_allowed_action_bullets(allowed_action_names=None, move_distance: float = 0.25, turn_angle: float = 30) -> str:
    ordered = _normalize_allowed_action_names(allowed_action_names)
    turn_text = _fmt_degrees(turn_angle)
    lines = []
    if "MOVE_FORWARD" in ordered:
        lines.append(
            f"- `MOVE_FORWARD {{{_move_action_set_text(move_distance)}}}`: move forward by the selected distance"
        )
    if "TURN_LEFT" in ordered:
        lines.append(
            f"- `TURN_LEFT_AVOID {turn_text}deg` to avoid obstacle only when FRONT <{_fmt_threshold_m(OBS_BLOCKED_M)}m or the current FRONT route is unusable | "
            f"`TURN_LEFT_ALIGN {turn_text}deg` to align destination landmark"
        )
    if "TURN_RIGHT" in ordered:
        lines.append(
            f"- `TURN_RIGHT_AVOID {turn_text}deg` to avoid obstacle only when FRONT <{_fmt_threshold_m(OBS_BLOCKED_M)}m or the current FRONT route is unusable | "
            f"`TURN_RIGHT_ALIGN {turn_text}deg` to align destination landmark"
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


def _normalize_action_prompt_text(prompt: str, *, move_distance: float = 0.25, turn_angle: float = 30) -> str:
    normalized = str(prompt or "")
    turn_text = _fmt_degrees(turn_angle)
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
            f"Output `action` only from the fixed action space: `TURN_LEFT_AVOID 30deg` / `TURN_LEFT_ALIGN 30deg` / `TURN_RIGHT_AVOID 30deg` / `TURN_RIGHT_ALIGN 30deg` / `MOVE_FORWARD {{{{{_move_action_set_text(move_distance)}}}}}` / `STOP`.",
        ),
        (
            "Output `action` only from the fixed action space: `TURN_LEFT 30deg` / `TURN_RIGHT 30deg` / `MOVE_FORWARD {0.25m, 0.5m, 0.75m, 1.0m, 1.25m}` / `STOP`.",
            f"Output `action` only from the fixed action space: `TURN_LEFT_AVOID 30deg` / `TURN_LEFT_ALIGN 30deg` / `TURN_RIGHT_AVOID 30deg` / `TURN_RIGHT_ALIGN 30deg` / `MOVE_FORWARD {{{_move_action_set_text(move_distance)}}}` / `STOP`.",
        ),
        (
            "toward the current-stage destination",
            "toward the current destination",
        ),
    )
    for old, new in literal_replacements:
        normalized = normalized.replace(old, new)

    if turn_text != "30":
        normalized = normalized.replace(
            "Left 30deg, FRONT, Right 30deg",
            f"Left {turn_text}deg, FRONT, Right {turn_text}deg",
        )
        normalized = normalized.replace(
            "FRONT / Left 30deg / Right 30deg",
            f"FRONT / Left {turn_text}deg / Right {turn_text}deg",
        )
        for action_prefix in (
            "TURN_LEFT_AVOID",
            "TURN_LEFT_ALIGN",
            "TURN_RIGHT_AVOID",
            "TURN_RIGHT_ALIGN",
            "TURN_LEFT",
            "TURN_RIGHT",
        ):
            normalized = normalized.replace(
                f"{action_prefix} 30deg",
                f"{action_prefix} {turn_text}deg",
            )

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


def _render_executor_system_prompt(*, model_name=None, prompt_profile=None) -> str:
    template = _select_prompt_template(
        EXECUTOR_SYSTEM_PROMPT,
        FAST_EXECUTOR_SYSTEM_PROMPT,
        model_name=model_name,
        prompt_profile=prompt_profile,
    )
    return template.format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        solid_autocomplete_m=_fmt_threshold_m(
            ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M
        ),
        open_autocomplete_m=_fmt_threshold_m(
            ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M
        ),
    )


def build_executor_prompt_bundle(
    *,
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
    model_name: str = None,
    prompt_profile: str = None,
) -> PromptBundle:
    """Render the executor system/user prompt bundle."""
    del waypoint_summary
    if not progress_summary:
        progress_summary = "Just started"

    system_prompt = _normalize_action_prompt_text(_render_executor_system_prompt(
        model_name=model_name,
        prompt_profile=prompt_profile,
    ), move_distance=move_distance, turn_angle=turn_angle)
    user_template = _select_prompt_template(
        EXECUTOR_USER_PROMPT,
        FAST_EXECUTOR_USER_PROMPT,
        model_name=model_name,
        prompt_profile=prompt_profile,
    )
    user_prompt = _normalize_action_prompt_text(user_template.format(
        subtask_destination=next_waypoint,
        subtask_landmark=str(subtask_landmark or "").strip() or "none",
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary,
        detected_landmarks=detected_landmarks or "none",
            obstacle_perception_summary=_build_obstacle_perception_summary(obstacle_distances, turn_angle),
        landmark_perception_summary=_build_landmark_perception_summary(
            detected_landmarks=detected_landmarks,
            landmark_map_info=landmark_map_info,
        ),
        allowed_action_output=_build_allowed_action_output(allowed_action_names, move_distance, turn_angle),
        allowed_action_bullets=_build_allowed_action_bullets(allowed_action_names, move_distance, turn_angle),
        action_space_constraint_notice=_build_action_space_constraint_notice(allowed_action_names),
    ), move_distance=move_distance, turn_angle=turn_angle)
    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=compose_full_prompt(system_prompt, user_prompt),
    )


def get_executor_prompt(
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
    model_name: str = None,
    prompt_profile: str = None,
) -> str:
    """Return the combined executor prompt."""
    return build_executor_prompt_bundle(
        next_waypoint=next_waypoint,
        subtask_instruction=subtask_instruction,
        subtask_landmark=subtask_landmark,
        progress_summary=progress_summary,
        waypoint_summary=waypoint_summary,
        detected_landmarks=detected_landmarks,
        obstacle_distances=obstacle_distances,
        landmark_map_info=landmark_map_info,
        allowed_action_names=allowed_action_names,
        move_distance=move_distance,
        turn_angle=turn_angle,
        model_name=model_name,
        prompt_profile=prompt_profile,
    ).full_prompt


__all__ = [
    "DEFAULT_ALLOWED_ACTION_NAMES",
    "EXECUTOR_SYSTEM_PROMPT",
    "EXECUTOR_USER_PROMPT",
    "INITIAL_PLANNING_SYSTEM_PROMPT",
    "INITIAL_PLANNING_USER_PROMPT",
    "VERIFY_PLANNING_SYSTEM_PROMPT",
    "VERIFY_PLANNING_USER_PROMPT",
    "_build_allowed_action_bullets",
    "_build_allowed_action_output",
    "_build_action_space_constraint_notice",
    "_build_landmark_perception_summary",
    "_normalize_action_prompt_text",
    "_build_obstacle_perception_summary",
    "_fmt_threshold_m",
    "_get_verify_view_count",
    "build_executor_prompt_bundle",
    "build_initial_planner_prompt_bundle",
    "build_verify_planner_prompt_bundle",
    "get_executor_prompt",
    "get_initial_planning_prompt",
    "get_verification_replanning_prompt",
]
