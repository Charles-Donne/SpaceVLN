"""OVON-specific prompt builders."""

from navigation_system.config.core.params.thresholds import (
    OBS_BLOCKED_M,
    OBS_OPEN_M,
    OBS_RISKY_M,
)
from navigation_system.object_navigation.goal_task import (
    build_object_goal_plan,
    parse_object_goal_instruction,
)
from navigation_system.object_navigation.prompts.common import (
    load_objectnav_prompt_template,
)
from navigation_system.object_navigation.thresholds import (
    OVON_ARRIVAL_NEAR_M,
    OVON_AUTOCOMPLETE_OPENING_M,
    OVON_AUTOCOMPLETE_SOLID_M,
)
from navigation_system.vlm.prompts.builders import (
    _build_allowed_action_bullets,
    _build_allowed_action_output,
    _build_landmark_perception_summary,
    _build_obstacle_perception_summary,
)


INITIAL_PLANNING_PROMPT = load_objectnav_prompt_template("planning_initial.prompt.md")
VERIFICATION_REPLANNING_PROMPT = load_objectnav_prompt_template("planning_verify.prompt.md")
ACTION_EXECUTION_PROMPT = load_objectnav_prompt_template("action_execution.prompt.md")


def _fmt_threshold_m(value: float) -> str:
    text = f"{float(value):.2f}"
    if text.endswith("00"):
        return f"{float(value):.1f}"
    return text.rstrip("0").rstrip(".")


def _build_goal_context(instruction: str):
    object_goal, aliases = parse_object_goal_instruction(instruction)
    plan = build_object_goal_plan(object_goal, aliases)
    likely_spaces = ", ".join(plan.likely_spaces) if plan.likely_spaces else "unknown"
    proxy_landmarks = ", ".join(plan.proxy_landmarks) if plan.proxy_landmarks else "none"
    alias_text = ", ".join(plan.child_categories) if plan.child_categories else "none"
    likely_spaces_hint_block = (
        "Soft semantic room priors (hints only; observations override them): "
        f"{likely_spaces}.\n"
        f"Useful supporting landmark cues (hints only): {proxy_landmarks}."
    )
    return object_goal, alias_text, likely_spaces_hint_block


def get_ovon_initial_planning_prompt(instruction: str, action_space: str) -> str:
    object_goal, alias_text, likely_spaces_hint_block = _build_goal_context(instruction)
    return INITIAL_PLANNING_PROMPT.format(
        instruction=instruction,
        object_goal=object_goal,
        goal_aliases=alias_text,
        likely_spaces_hint_block=likely_spaces_hint_block,
        action_space=action_space,
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(OVON_ARRIVAL_NEAR_M),
    )


def get_ovon_verification_replanning_prompt(
    instruction: str,
    subtask_destination: str,
    subtask_instruction: str,
    action_space: str,
    waypoint_summary: str = None,
    previous_subtask_landmark_summary: str = None,
    verify_replan_prompt_notice: str = None,
) -> str:
    object_goal, alias_text, likely_spaces_hint_block = _build_goal_context(instruction)
    verify_replan_prompt_notice_block = (
        f"\n**Recovery Notice**: {verify_replan_prompt_notice.strip()}"
        if str(verify_replan_prompt_notice or "").strip()
        else ""
    )
    previous_subtask_landmark_block = (
        f"- {previous_subtask_landmark_summary.strip()}"
        if str(previous_subtask_landmark_summary or "").strip()
        else ""
    )
    return VERIFICATION_REPLANNING_PROMPT.format(
        instruction=instruction,
        object_goal=object_goal,
        goal_aliases=alias_text,
        likely_spaces_hint_block=likely_spaces_hint_block,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        waypoint_summary=str(waypoint_summary or "Unavailable"),
        previous_subtask_landmark_block=previous_subtask_landmark_block,
        verify_replan_prompt_notice_block=verify_replan_prompt_notice_block,
        action_space=action_space,
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(OVON_ARRIVAL_NEAR_M),
    )


def get_ovon_action_execution_prompt(
    next_waypoint: str,
    subtask_instruction: str,
    progress_summary: str = "",
    detected_landmarks: str = None,
    obstacle_distances: dict = None,
    landmark_map_info: str = None,
    allowed_action_names=None,
) -> str:
    return ACTION_EXECUTION_PROMPT.format(
        subtask_destination=next_waypoint,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary or "(Just started - no actions yet)",
        obstacle_perception_summary=_build_obstacle_perception_summary(obstacle_distances),
        landmark_perception_summary=_build_landmark_perception_summary(
            detected_landmarks=detected_landmarks,
            landmark_map_info=landmark_map_info,
        ),
        detected_landmarks=str(detected_landmarks or "none"),
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        solid_autocomplete_m=_fmt_threshold_m(OVON_AUTOCOMPLETE_SOLID_M),
        open_autocomplete_m=_fmt_threshold_m(OVON_AUTOCOMPLETE_OPENING_M),
        arrival_near_m=_fmt_threshold_m(OVON_ARRIVAL_NEAR_M),
        allowed_action_output=_build_allowed_action_output(allowed_action_names),
        allowed_action_bullets=_build_allowed_action_bullets(allowed_action_names),
    )
