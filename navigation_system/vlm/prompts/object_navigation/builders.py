"""OVON-specific prompt builders."""

from navigation_system.config.core.params.thresholds import (
    OBS_BLOCKED_M,
    OBS_OPEN_M,
    OBS_RISKY_M,
)
from navigation_system.runtime.object_navigation.thresholds import (
    OVON_ARRIVAL_NEAR_M,
    OVON_AUTOCOMPLETE_OPENING_M,
    OVON_AUTOCOMPLETE_SOLID_M,
    OVON_FINAL_OBJECT_STOP_DISTANCE_M,
)
from navigation_system.vlm.prompts.object_navigation.common import (
    load_objectnav_prompt_template,
)
from navigation_system.vlm.prompts.vlnce.builders import (
    _build_allowed_action_bullets,
    _build_allowed_action_output,
    _build_landmark_perception_summary,
    _build_obstacle_perception_summary,
    _normalize_anchor_notation_text,
)


INITIAL_PLANNING_PROMPT = load_objectnav_prompt_template("planning_initial.prompt.md")
VERIFICATION_REPLANNING_PROMPT = load_objectnav_prompt_template("planning_verify.prompt.md")
ACTION_EXECUTION_PROMPT = load_objectnav_prompt_template("action_execution.prompt.md")


def _fmt_threshold_m(value: float) -> str:
    text = f"{float(value):.2f}"
    if text.endswith("00"):
        return f"{float(value):.1f}"
    return text.rstrip("0").rstrip(".")


def get_ovon_initial_planning_prompt(instruction: str, action_space: str) -> str:
    return _normalize_anchor_notation_text(INITIAL_PLANNING_PROMPT.format(
        instruction=instruction,
        action_space=action_space,
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(OVON_ARRIVAL_NEAR_M),
        strict_stop_m=_fmt_threshold_m(OVON_FINAL_OBJECT_STOP_DISTANCE_M),
    ))


def get_ovon_verification_replanning_prompt(
    instruction: str,
    subtask_destination: str,
    subtask_instruction: str,
    action_space: str,
    waypoint_summary: str = None,
    previous_subtask_landmark_summary: str = None,
    verify_replan_prompt_notice: str = None,
) -> str:
    verify_replan_prompt_notice_block = (
        f"\n**Stuck Notice**: {verify_replan_prompt_notice.strip()}"
        if str(verify_replan_prompt_notice or "").strip()
        else ""
    )
    previous_subtask_landmark_block = (
        f"- {previous_subtask_landmark_summary.strip()}"
        if str(previous_subtask_landmark_summary or "").strip()
        else ""
    )
    return _normalize_anchor_notation_text(VERIFICATION_REPLANNING_PROMPT.format(
        instruction=instruction,
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
        strict_stop_m=_fmt_threshold_m(OVON_FINAL_OBJECT_STOP_DISTANCE_M),
    ))


def get_ovon_action_execution_prompt(
    next_waypoint: str,
    subtask_instruction: str,
    subtask_landmark: str = "",
    progress_summary: str = "",
    detected_landmarks: str = None,
    obstacle_distances: dict = None,
    landmark_map_info: str = None,
    allowed_action_names=None,
) -> str:
    return ACTION_EXECUTION_PROMPT.format(
        subtask_destination=next_waypoint,
        subtask_landmark=str(subtask_landmark or "").strip() or "none",
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
        strict_stop_m=_fmt_threshold_m(OVON_FINAL_OBJECT_STOP_DISTANCE_M),
        allowed_action_output=_build_allowed_action_output(allowed_action_names),
        allowed_action_bullets=_build_allowed_action_bullets(allowed_action_names),
    )
