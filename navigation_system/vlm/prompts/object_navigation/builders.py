"""OVON-specific system/user prompt builders."""

from navigation_system.config.core.params.thresholds import (
    OBS_BLOCKED_M,
    OBS_OPEN_M,
    OBS_RISKY_M,
)
from navigation_system.runtime.object_navigation.ovon.thresholds import (
    OVON_ARRIVAL_NEAR_M,
    OVON_AUTOCOMPLETE_OPENING_M,
    OVON_AUTOCOMPLETE_SOLID_M,
    OVON_FINAL_OBJECT_STOP_DISTANCE_M,
)
from navigation_system.vlm.prompts.common import (
    PromptBundle,
    compose_full_prompt,
)
from navigation_system.vlm.prompts.object_navigation.common import (
    load_objectnav_prompt_template,
)
from navigation_system.vlm.prompts.vlnce.builders import (
    _build_action_space_constraint_notice,
    _build_allowed_action_bullets,
    _build_allowed_action_output,
    _build_landmark_perception_summary,
    _build_obstacle_perception_summary,
    _build_previous_subtask_landmark_block,
    _normalize_action_prompt_text,
    _normalize_anchor_notation_text,
)


INITIAL_PLANNING_SYSTEM_PROMPT = load_objectnav_prompt_template("planning_initial.system.prompt.md")
INITIAL_PLANNING_USER_PROMPT = load_objectnav_prompt_template("planning_initial.user.prompt.md")
VERIFY_PLANNING_SYSTEM_PROMPT = load_objectnav_prompt_template("planning_verify.system.prompt.md")
VERIFY_PLANNING_USER_PROMPT = load_objectnav_prompt_template("planning_verify.user.prompt.md")
EXECUTOR_SYSTEM_PROMPT = load_objectnav_prompt_template("executor.system.prompt.md")
EXECUTOR_USER_PROMPT = load_objectnav_prompt_template("executor.user.prompt.md")


def _fmt_threshold_m(value: float) -> str:
    text = f"{float(value):.2f}"
    if text.endswith("00"):
        return f"{float(value):.1f}"
    return text.rstrip("0").rstrip(".")


def _render_initial_planning_system_prompt() -> str:
    return _normalize_anchor_notation_text(INITIAL_PLANNING_SYSTEM_PROMPT.format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(OVON_ARRIVAL_NEAR_M),
        strict_stop_m=_fmt_threshold_m(OVON_FINAL_OBJECT_STOP_DISTANCE_M),
    ))


def build_ovon_initial_planner_prompt_bundle(
    *,
    instruction: str,
    action_space: str,
) -> PromptBundle:
    del action_space
    system_prompt = _render_initial_planning_system_prompt()
    user_prompt = INITIAL_PLANNING_USER_PROMPT.format(instruction=instruction)
    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=compose_full_prompt(system_prompt, user_prompt),
    )


def get_ovon_initial_planning_prompt(instruction: str, action_space: str) -> str:
    return build_ovon_initial_planner_prompt_bundle(
        instruction=instruction,
        action_space=action_space,
    ).full_prompt


def _render_verify_planning_system_prompt() -> str:
    return _normalize_anchor_notation_text(VERIFY_PLANNING_SYSTEM_PROMPT.format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(OVON_ARRIVAL_NEAR_M),
        strict_stop_m=_fmt_threshold_m(OVON_FINAL_OBJECT_STOP_DISTANCE_M),
    ))


def build_ovon_verify_planner_prompt_bundle(
    *,
    instruction: str,
    subtask_destination: str,
    subtask_instruction: str,
    action_space: str,
    detected_landmarks=None,
    waypoint_summary: str = None,
    previous_subtask_landmark_summary: str = None,
    verify_replan_prompt_notice: str = None,
    direction_names=None,
) -> PromptBundle:
    del action_space
    del detected_landmarks
    del direction_names
    previous_subtask_landmark_block = _build_previous_subtask_landmark_block(
        previous_subtask_landmark_summary
    )
    verify_notice_block = str(verify_replan_prompt_notice or "").strip()
    system_prompt = _render_verify_planning_system_prompt()
    user_prompt = VERIFY_PLANNING_USER_PROMPT.format(
        verify_replan_prompt_notice_block=verify_notice_block,
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        previous_subtask_landmark_block=previous_subtask_landmark_block,
        waypoint_summary=str(waypoint_summary or "Unavailable"),
    )
    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=compose_full_prompt(system_prompt, user_prompt),
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
    return build_ovon_verify_planner_prompt_bundle(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        action_space=action_space,
        waypoint_summary=waypoint_summary,
        previous_subtask_landmark_summary=previous_subtask_landmark_summary,
        verify_replan_prompt_notice=verify_replan_prompt_notice,
    ).full_prompt


def _render_executor_system_prompt() -> str:
    return EXECUTOR_SYSTEM_PROMPT.format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        solid_autocomplete_m=_fmt_threshold_m(OVON_AUTOCOMPLETE_SOLID_M),
        open_autocomplete_m=_fmt_threshold_m(OVON_AUTOCOMPLETE_OPENING_M),
        arrival_near_m=_fmt_threshold_m(OVON_ARRIVAL_NEAR_M),
        strict_stop_m=_fmt_threshold_m(OVON_FINAL_OBJECT_STOP_DISTANCE_M),
    )


def build_ovon_executor_prompt_bundle(
    *,
    next_waypoint: str,
    subtask_instruction: str,
    subtask_landmark: str = "",
    progress_summary: str = "",
    waypoint_summary: str = "",
    detected_landmarks: str = None,
    obstacle_distances: dict = None,
    landmark_map_info: str = None,
    allowed_action_names=None,
    move_distance: float = 0.25,
    turn_angle: int = 30,
) -> PromptBundle:
    del waypoint_summary
    del move_distance
    del turn_angle
    system_prompt = _normalize_action_prompt_text(_render_executor_system_prompt())
    user_prompt = _normalize_action_prompt_text(EXECUTOR_USER_PROMPT.format(
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
        allowed_action_output=_build_allowed_action_output(allowed_action_names),
        allowed_action_bullets=_build_allowed_action_bullets(allowed_action_names),
        action_space_constraint_notice=_build_action_space_constraint_notice(
            allowed_action_names
        ),
    ))
    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=compose_full_prompt(system_prompt, user_prompt),
    )


def get_ovon_executor_prompt(
    next_waypoint: str,
    subtask_instruction: str,
    subtask_landmark: str = "",
    progress_summary: str = "",
    detected_landmarks: str = None,
    obstacle_distances: dict = None,
    landmark_map_info: str = None,
    allowed_action_names=None,
) -> str:
    return build_ovon_executor_prompt_bundle(
        next_waypoint=next_waypoint,
        subtask_instruction=subtask_instruction,
        subtask_landmark=subtask_landmark,
        progress_summary=progress_summary,
        detected_landmarks=detected_landmarks,
        obstacle_distances=obstacle_distances,
        landmark_map_info=landmark_map_info,
        allowed_action_names=allowed_action_names,
    ).full_prompt

