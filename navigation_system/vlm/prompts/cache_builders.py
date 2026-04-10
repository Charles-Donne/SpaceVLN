"""Explicit-cache prompt builders backed by dedicated markdown templates."""

from typing import Dict, Optional, Sequence

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
from navigation_system.vlm.prompts.builders import (
    _build_allowed_action_bullets,
    _build_landmark_perception_summary,
    _build_obstacle_perception_summary,
    _fmt_threshold_m,
)
from navigation_system.vlm.prompts.common import (
    ExplicitCachePromptBundle,
    compose_full_prompt,
    load_prompt_template,
)


INITIAL_PLANNING_CACHE_SYSTEM_PROMPT = load_prompt_template(
    "cache/planning_initial.system.prompt.md"
)
INITIAL_PLANNING_CACHE_USER_PROMPT = load_prompt_template(
    "cache/planning_initial.user.prompt.md"
)
VERIFY_PLANNING_CACHE_SYSTEM_PROMPT = load_prompt_template(
    "cache/planning_verify.system.prompt.md"
)
VERIFY_PLANNING_CACHE_USER_PROMPT = load_prompt_template(
    "cache/planning_verify.user.prompt.md"
)
ACTION_CACHE_SYSTEM_PROMPT = load_prompt_template("cache/action.system.prompt.md")
ACTION_CACHE_USER_PROMPT = load_prompt_template("cache/action.user.prompt.md")


def _render_initial_planning_system_prompt() -> str:
    return INITIAL_PLANNING_CACHE_SYSTEM_PROMPT.format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(ARRIVAL_NEAR_M),
    )


def _render_verify_planning_system_prompt() -> str:
    return VERIFY_PLANNING_CACHE_SYSTEM_PROMPT.format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(ARRIVAL_NEAR_M),
    )


def _render_action_system_prompt() -> str:
    return ACTION_CACHE_SYSTEM_PROMPT.format(
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


def build_initial_planner_cache_prompt_bundle(
    *,
    instruction: str,
    action_space: str,
) -> ExplicitCachePromptBundle:
    """Build the initial planning cache prompt from dedicated templates."""
    del action_space

    system_prompt = _render_initial_planning_system_prompt()
    user_prompt = INITIAL_PLANNING_CACHE_USER_PROMPT.format(
        instruction=instruction,
    )
    full_prompt = compose_full_prompt(system_prompt, user_prompt)
    return ExplicitCachePromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=full_prompt,
    )


def build_verify_planner_cache_prompt_bundle(
    *,
    instruction: str,
    subtask_destination: str,
    subtask_instruction: str,
    action_space: str,
    detected_landmarks: Optional[str],
    waypoint_summary: Optional[str],
    previous_subtask_landmark_summary: Optional[str],
    verify_replan_prompt_notice: Optional[str],
    direction_names,
) -> ExplicitCachePromptBundle:
    """Build the verification/replan cache prompt from dedicated templates."""
    del action_space
    del detected_landmarks
    del direction_names

    waypoint_text = str(waypoint_summary or "Unavailable").strip() or "Unavailable"
    previous_subtask_landmark_summary = str(previous_subtask_landmark_summary or "").strip()
    previous_subtask_landmark_block = (
        f"- {previous_subtask_landmark_summary}"
        if previous_subtask_landmark_summary
        else ""
    )
    notice_block_for_user = ""
    if str(verify_replan_prompt_notice or "").strip():
        notice_text = str(verify_replan_prompt_notice).strip()
        notice_block_for_user = f"**Stuck Notice**: {notice_text}"

    system_prompt = _render_verify_planning_system_prompt()
    user_prompt = VERIFY_PLANNING_CACHE_USER_PROMPT.format(
        verify_replan_prompt_notice_block=notice_block_for_user,
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        previous_subtask_landmark_block=previous_subtask_landmark_block,
        waypoint_summary=waypoint_text,
    )
    full_prompt = compose_full_prompt(system_prompt, user_prompt)
    return ExplicitCachePromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=full_prompt,
    )


def build_action_cache_prompt_bundle(
    *,
    next_waypoint: str,
    subtask_instruction: str,
    progress_summary: Optional[str],
    waypoint_summary: Optional[str],
    detected_landmarks: Optional[str],
    previous_action_reason: Optional[str],
    obstacle_distances: Optional[Dict[str, str]],
    landmark_map_info: Optional[str],
    allowed_action_names: Optional[Sequence[str]],
    move_distance: float = 0.25,
    turn_angle: int = 30,
) -> ExplicitCachePromptBundle:
    """Build the action cache prompt from dedicated templates."""
    del move_distance
    del turn_angle
    del waypoint_summary

    progress_text = str(progress_summary or "").strip() or "Just started"
    previous_action_text = str(previous_action_reason or "").strip() or "N/A (first step)"
    detected_landmark_text = str(detected_landmarks or "").strip() or "none"
    obstacle_summary = _build_obstacle_perception_summary(obstacle_distances)
    landmark_summary = _build_landmark_perception_summary(
        detected_landmarks=detected_landmarks,
        landmark_map_info=landmark_map_info,
    )
    allowed_action_bullets = _build_allowed_action_bullets(allowed_action_names)
    system_prompt = _render_action_system_prompt()
    user_prompt = ACTION_CACHE_USER_PROMPT.format(
        subtask_destination=next_waypoint,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_text,
        previous_action_reason=previous_action_text,
        obstacle_perception_summary=obstacle_summary,
        landmark_perception_summary=landmark_summary,
        detected_landmarks=detected_landmark_text,
        allowed_action_bullets=allowed_action_bullets,
    )
    full_prompt = compose_full_prompt(system_prompt, user_prompt)
    return ExplicitCachePromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=full_prompt,
    )


__all__ = [
    "ACTION_CACHE_SYSTEM_PROMPT",
    "ACTION_CACHE_USER_PROMPT",
    "INITIAL_PLANNING_CACHE_SYSTEM_PROMPT",
    "INITIAL_PLANNING_CACHE_USER_PROMPT",
    "VERIFY_PLANNING_CACHE_SYSTEM_PROMPT",
    "VERIFY_PLANNING_CACHE_USER_PROMPT",
    "build_action_cache_prompt_bundle",
    "build_initial_planner_cache_prompt_bundle",
    "build_verify_planner_cache_prompt_bundle",
]
