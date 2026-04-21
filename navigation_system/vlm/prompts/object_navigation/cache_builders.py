"""Explicit-cache prompt builders for OVON object navigation."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

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
    _build_landmark_perception_summary,
    _build_obstacle_perception_summary,
    _fmt_threshold_m,
    _normalize_anchor_notation_text,
    _normalize_action_prompt_text,
)
from navigation_system.vlm.prompts.common import (
    ExplicitCachePromptBundle,
    compose_full_prompt,
)


INITIAL_PLANNING_CACHE_SYSTEM_PROMPT = load_objectnav_prompt_template(
    "cache/planning_initial.system.prompt.md"
)
INITIAL_PLANNING_CACHE_USER_PROMPT = load_objectnav_prompt_template(
    "cache/planning_initial.user.prompt.md"
)
VERIFY_PLANNING_CACHE_SYSTEM_PROMPT = load_objectnav_prompt_template(
    "cache/planning_verify.system.prompt.md"
)
VERIFY_PLANNING_CACHE_USER_PROMPT = load_objectnav_prompt_template(
    "cache/planning_verify.user.prompt.md"
)
ACTION_CACHE_SYSTEM_PROMPT = load_objectnav_prompt_template(
    "cache/action.system.prompt.md"
)
ACTION_CACHE_USER_PROMPT = load_objectnav_prompt_template(
    "cache/action.user.prompt.md"
)


def _render_initial_planning_system_prompt() -> str:
    return _normalize_anchor_notation_text(INITIAL_PLANNING_CACHE_SYSTEM_PROMPT.format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(OVON_ARRIVAL_NEAR_M),
        strict_stop_m=_fmt_threshold_m(OVON_FINAL_OBJECT_STOP_DISTANCE_M),
    ))


def _render_verify_planning_system_prompt() -> str:
    return _normalize_anchor_notation_text(VERIFY_PLANNING_CACHE_SYSTEM_PROMPT.format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(OVON_ARRIVAL_NEAR_M),
        strict_stop_m=_fmt_threshold_m(OVON_FINAL_OBJECT_STOP_DISTANCE_M),
    ))


def _render_action_system_prompt() -> str:
    return ACTION_CACHE_SYSTEM_PROMPT.format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        solid_autocomplete_m=_fmt_threshold_m(OVON_AUTOCOMPLETE_SOLID_M),
        open_autocomplete_m=_fmt_threshold_m(OVON_AUTOCOMPLETE_OPENING_M),
        arrival_near_m=_fmt_threshold_m(OVON_ARRIVAL_NEAR_M),
        strict_stop_m=_fmt_threshold_m(OVON_FINAL_OBJECT_STOP_DISTANCE_M),
    )


def build_ovon_initial_planner_cache_prompt_bundle(
    *,
    instruction: str,
    action_space: str,
) -> ExplicitCachePromptBundle:
    """Build the OVON initial-planning cache prompt."""
    del action_space
    system_prompt = _render_initial_planning_system_prompt()
    user_prompt = INITIAL_PLANNING_CACHE_USER_PROMPT.format(
        instruction=instruction,
    )
    return ExplicitCachePromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=compose_full_prompt(system_prompt, user_prompt),
    )


def build_ovon_verify_planner_cache_prompt_bundle(
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
    """Build the OVON verification/replan cache prompt."""
    del action_space
    del detected_landmarks
    del direction_names
    waypoint_text = str(waypoint_summary or "Unavailable").strip() or "Unavailable"
    previous_summary = str(previous_subtask_landmark_summary or "").strip()
    previous_block = f"- {previous_summary}" if previous_summary else ""
    notice_block = ""
    if str(verify_replan_prompt_notice or "").strip():
        notice_block = f"**Stuck Notice**: {str(verify_replan_prompt_notice).strip()}"

    system_prompt = _render_verify_planning_system_prompt()
    user_prompt = VERIFY_PLANNING_CACHE_USER_PROMPT.format(
        verify_replan_prompt_notice_block=notice_block,
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        previous_subtask_landmark_block=previous_block,
        waypoint_summary=waypoint_text,
    )
    return ExplicitCachePromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=compose_full_prompt(system_prompt, user_prompt),
    )


def build_ovon_action_cache_prompt_bundle(
    *,
    next_waypoint: str,
    subtask_instruction: str,
    progress_summary: Optional[str],
    waypoint_summary: Optional[str],
    detected_landmarks: Optional[str],
    obstacle_distances: Optional[Dict[str, str]],
    landmark_map_info: Optional[str],
    allowed_action_names: Optional[Sequence[str]],
    move_distance: float = 0.25,
    turn_angle: int = 30,
) -> ExplicitCachePromptBundle:
    """Build the OVON action cache prompt."""
    del waypoint_summary
    del move_distance
    del turn_angle
    progress_text = str(progress_summary or "").strip() or "Just started"
    detected_landmark_text = str(detected_landmarks or "").strip() or "none"
    obstacle_summary = _build_obstacle_perception_summary(obstacle_distances)
    landmark_summary = _build_landmark_perception_summary(
        detected_landmarks=detected_landmarks,
        landmark_map_info=landmark_map_info,
    )
    allowed_action_bullets = _build_allowed_action_bullets(allowed_action_names)
    system_prompt = _normalize_action_prompt_text(_render_action_system_prompt())
    user_prompt = _normalize_action_prompt_text(
        ACTION_CACHE_USER_PROMPT.format(
            subtask_destination=next_waypoint,
            subtask_instruction=subtask_instruction,
            progress_summary=progress_text,
            obstacle_perception_summary=obstacle_summary,
            landmark_perception_summary=landmark_summary,
            detected_landmarks=detected_landmark_text,
            allowed_action_bullets=allowed_action_bullets,
        )
    )
    return ExplicitCachePromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=compose_full_prompt(system_prompt, user_prompt),
    )


__all__ = [
    "build_ovon_action_cache_prompt_bundle",
    "build_ovon_initial_planner_cache_prompt_bundle",
    "build_ovon_verify_planner_cache_prompt_bundle",
]
