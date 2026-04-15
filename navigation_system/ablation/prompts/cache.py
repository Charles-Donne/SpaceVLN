"""Explicit-cache prompt builders using static ablation template copies."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from navigation_system.ablation.config import AblationSpec, load_ablation_spec
from navigation_system.ablation.prompts.standard import _hidden_obstacle_distances
from navigation_system.ablation.prompts.templates import load_ablation_template
from navigation_system.vlm.prompts import cache_builders as standard_cache_builders
from navigation_system.vlm.prompts.common import (
    ExplicitCachePromptBundle,
    compose_full_prompt,
)


def _active_spec(spec: Optional[AblationSpec] = None) -> AblationSpec:
    return spec or load_ablation_spec()


def build_initial_planner_cache_prompt_bundle(
    *,
    instruction: str,
    action_space: str,
    spec: Optional[AblationSpec] = None,
) -> ExplicitCachePromptBundle:
    del action_space

    resolved_spec = _active_spec(spec)
    system_template = load_ablation_template(
        resolved_spec,
        "cache/planning_initial.system.prompt.md",
    )
    user_template = load_ablation_template(
        resolved_spec,
        "cache/planning_initial.user.prompt.md",
    )
    system_prompt = system_template.format(
        obs_blocked_m=standard_cache_builders._fmt_threshold_m(standard_cache_builders.OBS_BLOCKED_M),
        obs_risky_m=standard_cache_builders._fmt_threshold_m(standard_cache_builders.OBS_RISKY_M),
        obs_open_m=standard_cache_builders._fmt_threshold_m(standard_cache_builders.OBS_OPEN_M),
        arrival_near_m=standard_cache_builders._fmt_threshold_m(standard_cache_builders.ARRIVAL_NEAR_M),
    )
    user_prompt = user_template.format(
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
    spec: Optional[AblationSpec] = None,
) -> ExplicitCachePromptBundle:
    resolved_spec = _active_spec(spec)
    system_template = load_ablation_template(
        resolved_spec,
        "cache/planning_verify.system.prompt.md",
    )
    user_template = load_ablation_template(
        resolved_spec,
        "cache/planning_verify.user.prompt.md",
    )

    prompt_detected_landmarks = (
        detected_landmarks
        if resolved_spec.thinking_prompt.include_detected_landmarks
        else None
    )
    prompt_waypoint_summary = (
        waypoint_summary
        if resolved_spec.thinking_prompt.include_waypoint_summary
        else None
    )
    prompt_previous_subtask_landmark_summary = (
        previous_subtask_landmark_summary
        if resolved_spec.thinking_prompt.include_previous_subtask_landmark_summary
        else None
    )
    prompt_verify_replan_prompt_notice = (
        verify_replan_prompt_notice
        if resolved_spec.thinking_prompt.include_verify_notice
        else None
    )

    waypoint_text = str(prompt_waypoint_summary or "Unavailable").strip() or "Unavailable"
    previous_subtask_text = str(prompt_previous_subtask_landmark_summary or "").strip()
    previous_subtask_landmark_block = (
        f"- {previous_subtask_text}"
        if previous_subtask_text
        else ""
    )
    notice_block_for_user = ""
    if str(prompt_verify_replan_prompt_notice or "").strip():
        notice_text = str(prompt_verify_replan_prompt_notice).strip()
        notice_block_for_user = f"**Stuck Notice**: {notice_text}"

    system_prompt = system_template.format(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        action_space=action_space,
        detected_landmarks=prompt_detected_landmarks,
        waypoint_summary=waypoint_text,
        previous_subtask_landmark_summary=previous_subtask_text,
        verify_replan_prompt_notice=prompt_verify_replan_prompt_notice,
        obs_blocked_m=standard_cache_builders._fmt_threshold_m(standard_cache_builders.OBS_BLOCKED_M),
        obs_risky_m=standard_cache_builders._fmt_threshold_m(standard_cache_builders.OBS_RISKY_M),
        obs_open_m=standard_cache_builders._fmt_threshold_m(standard_cache_builders.OBS_OPEN_M),
        arrival_near_m=standard_cache_builders._fmt_threshold_m(standard_cache_builders.ARRIVAL_NEAR_M),
    )
    user_prompt = user_template.format(
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
    controller_action_notice: Optional[str],
    obstacle_distances: Optional[Dict[str, str]],
    landmark_map_info: Optional[str],
    allowed_action_names: Optional[Sequence[str]],
    move_distance: float = 0.25,
    turn_angle: int = 30,
    spec: Optional[AblationSpec] = None,
) -> ExplicitCachePromptBundle:
    del waypoint_summary
    del move_distance
    del turn_angle

    resolved_spec = _active_spec(spec)
    system_template = load_ablation_template(
        resolved_spec,
        "cache/action.system.prompt.md",
    )
    user_template = load_ablation_template(
        resolved_spec,
        "cache/action.user.prompt.md",
    )

    prompt_progress_summary = (
        progress_summary
        if resolved_spec.action_prompt.include_progress_summary
        else ""
    )
    prompt_previous_action_reason = (
        previous_action_reason
        if resolved_spec.action_prompt.include_previous_action_reason
        else ""
    )
    prompt_detected_landmarks = (
        detected_landmarks
        if resolved_spec.action_prompt.include_detected_landmarks
        else None
    )
    prompt_obstacle_distances = (
        obstacle_distances
        if resolved_spec.action_prompt.include_obstacle_summary
        else _hidden_obstacle_distances()
    )
    prompt_landmark_map_info = (
        landmark_map_info
        if resolved_spec.action_prompt.include_landmark_map_info
        else None
    )

    progress_text = str(prompt_progress_summary or "").strip() or "Just started"
    previous_action_text = str(prompt_previous_action_reason or "").strip() or "N/A (first step)"
    controller_notice_text = str(controller_action_notice or "").strip() or "None"
    detected_landmark_text = str(prompt_detected_landmarks or "").strip() or "none"
    obstacle_summary = standard_cache_builders._build_obstacle_perception_summary(
        prompt_obstacle_distances
    )
    landmark_summary = standard_cache_builders._build_landmark_perception_summary(
        detected_landmarks=prompt_detected_landmarks,
        landmark_map_info=prompt_landmark_map_info,
    )
    allowed_action_bullets = standard_cache_builders._build_allowed_action_bullets(
        allowed_action_names
    )

    system_prompt = system_template.format(
        obs_blocked_m=standard_cache_builders._fmt_threshold_m(standard_cache_builders.OBS_BLOCKED_M),
        obs_risky_m=standard_cache_builders._fmt_threshold_m(standard_cache_builders.OBS_RISKY_M),
        obs_open_m=standard_cache_builders._fmt_threshold_m(standard_cache_builders.OBS_OPEN_M),
        solid_autocomplete_m=standard_cache_builders._fmt_threshold_m(
            standard_cache_builders.ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M
        ),
        open_autocomplete_m=standard_cache_builders._fmt_threshold_m(
            standard_cache_builders.ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M
        ),
    )
    user_prompt = user_template.format(
        subtask_destination=next_waypoint,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_text,
        previous_action_reason=previous_action_text,
        controller_action_notice=controller_notice_text,
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
    "build_action_cache_prompt_bundle",
    "build_initial_planner_cache_prompt_bundle",
    "build_verify_planner_cache_prompt_bundle",
]
