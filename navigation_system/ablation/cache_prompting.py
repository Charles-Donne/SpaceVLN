"""Explicit-cache prompt bundles for ablation runs using the original templates only."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from navigation_system.ablation.config import AblationSpec, load_ablation_spec
from navigation_system.ablation.prompting import _hidden_obstacle_distances
from navigation_system.vlm.prompts import cache_builders as standard_cache_builders
from navigation_system.vlm.prompts.common import ExplicitCachePromptBundle


def _active_spec(spec: Optional[AblationSpec] = None) -> AblationSpec:
    return spec or load_ablation_spec()


def build_initial_planner_cache_prompt_bundle(
    *,
    instruction: str,
    action_space: str,
    spec: Optional[AblationSpec] = None,
) -> ExplicitCachePromptBundle:
    _active_spec(spec)
    bundle = standard_cache_builders.build_initial_planner_cache_prompt_bundle(
        instruction=instruction,
        action_space=action_space,
    )
    return bundle


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
    bundle = standard_cache_builders.build_verify_planner_cache_prompt_bundle(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        action_space=action_space,
        detected_landmarks=prompt_detected_landmarks,
        waypoint_summary=prompt_waypoint_summary,
        previous_subtask_landmark_summary=prompt_previous_subtask_landmark_summary,
        verify_replan_prompt_notice=prompt_verify_replan_prompt_notice,
        direction_names=direction_names,
    )
    return bundle


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
    spec: Optional[AblationSpec] = None,
) -> ExplicitCachePromptBundle:
    resolved_spec = _active_spec(spec)
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
    bundle = standard_cache_builders.build_action_cache_prompt_bundle(
        next_waypoint=next_waypoint,
        subtask_instruction=subtask_instruction,
        progress_summary=prompt_progress_summary,
        waypoint_summary=waypoint_summary,
        detected_landmarks=prompt_detected_landmarks,
        previous_action_reason=prompt_previous_action_reason,
        obstacle_distances=prompt_obstacle_distances,
        landmark_map_info=prompt_landmark_map_info,
        allowed_action_names=allowed_action_names,
        move_distance=move_distance,
        turn_angle=turn_angle,
    )
    return bundle


__all__ = [
    "build_action_cache_prompt_bundle",
    "build_initial_planner_cache_prompt_bundle",
    "build_verify_planner_cache_prompt_bundle",
]
