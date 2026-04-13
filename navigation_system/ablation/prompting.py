"""Prompt wrappers for ablation runs using the original templates only."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from navigation_system.ablation.config import AblationSpec, load_ablation_spec
from navigation_system.vlm.prompts import builders as standard_builders


def _active_spec(spec: Optional[AblationSpec] = None) -> AblationSpec:
    return spec or load_ablation_spec()


def _hidden_obstacle_distances() -> Dict[str, str]:
    return {
        "front": "Unknown",
        "left_30": "Unknown",
        "right_30": "Unknown",
        "left_90": "Unknown",
        "right_90": "Unknown",
    }


def get_initial_planning_prompt(
    instruction: str,
    action_space: str,
    *,
    spec: Optional[AblationSpec] = None,
) -> str:
    _active_spec(spec)
    return standard_builders.get_initial_planning_prompt(instruction, action_space)


def get_verification_replanning_prompt(
    instruction: str,
    subtask_destination: str,
    subtask_instruction: str,
    action_space: str,
    detected_landmarks: str = None,
    waypoint_summary: str = None,
    previous_subtask_landmark_summary: str = None,
    verify_replan_prompt_notice: str = None,
    direction_names: Sequence[str] = None,
    *,
    spec: Optional[AblationSpec] = None,
) -> str:
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
    prompt_verify_notice = (
        verify_replan_prompt_notice
        if resolved_spec.thinking_prompt.include_verify_notice
        else None
    )

    base_prompt = standard_builders.get_verification_replanning_prompt(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        action_space=action_space,
        detected_landmarks=prompt_detected_landmarks,
        waypoint_summary=prompt_waypoint_summary,
        previous_subtask_landmark_summary=prompt_previous_subtask_landmark_summary,
        verify_replan_prompt_notice=prompt_verify_notice,
        direction_names=list(direction_names or []),
    )
    return base_prompt


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
    *,
    spec: Optional[AblationSpec] = None,
) -> str:
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
    base_prompt = standard_builders.get_action_execution_prompt(
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
    return base_prompt


__all__ = [
    "get_action_execution_prompt",
    "get_initial_planning_prompt",
    "get_verification_replanning_prompt",
]
