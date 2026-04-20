"""Prompt wrappers for ablation runs using static ablation template copies."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from navigation_system.ablation.config import AblationSpec, load_ablation_spec
from navigation_system.ablation.prompts.templates import load_ablation_template
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
    resolved_spec = _active_spec(spec)
    template = load_ablation_template(resolved_spec, "planning_initial.prompt.md")
    return standard_builders._normalize_anchor_notation_text(template.format(
        instruction=instruction,
        action_space=action_space,
        obs_blocked_m=standard_builders._fmt_threshold_m(standard_builders.OBS_BLOCKED_M),
        obs_risky_m=standard_builders._fmt_threshold_m(standard_builders.OBS_RISKY_M),
        obs_open_m=standard_builders._fmt_threshold_m(standard_builders.OBS_OPEN_M),
        arrival_near_m=standard_builders._fmt_threshold_m(standard_builders.ARRIVAL_NEAR_M),
    ))


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
    template = load_ablation_template(resolved_spec, "planning_verify.prompt.md")

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

    if not prompt_waypoint_summary:
        prompt_waypoint_summary = "Unavailable"
    if prompt_verify_notice:
        verify_replan_prompt_notice_block = (
            f"\n**Stuck Notice**: {prompt_verify_notice.strip()}"
        )
    else:
        verify_replan_prompt_notice_block = ""

    prompt_previous_subtask_landmark_summary = str(
        prompt_previous_subtask_landmark_summary or ""
    ).strip()
    previous_subtask_landmark_block = (
        f"- {prompt_previous_subtask_landmark_summary}"
        if prompt_previous_subtask_landmark_summary
        else ""
    )
    verify_view_count = standard_builders._get_verify_view_count(direction_names)

    return standard_builders._normalize_anchor_notation_text(template.format(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        action_space=action_space,
        detected_landmarks=prompt_detected_landmarks,
        waypoint_summary=prompt_waypoint_summary,
        previous_subtask_landmark_summary=prompt_previous_subtask_landmark_summary,
        previous_subtask_landmark_block=previous_subtask_landmark_block,
        verify_replan_prompt_notice=prompt_verify_notice,
        verify_replan_prompt_notice_block=verify_replan_prompt_notice_block,
        verify_view_count=verify_view_count,
        obs_blocked_m=standard_builders._fmt_threshold_m(standard_builders.OBS_BLOCKED_M),
        obs_risky_m=standard_builders._fmt_threshold_m(standard_builders.OBS_RISKY_M),
        obs_open_m=standard_builders._fmt_threshold_m(standard_builders.OBS_OPEN_M),
        arrival_near_m=standard_builders._fmt_threshold_m(standard_builders.ARRIVAL_NEAR_M),
    ))


def get_action_execution_prompt(
    next_waypoint: str,
    subtask_instruction: str,
    progress_summary: str = "",
    waypoint_summary: str = "",
    detected_landmarks: str = None,
    obstacle_distances=None,
    landmark_map_info: str = None,
    allowed_action_names=None,
    move_distance: float = 0.25,
    turn_angle: int = 30,
    *,
    spec: Optional[AblationSpec] = None,
) -> str:
    resolved_spec = _active_spec(spec)
    template = load_ablation_template(resolved_spec, "action_execution.prompt.md")

    prompt_progress_summary = (
        progress_summary
        if resolved_spec.action_prompt.include_progress_summary
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
    if not prompt_progress_summary:
        prompt_progress_summary = "Just started"

    return standard_builders._normalize_action_prompt_text(template.format(
        subtask_destination=next_waypoint,
        subtask_instruction=subtask_instruction,
        progress_summary=prompt_progress_summary,
        waypoint_summary=waypoint_summary,
        detected_landmarks=prompt_detected_landmarks or "none",
        obstacle_perception_summary=standard_builders._build_obstacle_perception_summary(
            prompt_obstacle_distances
        ),
        landmark_perception_summary=standard_builders._build_landmark_perception_summary(
            detected_landmarks=prompt_detected_landmarks,
            landmark_map_info=prompt_landmark_map_info,
        ),
        allowed_action_output=standard_builders._build_allowed_action_output(allowed_action_names),
        allowed_action_bullets=standard_builders._build_allowed_action_bullets(allowed_action_names),
        obs_blocked_m=standard_builders._fmt_threshold_m(standard_builders.OBS_BLOCKED_M),
        obs_risky_m=standard_builders._fmt_threshold_m(standard_builders.OBS_RISKY_M),
        obs_open_m=standard_builders._fmt_threshold_m(standard_builders.OBS_OPEN_M),
        open_autocomplete_m=standard_builders._fmt_threshold_m(
            standard_builders.ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M
        ),
        solid_autocomplete_m=standard_builders._fmt_threshold_m(
            standard_builders.ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M
        ),
        move_distance=move_distance,
        turn_angle=turn_angle,
    ))


__all__ = [
    "get_action_execution_prompt",
    "get_initial_planning_prompt",
    "get_verification_replanning_prompt",
]
