"""NavGBench-specific system/user planning prompt builders."""

from navigation_system.vlm.prompts.common import (
    PromptBundle,
    compose_full_prompt,
    join_prompt_blocks,
)
from navigation_system.vlm.prompts.vlnce.builders import (
    build_initial_planner_prompt_bundle,
    build_verify_planner_prompt_bundle,
)
from navigation_system.vlm.prompts.vlnce.navgbench.common import (
    complex_instruction_policy_block,
    is_complex_navgbench_prompt_mode,
)


def _with_complex_policy(
    bundle: PromptBundle,
    *,
    instruction_mode: str,
) -> PromptBundle:
    if not is_complex_navgbench_prompt_mode(instruction_mode):
        return bundle

    system_prompt = join_prompt_blocks(
        [
            bundle.system_prompt,
            complex_instruction_policy_block(),
        ]
    )
    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=bundle.user_prompt,
        full_prompt=compose_full_prompt(system_prompt, bundle.user_prompt),
    )


def build_navgbench_initial_planner_prompt_bundle(
    *,
    instruction: str,
    action_space: str,
    instruction_mode: str = "complex",
) -> PromptBundle:
    return _with_complex_policy(
        build_initial_planner_prompt_bundle(
            instruction=instruction,
            action_space=action_space,
        ),
        instruction_mode=instruction_mode,
    )


def build_navgbench_verify_planner_prompt_bundle(
    *,
    instruction: str,
    subtask_destination: str,
    subtask_instruction: str,
    action_space: str,
    detected_landmarks=None,
    waypoint_summary=None,
    previous_subtask_landmark_summary=None,
    verify_replan_prompt_notice=None,
    direction_names=None,
    instruction_mode: str = "complex",
) -> PromptBundle:
    return _with_complex_policy(
        build_verify_planner_prompt_bundle(
            instruction=instruction,
            subtask_destination=subtask_destination,
            subtask_instruction=subtask_instruction,
            action_space=action_space,
            detected_landmarks=detected_landmarks,
            waypoint_summary=waypoint_summary,
            previous_subtask_landmark_summary=previous_subtask_landmark_summary,
            verify_replan_prompt_notice=verify_replan_prompt_notice,
            direction_names=direction_names,
        ),
        instruction_mode=instruction_mode,
    )


def get_navgbench_initial_planning_prompt(
    instruction: str,
    action_space: str,
    *,
    instruction_mode: str = "complex",
) -> str:
    return build_navgbench_initial_planner_prompt_bundle(
        instruction=instruction,
        action_space=action_space,
        instruction_mode=instruction_mode,
    ).full_prompt


def get_navgbench_verification_replanning_prompt(
    instruction: str,
    subtask_destination: str,
    subtask_instruction: str,
    action_space: str,
    detected_landmarks: str = None,
    waypoint_summary: str = None,
    previous_subtask_landmark_summary: str = None,
    verify_replan_prompt_notice: str = None,
    direction_names: list = None,
    *,
    instruction_mode: str = "complex",
) -> str:
    return build_navgbench_verify_planner_prompt_bundle(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        action_space=action_space,
        detected_landmarks=detected_landmarks,
        waypoint_summary=waypoint_summary,
        previous_subtask_landmark_summary=previous_subtask_landmark_summary,
        verify_replan_prompt_notice=verify_replan_prompt_notice,
        direction_names=direction_names,
        instruction_mode=instruction_mode,
    ).full_prompt


__all__ = [
    "build_navgbench_initial_planner_prompt_bundle",
    "build_navgbench_verify_planner_prompt_bundle",
    "get_navgbench_initial_planning_prompt",
    "get_navgbench_verification_replanning_prompt",
]
