"""Explicit-cache NavGBench planning prompt builders."""

from navigation_system.vlm.prompts.common import (
    ExplicitCachePromptBundle,
    compose_full_prompt,
    join_prompt_blocks,
)
from navigation_system.vlm.prompts.vlnce.cache_builders import (
    build_initial_planner_cache_prompt_bundle,
    build_verify_planner_cache_prompt_bundle,
)
from navigation_system.vlm.prompts.vlnce.navgbench.common import (
    complex_instruction_policy_block,
    is_complex_navgbench_prompt_mode,
)


def _with_complex_policy(
    bundle: ExplicitCachePromptBundle,
    *,
    instruction_mode: str,
) -> ExplicitCachePromptBundle:
    if not is_complex_navgbench_prompt_mode(instruction_mode):
        return bundle

    system_prompt = join_prompt_blocks(
        [
            bundle.system_prompt,
            complex_instruction_policy_block(),
        ]
    )
    return ExplicitCachePromptBundle(
        system_prompt=system_prompt,
        user_prompt=bundle.user_prompt,
        full_prompt=compose_full_prompt(system_prompt, bundle.user_prompt),
    )


def build_navgbench_initial_planner_cache_prompt_bundle(
    *,
    instruction: str,
    action_space: str,
    instruction_mode: str = "complex",
) -> ExplicitCachePromptBundle:
    return _with_complex_policy(
        build_initial_planner_cache_prompt_bundle(
            instruction=instruction,
            action_space=action_space,
        ),
        instruction_mode=instruction_mode,
    )


def build_navgbench_verify_planner_cache_prompt_bundle(
    *,
    instruction: str,
    subtask_destination: str,
    subtask_instruction: str,
    action_space: str,
    detected_landmarks,
    waypoint_summary,
    previous_subtask_landmark_summary,
    verify_replan_prompt_notice,
    direction_names,
    instruction_mode: str = "complex",
) -> ExplicitCachePromptBundle:
    return _with_complex_policy(
        build_verify_planner_cache_prompt_bundle(
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


__all__ = [
    "build_navgbench_initial_planner_cache_prompt_bundle",
    "build_navgbench_verify_planner_cache_prompt_bundle",
]
