"""NavGBench-specific planning prompt builders.

The underlying VLNCE prompt remains the shared base; NavGBench adds an overlay
only for complex/grounded route instructions so R2R semantics stay untouched.
"""

from navigation_system.vlm.prompts.vlnce.builders import (
    get_initial_planning_prompt,
    get_verification_replanning_prompt,
)
from navigation_system.vlm.prompts.vlnce.navgbench.common import (
    inject_complex_instruction_policy,
)


def get_navgbench_initial_planning_prompt(
    instruction: str,
    action_space: str,
    *,
    instruction_mode: str = "complex",
) -> str:
    prompt = get_initial_planning_prompt(
        instruction=instruction,
        action_space=action_space,
    )
    return inject_complex_instruction_policy(
        prompt,
        instruction_mode=instruction_mode,
    )


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
    prompt = get_verification_replanning_prompt(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        action_space=action_space,
        detected_landmarks=detected_landmarks,
        waypoint_summary=waypoint_summary,
        previous_subtask_landmark_summary=previous_subtask_landmark_summary,
        verify_replan_prompt_notice=verify_replan_prompt_notice,
        direction_names=direction_names,
    )
    return inject_complex_instruction_policy(
        prompt,
        instruction_mode=instruction_mode,
    )


__all__ = [
    "get_navgbench_initial_planning_prompt",
    "get_navgbench_verification_replanning_prompt",
]
