"""OVON-specific planner built on top of the shared API client/runtime."""

from typing import Any, Dict, List, Optional, Tuple

from navigation_system.vlm.prompts.object_navigation.builders import (
    get_ovon_initial_planning_prompt,
    get_ovon_verification_replanning_prompt,
)
from navigation_system.vlm.planning.planner import LLMPlanner


class OVONPlanner(LLMPlanner):
    """Planner variant with OVON/object-search-specific prompts."""

    def generate_initial_subtask(
        self,
        instruction: str,
        observation_images: List[Any],
        direction_names: List[str],
        global_map_image: str,
        local_map_image: str = None,
        obstacle_distances: Dict[str, str] = None,
        save_dir: str = None,
    ) -> Tuple[Optional[Dict], str]:
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None, ""

        prompt = get_ovon_initial_planning_prompt(instruction, self.action_space)
        images = observation_images.copy()
        images.append(global_map_image)
        no_compress = {len(observation_images)}

        return self._call_planner_with_retry(
            prompt=prompt,
            images=images,
            direction_names=direction_names,
            mode="initial",
            save_dir=save_dir,
            no_compress=no_compress,
            failure_label="OVON LLM Planning",
        )

    def verify_and_replan(
        self,
        instruction: str,
        current_subtask: Dict,
        observation_images: List[Any],
        direction_names: List[str],
        global_map_image: str,
        local_map_image: str = None,
        detected_landmarks: List[str] = None,
        waypoint_summary: str = None,
        previous_subtask_landmark_summary: str = None,
        obstacle_distances: Dict[str, str] = None,
        verify_replan_prompt_notice: str = None,
        save_dir: str = None,
    ) -> Tuple[Optional[Dict], str]:
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None, ""

        prompt = get_ovon_verification_replanning_prompt(
            instruction=instruction,
            subtask_destination=str(current_subtask.get("next_waypoint", "") or ""),
            subtask_instruction=str(current_subtask.get("subtask_instruction", "") or ""),
            action_space=self.action_space,
            waypoint_summary=waypoint_summary,
            previous_subtask_landmark_summary=previous_subtask_landmark_summary,
            verify_replan_prompt_notice=verify_replan_prompt_notice,
        )

        images = observation_images.copy()
        images.append(global_map_image)
        no_compress = {len(observation_images)}

        return self._call_planner_with_retry(
            prompt=prompt,
            images=images,
            direction_names=direction_names,
            mode="verify",
            save_dir=save_dir,
            no_compress=no_compress,
            failure_label="OVON Verify/Replan",
        )
