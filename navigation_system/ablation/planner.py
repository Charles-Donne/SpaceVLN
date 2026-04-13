"""Ablation planner wrappers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from navigation_system.ablation.config import AblationSpec, load_ablation_spec
from navigation_system.ablation.prompting import (
    get_initial_planning_prompt,
    get_verification_replanning_prompt,
)
from navigation_system.vlm.contracts.schema import get_next_waypoint
from navigation_system.vlm.planning.planner import LLMPlanner


class AblationLLMPlanner(LLMPlanner):
    """Planner that keeps the original logic but swaps in ablation prompt wrappers."""

    def __init__(
        self,
        config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml",
        action_space: str = None,
        ablation_spec: Optional[AblationSpec] = None,
    ):
        self.ablation_spec = ablation_spec or load_ablation_spec()
        super().__init__(config_path=config_path, action_space=action_space)
        print(f"  AblationPlanner: {self.ablation_spec.slug}")

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
        del local_map_image
        del obstacle_distances
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None, ""

        prompt = get_initial_planning_prompt(
            instruction=instruction,
            action_space=self.action_space,
            spec=self.ablation_spec,
        )
        images = list(observation_images or [])
        images.append(global_map_image)
        no_compress = {len(observation_images)}

        return self._call_planner_with_retry(
            prompt=prompt,
            images=images,
            direction_names=direction_names,
            mode="initial",
            save_dir=save_dir,
            no_compress=no_compress,
            failure_label="LLM Planning",
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
        del local_map_image
        del obstacle_distances
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None, ""

        subtask_destination = get_next_waypoint(current_subtask) or "Unknown"
        subtask_instruction = str((current_subtask or {}).get("subtask_instruction", "") or "Unknown")
        prompt = get_verification_replanning_prompt(
            instruction=instruction,
            subtask_destination=subtask_destination,
            subtask_instruction=subtask_instruction,
            action_space=self.action_space,
            detected_landmarks=detected_landmarks,
            waypoint_summary=waypoint_summary,
            previous_subtask_landmark_summary=previous_subtask_landmark_summary,
            verify_replan_prompt_notice=verify_replan_prompt_notice,
            direction_names=direction_names,
            spec=self.ablation_spec,
        )

        images = list(observation_images or [])
        images.append(global_map_image)
        no_compress = {len(observation_images)}
        return self._call_planner_with_retry(
            prompt=prompt,
            images=images,
            direction_names=direction_names,
            mode="verify",
            save_dir=save_dir,
            no_compress=no_compress,
            failure_label="LLM Verify",
        )


__all__ = [
    "AblationLLMPlanner",
]
