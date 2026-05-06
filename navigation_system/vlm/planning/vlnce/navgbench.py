"""NavGBench planner variants for the VLNCE task family."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from navigation_system.vlm.contracts.schema import get_next_waypoint
from navigation_system.vlm.planning.vlnce.planner import LLMPlanner
from navigation_system.vlm.planning.vlnce.planner_context_cache import ContextCachePlanner
from navigation_system.vlm.prompts.vlnce.navgbench.builders import (
    build_navgbench_initial_planner_prompt_bundle,
    build_navgbench_verify_planner_prompt_bundle,
)


class NavGBenchPlanner(LLMPlanner):
    """Standard planner with NavGBench-specific planning prompt overlays."""

    def __init__(
        self,
        config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml",
        action_space: str = None,
        instruction_mode: str = "complex",
    ):
        self.instruction_mode = str(instruction_mode or "complex").strip() or "complex"
        super().__init__(config_path=config_path, action_space=action_space)

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

        prompt = build_navgbench_initial_planner_prompt_bundle(
            instruction=instruction,
            action_space=self.action_space,
            instruction_mode=self.instruction_mode,
        )
        images = list(observation_images or [])
        images.append(global_map_image)
        no_compress = {len(observation_images or [])}
        return self._call_planner_with_retry(
            prompt=prompt,
            images=images,
            direction_names=direction_names,
            mode="initial",
            save_dir=save_dir,
            no_compress=no_compress,
            failure_label="NavGBench LLM Planning",
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

        subtask_destination = get_next_waypoint(current_subtask) or "not set"
        subtask_instruction = str((current_subtask or {}).get("subtask_instruction", "") or "not set")
        prompt = build_navgbench_verify_planner_prompt_bundle(
            instruction=instruction,
            subtask_destination=subtask_destination,
            subtask_instruction=subtask_instruction,
            action_space=self.action_space,
            detected_landmarks=None,
            waypoint_summary=waypoint_summary,
            previous_subtask_landmark_summary=previous_subtask_landmark_summary,
            verify_replan_prompt_notice=verify_replan_prompt_notice,
            direction_names=direction_names,
            instruction_mode=self.instruction_mode,
        )
        images = list(observation_images or [])
        images.append(global_map_image)
        no_compress = {len(observation_images or [])}
        return self._call_planner_with_retry(
            prompt=prompt,
            images=images,
            direction_names=direction_names,
            mode="verify",
            save_dir=save_dir,
            no_compress=no_compress,
            failure_label="NavGBench LLM Verify",
        )


class NavGBenchContextCachePlanner(ContextCachePlanner):
    """Explicit-cache planner with NavGBench-specific planning prompt overlays."""

    def __init__(
        self,
        config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml",
        action_space: str = None,
        instruction_mode: str = "complex",
    ):
        self.instruction_mode = str(instruction_mode or "complex").strip() or "complex"
        super().__init__(config_path=config_path, action_space=action_space)

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

        prompt_bundle = build_navgbench_initial_planner_prompt_bundle(
            instruction=instruction,
            action_space=self.action_space,
            instruction_mode=self.instruction_mode,
        )
        images = list(observation_images or [])
        images.append(global_map_image)
        return self._call_planner_with_retry(
            prompt=prompt_bundle,
            images=images,
            direction_names=direction_names,
            mode="initial",
            save_dir=save_dir,
            no_compress=None,
            failure_label="NavGBench LLM Planning",
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

        subtask_destination = get_next_waypoint(current_subtask) or "not set"
        subtask_instruction = str((current_subtask or {}).get("subtask_instruction", "") or "not set")
        prompt_bundle = build_navgbench_verify_planner_prompt_bundle(
            instruction=instruction,
            subtask_destination=subtask_destination,
            subtask_instruction=subtask_instruction,
            action_space=self.action_space,
            detected_landmarks=detected_landmarks,
            waypoint_summary=waypoint_summary,
            previous_subtask_landmark_summary=previous_subtask_landmark_summary,
            verify_replan_prompt_notice=verify_replan_prompt_notice,
            direction_names=direction_names,
            instruction_mode=self.instruction_mode,
        )
        images = list(observation_images or [])
        images.append(global_map_image)
        return self._call_planner_with_retry(
            prompt=prompt_bundle,
            images=images,
            direction_names=direction_names,
            mode="verify",
            save_dir=save_dir,
            no_compress=None,
            failure_label="NavGBench LLM Verify",
        )


__all__ = [
    "NavGBenchContextCachePlanner",
    "NavGBenchPlanner",
]
