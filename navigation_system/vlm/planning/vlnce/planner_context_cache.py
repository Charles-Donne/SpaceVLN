"""Planner runtime with DashScope explicit context cache."""

from typing import Any, Dict, List, Optional, Tuple

from navigation_system.vlm.planning.vlnce.planner import LLMPlanner
from navigation_system.vlm.prompts.common import ExplicitCachePromptBundle
from navigation_system.vlm.prompts.vlnce.cache_builders import (
    build_initial_planner_cache_prompt_bundle,
    build_verify_planner_cache_prompt_bundle,
)
from navigation_system.vlm.contracts.schema import get_next_waypoint
from navigation_system.vlm.api.qwen_context_cache_client import QwenContextCacheMixin


class ContextCachePlanner(QwenContextCacheMixin, LLMPlanner):
    """Planner variant that reuses a long stable system prompt via explicit cache."""

    def __init__(
        self,
        config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml",
        action_space: str = None,
    ):
        super().__init__(config_path=config_path, action_space=action_space)
        self._init_qwen_context_cache(config_path)
        print(f"  LLMPlanner(ContextCache): {self.config.model} | explicit-context-cache")

    def call_api(
        self,
        prompt_bundle: ExplicitCachePromptBundle,
        image_paths: List[Any],
        save_dir: str = None,
        no_compress_indices: set = None,
    ) -> Optional[Dict]:
        return self.call_api_with_explicit_context_cache(
            prompt_bundle=prompt_bundle,
            image_paths=image_paths,
            save_dir=save_dir,
            no_compress_indices=no_compress_indices,
        )

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

        prompt_bundle = build_initial_planner_cache_prompt_bundle(
            instruction=instruction,
            action_space=self.action_space,
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
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None, ""

        subtask_destination = get_next_waypoint(current_subtask) or "not set"
        subtask_instruction = str((current_subtask or {}).get("subtask_instruction", "") or "not set")
        prompt_bundle = build_verify_planner_cache_prompt_bundle(
            instruction=instruction,
            subtask_destination=subtask_destination,
            subtask_instruction=subtask_instruction,
            action_space=self.action_space,
            detected_landmarks=detected_landmarks,
            waypoint_summary=waypoint_summary,
            previous_subtask_landmark_summary=previous_subtask_landmark_summary,
            verify_replan_prompt_notice=verify_replan_prompt_notice,
            direction_names=direction_names,
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
            failure_label="LLM Verify",
        )
