"""Explicit-cache planner wrapper for ablation runs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from navigation_system.ablation.prompts.builders import (
    build_initial_planner_prompt_bundle,
    build_verify_planner_prompt_bundle,
)
from navigation_system.ablation.config import AblationSpec, load_ablation_spec
from navigation_system.vlm.api.qwen_context_cache_client import QwenContextCacheMixin
from navigation_system.vlm.contracts.schema import get_next_waypoint
from navigation_system.vlm.planning.vlnce.planner import LLMPlanner
from navigation_system.vlm.prompts.common import PromptBundle


class AblationContextCachePlanner(QwenContextCacheMixin, LLMPlanner):
    """Ablation planner variant that preserves the explicit-cache runtime."""

    def __init__(
        self,
        config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml",
        action_space: str = None,
        ablation_spec: Optional[AblationSpec] = None,
    ):
        self.ablation_spec = ablation_spec or load_ablation_spec()
        super().__init__(config_path=config_path, action_space=action_space)
        self._init_qwen_context_cache(config_path)
        print(
            f"  AblationPlanner(ContextCache): {self.ablation_spec.slug} | explicit-context-cache"
        )

    def call_api(
        self,
        prompt_bundle: PromptBundle,
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
        del local_map_image
        del obstacle_distances
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None, ""

        prompt_bundle = build_initial_planner_prompt_bundle(
            instruction=instruction,
            action_space=self.action_space,
            spec=self.ablation_spec,
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
        del local_map_image
        del obstacle_distances
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None, ""

        subtask_destination = get_next_waypoint(current_subtask) or "Unknown"
        subtask_instruction = str((current_subtask or {}).get("subtask_instruction", "") or "Unknown")
        prompt_bundle = build_verify_planner_prompt_bundle(
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
        return self._call_planner_with_retry(
            prompt=prompt_bundle,
            images=images,
            direction_names=direction_names,
            mode="verify",
            save_dir=save_dir,
            no_compress=None,
            failure_label="LLM Verify",
        )


__all__ = [
    "AblationContextCachePlanner",
]
