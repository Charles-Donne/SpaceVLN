"""OVON planner runtime with explicit context cache."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from navigation_system.vlm.prompts.object_navigation.cache_builders import (
    build_ovon_initial_planner_cache_prompt_bundle,
    build_ovon_verify_planner_cache_prompt_bundle,
)
from navigation_system.vlm.api.qwen_context_cache_client import QwenContextCacheMixin
from navigation_system.vlm.contracts.schema import (
    get_next_waypoint,
    normalize_objectnav_subtask_payload,
)
from navigation_system.vlm.planning.vlnce.planner import LLMPlanner
from navigation_system.vlm.prompts.common import ExplicitCachePromptBundle


class OVONContextCachePlanner(QwenContextCacheMixin, LLMPlanner):
    """OVON high-level planner using the same explicit-cache protocol as CE."""

    def _normalize_response_payload(
        self,
        payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return normalize_objectnav_subtask_payload(payload)

    @classmethod
    def _direction_is_available(
        cls,
        chosen_direction: Optional[str],
        direction_names: Optional[List[str]],
    ) -> bool:
        chosen_text = str(chosen_direction or "").strip()
        if not chosen_text:
            return False
        if cls._extract_image_index(chosen_text) is None:
            return False
        return super()._direction_is_available(chosen_direction, direction_names)

    def __init__(
        self,
        config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml",
        action_space: str = None,
    ):
        super().__init__(config_path=config_path, action_space=action_space)
        self._init_qwen_context_cache(config_path)
        print(f"  OVONPlanner(ContextCache): {self.config.model} | explicit-context-cache")

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
        del local_map_image
        del obstacle_distances
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None, ""

        prompt_bundle = build_ovon_initial_planner_cache_prompt_bundle(
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
        del local_map_image
        del obstacle_distances
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None, ""

        prompt_bundle = build_ovon_verify_planner_cache_prompt_bundle(
            instruction=instruction,
            subtask_destination=get_next_waypoint(current_subtask) or "not set",
            subtask_instruction=str(
                (current_subtask or {}).get("subtask_instruction", "") or "not set"
            ),
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
            failure_label="OVON Verify/Replan",
        )
