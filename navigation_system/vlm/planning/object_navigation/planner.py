"""OVON-specific planner built on top of the shared API client/runtime."""

from typing import Any, Dict, List, Optional, Tuple

from navigation_system.vlm.prompts.object_navigation.builders import (
    build_ovon_initial_planner_prompt_bundle,
    build_ovon_verify_planner_prompt_bundle,
)
from navigation_system.vlm.contracts.schema import normalize_objectnav_subtask_payload
from navigation_system.vlm.planning.vlnce.planner import LLMPlanner


class OVONPlanner(LLMPlanner):
    """Planner variant with OVON/object-search-specific prompts."""

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

        prompt = build_ovon_initial_planner_prompt_bundle(
            instruction=instruction,
            action_space=self.action_space,
        )
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

        prompt = build_ovon_verify_planner_prompt_bundle(
            instruction=instruction,
            subtask_destination=str(current_subtask.get("next_waypoint", "") or ""),
            subtask_instruction=str(current_subtask.get("subtask_instruction", "") or ""),
            action_space=self.action_space,
            detected_landmarks=detected_landmarks,
            waypoint_summary=waypoint_summary,
            previous_subtask_landmark_summary=previous_subtask_landmark_summary,
            verify_replan_prompt_notice=verify_replan_prompt_notice,
            direction_names=direction_names,
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
