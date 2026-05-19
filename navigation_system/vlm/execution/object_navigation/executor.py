"""OVON-specific low-level action executor."""

from typing import Any, Dict, Optional, Sequence, Tuple

from navigation_system.vlm.prompts.object_navigation.builders import (
    build_ovon_executor_prompt_bundle,
)
from navigation_system.vlm.execution.vlnce.executor import Executor


class OVONExecutor(Executor):
    """Executor variant with OVON-specific executor prompt wording."""

    def decide_action(
        self,
        next_waypoint: str,
        subtask_instruction: str,
        first_person_image: Any,
        action_mapping: Dict[str, int],
        progress_summary: str = "",
        waypoint_summary: str = "",
        subtask_landmark: str = "",
        detection_image: Any = None,
        detected_landmarks: str = None,
        obstacle_distances: Dict[str, str] = None,
        landmark_map_info: str = None,
        allowed_action_names: Optional[Sequence[str]] = None,
        save_dir: str = None,
    ) -> Tuple[Optional[int], Optional[str], Optional[Dict], int, float, str]:
        if not obstacle_distances:
            obstacle_distances = {
                "front": "Unknown",
                "left_30": "Unknown",
                "right_30": "Unknown",
            }

        prompt = build_ovon_executor_prompt_bundle(
            next_waypoint=next_waypoint,
            subtask_instruction=subtask_instruction,
            subtask_landmark=subtask_landmark,
            progress_summary=progress_summary,
            waypoint_summary=waypoint_summary,
            detected_landmarks=detected_landmarks,
            obstacle_distances=obstacle_distances,
            landmark_map_info=landmark_map_info,
            allowed_action_names=allowed_action_names,
        )

        return self._decide_action_from_prompt(
            prompt=prompt,
            first_person_image=first_person_image,
            detection_image=detection_image,
            action_mapping=action_mapping,
            allowed_action_names=allowed_action_names,
            save_dir=save_dir,
        )
