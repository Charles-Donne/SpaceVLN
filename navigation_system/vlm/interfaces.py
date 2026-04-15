"""Typed interfaces for pluggable planner/action runtime adapters."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple


class PlannerLike(Protocol):
    def set_request_artifact_saving(self, enabled: bool) -> None:
        ...

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
        ...

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
        ...


class ActionExecutorLike(Protocol):
    def set_request_artifact_saving(self, enabled: bool) -> None:
        ...

    def decide_action(
        self,
        next_waypoint: str,
        subtask_instruction: str,
        first_person_image: Any,
        action_mapping: Dict[str, int],
        progress_summary: str = "",
        waypoint_summary: str = "",
        detection_image: Any = None,
        detected_landmarks: str = None,
        previous_action_reason: str = "",
        controller_action_notice: str = "",
        obstacle_distances: Dict[str, str] = None,
        landmark_map_info: str = None,
        allowed_action_names: Optional[Sequence[str]] = None,
        save_dir: str = None,
    ) -> Tuple[Optional[int], Optional[str], Optional[Dict], int, float, str]:
        ...


@dataclass(frozen=True)
class NavigationModelStack:
    planner: Optional[PlannerLike]
    action_executor: Optional[ActionExecutorLike]


class NavigationModelStackBuilder(Protocol):
    def __call__(
        self,
        *,
        config_path: str,
        action_space: str,
        turn_angle: float,
        move_distance: float,
        save_request_artifacts: bool,
    ) -> NavigationModelStack:
        ...
