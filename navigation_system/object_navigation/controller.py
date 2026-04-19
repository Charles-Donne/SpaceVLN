"""OVON-specific navigation controller wrapper."""

from __future__ import annotations

from typing import Optional

from navigation_system.controller.navigation_controller import VLMNavigationController
from navigation_system.object_navigation.runtime_factory import (
    build_ovon_navigation_model_stack,
)
from navigation_system.object_navigation.thresholds import (
    OVON_AUTOCOMPLETE_OPENING_M,
    OVON_AUTOCOMPLETE_SOLID_M,
    OVON_AUTOCOMPLETE_TOPK,
    OVON_SUCCESS_DISTANCE_M,
)
from navigation_system.vlm.interfaces import NavigationModelStackBuilder


class OVONObjectNavigationController(VLMNavigationController):
    """Separate controller entrypoint for OVON/object-navigation runs."""

    ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M = OVON_AUTOCOMPLETE_OPENING_M
    ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M = OVON_AUTOCOMPLETE_SOLID_M
    ACTION_SUBTASK_AUTOCOMPLETE_TOPK = OVON_AUTOCOMPLETE_TOPK
    FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK = 2
    FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M = OVON_SUCCESS_DISTANCE_M

    def __init__(
        self,
        config,
        config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml",
        model_stack_builder: Optional[NavigationModelStackBuilder] = None,
        envs=None,
    ):
        super().__init__(
            config,
            config_path=config_path,
            model_stack_builder=model_stack_builder or build_ovon_navigation_model_stack,
            envs=envs,
        )

    def _should_autostop_from_goal_distance(self) -> bool:
        latest_info = getattr(self, "latest_info", None)
        if not isinstance(latest_info, dict):
            return False
        try:
            distance_to_goal = float(latest_info.get("distance_to_goal", -1.0))
        except (TypeError, ValueError):
            return False
        if distance_to_goal < 0.0:
            return False
        return distance_to_goal <= float(self._get_success_distance_m())
