"""Ablation-specific controller that keeps the original system untouched."""

from __future__ import annotations

import cv2
from typing import Any, Dict, Optional

from navigation_system.ablation.config import load_ablation_spec
from navigation_system.ablation.render.map_visualizer import AblationMapVisualizer
from navigation_system.ablation.render.thinking_view_renderer import (
    AblationThinkingViewRenderer,
)
from navigation_system.config.core.params.api import ACTION_VIEW_MODEL_CONTENT_WIDTH
from navigation_system.controller.agent.controller import NavigationAgentController
from navigation_system.render.image_resize import resize_image_to_width


class AblationNavigationAgentController(NavigationAgentController):
    """Drop-in controller that only changes final prompt/image exposure."""

    def _uses_landmark_perception(self) -> bool:
        spec = getattr(self, "ablation_spec", None)
        if spec is None:
            return True
        return any(
            (
                bool(spec.thinking_prompt.include_detected_landmarks),
                bool(spec.thinking_prompt.include_previous_subtask_landmark_summary),
                bool(spec.thinking_image.include_detection_boxes),
                bool(spec.thinking_image.include_landmark_strip),
                bool(spec.action_prompt.include_detected_landmarks),
                bool(spec.action_prompt.include_landmark_map_info),
                bool(spec.action_image.use_detection_overlay),
            )
        )

    def _should_initialize_segment_module(self) -> bool:
        return self._uses_landmark_perception() and super()._should_initialize_segment_module()

    def __init__(
        self,
        config,
        config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml",
        model_stack_builder=None,
        envs=None,
    ):
        self.ablation_spec = load_ablation_spec()
        super().__init__(
            config,
            config_path=config_path,
            model_stack_builder=model_stack_builder,
            envs=envs,
        )
        self.visualizer = AblationMapVisualizer.from_existing(
            self.visualizer,
            ablation_spec=self.ablation_spec,
        )
        self.thinking_view_renderer = AblationThinkingViewRenderer(
            ablation_spec=self.ablation_spec,
        )
        print(f"  AblationController: {self.ablation_spec.slug}")
        if not self._uses_landmark_perception():
            print("  AblationController: landmark perception disabled; GroundedSAM is not loaded")

    def _capture_lookaround_scan(self, *args, **kwargs):
        if not self._uses_landmark_perception():
            kwargs["enable_landmark_detection"] = False
            kwargs["prepare_thinking_detection"] = False
        return super()._capture_lookaround_scan(*args, **kwargs)

    def _detect_landmarks_for_visualization(self, rgb, landmark_queries=None):
        if not self._uses_landmark_perception():
            return None, [], None
        return super()._detect_landmarks_for_visualization(rgb, landmark_queries)

    def _run_pre_action_detection_snapshot(self, action_phase: str) -> bool:
        if not self._uses_landmark_perception():
            return False
        return super()._run_pre_action_detection_snapshot(action_phase)

    def _refresh_post_action_landmark_detection_state(self, action_phase: str) -> bool:
        if not self._uses_landmark_perception():
            return False
        return super()._refresh_post_action_landmark_detection_state(action_phase)

    def _check_post_action_landmark_autocomplete(self, action_phase: str):
        if not self._uses_landmark_perception():
            return None
        return super()._check_post_action_landmark_autocomplete(action_phase)

    def _build_action_detection_image_input(self, last_step: int) -> Optional[Dict[str, Any]]:
        if self.ablation_spec.action_image.use_detection_overlay:
            return super()._build_action_detection_image_input(last_step)

        if self.latest_obs is not None and "rgb" in self.latest_obs:
            rgb_bgr = cv2.cvtColor(self.latest_obs["rgb"], cv2.COLOR_RGB2BGR)
            obstacle_distances = getattr(
                self,
                "latest_obstacle_distances",
                {"front": "Unknown", "left_30": "Unknown", "right_30": "Unknown"},
            )
            rgb_bgr = self.visualizer.draw_distance_on_action_view(rgb_bgr, obstacle_distances)
            rgb_bgr = resize_image_to_width(rgb_bgr, ACTION_VIEW_MODEL_CONTENT_WIDTH)
            return {
                "image_array": rgb_bgr,
                "color_space": "bgr",
                "artifact_name": "action_view_rgb.jpg",
                "name": f"action_view_rgb_step_{last_step:04d}",
            }

        print(f"  [WARN] Action RGB input not available for step {last_step}")
        return None


__all__ = [
    "AblationNavigationAgentController",
]
