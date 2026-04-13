"""Ablation-specific controller that keeps the original system untouched."""

from __future__ import annotations

import cv2
from typing import Any, Dict, Optional

from navigation_system.ablation.config import load_ablation_spec
from navigation_system.ablation.map_visualizer import AblationMapVisualizer
from navigation_system.ablation.thinking_view_renderer import AblationThinkingViewRenderer
from navigation_system.config.core.params.api import ACTION_VIEW_MODEL_CONTENT_WIDTH
from navigation_system.controller.navigation_controller import VLMNavigationController
from navigation_system.render.image_resize import resize_image_to_width


class AblationVLMNavigationController(VLMNavigationController):
    """Drop-in controller that only changes final prompt/image exposure."""

    def __init__(
        self,
        config,
        config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml",
        model_stack_builder=None,
    ):
        self.ablation_spec = load_ablation_spec()
        super().__init__(
            config,
            config_path=config_path,
            model_stack_builder=model_stack_builder,
        )
        self.visualizer = AblationMapVisualizer.from_existing(
            self.visualizer,
            ablation_spec=self.ablation_spec,
        )
        self.thinking_view_renderer = AblationThinkingViewRenderer(
            ablation_spec=self.ablation_spec,
        )
        print(f"  AblationController: {self.ablation_spec.slug}")

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
    "AblationVLMNavigationController",
]
