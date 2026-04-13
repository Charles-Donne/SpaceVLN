"""Ablation-aware map visualizer wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from navigation_system.ablation.config import AblationSpec, load_ablation_spec
from navigation_system.render.map.map_visualizer import MapVisualizer


class AblationMapVisualizer(MapVisualizer):
    """Wrap the original visualizer and only change the final rendered outputs."""

    def __init__(
        self,
        results_dir: str,
        resolution: int = 5,
        map_shape: Tuple[int, int] = (480, 480),
        enable_global_map_crop: bool = False,
        enable_adaptive_zoom: bool = False,
        debug_save_renderings: bool = True,
        save_step_map_artifacts: bool = False,
        ablation_spec: Optional[AblationSpec] = None,
    ):
        self.ablation_spec = ablation_spec or load_ablation_spec()
        self._ablation_phase = ""
        super().__init__(
            results_dir=results_dir,
            resolution=resolution,
            map_shape=map_shape,
            enable_global_map_crop=enable_global_map_crop,
            enable_adaptive_zoom=enable_adaptive_zoom,
            debug_save_renderings=debug_save_renderings,
            save_step_map_artifacts=save_step_map_artifacts,
        )

    @classmethod
    def from_existing(
        cls,
        existing: MapVisualizer,
        *,
        ablation_spec: Optional[AblationSpec] = None,
    ) -> "AblationMapVisualizer":
        visualizer = cls(
            results_dir=existing.results_dir,
            resolution=int(existing.resolution),
            map_shape=tuple(existing.map_shape),
            enable_global_map_crop=bool(existing.enable_global_map_crop),
            enable_adaptive_zoom=bool(existing.enable_adaptive_zoom),
            debug_save_renderings=bool(existing.debug_save_renderings),
            save_step_map_artifacts=bool(existing.save_step_map_artifacts),
            ablation_spec=ablation_spec,
        )
        visualizer.color_palette = list(existing.color_palette)
        return visualizer

    @staticmethod
    def _is_thinking_phase(phase: str) -> bool:
        phase_text = str(phase or "").strip().lower()
        return phase_text == "initial" or phase_text.startswith("verify")

    def save_step_visualization(self, *args, **kwargs):
        phase = str(kwargs.get("phase", "") or "").strip()
        previous_phase = self._ablation_phase
        self._ablation_phase = phase
        try:
            return super().save_step_visualization(*args, **kwargs)
        finally:
            self._ablation_phase = previous_phase

    def render_global_map(
        self,
        full_map,
        trajectory_points,
        detected_classes,
        floor=None,
        current_pose=None,
        landmark_classes=None,
        landmark_instances=None,
        landmark_config=None,
        waypoint_positions=None,
        waypoint_ids=None,
        space_area_layer=None,
        space_area_records=None,
        crop_offset=None,
        mapping_classes=None,
    ):
        if (
            self._is_thinking_phase(self._ablation_phase)
            and not self.ablation_spec.thinking_image.include_global_map_space_structure
        ):
            space_area_layer = None
            space_area_records = None
        return super().render_global_map(
            full_map,
            trajectory_points,
            detected_classes,
            floor=floor,
            current_pose=current_pose,
            landmark_classes=landmark_classes,
            landmark_instances=landmark_instances,
            landmark_config=landmark_config,
            waypoint_positions=waypoint_positions,
            waypoint_ids=waypoint_ids,
            space_area_layer=space_area_layer,
            space_area_records=space_area_records,
            crop_offset=crop_offset,
            mapping_classes=mapping_classes,
        )


__all__ = [
    "AblationMapVisualizer",
]
