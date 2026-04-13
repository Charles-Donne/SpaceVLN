"""Thinking-view renderer with ablation-aware output switches."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from navigation_system.ablation.config import AblationSpec, load_ablation_spec
from navigation_system.render.image_resize import resize_image_to_width
from navigation_system.render.map.landmark_overlay import LandmarkStripLine, render_landmark_strip
from navigation_system.render.views.thinking_view_renderer import ThinkingViewRenderer
from navigation_system.vlm.contracts.schema import DIRECTION_CONFIG


class AblationThinkingViewRenderer(ThinkingViewRenderer):
    """Render the same 12 views while controlling what finally appears on the image."""

    def __init__(self, ablation_spec: Optional[AblationSpec] = None):
        super().__init__()
        self.ablation_spec = ablation_spec or load_ablation_spec()

    def _filter_bottom_lines(
        self,
        lines: List[LandmarkStripLine],
    ) -> List[LandmarkStripLine]:
        if not lines:
            return []

        include_landmarks = bool(self.ablation_spec.thinking_image.include_landmark_strip)
        include_space_waypoints = bool(self.ablation_spec.thinking_image.include_space_waypoint_strip)
        if include_landmarks and include_space_waypoints:
            return lines

        filtered: List[LandmarkStripLine] = []
        for line in lines:
            first_text = ""
            if getattr(line, "segments", None):
                first_text = str(line.segments[0].text or "").strip().lower()
            is_landmark_line = first_text.startswith("landmark:")
            is_space_line = (
                first_text.startswith("space waypoint:")
                or first_text.startswith("your current area:")
            )
            if is_landmark_line and not include_landmarks:
                continue
            if is_space_line and not include_space_waypoints:
                continue
            filtered.append(line)
        return filtered

    def render_direction_views(
        self,
        phase: str,
        lookaround_images: List[np.ndarray],
        lookaround_depths: List[Optional[np.ndarray]],
        lookaround_detection_payloads: Optional[List[Tuple[Any, List[str], Any]]],
        landmark_classes: Optional[List[str]],
        detect_landmarks_fn: Callable[[np.ndarray, List[str]], Tuple[Any, List[str], Any]],
        render_detection_fn: Callable[[np.ndarray, Any, List[str], Optional[np.ndarray]], Tuple[np.ndarray, List[Tuple[str, float]], Any, Any]],
        draw_distance_fn: Callable[[np.ndarray, str], np.ndarray],
        distance_lookup: Dict[str, str],
        waypoint_info: Optional[tuple],
        waypoint_area_labels: Optional[List[str]],
        waypoint_floor_ids: Optional[List[int]],
        current_pose: Optional[np.ndarray],
        resolution_cm: float,
        current_space_area_label: str,
        full_map: Optional[np.ndarray],
        crop_offset: Optional[Tuple[int, int]],
        waypoint_angle_deg: Optional[float],
        draw_waypoints_fn: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
        current_floor_id: int = 0,
        initial_waypoint_index: Optional[int] = 0,
        detection_topk: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        rendered_views: List[Dict[str, Any]] = []
        view_payloads: List[Tuple[Any, List[str], np.ndarray, Optional[np.ndarray], float, str]] = []
        waypoint_entries = self._build_waypoint_view_entries(
            waypoint_info=waypoint_info,
            waypoint_area_labels=waypoint_area_labels,
            waypoint_floor_ids=waypoint_floor_ids,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            current_space_area_label=current_space_area_label,
            full_map=full_map,
            crop_offset=crop_offset,
            current_floor_id=current_floor_id,
            initial_waypoint_index=initial_waypoint_index,
        )
        view_angles = [float(config["angle"]) for config in DIRECTION_CONFIG]
        waypoint_entries = self._apply_waypoint_visibility(
            waypoint_entries=waypoint_entries,
            view_angles_deg=view_angles,
            full_map=full_map,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            crop_offset=crop_offset,
        )
        waypoint_entries_by_view = self._assign_waypoints_to_views(
            waypoint_entries=waypoint_entries,
            view_angles_deg=view_angles,
        )

        for config in DIRECTION_CONFIG:
            step_idx = config["step"]
            angle = config["angle"]
            direction_name = config["name"]

            image = lookaround_images[step_idx - 1].copy()
            depth_meters = lookaround_depths[step_idx - 1] if step_idx - 1 < len(lookaround_depths) else None
            dets_view, labels_view = None, []
            if landmark_classes:
                if lookaround_detection_payloads and step_idx - 1 < len(lookaround_detection_payloads):
                    dets_view, labels_view, _ = lookaround_detection_payloads[step_idx - 1]
                    labels_view = list(labels_view or [])
                else:
                    dets_view, labels_view, _ = detect_landmarks_fn(image, landmark_classes)
                dets_view, labels_view = self._remove_transition_like_detections(
                    dets_view,
                    labels_view,
                )
            view_payloads.append((dets_view, labels_view, image, depth_meters, angle, direction_name))

        topk = int(detection_topk) if detection_topk is not None else int(self.THINKING_DETECTION_TOPK)
        keep_by_view = self._select_grouped_detection_indices(view_payloads, topk=topk)
        include_detection_boxes = bool(self.ablation_spec.thinking_image.include_detection_boxes)
        include_obstacle_text = bool(self.ablation_spec.thinking_image.include_obstacle_text)
        include_last_visited_marker = bool(self.ablation_spec.thinking_image.include_last_visited_marker)

        for view_idx, (dets_view, labels_view, original_image, depth_meters, angle, direction_name) in enumerate(view_payloads):
            image = original_image.copy()
            visible_entries_meta: List[Dict[str, Any]] = []
            if landmark_classes:
                filtered_dets, filtered_labels = self._filter_detection_payload(
                    dets_view,
                    labels_view,
                    keep_by_view.get(view_idx, []),
                )
                rendered_detection_image, _detected_landmarks_view, _, _, visible_entries_meta = render_detection_fn(
                    original_image.copy(),
                    filtered_dets,
                    filtered_labels,
                    depth_meters,
                )
                if include_detection_boxes:
                    image = rendered_detection_image

            dist_key = f"angle_{angle}"
            dist_str = distance_lookup.get(dist_key, "Unknown")
            if include_obstacle_text:
                image = draw_distance_fn(image, dist_str)
            image = resize_image_to_width(image, self.MODEL_CONTENT_WIDTH)

            _h, width = image.shape[:2]
            top_text = (
                f"{direction_name} | Obstacle: {dist_str}"
                if include_obstacle_text
                else str(direction_name)
            )
            top_label = self._build_text_strip(
                width,
                top_text,
                height=28,
                font_scale=0.68,
                font_thickness=1,
                text_color=(0, 0, 255),
            )
            view_waypoint_entries = list(waypoint_entries_by_view.get(float(angle), []))
            marker_entry = next(
                (entry for entry in view_waypoint_entries if bool(entry.get("is_last_visited"))),
                None,
            )
            if marker_entry is not None and include_last_visited_marker:
                image = draw_waypoints_fn(image, marker_entry)

            bottom_lines = self._filter_bottom_lines(
                self._build_bottom_strip_lines(
                    visible_entries_meta=visible_entries_meta,
                    waypoint_entries=view_waypoint_entries,
                )
            )
            if bottom_lines:
                bottom_label = render_landmark_strip(
                    width,
                    bottom_lines,
                    font_scale=0.48,
                    font_thickness=1,
                    compact=True,
                )
                labeled_image = np.vstack([top_label, image, bottom_label])
            else:
                labeled_image = np.vstack([top_label, image])

            rendered_views.append(
                {
                    "phase": str(phase),
                    "angle": int(angle),
                    "direction_name": str(direction_name),
                    "image": labeled_image,
                }
            )

        return rendered_views


__all__ = [
    "AblationThinkingViewRenderer",
]
