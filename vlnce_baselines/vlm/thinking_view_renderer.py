"""
Thinking-view renderer.

Keep 12-view image annotation and saving out of the main controller so the
controller stays focused on orchestration instead of per-image rendering.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vlnce_baselines.vlm.navigation_config import DIRECTION_CONFIG


class ThinkingViewRenderer:
    """Render and save the 12 annotated direction views used by the thinking model."""

    @staticmethod
    def _build_text_strip(
        width: int,
        text: str,
        height: int,
        font_scale: float,
        font_thickness: int,
        text_color: Tuple[int, int, int],
    ) -> np.ndarray:
        strip = np.ones((height, width, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = (text or "").strip()
        if not text:
            return strip

        scale = font_scale
        min_scale = 0.38
        max_text_width = max(10, width - 16)
        text_size = cv2.getTextSize(text, font, scale, font_thickness)[0]
        while text_size[0] > max_text_width and scale > min_scale:
            scale = max(min_scale, scale - 0.04)
            text_size = cv2.getTextSize(text, font, scale, font_thickness)[0]

        if text_size[0] > max_text_width:
            trimmed = text
            while len(trimmed) > 3:
                trimmed = trimmed[:-1]
                candidate = trimmed.rstrip() + "..."
                text_size = cv2.getTextSize(candidate, font, scale, font_thickness)[0]
                if text_size[0] <= max_text_width:
                    text = candidate
                    break

        text_width, text_height = cv2.getTextSize(text, font, scale, font_thickness)[0]
        baseline = cv2.getTextSize(text, font, scale, font_thickness)[1]
        text_x = max(8, (width - text_width) // 2)
        text_y = max(text_height + 4, (height + text_height) // 2 - max(0, baseline // 2))
        cv2.putText(strip, text, (text_x, text_y), font, scale, text_color, font_thickness, cv2.LINE_AA)
        return strip

    @staticmethod
    def _summarize_detected_landmarks(detected_landmarks: List[Tuple[str, float]]) -> str:
        counts: Dict[str, int] = {}
        for name, _confidence in detected_landmarks or []:
            counts[name] = counts.get(name, 0) + 1

        if not counts:
            return "Detected landmark: none"

        parts = []
        for name, count in counts.items():
            parts.append(f"{name} x{count}" if count > 1 else name)
        return "Detected landmark: " + ", ".join(parts)

    @staticmethod
    def _should_draw_waypoint(view_angle: float, waypoint_angle_deg: Optional[float]) -> bool:
        if waypoint_angle_deg is None:
            return False

        waypoint_view_angle = -float(waypoint_angle_deg)
        while waypoint_view_angle < 0:
            waypoint_view_angle += 360
        while waypoint_view_angle >= 360:
            waypoint_view_angle -= 360

        angle_diff = waypoint_view_angle - float(view_angle)
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        return abs(angle_diff) <= 15

    def save_direction_views(
        self,
        directions_dir: str,
        phase: str,
        lookaround_images: List[np.ndarray],
        lookaround_depths: List[Optional[np.ndarray]],
        landmark_classes: Optional[List[str]],
        detect_landmarks_fn: Callable[[np.ndarray, List[str]], Tuple[Any, List[str], Any]],
        render_detection_fn: Callable[[np.ndarray, Any, List[str], Optional[np.ndarray]], Tuple[np.ndarray, List[Tuple[str, float]], Any, Any]],
        draw_distance_fn: Callable[[np.ndarray, str], np.ndarray],
        distance_lookup: Dict[str, str],
        waypoint_info: Optional[tuple],
        waypoint_angle_deg: Optional[float],
        draw_waypoints_fn: Callable[[np.ndarray, float, tuple], np.ndarray],
    ) -> Tuple[List[str], List[str]]:
        os.makedirs(directions_dir, exist_ok=True)

        direction_paths: List[str] = []
        direction_names: List[str] = []

        for config in DIRECTION_CONFIG:
            step_idx = config["step"]
            angle = config["angle"]
            direction_name = config["name"]

            image = lookaround_images[step_idx - 1].copy()
            depth_meters = lookaround_depths[step_idx - 1] if step_idx - 1 < len(lookaround_depths) else None
            detected_landmarks_view: List[Tuple[str, float]] = []

            if landmark_classes:
                dets_view, labels_view, _ = detect_landmarks_fn(image, landmark_classes)
                image, detected_landmarks_view, _, _ = render_detection_fn(
                    image,
                    dets_view,
                    labels_view,
                    depth_meters,
                )

            dist_key = f"angle_{angle}"
            dist_str = distance_lookup.get(dist_key, "Unknown")
            image = draw_distance_fn(image, dist_str)

            if waypoint_info and self._should_draw_waypoint(angle, waypoint_angle_deg):
                image = draw_waypoints_fn(image, angle, waypoint_info)

            _h, width = image.shape[:2]
            top_label = self._build_text_strip(
                width,
                direction_name,
                height=28,
                font_scale=0.68,
                font_thickness=1,
                text_color=(0, 0, 255),
            )
            bottom_label = self._build_text_strip(
                width,
                self._summarize_detected_landmarks(detected_landmarks_view),
                height=30,
                font_scale=0.52,
                font_thickness=1,
                text_color=(40, 40, 40),
            )
            labeled_image = np.vstack([top_label, image, bottom_label])

            direction_filename = f"{phase}_direction_{angle:03d}.png"
            direction_path = os.path.join(directions_dir, direction_filename)
            cv2.imwrite(direction_path, labeled_image)

            direction_paths.append(direction_path)
            direction_names.append(direction_name)

        return direction_paths, direction_names
