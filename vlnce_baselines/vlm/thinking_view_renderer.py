"""
Thinking-view renderer.

Keep 12-view image annotation and saving out of the main controller so the
controller stays focused on orchestration instead of per-image rendering.
"""

import os
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vlnce_baselines.vlm.navigation_config import DIRECTION_CONFIG


class ThinkingViewRenderer:
    """Render and save the 12 annotated direction views used by the thinking model."""

    THINKING_DETECTION_TOPK = 3

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
    def _summarize_detected_landmarks(detected_landmarks: List[Tuple[str, float]]) -> List[Tuple[str, Tuple[int, int, int]]]:
        if not detected_landmarks:
            return []

        ranked = sorted(
            [(str(name), float(confidence)) for name, confidence in detected_landmarks],
            key=lambda item: item[1],
            reverse=True,
        )
        blue = (255, 0, 0)
        dark = (40, 40, 40)
        segments: List[Tuple[str, Tuple[int, int, int]]] = [("landmark: ", dark)]
        for idx, (name, confidence) in enumerate(ranked):
            if idx > 0:
                segments.append((", ", dark))
            segments.extend([
                (name, blue),
                (" (confidence: ", dark),
                (f"{confidence:.2f}", blue),
                (")", dark),
            ])
        return segments

    @staticmethod
    def _measure_segmented_text(
        segments: List[Tuple[str, Tuple[int, int, int]]],
        font: int,
        scale: float,
        thickness: int,
    ) -> Tuple[int, int, int]:
        total_width = 0
        max_height = 0
        max_baseline = 0
        for text, _color in segments:
            (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
            total_width += text_width
            max_height = max(max_height, text_height)
            max_baseline = max(max_baseline, baseline)
        return total_width, max_height, max_baseline

    def _build_segmented_text_strip(
        self,
        width: int,
        segments: List[Tuple[str, Tuple[int, int, int]]],
        height: int,
        font_scale: float,
        font_thickness: int,
    ) -> np.ndarray:
        strip = np.ones((height, width, 3), dtype=np.uint8) * 255
        if not segments:
            return strip

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = font_scale
        min_scale = 0.38
        max_text_width = max(10, width - 16)
        draw_segments = list(segments)

        while True:
            total_width, text_height, baseline = self._measure_segmented_text(
                draw_segments, font, scale, font_thickness
            )
            if total_width <= max_text_width or scale <= min_scale:
                break
            scale = max(min_scale, scale - 0.04)

        if total_width > max_text_width and len(draw_segments) > 5:
            while len(draw_segments) > 5 and total_width > max_text_width:
                draw_segments = draw_segments[:-4] + [(", ...", (40, 40, 40))]
                total_width, text_height, baseline = self._measure_segmented_text(
                    draw_segments, font, scale, font_thickness
                )

        text_x = max(8, (width - total_width) // 2)
        text_y = max(text_height + 4, (height + text_height) // 2 - max(0, baseline // 2))
        cursor_x = text_x
        for text, color in draw_segments:
            cv2.putText(strip, text, (cursor_x, text_y), font, scale, color, font_thickness, cv2.LINE_AA)
            text_width = cv2.getTextSize(text, font, scale, font_thickness)[0][0]
            cursor_x += text_width
        return strip

    @staticmethod
    def _filter_detection_payload(detections, labels: List[str], keep_indices: List[int]):
        if detections is None or getattr(detections, "xyxy", None) is None:
            return None, []

        if not keep_indices:
            return (
                SimpleNamespace(
                    xyxy=np.zeros((0, 4), dtype=np.float32),
                    confidence=np.zeros((0,), dtype=np.float32),
                    class_id=np.zeros((0,), dtype=np.int32),
                    tracker_id=None,
                    mask=None,
                ),
                [],
            )

        keep = np.asarray(sorted(set(int(idx) for idx in keep_indices)), dtype=np.int32)
        all_xyxy = np.asarray(detections.xyxy, dtype=np.float32)
        all_conf = getattr(detections, "confidence", None)
        all_class_id = getattr(detections, "class_id", None)
        all_tracker_id = getattr(detections, "tracker_id", None)
        all_mask = getattr(detections, "mask", None)

        return (
            SimpleNamespace(
                xyxy=all_xyxy[keep],
                confidence=(
                    np.asarray(all_conf, dtype=np.float32)[keep]
                    if all_conf is not None else np.zeros((len(keep),), dtype=np.float32)
                ),
                class_id=(
                    np.asarray(all_class_id, dtype=np.int32)[keep]
                    if all_class_id is not None else np.full((len(keep),), -1, dtype=np.int32)
                ),
                tracker_id=(
                    np.asarray(all_tracker_id, dtype=np.int32)[keep]
                    if all_tracker_id is not None else None
                ),
                mask=(
                    np.asarray(all_mask, dtype=np.float32)[keep]
                    if all_mask is not None else None
                ),
            ),
            [labels[idx] for idx in keep if 0 <= idx < len(labels)],
        )

    def _collect_topk_detection_indices(
        self,
        payloads: List[Tuple[Any, List[str], np.ndarray, Optional[np.ndarray], float, str]],
        topk: int,
    ) -> Dict[int, List[int]]:
        if topk <= 0:
            return {}

        ranked: List[Tuple[float, int, int]] = []
        for view_idx, (detections, _labels, _image, _depth, _angle, _name) in enumerate(payloads):
            if detections is None or getattr(detections, "xyxy", None) is None:
                continue
            confidences = getattr(detections, "confidence", None)
            if confidences is None:
                confidences = np.zeros((len(detections.xyxy),), dtype=np.float32)
            else:
                confidences = np.asarray(confidences, dtype=np.float32)
            for det_idx, confidence in enumerate(confidences):
                ranked.append((float(confidence), view_idx, det_idx))

        ranked.sort(key=lambda item: item[0], reverse=True)
        keep_by_view: Dict[int, List[int]] = {}
        for _confidence, view_idx, det_idx in ranked[:topk]:
            keep_by_view.setdefault(view_idx, []).append(det_idx)
        return keep_by_view

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
        detection_topk: int = THINKING_DETECTION_TOPK,
    ) -> Tuple[List[str], List[str]]:
        os.makedirs(directions_dir, exist_ok=True)

        direction_paths: List[str] = []
        direction_names: List[str] = []
        view_payloads: List[Tuple[Any, List[str], np.ndarray, Optional[np.ndarray], float, str]] = []

        for config in DIRECTION_CONFIG:
            step_idx = config["step"]
            angle = config["angle"]
            direction_name = config["name"]

            image = lookaround_images[step_idx - 1].copy()
            depth_meters = lookaround_depths[step_idx - 1] if step_idx - 1 < len(lookaround_depths) else None
            dets_view, labels_view = None, []
            if landmark_classes:
                dets_view, labels_view, _ = detect_landmarks_fn(image, landmark_classes)
            view_payloads.append((dets_view, labels_view, image, depth_meters, angle, direction_name))

        keep_by_view = self._collect_topk_detection_indices(view_payloads, topk=detection_topk)

        for view_idx, (dets_view, labels_view, image, depth_meters, angle, direction_name) in enumerate(view_payloads):
            detected_landmarks_view: List[Tuple[str, float]] = []
            if landmark_classes:
                filtered_dets, filtered_labels = self._filter_detection_payload(
                    dets_view,
                    labels_view,
                    keep_by_view.get(view_idx, []),
                )
                image, detected_landmarks_view, _, _ = render_detection_fn(
                    image,
                    filtered_dets,
                    filtered_labels,
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
            bottom_segments = self._summarize_detected_landmarks(detected_landmarks_view)
            if bottom_segments:
                bottom_label = self._build_segmented_text_strip(
                    width,
                    bottom_segments,
                    height=30,
                    font_scale=0.52,
                    font_thickness=1,
                )
                labeled_image = np.vstack([top_label, image, bottom_label])
            else:
                labeled_image = np.vstack([top_label, image])

            direction_filename = f"{phase}_direction_{angle:03d}.png"
            direction_path = os.path.join(directions_dir, direction_filename)
            cv2.imwrite(direction_path, labeled_image)

            direction_paths.append(direction_path)
            direction_names.append(direction_name)

        return direction_paths, direction_names
