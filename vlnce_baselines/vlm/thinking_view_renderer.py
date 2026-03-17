"""
Thinking-view renderer.

Keep 12-view image annotation and saving out of the main controller so the
controller stays focused on orchestration instead of per-image rendering.
"""

import os
import math
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vlnce_baselines.visualization.landmark_overlay import (
    LandmarkStripLine,
    LandmarkStripSegment,
    render_landmark_strip,
)
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
    def _normalize_angle_deg(angle_deg: float) -> float:
        angle = float(angle_deg) % 360.0
        if angle < 0:
            angle += 360.0
        return angle

    @classmethod
    def _angle_delta_deg(cls, angle_a: float, angle_b: float) -> float:
        diff = cls._normalize_angle_deg(angle_a) - cls._normalize_angle_deg(angle_b)
        while diff > 180.0:
            diff -= 360.0
        while diff < -180.0:
            diff += 360.0
        return diff

    @staticmethod
    def _short_text(text: str, max_len: int = 40) -> str:
        text = str(text or "").strip()
        if len(text) <= max_len:
            return text
        return text[: max(0, max_len - 2)].rstrip() + ".."

    @classmethod
    def _build_waypoint_view_entries(
        cls,
        waypoint_info: Optional[tuple],
        waypoint_area_labels: Optional[List[str]],
        current_pose: Optional[np.ndarray],
        resolution_cm: float,
    ) -> List[Dict[str, Any]]:
        if not waypoint_info or current_pose is None:
            return []

        waypoint_positions, waypoint_ids, waypoint_descriptions = waypoint_info
        if not waypoint_ids:
            return []

        curr_x_m, curr_y_m, curr_orientation_deg = [float(v) for v in current_pose[:3]]
        area_labels = list(waypoint_area_labels or [])
        entries: List[Dict[str, Any]] = []

        for index, (wp_id, wp_desc, (wp_py, wp_px)) in enumerate(
            zip(waypoint_ids, waypoint_descriptions, waypoint_positions)
        ):
            wp_x_m = float(wp_px) * float(resolution_cm) / 100.0
            wp_y_m = float(wp_py) * float(resolution_cm) / 100.0
            dx = wp_x_m - curr_x_m
            dy = wp_y_m - curr_y_m
            distance_m = float(math.hypot(dx, dy))
            absolute_angle_deg = float(math.degrees(math.atan2(dy, dx)))
            relative_bearing_deg = float(curr_orientation_deg - absolute_angle_deg)
            view_angle_deg = cls._normalize_angle_deg(-relative_bearing_deg)

            area_label = str(area_labels[index] if index < len(area_labels) else "").strip()
            description = str(wp_desc or "").strip()
            label_text = area_label or description.split(" - ", 1)[0].strip() or f"WP#{wp_id}"

            entries.append({
                "id": int(wp_id),
                "label": cls._short_text(label_text, max_len=34),
                "description": description,
                "area_label": area_label,
                "distance_m": distance_m,
                "relative_bearing_deg": relative_bearing_deg,
                "view_angle_deg": view_angle_deg,
                "is_last_visited": index == len(waypoint_ids) - 1,
            })

        return entries

    @classmethod
    def _select_waypoints_for_view(
        cls,
        waypoint_entries: List[Dict[str, Any]],
        view_angle_deg: float,
        hfov_deg: float,
    ) -> List[Dict[str, Any]]:
        if not waypoint_entries:
            return []

        half_fov = max(1.0, float(hfov_deg) / 2.0)
        selected = [
            dict(entry)
            for entry in waypoint_entries
            if abs(cls._angle_delta_deg(float(entry["view_angle_deg"]), float(view_angle_deg))) <= half_fov
        ]
        selected.sort(key=lambda item: (float(item["distance_m"]), int(item["id"])))
        return selected

    @classmethod
    def _build_bottom_strip_lines(
        cls,
        visible_entries_meta: List[Dict[str, Any]],
        waypoint_entries: List[Dict[str, Any]],
    ) -> List[LandmarkStripLine]:
        lines: List[LandmarkStripLine] = []
        prefix_color = (40, 40, 40)
        value_color = (255, 0, 0)

        same_name_counts: Dict[str, int] = {}
        for entry in visible_entries_meta:
            same_name_counts[str(entry.get("name", ""))] = same_name_counts.get(str(entry.get("name", "")), 0) + 1

        seen_name_indices: Dict[str, int] = {}
        for entry in visible_entries_meta:
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            seen_name_indices[name] = seen_name_indices.get(name, 0) + 1
            suffix = f" #{seen_name_indices[name]}" if same_name_counts.get(name, 0) > 1 else ""
            lines.append(
                LandmarkStripLine(
                    distance_m=float(entry.get("distance_m", 1e9)),
                    confidence=float(entry.get("confidence", 0.0)),
                    priority=0,
                    segments=(
                        LandmarkStripSegment("landmark: ", prefix_color),
                        LandmarkStripSegment(cls._short_text(f"{name}{suffix}", max_len=30), value_color),
                        LandmarkStripSegment(f"  {float(entry.get('distance_m', 0.0)):.1f}m", value_color),
                        LandmarkStripSegment("  confidence: ", prefix_color),
                        LandmarkStripSegment(f"{float(entry.get('confidence', 0.0)):.3f}", value_color),
                    ),
                )
            )

        for entry in waypoint_entries:
            note = " (came from here)" if bool(entry.get("is_last_visited")) else ""
            waypoint_text = f"WP#{int(entry.get('id', 0))} {entry.get('label', 'Unknown')}".strip()
            lines.append(
                LandmarkStripLine(
                    distance_m=float(entry.get("distance_m", 1e9)),
                    confidence=0.0,
                    priority=1,
                    segments=(
                        LandmarkStripSegment("waypoint area: ", prefix_color),
                        LandmarkStripSegment(cls._short_text(waypoint_text, max_len=34), value_color),
                        LandmarkStripSegment(f"  {float(entry.get('distance_m', 0.0)):.1f}m", value_color),
                        LandmarkStripSegment(note, prefix_color),
                    ),
                )
            )

        lines.sort(key=lambda line: (float(line.distance_m), int(line.priority), -float(line.confidence)))
        return lines

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
        waypoint_area_labels: Optional[List[str]],
        current_pose: Optional[np.ndarray],
        resolution_cm: float,
        waypoint_angle_deg: Optional[float],
        draw_waypoints_fn: Callable[[np.ndarray, float, tuple], np.ndarray],
        detection_topk: int = THINKING_DETECTION_TOPK,
    ) -> Tuple[List[str], List[str]]:
        os.makedirs(directions_dir, exist_ok=True)

        direction_paths: List[str] = []
        direction_names: List[str] = []
        view_payloads: List[Tuple[Any, List[str], np.ndarray, Optional[np.ndarray], float, str]] = []
        waypoint_entries = self._build_waypoint_view_entries(
            waypoint_info=waypoint_info,
            waypoint_area_labels=waypoint_area_labels,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
        )

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
            visible_entries_meta: List[Dict[str, Any]] = []
            if landmark_classes:
                filtered_dets, filtered_labels = self._filter_detection_payload(
                    dets_view,
                    labels_view,
                    keep_by_view.get(view_idx, []),
                )
                image, detected_landmarks_view, _, _, visible_entries_meta = render_detection_fn(
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
            view_waypoint_entries = self._select_waypoints_for_view(
                waypoint_entries=waypoint_entries,
                view_angle_deg=float(angle),
                hfov_deg=79.0,
            )
            bottom_lines = self._build_bottom_strip_lines(
                visible_entries_meta=visible_entries_meta,
                waypoint_entries=view_waypoint_entries,
            )
            if bottom_lines:
                bottom_label = render_landmark_strip(
                    width,
                    bottom_lines,
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
