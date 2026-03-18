"""
Thinking-view renderer.

Keep 12-view image annotation and saving out of the main controller so the
controller stays focused on orchestration instead of per-image rendering.
"""

import os
import math
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from vlnce_baselines.visualization.landmark_overlay import (
    LandmarkStripLine,
    LandmarkStripSegment,
    render_landmark_strip,
)
from vlnce_baselines.common.spatial_formatter import snap_relative_bearing
from vlnce_baselines.vlm.navigation_config import DIRECTION_CONFIG
from vlnce_baselines.visualization.map_projection import RotatedMapProjector


class ThinkingViewRenderer:
    """Render and save the 12 annotated direction views used by the thinking model."""

    THINKING_DETECTION_TOPK = 3
    CURRENT_AREA_OVERLAP_THRESHOLD_M = 1.0
    VIEW_HFOV_DEG = 79.0
    WAYPOINT_VISIBILITY_RADIUS_M = 1.0
    WAYPOINT_VISIBILITY_SAMPLES = 16

    @staticmethod
    def _is_known_area_label(area_label: str) -> bool:
        return str(area_label or "").strip().lower() not in {"", "unknown"}

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
        current_room_area_label: str,
    ) -> List[Dict[str, Any]]:
        if current_pose is None:
            return []

        if not waypoint_info:
            current_area_text = str(current_room_area_label or "Unknown").strip() or "Unknown"
            if not cls._is_known_area_label(current_area_text):
                return []
            return [{
                "id": 0,
                "label": cls._short_text(current_area_text, max_len=34),
                "description": current_area_text,
                "area_label": current_area_text,
                "distance_m": 0.0,
                "relative_bearing_deg": 0.0,
                "snapped_relative_bearing_deg": 0.0,
                "view_angle_deg": 0.0,
                "is_last_visited": False,
                "is_current_area": True,
            }]

        waypoint_positions, waypoint_ids, waypoint_descriptions = waypoint_info
        if not waypoint_ids:
            current_area_text = str(current_room_area_label or "Unknown").strip() or "Unknown"
            if not cls._is_known_area_label(current_area_text):
                return []
            return [{
                "id": 0,
                "label": cls._short_text(current_area_text, max_len=34),
                "description": current_area_text,
                "area_label": current_area_text,
                "distance_m": 0.0,
                "relative_bearing_deg": 0.0,
                "snapped_relative_bearing_deg": 0.0,
                "view_angle_deg": 0.0,
                "is_last_visited": False,
                "is_current_area": True,
            }]

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
            snapped_relative_bearing_deg = float(snap_relative_bearing(relative_bearing_deg))
            view_angle_deg = cls._normalize_angle_deg(-snapped_relative_bearing_deg)

            area_label = str(area_labels[index] if index < len(area_labels) else "").strip()
            description = str(wp_desc or "").strip()
            label_text = area_label or description.split(" - ", 1)[0].strip() or f"WP#{wp_id}"

            entries.append({
                "id": int(wp_id),
                "label": cls._short_text(label_text, max_len=34),
                "description": description,
                "area_label": area_label,
                "world_py": int(wp_py),
                "world_px": int(wp_px),
                "distance_m": distance_m,
                "relative_bearing_deg": relative_bearing_deg,
                "snapped_relative_bearing_deg": snapped_relative_bearing_deg,
                "view_angle_deg": view_angle_deg,
                "is_last_visited": index == len(waypoint_ids) - 1,
            })

        current_area_text = str(current_room_area_label or "Unknown").strip() or "Unknown"
        current_area_view_angle = 0.0
        current_area_relative_bearing = 0.0
        current_area_snapped_bearing = 0.0
        if entries:
            last_entry = entries[-1]
            current_area_view_angle = float(last_entry["view_angle_deg"])
            current_area_relative_bearing = float(last_entry["relative_bearing_deg"])
            current_area_snapped_bearing = float(last_entry.get("snapped_relative_bearing_deg", 0.0))
            if float(last_entry["distance_m"]) <= cls.CURRENT_AREA_OVERLAP_THRESHOLD_M:
                entries.pop()

        if cls._is_known_area_label(current_area_text):
            entries.append({
                "id": 0,
                "label": cls._short_text(current_area_text, max_len=34),
                "description": current_area_text,
                "area_label": current_area_text,
                "distance_m": 0.0,
                "relative_bearing_deg": current_area_relative_bearing,
                "snapped_relative_bearing_deg": current_area_snapped_bearing,
                "view_angle_deg": current_area_view_angle,
                "is_last_visited": False,
                "is_current_area": True,
            })

        return entries

    @staticmethod
    def _build_projector(
        full_map: Optional[np.ndarray],
        current_pose: Optional[np.ndarray],
        crop_offset: Optional[Tuple[int, int]],
    ) -> Optional[RotatedMapProjector]:
        if full_map is None or current_pose is None or crop_offset is None:
            return None
        return RotatedMapProjector(
            map_h=full_map.shape[1],
            map_w=full_map.shape[2],
            crop_offset=crop_offset,
            agent_orientation_deg=float(current_pose[2]),
        )

    @staticmethod
    def _line_is_clear(
        obstacle_mask: np.ndarray,
        start_row: float,
        start_col: float,
        end_row: float,
        end_col: float,
    ) -> bool:
        steps = max(int(math.ceil(max(abs(end_row - start_row), abs(end_col - start_col)))), 1)
        rows = np.linspace(start_row, end_row, steps + 1)
        cols = np.linspace(start_col, end_col, steps + 1)
        height, width = obstacle_mask.shape
        for idx in range(1, steps):
            row = int(round(rows[idx]))
            col = int(round(cols[idx]))
            if not (0 <= row < height and 0 <= col < width):
                return False
            if bool(obstacle_mask[row, col]):
                return False
        return True

    @classmethod
    def _has_visible_waypoint_ray(
        cls,
        entry: Dict[str, Any],
        obstacle_mask: np.ndarray,
        projector: RotatedMapProjector,
        current_pose: np.ndarray,
        resolution_cm: float,
    ) -> bool:
        curr_py = float(current_pose[1]) * 100.0 / float(resolution_cm)
        curr_px = float(current_pose[0]) * 100.0 / float(resolution_cm)
        start_rot = projector.world_to_rotated_pixel(curr_py, curr_px)
        if start_rot is None:
            return False

        center_py = float(entry.get("world_py", 0.0))
        center_px = float(entry.get("world_px", 0.0))
        radius_px = (cls.WAYPOINT_VISIBILITY_RADIUS_M * 100.0) / float(resolution_cm)

        candidate_points: List[Tuple[float, float]] = [(center_py, center_px)]
        for sample_idx in range(cls.WAYPOINT_VISIBILITY_SAMPLES):
            theta = (2.0 * math.pi * float(sample_idx)) / float(cls.WAYPOINT_VISIBILITY_SAMPLES)
            candidate_points.append((
                center_py + radius_px * math.sin(theta),
                center_px + radius_px * math.cos(theta),
            ))

        for world_py, world_px in candidate_points:
            end_rot = projector.world_to_rotated_pixel(world_py, world_px)
            if end_rot is None:
                continue
            if cls._line_is_clear(
                obstacle_mask=obstacle_mask,
                start_row=float(start_rot[0]),
                start_col=float(start_rot[1]),
                end_row=float(end_rot[0]),
                end_col=float(end_rot[1]),
            ):
                return True
        return False

    @classmethod
    def _visible_view_angles_for_waypoint(
        cls,
        entry: Dict[str, Any],
        view_angles_deg: List[float],
    ) -> List[float]:
        if not view_angles_deg:
            return []

        distance_m = float(entry.get("distance_m", 0.0))
        if distance_m <= 1e-6:
            return []

        view_angle_deg = cls._normalize_angle_deg(float(entry.get("view_angle_deg", 0.0)))
        ratio = min(1.0, cls.WAYPOINT_VISIBILITY_RADIUS_M / max(distance_m, cls.WAYPOINT_VISIBILITY_RADIUS_M))
        half_span_deg = math.degrees(math.asin(ratio))
        max_view_delta = (cls.VIEW_HFOV_DEG / 2.0) + half_span_deg
        visible_angles = [
            float(angle)
            for angle in view_angles_deg
            if abs(cls._angle_delta_deg(float(angle), view_angle_deg)) <= max_view_delta
        ]
        if visible_angles:
            return visible_angles

        nearest = cls._nearest_view_angle(view_angle_deg, view_angles_deg)
        return [float(nearest)] if nearest is not None else []

    @classmethod
    def _apply_waypoint_visibility(
        cls,
        waypoint_entries: List[Dict[str, Any]],
        view_angles_deg: List[float],
        full_map: Optional[np.ndarray],
        current_pose: Optional[np.ndarray],
        resolution_cm: float,
        crop_offset: Optional[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        if not waypoint_entries:
            return waypoint_entries

        obstacle_mask = None
        projector = cls._build_projector(full_map, current_pose, crop_offset)
        if full_map is not None and full_map.shape[0] > 0:
            obstacle_mask = np.asarray(full_map[0] > 0.5, dtype=bool)

        filtered_entries: List[Dict[str, Any]] = []
        for entry in waypoint_entries:
            if bool(entry.get("is_current_area")):
                filtered_entries.append(dict(entry))
                continue

            visible_view_angles: List[float] = []
            if obstacle_mask is not None and projector is not None and current_pose is not None:
                if cls._has_visible_waypoint_ray(
                    entry=entry,
                    obstacle_mask=obstacle_mask,
                    projector=projector,
                    current_pose=current_pose,
                    resolution_cm=resolution_cm,
                ):
                    visible_view_angles = cls._visible_view_angles_for_waypoint(entry, view_angles_deg)
            if not visible_view_angles:
                continue

            updated_entry = dict(entry)
            updated_entry["visible_view_angles"] = list(visible_view_angles)
            filtered_entries.append(updated_entry)

        return filtered_entries

    @classmethod
    def _nearest_view_angle(
        cls,
        target_angle_deg: float,
        view_angles_deg: List[float],
    ) -> Optional[float]:
        if not view_angles_deg:
            return None

        normalized_target = cls._normalize_angle_deg(target_angle_deg)
        return min(
            [float(angle) for angle in view_angles_deg],
            key=lambda angle: (
                abs(cls._angle_delta_deg(normalized_target, float(angle))),
                cls._normalize_angle_deg(float(angle) - normalized_target),
            ),
        )

    @classmethod
    def _assign_waypoints_to_views(
        cls,
        waypoint_entries: List[Dict[str, Any]],
        view_angles_deg: List[float],
    ) -> Dict[float, List[Dict[str, Any]]]:
        assignments: Dict[float, List[Dict[str, Any]]] = {
            float(angle): [] for angle in view_angles_deg
        }
        if not waypoint_entries or not view_angles_deg:
            return assignments

        for entry in waypoint_entries:
            visible_view_angles = [float(angle) for angle in entry.get("visible_view_angles", [])]
            if visible_view_angles:
                for visible_angle in visible_view_angles:
                    assignments.setdefault(float(visible_angle), []).append(dict(entry))
                continue

            entry_view_angle = cls._normalize_angle_deg(float(entry.get("view_angle_deg", 0.0)))
            assigned_angle = (
                entry_view_angle
                if any(abs(entry_view_angle - float(angle)) < 1e-3 for angle in view_angles_deg)
                else cls._nearest_view_angle(entry_view_angle, view_angles_deg)
            )
            if assigned_angle is None:
                continue
            assignments.setdefault(float(assigned_angle), []).append(dict(entry))

        for angle in assignments:
            assignments[angle].sort(
                key=lambda item: (float(item["distance_m"]), int(item["id"]))
            )
        return assignments

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
            if bool(entry.get("is_current_area")):
                lines.append(
                    LandmarkStripLine(
                        distance_m=float(entry.get("distance_m", 0.0)),
                        confidence=0.0,
                        priority=1,
                        segments=(
                            LandmarkStripSegment("your current area: ", prefix_color),
                            LandmarkStripSegment(cls._short_text(entry.get("label", "Unknown"), max_len=34), value_color),
                        ),
                    )
                )
                continue

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

    @classmethod
    def _should_draw_waypoint(
        cls,
        view_angle: float,
        waypoint_angle_deg: Optional[float],
        view_angles_deg: List[float],
    ) -> bool:
        if waypoint_angle_deg is None:
            return False

        snapped_waypoint_bearing_deg = float(snap_relative_bearing(float(waypoint_angle_deg)))
        waypoint_view_angle = cls._normalize_angle_deg(-snapped_waypoint_bearing_deg)
        assigned_view_angle = cls._nearest_view_angle(waypoint_view_angle, view_angles_deg)
        if assigned_view_angle is None:
            return False
        return abs(cls._angle_delta_deg(float(view_angle), float(assigned_view_angle))) < 1e-3

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
        current_room_area_label: str,
        full_map: Optional[np.ndarray],
        crop_offset: Optional[Tuple[int, int]],
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
            current_room_area_label=current_room_area_label,
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
        waypoint_marker_view_angles: Set[float] = set()
        for entry in waypoint_entries:
            if bool(entry.get("is_last_visited")):
                waypoint_marker_view_angles.update(
                    float(angle) for angle in entry.get("visible_view_angles", [])
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

            if waypoint_info and float(angle) in waypoint_marker_view_angles:
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
            view_waypoint_entries = list(waypoint_entries_by_view.get(float(angle), []))
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
