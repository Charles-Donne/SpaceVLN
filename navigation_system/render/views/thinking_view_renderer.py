"""
Thinking-view renderer.

Keep 12-view image annotation and saving out of the main controller so the
controller stays focused on orchestration instead of per-image rendering.
"""

import os
import math
import re
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from navigation_system.config.core.params.api import (
    THINKING_VIEW_MODEL_CONTENT_WIDTH,
)
from navigation_system.config.core.params.spatial import (
    CURRENT_AREA_OVERLAP_THRESHOLD_M as CFG_CURRENT_AREA_OVERLAP_THRESHOLD_M,
    SPACE_AREA_CURRENT_WAYPOINT_MAX_DISTANCE_M as CFG_SPACE_AREA_CURRENT_WAYPOINT_MAX_DISTANCE_M,
    THINKING_DETECTION_GROUP_MAX_VIEWS as CFG_THINKING_DETECTION_GROUP_MAX_VIEWS,
    THINKING_DETECTION_OBJECT_TOTAL_MAX_VIEWS as CFG_THINKING_DETECTION_OBJECT_TOTAL_MAX_VIEWS,
    THINKING_DETECTION_TRANSITION_TOTAL_MAX_VIEWS as CFG_THINKING_DETECTION_TRANSITION_TOTAL_MAX_VIEWS,
    THINKING_DETECTION_TOPK as CFG_THINKING_DETECTION_TOPK,
    THINKING_SAME_OBJECT_BEARING_THRESHOLD_DEG as CFG_THINKING_SAME_OBJECT_BEARING_THRESHOLD_DEG,
    THINKING_SAME_OBJECT_DISTANCE_RATIO as CFG_THINKING_SAME_OBJECT_DISTANCE_RATIO,
    THINKING_SAME_OBJECT_DISTANCE_THRESHOLD_M as CFG_THINKING_SAME_OBJECT_DISTANCE_THRESHOLD_M,
    THINKING_VIEW_HFOV_DEG as CFG_THINKING_VIEW_HFOV_DEG,
    WAYPOINT_VISIBILITY_RADIUS_M as CFG_WAYPOINT_VISIBILITY_RADIUS_M,
    WAYPOINT_VISIBILITY_SAMPLES as CFG_WAYPOINT_VISIBILITY_SAMPLES,
)
from navigation_system.config.core.params.landmarks import (
    LANDMARK_EDGE_DEPTH_KEYWORDS as CFG_LANDMARK_EDGE_DEPTH_KEYWORDS,
)
from navigation_system.render.map.landmark_overlay import (
    LandmarkStripLine,
    LandmarkStripSegment,
    render_landmark_strip,
)
from navigation_system.render.image_resize import resize_image_to_width
from navigation_system.space.landmarks.landmark_selection import (
    _build_outer_ring_sampling_mask,
    _sample_random_mask_coords,
)
from navigation_system.space.description.direction_format import snap_relative_bearing
from navigation_system.space.description.spatial_formatter import (
    resolve_display_current_area,
    resolve_last_distinct_waypoint_index,
    select_display_waypoint_indices,
)
from navigation_system.space.structure.space_types import (
    infer_space_type_from_texts,
    normalize_space_type,
    strip_space_type_variant_suffixes,
)
from navigation_system.vlm.contracts.schema import DIRECTION_CONFIG
from navigation_system.space.geometry.map_projection import RotatedMapProjector


class ThinkingViewRenderer:
    """Render and save the 12 annotated direction views used by the thinking model."""

    MODEL_CONTENT_WIDTH = THINKING_VIEW_MODEL_CONTENT_WIDTH
    THINKING_DETECTION_TOPK = CFG_THINKING_DETECTION_TOPK
    THINKING_DETECTION_GROUP_MAX_VIEWS = CFG_THINKING_DETECTION_GROUP_MAX_VIEWS
    THINKING_DETECTION_TRANSITION_TOTAL_MAX_VIEWS = CFG_THINKING_DETECTION_TRANSITION_TOTAL_MAX_VIEWS
    THINKING_DETECTION_OBJECT_TOTAL_MAX_VIEWS = CFG_THINKING_DETECTION_OBJECT_TOTAL_MAX_VIEWS
    CURRENT_AREA_OVERLAP_THRESHOLD_M = CFG_CURRENT_AREA_OVERLAP_THRESHOLD_M
    VIEW_HFOV_DEG = CFG_THINKING_VIEW_HFOV_DEG
    WAYPOINT_VISIBILITY_RADIUS_M = CFG_WAYPOINT_VISIBILITY_RADIUS_M
    WAYPOINT_VISIBILITY_SAMPLES = CFG_WAYPOINT_VISIBILITY_SAMPLES
    WAYPOINT_STRIP_MIN_DISTANCE_M = CFG_SPACE_AREA_CURRENT_WAYPOINT_MAX_DISTANCE_M
    SAME_OBJECT_BEARING_THRESHOLD_DEG = CFG_THINKING_SAME_OBJECT_BEARING_THRESHOLD_DEG
    SAME_OBJECT_DISTANCE_THRESHOLD_M = CFG_THINKING_SAME_OBJECT_DISTANCE_THRESHOLD_M
    SAME_OBJECT_DISTANCE_RATIO = CFG_THINKING_SAME_OBJECT_DISTANCE_RATIO
    TRANSITION_DETECTION_KEYWORDS = tuple(
        list(
            str(keyword).strip().lower()
            for keyword in CFG_LANDMARK_EDGE_DEPTH_KEYWORDS
            if str(keyword).strip()
        ) + ["hallway"]
    )

    @staticmethod
    def _is_known_area_label(area_label: str) -> bool:
        return str(area_label or "").strip().lower() not in {"", "unknown"}

    @classmethod
    def _split_area_label_links(cls, area_label: str) -> Tuple[str, List[str]]:
        text = str(area_label or "").strip()
        if not text:
            return "", []

        lower_text = text.lower()
        marker = " [links:"
        marker_idx = lower_text.find(marker)
        if marker_idx < 0:
            return cls._strip_visual_brackets(text), []

        clean_text = cls._strip_visual_brackets(text[:marker_idx].strip() or text)
        link_text = text[marker_idx + len(marker):].strip()
        if link_text.endswith("]"):
            link_text = link_text[:-1].strip()
        links = [
            cls._strip_visual_brackets(item.strip())
            for item in link_text.split(",")
            if item.strip()
        ]
        return clean_text, links

    @staticmethod
    def _strip_visual_brackets(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return cleaned
        cleaned = cleaned.replace("【", "[").replace("】", "]")
        cleaned = re.sub(r"\[\s*([^\[\]]+?)\s*\]", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

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
                (" (conf: ", dark),
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
        text = ThinkingViewRenderer._strip_visual_brackets(str(text or "").strip())
        if len(text) <= max_len:
            return text
        return text[: max(0, max_len - 2)].rstrip() + ".."

    @staticmethod
    def _title_case_space_type(text: str) -> str:
        return " ".join(word.capitalize() for word in str(text or "").split())

    @classmethod
    def _strip_space_waypoint_suffix(cls, text: str) -> str:
        cleaned = cls._strip_visual_brackets(strip_space_type_variant_suffixes(text) or text)
        cleaned = cleaned.strip()
        if not cleaned:
            return ""
        for separator in (" - ", " / ", " | "):
            if separator in cleaned:
                cleaned = cleaned.split(separator, 1)[0].strip()
        possessive_match = re.match(r"^(.+?)'s\b", cleaned, flags=re.IGNORECASE)
        if possessive_match:
            cleaned = possessive_match.group(1).strip()
        return cleaned

    @classmethod
    def _build_waypoint_center_tag_text(
        cls,
        area_label: str,
        description: str,
        display_text: str,
    ) -> str:
        clean_area_label, _connected_area_labels = cls._split_area_label_links(area_label)
        candidates = [
            cls._strip_space_waypoint_suffix(clean_area_label),
            cls._strip_space_waypoint_suffix(description),
            cls._strip_space_waypoint_suffix(display_text),
        ]
        for candidate in candidates:
            normalized_space_type = normalize_space_type(candidate)
            if str(normalized_space_type).strip().lower() != "unknown":
                return cls._title_case_space_type(normalized_space_type)
        inferred_space_type = infer_space_type_from_texts(candidates)
        if str(inferred_space_type).strip().lower() != "unknown":
            return cls._title_case_space_type(inferred_space_type)
        for candidate in candidates:
            lowered = str(candidate or "").strip().lower()
            if lowered and lowered not in {"unknown", "area", "room", "space", "zone", "section", "place", "location"}:
                return candidate
        return ""

    @classmethod
    def _build_waypoint_view_entries(
        cls,
        waypoint_info: Optional[tuple],
        waypoint_area_labels: Optional[List[str]],
        waypoint_floor_ids: Optional[List[int]],
        current_pose: Optional[np.ndarray],
        resolution_cm: float,
        current_space_area_label: str,
        full_map: Optional[np.ndarray],
        crop_offset: Optional[Tuple[int, int]],
        current_space_area_type: str = "",
        current_floor_id: int = 0,
        initial_waypoint_index: Optional[int] = 0,
    ) -> List[Dict[str, Any]]:
        if current_pose is None:
            return []

        if not waypoint_info:
            current_area_text = cls._strip_visual_brackets(
                str(current_space_area_label or "Unknown").strip() or "Unknown"
            )
            clean_current_area_text, connected_area_labels = cls._split_area_label_links(current_area_text)
            if not cls._is_known_area_label(clean_current_area_text):
                return []
            return [{
                "id": 0,
                "label": cls._short_text(clean_current_area_text, max_len=34),
                "display_text": clean_current_area_text,
                "description": clean_current_area_text,
                "area_label": current_area_text,
                "clean_area_label": clean_current_area_text,
                "connected_area_labels": connected_area_labels,
                "distance_m": 0.0,
                "relative_bearing_deg": 0.0,
                "snapped_relative_bearing_deg": 0.0,
                "view_angle_deg": 0.0,
                "is_last_visited": False,
                "is_current_area": True,
            }]

        waypoint_positions, waypoint_ids, waypoint_descriptions = waypoint_info
        if not waypoint_ids:
            current_area_text = str(current_space_area_label or "Unknown").strip() or "Unknown"
            clean_current_area_text, connected_area_labels = cls._split_area_label_links(current_area_text)
            if not cls._is_known_area_label(clean_current_area_text):
                return []
            return [{
                "id": 0,
                "label": cls._short_text(clean_current_area_text, max_len=34),
                "display_text": clean_current_area_text,
                "description": clean_current_area_text,
                "area_label": current_area_text,
                "clean_area_label": clean_current_area_text,
                "connected_area_labels": connected_area_labels,
                "distance_m": 0.0,
                "relative_bearing_deg": 0.0,
                "snapped_relative_bearing_deg": 0.0,
                "view_angle_deg": 0.0,
                "is_last_visited": False,
                "is_current_area": True,
            }]

        area_labels = list(waypoint_area_labels or [])
        floor_ids = [
            int(waypoint_floor_ids[index]) if waypoint_floor_ids and index < len(waypoint_floor_ids) else int(current_floor_id)
            for index in range(len(waypoint_ids))
        ]
        current_floor_global_indices = [
            index
            for index, floor_id in enumerate(floor_ids)
            if int(floor_id) == int(current_floor_id)
        ]
        current_floor_index_map = {
            global_index: local_index
            for local_index, global_index in enumerate(current_floor_global_indices)
        }
        current_floor_positions = [waypoint_positions[index] for index in current_floor_global_indices]
        current_floor_ids = [waypoint_ids[index] for index in current_floor_global_indices]
        current_floor_descriptions = [
            waypoint_descriptions[index] if index < len(waypoint_descriptions) else ""
            for index in current_floor_global_indices
        ]
        current_floor_area_labels = [
            area_labels[index] if index < len(area_labels) else ""
            for index in current_floor_global_indices
        ]
        current_floor_initial_index = (
            current_floor_index_map.get(int(initial_waypoint_index))
            if initial_waypoint_index is not None
            else None
        )

        curr_x_m, curr_y_m, curr_orientation_deg = [float(v) for v in current_pose[:3]]
        resolved_current_area_text, current_area_anchor_index = resolve_display_current_area(
            waypoint_positions=current_floor_positions,
            waypoint_area_labels=current_floor_area_labels,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            current_space_area_label=current_space_area_label,
            current_space_area_type=current_space_area_type,
            waypoint_descriptions=current_floor_descriptions,
            full_map=full_map,
            crop_offset=crop_offset,
        )
        display_indices = select_display_waypoint_indices(
            waypoint_positions=current_floor_positions,
            waypoint_ids=current_floor_ids,
            waypoint_descriptions=current_floor_descriptions,
            waypoint_area_labels=current_floor_area_labels,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            full_map=full_map,
            crop_offset=crop_offset,
            initial_waypoint_index=current_floor_initial_index,
            skip_current_overlap=True,
        )
        last_visited_local_index = resolve_last_distinct_waypoint_index(
            waypoint_positions=current_floor_positions,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            full_map=full_map,
            crop_offset=crop_offset,
        )
        if (
            last_visited_local_index is not None
            and 0 <= int(last_visited_local_index) < len(current_floor_ids)
            and int(last_visited_local_index) not in display_indices
        ):
            display_indices.append(int(last_visited_local_index))
            display_indices = sorted(set(display_indices))
        entries: List[Dict[str, Any]] = []

        for local_index in display_indices:
            global_index = current_floor_global_indices[local_index]
            wp_id = current_floor_ids[local_index]
            wp_desc = current_floor_descriptions[local_index]
            wp_py, wp_px = current_floor_positions[local_index]
            wp_x_m = float(wp_px) * float(resolution_cm) / 100.0
            wp_y_m = float(wp_py) * float(resolution_cm) / 100.0
            dx = wp_x_m - curr_x_m
            dy = wp_y_m - curr_y_m
            distance_m = float(math.hypot(dx, dy))
            absolute_angle_deg = float(math.degrees(math.atan2(dy, dx)))
            relative_bearing_deg = float(curr_orientation_deg - absolute_angle_deg)
            snapped_relative_bearing_deg = float(snap_relative_bearing(relative_bearing_deg))
            view_angle_deg = cls._normalize_angle_deg(-snapped_relative_bearing_deg)

            area_label = str(current_floor_area_labels[local_index] if local_index < len(current_floor_area_labels) else "").strip()
            clean_area_label, connected_area_labels = cls._split_area_label_links(area_label)
            description = cls._strip_visual_brackets(str(wp_desc or "").strip())
            display_text = cls._strip_visual_brackets(
                description or clean_area_label or f"WP#{wp_id}"
            )

            entries.append({
                "id": int(wp_id),
                "label": cls._short_text(display_text, max_len=34),
                "display_text": display_text,
                "description": description,
                "area_label": area_label,
                "clean_area_label": clean_area_label,
                "connected_area_labels": connected_area_labels,
                "world_py": int(wp_py),
                "world_px": int(wp_px),
                "distance_m": distance_m,
                "relative_bearing_deg": relative_bearing_deg,
                "snapped_relative_bearing_deg": snapped_relative_bearing_deg,
                "view_angle_deg": view_angle_deg,
                "center_tag_text": cls._build_waypoint_center_tag_text(
                    area_label=area_label,
                    description=description,
                    display_text=display_text,
                ),
                "is_last_visited": local_index == last_visited_local_index,
                "is_task_initial_position": (
                    current_floor_initial_index is not None
                    and int(local_index) == int(current_floor_initial_index)
                ),
            })

        current_area_text = cls._strip_visual_brackets(
            str(resolved_current_area_text or current_space_area_label or "Unknown").strip() or "Unknown"
        )
        clean_current_area_text, connected_current_area_labels = cls._split_area_label_links(current_area_text)
        current_area_view_angle = 0.0
        current_area_relative_bearing = 0.0
        current_area_snapped_bearing = 0.0
        if current_area_anchor_index is not None and current_area_anchor_index < len(current_floor_positions):
            anchor_py, anchor_px = current_floor_positions[current_area_anchor_index]
            anchor_x_m = float(anchor_px) * float(resolution_cm) / 100.0
            anchor_y_m = float(anchor_py) * float(resolution_cm) / 100.0
            dx = anchor_x_m - curr_x_m
            dy = anchor_y_m - curr_y_m
            absolute_angle_deg = float(math.degrees(math.atan2(dy, dx)))
            current_area_relative_bearing = float(curr_orientation_deg - absolute_angle_deg)
            current_area_snapped_bearing = float(snap_relative_bearing(current_area_relative_bearing))
            current_area_view_angle = cls._normalize_angle_deg(-current_area_snapped_bearing)
        elif entries:
            last_entry = entries[-1]
            current_area_view_angle = float(last_entry["view_angle_deg"])
            current_area_relative_bearing = float(last_entry["relative_bearing_deg"])
            current_area_snapped_bearing = float(last_entry.get("snapped_relative_bearing_deg", 0.0))

        if cls._is_known_area_label(clean_current_area_text) and (current_area_anchor_index is not None or not waypoint_ids):
            entries.append({
                "id": 0,
                "label": cls._short_text(clean_current_area_text, max_len=34),
                "display_text": clean_current_area_text,
                "description": clean_current_area_text,
                "area_label": current_area_text,
                "clean_area_label": clean_current_area_text,
                "connected_area_labels": connected_current_area_labels,
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
    def _assigned_view_angle_for_waypoint(
        cls,
        entry: Dict[str, Any],
        view_angles_deg: List[float],
    ) -> Optional[float]:
        if not view_angles_deg:
            return None
        view_angle_deg = cls._normalize_angle_deg(float(entry.get("view_angle_deg", 0.0)))
        nearest = cls._nearest_view_angle(view_angle_deg, view_angles_deg)
        return float(nearest) if nearest is not None else None

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
                updated_entry = dict(entry)
                updated_entry["is_connected_to_current"] = True
                assigned_view_angle = cls._assigned_view_angle_for_waypoint(updated_entry, view_angles_deg)
                if assigned_view_angle is not None:
                    updated_entry["assigned_view_angle"] = float(assigned_view_angle)
                filtered_entries.append(updated_entry)
                continue

            is_connected_to_current = False
            if obstacle_mask is not None and projector is not None and current_pose is not None:
                is_connected_to_current = cls._has_visible_waypoint_ray(
                    entry=entry,
                    obstacle_mask=obstacle_mask,
                    projector=projector,
                    current_pose=current_pose,
                    resolution_cm=resolution_cm,
                )
            if not is_connected_to_current:
                continue

            updated_entry = dict(entry)
            updated_entry["is_connected_to_current"] = True
            assigned_view_angle = cls._assigned_view_angle_for_waypoint(updated_entry, view_angles_deg)
            if assigned_view_angle is None:
                continue
            updated_entry["assigned_view_angle"] = float(assigned_view_angle)
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
            assigned_angle = entry.get("assigned_view_angle")
            if assigned_angle is None:
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

    @staticmethod
    def _distance_sort_value(distance_m: Optional[float]) -> float:
        try:
            value = float(distance_m)
        except (TypeError, ValueError):
            return float("inf")
        return value if np.isfinite(value) else float("inf")

    @staticmethod
    def _distance_text(distance_m: Optional[float]) -> str:
        try:
            value = float(distance_m)
        except (TypeError, ValueError):
            return "unknown"
        if not np.isfinite(value):
            return "unknown"
        return f"{value:.1f}m"

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
            distance_m = entry.get("distance_m")
            confidence = float(entry.get("confidence", 0.0))
            sort_key = (cls._distance_sort_value(distance_m), 0.0, -confidence)
            lines.append(
                LandmarkStripLine(
                    distance_m=cls._distance_sort_value(distance_m),
                    confidence=confidence,
                    priority=0,
                    sort_key=sort_key,
                    segments=(
                        LandmarkStripSegment("landmark: ", prefix_color),
                        LandmarkStripSegment(cls._short_text(f"{name}{suffix}", max_len=30), value_color),
                        LandmarkStripSegment(f"  {cls._distance_text(distance_m)}", value_color),
                        LandmarkStripSegment("  conf: ", prefix_color),
                        LandmarkStripSegment(f"{confidence:.3f}", value_color),
                    ),
                )
            )

        for entry in waypoint_entries:
            if bool(entry.get("is_current_area")):
                distance_m = entry.get("distance_m", 0.0)
                connected_area_labels = [
                    str(label).strip()
                    for label in list(entry.get("connected_area_labels") or [])
                    if str(label).strip()
                ]
                display_text = str(
                    entry.get("display_text")
                    or entry.get("clean_area_label")
                    or entry.get("label")
                    or "Unknown"
                ).strip() or "Unknown"
                segments = [
                    LandmarkStripSegment("your current area: ", prefix_color),
                    LandmarkStripSegment(display_text, value_color),
                ]
                if connected_area_labels:
                    segments.extend((
                        LandmarkStripSegment("  connects: ", prefix_color),
                        LandmarkStripSegment(", ".join(connected_area_labels), value_color),
                    ))
                sort_key = (cls._distance_sort_value(distance_m), 1.0, 0.0)
                lines.append(
                    LandmarkStripLine(
                        distance_m=cls._distance_sort_value(distance_m),
                        confidence=0.0,
                        priority=1,
                        sort_key=sort_key,
                        segments=tuple(segments),
                    )
                )
                continue

            if not bool(entry.get("is_connected_to_current")):
                continue
            try:
                waypoint_distance_m = float(entry.get("distance_m"))
            except (TypeError, ValueError):
                continue
            if (
                not np.isfinite(waypoint_distance_m)
                or waypoint_distance_m <= float(cls.WAYPOINT_STRIP_MIN_DISTANCE_M) + 1e-6
            ):
                continue
            note_parts: List[str] = []
            if bool(entry.get("is_last_visited")) and not bool(entry.get("is_task_initial_position")):
                note_parts.append("LAST POSITION")
            if bool(entry.get("is_task_initial_position")):
                note_parts.append("INITIAL POSITION")
            note = f"  <- {' | '.join(note_parts)}" if note_parts else ""
            connected_area_labels = [
                str(label).strip()
                for label in list(entry.get("connected_area_labels") or [])
                if str(label).strip()
            ]
            waypoint_display_text = str(
                entry.get("display_text")
                or entry.get("description")
                or entry.get("clean_area_label")
                or entry.get("label")
                or "Unknown"
            ).strip() or "Unknown"
            waypoint_text = f"WP#{int(entry.get('id', 0))} {waypoint_display_text}".strip()
            distance_m = waypoint_distance_m
            sort_key = (cls._distance_sort_value(distance_m), 1.0, 0.0)
            segments = [
                LandmarkStripSegment("space waypoint: ", prefix_color),
                LandmarkStripSegment(waypoint_text, value_color),
                LandmarkStripSegment(f"  {cls._distance_text(distance_m)}", value_color),
            ]
            if connected_area_labels:
                segments.extend((
                    LandmarkStripSegment("  connects: ", prefix_color),
                    LandmarkStripSegment(", ".join(connected_area_labels), value_color),
                ))
            if note:
                segments.append(LandmarkStripSegment(note, prefix_color))
            lines.append(
                LandmarkStripLine(
                    distance_m=cls._distance_sort_value(distance_m),
                    confidence=0.0,
                    priority=1,
                    sort_key=sort_key,
                    segments=tuple(segments),
                )
            )

        lines.sort(key=lambda line: line.sort_key)
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

    @classmethod
    def _remove_transition_like_detections(cls, detections, labels: List[str]):
        if detections is None or getattr(detections, "xyxy", None) is None:
            return detections, labels or []

        keep_indices: List[int] = []
        for idx, label_text in enumerate(labels or []):
            name, _confidence = cls._parse_detection_label(label_text)
            if cls._is_transition_like_detection(name):
                continue
            keep_indices.append(int(idx))
        return cls._filter_detection_payload(detections, labels or [], keep_indices)

    @staticmethod
    def _parse_detection_label(label: str) -> Tuple[str, float]:
        parts = str(label or "").split()
        if not parts:
            return "unknown", 0.0
        if len(parts) == 1:
            return parts[0], 0.0
        try:
            confidence = float(parts[-1])
            name = " ".join(parts[:-1]).strip() or parts[0]
            return name, confidence
        except ValueError:
            return " ".join(parts).strip(), 0.0

    @staticmethod
    def _normalize_detection_name(name: str) -> str:
        return " ".join(str(name or "").strip().lower().split())

    @classmethod
    def _is_transition_like_detection(cls, name: str) -> bool:
        normalized_name = cls._normalize_detection_name(name)
        if not normalized_name:
            return False
        normalized_text = f" {normalized_name.replace('-', ' ').replace('/', ' ')} "
        tokens = set(normalized_text.split())
        for keyword in cls.TRANSITION_DETECTION_KEYWORDS:
            keyword_text = str(keyword).strip().lower()
            if not keyword_text:
                continue
            keyword_tokens = keyword_text.split()
            if len(keyword_tokens) == 1:
                if keyword_tokens[0] in tokens:
                    return True
                continue
            if f" {' '.join(keyword_tokens)} " in normalized_text:
                return True
        return False

    @classmethod
    def _cross_view_detection_family_key(cls, name: str) -> str:
        normalized_name = cls._normalize_detection_name(name)
        if cls._is_transition_like_detection(normalized_name):
            return "__transition_like__"
        return normalized_name or "__unknown__"

    @classmethod
    def _cross_view_total_limit(cls, name: str) -> int:
        if cls._is_transition_like_detection(name):
            return int(max(1, cls.THINKING_DETECTION_TRANSITION_TOTAL_MAX_VIEWS))
        return int(max(1, cls.THINKING_DETECTION_OBJECT_TOTAL_MAX_VIEWS))

    @classmethod
    def _estimate_detection_distance_m(
        cls,
        detections: Any,
        det_idx: int,
        depth_meters: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
    ) -> Optional[float]:
        if depth_meters is None:
            return None

        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), depth_meters.shape[1] - 1))
        x2 = max(0, min(int(x2), depth_meters.shape[1]))
        y1 = max(0, min(int(y1), depth_meters.shape[0] - 1))
        y2 = max(0, min(int(y2), depth_meters.shape[0]))
        if x2 <= x1 or y2 <= y1:
            return None

        depth_region = depth_meters[y1:y2, x1:x2]
        valid_depths = depth_region[depth_region > 0.05]
        if getattr(detections, "mask", None) is not None and det_idx < len(detections.mask):
            det_mask = np.asarray(detections.mask[det_idx]).astype(bool)
            if det_mask.shape[:2] == depth_meters.shape[:2]:
                sample_mask = _build_outer_ring_sampling_mask(
                    det_mask,
                    depth_meters,
                    min_depth=0.05,
                )
                ys, xs = _sample_random_mask_coords(sample_mask)
                if ys.size > 0:
                    sampled_depths = depth_meters[ys, xs].astype(np.float32)
                    sampled_depths = sampled_depths[np.isfinite(sampled_depths) & (sampled_depths > 0.05)]
                    if sampled_depths.size > 0:
                        valid_depths = sampled_depths

        if valid_depths.size == 0:
            return None
        return float(np.median(valid_depths))

    @classmethod
    def _build_cross_view_detection_candidates(
        cls,
        payloads: List[Tuple[Any, List[str], np.ndarray, Optional[np.ndarray], float, str]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for view_idx, (detections, _labels, _image, _depth, _angle, _name) in enumerate(payloads):
            if detections is None or getattr(detections, "xyxy", None) is None:
                continue
            labels = _labels or []
            image_w = max(1, int(_image.shape[1]))
            for det_idx, bbox_values in enumerate(np.asarray(detections.xyxy, dtype=np.float32)):
                bbox = tuple(int(round(v)) for v in bbox_values.tolist())
                label_text = labels[det_idx] if det_idx < len(labels) else f"object_{det_idx}"
                name, confidence = cls._parse_detection_label(label_text)
                bbox_center_x = 0.5 * (float(bbox[0]) + float(bbox[2]))
                rel_angle_deg = (0.5 - (bbox_center_x / float(image_w))) * cls.VIEW_HFOV_DEG
                global_bearing_deg = cls._normalize_angle_deg(float(_angle) + rel_angle_deg)
                distance_m = cls._estimate_detection_distance_m(
                    detections=detections,
                    det_idx=det_idx,
                    depth_meters=_depth,
                    bbox=bbox,
                )
                bbox_area = max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))
                candidates.append({
                    "view_idx": int(view_idx),
                    "det_idx": int(det_idx),
                    "name": str(name or "unknown"),
                    "confidence": float(confidence),
                    "bbox": bbox,
                    "distance_m": float(distance_m) if distance_m is not None else None,
                    "global_bearing_deg": float(global_bearing_deg),
                    "bbox_area": float(bbox_area),
                })
        return candidates

    @classmethod
    def _distance_compatible(cls, dist_a: Optional[float], dist_b: Optional[float]) -> bool:
        if dist_a is None or dist_b is None:
            return True
        distance_gap = abs(float(dist_a) - float(dist_b))
        if distance_gap <= cls.SAME_OBJECT_DISTANCE_THRESHOLD_M:
            return True
        max_dist = max(float(dist_a), float(dist_b), 1e-6)
        return distance_gap <= cls.SAME_OBJECT_DISTANCE_RATIO * max_dist

    @classmethod
    def _match_cross_view_group(
        cls,
        candidate: Dict[str, Any],
        groups: List[Dict[str, Any]],
    ) -> Optional[int]:
        best_group_idx = None
        best_cost = None
        for group_idx, group in enumerate(groups):
            if str(group.get("name")) != str(candidate.get("name")):
                continue
            bearing_gap = abs(cls._angle_delta_deg(
                float(candidate.get("global_bearing_deg", 0.0)),
                float(group.get("bearing_center_deg", 0.0)),
            ))
            if bearing_gap > cls.SAME_OBJECT_BEARING_THRESHOLD_DEG:
                continue
            if not cls._distance_compatible(
                candidate.get("distance_m"),
                group.get("distance_center_m"),
            ):
                continue
            cost = bearing_gap
            if candidate.get("distance_m") is not None and group.get("distance_center_m") is not None:
                cost += abs(float(candidate["distance_m"]) - float(group["distance_center_m"]))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_group_idx = group_idx
        return best_group_idx

    @classmethod
    def _select_grouped_detection_indices(
        cls,
        payloads: List[Tuple[Any, List[str], np.ndarray, Optional[np.ndarray], float, str]],
        topk: int,
    ) -> Dict[int, List[int]]:
        if topk <= 0:
            return {}

        candidates = cls._build_cross_view_detection_candidates(payloads)
        if not candidates:
            return {}

        candidates.sort(
            key=lambda item: (
                -float(item.get("confidence", 0.0)),
                float(item.get("distance_m")) if item.get("distance_m") is not None else float("inf"),
                -float(item.get("bbox_area", 0.0)),
            )
        )

        groups: List[Dict[str, Any]] = []
        for candidate in candidates:
            match_idx = cls._match_cross_view_group(candidate, groups)
            if match_idx is None:
                groups.append({
                    "name": str(candidate.get("name", "unknown")),
                    "bearing_center_deg": float(candidate.get("global_bearing_deg", 0.0)),
                    "distance_center_m": candidate.get("distance_m"),
                    "best_confidence": float(candidate.get("confidence", 0.0)),
                    "nearest_distance_m": candidate.get("distance_m"),
                    "members": [candidate],
                })
                continue

            group = groups[match_idx]
            group["members"].append(candidate)
            members = group["members"]
            group["bearing_center_deg"] = float(np.mean([
                float(item.get("global_bearing_deg", 0.0)) for item in members
            ]))
            valid_distances = [
                float(item["distance_m"]) for item in members if item.get("distance_m") is not None
            ]
            group["distance_center_m"] = (
                float(np.mean(valid_distances)) if valid_distances else None
            )
            group["best_confidence"] = max(
                float(group.get("best_confidence", 0.0)),
                float(candidate.get("confidence", 0.0)),
            )
            if candidate.get("distance_m") is not None:
                if group.get("nearest_distance_m") is None:
                    group["nearest_distance_m"] = float(candidate["distance_m"])
                else:
                    group["nearest_distance_m"] = min(
                        float(group["nearest_distance_m"]),
                        float(candidate["distance_m"]),
                    )

        groups.sort(
            key=lambda item: (
                -float(item.get("best_confidence", 0.0)),
                float(item.get("nearest_distance_m")) if item.get("nearest_distance_m") is not None else float("inf"),
            )
        )

        keep_by_view: Dict[int, List[int]] = {}
        kept_family_counts: Dict[str, int] = {}
        for group in groups[:max(1, int(topk))]:
            group_name = str(group.get("name", "unknown"))
            family_key = cls._cross_view_detection_family_key(group_name)
            family_kept_count = int(kept_family_counts.get(family_key, 0))
            family_limit = cls._cross_view_total_limit(group_name)
            remaining_family_budget = max(0, family_limit - family_kept_count)
            if remaining_family_budget <= 0:
                continue

            members = sorted(
                group.get("members", []),
                key=lambda item: (
                    float(item.get("distance_m")) if item.get("distance_m") is not None else float("inf"),
                    -float(item.get("confidence", 0.0)),
                    float(item.get("bbox_area", 0.0)) * -1.0,
                ),
            )
            used_views = set()
            kept_count = 0
            # Preserve the original same-object grouping and per-group 3-view cap.
            # Only the 12-view detection retention budget changes here:
            # transition-like detections can occupy up to 4 views, regular objects up to 2.
            group_limit = min(int(max(1, cls.THINKING_DETECTION_GROUP_MAX_VIEWS)), remaining_family_budget)
            for member in members:
                view_idx = int(member.get("view_idx", -1))
                det_idx = int(member.get("det_idx", -1))
                if view_idx < 0 or det_idx < 0 or view_idx in used_views:
                    continue
                keep_by_view.setdefault(view_idx, []).append(det_idx)
                used_views.add(view_idx)
                kept_count += 1
                if kept_count >= group_limit:
                    break
            if kept_count > 0:
                kept_family_counts[family_key] = family_kept_count + kept_count
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
        lookaround_detection_payloads: Optional[List[Tuple[Any, List[str], Any]]] = None,
        detection_topk: int = THINKING_DETECTION_TOPK,
    ) -> Tuple[List[str], List[str]]:
        os.makedirs(directions_dir, exist_ok=True)
        rendered_views = self.render_direction_views(
            phase=phase,
            lookaround_images=lookaround_images,
            lookaround_depths=lookaround_depths,
            lookaround_detection_payloads=lookaround_detection_payloads,
            landmark_classes=landmark_classes,
            detect_landmarks_fn=detect_landmarks_fn,
            render_detection_fn=render_detection_fn,
            draw_distance_fn=draw_distance_fn,
            distance_lookup=distance_lookup,
            waypoint_info=waypoint_info,
            waypoint_area_labels=waypoint_area_labels,
            waypoint_floor_ids=waypoint_floor_ids,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            current_space_area_label=current_space_area_label,
            full_map=full_map,
            crop_offset=crop_offset,
            waypoint_angle_deg=waypoint_angle_deg,
            draw_waypoints_fn=draw_waypoints_fn,
            current_floor_id=current_floor_id,
            initial_waypoint_index=initial_waypoint_index,
            detection_topk=detection_topk,
        )
        direction_paths: List[str] = []
        direction_names: List[str] = []
        for view in rendered_views:
            angle = int(view["angle"])
            labeled_image = view["image"]
            direction_filename = f"{phase}_direction_{angle:03d}.png"
            direction_path = os.path.join(directions_dir, direction_filename)
            cv2.imwrite(direction_path, labeled_image)
            direction_paths.append(direction_path)
            direction_names.append(str(view["direction_name"]))
        return direction_paths, direction_names

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
        detection_topk: int = THINKING_DETECTION_TOPK,
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

        keep_by_view = self._select_grouped_detection_indices(view_payloads, topk=detection_topk)

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
            image = resize_image_to_width(image, self.MODEL_CONTENT_WIDTH)

            _h, width = image.shape[:2]
            top_text = f"{direction_name} | Obstacle: {dist_str}"
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
            if marker_entry is not None:
                image = draw_waypoints_fn(image, marker_entry)
            bottom_lines = self._build_bottom_strip_lines(
                visible_entries_meta=visible_entries_meta,
                waypoint_entries=view_waypoint_entries,
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

            rendered_views.append({
                "phase": str(phase),
                "angle": int(angle),
                "direction_name": str(direction_name),
                "image": labeled_image,
            })

        return rendered_views
