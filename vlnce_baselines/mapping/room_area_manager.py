"""
Persistent room-area manager built on top of the world semantic map.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from vlnce_baselines.mapping.space_types import normalize_space_type
from vlnce_baselines.visualization.map_projection import RotatedMapProjector


class RoomAreaManager:
    """Track room-type areas, variants, and the current area label on the world map."""

    STEP_MAINTENANCE_RADIUS_M = 2.0

    def __init__(self, map_shape: Tuple[int, int], resolution: int = 5):
        self.map_shape = map_shape
        self.resolution = resolution
        self.reset()

    def reset(self) -> None:
        self.room_area_records: List[Dict[str, Any]] = []
        self.room_area_counter = 0
        self.current_room_area_label = "Unknown"
        self.current_room_area_type = "Unknown"

    def update_from_waypoint(
        self,
        description: str,
        pixel_y: int,
        pixel_x: int,
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
    ) -> str:
        room_type = self._parse_room_type(description)
        if room_type == "Unknown":
            return "Unknown"
        room_key = self._room_type_key(room_type)
        world_pixels = self._compute_room_area_world_pixels(
            pixel_y=pixel_y,
            pixel_x=pixel_x,
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
        )

        overlapping_records = [
            record for record in self.room_area_records
            if record["room_key"] == room_key
            and self._pixel_sets_overlap(record["pixels"], world_pixels)
        ]

        if overlapping_records:
            merged_record = overlapping_records[0]
            merged_record["pixels"].update(world_pixels)
            merged_record["center_world_px"] = (int(pixel_y), int(pixel_x))
            merged_record["description"] = description

            if len(overlapping_records) > 1:
                for extra_record in overlapping_records[1:]:
                    merged_record["pixels"].update(extra_record["pixels"])
                    self.room_area_records.remove(extra_record)

            self.current_room_area_label = str(merged_record["label"])
            self.current_room_area_type = str(merged_record["room_type"])
            return str(merged_record["label"])

        existing_variants = [
            int(record["variant"])
            for record in self.room_area_records
            if record["room_key"] == room_key
        ]
        variant = (max(existing_variants) + 1) if existing_variants else 1
        self.room_area_counter += 1
        label = self._room_label(room_type, variant)
        record = {
            "id": self.room_area_counter,
            "label": label,
            "room_type": room_type,
            "room_key": room_key,
            "variant": variant,
            "center_world_px": (int(pixel_y), int(pixel_x)),
            "pixels": set(world_pixels),
            "description": description,
        }
        self.room_area_records.append(record)
        self.current_room_area_label = label
        self.current_room_area_type = room_type
        return label

    def build_layer(
        self,
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if full_map is None:
            self._set_unknown_current_area()
            return np.zeros(self.map_shape, dtype=np.int32), []

        self._maintain_current_area_with_pose(
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
        )

        h_map, w_map = full_map.shape[1], full_map.shape[2]
        layer = np.zeros((h_map, w_map), dtype=np.int32)
        best_distance = np.full((h_map, w_map), np.inf, dtype=np.float32)
        projector = self._build_projector(full_map, full_pose, crop_offset)
        if projector is None:
            self._set_unknown_current_area()
            return layer, []

        area_records: List[Dict[str, Any]] = []
        for record in self.room_area_records:
            center_py, center_px = record["center_world_px"]
            area_records.append({
                "id": int(record["id"]),
                "label": str(record["label"]),
                "room_type": str(record["room_type"]),
                "variant": int(record["variant"]),
                "center_world_px": (int(center_py), int(center_px)),
            })

            for world_py, world_px in record["pixels"]:
                rotated = projector.world_to_rotated_pixel(world_py, world_px)
                if rotated is None:
                    continue
                row = int(round(rotated[0]))
                col = int(round(rotated[1]))
                if not (0 <= row < h_map and 0 <= col < w_map):
                    continue
                dist = float(np.hypot(world_py - center_py, world_px - center_px))
                if dist < best_distance[row, col]:
                    best_distance[row, col] = dist
                    layer[row, col] = int(record["id"])

        self._set_current_room_area_from_layer(layer, full_pose, projector)
        return layer, area_records

    @staticmethod
    def _parse_room_type(description: str) -> str:
        text = (description or "").strip()
        if not text:
            return "Unknown"

        for sep in ("|", "-", "Nearby", "Connected"):
            if sep in text:
                text = text.split(sep)[0].strip()
        return normalize_space_type(" ".join(text.split()))

    @staticmethod
    def _room_type_key(room_type: str) -> str:
        return "".join(ch.lower() for ch in room_type if ch.isalnum())

    @staticmethod
    def _room_label(room_type: str, variant: int) -> str:
        words = [word.capitalize() for word in room_type.split() if word]
        base = "".join(words) if words else "Unknown"
        return f"{base}{variant}"

    @staticmethod
    def _pixel_sets_overlap(pixels_a: Set[Tuple[int, int]], pixels_b: Set[Tuple[int, int]]) -> bool:
        if not pixels_a or not pixels_b:
            return False
        if len(pixels_a) > len(pixels_b):
            pixels_a, pixels_b = pixels_b, pixels_a
        return any(pixel in pixels_b for pixel in pixels_a)

    def _build_projector(
        self,
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
    ) -> Optional[RotatedMapProjector]:
        if full_map is None or full_pose is None or crop_offset is None:
            return None
        return RotatedMapProjector(
            map_h=full_map.shape[1],
            map_w=full_map.shape[2],
            crop_offset=crop_offset,
            agent_orientation_deg=float(full_pose[2]),
        )

    def _find_room_area_start(
        self,
        traversible: np.ndarray,
        center_row: int,
        center_col: int,
    ) -> Optional[Tuple[int, int]]:
        h_map, w_map = traversible.shape
        if 0 <= center_row < h_map and 0 <= center_col < w_map and traversible[center_row, center_col]:
            return center_row, center_col

        radius = int(round(100.0 / float(self.resolution)))
        ys, xs = np.nonzero(traversible)
        if ys.size == 0:
            return None
        d2 = (ys - center_row) ** 2 + (xs - center_col) ** 2
        within = d2 <= radius ** 2
        if not np.any(within):
            return None
        indices = np.where(within)[0]
        best_idx = int(indices[np.argmin(d2[within])])
        return int(ys[best_idx]), int(xs[best_idx])

    def _compute_room_area_world_pixels(
        self,
        pixel_y: int,
        pixel_x: int,
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
        max_radius_m: float = 2.0,
    ) -> Set[Tuple[int, int]]:
        fallback = {(int(pixel_y), int(pixel_x))}
        projector = self._build_projector(full_map, full_pose, crop_offset)
        if full_map is None or projector is None:
            return fallback

        obstacle_mask = full_map[0] > 0.5
        explored_mask = full_map[1] > 0.5
        traversible = explored_mask & (~obstacle_mask)
        center_rot = projector.world_to_rotated_pixel(pixel_y, pixel_x)
        if center_rot is None:
            return fallback

        center_row = int(round(center_rot[0]))
        center_col = int(round(center_rot[1]))
        start = self._find_room_area_start(traversible, center_row, center_col)
        if start is None:
            return fallback

        max_radius_px = int(round((max_radius_m * 100.0) / float(self.resolution)))
        h_map, w_map = traversible.shape
        visited = np.zeros((h_map, w_map), dtype=bool)
        queue = deque([start])
        visited[start[0], start[1]] = True
        selected_rotated: List[Tuple[int, int]] = []

        while queue:
            row, col = queue.popleft()
            if ((row - start[0]) ** 2 + (col - start[1]) ** 2) > max_radius_px ** 2:
                continue
            selected_rotated.append((row, col))

            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    if d_row == 0 and d_col == 0:
                        continue
                    next_row = row + d_row
                    next_col = col + d_col
                    if not (0 <= next_row < h_map and 0 <= next_col < w_map):
                        continue
                    if visited[next_row, next_col] or not traversible[next_row, next_col]:
                        continue
                    if ((next_row - start[0]) ** 2 + (next_col - start[1]) ** 2) > max_radius_px ** 2:
                        continue
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))

        world_pixels: Set[Tuple[int, int]] = set()
        for row, col in selected_rotated:
            world = projector.rotated_to_world_pixel(row, col)
            if world is None:
                continue
            world_pixels.add((int(round(world[0])), int(round(world[1]))))
        return world_pixels or fallback

    def _set_current_room_area_from_layer(
        self,
        layer: np.ndarray,
        full_pose: Optional[Sequence[float]],
        projector: RotatedMapProjector,
    ) -> None:
        if full_pose is None or not self.room_area_records:
            self._set_unknown_current_area()
            return

        curr_py = int(round(float(full_pose[1]) * 100.0 / float(self.resolution)))
        curr_px = int(round(float(full_pose[0]) * 100.0 / float(self.resolution)))
        rotated = projector.world_to_rotated_pixel(curr_py, curr_px)
        if rotated is not None:
            row = int(round(rotated[0]))
            col = int(round(rotated[1]))
            if 0 <= row < layer.shape[0] and 0 <= col < layer.shape[1]:
                area_id = int(layer[row, col])
                if area_id > 0:
                    current_record = next(
                        (record for record in self.room_area_records if int(record["id"]) == area_id),
                        None,
                    )
                    if current_record is not None:
                        self.current_room_area_label = str(current_record["label"])
                        self.current_room_area_type = str(current_record["room_type"])
                        return

        self._set_unknown_current_area()

    def _set_unknown_current_area(self) -> None:
        self.current_room_area_label = "Unknown"
        self.current_room_area_type = "Unknown"

    def _maintain_current_area_with_pose(
        self,
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
    ) -> None:
        if full_map is None or full_pose is None or not self.room_area_records:
            self._set_unknown_current_area()
            return

        curr_py = int(round(float(full_pose[1]) * 100.0 / float(self.resolution)))
        curr_px = int(round(float(full_pose[0]) * 100.0 / float(self.resolution)))

        containing_record = self._find_record_containing_pixel(curr_py, curr_px)
        if containing_record is not None:
            self._set_current_area_from_record(containing_record)
            return

        candidate_record = self._select_record_for_step_maintenance(curr_py, curr_px)
        if candidate_record is None:
            self._set_unknown_current_area()
            return

        expanded_pixels = self._compute_room_area_world_pixels(
            pixel_y=curr_py,
            pixel_x=curr_px,
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
            max_radius_m=self.STEP_MAINTENANCE_RADIUS_M,
        )
        candidate_record["pixels"].update(expanded_pixels)
        self._set_current_area_from_record(candidate_record)

    def _find_record_containing_pixel(
        self,
        pixel_y: int,
        pixel_x: int,
    ) -> Optional[Dict[str, Any]]:
        target_pixel = (int(pixel_y), int(pixel_x))
        for record in reversed(self.room_area_records):
            if target_pixel in record["pixels"]:
                return record
        return None

    def _select_record_for_step_maintenance(
        self,
        pixel_y: int,
        pixel_x: int,
    ) -> Optional[Dict[str, Any]]:
        max_distance_px = (self.STEP_MAINTENANCE_RADIUS_M * 100.0) / float(self.resolution)
        max_distance_sq = max_distance_px ** 2

        preferred_records: List[Dict[str, Any]] = []
        current_record = self._find_record_by_label(self.current_room_area_label)
        if current_record is not None:
            preferred_records.append(current_record)
        if self.room_area_records:
            latest_record = self.room_area_records[-1]
            if latest_record not in preferred_records:
                preferred_records.append(latest_record)

        for record in preferred_records:
            if self._record_min_distance_sq(record, pixel_y, pixel_x, max_distance_sq) <= max_distance_sq:
                return record

        best_record = None
        best_distance_sq = max_distance_sq
        for record in reversed(self.room_area_records):
            distance_sq = self._record_min_distance_sq(record, pixel_y, pixel_x, best_distance_sq)
            if distance_sq <= best_distance_sq:
                best_record = record
                best_distance_sq = distance_sq
        return best_record

    def _find_record_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        target_label = str(label or "").strip()
        if not target_label or target_label == "Unknown":
            return None
        for record in reversed(self.room_area_records):
            if str(record.get("label", "")) == target_label:
                return record
        return None

    @staticmethod
    def _record_min_distance_sq(
        record: Dict[str, Any],
        pixel_y: int,
        pixel_x: int,
        early_stop_sq: float,
    ) -> float:
        best_distance_sq = np.inf
        for world_py, world_px in record.get("pixels", set()):
            distance_sq = float((world_py - pixel_y) ** 2 + (world_px - pixel_x) ** 2)
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                if best_distance_sq <= early_stop_sq:
                    return best_distance_sq
        return best_distance_sq

    def _set_current_area_from_record(self, record: Dict[str, Any]) -> None:
        self.current_room_area_label = str(record.get("label", "Unknown") or "Unknown")
        self.current_room_area_type = str(record.get("room_type", "Unknown") or "Unknown")
