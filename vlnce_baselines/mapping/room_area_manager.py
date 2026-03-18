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

    CONNECTOR_ROOM_TYPES = {"hallway", "entryway"}
    MAX_CONNECTED_AREAS = 3
    MAX_CONNECTION_DISTANCE_M = 3.0
    MAX_SAME_TYPE_WAYPOINT_MERGE_DISTANCE_M = 5.0
    SAMPLE_PIXELS_PER_AREA = 9
    CURRENT_AREA_WAYPOINT_MAX_DISTANCE_M = 1.25

    def __init__(self, map_shape: Tuple[int, int], resolution: int = 5):
        self.map_shape = map_shape
        self.resolution = resolution
        self.reset()

    def reset(self) -> None:
        self.room_area_records: List[Dict[str, Any]] = []
        self.room_area_counter = 0
        self.label_aliases: Dict[str, str] = {}
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

        projector = self._build_projector(full_map, full_pose, crop_offset)
        obstacle_mask = (
            np.asarray(full_map[0] > 0.5, dtype=bool)
            if full_map is not None and projector is not None
            else None
        )

        overlapping_records = [
            record for record in self.room_area_records
            if record["room_key"] == room_key
            and self._records_match_new_area(
                existing_record=record,
                new_pixels=world_pixels,
                new_center_world_px=(int(pixel_y), int(pixel_x)),
                projector=projector,
                obstacle_mask=obstacle_mask,
            )
        ]

        if overlapping_records:
            merged_record = min(
                overlapping_records,
                key=lambda record: (int(record["variant"]), int(record["id"])),
            )
            merged_record["pixels"].update(world_pixels)
            merged_record.setdefault("waypoint_points", set()).add((int(pixel_y), int(pixel_x)))
            merged_record["center_world_px"] = (int(pixel_y), int(pixel_x))
            merged_record["description"] = description

            for extra_record in overlapping_records:
                if extra_record is merged_record:
                    continue
                merged_record["pixels"].update(extra_record["pixels"])
                merged_record.setdefault("waypoint_points", set()).update(
                    set(extra_record.get("waypoint_points", set()) or set())
                )
                self._register_label_alias(
                    old_label=str(extra_record["label"]),
                    new_label=str(merged_record["label"]),
                )
                self.room_area_records.remove(extra_record)

            self.current_room_area_label = str(merged_record["label"])
            self.current_room_area_type = str(merged_record["room_type"])
            self._consolidate_same_type_records(
                obstacle_mask=obstacle_mask,
                projector=projector,
            )
            return self._resolve_label_alias(str(merged_record["label"]))

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
            "waypoint_points": {(int(pixel_y), int(pixel_x))},
            "description": description,
        }
        self.room_area_records.append(record)
        self.current_room_area_label = label
        self.current_room_area_type = room_type
        self._consolidate_same_type_records(
            obstacle_mask=obstacle_mask,
            projector=projector,
        )
        return self._resolve_label_alias(label)

    def build_layer(
        self,
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
        waypoint_positions: Optional[Sequence[Tuple[int, int]]] = None,
        waypoint_area_labels: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if full_map is None:
            self._set_unknown_current_area()
            return np.zeros(self.map_shape, dtype=np.int32), []

        obstacle_mask = np.asarray(full_map[0] > 0.5, dtype=bool)
        self._consolidate_same_type_records(
            obstacle_mask=obstacle_mask,
            projector=self._build_projector(full_map, full_pose, crop_offset),
        )
        self._maintain_current_area_with_pose(
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
            waypoint_positions=waypoint_positions,
            waypoint_area_labels=waypoint_area_labels,
        )

        h_map, w_map = full_map.shape[1], full_map.shape[2]
        layer = np.zeros((h_map, w_map), dtype=np.int32)
        best_distance = np.full((h_map, w_map), np.inf, dtype=np.float32)
        projector = self._build_projector(full_map, full_pose, crop_offset)
        if projector is None:
            self._set_unknown_current_area()
            return layer, []

        area_records: List[Dict[str, Any]] = []
        self._refresh_connection_metadata(full_map, full_pose, crop_offset)
        for record in self.room_area_records:
            center_py, center_px = record["center_world_px"]
            area_records.append({
                "id": int(record["id"]),
                "label": str(record["label"]),
                "display_label": str(record.get("display_label", record["label"])),
                "room_type": str(record["room_type"]),
                "variant": int(record["variant"]),
                "center_world_px": (int(center_py), int(center_px)),
                "connected_area_labels": list(record.get("connected_area_labels", [])),
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

        self._set_current_room_area_from_layer(
            layer=layer,
            full_pose=full_pose,
            projector=projector,
            waypoint_positions=waypoint_positions,
            waypoint_area_labels=waypoint_area_labels,
        )
        return layer, area_records

    def _consolidate_same_type_records(
        self,
        obstacle_mask: Optional[np.ndarray] = None,
        projector: Optional[RotatedMapProjector] = None,
    ) -> None:
        """Merge any same-type area records that now overlap or touch."""
        changed = True
        while changed:
            changed = False
            for idx, record in enumerate(list(self.room_area_records)):
                for other in self.room_area_records[idx + 1:]:
                    if record.get("room_key") != other.get("room_key"):
                        continue
                    should_merge = (
                        self._pixel_sets_overlap(record.get("pixels", set()), other.get("pixels", set()))
                        or self._pixel_sets_are_adjacent(record.get("pixels", set()), other.get("pixels", set()))
                    )
                    if (
                        not should_merge
                        and obstacle_mask is not None
                        and projector is not None
                    ):
                        should_merge = self._records_have_waypoint_connection(
                            record_a=record,
                            record_b=other,
                            obstacle_mask=obstacle_mask,
                            projector=projector,
                            max_distance_m=self.MAX_SAME_TYPE_WAYPOINT_MERGE_DISTANCE_M,
                        )
                    if not should_merge:
                        continue
                    primary, secondary = sorted(
                        (record, other),
                        key=lambda item: (int(item.get("variant", 0)), int(item.get("id", 0))),
                    )
                    self._merge_room_area_records(primary, secondary)
                    changed = True
                    break
                if changed:
                    break

        resolved_current = self._resolve_label_alias(self.current_room_area_label)
        if resolved_current and resolved_current != "Unknown":
            current_record = next(
                (
                    item for item in self.room_area_records
                    if str(item.get("label", "")) == resolved_current
                ),
                None,
            )
            if current_record is not None:
                self._set_current_area_from_record(current_record)

    def _merge_room_area_records(
        self,
        primary: Dict[str, Any],
        secondary: Dict[str, Any],
    ) -> None:
        primary["pixels"].update(secondary.get("pixels", set()))
        primary.setdefault("waypoint_points", set()).update(
            set(secondary.get("waypoint_points", set()) or set())
        )
        primary["description"] = str(primary.get("description") or secondary.get("description") or "")
        primary["center_world_px"] = self._compute_record_center_from_pixels(primary)
        self._register_label_alias(
            old_label=str(secondary.get("label", "")),
            new_label=str(primary.get("label", "")),
        )
        if secondary in self.room_area_records:
            self.room_area_records.remove(secondary)

    @staticmethod
    def _compute_record_center_from_pixels(record: Dict[str, Any]) -> Tuple[int, int]:
        pixels = list(record.get("pixels", set()) or [])
        if not pixels:
            center = record.get("center_world_px", (0, 0))
            return int(center[0]), int(center[1])

        rows = np.asarray([pixel[0] for pixel in pixels], dtype=np.float32)
        cols = np.asarray([pixel[1] for pixel in pixels], dtype=np.float32)
        return int(round(float(rows.mean()))), int(round(float(cols.mean())))

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

    def _records_match_new_area(
        self,
        existing_record: Dict[str, Any],
        new_pixels: Set[Tuple[int, int]],
        new_center_world_px: Tuple[int, int],
        projector: Optional[RotatedMapProjector],
        obstacle_mask: Optional[np.ndarray],
    ) -> bool:
        existing_pixels = existing_record.get("pixels", set())
        if self._pixel_sets_overlap(existing_pixels, new_pixels):
            return True
        if self._pixel_sets_are_adjacent(existing_pixels, new_pixels):
            return True
        if projector is None or obstacle_mask is None:
            return False

        return self._records_have_waypoint_connection(
            record_a=existing_record,
            record_b={
                "center_world_px": new_center_world_px,
                "waypoint_points": {tuple(new_center_world_px)},
            },
            obstacle_mask=obstacle_mask,
            projector=projector,
            max_distance_m=self.MAX_SAME_TYPE_WAYPOINT_MERGE_DISTANCE_M,
        )

    @staticmethod
    def _pixel_sets_are_adjacent(
        pixels_a: Set[Tuple[int, int]],
        pixels_b: Set[Tuple[int, int]],
    ) -> bool:
        if not pixels_a or not pixels_b:
            return False
        if len(pixels_a) > len(pixels_b):
            pixels_a, pixels_b = pixels_b, pixels_a
        for row, col in pixels_a:
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    if (int(row + d_row), int(col + d_col)) in pixels_b:
                        return True
        return False

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
        world_pixels.add((int(pixel_y), int(pixel_x)))
        filtered_pixels = self._filter_obstacle_pixels(
            world_pixels=world_pixels,
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
        )
        if filtered_pixels:
            return filtered_pixels
        if self._world_pixel_is_traversible(
            pixel_y=int(pixel_y),
            pixel_x=int(pixel_x),
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
        ):
            return {(int(pixel_y), int(pixel_x))}
        return fallback

    def _set_current_room_area_from_layer(
        self,
        layer: np.ndarray,
        full_pose: Optional[Sequence[float]],
        projector: RotatedMapProjector,
        waypoint_positions: Optional[Sequence[Tuple[int, int]]] = None,
        waypoint_area_labels: Optional[Sequence[str]] = None,
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
                    if current_record is not None and self._record_has_nearby_area_waypoint(
                        record=current_record,
                        pixel_y=curr_py,
                        pixel_x=curr_px,
                        waypoint_positions=waypoint_positions,
                        waypoint_area_labels=waypoint_area_labels,
                    ):
                        self.current_room_area_label = str(current_record["label"])
                        self.current_room_area_type = str(current_record["room_type"])
                        return

        containing_record = self._find_current_record_from_waypoints(
            pixel_y=curr_py,
            pixel_x=curr_px,
            waypoint_positions=waypoint_positions,
            waypoint_area_labels=waypoint_area_labels,
        )
        if containing_record is not None:
            self._set_current_area_from_record(containing_record)
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
        waypoint_positions: Optional[Sequence[Tuple[int, int]]] = None,
        waypoint_area_labels: Optional[Sequence[str]] = None,
    ) -> None:
        if full_map is None or full_pose is None or not self.room_area_records:
            self._set_unknown_current_area()
            return

        curr_py = int(round(float(full_pose[1]) * 100.0 / float(self.resolution)))
        curr_px = int(round(float(full_pose[0]) * 100.0 / float(self.resolution)))

        containing_record = self._find_current_record_from_waypoints(
            pixel_y=curr_py,
            pixel_x=curr_px,
            waypoint_positions=waypoint_positions,
            waypoint_area_labels=waypoint_area_labels,
        )
        if containing_record is not None:
            self._set_current_area_from_record(containing_record)
            return

        self._set_unknown_current_area()

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

    def _find_current_record_from_waypoints(
        self,
        pixel_y: int,
        pixel_x: int,
        waypoint_positions: Optional[Sequence[Tuple[int, int]]] = None,
        waypoint_area_labels: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        containing_record = self._find_record_containing_pixel(pixel_y, pixel_x)
        if containing_record is None:
            return None
        if self._record_has_nearby_area_waypoint(
            record=containing_record,
            pixel_y=pixel_y,
            pixel_x=pixel_x,
            waypoint_positions=waypoint_positions,
            waypoint_area_labels=waypoint_area_labels,
        ):
            return containing_record
        return None

    def _record_has_nearby_area_waypoint(
        self,
        record: Dict[str, Any],
        pixel_y: int,
        pixel_x: int,
        waypoint_positions: Optional[Sequence[Tuple[int, int]]],
        waypoint_area_labels: Optional[Sequence[str]],
    ) -> bool:
        if not waypoint_positions:
            return False

        record_label = self._resolve_label_alias(str(record.get("label", "")).strip())
        if not record_label:
            return False

        max_distance_px = (
            self.CURRENT_AREA_WAYPOINT_MAX_DISTANCE_M * 100.0
        ) / float(self.resolution)
        area_labels = list(waypoint_area_labels or [])
        record_pixels = set(record.get("pixels", set()) or [])

        for index, waypoint_pos in enumerate(waypoint_positions):
            if waypoint_pos is None:
                continue
            waypoint_label = self._resolve_label_alias(
                str(area_labels[index]).strip() if index < len(area_labels) else ""
            )
            if waypoint_label != record_label:
                continue

            wp_py, wp_px = int(waypoint_pos[0]), int(waypoint_pos[1])
            if (wp_py, wp_px) not in record_pixels:
                continue

            distance_px = float(np.hypot(float(pixel_y) - float(wp_py), float(pixel_x) - float(wp_px)))
            if distance_px <= max_distance_px + 1e-6:
                return True
        return False

    def _set_current_area_from_record(self, record: Dict[str, Any]) -> None:
        self.current_room_area_label = str(record.get("label", "Unknown") or "Unknown")
        self.current_room_area_type = str(record.get("room_type", "Unknown") or "Unknown")

    def get_display_label(self, label: str) -> str:
        target_label = self._resolve_label_alias(str(label or "").strip())
        if not target_label:
            return "Unknown"
        if target_label == "Unknown":
            return "Unknown"
        record = next(
            (item for item in self.room_area_records if str(item.get("label", "")) == target_label),
            None,
        )
        if record is None:
            return target_label
        return str(record.get("display_label", target_label) or target_label)

    def _resolve_label_alias(self, label: str) -> str:
        target_label = str(label or "").strip()
        if not target_label or target_label == "Unknown":
            return "Unknown" if target_label == "Unknown" else target_label

        visited: Set[str] = set()
        while target_label in self.label_aliases and target_label not in visited:
            visited.add(target_label)
            target_label = str(self.label_aliases[target_label])
        return target_label

    def _register_label_alias(self, old_label: str, new_label: str) -> None:
        src_label = str(old_label or "").strip()
        dst_label = self._resolve_label_alias(str(new_label or "").strip())
        if not src_label or not dst_label or src_label == dst_label:
            return

        for alias, target in list(self.label_aliases.items()):
            if self._resolve_label_alias(target) == src_label:
                self.label_aliases[alias] = dst_label
        self.label_aliases[src_label] = dst_label

    def _refresh_connection_metadata(
        self,
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
    ) -> None:
        for record in self.room_area_records:
            record["connected_area_labels"] = []
            record["display_label"] = str(record.get("label", "Unknown") or "Unknown")

        if full_map is None or full_pose is None or crop_offset is None:
            return

        projector = self._build_projector(full_map, full_pose, crop_offset)
        if projector is None:
            return

        obstacle_mask = np.asarray(full_map[0] > 0.5, dtype=bool)
        for record in self.room_area_records:
            if str(record.get("room_type", "")) not in self.CONNECTOR_ROOM_TYPES:
                continue

            connected = self._compute_connected_areas_for_record(
                record=record,
                obstacle_mask=obstacle_mask,
                projector=projector,
            )
            connected_labels = [str(item.get("label", "")) for item in connected[: self.MAX_CONNECTED_AREAS]]
            record["connected_area_labels"] = connected_labels
            if connected_labels:
                record["display_label"] = (
                    f"{record['label']} [links: {', '.join(connected_labels[: self.MAX_CONNECTED_AREAS])}]"
                )

    def _compute_connected_areas_for_record(
        self,
        record: Dict[str, Any],
        obstacle_mask: np.ndarray,
        projector: RotatedMapProjector,
    ) -> List[Dict[str, Any]]:
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        max_distance_px = (self.MAX_CONNECTION_DISTANCE_M * 100.0) / float(self.resolution)

        for other in self.room_area_records:
            if other is record:
                continue
            if self._records_are_adjacent(record, other):
                candidates.append((0.0, other))
                continue

            center_dist_px = float(np.hypot(
                float(record["center_world_px"][0]) - float(other["center_world_px"][0]),
                float(record["center_world_px"][1]) - float(other["center_world_px"][1]),
            ))
            if center_dist_px > max_distance_px:
                continue
            if self._records_have_clear_connection(record, other, obstacle_mask, projector):
                candidates.append((center_dist_px, other))

        candidates.sort(key=lambda item: (float(item[0]), str(item[1].get("label", ""))))
        unique_records: List[Dict[str, Any]] = []
        seen_labels: Set[str] = set()
        for _distance_px, other in candidates:
            label = str(other.get("label", ""))
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            unique_records.append(other)
        return unique_records

    @staticmethod
    def _records_are_adjacent(record_a: Dict[str, Any], record_b: Dict[str, Any]) -> bool:
        return RoomAreaManager._pixel_sets_are_adjacent(
            record_a.get("pixels", set()),
            record_b.get("pixels", set()),
        )

    def _records_have_clear_connection(
        self,
        record_a: Dict[str, Any],
        record_b: Dict[str, Any],
        obstacle_mask: np.ndarray,
        projector: RotatedMapProjector,
    ) -> bool:
        max_distance_px = (self.MAX_CONNECTION_DISTANCE_M * 100.0) / float(self.resolution)
        return self._pixel_sets_have_clear_connection(
            pixels_a=record_a.get("pixels", set()),
            center_a=tuple(record_a.get("center_world_px", (0, 0))),
            pixels_b=record_b.get("pixels", set()),
            center_b=tuple(record_b.get("center_world_px", (0, 0))),
            obstacle_mask=obstacle_mask,
            projector=projector,
            max_distance_px=max_distance_px,
        )

    def _records_have_waypoint_connection(
        self,
        record_a: Dict[str, Any],
        record_b: Dict[str, Any],
        obstacle_mask: np.ndarray,
        projector: RotatedMapProjector,
        max_distance_m: float,
    ) -> bool:
        max_distance_px = (float(max_distance_m) * 100.0) / float(self.resolution)
        waypoint_points_a = self._record_waypoint_points(record_a)
        waypoint_points_b = self._record_waypoint_points(record_b)
        for start_world in waypoint_points_a:
            for end_world in waypoint_points_b:
                if float(np.hypot(
                    float(start_world[0]) - float(end_world[0]),
                    float(start_world[1]) - float(end_world[1]),
                )) > max_distance_px + 1e-6:
                    continue
                if self._world_line_is_clear(
                    obstacle_mask=obstacle_mask,
                    projector=projector,
                    start_world=start_world,
                    end_world=end_world,
                ):
                    return True
        return False

    @staticmethod
    def _record_waypoint_points(record: Dict[str, Any]) -> List[Tuple[int, int]]:
        waypoint_points = list(record.get("waypoint_points", set()) or [])
        if waypoint_points:
            return [
                (int(point[0]), int(point[1]))
                for point in waypoint_points
            ]
        center = record.get("center_world_px", (0, 0))
        return [(int(center[0]), int(center[1]))]

    def _sample_record_pixels(self, record: Dict[str, Any]) -> List[Tuple[int, int]]:
        return self._sample_pixels(
            pixels=record.get("pixels", set()),
            center_world_px=tuple(record.get("center_world_px", (0, 0))),
        )

    def _sample_pixels(
        self,
        pixels: Set[Tuple[int, int]],
        center_world_px: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        pixel_list = list(pixels)
        if not pixel_list:
            return [(int(center_world_px[0]), int(center_world_px[1]))]

        sampled: List[Tuple[int, int]] = []
        center = tuple(center_world_px if center_world_px else pixel_list[0])
        sampled.append((int(center[0]), int(center[1])))
        anchors = [
            min(pixel_list, key=lambda p: (p[0], p[1])),
            max(pixel_list, key=lambda p: (p[0], p[1])),
            min(pixel_list, key=lambda p: (p[1], p[0])),
            max(pixel_list, key=lambda p: (p[1], p[0])),
            min(pixel_list, key=lambda p: (p[0] + p[1], p[0])),
            max(pixel_list, key=lambda p: (p[0] + p[1], p[0])),
            min(pixel_list, key=lambda p: (p[0] - p[1], p[0])),
            max(pixel_list, key=lambda p: (p[0] - p[1], p[0])),
        ]
        for row, col in anchors:
            sampled.append((int(row), int(col)))

        deduped: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for pixel in sampled:
            if pixel in seen:
                continue
            seen.add(pixel)
            deduped.append(pixel)
            if len(deduped) >= self.SAMPLE_PIXELS_PER_AREA:
                break
        return deduped

    def _pixel_sets_have_clear_connection(
        self,
        pixels_a: Set[Tuple[int, int]],
        center_a: Tuple[int, int],
        pixels_b: Set[Tuple[int, int]],
        center_b: Tuple[int, int],
        obstacle_mask: np.ndarray,
        projector: RotatedMapProjector,
        max_distance_px: float,
    ) -> bool:
        sample_pixels_a = self._sample_pixels(pixels_a, center_a)
        sample_pixels_b = self._sample_pixels(pixels_b, center_b)
        for start_world in sample_pixels_a:
            for end_world in sample_pixels_b:
                if float(np.hypot(
                    float(start_world[0]) - float(end_world[0]),
                    float(start_world[1]) - float(end_world[1]),
                )) > max_distance_px:
                    continue
                if self._world_line_is_clear(
                    obstacle_mask=obstacle_mask,
                    projector=projector,
                    start_world=start_world,
                    end_world=end_world,
                ):
                    return True
        return False

    @staticmethod
    def _line_is_clear(
        obstacle_mask: np.ndarray,
        start_row: float,
        start_col: float,
        end_row: float,
        end_col: float,
    ) -> bool:
        steps = max(int(np.ceil(max(abs(end_row - start_row), abs(end_col - start_col)))), 1)
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

    def _world_line_is_clear(
        self,
        obstacle_mask: np.ndarray,
        projector: RotatedMapProjector,
        start_world: Tuple[int, int],
        end_world: Tuple[int, int],
    ) -> bool:
        start_rot = projector.world_to_rotated_pixel(float(start_world[0]), float(start_world[1]))
        end_rot = projector.world_to_rotated_pixel(float(end_world[0]), float(end_world[1]))
        if start_rot is None or end_rot is None:
            return False
        return self._line_is_clear(
            obstacle_mask=obstacle_mask,
            start_row=float(start_rot[0]),
            start_col=float(start_rot[1]),
            end_row=float(end_rot[0]),
            end_col=float(end_rot[1]),
        )

    def _filter_obstacle_pixels(
        self,
        world_pixels: Set[Tuple[int, int]],
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
    ) -> Set[Tuple[int, int]]:
        if not world_pixels or full_map is None:
            return set(world_pixels)

        projector = self._build_projector(full_map, full_pose, crop_offset)
        if projector is None:
            return set(world_pixels)

        filtered: Set[Tuple[int, int]] = set()
        for pixel_y, pixel_x in world_pixels:
            if self._world_pixel_is_traversible(
                pixel_y=int(pixel_y),
                pixel_x=int(pixel_x),
                full_map=full_map,
                full_pose=full_pose,
                crop_offset=crop_offset,
            ):
                filtered.add((int(pixel_y), int(pixel_x)))
        return filtered

    def _world_pixel_is_traversible(
        self,
        pixel_y: int,
        pixel_x: int,
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
    ) -> bool:
        projector = self._build_projector(full_map, full_pose, crop_offset)
        if full_map is None or projector is None:
            return True
        rotated = projector.world_to_rotated_pixel(float(pixel_y), float(pixel_x))
        if rotated is None:
            return False
        row = int(round(rotated[0]))
        col = int(round(rotated[1]))
        if not (0 <= row < full_map.shape[1] and 0 <= col < full_map.shape[2]):
            return False
        obstacle = bool(full_map[0, row, col] > 0.5)
        explored = bool(full_map[1, row, col] > 0.5)
        return explored and not obstacle
