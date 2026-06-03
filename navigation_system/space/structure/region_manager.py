"""
Persistent region manager built on top of the world semantic map.
"""

from collections import deque
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from navigation_system.config.core.params.spatial import (
    SPACE_AREA_CONNECTOR_TYPES,
    SPACE_AREA_CURRENT_WAYPOINT_MAX_DISTANCE_M,
    SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M,
    SPACE_AREA_MAX_CONNECTED_AREAS,
    SPACE_AREA_MAX_CONNECTION_DISTANCE_M,
    SPACE_AREA_NARROW_PASSAGE_CLEARANCE_M,
    SPACE_AREA_REGION_RADIUS_M,
    SPACE_AREA_MAX_SAME_TYPE_WAYPOINT_MERGE_DISTANCE_M,
    SPACE_AREA_SAME_TYPE_CONNECTOR_SPLIT_DISTANCE_M,
    SPACE_AREA_SAME_TYPE_MERGE_MIN_CLEARANCE_M,
    SPACE_AREA_SAMPLE_PIXELS_PER_AREA,
    WAYPOINT_VISIBILITY_RADIUS_M,
    WAYPOINT_VISIBILITY_SAMPLES,
)
from navigation_system.space.geometry.connectivity import (
    build_bounded_geodesic_distance_field,
    query_world_distance_from_field_m,
)
from navigation_system.space.structure.space_types import (
    normalize_space_type,
    strip_space_type_label_variant_suffix,
)
from navigation_system.space.geometry.map_projection import RotatedMapProjector


class RegionManager:
    """Track region types, variants, and the current region label on the world map."""

    CONNECTOR_SPACE_TYPES = SPACE_AREA_CONNECTOR_TYPES
    MAX_CONNECTED_AREAS = SPACE_AREA_MAX_CONNECTED_AREAS
    MAX_CONNECTION_DISTANCE_M = SPACE_AREA_MAX_CONNECTION_DISTANCE_M
    MAX_SAME_TYPE_WAYPOINT_MERGE_DISTANCE_M = SPACE_AREA_MAX_SAME_TYPE_WAYPOINT_MERGE_DISTANCE_M
    SAME_TYPE_CONNECTOR_SPLIT_DISTANCE_M = SPACE_AREA_SAME_TYPE_CONNECTOR_SPLIT_DISTANCE_M
    SAMPLE_PIXELS_PER_AREA = SPACE_AREA_SAMPLE_PIXELS_PER_AREA
    CURRENT_AREA_WAYPOINT_MAX_DISTANCE_M = SPACE_AREA_CURRENT_WAYPOINT_MAX_DISTANCE_M
    CURRENT_AREA_INITIAL_WAYPOINT_MAX_DISTANCE_M = SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M
    NARROW_PASSAGE_CLEARANCE_M = SPACE_AREA_NARROW_PASSAGE_CLEARANCE_M
    SAME_TYPE_MERGE_MIN_CLEARANCE_M = SPACE_AREA_SAME_TYPE_MERGE_MIN_CLEARANCE_M
    MIN_RENDERABLE_AREA_PIXELS = 24
    MIN_SEED_AREA_RADIUS_M = 0.60
    REGION_RADIUS_M = SPACE_AREA_REGION_RADIUS_M

    def __init__(self, map_shape: Tuple[int, int], resolution: int = 5):
        self.map_shape = map_shape
        self.resolution = resolution
        self.region_radius_m = self._resolve_region_radius_m()
        self.MAX_CONNECTION_DISTANCE_M = self._resolve_positive_float_env(
            "SPACEVLN_SPACE_AREA_MAX_CONNECTION_DISTANCE_M",
            self.MAX_CONNECTION_DISTANCE_M,
        )
        self.MAX_SAME_TYPE_WAYPOINT_MERGE_DISTANCE_M = self._resolve_positive_float_env(
            "SPACEVLN_SPACE_AREA_MAX_SAME_TYPE_WAYPOINT_MERGE_DISTANCE_M",
            self.MAX_SAME_TYPE_WAYPOINT_MERGE_DISTANCE_M,
        )
        self.reset()

    @classmethod
    def _resolve_positive_float_env(cls, name: str, default: float, *, min_value: float = 0.1) -> float:
        raw_value = str(os.getenv(name, "") or "").strip()
        if raw_value:
            try:
                return max(float(min_value), float(raw_value))
            except ValueError:
                pass
        return float(default)

    @classmethod
    def _resolve_region_radius_m(cls) -> float:
        return cls._resolve_positive_float_env(
            "SPACEVLN_SPACE_AREA_REGION_RADIUS_M",
            cls.REGION_RADIUS_M,
        )

    def reset(self) -> None:
        self.region_records: List[Dict[str, Any]] = []
        self.region_counter = 0
        self.label_aliases: Dict[str, str] = {}
        self.current_region_label = "Unknown"
        self.current_region_type = "Unknown"

    def export_state(self) -> Dict[str, Any]:
        records: List[Dict[str, Any]] = []
        for record in self.region_records:
            center_world_px = record.get("center_world_px", (0, 0))
            records.append({
                "id": int(record.get("id", 0) or 0),
                "label": str(record.get("label", "")),
                "space_type": str(record.get("space_type", "Unknown") or "Unknown"),
                "space_key": str(record.get("space_key", "")),
                "variant": int(record.get("variant", 0) or 0),
                "center_world_px": (int(center_world_px[0]), int(center_world_px[1])),
                "pixels": [
                    (int(pixel_y), int(pixel_x))
                    for pixel_y, pixel_x in set(record.get("pixels", set()) or set())
                ],
                "waypoint_points": [
                    (int(pixel_y), int(pixel_x))
                    for pixel_y, pixel_x in set(record.get("waypoint_points", set()) or set())
                ],
                "description": str(record.get("description", "")),
                "connected_area_labels": [str(item) for item in list(record.get("connected_area_labels", []) or [])],
                "display_label": str(record.get("display_label", record.get("label", ""))),
            })
        return {
            "region_records": records,
            "region_counter": int(self.region_counter),
            "label_aliases": {str(key): str(value) for key, value in self.label_aliases.items()},
            "current_region_label": str(self.current_region_label or "Unknown"),
            "current_region_type": str(self.current_region_type or "Unknown"),
        }

    def import_state(self, state: Optional[Dict[str, Any]]) -> None:
        self.reset()
        if not state:
            return
        self.region_counter = int(state.get("region_counter", 0) or 0)
        self.label_aliases = {
            str(key): str(value)
            for key, value in dict(state.get("label_aliases", {}) or {}).items()
        }
        self.current_region_label = str(state.get("current_region_label", "Unknown") or "Unknown")
        self.current_region_type = str(state.get("current_region_type", "Unknown") or "Unknown")
        restored_records: List[Dict[str, Any]] = []
        for raw_record in list(state.get("region_records", []) or []):
            center_world_px = raw_record.get("center_world_px", (0, 0))
            restored_records.append({
                "id": int(raw_record.get("id", 0) or 0),
                "label": str(raw_record.get("label", "")),
                "space_type": str(raw_record.get("space_type", "Unknown") or "Unknown"),
                "space_key": str(raw_record.get("space_key", "")),
                "variant": int(raw_record.get("variant", 0) or 0),
                "center_world_px": (int(center_world_px[0]), int(center_world_px[1])),
                "pixels": {
                    (int(pixel_y), int(pixel_x))
                    for pixel_y, pixel_x in list(raw_record.get("pixels", []) or [])
                    if pixel_y is not None and pixel_x is not None
                },
                "waypoint_points": {
                    (int(pixel_y), int(pixel_x))
                    for pixel_y, pixel_x in list(raw_record.get("waypoint_points", []) or [])
                    if pixel_y is not None and pixel_x is not None
                },
                "description": str(raw_record.get("description", "")),
                "connected_area_labels": [str(item) for item in list(raw_record.get("connected_area_labels", []) or [])],
                "display_label": str(raw_record.get("display_label", raw_record.get("label", ""))),
            })
        self.region_records = restored_records

    def update_from_waypoint(
        self,
        description: str,
        pixel_y: int,
        pixel_x: int,
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
    ) -> str:
        space_type = self._parse_space_type(description)
        if space_type == "Unknown":
            return "Unknown"
        space_key = self._space_type_key(space_type)
        world_pixels = self._compute_region_world_pixels(
            pixel_y=pixel_y,
            pixel_x=pixel_x,
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
            max_radius_m=float(self.region_radius_m),
        )
        if not world_pixels:
            # Keep a minimal seed area at the waypoint pose so the parsed label
            # remains part of the managed area graph and can merge only with the
            # same normalized space category later.
            world_pixels = {(int(pixel_y), int(pixel_x))}

        projector = self._build_projector(full_map, full_pose, crop_offset)
        obstacle_mask = (
            np.asarray(full_map[0] > 0.5, dtype=bool)
            if full_map is not None and projector is not None
            else None
        )
        clearance_map = (
            self._build_clearance_map(obstacle_mask)
            if obstacle_mask is not None
            else None
        )
        world_pixels = self._filter_area_pixels_for_space_type(
            world_pixels=world_pixels,
            space_type=space_type,
            projector=projector,
            clearance_map=clearance_map,
        )

        overlapping_records = [
            record for record in self.region_records
            if record["space_key"] == space_key
            and self._records_match_new_area(
                existing_record=record,
                new_pixels=world_pixels,
                new_center_world_px=(int(pixel_y), int(pixel_x)),
                projector=projector,
                obstacle_mask=obstacle_mask,
                clearance_map=clearance_map,
            )
        ]

        if overlapping_records:
            merged_record = min(
                overlapping_records,
                key=lambda record: (int(record["variant"]), int(record["id"])),
            )
            merged_record["pixels"].update(world_pixels)
            merged_record.setdefault("waypoint_points", set()).add((int(pixel_y), int(pixel_x)))
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
                self.region_records.remove(extra_record)
            merged_record["center_world_px"] = self._compute_record_center_from_pixels(merged_record)

            self.current_region_label = str(merged_record["label"])
            self.current_region_type = str(merged_record["space_type"])
            self._consolidate_same_type_records(
                obstacle_mask=obstacle_mask,
                projector=projector,
                clearance_map=clearance_map,
            )
            return self._resolve_label_alias(str(merged_record["label"]))

        existing_variants = [
            int(record["variant"])
            for record in self.region_records
            if record["space_key"] == space_key
        ]
        variant = (max(existing_variants) + 1) if existing_variants else 1
        self.region_counter += 1
        label = self._region_label(space_type, variant)
        record = {
            "id": self.region_counter,
            "label": label,
            "space_type": space_type,
            "space_key": space_key,
            "variant": variant,
            "center_world_px": (int(pixel_y), int(pixel_x)),
            "pixels": set(world_pixels),
            "waypoint_points": {(int(pixel_y), int(pixel_x))},
            "description": description,
        }
        record["center_world_px"] = self._compute_record_center_from_pixels(record)
        self.region_records.append(record)
        self.current_region_label = label
        self.current_region_type = space_type
        self._consolidate_same_type_records(
            obstacle_mask=obstacle_mask,
            projector=projector,
            clearance_map=clearance_map,
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
        projector = self._build_projector(full_map, full_pose, crop_offset)
        self._prune_records_against_obstacles(
            full_map=full_map,
            projector=projector,
        )
        self._consolidate_same_type_records(
            obstacle_mask=obstacle_mask,
            projector=projector,
            clearance_map=self._build_clearance_map(obstacle_mask),
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
        if projector is None:
            self._set_unknown_current_area()
            return layer, []

        traversible = ~obstacle_mask
        area_records: List[Dict[str, Any]] = []
        self._refresh_connection_metadata(full_map, full_pose, crop_offset)
        for record in self.region_records:
            record_pixels = set(record.get("pixels", set()) or set())
            if not record_pixels:
                continue

            center_py, center_px = record["center_world_px"]
            record_id = int(record["id"])
            area_records.append({
                "id": record_id,
                "label": str(record["label"]),
                "display_label": str(record.get("display_label", record["label"])),
                "space_type": str(record["space_type"]),
                "variant": int(record["variant"]),
                "center_world_px": (int(center_py), int(center_px)),
                "connected_area_labels": list(record.get("connected_area_labels", [])),
            })

            projected_pixel_count = 0
            for world_py, world_px in record_pixels:
                rotated = projector.world_to_rotated_pixel(world_py, world_px)
                if rotated is None:
                    continue
                row = int(round(rotated[0]))
                col = int(round(rotated[1]))
                if not (0 <= row < h_map and 0 <= col < w_map):
                    continue
                dist = self._distance_to_record_waypoints(
                    record=record,
                    pixel_y=int(world_py),
                    pixel_x=int(world_px),
                )
                if dist < best_distance[row, col]:
                    if layer[row, col] != record_id:
                        projected_pixel_count += 1
                    best_distance[row, col] = dist
                    layer[row, col] = record_id

            if projected_pixel_count < self.MIN_RENDERABLE_AREA_PIXELS:
                self._paint_record_seed_region(
                    layer=layer,
                    best_distance=best_distance,
                    record_id=record_id,
                    center_world_px=(int(center_py), int(center_px)),
                    traversible=traversible,
                    projector=projector,
                )

        self._set_current_region_from_layer(
            layer=layer,
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
            projector=projector,
            waypoint_positions=waypoint_positions,
            waypoint_area_labels=waypoint_area_labels,
        )
        return layer, area_records

    def _prune_records_against_obstacles(
        self,
        full_map: np.ndarray,
        projector: Optional[RotatedMapProjector],
    ) -> None:
        """Shrink persisted area pixels against the latest obstacle map before rendering."""
        if projector is None or full_map is None:
            return

        obstacle_mask = np.asarray(full_map[0] > 0.5, dtype=bool)
        for record in self.region_records:
            filtered_pixels = self._filter_record_pixels_against_current_visibility(
                record=record,
                obstacle_mask=obstacle_mask,
                projector=projector,
            )

            if not filtered_pixels:
                for world_py, world_px in self._record_waypoint_points(record):
                    rotated = projector.world_to_rotated_pixel(float(world_py), float(world_px))
                    if rotated is None:
                        continue
                    row = int(round(rotated[0]))
                    col = int(round(rotated[1]))
                    if (
                        0 <= row < obstacle_mask.shape[0]
                        and 0 <= col < obstacle_mask.shape[1]
                        and not bool(obstacle_mask[row, col])
                    ):
                        filtered_pixels.add((int(world_py), int(world_px)))

            record["pixels"] = filtered_pixels
            if filtered_pixels:
                record["center_world_px"] = self._compute_record_center_from_pixels(record)

    def _filter_record_pixels_against_current_visibility(
        self,
        record: Dict[str, Any],
        obstacle_mask: np.ndarray,
        projector: RotatedMapProjector,
    ) -> Set[Tuple[int, int]]:
        anchor_world_points: List[Tuple[int, int]] = []
        anchor_rotated_points: List[Tuple[int, int]] = []
        seen_anchor_rotated: Set[Tuple[int, int]] = set()

        for candidate in self._record_waypoint_points(record):
            world_py = int(candidate[0])
            world_px = int(candidate[1])
            rotated = projector.world_to_rotated_pixel(float(world_py), float(world_px))
            if rotated is None:
                continue
            row = int(round(rotated[0]))
            col = int(round(rotated[1]))
            if not (0 <= row < obstacle_mask.shape[0] and 0 <= col < obstacle_mask.shape[1]):
                continue
            if bool(obstacle_mask[row, col]):
                continue
            rotated_point = (row, col)
            if rotated_point in seen_anchor_rotated:
                continue
            seen_anchor_rotated.add(rotated_point)
            anchor_world_points.append((world_py, world_px))
            anchor_rotated_points.append(rotated_point)

        if not anchor_rotated_points:
            center = tuple(record.get("center_world_px", (0, 0)))
            center_py = int(center[0])
            center_px = int(center[1])
            rotated = projector.world_to_rotated_pixel(float(center_py), float(center_px))
            if rotated is not None:
                row = int(round(rotated[0]))
                col = int(round(rotated[1]))
                if (
                    0 <= row < obstacle_mask.shape[0]
                    and 0 <= col < obstacle_mask.shape[1]
                    and not bool(obstacle_mask[row, col])
                ):
                    anchor_world_points.append((center_py, center_px))
                    anchor_rotated_points.append((row, col))

        if not anchor_rotated_points:
            return set()

        anchor_world_set = set(anchor_world_points)
        filtered_pixels: Set[Tuple[int, int]] = set()
        for candidate in set(record.get("pixels", set()) or set()):
            world_py = int(candidate[0])
            world_px = int(candidate[1])
            rotated = projector.world_to_rotated_pixel(float(world_py), float(world_px))
            if rotated is None:
                continue
            row = int(round(rotated[0]))
            col = int(round(rotated[1]))
            if not (0 <= row < obstacle_mask.shape[0] and 0 <= col < obstacle_mask.shape[1]):
                continue
            if bool(obstacle_mask[row, col]):
                continue

            world_point = (world_py, world_px)
            if world_point in anchor_world_set:
                filtered_pixels.add(world_point)
                continue

            for anchor_row, anchor_col in anchor_rotated_points:
                if (row, col) == (anchor_row, anchor_col):
                    filtered_pixels.add(world_point)
                    break
                if self._line_is_clear(
                    obstacle_mask=obstacle_mask,
                    start_row=float(anchor_row),
                    start_col=float(anchor_col),
                    end_row=float(row),
                    end_col=float(col),
                ):
                    filtered_pixels.add(world_point)
                    break
        return filtered_pixels

    def _consolidate_same_type_records(
        self,
        obstacle_mask: Optional[np.ndarray] = None,
        projector: Optional[RotatedMapProjector] = None,
        clearance_map: Optional[np.ndarray] = None,
    ) -> None:
        """Merge any same-type area records that now overlap or touch."""
        self._normalize_region_records()

        changed = True
        while changed:
            changed = False
            for idx, record in enumerate(list(self.region_records)):
                for other in self.region_records[idx + 1:]:
                    if record.get("space_key") != other.get("space_key"):
                        continue
                    should_merge = (
                        self._pixel_sets_overlap(record.get("pixels", set()), other.get("pixels", set()))
                        or self._pixel_sets_are_adjacent(record.get("pixels", set()), other.get("pixels", set()))
                    )
                    if (
                        should_merge
                        and obstacle_mask is not None
                        and projector is not None
                        and self._same_type_records_should_stay_distinct(
                            record_a=record,
                            record_b=other,
                            obstacle_mask=obstacle_mask,
                            projector=projector,
                            clearance_map=clearance_map,
                        )
                    ):
                        should_merge = False
                    if (
                        not should_merge
                        and obstacle_mask is not None
                        and projector is not None
                    ):
                        if not self._same_type_records_should_stay_distinct(
                            record_a=record,
                            record_b=other,
                            obstacle_mask=obstacle_mask,
                            projector=projector,
                            clearance_map=clearance_map,
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
                    self._merge_region_records(primary, secondary)
                    self._normalize_region_records()
                    changed = True
                    break
                if changed:
                    break

        resolved_current = self._resolve_label_alias(self.current_region_label)
        if resolved_current and resolved_current != "Unknown":
            current_record = next(
                (
                    item for item in self.region_records
                    if str(item.get("label", "")) == resolved_current
                ),
                None,
            )
            if current_record is not None:
                self._set_current_area_from_record(current_record)

    def _merge_region_records(
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
        if secondary in self.region_records:
            self.region_records.remove(secondary)

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
    def _normalize_space_type_name(space_type: str) -> str:
        compact_text = strip_space_type_label_variant_suffix(space_type)
        compact_text = " ".join(str(compact_text or "").split())
        if not compact_text:
            return "Unknown"

        canonical = normalize_space_type(compact_text)
        if canonical != "Unknown":
            return canonical
        return compact_text

    def _normalize_region_records(self) -> None:
        grouped_records: Dict[str, List[Dict[str, Any]]] = {}
        for record in self.region_records:
            normalized_type = self._normalize_space_type_name(str(record.get("space_type", "Unknown") or "Unknown"))
            record["space_type"] = normalized_type
            record["space_key"] = self._space_type_key(normalized_type)
            grouped_records.setdefault(record["space_key"], []).append(record)

        for records in grouped_records.values():
            records.sort(key=lambda item: (int(item.get("variant", 0) or 0), int(item.get("id", 0) or 0)))
            for variant, record in enumerate(records, start=1):
                old_label = str(record.get("label", "") or "")
                new_label = self._region_label(str(record.get("space_type", "Unknown") or "Unknown"), variant)
                record["variant"] = variant
                record["label"] = new_label
                if old_label and old_label != new_label:
                    self._register_label_alias(old_label=old_label, new_label=new_label)

    @staticmethod
    def _parse_space_type(description: str) -> str:
        text = (description or "").strip()
        if not text:
            return "Unknown"

        for sep in ("|", "-", "Nearby", "Connected"):
            if sep in text:
                text = text.split(sep)[0].strip()
        return RegionManager._normalize_space_type_name(text)

    @staticmethod
    def _space_type_key(space_type: str) -> str:
        return "".join(ch.lower() for ch in space_type if ch.isalnum())

    @staticmethod
    def _region_label(space_type: str, variant: int) -> str:
        words = [word.capitalize() for word in space_type.split() if word]
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
        clearance_map: Optional[np.ndarray] = None,
    ) -> bool:
        candidate_record = {
            "space_type": str(existing_record.get("space_type", "Unknown") or "Unknown"),
            "space_key": str(existing_record.get("space_key", "")),
            "center_world_px": tuple(new_center_world_px),
            "pixels": set(new_pixels),
            "waypoint_points": {tuple(new_center_world_px)},
        }
        existing_pixels = existing_record.get("pixels", set())
        if self._pixel_sets_overlap(existing_pixels, new_pixels):
            if (
                projector is not None
                and obstacle_mask is not None
                and self._same_type_records_should_stay_distinct(
                    record_a=existing_record,
                    record_b=candidate_record,
                    obstacle_mask=obstacle_mask,
                    projector=projector,
                    clearance_map=clearance_map,
                )
            ):
                return False
            return True
        if self._pixel_sets_are_adjacent(existing_pixels, new_pixels):
            if (
                projector is not None
                and obstacle_mask is not None
                and self._same_type_records_should_stay_distinct(
                    record_a=existing_record,
                    record_b=candidate_record,
                    obstacle_mask=obstacle_mask,
                    projector=projector,
                    clearance_map=clearance_map,
                )
            ):
                return False
            return True
        if projector is None or obstacle_mask is None:
            return False
        if self._same_type_records_should_stay_distinct(
            record_a=existing_record,
            record_b=candidate_record,
            obstacle_mask=obstacle_mask,
            projector=projector,
            clearance_map=clearance_map,
        ):
            return False

        return self._records_have_waypoint_connection(
            record_a=existing_record,
            record_b=candidate_record,
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

    @staticmethod
    def _build_clearance_map(obstacle_mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if obstacle_mask is None:
            return None
        free_mask = np.asarray(~obstacle_mask, dtype=np.uint8)
        if free_mask.size == 0:
            return None
        return cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)

    def _world_pixel_clearance_px(
        self,
        pixel_y: int,
        pixel_x: int,
        projector: Optional[RotatedMapProjector],
        clearance_map: Optional[np.ndarray],
    ) -> Optional[float]:
        if projector is None or clearance_map is None:
            return None
        rotated = projector.world_to_rotated_pixel(float(pixel_y), float(pixel_x))
        if rotated is None:
            return None
        row = int(round(rotated[0]))
        col = int(round(rotated[1]))
        if not (0 <= row < clearance_map.shape[0] and 0 <= col < clearance_map.shape[1]):
            return None
        return float(clearance_map[row, col])

    def _filter_area_pixels_for_space_type(
        self,
        world_pixels: Set[Tuple[int, int]],
        space_type: str,
        projector: Optional[RotatedMapProjector],
        clearance_map: Optional[np.ndarray],
    ) -> Set[Tuple[int, int]]:
        del space_type, projector, clearance_map
        return set(world_pixels)

    def _find_region_start(
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

    def _compute_region_world_pixels(
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

        obstacle_mask = np.asarray(full_map[0] > 0.5, dtype=bool)
        traversible = ~obstacle_mask
        center_rot = projector.world_to_rotated_pixel(pixel_y, pixel_x)
        if center_rot is None:
            return set()

        center_row = int(round(center_rot[0]))
        center_col = int(round(center_rot[1]))
        if not (
            0 <= center_row < traversible.shape[0]
            and 0 <= center_col < traversible.shape[1]
            and traversible[center_row, center_col]
        ):
            nearest = self._find_region_start(traversible, center_row, center_col)
            if nearest is None:
                if self._world_pixel_is_traversible_with_projector(
                    pixel_y=int(pixel_y),
                    pixel_x=int(pixel_x),
                    full_map=full_map,
                    projector=projector,
                ):
                    return {(int(pixel_y), int(pixel_x))}
                return set()
            center_row, center_col = int(nearest[0]), int(nearest[1])

        start = (center_row, center_col)

        max_radius_px = int(round((max_radius_m * 100.0) / float(self.resolution)))
        selected_rotated = self._collect_visible_rotated_pixels(
            obstacle_mask=obstacle_mask,
            start=start,
            max_radius_px=max_radius_px,
        )
        world_pixels = self._rotated_pixels_to_world_pixels(
            rotated_pixels=selected_rotated,
            projector=projector,
            full_map=full_map,
        )

        if len(world_pixels) < self.MIN_RENDERABLE_AREA_PIXELS:
            seed_radius_px = int(round((self.MIN_SEED_AREA_RADIUS_M * 100.0) / float(self.resolution)))
            seed_rotated = self._collect_visible_rotated_pixels(
                obstacle_mask=obstacle_mask,
                start=start,
                max_radius_px=max(1, min(seed_radius_px, max_radius_px)),
            )
            world_pixels.update(
                self._rotated_pixels_to_world_pixels(
                    rotated_pixels=seed_rotated,
                    projector=projector,
                    full_map=full_map,
                )
            )

        if world_pixels:
            return world_pixels
        if self._world_pixel_is_traversible_with_projector(
            pixel_y=int(pixel_y),
            pixel_x=int(pixel_x),
            full_map=full_map,
            projector=projector,
        ):
            return {(int(pixel_y), int(pixel_x))}
        return set()

    @staticmethod
    def _collect_connected_rotated_pixels(
        traversible: np.ndarray,
        start: Tuple[int, int],
        max_radius_px: int,
    ) -> List[Tuple[int, int]]:
        h_map, w_map = traversible.shape
        visited = np.zeros((h_map, w_map), dtype=bool)
        queue = deque([start])
        visited[start[0], start[1]] = True
        selected: List[Tuple[int, int]] = []

        while queue:
            row, col = queue.popleft()
            if ((row - start[0]) ** 2 + (col - start[1]) ** 2) > max_radius_px ** 2:
                continue
            selected.append((row, col))

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

        return selected

    @classmethod
    def _collect_visible_rotated_pixels(
        cls,
        obstacle_mask: np.ndarray,
        start: Tuple[int, int],
        max_radius_px: int,
    ) -> List[Tuple[int, int]]:
        h_map, w_map = obstacle_mask.shape
        free_mask = ~np.asarray(obstacle_mask, dtype=bool)
        start_row = int(start[0])
        start_col = int(start[1])
        if not (0 <= start_row < h_map and 0 <= start_col < w_map):
            return []
        if not free_mask[start_row, start_col]:
            return []

        radius_sq = float(max_radius_px * max_radius_px)
        selected: List[Tuple[int, int]] = []
        row_min = max(0, start_row - int(max_radius_px))
        row_max = min(h_map - 1, start_row + int(max_radius_px))
        col_min = max(0, start_col - int(max_radius_px))
        col_max = min(w_map - 1, start_col + int(max_radius_px))

        for row in range(row_min, row_max + 1):
            row_delta_sq = float((row - start_row) ** 2)
            for col in range(col_min, col_max + 1):
                if not free_mask[row, col]:
                    continue
                dist_sq = row_delta_sq + float((col - start_col) ** 2)
                if dist_sq > radius_sq:
                    continue
                if not cls._line_is_clear(
                    obstacle_mask=obstacle_mask,
                    start_row=float(start_row),
                    start_col=float(start_col),
                    end_row=float(row),
                    end_col=float(col),
                ):
                    continue
                selected.append((int(row), int(col)))

        return selected

    def _rotated_pixels_to_world_pixels(
        self,
        rotated_pixels: Sequence[Tuple[int, int]],
        projector: RotatedMapProjector,
        full_map: np.ndarray,
    ) -> Set[Tuple[int, int]]:
        world_pixels: Set[Tuple[int, int]] = set()
        for row, col in rotated_pixels:
            world_pixel = self._select_world_pixel_for_rotated(
                rotated_row=int(row),
                rotated_col=int(col),
                projector=projector,
                full_map=full_map,
            )
            if world_pixel is None:
                continue
            world_pixels.add(world_pixel)
        return world_pixels

    def _select_world_pixel_for_rotated(
        self,
        rotated_row: int,
        rotated_col: int,
        projector: RotatedMapProjector,
        full_map: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        world = projector.rotated_to_world_pixel(float(rotated_row), float(rotated_col))
        if world is None:
            return None

        world_row = float(world[0])
        world_col = float(world[1])
        base_row = int(np.floor(world_row))
        base_col = int(np.floor(world_col))
        best_candidate: Optional[Tuple[float, int, int]] = None

        for cand_row in range(base_row - 1, base_row + 2):
            for cand_col in range(base_col - 1, base_col + 2):
                if not self._world_pixel_is_traversible_with_projector(
                    pixel_y=int(cand_row),
                    pixel_x=int(cand_col),
                    full_map=full_map,
                    projector=projector,
                ):
                    continue
                projected = projector.world_to_rotated_pixel(float(cand_row), float(cand_col))
                if projected is None:
                    continue
                reproj_error = float(np.hypot(
                    float(projected[0]) - float(rotated_row),
                    float(projected[1]) - float(rotated_col),
                ))
                candidate = (reproj_error, int(cand_row), int(cand_col))
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate

        if best_candidate is not None:
            return int(best_candidate[1]), int(best_candidate[2])

        rounded_row = int(round(world_row))
        rounded_col = int(round(world_col))
        if self._world_pixel_is_traversible_with_projector(
            pixel_y=rounded_row,
            pixel_x=rounded_col,
            full_map=full_map,
            projector=projector,
        ):
            return rounded_row, rounded_col
        return None

    def _paint_record_seed_region(
        self,
        layer: np.ndarray,
        best_distance: np.ndarray,
        record_id: int,
        center_world_px: Tuple[int, int],
        traversible: np.ndarray,
        projector: RotatedMapProjector,
    ) -> None:
        rotated = projector.world_to_rotated_pixel(
            float(center_world_px[0]),
            float(center_world_px[1]),
        )
        if rotated is None:
            return

        seed_row = int(round(rotated[0]))
        seed_col = int(round(rotated[1]))
        if not (0 <= seed_row < traversible.shape[0] and 0 <= seed_col < traversible.shape[1]):
            return
        if not traversible[seed_row, seed_col]:
            nearest = self._find_region_start(traversible, seed_row, seed_col)
            if nearest is None:
                return
            seed_row, seed_col = int(nearest[0]), int(nearest[1])

        seed_radius_px = max(1, int(round((self.MIN_SEED_AREA_RADIUS_M * 100.0) / float(self.resolution))))
        for row, col in self._collect_connected_rotated_pixels(
            traversible=traversible,
            start=(seed_row, seed_col),
            max_radius_px=seed_radius_px,
        ):
            dist = float(np.hypot(float(row) - float(seed_row), float(col) - float(seed_col)))
            if dist < best_distance[row, col]:
                best_distance[row, col] = dist
                layer[row, col] = int(record_id)

    def _set_current_region_from_layer(
        self,
        layer: np.ndarray,
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
        projector: RotatedMapProjector,
        waypoint_positions: Optional[Sequence[Tuple[int, int]]] = None,
        waypoint_area_labels: Optional[Sequence[str]] = None,
    ) -> None:
        if full_pose is None or not self.region_records:
            self._set_unknown_current_area()
            return

        curr_py = int(round(float(full_pose[1]) * 100.0 / float(self.resolution)))
        curr_px = int(round(float(full_pose[0]) * 100.0 / float(self.resolution)))
        probe_py, probe_px = self._resolve_current_area_probe_pixel(
            pixel_y=curr_py,
            pixel_x=curr_px,
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
        )
        current_record = self._resolve_current_region_record(
            pixel_y=curr_py,
            pixel_x=curr_px,
            probe_py=probe_py,
            probe_px=probe_px,
            waypoint_positions=waypoint_positions,
            waypoint_area_labels=waypoint_area_labels,
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
            layer=layer,
            projector=projector,
        )
        if current_record is not None:
            self._set_current_area_from_record(current_record)
            return

        self._set_unknown_current_area()

    def _set_unknown_current_area(self) -> None:
        self.current_region_label = "Unknown"
        self.current_region_type = "Unknown"

    def _maintain_current_area_with_pose(
        self,
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
        waypoint_positions: Optional[Sequence[Tuple[int, int]]] = None,
        waypoint_area_labels: Optional[Sequence[str]] = None,
    ) -> None:
        if full_map is None or full_pose is None or not self.region_records:
            self._set_unknown_current_area()
            return

        curr_py = int(round(float(full_pose[1]) * 100.0 / float(self.resolution)))
        curr_px = int(round(float(full_pose[0]) * 100.0 / float(self.resolution)))
        probe_py, probe_px = self._resolve_current_area_probe_pixel(
            pixel_y=curr_py,
            pixel_x=curr_px,
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
        )

        current_record = self._resolve_current_region_record(
            pixel_y=curr_py,
            pixel_x=curr_px,
            probe_py=probe_py,
            probe_px=probe_px,
            waypoint_positions=waypoint_positions,
            waypoint_area_labels=waypoint_area_labels,
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
        )
        if current_record is not None:
            self._set_current_area_from_record(current_record)
            return

        self._set_unknown_current_area()

    def _find_record_containing_pixel(
        self,
        pixel_y: int,
        pixel_x: int,
    ) -> Optional[Dict[str, Any]]:
        target_pixel = (int(pixel_y), int(pixel_x))
        for record in reversed(self.region_records):
            if target_pixel in record["pixels"]:
                return record
        return None

    def _find_current_record_from_waypoints(
        self,
        pixel_y: int,
        pixel_x: int,
        waypoint_positions: Optional[Sequence[Tuple[int, int]]] = None,
        waypoint_area_labels: Optional[Sequence[str]] = None,
        containment_pixel: Optional[Tuple[int, int]] = None,
        full_map: Optional[np.ndarray] = None,
        full_pose: Optional[Sequence[float]] = None,
        crop_offset: Optional[Tuple[int, int]] = None,
        require_nearby_waypoint: bool = True,
        distance_field: Optional[np.ndarray] = None,
        obstacle_mask: Optional[np.ndarray] = None,
        projector: Optional[RotatedMapProjector] = None,
    ) -> Optional[Dict[str, Any]]:
        if containment_pixel is None:
            contain_y, contain_x = int(pixel_y), int(pixel_x)
        else:
            contain_y, contain_x = int(containment_pixel[0]), int(containment_pixel[1])
        containing_record = self._find_record_containing_pixel(contain_y, contain_x)
        if containing_record is None:
            return None
        if not require_nearby_waypoint:
            return containing_record
        if self._record_has_nearby_area_waypoint(
            record=containing_record,
            pixel_y=pixel_y,
            pixel_x=pixel_x,
            waypoint_positions=waypoint_positions,
            waypoint_area_labels=waypoint_area_labels,
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
            distance_field=distance_field,
            obstacle_mask=obstacle_mask,
            projector=projector,
        ):
            return containing_record
        return None

    def _resolve_current_region_record(
        self,
        pixel_y: int,
        pixel_x: int,
        probe_py: int,
        probe_px: int,
        waypoint_positions: Optional[Sequence[Tuple[int, int]]] = None,
        waypoint_area_labels: Optional[Sequence[str]] = None,
        full_map: Optional[np.ndarray] = None,
        full_pose: Optional[Sequence[float]] = None,
        crop_offset: Optional[Tuple[int, int]] = None,
        layer: Optional[np.ndarray] = None,
        projector: Optional[RotatedMapProjector] = None,
    ) -> Optional[Dict[str, Any]]:
        record_map = self._build_record_label_map()
        obstacle_mask = (
            np.asarray(full_map[0] > 0.5, dtype=bool)
            if full_map is not None and projector is not None
            else None
        )
        distance_field = None
        if obstacle_mask is not None and projector is not None:
            distance_field = build_bounded_geodesic_distance_field(
                obstacle_mask=obstacle_mask,
                projector=projector,
                source_world=(int(probe_py), int(probe_px)),
                max_distance_m=max(
                    float(self.CURRENT_AREA_WAYPOINT_MAX_DISTANCE_M),
                    float(self.CURRENT_AREA_INITIAL_WAYPOINT_MAX_DISTANCE_M),
                ),
                resolution_cm=float(self.resolution),
            )

        sticky_record = self._record_from_label_map(self.current_region_label, record_map)
        if sticky_record is not None and self._record_has_nearby_area_waypoint(
            record=sticky_record,
            pixel_y=probe_py,
            pixel_x=probe_px,
            waypoint_positions=waypoint_positions,
            waypoint_area_labels=waypoint_area_labels,
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
            distance_field=distance_field,
            obstacle_mask=obstacle_mask,
            projector=projector,
        ):
            return sticky_record

        if layer is not None and projector is not None:
            rotated = projector.world_to_rotated_pixel(probe_py, probe_px)
            if rotated is not None:
                row = int(round(rotated[0]))
                col = int(round(rotated[1]))
                if 0 <= row < layer.shape[0] and 0 <= col < layer.shape[1]:
                    area_id = int(layer[row, col])
                    if area_id > 0:
                        layer_record = next(
                            (record for record in self.region_records if int(record["id"]) == area_id),
                            None,
                        )
                        if layer_record is not None and self._record_has_nearby_area_waypoint(
                            record=layer_record,
                            pixel_y=probe_py,
                            pixel_x=probe_px,
                            waypoint_positions=waypoint_positions,
                            waypoint_area_labels=waypoint_area_labels,
                            full_map=full_map,
                            full_pose=full_pose,
                            crop_offset=crop_offset,
                            distance_field=distance_field,
                            obstacle_mask=obstacle_mask,
                            projector=projector,
                        ):
                            return layer_record

        containing_record = self._find_current_record_from_waypoints(
            pixel_y=pixel_y,
            pixel_x=pixel_x,
            waypoint_positions=waypoint_positions,
            waypoint_area_labels=waypoint_area_labels,
            containment_pixel=(probe_py, probe_px),
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
            require_nearby_waypoint=True,
            distance_field=distance_field,
            obstacle_mask=obstacle_mask,
            projector=projector,
        )
        if containing_record is not None:
            return containing_record

        fallback_record = self._find_record_from_nearby_waypoint(
            pixel_y=probe_py,
            pixel_x=probe_px,
            waypoint_positions=waypoint_positions,
            waypoint_area_labels=waypoint_area_labels,
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
            record_map=record_map,
            distance_field=distance_field,
            obstacle_mask=obstacle_mask,
            projector=projector,
        )
        if fallback_record is not None:
            return fallback_record

        return self._find_record_from_nearby_record_waypoints(
            pixel_y=probe_py,
            pixel_x=probe_px,
            waypoint_positions=waypoint_positions,
            full_map=full_map,
            full_pose=full_pose,
            crop_offset=crop_offset,
            distance_field=distance_field,
            obstacle_mask=obstacle_mask,
            projector=projector,
        )

    def _build_record_label_map(self) -> Dict[str, Dict[str, Any]]:
        record_map: Dict[str, Dict[str, Any]] = {}
        for record in self.region_records:
            resolved_label = self._resolve_label_alias(str(record.get("label", "")).strip())
            if resolved_label and resolved_label != "Unknown":
                record_map[resolved_label] = record
        return record_map

    def _record_from_label_map(
        self,
        label: str,
        record_map: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        resolved_label = self._resolve_label_alias(str(label or "").strip())
        if not resolved_label or resolved_label == "Unknown":
            return None
        return record_map.get(resolved_label)

    def _record_has_nearby_area_waypoint(
        self,
        record: Dict[str, Any],
        pixel_y: int,
        pixel_x: int,
        waypoint_positions: Optional[Sequence[Tuple[int, int]]],
        waypoint_area_labels: Optional[Sequence[str]],
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
        distance_field: Optional[np.ndarray] = None,
        obstacle_mask: Optional[np.ndarray] = None,
        projector: Optional[RotatedMapProjector] = None,
    ) -> bool:
        if not waypoint_positions:
            return False

        record_label = self._resolve_label_alias(str(record.get("label", "")).strip())
        if not record_label:
            return False

        local_projector = projector or self._build_projector(full_map, full_pose, crop_offset)
        local_obstacle_mask = (
            obstacle_mask
            if obstacle_mask is not None
            else (
                np.asarray(full_map[0] > 0.5, dtype=bool)
                if full_map is not None and local_projector is not None
                else None
            )
        )
        area_labels = list(waypoint_area_labels or [])

        for index, waypoint_pos in enumerate(waypoint_positions):
            if waypoint_pos is None:
                continue
            waypoint_label = self._resolve_label_alias(
                str(area_labels[index]).strip() if index < len(area_labels) else ""
            )
            if waypoint_label != record_label:
                continue

            wp_py, wp_px = int(waypoint_pos[0]), int(waypoint_pos[1])
            distance_px = float(np.hypot(float(pixel_y) - float(wp_py), float(pixel_x) - float(wp_px)))
            max_distance_m = (
                self.CURRENT_AREA_INITIAL_WAYPOINT_MAX_DISTANCE_M
                if int(index) == 0 else
                self.CURRENT_AREA_WAYPOINT_MAX_DISTANCE_M
            )
            max_distance_px = (float(max_distance_m) * 100.0) / float(self.resolution)
            if distance_field is not None and local_obstacle_mask is not None and local_projector is not None:
                geodesic_distance_m = query_world_distance_from_field_m(
                    distance_field=distance_field,
                    obstacle_mask=local_obstacle_mask,
                    projector=local_projector,
                    target_world=(wp_py, wp_px),
                    resolution_cm=float(self.resolution),
                    target_radius_m=WAYPOINT_VISIBILITY_RADIUS_M,
                    target_samples=WAYPOINT_VISIBILITY_SAMPLES,
                )
                if geodesic_distance_m is None or geodesic_distance_m > float(max_distance_m) + 1e-6:
                    continue
            else:
                if distance_px > max_distance_px + 1e-6:
                    continue
                if local_obstacle_mask is not None and local_projector is not None:
                    if not self._world_line_is_clear(
                        obstacle_mask=local_obstacle_mask,
                        projector=local_projector,
                        start_world=(int(pixel_y), int(pixel_x)),
                        end_world=(wp_py, wp_px),
                    ):
                        continue
            return True
        return False

    def _find_record_from_nearby_waypoint(
        self,
        pixel_y: int,
        pixel_x: int,
        waypoint_positions: Optional[Sequence[Tuple[int, int]]],
        waypoint_area_labels: Optional[Sequence[str]],
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
        record_map: Optional[Dict[str, Dict[str, Any]]] = None,
        distance_field: Optional[np.ndarray] = None,
        obstacle_mask: Optional[np.ndarray] = None,
        projector: Optional[RotatedMapProjector] = None,
    ) -> Optional[Dict[str, Any]]:
        if not waypoint_positions:
            return None

        local_projector = projector or self._build_projector(full_map, full_pose, crop_offset)
        local_obstacle_mask = (
            obstacle_mask
            if obstacle_mask is not None
            else (
                np.asarray(full_map[0] > 0.5, dtype=bool)
                if full_map is not None and local_projector is not None
                else None
            )
        )
        area_labels = list(waypoint_area_labels or [])
        record_label_map = record_map or self._build_record_label_map()
        candidates: List[Tuple[float, int, Dict[str, Any]]] = []

        for index, waypoint_pos in enumerate(waypoint_positions):
            if waypoint_pos is None:
                continue
            waypoint_label = self._resolve_label_alias(
                str(area_labels[index]).strip() if index < len(area_labels) else ""
            )
            if not waypoint_label or waypoint_label == "Unknown":
                continue
            record = self._record_from_label_map(waypoint_label, record_label_map)
            if record is None:
                continue

            wp_py, wp_px = int(waypoint_pos[0]), int(waypoint_pos[1])
            distance_px = float(np.hypot(float(pixel_y) - float(wp_py), float(pixel_x) - float(wp_px)))
            max_distance_m = (
                self.CURRENT_AREA_INITIAL_WAYPOINT_MAX_DISTANCE_M
                if int(index) == 0 else
                self.CURRENT_AREA_WAYPOINT_MAX_DISTANCE_M
            )
            max_distance_px = (float(max_distance_m) * 100.0) / float(self.resolution)
            if distance_field is not None and local_obstacle_mask is not None and local_projector is not None:
                geodesic_distance_m = query_world_distance_from_field_m(
                    distance_field=distance_field,
                    obstacle_mask=local_obstacle_mask,
                    projector=local_projector,
                    target_world=(wp_py, wp_px),
                    resolution_cm=float(self.resolution),
                    target_radius_m=WAYPOINT_VISIBILITY_RADIUS_M,
                    target_samples=WAYPOINT_VISIBILITY_SAMPLES,
                )
                if geodesic_distance_m is None or geodesic_distance_m > float(max_distance_m) + 1e-6:
                    continue
                candidates.append((float(geodesic_distance_m), int(index), record))
            else:
                if distance_px > max_distance_px + 1e-6:
                    continue

                if local_obstacle_mask is not None and local_projector is not None:
                    if not self._world_line_is_clear(
                        obstacle_mask=local_obstacle_mask,
                        projector=local_projector,
                        start_world=(int(pixel_y), int(pixel_x)),
                        end_world=(wp_py, wp_px),
                    ):
                        continue

                candidates.append((
                    float(distance_px) * float(self.resolution) / 100.0,
                    int(index),
                    record,
                ))

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                float(item[0]),
                0 if int(item[1]) != 0 else 1,
                str(item[2].get("label", "")),
            )
        )
        return candidates[0][2]

    def _find_record_from_nearby_record_waypoints(
        self,
        pixel_y: int,
        pixel_x: int,
        waypoint_positions: Optional[Sequence[Tuple[int, int]]],
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
        distance_field: Optional[np.ndarray] = None,
        obstacle_mask: Optional[np.ndarray] = None,
        projector: Optional[RotatedMapProjector] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.region_records:
            return None

        local_projector = projector or self._build_projector(full_map, full_pose, crop_offset)
        local_obstacle_mask = (
            obstacle_mask
            if obstacle_mask is not None
            else (
                np.asarray(full_map[0] > 0.5, dtype=bool)
                if full_map is not None and local_projector is not None
                else None
            )
        )
        initial_waypoint = None
        if waypoint_positions and waypoint_positions[0] is not None:
            initial_waypoint = (int(waypoint_positions[0][0]), int(waypoint_positions[0][1]))

        candidates: List[Tuple[float, int, Dict[str, Any]]] = []
        for record in self.region_records:
            best_match: Optional[Tuple[float, int]] = None
            for waypoint_point in self._record_waypoint_points(record):
                point = (int(waypoint_point[0]), int(waypoint_point[1]))
                distance_px = float(np.hypot(float(pixel_y) - float(point[0]), float(pixel_x) - float(point[1])))
                is_initial_point = int(initial_waypoint == point)
                max_distance_m = (
                    self.CURRENT_AREA_INITIAL_WAYPOINT_MAX_DISTANCE_M
                    if is_initial_point else
                    self.CURRENT_AREA_WAYPOINT_MAX_DISTANCE_M
                )
                max_distance_px = (float(max_distance_m) * 100.0) / float(self.resolution)
                if distance_field is not None and local_obstacle_mask is not None and local_projector is not None:
                    geodesic_distance_m = query_world_distance_from_field_m(
                        distance_field=distance_field,
                        obstacle_mask=local_obstacle_mask,
                        projector=local_projector,
                        target_world=point,
                        resolution_cm=float(self.resolution),
                        target_radius_m=WAYPOINT_VISIBILITY_RADIUS_M,
                        target_samples=WAYPOINT_VISIBILITY_SAMPLES,
                    )
                    if geodesic_distance_m is None or geodesic_distance_m > float(max_distance_m) + 1e-6:
                        continue
                    candidate = (float(geodesic_distance_m), is_initial_point)
                else:
                    if distance_px > max_distance_px + 1e-6:
                        continue
                    if local_obstacle_mask is not None and local_projector is not None:
                        if not self._world_line_is_clear(
                            obstacle_mask=local_obstacle_mask,
                            projector=local_projector,
                            start_world=(int(pixel_y), int(pixel_x)),
                            end_world=point,
                        ):
                            continue
                    candidate = (
                        float(distance_px) * float(self.resolution) / 100.0,
                        is_initial_point,
                    )
                if best_match is None or candidate < best_match:
                    best_match = candidate
            if best_match is not None:
                candidates.append((best_match[0], best_match[1], record))

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                float(item[0]),
                int(item[1]),
                str(item[2].get("label", "")),
            )
        )
        return candidates[0][2]

    def _set_current_area_from_record(self, record: Dict[str, Any]) -> None:
        self.current_region_label = str(record.get("label", "Unknown") or "Unknown")
        self.current_region_type = str(record.get("space_type", "Unknown") or "Unknown")

    def _resolve_current_area_probe_pixel(
        self,
        pixel_y: int,
        pixel_x: int,
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
    ) -> Tuple[int, int]:
        """Snap the current pose to the nearest traversible map cell for area membership checks."""
        projector = self._build_projector(full_map, full_pose, crop_offset)
        if full_map is None or projector is None:
            return int(pixel_y), int(pixel_x)

        obstacle_mask = np.asarray(full_map[0] > 0.5, dtype=bool)
        traversible = ~obstacle_mask
        rotated = projector.world_to_rotated_pixel(float(pixel_y), float(pixel_x))
        if rotated is None:
            return int(pixel_y), int(pixel_x)

        center_row = int(round(rotated[0]))
        center_col = int(round(rotated[1]))
        if (
            0 <= center_row < traversible.shape[0]
            and 0 <= center_col < traversible.shape[1]
            and traversible[center_row, center_col]
        ):
            return int(pixel_y), int(pixel_x)

        nearest = self._find_region_start(traversible, center_row, center_col)
        if nearest is None:
            return int(pixel_y), int(pixel_x)

        world = projector.rotated_to_world_pixel(nearest[0], nearest[1])
        if world is None:
            return int(pixel_y), int(pixel_x)
        return int(round(world[0])), int(round(world[1]))

    def get_display_label(self, label: str) -> str:
        target_label = self._resolve_label_alias(str(label or "").strip())
        if not target_label:
            return "Unknown"
        if target_label == "Unknown":
            return "Unknown"
        record = next(
            (item for item in self.region_records if str(item.get("label", "")) == target_label),
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
        for record in self.region_records:
            record["connected_area_labels"] = []
            record["display_label"] = self._region_label(
                str(record.get("space_type", "Unknown") or "Unknown"),
                int(record.get("variant", 0) or 1),
            )

        if full_map is None or full_pose is None or crop_offset is None:
            return

        projector = self._build_projector(full_map, full_pose, crop_offset)
        if projector is None:
            return

        obstacle_mask = np.asarray(full_map[0] > 0.5, dtype=bool)
        for record in self.region_records:
            if str(record.get("space_type", "")) not in self.CONNECTOR_SPACE_TYPES:
                continue

            connected = self._compute_connected_areas_for_record(
                record=record,
                obstacle_mask=obstacle_mask,
                projector=projector,
            )
            connected_labels = [
                self._region_label(
                    str(item.get("space_type", "Unknown") or "Unknown"),
                    int(item.get("variant", 0) or 1),
                )
                for item in connected[: self.MAX_CONNECTED_AREAS]
            ]
            record["connected_area_labels"] = connected_labels
            if connected_labels:
                base_label = self._region_label(
                    str(record.get("space_type", "Unknown") or "Unknown"),
                    int(record.get("variant", 0) or 1),
                )
                record["display_label"] = (
                    f"{base_label} [links: {', '.join(connected_labels[: self.MAX_CONNECTED_AREAS])}]"
                )

    def _compute_connected_areas_for_record(
        self,
        record: Dict[str, Any],
        obstacle_mask: np.ndarray,
        projector: RotatedMapProjector,
    ) -> List[Dict[str, Any]]:
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        max_distance_px = (self.MAX_CONNECTION_DISTANCE_M * 100.0) / float(self.resolution)

        for other in self.region_records:
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
        return RegionManager._pixel_sets_are_adjacent(
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

    def _same_type_records_should_stay_distinct(
        self,
        record_a: Dict[str, Any],
        record_b: Dict[str, Any],
        obstacle_mask: np.ndarray,
        projector: RotatedMapProjector,
        clearance_map: Optional[np.ndarray],
    ) -> bool:
        if str(record_a.get("space_key", "")) != str(record_b.get("space_key", "")):
            return False

        space_type = str(record_a.get("space_type", "Unknown") or "Unknown")
        if space_type in self.CONNECTOR_SPACE_TYPES:
            return False

        # Revisited same-type areas should merge once they have a broad, open,
        # obstacle-free connection again, even if both records also touch the
        # same hallway / entry connector. The connector bridge should only keep
        # them distinct when there is no sufficiently open same-type passage.
        if self._records_have_broad_open_connection(
            record_a=record_a,
            record_b=record_b,
            obstacle_mask=obstacle_mask,
            projector=projector,
            clearance_map=clearance_map,
            max_distance_m=self.MAX_SAME_TYPE_WAYPOINT_MERGE_DISTANCE_M,
            min_clearance_m=self.SAME_TYPE_MERGE_MIN_CLEARANCE_M,
        ):
            return False

        # Same-type areas should merge by default. Keep distinct variants only
        # when a connector genuinely separates them and the two sides are not
        # just nearby fragments of the same room.
        if not self._records_share_connector_bridge(
            record_a=record_a,
            record_b=record_b,
            obstacle_mask=obstacle_mask,
            projector=projector,
        ):
            return False

        if self._records_have_waypoint_connection(
            record_a=record_a,
            record_b=record_b,
            obstacle_mask=obstacle_mask,
            projector=projector,
            max_distance_m=self.SAME_TYPE_CONNECTOR_SPLIT_DISTANCE_M,
        ):
            return False

        return True

    def _records_share_connector_bridge(
        self,
        record_a: Dict[str, Any],
        record_b: Dict[str, Any],
        obstacle_mask: np.ndarray,
        projector: RotatedMapProjector,
    ) -> bool:
        for connector_record in self.region_records:
            if connector_record is record_a or connector_record is record_b:
                continue
            if str(connector_record.get("space_type", "")) not in self.CONNECTOR_SPACE_TYPES:
                continue
            if not self._records_touch_or_connect(
                record_a,
                connector_record,
                obstacle_mask=obstacle_mask,
                projector=projector,
            ):
                continue
            if self._records_touch_or_connect(
                record_b,
                connector_record,
                obstacle_mask=obstacle_mask,
                projector=projector,
            ):
                return True
        return False

    def _records_touch_or_connect(
        self,
        record_a: Dict[str, Any],
        record_b: Dict[str, Any],
        obstacle_mask: np.ndarray,
        projector: RotatedMapProjector,
    ) -> bool:
        if self._records_are_adjacent(record_a, record_b):
            return True
        return self._records_have_clear_connection(
            record_a=record_a,
            record_b=record_b,
            obstacle_mask=obstacle_mask,
            projector=projector,
        )

    def _record_connection_points(
        self,
        record: Dict[str, Any],
    ) -> List[Tuple[int, int]]:
        points: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for candidate in self._record_waypoint_points(record) + self._sample_record_pixels(record):
            point = (int(candidate[0]), int(candidate[1]))
            if point in seen:
                continue
            seen.add(point)
            points.append(point)
        return points

    def _records_have_broad_open_connection(
        self,
        record_a: Dict[str, Any],
        record_b: Dict[str, Any],
        obstacle_mask: np.ndarray,
        projector: RotatedMapProjector,
        clearance_map: Optional[np.ndarray],
        max_distance_m: float,
        min_clearance_m: float,
    ) -> bool:
        if clearance_map is None:
            return False

        max_distance_px = (float(max_distance_m) * 100.0) / float(self.resolution)
        min_clearance_px = (float(min_clearance_m) * 100.0) / float(self.resolution)
        points_a = self._record_connection_points(record_a)
        points_b = self._record_connection_points(record_b)
        for start_world in points_a:
            for end_world in points_b:
                if float(np.hypot(
                    float(start_world[0]) - float(end_world[0]),
                    float(start_world[1]) - float(end_world[1]),
                )) > max_distance_px + 1e-6:
                    continue
                min_path_clearance_px = self._world_line_min_clearance_px(
                    clearance_map=clearance_map,
                    obstacle_mask=obstacle_mask,
                    projector=projector,
                    start_world=start_world,
                    end_world=end_world,
                )
                if min_path_clearance_px is None:
                    continue
                if min_path_clearance_px >= min_clearance_px:
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

    def _distance_to_record_waypoints(
        self,
        record: Dict[str, Any],
        pixel_y: int,
        pixel_x: int,
    ) -> float:
        waypoint_points = self._record_waypoint_points(record)
        if not waypoint_points:
            center = tuple(record.get("center_world_px", (0, 0)))
            return float(np.hypot(float(pixel_y) - float(center[0]), float(pixel_x) - float(center[1])))
        return min(
            float(np.hypot(float(pixel_y) - float(point[0]), float(pixel_x) - float(point[1])))
            for point in waypoint_points
        )

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

    @staticmethod
    def _line_min_clearance_px(
        clearance_map: np.ndarray,
        obstacle_mask: np.ndarray,
        start_row: float,
        start_col: float,
        end_row: float,
        end_col: float,
    ) -> Optional[float]:
        steps = max(int(np.ceil(max(abs(end_row - start_row), abs(end_col - start_col)))), 1)
        rows = np.linspace(start_row, end_row, steps + 1)
        cols = np.linspace(start_col, end_col, steps + 1)
        height, width = obstacle_mask.shape
        min_clearance_px: Optional[float] = None

        for idx in range(0, steps + 1):
            row = int(round(rows[idx]))
            col = int(round(cols[idx]))
            if not (0 <= row < height and 0 <= col < width):
                return None
            if bool(obstacle_mask[row, col]):
                return None
            clearance_px = float(clearance_map[row, col])
            if min_clearance_px is None or clearance_px < min_clearance_px:
                min_clearance_px = clearance_px
        return min_clearance_px

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

    def _world_line_min_clearance_px(
        self,
        clearance_map: np.ndarray,
        obstacle_mask: np.ndarray,
        projector: RotatedMapProjector,
        start_world: Tuple[int, int],
        end_world: Tuple[int, int],
    ) -> Optional[float]:
        start_rot = projector.world_to_rotated_pixel(float(start_world[0]), float(start_world[1]))
        end_rot = projector.world_to_rotated_pixel(float(end_world[0]), float(end_world[1]))
        if start_rot is None or end_rot is None:
            return None
        return self._line_min_clearance_px(
            clearance_map=clearance_map,
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
            if self._world_pixel_is_traversible_with_projector(
                pixel_y=int(pixel_y),
                pixel_x=int(pixel_x),
                full_map=full_map,
                projector=projector,
            ):
                filtered.add((int(pixel_y), int(pixel_x)))
        return filtered

    def _world_pixel_is_traversible_with_projector(
        self,
        pixel_y: int,
        pixel_x: int,
        full_map: Optional[np.ndarray],
        projector: Optional[RotatedMapProjector],
    ) -> bool:
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
        return not obstacle

    def _world_pixel_is_traversible(
        self,
        pixel_y: int,
        pixel_x: int,
        full_map: Optional[np.ndarray],
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
    ) -> bool:
        projector = self._build_projector(full_map, full_pose, crop_offset)
        return self._world_pixel_is_traversible_with_projector(
            pixel_y=pixel_y,
            pixel_x=pixel_x,
            full_map=full_map,
            projector=projector,
        )
