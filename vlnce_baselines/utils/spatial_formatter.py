import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vlnce_baselines.config.core.params.spatial import (
    CURRENT_AREA_OVERLAP_THRESHOLD_M,
    WAYPOINT_VISIBILITY_RADIUS_M,
    WAYPOINT_VISIBILITY_SAMPLES,
)
from vlnce_baselines.visualization.map_projection import RotatedMapProjector


def normalize_relative_bearing(bearing_deg: float) -> float:
    """Normalize relative bearing into [-180, 180]."""
    return ((bearing_deg + 180.0) % 360.0) - 180.0


def snap_relative_bearing(bearing_deg: float) -> int:
    """Snap bearing to the Front/Back and 30-degree bins used in prompts."""
    bearing = normalize_relative_bearing(bearing_deg)
    magnitude = abs(bearing)
    if magnitude <= 15.0:
        return 0
    if magnitude >= 165.0:
        return 180 if bearing >= 0 else -180
    bucket = int((magnitude - 15.0 - 1e-6) // 30.0) + 1
    snapped = min(bucket * 30, 150)
    return snapped if bearing > 0 else -snapped


def format_relative_direction(bearing_deg: float) -> str:
    """Format bearing using the snapped prompt-side direction labels."""
    snapped = snap_relative_bearing(bearing_deg)
    magnitude = abs(snapped)
    if magnitude <= 15.0:
        return "Front 0deg"
    if magnitude >= 165.0:
        return "Back 180deg"
    side = "Right" if snapped > 0 else "Left"
    return f"{side} {magnitude:.0f}deg"


def build_landmark_turn_hint(bearing_deg: float, is_visible: bool = False) -> str:
    """Produce the short action-side hint used for visible/off-screen landmarks."""
    snapped = snap_relative_bearing(bearing_deg)
    if is_visible and snapped == 0:
        return ""
    if snapped == 0:
        return " -> move forward"
    if snapped > 0:
        return f" -> TURN RIGHT {abs(snapped)}deg then move forward"
    return f" -> TURN LEFT {abs(snapped)}deg then move forward"


def build_waypoint_summary(
    waypoint_positions: Sequence[Tuple[int, int]],
    waypoint_ids: Sequence[int],
    waypoint_descriptions: Sequence[str],
    waypoint_area_labels: Optional[Sequence[str]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    current_space_area_label: str = "",
    current_space_area_type: str = "",
    full_map: Optional[np.ndarray] = None,
    crop_offset: Optional[Tuple[int, int]] = None,
    include_area_chain: bool = True,
    include_path: bool = True,
) -> str:
    """Summarize visited waypoints relative to the current pose."""
    header_lines: List[str] = []
    display_area_label = current_space_area_label or "Unknown"
    display_area_type = current_space_area_type or "Unknown"
    space_type_note = (
        f" ({display_area_type})"
        if display_area_type and display_area_type != "Unknown" and display_area_type != display_area_label
        else ""
    )
    header_lines.append(f"Your Current Area: {display_area_label}{space_type_note}")

    empty_area_path_line = None
    if include_area_chain:
        current_area_display = str(current_space_area_label or "Unknown").strip() or "Unknown"
        empty_area_path_line = f"Space Waypoint Path: Current({current_area_display})"

    if not waypoint_ids:
        lines = list(header_lines)
        if empty_area_path_line:
            lines.append(empty_area_path_line)
        lines.append("No space waypoints recorded yet.")
        return "\n".join(lines)

    waypoint_distances_m: List[Optional[float]] = []
    close_last_waypoint = False
    if current_pose is not None:
        curr_x_m, curr_y_m, _curr_orientation_deg = current_pose[:3]
        for wp_py, wp_px in waypoint_positions:
            wp_x_m = wp_px * resolution_cm / 100.0
            wp_y_m = wp_py * resolution_cm / 100.0
            waypoint_distances_m.append(math.hypot(wp_x_m - curr_x_m, wp_y_m - curr_y_m))
        if waypoint_distances_m and waypoint_distances_m[-1] <= CURRENT_AREA_OVERLAP_THRESHOLD_M:
            close_last_waypoint = True
    else:
        waypoint_distances_m = [None] * len(waypoint_ids)

    visible_indices = list(range(len(waypoint_ids)))
    if close_last_waypoint and visible_indices:
        visible_indices = visible_indices[:-1]
    last_visible_index = visible_indices[-1] if visible_indices else None

    all_area_labels = list(waypoint_area_labels or [])
    node_lines: List[str] = []
    for index in visible_indices:
        wp_id = waypoint_ids[index]
        wp_desc = waypoint_descriptions[index]
        wp_py, wp_px = waypoint_positions[index]
        is_last = last_visible_index is not None and index == last_visible_index
        suffix = "  <- LAST VISITED (came from here)" if is_last else ""

        distance_m = None
        relative_bearing_deg = None
        if current_pose is None:
            spatial_info = "distance unknown"
        else:
            wp_x_m = wp_px * resolution_cm / 100.0
            wp_y_m = wp_py * resolution_cm / 100.0
            curr_x_m, curr_y_m, curr_orientation_deg = current_pose[:3]

            dx = wp_x_m - curr_x_m
            dy = wp_y_m - curr_y_m
            distance_m = math.sqrt(dx ** 2 + dy ** 2)
            absolute_angle_deg = math.degrees(math.atan2(dy, dx))
            relative_bearing_deg = curr_orientation_deg - absolute_angle_deg
            direction = format_relative_direction(relative_bearing_deg)
            spatial_info = f"{distance_m:.1f}m, {direction}"

        area_label = ""
        if waypoint_area_labels and index < len(waypoint_area_labels):
            area_label = waypoint_area_labels[index]
        area_note = f" | area={area_label}" if area_label else ""
        reachability_note = _build_waypoint_reachability_note(
            waypoint_index=index,
            waypoint_id=wp_id,
            waypoint_ids=waypoint_ids,
            waypoint_positions=waypoint_positions,
            waypoint_area_labels=all_area_labels,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            full_map=full_map,
            crop_offset=crop_offset,
            visible_indices=visible_indices,
            current_space_area_label=current_space_area_label,
        )
        if reachability_note:
            spatial_info = f"{spatial_info} | {reachability_note}"
        node_lines.append(f"Space WP#{wp_id} [{wp_desc}{area_note}] -- {spatial_info}{suffix}")

    visible_waypoint_ids = [waypoint_ids[index] for index in visible_indices]
    visible_waypoint_positions = [waypoint_positions[index] for index in visible_indices]
    visible_waypoint_area_labels = (
        [waypoint_area_labels[index] for index in visible_indices]
        if waypoint_area_labels else []
    )

    current_area_display = str(current_space_area_label or "Unknown").strip() or "Unknown"
    waypoint_area_path_line = _build_waypoint_area_path_line(
        visible_waypoint_ids=visible_waypoint_ids,
        visible_waypoint_positions=visible_waypoint_positions,
        visible_waypoint_area_labels=visible_waypoint_area_labels,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        current_area_display=current_area_display,
        include_area_chain=include_area_chain,
        include_path=include_path,
    )

    lines = header_lines + node_lines
    if waypoint_area_path_line:
        lines.append(waypoint_area_path_line)
    return "\n".join(lines)


def _build_projector(
    full_map: Optional[np.ndarray],
    current_pose: Optional[Sequence[float]],
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


def _has_clear_path_to_waypoint(
    waypoint_py: int,
    waypoint_px: int,
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    full_map: Optional[np.ndarray],
    crop_offset: Optional[Tuple[int, int]],
) -> Optional[bool]:
    if current_pose is None or full_map is None:
        return None
    projector = _build_projector(full_map, current_pose, crop_offset)
    if projector is None:
        return None

    obstacle_mask = np.asarray(full_map[0] > 0.5, dtype=bool)
    curr_py = float(current_pose[1]) * 100.0 / float(resolution_cm)
    curr_px = float(current_pose[0]) * 100.0 / float(resolution_cm)
    start_rot = projector.world_to_rotated_pixel(curr_py, curr_px)
    if start_rot is None:
        return None

    center_py = float(waypoint_py)
    center_px = float(waypoint_px)
    radius_px = (WAYPOINT_VISIBILITY_RADIUS_M * 100.0) / float(resolution_cm)

    candidate_points: List[Tuple[float, float]] = [(center_py, center_px)]
    for sample_idx in range(WAYPOINT_VISIBILITY_SAMPLES):
        theta = (2.0 * math.pi * float(sample_idx)) / float(WAYPOINT_VISIBILITY_SAMPLES)
        candidate_points.append((
            center_py + radius_px * math.sin(theta),
            center_px + radius_px * math.cos(theta),
        ))

    for world_py, world_px in candidate_points:
        end_rot = projector.world_to_rotated_pixel(world_py, world_px)
        if end_rot is None:
            continue
        if _line_is_clear(
            obstacle_mask=obstacle_mask,
            start_row=float(start_rot[0]),
            start_col=float(start_rot[1]),
            end_row=float(end_rot[0]),
            end_col=float(end_rot[1]),
        ):
            return True
    return False


def _format_waypoint_area_ref(waypoint_id: int, area_label: str) -> str:
    clean_area = str(area_label or "Unknown").strip() or "Unknown"
    return f"Space WP#{int(waypoint_id)}({clean_area})"


def _build_waypoint_reachability_note(
    waypoint_index: int,
    waypoint_id: int,
    waypoint_ids: Sequence[int],
    waypoint_positions: Sequence[Tuple[int, int]],
    waypoint_area_labels: Sequence[str],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    full_map: Optional[np.ndarray],
    crop_offset: Optional[Tuple[int, int]],
    visible_indices: Sequence[int],
    current_space_area_label: str,
) -> str:
    if waypoint_index >= len(waypoint_positions):
        return ""

    wp_py, wp_px = waypoint_positions[waypoint_index]
    clear_path = _has_clear_path_to_waypoint(
        waypoint_py=int(wp_py),
        waypoint_px=int(wp_px),
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        full_map=full_map,
        crop_offset=crop_offset,
    )
    if clear_path is None:
        return ""
    if clear_path:
        return "connected to current"

    visible_index_set = set(int(idx) for idx in visible_indices)
    for next_index in range(waypoint_index + 1, len(waypoint_positions)):
        if next_index not in visible_index_set:
            continue
        next_wp_py, next_wp_px = waypoint_positions[next_index]
        next_clear_path = _has_clear_path_to_waypoint(
            waypoint_py=int(next_wp_py),
            waypoint_px=int(next_wp_px),
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            full_map=full_map,
            crop_offset=crop_offset,
        )
        if next_clear_path is False:
            continue
        next_waypoint_id = int(waypoint_ids[next_index]) if next_index < len(waypoint_ids) else int(next_index + 1)
        next_area = (
            str(waypoint_area_labels[next_index]).strip()
            if next_index < len(waypoint_area_labels) else "Unknown"
        ) or "Unknown"
        return f"blocked to current position; reach via {_format_waypoint_area_ref(next_waypoint_id, next_area)}"

    if current_pose is not None:
        current_area_display = str(current_space_area_label or "Unknown").strip() or "Unknown"
        return f"blocked to current position; reach via Current({current_area_display})"
    return "blocked to current position"


def _world_point_to_meters(point: Tuple[int, int], resolution_cm: float) -> Tuple[float, float]:
    py, px = point
    return float(px) * float(resolution_cm) / 100.0, float(py) * float(resolution_cm) / 100.0


def _segment_heading_deg(
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    resolution_cm: float,
) -> float:
    start_x_m, start_y_m = _world_point_to_meters(start_point, resolution_cm)
    end_x_m, end_y_m = _world_point_to_meters(end_point, resolution_cm)
    return float(math.degrees(math.atan2(end_y_m - start_y_m, end_x_m - start_x_m)))


def _segment_distance_m(
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    resolution_cm: float,
) -> float:
    start_x_m, start_y_m = _world_point_to_meters(start_point, resolution_cm)
    end_x_m, end_y_m = _world_point_to_meters(end_point, resolution_cm)
    return float(math.hypot(end_x_m - start_x_m, end_y_m - start_y_m))


def _format_turn_step(turn_delta_deg: float) -> str:
    magnitude = abs(normalize_relative_bearing(turn_delta_deg))
    if magnitude < 15.0:
        return ""
    if magnitude >= 165.0:
        return " turn 180deg"
    side = "right" if turn_delta_deg > 0.0 else "left"
    return f" turn {side} {int(round(magnitude))}deg"


def _build_waypoint_area_path_line(
    visible_waypoint_ids: Sequence[int],
    visible_waypoint_positions: Sequence[Tuple[int, int]],
    visible_waypoint_area_labels: Sequence[str],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    current_area_display: str,
    include_area_chain: bool,
    include_path: bool,
) -> Optional[str]:
    if not include_area_chain and not include_path:
        return None

    node_entries: List[Dict[str, Any]] = []
    for index, wp_id in enumerate(visible_waypoint_ids):
        area_label = (
            str(visible_waypoint_area_labels[index]).strip()
            if index < len(visible_waypoint_area_labels) else "Unknown"
        ) or "Unknown"
        node_entries.append({
            "area_label": area_label,
            "token": f"WP#{int(wp_id)}",
            "point": visible_waypoint_positions[index],
            "is_current": False,
        })

    if current_pose is not None:
        curr_py = int(round(float(current_pose[1]) * 100.0 / float(resolution_cm)))
        curr_px = int(round(float(current_pose[0]) * 100.0 / float(resolution_cm)))
        node_entries.append({
            "area_label": current_area_display,
            "token": "Current",
            "point": (curr_py, curr_px),
            "is_current": True,
        })
    elif node_entries:
        node_entries.append({
            "area_label": current_area_display,
            "token": "Current",
            "point": None,
            "is_current": True,
        })

    if not node_entries:
        return "Space Waypoint Path: Current(Unknown)" if include_area_chain else None

    grouped_entries: List[Dict[str, Any]] = []
    for node_index, entry in enumerate(node_entries):
        area_label = str(entry.get("area_label", "Unknown") or "Unknown").strip() or "Unknown"
        if (
            grouped_entries
            and area_label != "Unknown"
            and area_label == grouped_entries[-1]["area_label"]
        ):
            grouped_entries[-1]["tokens"].append(str(entry.get("token", "")))
            if entry.get("point") is not None:
                grouped_entries[-1]["last_point"] = entry.get("point")
            grouped_entries[-1]["has_current"] = (
                grouped_entries[-1]["has_current"] or bool(entry.get("is_current", False))
            )
            continue

        grouped_entries.append({
            "area_label": area_label,
            "tokens": [str(entry.get("token", ""))],
            "first_point": entry.get("point"),
            "last_point": entry.get("point"),
            "has_current": bool(entry.get("is_current", False)),
            "start_node_index": int(node_index),
        })

    def _format_group(entry: Dict[str, Any]) -> str:
        area_label = str(entry.get("area_label", "Unknown") or "Unknown").strip() or "Unknown"
        tokens = [token for token in entry.get("tokens", []) if token]
        token_chain = "->".join(tokens)
        if area_label == "Unknown" and tokens == ["Current"]:
            return "Current(Unknown)"
        if not token_chain:
            return area_label
        return f"{area_label} ({token_chain})"

    if len(grouped_entries) == 1:
        return "Space Waypoint Path: " + _format_group(grouped_entries[0])

    parts: List[str] = [_format_group(grouped_entries[0])]
    for index in range(1, len(grouped_entries)):
        prev_entry = grouped_entries[index - 1]
        curr_entry = grouped_entries[index]
        prev_point = prev_entry.get("last_point")
        curr_point = curr_entry.get("first_point")
        prev_group_start = int(prev_entry.get("start_node_index", max(index - 1, 0)))
        prev_group_len = len(prev_entry.get("tokens", []))
        prev_group_end = prev_group_start + max(prev_group_len - 1, 0)

        if prev_group_end >= 1 and prev_group_end < len(node_entries) and prev_point is not None and curr_point is not None:
            prior_point = node_entries[prev_group_end - 1].get("point")
            if prior_point is not None:
                previous_heading_deg = _segment_heading_deg(
                    start_point=prior_point,
                    end_point=prev_point,
                    resolution_cm=resolution_cm,
                )
                current_heading_deg = _segment_heading_deg(
                    start_point=prev_point,
                    end_point=curr_point,
                    resolution_cm=resolution_cm,
                )
                turn_delta_deg = normalize_relative_bearing(previous_heading_deg - current_heading_deg)
                turn_text = _format_turn_step(turn_delta_deg)
                if turn_text:
                    parts.append(turn_text)
        if prev_point is not None and curr_point is not None:
            distance_m = _segment_distance_m(
                start_point=prev_point,
                end_point=curr_point,
                resolution_cm=resolution_cm,
            )
            parts.append(f" move {distance_m:.1f}m to ")
        else:
            parts.append(" -> ")
        parts.append(_format_group(curr_entry))

    return "Space Waypoint Path: " + "".join(parts)


def build_action_landmark_map_info(
    step_landmark_entries: Sequence[Dict[str, Any]],
    landmark_dist_map: Optional[Dict[str, Tuple[float, float]]] = None,
    landmark_dist_map_multi: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    landmark_instances_world: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Build the action prompt's visible/off-screen landmark summary."""
    landmark_dist_map = landmark_dist_map or {}
    landmark_dist_map_multi = landmark_dist_map_multi or {}
    landmark_instances_world = landmark_instances_world or []
    if not (step_landmark_entries or landmark_dist_map or landmark_dist_map_multi or landmark_instances_world):
        return None

    if step_landmark_entries and any(entry.get("source") in {"vis", "off"} for entry in step_landmark_entries):
        cls_totals: Dict[str, int] = {}
        for item in landmark_instances_world:
            cls_name = item.get("name")
            if cls_name is None:
                continue
            cls_totals[str(cls_name)] = cls_totals.get(str(cls_name), 0) + 1
        for entry in step_landmark_entries:
            cls_name = entry.get("name")
            if cls_name is None:
                continue
            cls_key = str(cls_name)
            cls_totals[cls_key] = max(
                cls_totals.get(cls_key, 0),
                int(entry.get("class_total", 1) or 1),
                int(_maybe_int(entry.get("instance_idx")) or 0) + 1,
                sum(1 for other in step_landmark_entries if str(other.get("name")) == cls_key),
            )

        lines: List[str] = []
        for entry in step_landmark_entries:
            name = entry.get("name")
            distance_m = entry.get("distance_m")
            angle_deg = entry.get("angle_deg")
            if name is None or distance_m is None or angle_deg is None:
                continue
            try:
                cls_name = str(name)
                dist_m = float(distance_m)
                rel_angle_deg = float(angle_deg)
                confidence = float(entry.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue

            instance_idx = _maybe_int(entry.get("instance_idx"))
            suffix = ""
            if instance_idx is not None and cls_totals.get(cls_name, 0) > 1:
                suffix = f" #{instance_idx + 1}"
            source = "vis" if str(entry.get("source", "vis")) == "vis" else "off vis"
            lines.append(
                f"  - {source} {cls_name}{suffix}: {dist_m:.1f}m, "
                f"{format_relative_direction(rel_angle_deg)}, confidence: {confidence:.3f}"
                f"{build_landmark_turn_hint(rel_angle_deg, is_visible=(source == 'vis'))}"
            )
        return "\n".join(lines) if lines else None

    visible_entries: List[Tuple[str, float, float, Optional[int], float]] = []
    for entry in step_landmark_entries:
        name = entry.get("name")
        distance_m = entry.get("distance_m")
        angle_deg = entry.get("angle_deg")
        if name is None or distance_m is None or angle_deg is None:
            continue
        try:
            visible_entries.append(
                (
                    str(name),
                    float(distance_m),
                    float(angle_deg),
                    _maybe_int(entry.get("instance_idx")),
                    float(entry.get("confidence", 0.0)),
                )
            )
        except (TypeError, ValueError):
            continue

    visible_entries.sort(key=lambda item: item[1])
    visible_instance_indices: Dict[str, set] = {}
    lines: List[str] = []

    for cls_name, dist_m, angle_deg, instance_idx, confidence in visible_entries:
        if instance_idx is not None:
            visible_instance_indices.setdefault(cls_name, set()).add(instance_idx)
        same_cls_count = len(landmark_dist_map_multi.get(cls_name, [])) or sum(
            1 for name, *_rest in visible_entries if name == cls_name
        )
        suffix = f" #{instance_idx + 1}" if instance_idx is not None and same_cls_count > 1 else ""
        lines.append(
            f"  - vis {cls_name}{suffix}: {dist_m:.1f}m, "
            f"{format_relative_direction(angle_deg)}, confidence: {confidence:.3f}"
            f"{build_landmark_turn_hint(angle_deg, is_visible=True)}"
        )

    if landmark_instances_world:
        offscreen_items: List[Tuple[str, str, float, float, float]] = []
        for item in sorted(
            landmark_instances_world,
            key=lambda entry: float(entry.get("distance_m", 1e9)),
        ):
            cls_name = item.get("name")
            instance_idx = _maybe_int(item.get("instance_idx"))
            distance_m = item.get("distance_m")
            angle_deg = item.get("angle_deg")
            if cls_name is None or instance_idx is None or distance_m is None or angle_deg is None:
                continue
            if instance_idx in visible_instance_indices.get(str(cls_name), set()):
                continue
            same_cls_count = sum(1 for inst in landmark_instances_world if inst.get("name") == cls_name)
            suffix = f" #{instance_idx + 1}" if same_cls_count > 1 else ""
            offscreen_items.append(
                (
                    str(cls_name),
                    suffix,
                    float(distance_m),
                    float(angle_deg),
                    float(item.get("confidence", 0.0)),
                )
            )

        for cls_name, suffix, dist_m, angle_deg, confidence in offscreen_items:
            lines.append(
                f"  - off vis {cls_name}{suffix}: {dist_m:.1f}m, "
                f"{format_relative_direction(angle_deg)}, confidence: {confidence:.3f}"
                f"{build_landmark_turn_hint(angle_deg)}"
            )
    elif landmark_dist_map_multi:
        offscreen_items: List[Tuple[str, str, float, float]] = []
        for cls_name, candidates in sorted(
            landmark_dist_map_multi.items(),
            key=lambda item: min([pair[0] for pair in item[1]]) if item[1] else 1e9,
        ):
            if not candidates:
                continue
            used_set = visible_instance_indices.get(cls_name, set())
            sorted_candidates = sorted(candidates, key=lambda pair: pair[0])
            cls_total = len(sorted_candidates)
            for index, (dist_m, angle_deg) in enumerate(sorted_candidates):
                if index in used_set:
                    continue
                suffix = f" #{index + 1}" if cls_total > 1 else ""
                offscreen_items.append((cls_name, suffix, float(dist_m), float(angle_deg)))

        for cls_name, suffix, dist_m, angle_deg in sorted(offscreen_items, key=lambda item: item[2]):
            lines.append(
                f"  - off vis {cls_name}{suffix}: {dist_m:.1f}m, "
                f"{format_relative_direction(angle_deg)}, confidence: 0.000"
                f"{build_landmark_turn_hint(angle_deg)}"
            )
    else:
        visible_names = {name for name, *_rest in visible_entries}
        for cls_name, (dist_m, angle_deg) in sorted(landmark_dist_map.items(), key=lambda item: item[1][0]):
            if cls_name in visible_names:
                continue
            lines.append(
                f"  - off vis {cls_name}: {dist_m:.1f}m, "
                f"{format_relative_direction(angle_deg)}, confidence: 0.000"
                f"{build_landmark_turn_hint(angle_deg)}"
            )

    return "\n".join(lines) if lines else None


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
