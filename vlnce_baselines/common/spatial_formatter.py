import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

CURRENT_AREA_OVERLAP_THRESHOLD_M = 2.0


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
    current_room_area_label: str = "",
    current_room_area_type: str = "",
    include_area_chain: bool = True,
    include_path: bool = True,
) -> str:
    """Summarize visited waypoints relative to the current pose."""
    header_lines: List[str] = []
    display_area_label = current_room_area_label or "Unknown"
    display_area_type = current_room_area_type or "Unknown"
    room_type_note = (
        f" ({display_area_type})"
        if display_area_type and display_area_type != "Unknown" and display_area_type != display_area_label
        else ""
    )
    header_lines.append(f"Your Current Area: {display_area_label}{room_type_note}")

    if not waypoint_ids:
        return "\n".join(header_lines + ["No waypoints visited yet."])

    current_area_known = bool(
        display_area_label
        and str(display_area_label).strip()
        and str(display_area_label).strip().lower() != "unknown"
    )
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
    if (current_area_known or close_last_waypoint) and visible_indices:
        visible_indices = visible_indices[:-1]
    last_visible_index = visible_indices[-1] if visible_indices else None

    node_lines: List[str] = []
    for index in visible_indices:
        wp_id = waypoint_ids[index]
        wp_desc = waypoint_descriptions[index]
        wp_py, wp_px = waypoint_positions[index]
        is_last = last_visible_index is not None and index == last_visible_index
        suffix = "  <- LAST VISITED (came from here)" if is_last else ""

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
        node_lines.append(f"WP#{wp_id} [{wp_desc}{area_note}] -- {spatial_info}{suffix}")

    path_segments: List[str] = []
    visible_waypoint_ids = [waypoint_ids[index] for index in visible_indices]
    visible_waypoint_positions = [waypoint_positions[index] for index in visible_indices]
    visible_waypoint_area_labels = (
        [waypoint_area_labels[index] for index in visible_indices]
        if waypoint_area_labels else []
    )

    for index in range(len(visible_waypoint_ids) - 1):
        py1, px1 = visible_waypoint_positions[index]
        py2, px2 = visible_waypoint_positions[index + 1]
        segment_distance_m = math.sqrt(
            ((px2 - px1) * resolution_cm / 100.0) ** 2
            + ((py2 - py1) * resolution_cm / 100.0) ** 2
        )
        path_segments.append(
            f"WP#{visible_waypoint_ids[index]}->WP#{visible_waypoint_ids[index + 1]}({segment_distance_m:.1f}m)"
        )

    first_waypoint_id = visible_waypoint_ids[0] if visible_waypoint_ids else None
    path_line = None
    if first_waypoint_id is not None:
        path_line = (
            "Path: " + " -> ".join(path_segments) + " -> Current"
            if path_segments
            else f"Path: WP#{first_waypoint_id} -> Current"
        )

    waypoint_area_path_line = None
    if include_area_chain:
        area_nodes: List[str] = []
        for index, wp_id in enumerate(visible_waypoint_ids):
            area_label = ""
            if index < len(visible_waypoint_area_labels):
                area_label = str(visible_waypoint_area_labels[index] or "").strip()
            if area_label:
                area_nodes.append(f"WP#{wp_id}({area_label})")
            else:
                area_nodes.append(f"WP#{wp_id}(Unknown)")
        current_area_display = str(current_room_area_label or "Unknown").strip() or "Unknown"
        area_nodes.append(f"Current({current_area_display})")
        waypoint_area_path_line = "Waypoint Area Path: " + " -> ".join(area_nodes)

    lines = header_lines + node_lines
    if waypoint_area_path_line:
        lines.append(waypoint_area_path_line)
    elif include_path and path_line:
        lines.append(path_line)
    return "\n".join(lines)


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
