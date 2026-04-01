import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vlnce_baselines.config.core.params.spatial import (
    SPACE_AREA_CURRENT_WAYPOINT_MAX_DISTANCE_M,
    SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M,
    WAYPOINT_STRIP_SAME_DIRECTION_GAP_M,
    WAYPOINT_VISIBILITY_RADIUS_M,
    WAYPOINT_VISIBILITY_SAMPLES,
)
from vlnce_baselines.config.core.params.landmarks import LANDMARK_STRIP_TOPK
from vlnce_baselines.mapping.space_types import strip_space_type_variant_suffixes
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


def _counterclockwise_direction_sort_key(bearing_deg: float) -> Tuple[int, float]:
    """Sort bearings in counterclockwise 12-view order starting from Front."""
    snapped = snap_relative_bearing(bearing_deg)
    if abs(snapped) >= 165.0:
        snapped = 180
    direction_order = {
        0: 0,
        -30: 1,
        -60: 2,
        -90: 3,
        -120: 4,
        -150: 5,
        180: 6,
        150: 7,
        120: 8,
        90: 9,
        60: 10,
        30: 11,
    }
    direction_rank = direction_order.get(int(snapped), 12)
    return direction_rank, abs(normalize_relative_bearing(bearing_deg) - float(snapped))


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


def _clean_area_label(area_label: str) -> str:
    clean_area = str(area_label or "Unknown").strip() or "Unknown"
    if " [links:" in clean_area:
        clean_area = clean_area.split(" [links:", 1)[0].strip()
    return clean_area or "Unknown"


def _clean_waypoint_description(description: str) -> str:
    return str(
        strip_space_type_variant_suffixes(description) or description or ""
    ).strip()


def _split_waypoint_description(description: str) -> Tuple[str, str]:
    clean_desc = _clean_waypoint_description(description)
    if not clean_desc:
        return "", ""
    if " - " in clean_desc:
        space_text, local_text = clean_desc.split(" - ", 1)
        return space_text.strip(), local_text.strip()
    return clean_desc, ""


def _trim_local_place_prefix(local_text: str) -> str:
    clean_text = str(local_text or "").strip()
    lowered = clean_text.lower()
    for prefix in ("near ", "by ", "at "):
        if lowered.startswith(prefix):
            trimmed = clean_text[len(prefix):].strip()
            if trimmed:
                return trimmed
    return clean_text


def _format_space_waypoint_chain_member(
    waypoint_token: str,
    waypoint_description: str = "",
    is_current: bool = False,
) -> str:
    if is_current:
        return "Current"

    clean_desc = _clean_waypoint_description(waypoint_description)
    _space_text, local_text = _split_waypoint_description(clean_desc)
    member_label = _trim_local_place_prefix(local_text) or clean_desc or "waypoint"
    if waypoint_token:
        return f"{waypoint_token}: {member_label}"
    return member_label


def _format_space_waypoint_chain_group(area_label: str, member_labels: Sequence[str]) -> str:
    clean_area = _clean_area_label(area_label)
    members = [str(item).strip() for item in member_labels if str(item).strip()]
    if not members:
        return clean_area
    return f"{clean_area} ({' -> '.join(members)})"


def _build_current_area_initial_waypoint_note(
    waypoint_positions: Sequence[Tuple[int, int]],
    waypoint_ids: Sequence[int],
    waypoint_area_labels: Optional[Sequence[str]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    current_space_area_label: str,
    full_map: Optional[np.ndarray],
    crop_offset: Optional[Tuple[int, int]],
    initial_waypoint_index: Optional[int] = 0,
) -> str:
    if (
        not waypoint_positions
        or current_pose is None
        or initial_waypoint_index is None
        or int(initial_waypoint_index) < 0
        or int(initial_waypoint_index) >= len(waypoint_positions)
    ):
        return ""

    initial_waypoint = waypoint_positions[int(initial_waypoint_index)]
    if initial_waypoint is None:
        return ""

    current_area = _clean_area_label(current_space_area_label)
    initial_area = _clean_area_label(
        waypoint_area_labels[int(initial_waypoint_index)]
        if waypoint_area_labels and int(initial_waypoint_index) < len(waypoint_area_labels)
        else ""
    )
    if not current_area or current_area == "Unknown" or current_area != initial_area:
        return ""

    wp_py, wp_px = int(initial_waypoint[0]), int(initial_waypoint[1])
    curr_x_m, curr_y_m, _ = current_pose[:3]
    curr_py = int(round(float(curr_y_m) * 100.0 / float(resolution_cm)))
    curr_px = int(round(float(curr_x_m) * 100.0 / float(resolution_cm)))
    distance_m = float(np.hypot(float(curr_py) - float(wp_py), float(curr_px) - float(wp_px))) * float(resolution_cm) / 100.0
    if distance_m > float(SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M) + 1e-6:
        return ""

    clear_path = _has_clear_path_to_waypoint(
        waypoint_py=wp_py,
        waypoint_px=wp_px,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        full_map=full_map,
        crop_offset=crop_offset,
    )
    if clear_path is False:
        return ""

    initial_wp_id = (
        int(waypoint_ids[int(initial_waypoint_index)])
        if waypoint_ids and int(initial_waypoint_index) < len(waypoint_ids)
        else 1
    )
    return (
        f" (near INITIAL POSITION Space WP#{initial_wp_id}; "
        "leave the initial-position neighborhood and continue toward the next task-relevant space waypoint)"
    )


def _waypoint_distance_to_current_m(
    waypoint_index: int,
    waypoint_positions: Sequence[Tuple[int, int]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
) -> Optional[float]:
    if current_pose is None or waypoint_index >= len(waypoint_positions):
        return None
    wp_py, wp_px = waypoint_positions[waypoint_index]
    wp_x_m = float(wp_px) * float(resolution_cm) / 100.0
    wp_y_m = float(wp_py) * float(resolution_cm) / 100.0
    curr_x_m, curr_y_m, _ = current_pose[:3]
    return float(math.hypot(wp_x_m - float(curr_x_m), wp_y_m - float(curr_y_m)))


def _waypoint_relative_bearing_deg(
    waypoint_index: int,
    waypoint_positions: Sequence[Tuple[int, int]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
) -> Optional[float]:
    if current_pose is None or waypoint_index >= len(waypoint_positions):
        return None

    curr_x_m, curr_y_m, curr_orientation_deg = [float(v) for v in current_pose[:3]]
    wp_py, wp_px = waypoint_positions[waypoint_index]
    wp_x_m = float(wp_px) * float(resolution_cm) / 100.0
    wp_y_m = float(wp_py) * float(resolution_cm) / 100.0
    absolute_angle_deg = float(math.degrees(math.atan2(wp_y_m - curr_y_m, wp_x_m - curr_x_m)))
    return float(curr_orientation_deg - absolute_angle_deg)


def _should_skip_display_waypoint(
    candidate_index: int,
    newer_kept_index: int,
    waypoint_positions: Sequence[Tuple[int, int]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
) -> bool:
    if candidate_index >= len(waypoint_positions) or newer_kept_index >= len(waypoint_positions):
        return False
    if current_pose is None:
        return False

    candidate_bearing_deg = _waypoint_relative_bearing_deg(
        waypoint_index=candidate_index,
        waypoint_positions=waypoint_positions,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
    )
    newer_bearing_deg = _waypoint_relative_bearing_deg(
        waypoint_index=newer_kept_index,
        waypoint_positions=waypoint_positions,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
    )
    if candidate_bearing_deg is None or newer_bearing_deg is None:
        return False
    if snap_relative_bearing(candidate_bearing_deg) != snap_relative_bearing(newer_bearing_deg):
        return False

    candidate_distance_m = _waypoint_distance_to_current_m(
        waypoint_index=candidate_index,
        waypoint_positions=waypoint_positions,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
    )
    newer_distance_m = _waypoint_distance_to_current_m(
        waypoint_index=newer_kept_index,
        waypoint_positions=waypoint_positions,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
    )
    if candidate_distance_m is None or newer_distance_m is None:
        return False
    return abs(float(candidate_distance_m) - float(newer_distance_m)) <= float(
        WAYPOINT_STRIP_SAME_DIRECTION_GAP_M
    ) + 1e-6


def _is_waypoint_overlapping_current_display(
    waypoint_index: int,
    waypoint_positions: Sequence[Tuple[int, int]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    full_map: Optional[np.ndarray],
    crop_offset: Optional[Tuple[int, int]],
) -> bool:
    if current_pose is None or waypoint_index >= len(waypoint_positions):
        return False

    distance_m = _waypoint_distance_to_current_m(
        waypoint_index=waypoint_index,
        waypoint_positions=waypoint_positions,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
    )
    if distance_m is None:
        return False

    max_distance_m = (
        float(SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M)
        if int(waypoint_index) == 0
        else float(SPACE_AREA_CURRENT_WAYPOINT_MAX_DISTANCE_M)
    )
    if distance_m > max_distance_m + 1e-6:
        return False

    clear_path = _has_clear_path_to_waypoint(
        waypoint_py=int(waypoint_positions[waypoint_index][0]),
        waypoint_px=int(waypoint_positions[waypoint_index][1]),
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        full_map=full_map,
        crop_offset=crop_offset,
    )
    return clear_path is True


def _find_current_area_waypoint_anchor_index(
    waypoint_positions: Sequence[Tuple[int, int]],
    waypoint_area_labels: Optional[Sequence[str]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    current_space_area_label: str = "",
    full_map: Optional[np.ndarray] = None,
    crop_offset: Optional[Tuple[int, int]] = None,
) -> Optional[int]:
    if current_pose is None or not waypoint_positions:
        return None

    target_area_label = _clean_area_label(current_space_area_label)
    area_labels = list(waypoint_area_labels or [])
    matching_candidates: List[Tuple[float, int]] = []
    fallback_candidates: List[Tuple[float, int, str]] = []

    for index, waypoint_pos in enumerate(waypoint_positions):
        if waypoint_pos is None:
            continue
        area_label = str(area_labels[index] if index < len(area_labels) else "").strip()
        clean_area_label = _clean_area_label(area_label)
        if clean_area_label == "Unknown":
            continue

        distance_m = _waypoint_distance_to_current_m(
            waypoint_index=index,
            waypoint_positions=waypoint_positions,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
        )
        if distance_m is None:
            continue
        max_distance_m = (
            float(SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M)
            if int(index) == 0 else
            float(SPACE_AREA_CURRENT_WAYPOINT_MAX_DISTANCE_M)
        )
        if distance_m > max_distance_m + 1e-6:
            continue

        clear_path = _has_clear_path_to_waypoint(
            waypoint_py=int(waypoint_pos[0]),
            waypoint_px=int(waypoint_pos[1]),
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            full_map=full_map,
            crop_offset=crop_offset,
        )
        if clear_path is False:
            continue

        if target_area_label != "Unknown" and clean_area_label == target_area_label:
            matching_candidates.append((float(distance_m), int(index)))
        fallback_candidates.append((float(distance_m), int(index), area_label))

    if matching_candidates:
        matching_candidates.sort(key=lambda item: (float(item[0]), int(item[1]) == 0, int(item[1])))
        return int(matching_candidates[0][1])

    if target_area_label != "Unknown":
        return None

    if not fallback_candidates:
        return None

    fallback_candidates.sort(key=lambda item: (float(item[0]), int(item[1]) == 0, int(item[1])))
    return int(fallback_candidates[0][1])


def resolve_display_current_area(
    waypoint_positions: Sequence[Tuple[int, int]],
    waypoint_area_labels: Optional[Sequence[str]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    current_space_area_label: str = "",
    full_map: Optional[np.ndarray] = None,
    crop_offset: Optional[Tuple[int, int]] = None,
) -> Tuple[str, Optional[int]]:
    current_area_label = str(current_space_area_label or "").strip()
    anchor_index = _find_current_area_waypoint_anchor_index(
        waypoint_positions=waypoint_positions,
        waypoint_area_labels=waypoint_area_labels,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        current_space_area_label=current_space_area_label,
        full_map=full_map,
        crop_offset=crop_offset,
    )
    if anchor_index is not None and waypoint_area_labels and anchor_index < len(waypoint_area_labels):
        anchor_label = str(waypoint_area_labels[anchor_index] or "").strip()
        if anchor_label:
            if _clean_area_label(current_area_label) == "Unknown":
                return anchor_label, int(anchor_index)
            if _clean_area_label(anchor_label) == _clean_area_label(current_area_label):
                return anchor_label, int(anchor_index)
    return current_area_label or "Unknown", anchor_index


def select_display_waypoint_indices(
    waypoint_positions: Sequence[Tuple[int, int]],
    waypoint_ids: Sequence[int],
    waypoint_descriptions: Sequence[str],
    waypoint_area_labels: Optional[Sequence[str]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    full_map: Optional[np.ndarray],
    crop_offset: Optional[Tuple[int, int]],
) -> List[int]:
    if not waypoint_ids:
        return []

    latest_anchor_index = len(waypoint_ids) - 1
    while latest_anchor_index > 0:
        if not _is_waypoint_overlapping_current_display(
            waypoint_index=latest_anchor_index,
            waypoint_positions=waypoint_positions,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            full_map=full_map,
            crop_offset=crop_offset,
        ):
            break
        latest_anchor_index -= 1

    kept_reversed: List[int] = [latest_anchor_index]
    for candidate_index in range(latest_anchor_index - 1, -1, -1):
        if _is_waypoint_overlapping_current_display(
            waypoint_index=candidate_index,
            waypoint_positions=waypoint_positions,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            full_map=full_map,
            crop_offset=crop_offset,
        ):
            continue
        if any(
            _should_skip_display_waypoint(
                candidate_index=candidate_index,
                newer_kept_index=newer_kept_index,
                waypoint_positions=waypoint_positions,
                current_pose=current_pose,
                resolution_cm=resolution_cm,
            )
            for newer_kept_index in kept_reversed
        ):
            continue
        kept_reversed.append(candidate_index)

    display_indices = sorted(set(kept_reversed))
    if latest_anchor_index not in display_indices:
        display_indices.append(latest_anchor_index)
    return display_indices


def _normalize_waypoint_floor_ids(
    waypoint_ids: Sequence[int],
    waypoint_floor_ids: Optional[Sequence[int]],
    current_floor_id: int,
) -> List[int]:
    raw_floor_ids = list(waypoint_floor_ids or [])
    normalized = [
        int(raw_floor_ids[index]) if index < len(raw_floor_ids) else int(current_floor_id)
        for index in range(len(waypoint_ids))
    ]
    return normalized


def _format_floor_label(floor_id: int) -> str:
    return f"F{int(floor_id) + 1}"


def _coerce_floor_id(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _format_area_with_floor(
    area_label: str,
    floor_id: int,
    multi_floor_active: bool,
) -> str:
    clean_area = _clean_area_label(area_label)
    if not multi_floor_active:
        return clean_area
    return f"{clean_area} [{_format_floor_label(floor_id)}]"


def _find_stair_connector_transition_label(
    from_floor_id: int,
    to_floor_id: int,
    stair_connectors: Optional[Sequence[Dict[str, Any]]],
) -> str:
    start_floor = int(from_floor_id)
    end_floor = int(to_floor_id)
    for connector in list(stair_connectors or []):
        connector_from = int(connector.get("from_floor_id", 0) or 0)
        connector_to = int(connector.get("to_floor_id", 0) or 0)
        if connector_from == start_floor and connector_to == end_floor:
            label = str(connector.get("label", "stairs")).strip() or "stairs"
            return f"{label}({_format_floor_label(start_floor)}->{_format_floor_label(end_floor)})"
        if connector_from == end_floor and connector_to == start_floor:
            label = str(connector.get("label", "stairs")).strip() or "stairs"
            return f"{label}({_format_floor_label(start_floor)}->{_format_floor_label(end_floor)})"
    return f"stairs({_format_floor_label(start_floor)}->{_format_floor_label(end_floor)})"


def build_waypoint_summary(
    waypoint_positions: Sequence[Tuple[int, int]],
    waypoint_ids: Sequence[int],
    waypoint_descriptions: Sequence[str],
    waypoint_area_labels: Optional[Sequence[str]],
    waypoint_floor_ids: Optional[Sequence[int]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    current_space_area_label: str = "",
    current_space_area_type: str = "",
    full_map: Optional[np.ndarray] = None,
    crop_offset: Optional[Tuple[int, int]] = None,
    initial_waypoint_index: Optional[int] = 0,
    current_world_z: Optional[float] = None,
    current_floor_id: int = 0,
    multi_floor_active: bool = False,
    on_stairs_connector: bool = False,
    stair_connectors: Optional[Sequence[Dict[str, Any]]] = None,
    include_area_chain: bool = True,
    include_path: bool = True,
) -> str:
    """Summarize visited waypoints relative to the current pose as a maintained chain."""
    header_lines: List[str] = []
    display_area_label = current_space_area_label or "Unknown"
    display_area_type = current_space_area_type or "Unknown"
    normalized_floor_ids = _normalize_waypoint_floor_ids(
        waypoint_ids=waypoint_ids,
        waypoint_floor_ids=waypoint_floor_ids,
        current_floor_id=current_floor_id,
    )
    space_type_note = (
        f" ({display_area_type})"
        if display_area_type and display_area_type != "Unknown" and display_area_type != display_area_label
        else ""
    )
    current_floor_global_indices = [
        index
        for index, floor_id in enumerate(normalized_floor_ids)
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
        _clean_area_label(
            waypoint_area_labels[index]
            if waypoint_area_labels and index < len(waypoint_area_labels) else ""
        )
        for index in current_floor_global_indices
    ]
    current_floor_initial_index = (
        current_floor_index_map.get(int(initial_waypoint_index))
        if initial_waypoint_index is not None
        else None
    )
    current_area_initial_note = _build_current_area_initial_waypoint_note(
        waypoint_positions=current_floor_positions,
        waypoint_ids=current_floor_ids,
        waypoint_area_labels=current_floor_area_labels,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        current_space_area_label=current_space_area_label,
        full_map=full_map,
        crop_offset=crop_offset,
        initial_waypoint_index=current_floor_initial_index,
    )
    header_lines.append(
        f"Your Current Area: {display_area_label}{space_type_note}{current_area_initial_note}"
    )
    if multi_floor_active:
        floor_line = f"Current Floor: F{int(current_floor_id) + 1}"
        if on_stairs_connector:
            floor_line += " | stair connector active"
        if stair_connectors:
            connector_bits: List[str] = []
            for connector in list(stair_connectors or []):
                lower = dict(connector.get("lower_landing", {}) or {})
                upper = dict(connector.get("upper_landing", {}) or {})
                lower_label = str(lower.get("floor_label", f"F{int(lower.get('floor_id', 0)) + 1}"))
                upper_label = str(upper.get("floor_label", f"F{int(upper.get('floor_id', 0)) + 1}"))
                connector_bits.append(f"{str(connector.get('label', 'stairs'))}: {lower_label}->{upper_label}")
            if connector_bits:
                floor_line += " | Stair Connectors: " + "; ".join(connector_bits)
        header_lines.append(floor_line)

    empty_area_chain_line = None
    if include_area_chain:
        current_area_display = _clean_area_label(current_space_area_label)
        empty_area_chain_line = f"Space Waypoint Chain: {current_area_display} (Current)"

    if not waypoint_ids:
        lines = list(header_lines)
        if empty_area_chain_line:
            lines.append(empty_area_chain_line)
        lines.append("No space waypoints recorded yet.")
        return "\n".join(lines)

    current_floor_waypoint_distances_m = [
        _waypoint_distance_to_current_m(
            waypoint_index=index,
            waypoint_positions=current_floor_positions,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
        )
        for index in range(len(current_floor_ids))
    ]

    resolved_current_area_label, _current_area_anchor_index = resolve_display_current_area(
        waypoint_positions=current_floor_positions,
        waypoint_area_labels=current_floor_area_labels,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        current_space_area_label=current_space_area_label,
        full_map=full_map,
        crop_offset=crop_offset,
    )
    display_area_label = resolved_current_area_label or display_area_label
    current_area_initial_note = _build_current_area_initial_waypoint_note(
        waypoint_positions=current_floor_positions,
        waypoint_ids=current_floor_ids,
        waypoint_area_labels=current_floor_area_labels,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        current_space_area_label=display_area_label,
        full_map=full_map,
        crop_offset=crop_offset,
        initial_waypoint_index=current_floor_initial_index,
    )
    header_lines[0] = f"Your Current Area: {display_area_label}{space_type_note}{current_area_initial_note}"

    current_floor_visible_local_indices = select_display_waypoint_indices(
        waypoint_positions=current_floor_positions,
        waypoint_ids=current_floor_ids,
        waypoint_descriptions=current_floor_descriptions,
        waypoint_area_labels=current_floor_area_labels,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        full_map=full_map,
        crop_offset=crop_offset,
    )
    last_visible_global_index = (
        current_floor_global_indices[current_floor_visible_local_indices[-1]]
        if current_floor_visible_local_indices
        else None
    )
    current_floor_display_global_indices = [
        current_floor_global_indices[index]
        for index in current_floor_visible_local_indices
    ]
    if current_pose is not None:
        current_floor_display_global_indices.sort(
            key=lambda index: (
                _counterclockwise_direction_sort_key(
                    float(current_pose[2]) - math.degrees(
                        math.atan2(
                            (
                                float(waypoint_positions[index][0]) * float(resolution_cm) / 100.0
                                - float(current_pose[1])
                            ),
                            (
                                float(waypoint_positions[index][1]) * float(resolution_cm) / 100.0
                                - float(current_pose[0])
                            ),
                        )
                    )
                ),
                float(
                    current_floor_waypoint_distances_m[current_floor_index_map[index]]
                )
                if index in current_floor_index_map
                and current_floor_waypoint_distances_m[current_floor_index_map[index]] is not None
                else float("inf"),
                int(waypoint_ids[index]) if index < len(waypoint_ids) else int(index),
            )
        )

    def _waypoint_ccw_sort_key(global_index: int) -> Tuple[int, float, int]:
        if current_pose is None:
            return (0, 0.0, int(waypoint_ids[global_index]) if global_index < len(waypoint_ids) else int(global_index))
        wp_py, wp_px = waypoint_positions[global_index]
        return (
            _counterclockwise_direction_sort_key(
                float(current_pose[2]) - math.degrees(
                    math.atan2(
                        (
                            float(wp_py) * float(resolution_cm) / 100.0
                            - float(current_pose[1])
                        ),
                        (
                            float(wp_px) * float(resolution_cm) / 100.0
                            - float(current_pose[0])
                        ),
                    )
                )
            ),
            float(
                current_floor_waypoint_distances_m[current_floor_index_map[global_index]]
            )
            if global_index in current_floor_index_map
            and current_floor_waypoint_distances_m[current_floor_index_map[global_index]] is not None
            else float("inf"),
            int(waypoint_ids[global_index]) if global_index < len(waypoint_ids) else int(global_index),
        )

    other_floor_groups: Dict[int, List[int]] = {}
    for index, floor_id in enumerate(normalized_floor_ids):
        if int(floor_id) == int(current_floor_id):
            continue
        other_floor_groups.setdefault(int(floor_id), []).append(int(index))

    other_floor_group_indices: List[List[int]] = []
    for floor_id in sorted(other_floor_groups.keys()):
        group = other_floor_groups[floor_id]
        group.sort(key=_waypoint_ccw_sort_key)
        other_floor_group_indices.append(group)

    grouped_display_indices: List[List[int]] = []
    grouped_floor_ids: List[int] = []
    if current_floor_display_global_indices:
        grouped_display_indices.append(current_floor_display_global_indices)
        grouped_floor_ids.append(int(current_floor_id))
    for floor_id in sorted(other_floor_groups.keys()):
        grouped_display_indices.append(other_floor_groups[floor_id])
        grouped_floor_ids.append(int(floor_id))

    node_lines: List[str] = []
    for group_idx, group in enumerate(grouped_display_indices):
        group_floor_id = grouped_floor_ids[group_idx] if group_idx < len(grouped_floor_ids) else int(current_floor_id)
        if multi_floor_active:
            node_lines.append(f"--- Floor {int(group_floor_id) + 1} ---")
        for index in group:
            wp_id = waypoint_ids[index]
            wp_desc = _clean_waypoint_description(
                waypoint_descriptions[index] if index < len(waypoint_descriptions) else ""
            )
            floor_id = int(normalized_floor_ids[index]) if index < len(normalized_floor_ids) else int(current_floor_id)
            floor_label = _format_floor_label(floor_id)
            is_last = last_visible_global_index is not None and index == last_visible_global_index
            suffix_notes: List[str] = []
            if initial_waypoint_index is not None and index == int(initial_waypoint_index):
                suffix_notes.append("INITIAL POSITION")
            if is_last and index != 0:
                suffix_notes.append("LAST POSITION")
            suffix = f"  <- {' | '.join(suffix_notes)}" if suffix_notes else ""

            area_label = _clean_area_label(
                waypoint_area_labels[index]
                if waypoint_area_labels and index < len(waypoint_area_labels) else ""
            )
            area_note = f" | area={area_label}" if area_label else ""
            if multi_floor_active:
                area_note += f" | floor={floor_label}"

            if floor_id != int(current_floor_id):
                transition_label = _find_stair_connector_transition_label(
                    from_floor_id=current_floor_id,
                    to_floor_id=floor_id,
                    stair_connectors=stair_connectors,
                )
                spatial_info = f"historical waypoint on {floor_label} | reach via {transition_label}"
            elif current_pose is None:
                spatial_info = "distance unknown"
            else:
                local_index = current_floor_index_map.get(index)
                wp_py, wp_px = waypoint_positions[index]
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
                if local_index is not None:
                    reachability_note = _build_waypoint_reachability_note(
                        waypoint_index=local_index,
                        waypoint_id=wp_id,
                        waypoint_ids=current_floor_ids,
                        waypoint_positions=current_floor_positions,
                        waypoint_area_labels=current_floor_area_labels,
                        current_pose=current_pose,
                        resolution_cm=resolution_cm,
                        full_map=full_map,
                        crop_offset=crop_offset,
                        visible_indices=current_floor_visible_local_indices,
                        current_space_area_label=display_area_label,
                    )
                    if reachability_note:
                        spatial_info = f"{spatial_info} | {reachability_note}"
            node_lines.append(f"Space WP#{wp_id} [{wp_desc}{area_note}] -- {spatial_info}{suffix}")

    current_area_display = _clean_area_label(display_area_label)
    waypoint_area_path_line = _build_waypoint_area_path_line(
        visible_waypoint_ids=waypoint_ids,
        visible_waypoint_positions=waypoint_positions,
        visible_waypoint_descriptions=waypoint_descriptions,
        visible_waypoint_area_labels=(
            [_clean_area_label(waypoint_area_labels[index]) for index in range(len(waypoint_ids))]
            if waypoint_area_labels else []
        ),
        visible_waypoint_floor_ids=normalized_floor_ids,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        current_area_display=current_area_display,
        include_area_chain=include_area_chain,
        include_path=include_path,
        current_floor_id=current_floor_id,
        multi_floor_active=multi_floor_active,
        stair_connectors=stair_connectors,
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
    clean_area = _clean_area_label(area_label)
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
        next_area = _clean_area_label(
            waypoint_area_labels[next_index]
            if next_index < len(waypoint_area_labels) else "Unknown"
        )
        return f"blocked to current position; reach via {_format_waypoint_area_ref(next_waypoint_id, next_area)}"

    if current_pose is not None:
        current_area_display = _clean_area_label(current_space_area_label)
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
    visible_waypoint_descriptions: Sequence[str],
    visible_waypoint_area_labels: Sequence[str],
    visible_waypoint_floor_ids: Optional[Sequence[int]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    current_area_display: str,
    include_area_chain: bool,
    include_path: bool,
    current_floor_id: int,
    multi_floor_active: bool,
    stair_connectors: Optional[Sequence[Dict[str, Any]]],
) -> Optional[str]:
    if not include_area_chain and not include_path:
        return None

    node_entries: List[Dict[str, Any]] = []
    normalized_floor_ids = _normalize_waypoint_floor_ids(
        waypoint_ids=visible_waypoint_ids,
        waypoint_floor_ids=visible_waypoint_floor_ids,
        current_floor_id=current_floor_id,
    )
    for index, wp_id in enumerate(visible_waypoint_ids):
        area_label = _clean_area_label(
            visible_waypoint_area_labels[index]
            if index < len(visible_waypoint_area_labels) else "Unknown"
        )
        waypoint_desc = (
            visible_waypoint_descriptions[index]
            if index < len(visible_waypoint_descriptions) else ""
        )
        node_entries.append({
            "area_label": area_label,
            "token": f"WP#{int(wp_id)}",
            "point": visible_waypoint_positions[index],
            "is_current": False,
            "description": waypoint_desc,
            "floor_id": int(normalized_floor_ids[index]) if index < len(normalized_floor_ids) else int(current_floor_id),
        })

    if current_pose is not None:
        curr_py = int(round(float(current_pose[1]) * 100.0 / float(resolution_cm)))
        curr_px = int(round(float(current_pose[0]) * 100.0 / float(resolution_cm)))
        node_entries.append({
            "area_label": current_area_display,
            "token": "Current",
            "point": (curr_py, curr_px),
            "is_current": True,
            "description": "",
            "floor_id": int(current_floor_id),
        })
    elif node_entries:
        node_entries.append({
            "area_label": current_area_display,
            "token": "Current",
            "point": None,
            "is_current": True,
            "description": "",
            "floor_id": int(current_floor_id),
        })

    if not node_entries:
        return "Space Waypoint Chain: Unknown (Current)" if include_area_chain else None

    grouped_entries: List[Dict[str, Any]] = []
    for entry in node_entries:
        area_label = _clean_area_label(str(entry.get("area_label", "Unknown") or "Unknown"))
        floor_id = _coerce_floor_id(entry.get("floor_id"), current_floor_id)
        point = entry.get("point")
        if (
            grouped_entries
            and area_label == grouped_entries[-1]["area_label"]
            and floor_id == grouped_entries[-1]["floor_id"]
        ):
            grouped_entries[-1]["members"].append(entry)
            if grouped_entries[-1]["first_point"] is None and point is not None:
                grouped_entries[-1]["first_point"] = point
            if point is not None:
                grouped_entries[-1]["last_point"] = point
            continue

        grouped_entries.append({
            "area_label": area_label,
            "members": [entry],
            "first_point": point,
            "last_point": point,
            "floor_id": floor_id,
        })

    def _format_group(entry: Dict[str, Any]) -> str:
        member_labels = [
            _format_space_waypoint_chain_member(
                waypoint_token=str(member.get("token", "")),
                waypoint_description=str(member.get("description", "") or ""),
                is_current=bool(member.get("is_current", False)),
            )
            for member in entry.get("members", [])
        ]
        return _format_space_waypoint_chain_group(
            area_label=_format_area_with_floor(
                area_label=str(entry.get("area_label", "Unknown") or "Unknown"),
                floor_id=_coerce_floor_id(entry.get("floor_id"), current_floor_id),
                multi_floor_active=multi_floor_active,
            ),
            member_labels=member_labels,
        )

    if len(grouped_entries) == 1:
        return "Space Waypoint Chain: " + _format_group(grouped_entries[0])

    parts: List[str] = [_format_group(grouped_entries[0])]
    for index in range(1, len(grouped_entries)):
        prev_group = grouped_entries[index - 1]
        curr_group = grouped_entries[index]
        prev_point = prev_group.get("last_point")
        curr_point = curr_group.get("first_point")
        prev_floor_id = _coerce_floor_id(prev_group.get("floor_id"), current_floor_id)
        curr_floor_id = _coerce_floor_id(curr_group.get("floor_id"), current_floor_id)

        if prev_floor_id != curr_floor_id:
            transition_label = _find_stair_connector_transition_label(
                from_floor_id=prev_floor_id,
                to_floor_id=curr_floor_id,
                stair_connectors=stair_connectors,
            )
            parts.append(f" -> {transition_label} -> ")
            parts.append(_format_group(curr_group))
            continue

        if include_path and index >= 2 and prev_point is not None and curr_point is not None:
            prior_point = grouped_entries[index - 2].get("last_point")
            prior_floor_id = _coerce_floor_id(
                grouped_entries[index - 2].get("floor_id"),
                current_floor_id,
            )
            if prior_point is not None and prior_floor_id == prev_floor_id:
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

        if include_path and prev_point is not None and curr_point is not None:
            distance_m = _segment_distance_m(
                start_point=prev_point,
                end_point=curr_point,
                resolution_cm=resolution_cm,
            )
            if distance_m < 0.05:
                parts.append(" -> ")
            else:
                parts.append(f" move {distance_m:.1f}m to ")
        else:
            parts.append(" -> ")
        parts.append(_format_group(curr_group))

    return "Space Waypoint Chain: " + "".join(parts)


def build_action_landmark_map_info(
    step_landmark_entries: Sequence[Dict[str, Any]],
    landmark_dist_map: Optional[Dict[str, Tuple[float, float]]] = None,
    landmark_dist_map_multi: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    landmark_instances_world: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Build the action prompt's landmark summary from the selected action top-k list."""
    _ = landmark_dist_map
    _ = landmark_dist_map_multi
    _ = landmark_instances_world
    if not step_landmark_entries:
        return None

    ordered_entries = [dict(entry) for entry in step_landmark_entries]
    ordered_entries.sort(
        key=lambda entry: (
            _maybe_int(entry.get("selection_rank"))
            if _maybe_int(entry.get("selection_rank")) is not None
            else 1e9,
            -float(entry.get("confidence", 0.0) or 0.0),
            float(entry.get("distance_m", 1e9) or 1e9),
            str(entry.get("name", "")),
        )
    )
    topk = max(1, int(LANDMARK_STRIP_TOPK))
    selected_entries: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for entry in ordered_entries:
        entry_id = entry.get("instance_uid", id(entry))
        if entry_id in seen_ids:
            continue
        selected_entries.append(entry)
        seen_ids.add(entry_id)
        if len(selected_entries) >= topk:
            break
    if not selected_entries:
        return None

    cls_totals: Dict[str, int] = {}
    for entry in selected_entries:
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

    parts: List[str] = []
    for entry in selected_entries:
        name = entry.get("name")
        distance_m = entry.get("distance_m")
        angle_deg = entry.get("angle_deg")
        if name is None or distance_m is None:
            continue
        try:
            cls_name = str(name)
            dist_m = float(distance_m)
            confidence = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        rel_angle_deg = _maybe_float(angle_deg)

        display_id = _maybe_int(entry.get("display_id"))
        instance_idx = _maybe_int(entry.get("instance_idx"))
        suffix = ""
        if display_id is not None and display_id > 0:
            suffix = f" #{display_id}"
        elif instance_idx is not None and cls_totals.get(cls_name, 0) > 1:
            suffix = f" #{instance_idx + 1}"
        source_tag = "vis" if str(entry.get("source", "vis")) == "vis" else "mem"
        direction_text = (
            format_relative_direction(rel_angle_deg)
            if rel_angle_deg is not None
            else "Unknown"
        )
        parts.append(
            f"{source_tag} {cls_name}{suffix}: {dist_m:.1f}m, "
            f"{direction_text}, conf: {confidence:.3f}"
        )
    return " || ".join(parts) if parts else None


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
