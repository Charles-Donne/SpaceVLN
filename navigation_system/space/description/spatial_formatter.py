import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from navigation_system.space.description.direction_format import (
    build_landmark_turn_hint,
    format_relative_direction,
    normalize_relative_bearing,
    snap_relative_bearing,
)
from navigation_system.config.core.params.spatial import (
    SPACE_AREA_CURRENT_WAYPOINT_MAX_DISTANCE_M,
    SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M,
    WAYPOINT_STRIP_SAME_DIRECTION_GAP_M,
    WAYPOINT_VISIBILITY_RADIUS_M,
    WAYPOINT_VISIBILITY_SAMPLES,
)
from navigation_system.config.core.params.landmarks import LANDMARK_STRIP_TOPK
from navigation_system.space.geometry.connectivity import (
    build_bounded_geodesic_distance_field,
    query_world_distance_from_field_m,
)
from navigation_system.space.structure.space_types import (
    infer_space_type_from_texts,
    strip_space_type_variant_suffixes,
)
from navigation_system.space.geometry.map_projection import RotatedMapProjector


_UNRESOLVED_AREA_LABELS = {
    "unknown",
    "area",
    "room",
    "space",
    "zone",
    "section",
    "place",
    "location",
}


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


def _clean_area_label(area_label: str) -> str:
    clean_area = str(area_label or "Unknown").strip() or "Unknown"
    if " [links:" in clean_area:
        clean_area = clean_area.split(" [links:", 1)[0].strip()
    clean_area = _strip_visual_brackets(clean_area)
    return clean_area or "Unknown"


def _is_unknown_area_label(area_label: str) -> bool:
    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        _clean_area_label(area_label).strip().lower(),
    ).strip()
    if not normalized:
        return True
    if normalized in _UNRESOLVED_AREA_LABELS:
        return True
    if re.fullmatch(r"(?:area|room|space|zone|section)\s*\d+", normalized):
        return True
    if normalized.startswith("unknown"):
        return True
    if "not yet determined" in normalized or "to infer" in normalized:
        return True
    if "infer" in normalized and "view" in normalized:
        return True
    return False


def _resolve_display_area_label(
    area_label: str,
    area_type: str = "",
    cue_texts: Optional[Sequence[str]] = None,
) -> str:
    clean_area = _clean_area_label(area_label)
    if not _is_unknown_area_label(clean_area):
        return clean_area

    clean_type = _clean_area_label(area_type)
    if not _is_unknown_area_label(clean_type):
        return clean_type

    inferred_type = infer_space_type_from_texts(
        [clean_area, clean_type, *list(cue_texts or [])]
    )
    if not _is_unknown_area_label(inferred_type):
        return inferred_type
    return ""


def _format_area_display_label(
    area_label: str,
    area_type: str = "",
    fallback: str = "infer from current views",
    cue_texts: Optional[Sequence[str]] = None,
) -> str:
    resolved_label = _resolve_display_area_label(
        area_label=area_label,
        area_type=area_type,
        cue_texts=cue_texts,
    )
    if resolved_label:
        return resolved_label
    return str(fallback or "infer from current views").strip() or "infer from current views"


def _format_current_area_header_label(area_label: str, area_type: str = "") -> str:
    display_label = _resolve_display_area_label(area_label, area_type)
    if display_label:
        return display_label
    return "you need to infer actual area from current views"


def _format_current_area_chain_label(area_label: str) -> str:
    clean_area = _clean_area_label(area_label)
    if _is_unknown_area_label(clean_area):
        return "Current Position"
    return clean_area


def _clean_waypoint_description(description: str) -> str:
    cleaned = str(
        strip_space_type_variant_suffixes(description) or description or ""
    ).strip()
    return _strip_visual_brackets(cleaned)


def _strip_visual_brackets(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return cleaned
    cleaned = re.sub(r"[\[\【]\s*([^\[\]【】]+?)\s*[\]\】]", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _split_waypoint_description(description: str) -> Tuple[str, str]:
    clean_desc = _clean_waypoint_description(description)
    if not clean_desc:
        return "", ""
    if " - " in clean_desc:
        space_text, local_text = clean_desc.split(" - ", 1)
        return space_text.strip(), local_text.strip()
    return clean_desc, ""


def _format_waypoint_description_for_display(
    description: str,
    area_label: str = "",
    area_type: str = "",
) -> str:
    clean_desc = _clean_waypoint_description(description)
    if not clean_desc:
        return ""

    space_text, local_text = _split_waypoint_description(clean_desc)
    if _is_unknown_area_label(space_text):
        display_space = _format_area_display_label(
            area_label,
            area_type,
            cue_texts=[clean_desc, local_text],
        )
        if local_text:
            return f"{display_space} - {local_text}"
        return display_space
    return clean_desc


def _trim_local_place_prefix(local_text: str) -> str:
    clean_text = str(local_text or "").strip()
    lowered = clean_text.lower()
    for prefix in ("near ", "by ", "at "):
        if lowered.startswith(prefix):
            trimmed = clean_text[len(prefix):].strip()
            if trimmed:
                return trimmed
    return clean_text


def _format_spatial_waypoint_chain_member(
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


def _format_spatial_waypoint_chain_group(area_label: str, member_labels: Sequence[str]) -> str:
    members = [str(item).strip() for item in member_labels if str(item).strip()]
    clean_area = _format_area_display_label(area_label, cue_texts=members)
    if not members:
        return clean_area
    return f"{clean_area} ({' -> '.join(members)})"


def _build_current_area_initial_waypoint_note(
    waypoint_positions: Sequence[Tuple[int, int]],
    waypoint_ids: Sequence[int],
    waypoint_area_labels: Optional[Sequence[str]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    current_region_label: str,
    full_map: Optional[np.ndarray],
    crop_offset: Optional[Tuple[int, int]],
    initial_waypoint_index: Optional[int] = 0,
    obstacle_mask: Optional[np.ndarray] = None,
    projector: Optional[RotatedMapProjector] = None,
    current_distance_field: Optional[np.ndarray] = None,
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

    current_area = _clean_area_label(current_region_label)
    initial_area = _clean_area_label(
        waypoint_area_labels[int(initial_waypoint_index)]
        if waypoint_area_labels and int(initial_waypoint_index) < len(waypoint_area_labels)
        else ""
    )
    if not current_area or current_area == "Unknown" or current_area != initial_area:
        return ""

    wp_py, wp_px = int(initial_waypoint[0]), int(initial_waypoint[1])
    distance_m: Optional[float] = None
    if current_distance_field is not None and obstacle_mask is not None and projector is not None:
        distance_m = query_world_distance_from_field_m(
            distance_field=current_distance_field,
            obstacle_mask=obstacle_mask,
            projector=projector,
            target_world=(wp_py, wp_px),
            resolution_cm=resolution_cm,
            target_radius_m=WAYPOINT_VISIBILITY_RADIUS_M,
            target_samples=WAYPOINT_VISIBILITY_SAMPLES,
        )
    if distance_m is None:
        curr_x_m, curr_y_m, _ = current_pose[:3]
        curr_py = int(round(float(curr_y_m) * 100.0 / float(resolution_cm)))
        curr_px = int(round(float(curr_x_m) * 100.0 / float(resolution_cm)))
        distance_m = (
            float(np.hypot(float(curr_py) - float(wp_py), float(curr_px) - float(wp_px)))
            * float(resolution_cm)
            / 100.0
        )
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
    elif distance_m > float(SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M) + 1e-6:
        return ""

    initial_wp_id = (
        int(waypoint_ids[int(initial_waypoint_index)])
        if waypoint_ids and int(initial_waypoint_index) < len(waypoint_ids)
        else 1
    )
    return (
        f" (still near INITIAL POSITION(Task start) Spatial WP#{initial_wp_id}; "
        "do not set global_task_finish=true; "
        "execute Task first stage first)"
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
    return abs(float(candidate_distance_m) - float(newer_distance_m)) < float(
        WAYPOINT_STRIP_SAME_DIRECTION_GAP_M
    )


def _is_waypoint_overlapping_current_display(
    waypoint_index: int,
    waypoint_positions: Sequence[Tuple[int, int]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    full_map: Optional[np.ndarray],
    crop_offset: Optional[Tuple[int, int]],
    obstacle_mask: Optional[np.ndarray] = None,
    projector: Optional[RotatedMapProjector] = None,
    current_distance_field: Optional[np.ndarray] = None,
) -> bool:
    if current_pose is None or waypoint_index >= len(waypoint_positions):
        return False

    distance_m: Optional[float] = None
    if current_distance_field is not None and obstacle_mask is not None and projector is not None:
        target_world = (
            int(waypoint_positions[waypoint_index][0]),
            int(waypoint_positions[waypoint_index][1]),
        )
        distance_m = query_world_distance_from_field_m(
            distance_field=current_distance_field,
            obstacle_mask=obstacle_mask,
            projector=projector,
            target_world=target_world,
            resolution_cm=resolution_cm,
            target_radius_m=WAYPOINT_VISIBILITY_RADIUS_M,
            target_samples=WAYPOINT_VISIBILITY_SAMPLES,
        )
    if distance_m is None:
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

    if current_distance_field is not None and obstacle_mask is not None and projector is not None:
        return True

    clear_path = _has_clear_path_to_waypoint(
        waypoint_py=int(waypoint_positions[waypoint_index][0]),
        waypoint_px=int(waypoint_positions[waypoint_index][1]),
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        full_map=full_map,
        crop_offset=crop_offset,
    )
    return clear_path is not False


def _resolve_last_distinct_waypoint_index(
    waypoint_positions: Sequence[Tuple[int, int]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    full_map: Optional[np.ndarray],
    crop_offset: Optional[Tuple[int, int]],
    obstacle_mask: Optional[np.ndarray] = None,
    projector: Optional[RotatedMapProjector] = None,
    current_distance_field: Optional[np.ndarray] = None,
) -> Optional[int]:
    if current_pose is None or not waypoint_positions:
        return None

    for waypoint_index in range(len(waypoint_positions) - 1, -1, -1):
        if not _is_waypoint_overlapping_current_display(
            waypoint_index=waypoint_index,
            waypoint_positions=waypoint_positions,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            full_map=full_map,
            crop_offset=crop_offset,
            obstacle_mask=obstacle_mask,
            projector=projector,
            current_distance_field=current_distance_field,
        ):
            return int(waypoint_index)
    return None


def resolve_last_distinct_waypoint_index(
    waypoint_positions: Sequence[Tuple[int, int]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    full_map: Optional[np.ndarray],
    crop_offset: Optional[Tuple[int, int]],
) -> Optional[int]:
    obstacle_mask = (
        np.asarray(full_map[0] > 0.5, dtype=bool)
        if full_map is not None and current_pose is not None and crop_offset is not None
        else None
    )
    projector = (
        _build_projector(full_map, current_pose, crop_offset)
        if obstacle_mask is not None
        else None
    )
    current_distance_field = None
    if obstacle_mask is not None and projector is not None and current_pose is not None:
        current_distance_field = build_bounded_geodesic_distance_field(
            obstacle_mask=obstacle_mask,
            projector=projector,
            source_world=(
                float(current_pose[1]) * 100.0 / float(resolution_cm),
                float(current_pose[0]) * 100.0 / float(resolution_cm),
            ),
            max_distance_m=max(
                float(SPACE_AREA_CURRENT_WAYPOINT_MAX_DISTANCE_M),
                float(SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M),
            ),
            resolution_cm=resolution_cm,
        )
    return _resolve_last_distinct_waypoint_index(
        waypoint_positions=waypoint_positions,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        full_map=full_map,
        crop_offset=crop_offset,
        obstacle_mask=obstacle_mask,
        projector=projector,
        current_distance_field=current_distance_field,
    )


def _find_current_area_waypoint_anchor_index(
    waypoint_positions: Sequence[Tuple[int, int]],
    waypoint_area_labels: Optional[Sequence[str]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    current_region_label: str = "",
    full_map: Optional[np.ndarray] = None,
    crop_offset: Optional[Tuple[int, int]] = None,
    obstacle_mask: Optional[np.ndarray] = None,
    projector: Optional[RotatedMapProjector] = None,
    current_distance_field: Optional[np.ndarray] = None,
) -> Optional[int]:
    if current_pose is None or not waypoint_positions:
        return None

    target_area_label = _clean_area_label(current_region_label)
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

        distance_m: Optional[float] = None
        if current_distance_field is not None and obstacle_mask is not None and projector is not None:
            distance_m = query_world_distance_from_field_m(
                distance_field=current_distance_field,
                obstacle_mask=obstacle_mask,
                projector=projector,
                target_world=(int(waypoint_pos[0]), int(waypoint_pos[1])),
                resolution_cm=resolution_cm,
                target_radius_m=WAYPOINT_VISIBILITY_RADIUS_M,
                target_samples=WAYPOINT_VISIBILITY_SAMPLES,
            )
        if distance_m is None:
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

        if current_distance_field is None or obstacle_mask is None or projector is None:
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
    current_region_label: str = "",
    current_region_type: str = "",
    waypoint_descriptions: Optional[Sequence[str]] = None,
    full_map: Optional[np.ndarray] = None,
    crop_offset: Optional[Tuple[int, int]] = None,
) -> Tuple[str, Optional[int]]:
    current_area_label = str(current_region_label or "").strip()
    obstacle_mask = (
        np.asarray(full_map[0] > 0.5, dtype=bool)
        if full_map is not None and current_pose is not None and crop_offset is not None
        else None
    )
    projector = (
        _build_projector(full_map, current_pose, crop_offset)
        if obstacle_mask is not None
        else None
    )
    current_distance_field = None
    if obstacle_mask is not None and projector is not None and current_pose is not None:
        current_distance_field = build_bounded_geodesic_distance_field(
            obstacle_mask=obstacle_mask,
            projector=projector,
            source_world=(
                float(current_pose[1]) * 100.0 / float(resolution_cm),
                float(current_pose[0]) * 100.0 / float(resolution_cm),
            ),
            max_distance_m=max(
                float(SPACE_AREA_CURRENT_WAYPOINT_MAX_DISTANCE_M),
                float(SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M),
            ),
            resolution_cm=resolution_cm,
        )
    anchor_index = _find_current_area_waypoint_anchor_index(
        waypoint_positions=waypoint_positions,
        waypoint_area_labels=waypoint_area_labels,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        current_region_label=current_region_label,
        full_map=full_map,
        crop_offset=crop_offset,
        obstacle_mask=obstacle_mask,
        projector=projector,
        current_distance_field=current_distance_field,
    )
    if anchor_index is not None and waypoint_area_labels and anchor_index < len(waypoint_area_labels):
        anchor_label = str(waypoint_area_labels[anchor_index] or "").strip()
        anchor_description = (
            str(waypoint_descriptions[anchor_index] or "").strip()
            if waypoint_descriptions and anchor_index < len(waypoint_descriptions)
            else ""
        )
        if anchor_label:
            if _is_unknown_area_label(current_area_label):
                display_label = _resolve_display_area_label(
                    anchor_label,
                    current_region_type,
                    cue_texts=[anchor_description],
                )
                if display_label:
                    return display_label, int(anchor_index)
            if _clean_area_label(anchor_label) == _clean_area_label(current_area_label):
                display_label = _resolve_display_area_label(
                    anchor_label,
                    current_region_type,
                    cue_texts=[anchor_description],
                )
                return display_label or anchor_label, int(anchor_index)

    display_current_label = _resolve_display_area_label(
        current_area_label,
        current_region_type,
        cue_texts=list(waypoint_descriptions or []),
    )
    return display_current_label or current_area_label or "Unknown", anchor_index


def select_display_waypoint_indices(
    waypoint_positions: Sequence[Tuple[int, int]],
    waypoint_ids: Sequence[int],
    waypoint_descriptions: Sequence[str],
    waypoint_area_labels: Optional[Sequence[str]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    full_map: Optional[np.ndarray],
    crop_offset: Optional[Tuple[int, int]],
    initial_waypoint_index: Optional[int] = None,
    skip_current_overlap: bool = True,
) -> List[int]:
    if not waypoint_ids:
        return []

    obstacle_mask = (
        np.asarray(full_map[0] > 0.5, dtype=bool)
        if full_map is not None and current_pose is not None and crop_offset is not None
        else None
    )
    projector = (
        _build_projector(full_map, current_pose, crop_offset)
        if obstacle_mask is not None
        else None
    )
    current_distance_field = None
    if obstacle_mask is not None and projector is not None and current_pose is not None:
        current_distance_field = build_bounded_geodesic_distance_field(
            obstacle_mask=obstacle_mask,
            projector=projector,
            source_world=(
                float(current_pose[1]) * 100.0 / float(resolution_cm),
                float(current_pose[0]) * 100.0 / float(resolution_cm),
            ),
            max_distance_m=max(
                float(SPACE_AREA_CURRENT_WAYPOINT_MAX_DISTANCE_M),
                float(SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M),
            ),
            resolution_cm=resolution_cm,
        )

    protected_anchor_indices = set()
    if initial_waypoint_index is not None:
        try:
            candidate_initial_index = int(initial_waypoint_index)
        except (TypeError, ValueError):
            candidate_initial_index = -1
        if 0 <= candidate_initial_index < len(waypoint_ids):
            protected_anchor_indices.add(candidate_initial_index)

    latest_anchor_index = len(waypoint_ids) - 1
    if latest_anchor_index >= 0:
        protected_anchor_indices.add(latest_anchor_index)

    kept_reversed: List[int] = [latest_anchor_index]
    for candidate_index in range(latest_anchor_index - 1, -1, -1):
        if candidate_index in protected_anchor_indices:
            kept_reversed.append(candidate_index)
            continue
        if skip_current_overlap and _is_waypoint_overlapping_current_display(
            waypoint_index=candidate_index,
            waypoint_positions=waypoint_positions,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            full_map=full_map,
            crop_offset=crop_offset,
            obstacle_mask=obstacle_mask,
            projector=projector,
            current_distance_field=current_distance_field,
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

    for protected_index in protected_anchor_indices:
        if protected_index not in kept_reversed:
            kept_reversed.append(protected_index)

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
    cue_texts: Optional[Sequence[str]] = None,
) -> str:
    clean_area = _format_area_display_label(area_label, cue_texts=cue_texts)
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
    waypoint_initial_neighborhood_flags: Optional[Sequence[bool]],
    waypoint_floor_ids: Optional[Sequence[int]],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    current_region_label: str = "",
    current_region_type: str = "",
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
    display_area_label = current_region_label or "Unknown"
    display_area_type = current_region_type or "Unknown"
    normalized_floor_ids = _normalize_waypoint_floor_ids(
        waypoint_ids=waypoint_ids,
        waypoint_floor_ids=waypoint_floor_ids,
        current_floor_id=current_floor_id,
    )
    rendered_floor_ids = {int(floor_id) for floor_id in normalized_floor_ids}
    render_multi_floor = bool(
        multi_floor_active
        and (len(rendered_floor_ids) > 1 or bool(on_stairs_connector))
    )
    space_type_note = (
        f" ({display_area_type})"
        if (
            not _is_unknown_area_label(display_area_label)
            and display_area_type
            and not _is_unknown_area_label(display_area_type)
            and display_area_type != display_area_label
        )
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
    global_initial_neighborhood_indices = {
        int(index)
        for index, flag in enumerate(list(waypoint_initial_neighborhood_flags or []))
        if bool(flag)
    }
    obstacle_mask = (
        np.asarray(full_map[0] > 0.5, dtype=bool)
        if full_map is not None and current_pose is not None and crop_offset is not None
        else None
    )
    projector = (
        _build_projector(full_map, current_pose, crop_offset)
        if obstacle_mask is not None
        else None
    )
    current_distance_field = None
    if obstacle_mask is not None and projector is not None and current_pose is not None:
        current_distance_field = build_bounded_geodesic_distance_field(
            obstacle_mask=obstacle_mask,
            projector=projector,
            source_world=(
                float(current_pose[1]) * 100.0 / float(resolution_cm),
                float(current_pose[0]) * 100.0 / float(resolution_cm),
            ),
            max_distance_m=max(
                float(SPACE_AREA_CURRENT_WAYPOINT_MAX_DISTANCE_M),
                float(SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M),
            ),
            resolution_cm=resolution_cm,
        )
    current_area_initial_note = _build_current_area_initial_waypoint_note(
        waypoint_positions=current_floor_positions,
        waypoint_ids=current_floor_ids,
        waypoint_area_labels=current_floor_area_labels,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        current_region_label=current_region_label,
        full_map=full_map,
        crop_offset=crop_offset,
        initial_waypoint_index=current_floor_initial_index,
        obstacle_mask=obstacle_mask,
        projector=projector,
        current_distance_field=current_distance_field,
    )
    header_lines.append(
        "Your Current Region: "
        f"{_format_current_area_header_label(display_area_label, display_area_type)}"
        f"{space_type_note}{current_area_initial_note}"
    )
    if render_multi_floor:
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
        current_area_display = _format_current_area_chain_label(current_region_label)
        if current_area_display == "Current Position":
            empty_area_chain_line = "Spatial Waypoint Chain: Current Position"
        else:
            empty_area_chain_line = f"Spatial Waypoint Chain: {current_area_display} (Current)"

    if not waypoint_ids:
        lines = list(header_lines)
        if empty_area_chain_line:
            lines.append(empty_area_chain_line)
        lines.append("No spatial waypoints recorded yet.")
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
        current_region_label=current_region_label,
        current_region_type=current_region_type,
        waypoint_descriptions=current_floor_descriptions,
        full_map=full_map,
        crop_offset=crop_offset,
    )
    display_area_label = resolved_current_area_label or display_area_label
    space_type_note = (
        f" ({display_area_type})"
        if (
            not _is_unknown_area_label(display_area_label)
            and display_area_type
            and not _is_unknown_area_label(display_area_type)
            and display_area_type != display_area_label
        )
        else ""
    )
    current_area_initial_note = _build_current_area_initial_waypoint_note(
        waypoint_positions=current_floor_positions,
        waypoint_ids=current_floor_ids,
        waypoint_area_labels=current_floor_area_labels,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        current_region_label=display_area_label,
        full_map=full_map,
        crop_offset=crop_offset,
        initial_waypoint_index=current_floor_initial_index,
        obstacle_mask=obstacle_mask,
        projector=projector,
        current_distance_field=current_distance_field,
    )
    header_lines[0] = (
        "Your Current Region: "
        f"{_format_current_area_header_label(display_area_label, display_area_type)}"
        f"{space_type_note}{current_area_initial_note}"
    )

    current_floor_visible_local_indices = select_display_waypoint_indices(
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
    global_last_distinct_index = _resolve_last_distinct_waypoint_index(
        waypoint_positions=waypoint_positions,
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        full_map=full_map,
        crop_offset=crop_offset,
        obstacle_mask=obstacle_mask,
        projector=projector,
        current_distance_field=current_distance_field,
    )
    current_floor_display_global_indices = [
        current_floor_global_indices[index]
        for index in current_floor_visible_local_indices
    ]
    if (
        global_last_distinct_index is not None
        and global_last_distinct_index in current_floor_global_indices
        and global_last_distinct_index not in current_floor_display_global_indices
    ):
        current_floor_display_global_indices.append(global_last_distinct_index)
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
        distance_m = _waypoint_distance_to_current_m(
            waypoint_index=global_index,
            waypoint_positions=waypoint_positions,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
        )
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
            float(distance_m) if distance_m is not None else float("inf"),
            int(waypoint_ids[global_index]) if global_index < len(waypoint_ids) else int(global_index),
        )

    other_floor_groups: Dict[int, List[int]] = {}
    for index, floor_id in enumerate(normalized_floor_ids):
        if int(floor_id) == int(current_floor_id):
            continue
        other_floor_groups.setdefault(int(floor_id), []).append(int(index))

    grouped_display_indices: List[List[int]] = []
    grouped_floor_ids: List[int] = []
    if current_floor_display_global_indices:
        grouped_display_indices.append(current_floor_display_global_indices)
        grouped_floor_ids.append(int(current_floor_id))
    for floor_id in sorted(other_floor_groups.keys()):
        floor_global_indices = list(other_floor_groups[floor_id])
        floor_positions = [waypoint_positions[index] for index in floor_global_indices]
        floor_ids = [waypoint_ids[index] for index in floor_global_indices]
        floor_descriptions = [
            waypoint_descriptions[index] if index < len(waypoint_descriptions) else ""
            for index in floor_global_indices
        ]
        floor_area_labels = [
            _clean_area_label(
                waypoint_area_labels[index]
                if waypoint_area_labels and index < len(waypoint_area_labels) else ""
            )
            for index in floor_global_indices
        ]
        floor_initial_local_index = None
        if initial_waypoint_index is not None:
            try:
                initial_global_index = int(initial_waypoint_index)
            except (TypeError, ValueError):
                initial_global_index = -1
            if initial_global_index in floor_global_indices:
                floor_initial_local_index = floor_global_indices.index(initial_global_index)
        floor_visible_local_indices = select_display_waypoint_indices(
            waypoint_positions=floor_positions,
            waypoint_ids=floor_ids,
            waypoint_descriptions=floor_descriptions,
            waypoint_area_labels=floor_area_labels,
            current_pose=current_pose,
            resolution_cm=resolution_cm,
            full_map=full_map,
            crop_offset=crop_offset,
            initial_waypoint_index=floor_initial_local_index,
            skip_current_overlap=False,
        )
        floor_display_global_indices = [
            floor_global_indices[index]
            for index in floor_visible_local_indices
        ]
        if (
            global_last_distinct_index is not None
            and global_last_distinct_index in floor_global_indices
            and global_last_distinct_index not in floor_display_global_indices
        ):
            floor_display_global_indices.append(global_last_distinct_index)
        floor_display_global_indices.sort(key=_waypoint_ccw_sort_key)
        grouped_display_indices.append(floor_display_global_indices)
        grouped_floor_ids.append(int(floor_id))

    displayed_global_index_set: set[int] = {
        int(index)
        for group in grouped_display_indices
        for index in list(group or [])
    }
    if initial_waypoint_index is not None:
        try:
            displayed_global_index_set.add(int(initial_waypoint_index))
        except (TypeError, ValueError):
            pass
    if global_last_distinct_index is not None:
        displayed_global_index_set.add(int(global_last_distinct_index))
    displayed_current_floor_visible_local_indices = [
        current_floor_index_map[index]
        for index in sorted(displayed_global_index_set)
        if index in current_floor_index_map
    ]

    grouped_indices_by_floor: Dict[int, List[int]] = {}
    for index in sorted(displayed_global_index_set):
        floor_id = (
            int(normalized_floor_ids[index])
            if index < len(normalized_floor_ids)
            else int(current_floor_id)
        )
        grouped_indices_by_floor.setdefault(floor_id, []).append(int(index))

    area_visit_counts: Dict[Tuple[int, str], int] = {}
    for index in range(len(waypoint_ids)):
        floor_id = (
            int(normalized_floor_ids[index])
            if index < len(normalized_floor_ids)
            else int(current_floor_id)
        )
        area_label = _clean_area_label(
            waypoint_area_labels[index]
            if waypoint_area_labels and index < len(waypoint_area_labels) else ""
        )
        area_visit_counts[(floor_id, area_label)] = (
            int(area_visit_counts.get((floor_id, area_label), 0)) + 1
        )

    def _floor_recency_sort_key(floor_id: int) -> Tuple[int, int]:
        floor_indices = grouped_indices_by_floor.get(int(floor_id), [])
        newest_index = max(floor_indices) if floor_indices else -1
        is_current_floor = int(floor_id) == int(current_floor_id)
        return (0 if is_current_floor else 1, -int(newest_index), int(floor_id))

    def _area_recency_sort_key(area_label: str, area_indices: Sequence[int], floor_id: int) -> Tuple[int, int, str]:
        clean_area = _clean_area_label(area_label)
        newest_index = max(int(index) for index in area_indices) if area_indices else -1
        is_current_area = (
            int(floor_id) == int(current_floor_id)
            and clean_area == _clean_area_label(display_area_label)
        )
        return (0 if is_current_area else 1, -int(newest_index), clean_area)

    def _ordered_area_waypoints(area_indices: Sequence[int]) -> List[int]:
        unique_indices = [int(index) for index in area_indices]
        seen: set[int] = set()
        ordered_unique: List[int] = []
        for index in unique_indices:
            if index in seen:
                continue
            seen.add(index)
            ordered_unique.append(index)

        fixed_indices: List[int] = []
        if initial_waypoint_index is not None:
            try:
                initial_global_index = int(initial_waypoint_index)
            except (TypeError, ValueError):
                initial_global_index = -1
            if initial_global_index in ordered_unique:
                fixed_indices.append(initial_global_index)
        if (
            global_last_distinct_index is not None
            and global_last_distinct_index in ordered_unique
            and global_last_distinct_index not in fixed_indices
        ):
            fixed_indices.append(int(global_last_distinct_index))

        remaining_indices = [
            index for index in ordered_unique
            if index not in set(fixed_indices)
        ]
        remaining_indices.sort(key=_waypoint_ccw_sort_key)
        return fixed_indices + remaining_indices

    node_lines: List[str] = []
    ordered_floor_ids = sorted(grouped_indices_by_floor.keys(), key=_floor_recency_sort_key)
    for floor_id in ordered_floor_ids:
        floor_indices = grouped_indices_by_floor.get(int(floor_id), [])
        if not floor_indices:
            continue
        if render_multi_floor:
            floor_label = _format_floor_label(floor_id)
            floor_notes: List[str] = []
            if int(floor_id) == int(current_floor_id):
                floor_notes.append("Current Floor")
            if on_stairs_connector and int(floor_id) == int(current_floor_id):
                floor_notes.append("stair connector active")
            floor_suffix = f" ({' | '.join(floor_notes)})" if floor_notes else ""
            node_lines.append(f"--- {floor_label}{floor_suffix} ---")

        indices_by_area: Dict[str, List[int]] = {}
        for index in floor_indices:
            area_label = _clean_area_label(
                waypoint_area_labels[index]
                if waypoint_area_labels and index < len(waypoint_area_labels) else ""
            )
            indices_by_area.setdefault(area_label, []).append(int(index))

        ordered_area_labels = sorted(
            indices_by_area.keys(),
            key=lambda area: _area_recency_sort_key(area, indices_by_area.get(area, []), floor_id),
        )
        for area_label in ordered_area_labels:
            area_indices = indices_by_area.get(area_label, [])
            if not area_indices:
                continue
            visit_count = int(area_visit_counts.get((int(floor_id), _clean_area_label(area_label)), 0))
            visit_label = "visit" if visit_count == 1 else "visits"
            area_cue_texts = [
                str(waypoint_descriptions[index] if index < len(waypoint_descriptions) else "")
                for index in area_indices
            ]
            display_node_area = _format_area_display_label(
                area_label,
                cue_texts=area_cue_texts,
            )
            area_prefix = "  Area:" if render_multi_floor else "Area:"
            node_lines.append(f"{area_prefix} {display_node_area} ({visit_count} {visit_label})")

            for index in _ordered_area_waypoints(area_indices):
                wp_id = waypoint_ids[index]
                raw_wp_desc = waypoint_descriptions[index] if index < len(waypoint_descriptions) else ""
                wp_desc = _format_waypoint_description_for_display(
                    raw_wp_desc,
                    area_label=display_node_area,
                    area_type=display_area_type,
                )
                local_index = current_floor_index_map.get(index)
                is_last = global_last_distinct_index is not None and index == int(global_last_distinct_index)
                suffix_notes: List[str] = []
                is_initial = initial_waypoint_index is not None and index == int(initial_waypoint_index)
                if is_initial:
                    suffix_notes.append("INITIAL POSITION")
                if is_last and not is_initial:
                    suffix_notes.append("LAST POSITION")
                if (not is_initial) and index in global_initial_neighborhood_indices:
                    suffix_notes.append("near INITIAL POSITION; not task goal")
                suffix = f"  <- {' | '.join(suffix_notes)}" if suffix_notes else ""

                if int(floor_id) != int(current_floor_id):
                    transition_label = _find_stair_connector_transition_label(
                        from_floor_id=current_floor_id,
                        to_floor_id=floor_id,
                        stair_connectors=stair_connectors,
                    )
                    spatial_info = f"reach via {transition_label}"
                elif current_pose is None:
                    spatial_info = "distance unknown"
                else:
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
                            waypoint_descriptions=current_floor_descriptions,
                            waypoint_area_labels=current_floor_area_labels,
                            current_pose=current_pose,
                            resolution_cm=resolution_cm,
                            full_map=full_map,
                            crop_offset=crop_offset,
                            visible_indices=displayed_current_floor_visible_local_indices,
                            current_region_label=display_area_label,
                        )
                        if reachability_note:
                            spatial_info = f"{spatial_info} | {reachability_note}"
                waypoint_prefix = "    -" if render_multi_floor else "  -"
                node_lines.append(f"{waypoint_prefix} Spatial WP#{wp_id} [{wp_desc}] -- {spatial_info}{suffix}")

    current_area_display = _format_current_area_chain_label(display_area_label)
    displayed_chain_global_indices = [
        index
        for index in range(len(waypoint_ids))
        if index in displayed_global_index_set
    ]
    waypoint_area_path_line = _build_waypoint_area_path_line(
        visible_waypoint_ids=[waypoint_ids[index] for index in displayed_chain_global_indices],
        visible_waypoint_positions=[waypoint_positions[index] for index in displayed_chain_global_indices],
        visible_waypoint_descriptions=[
            waypoint_descriptions[index] if index < len(waypoint_descriptions) else ""
            for index in displayed_chain_global_indices
        ],
        visible_waypoint_area_labels=[
            _clean_area_label(waypoint_area_labels[index])
            if waypoint_area_labels and index < len(waypoint_area_labels)
            else "Unknown"
            for index in displayed_chain_global_indices
        ],
        visible_waypoint_floor_ids=[
            normalized_floor_ids[index]
            for index in displayed_chain_global_indices
        ],
        current_pose=current_pose,
        resolution_cm=resolution_cm,
        current_area_display=current_area_display,
        include_area_chain=include_area_chain,
        include_path=include_path,
        current_floor_id=current_floor_id,
        multi_floor_active=render_multi_floor,
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
    clean_area = _format_area_display_label(area_label)
    return f"Spatial WP#{int(waypoint_id)}({clean_area})"


def _build_waypoint_reachability_note(
    waypoint_index: int,
    waypoint_id: int,
    waypoint_ids: Sequence[int],
    waypoint_positions: Sequence[Tuple[int, int]],
    waypoint_descriptions: Sequence[str],
    waypoint_area_labels: Sequence[str],
    current_pose: Optional[Sequence[float]],
    resolution_cm: float,
    full_map: Optional[np.ndarray],
    crop_offset: Optional[Tuple[int, int]],
    visible_indices: Sequence[int],
    current_region_label: str,
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
        next_description = (
            str(waypoint_descriptions[next_index] or "").strip()
            if next_index < len(waypoint_descriptions)
            else ""
        )
        next_area_ref = f"Spatial WP#{int(next_waypoint_id)}(" + _format_area_display_label(
            next_area,
            cue_texts=[next_description],
        ) + ")"
        return f"blocked to current position; reach via {next_area_ref}"

    if current_pose is not None:
        current_area_display = _format_area_display_label(current_region_label)
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
        return "Spatial Waypoint Chain: Current Position" if include_area_chain else None

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
        entry_members = list(entry.get("members", []) or [])
        cue_texts = [
            str(member.get("description", "") or "")
            for member in entry_members
            if str(member.get("description", "") or "").strip()
        ]
        formatted_area_label = _format_area_with_floor(
            area_label=str(entry.get("area_label", "Unknown") or "Unknown"),
            floor_id=_coerce_floor_id(entry.get("floor_id"), current_floor_id),
            multi_floor_active=multi_floor_active,
            cue_texts=cue_texts,
        )
        if (
            _clean_area_label(str(entry.get("area_label", "") or "")) == "Current Position"
            and entry_members
            and all(bool(member.get("is_current", False)) for member in entry_members)
        ):
            return formatted_area_label
        member_labels = [
            _format_spatial_waypoint_chain_member(
                waypoint_token=str(member.get("token", "")),
                waypoint_description=str(member.get("description", "") or ""),
                is_current=bool(member.get("is_current", False)),
            )
            for member in entry_members
        ]
        return _format_spatial_waypoint_chain_group(
            area_label=formatted_area_label,
            member_labels=member_labels,
        )

    if len(grouped_entries) == 1:
        return "Spatial Waypoint Chain: " + _format_group(grouped_entries[0])

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

    return "Spatial Waypoint Chain: " + "".join(parts)


def build_action_landmark_map_info(
    step_landmark_entries: Sequence[Dict[str, Any]],
    landmark_dist_map: Optional[Dict[str, Tuple[float, float]]] = None,
    landmark_dist_map_multi: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    landmark_instances_world: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Build the executor prompt's landmark summary from the selected action top-k list."""
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
