from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


ACTION_VIEW_DIRECTIONS = {
    "left_30": -30.0,
    "front": 0.0,
    "right_30": 30.0,
}

ACTION_DIRECTIONS = {
    "front": -90,
    "left_30": -120,
    "left_60": -150,
    "left_90": -180,
    "right_30": -60,
    "right_60": -30,
    "right_90": 0,
}

PANORAMA_DIRECTIONS = {
    "angle_0": -90,
    "angle_30": -120,
    "angle_60": -150,
    "angle_90": -180,
    "angle_120": 150,
    "angle_150": 120,
    "angle_180": 90,
    "angle_210": 60,
    "angle_240": 30,
    "angle_270": 0,
    "angle_300": -30,
    "angle_330": -60,
}

DEFAULT_RAYCAST_FOOTPRINT_RADIUS_PX = 3
DEFAULT_RAYCAST_FOOTPRINT_SAMPLE_COUNT = 9
DEFAULT_RAYCAST_FOOTPRINT_PERCENTILE = 60.0
DEFAULT_DEPTH_REGION_SAMPLE_COUNT = 96


def build_rotated_obstacle_mask(
    full_map: np.ndarray,
    threshold: float = 0.5,
    display_size: int = 480,
    open_kernel_size: int = 0,
    close_kernel_size: int = 0,
    axis_close_kernel_size: int = 0,
    min_component_area: int = 0,
    hole_fill_area: int = 0,
    orthogonal_kernel_size: int = 0,
    orthogonal_min_component_area: int = 0,
    dilate_kernel_size: int = 0,
) -> np.ndarray:
    obstacle_mask = full_map[0, ...] > threshold
    obstacle_mask = np.flipud(obstacle_mask)
    obstacle_mask = cv2.resize(
        obstacle_mask.astype(np.uint8) * 255,
        (display_size, display_size),
        interpolation=cv2.INTER_NEAREST,
    ) > 127
    if open_kernel_size and open_kernel_size > 1:
        kernel = np.ones((int(open_kernel_size), int(open_kernel_size)), dtype=np.uint8)
        obstacle_mask = cv2.morphologyEx(
            obstacle_mask.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            kernel,
        ) > 127
    if close_kernel_size and close_kernel_size > 1:
        kernel = np.ones((int(close_kernel_size), int(close_kernel_size)), dtype=np.uint8)
        obstacle_mask = cv2.morphologyEx(
            obstacle_mask.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            kernel,
        ) > 127
    if axis_close_kernel_size and axis_close_kernel_size > 1:
        obstacle_mask_u8 = obstacle_mask.astype(np.uint8) * 255
        horizontal_kernel = np.ones((3, int(axis_close_kernel_size)), dtype=np.uint8)
        vertical_kernel = np.ones((int(axis_close_kernel_size), 3), dtype=np.uint8)
        horizontal_closed = cv2.morphologyEx(
            obstacle_mask_u8,
            cv2.MORPH_CLOSE,
            horizontal_kernel,
        ) > 127
        vertical_closed = cv2.morphologyEx(
            obstacle_mask_u8,
            cv2.MORPH_CLOSE,
            vertical_kernel,
        ) > 127
        obstacle_mask = obstacle_mask | horizontal_closed | vertical_closed
    if min_component_area and min_component_area > 1:
        mask_u8 = obstacle_mask.astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        filtered = np.zeros_like(mask_u8)
        for component_id in range(1, num_components):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area >= int(min_component_area):
                filtered[labels == component_id] = 1
        obstacle_mask = filtered > 0
    if hole_fill_area and hole_fill_area > 1:
        obstacle_mask = _fill_small_holes(obstacle_mask, max_hole_area=int(hole_fill_area))
    if orthogonal_kernel_size and orthogonal_kernel_size > 1:
        obstacle_mask = _orthogonalize_obstacle_mask(
            obstacle_mask,
            line_kernel_size=int(orthogonal_kernel_size),
            min_component_area=max(orthogonal_min_component_area, min_component_area, 1),
        )
    if dilate_kernel_size and dilate_kernel_size > 1:
        kernel = np.ones((int(dilate_kernel_size), int(dilate_kernel_size)), dtype=np.uint8)
        obstacle_mask = cv2.dilate(
            obstacle_mask.astype(np.uint8) * 255,
            kernel,
            iterations=1,
        ) > 127
    return obstacle_mask


def _fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if max_hole_area <= 0 or not np.any(mask_bool):
        return mask_bool

    background = (~mask_bool).astype(np.uint8)
    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    border_labels = set(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())

    filled = mask_bool.copy()
    for component_id in range(1, num_components):
        if component_id in border_labels:
            continue
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area <= int(max_hole_area):
            filled[labels == component_id] = True
    return filled


def _orthogonalize_obstacle_mask(
    mask: np.ndarray,
    line_kernel_size: int,
    min_component_area: int,
) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
    if mask_u8.size == 0 or np.count_nonzero(mask_u8) == 0:
        return mask_u8 > 127

    horizontal_kernel = np.ones((1, int(line_kernel_size)), dtype=np.uint8)
    vertical_kernel = np.ones((int(line_kernel_size), 1), dtype=np.uint8)
    horizontal = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, horizontal_kernel)
    vertical = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, vertical_kernel)
    regularized = ((mask_u8 > 127) | (horizontal > 127) | (vertical > 127)).astype(np.uint8) * 255

    contours, _ = cv2.findContours(regularized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    snapped_fill = np.zeros_like(mask_u8)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(max(min_component_area, 1)):
            cv2.drawContours(snapped_fill, [contour], -1, 255, thickness=-1)
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = max(1.0, perimeter * 0.015)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = approx[:, 0, :]
        if len(pts) < 3:
            cv2.drawContours(snapped_fill, [contour], -1, 255, thickness=-1)
            continue

        snapped_pts = []
        for idx, point in enumerate(pts):
            prev_point = pts[idx - 1]
            curr_point = point.copy()
            dx = int(curr_point[0]) - int(prev_point[0])
            dy = int(curr_point[1]) - int(prev_point[1])
            if max(abs(dx), abs(dy)) >= int(max(4, line_kernel_size // 2)):
                if abs(dx) >= abs(dy):
                    curr_point[1] = prev_point[1]
                else:
                    curr_point[0] = prev_point[0]
            snapped_pts.append(curr_point)

        snapped_polygon = np.asarray(snapped_pts, dtype=np.int32)
        if snapped_polygon.shape[0] >= 3:
            cv2.fillPoly(snapped_fill, [snapped_polygon], 255)
        else:
            cv2.drawContours(snapped_fill, [contour], -1, 255, thickness=-1)

    regularized = cv2.morphologyEx(
        snapped_fill,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    return regularized > 127


def raycast_on_rotated_map(
    obstacle_mask: np.ndarray,
    start_x: int,
    start_y: int,
    angle_deg: float,
    max_distance_m: float = 2.0,
    step_size_px: float = 0.5,
    resolution_cm: int = 5,
) -> Optional[float]:
    h, w = obstacle_mask.shape
    angle_rad = np.deg2rad(angle_deg)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    distance_px = 0.0
    max_steps = int(max_distance_m * 100 / resolution_cm / step_size_px)
    for _ in range(max_steps):
        distance_px += step_size_px
        current_x = start_x + dx * distance_px
        current_y = start_y + dy * distance_px
        ix, iy = int(round(current_x)), int(round(current_y))
        if not (0 <= ix < w and 0 <= iy < h):
            return max_distance_m + 0.1
        if obstacle_mask[iy, ix]:
            return distance_px * resolution_cm / 100.0
    return max_distance_m + 0.1


def _build_footprint_start_offsets(
    radius_px: int,
    sample_count: Optional[int] = None,
) -> List[Tuple[int, int]]:
    radius_px = max(0, int(radius_px))
    if radius_px <= 0:
        return [(0, 0)]

    candidate_offsets: List[Tuple[int, int]] = []
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            if dx * dx + dy * dy > radius_px * radius_px:
                continue
            candidate_offsets.append((dx, dy))

    candidate_offsets.sort(key=lambda item: item[0] * item[0] + item[1] * item[1])
    if len(candidate_offsets) <= 1:
        return candidate_offsets

    center = (0, 0)
    other_offsets = [offset for offset in candidate_offsets if offset != center]
    target_sample_count = max(
        1,
        int(sample_count or DEFAULT_RAYCAST_FOOTPRINT_SAMPLE_COUNT),
    )
    extra_needed = max(0, min(len(other_offsets), target_sample_count - 1))
    if extra_needed <= 0:
        return [center]

    rng = np.random.default_rng()
    selected_indices = rng.choice(len(other_offsets), size=extra_needed, replace=False)
    sampled_offsets = [other_offsets[int(idx)] for idx in np.atleast_1d(selected_indices).tolist()]
    sampled_offsets.sort(key=lambda item: item[0] * item[0] + item[1] * item[1])
    return [center] + sampled_offsets


def format_distance(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return "Unknown"
    if distance_m > 2.0:
        return ">2.0m open"
    if distance_m < 0.5:
        return f"{distance_m:.2f}m WARNING"
    return f"{distance_m:.2f}m"


def calculate_distances_for_directions(
    obstacle_mask_rotated: np.ndarray,
    directions: Dict[str, float],
    center_x: int = 240,
    center_y: int = 240,
    footprint_radius_px: int = DEFAULT_RAYCAST_FOOTPRINT_RADIUS_PX,
    footprint_sample_count: int = DEFAULT_RAYCAST_FOOTPRINT_SAMPLE_COUNT,
    footprint_percentile: float = DEFAULT_RAYCAST_FOOTPRINT_PERCENTILE,
) -> Dict[str, str]:
    if obstacle_mask_rotated.dtype != bool:
        obstacle_mask_rotated = obstacle_mask_rotated > 127

    start_offsets = _build_footprint_start_offsets(
        radius_px=footprint_radius_px,
        sample_count=footprint_sample_count,
    )
    distances: Dict[str, str] = {}
    for key, angle in directions.items():
        footprint_distances = []
        for start_dx, start_dy in start_offsets:
            start_x = int(center_x + start_dx)
            start_y = int(center_y + start_dy)
            if not (
                0 <= start_x < obstacle_mask_rotated.shape[1] and
                0 <= start_y < obstacle_mask_rotated.shape[0]
            ):
                continue

            ray_distances = []
            for offset in (-5, -2.5, 0, 2.5, 5):
                dist_m = raycast_on_rotated_map(
                    obstacle_mask_rotated,
                    start_x,
                    start_y,
                    angle + offset,
                )
                if dist_m is not None:
                    ray_distances.append(dist_m)

            if ray_distances:
                footprint_distances.append(float(np.median(ray_distances)))

        if footprint_distances:
            percentile = float(np.clip(footprint_percentile, 0.0, 100.0))
            distances[key] = format_distance(float(np.percentile(footprint_distances, percentile)))
        else:
            distances[key] = "Unknown"
    return distances


def _prepare_depth_array(depth_meters: np.ndarray) -> Optional[np.ndarray]:
    if depth_meters is None:
        return None
    depth = np.asarray(depth_meters, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        return None
    return depth


def _parse_distance_text_m(distance_text: Optional[str]) -> Optional[float]:
    if not distance_text:
        return None
    text = str(distance_text).strip().lower()
    if not text or text == "unknown":
        return None
    try:
        number = float(text.replace("warning", "").replace("open", "").replace("m", "").replace(">", "").strip())
    except ValueError:
        return None
    if ">" in text:
        return number + 0.1
    return number


def sample_depth_distance_from_region(
    depth_meters: np.ndarray,
    center_x_ratio: float = 0.5,
    width_ratio: float = 0.24,
    row_start_ratio: float = 0.35,
    row_end_ratio: float = 0.90,
    max_distance_m: float = 5.0,
    sample_count: int = DEFAULT_DEPTH_REGION_SAMPLE_COUNT,
    sample_percentile: float = 20.0,
    min_depth_m: float = 0.02,
) -> Optional[float]:
    """Estimate a depth distance by randomly sampling a rectangular image region."""
    depth = _prepare_depth_array(depth_meters)
    if depth is None:
        return None

    height, width = depth.shape
    if height <= 0 or width <= 0:
        return None

    center_x_ratio = float(np.clip(center_x_ratio, 0.0, 1.0))
    width_ratio = float(np.clip(width_ratio, 1e-3, 1.0))
    half_width_ratio = 0.5 * width_ratio

    row_start = max(0, min(height - 1, int(round(height * float(row_start_ratio)))))
    row_end = max(row_start + 1, min(height, int(round(height * float(row_end_ratio)))))
    col_start = max(0, min(width - 1, int(round(width * (center_x_ratio - half_width_ratio)))))
    col_end = max(col_start + 1, min(width, int(round(width * (center_x_ratio + half_width_ratio)))))

    region = depth[row_start:row_end, col_start:col_end]
    valid_mask = np.isfinite(region) & (region > float(min_depth_m))
    if not np.any(valid_mask):
        return None

    ys, xs = np.where(valid_mask)
    sample_total = int(max(1, sample_count))
    if ys.size > sample_total:
        rng = np.random.default_rng()
        chosen = rng.choice(ys.size, size=sample_total, replace=False)
        ys = ys[chosen]
        xs = xs[chosen]

    sampled_depths = region[ys, xs].astype(np.float32)
    sampled_depths = sampled_depths[np.isfinite(sampled_depths) & (sampled_depths > float(min_depth_m))]
    if sampled_depths.size == 0:
        return None

    clipped = sampled_depths[sampled_depths <= float(max_distance_m)]
    if clipped.size == 0:
        return float(max_distance_m) + 0.1

    percentile = float(np.clip(sample_percentile, 0.0, 100.0))
    return float(np.percentile(clipped, percentile))


def sample_depth_distance_for_angle(
    depth_meters: np.ndarray,
    angle_deg: float,
    hfov_deg: float = 79.0,
    max_distance_m: float = 5.0,
    row_start_ratio: float = 0.30,
    row_end_ratio: float = 0.90,
    angle_band_deg: float = 5.0,
    sample_percentile: float = 20.0,
) -> Optional[float]:
    """Estimate obstacle distance from an angular depth band in the current view."""
    depth = _prepare_depth_array(depth_meters)
    if depth is None or hfov_deg <= 1e-6:
        return None

    half_fov = hfov_deg / 2.0
    if abs(angle_deg) > half_fov + 1e-6:
        return None

    height, width = depth.shape
    band_candidates = []
    for band_deg in (float(angle_band_deg), max(float(angle_band_deg) * 2.0, 10.0)):
        if band_deg > 0:
            band_candidates.append(band_deg)

    row_bands = [
        (float(row_start_ratio), float(row_end_ratio)),
        (max(float(row_start_ratio), 0.55), 0.98),
    ]
    best_distance = None
    for row_start_ratio_i, row_end_ratio_i in row_bands:
        row_start = max(0, min(height - 1, int(height * row_start_ratio_i)))
        row_end = max(row_start + 1, min(height, int(height * row_end_ratio_i)))
        for band_deg in band_candidates:
            left_angle = max(-half_fov, float(angle_deg) - band_deg / 2.0)
            right_angle = min(half_fov, float(angle_deg) + band_deg / 2.0)
            left_ratio = (left_angle + half_fov) / hfov_deg
            right_ratio = (right_angle + half_fov) / hfov_deg
            col_start = max(0, int(np.floor(left_ratio * (width - 1))))
            col_end = min(width, int(np.ceil(right_ratio * (width - 1))) + 1)
            if col_end <= col_start:
                center_x = ((angle_deg + half_fov) / hfov_deg) * (width - 1)
                center_col = int(round(center_x))
                col_start = max(0, center_col - 1)
                col_end = min(width, center_col + 2)

            window = depth[row_start:row_end, col_start:col_end]
            valid = window[np.isfinite(window) & (window > 0.02)]
            if valid.size == 0:
                continue

            clipped = valid[valid <= max_distance_m]
            candidate = max_distance_m + 0.1 if clipped.size == 0 else float(np.percentile(clipped, sample_percentile))
            if best_distance is None or candidate < best_distance:
                best_distance = candidate

    return best_distance


def calculate_obstacle_distances_from_depth(
    depth_meters: np.ndarray,
    hfov_deg: float = 79.0,
    directions: Optional[Dict[str, float]] = None,
    max_distance_m: float = 5.0,
    angle_band_deg: float = 5.0,
    fallback_distances: Optional[Dict[str, str]] = None,
    default_distance: str = ">2.0m open",
    conservative_map_distance_m: float = 1.5,
) -> Dict[str, str]:
    """Calculate lightweight action-side obstacle distances from the current depth frame."""
    direction_map = directions or ACTION_VIEW_DIRECTIONS
    distances: Dict[str, str] = {}
    for key, angle_deg in direction_map.items():
        distance_m = sample_depth_distance_for_angle(
            depth_meters,
            angle_deg=angle_deg,
            hfov_deg=hfov_deg,
            max_distance_m=max_distance_m,
            angle_band_deg=angle_band_deg,
        )
        fallback_text = (fallback_distances or {}).get(key, default_distance)
        fallback_distance_m = _parse_distance_text_m(fallback_text)
        chosen_distance_m = distance_m
        if chosen_distance_m is None:
            distances[key] = fallback_text
            continue
        if (
            fallback_distance_m is not None and
            fallback_distance_m <= float(conservative_map_distance_m)
        ):
            chosen_distance_m = min(float(chosen_distance_m), float(fallback_distance_m))
        distances[key] = format_distance(chosen_distance_m)
    return distances


def calculate_obstacle_distances_from_rotated_map(
    obstacle_mask_rotated: np.ndarray,
    center_x: int = 240,
    center_y: int = 240,
) -> Dict[str, str]:
    return calculate_distances_for_directions(
        obstacle_mask_rotated,
        ACTION_DIRECTIONS,
        center_x=center_x,
        center_y=center_y,
    )


def calculate_obstacle_distances_12_directions(
    obstacle_mask_rotated: np.ndarray,
    center_x: int = 240,
    center_y: int = 240,
) -> Dict[str, str]:
    return calculate_distances_for_directions(
        obstacle_mask_rotated,
        PANORAMA_DIRECTIONS,
        center_x=center_x,
        center_y=center_y,
    )
