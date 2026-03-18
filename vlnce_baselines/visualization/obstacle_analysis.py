from typing import Dict, Optional

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


def build_rotated_obstacle_mask(
    full_map: np.ndarray,
    threshold: float = 0.5,
    display_size: int = 480,
    open_kernel_size: int = 0,
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
    return obstacle_mask


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
) -> Dict[str, str]:
    if obstacle_mask_rotated.dtype != bool:
        obstacle_mask_rotated = obstacle_mask_rotated > 127

    distances: Dict[str, str] = {}
    for key, angle in directions.items():
        ray_distances = []
        for offset in (-5, -2.5, 0, 2.5, 5):
            dist_m = raycast_on_rotated_map(
                obstacle_mask_rotated,
                center_x,
                center_y,
                angle + offset,
            )
            if dist_m is not None:
                ray_distances.append(dist_m)

        distances[key] = format_distance(float(np.median(ray_distances))) if ray_distances else "Unknown"
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
