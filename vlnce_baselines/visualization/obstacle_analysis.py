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
) -> np.ndarray:
    obstacle_mask = full_map[0, ...] > threshold
    obstacle_mask = np.flipud(obstacle_mask)
    obstacle_mask = cv2.resize(
        obstacle_mask.astype(np.uint8) * 255,
        (display_size, display_size),
        interpolation=cv2.INTER_NEAREST,
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


def sample_depth_distance_for_angle(
    depth_meters: np.ndarray,
    angle_deg: float,
    hfov_deg: float = 79.0,
    max_distance_m: float = 5.0,
    row_start_ratio: float = 0.30,
    row_end_ratio: float = 0.90,
    column_band_ratio: float = 0.04,
    sample_percentile: float = 20.0,
) -> Optional[float]:
    """Estimate obstacle distance directly from the current depth frame."""
    depth = _prepare_depth_array(depth_meters)
    if depth is None or hfov_deg <= 1e-6:
        return None

    half_fov = hfov_deg / 2.0
    if abs(angle_deg) > half_fov + 1e-6:
        return None

    height, width = depth.shape
    row_start = max(0, min(height - 1, int(height * row_start_ratio)))
    row_end = max(row_start + 1, min(height, int(height * row_end_ratio)))
    center_x = ((angle_deg + half_fov) / hfov_deg) * (width - 1)
    half_band = max(2, int(width * column_band_ratio))
    col_start = max(0, int(round(center_x)) - half_band)
    col_end = min(width, int(round(center_x)) + half_band + 1)

    window = depth[row_start:row_end, col_start:col_end]
    valid = window[np.isfinite(window) & (window > 0.02)]
    if valid.size == 0:
        return None

    clipped = valid[valid <= max_distance_m]
    if clipped.size == 0:
        return max_distance_m + 0.1
    return float(np.percentile(clipped, sample_percentile))


def calculate_obstacle_distances_from_depth(
    depth_meters: np.ndarray,
    hfov_deg: float = 79.0,
    directions: Optional[Dict[str, float]] = None,
    max_distance_m: float = 5.0,
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
        )
        distances[key] = format_distance(distance_m)
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
