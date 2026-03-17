from typing import Dict, Optional

import cv2
import numpy as np


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
