"""Top-down visualization helpers for NavGBench adapters."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from navigation_system.env.common import yaw_from_quaternion_like


GT_PATH_COLOR_RGB = (0, 210, 0)
START_SQUARE_COLOR_RGB = (0, 85, 35)
GOAL_SQUARE_COLOR_RGB = (255, 0, 0)
AGENT_ARROW_COLOR_RGB = (255, 45, 45)
EXECUTED_PATH_COLOR_RGB = (255, 140, 0)


def _as_pixel_tuple(pixel: Any) -> Optional[tuple[int, int]]:
    try:
        if isinstance(pixel, np.ndarray):
            pixel = pixel.reshape(-1).tolist()
        if isinstance(pixel, (list, tuple)) and len(pixel) >= 2:
            return int(pixel[0]), int(pixel[1])
    except Exception:
        return None
    return None


def _draw_square_marker(
    image: np.ndarray,
    center: tuple[int, int],
    color: tuple[int, int, int],
    *,
    half_size: int = 7,
) -> None:
    x, y = int(center[0]), int(center[1])
    outer = int(half_size) + 2
    cv2.rectangle(
        image,
        (x - outer, y - outer),
        (x + outer, y + outer),
        (255, 255, 255),
        -1,
    )
    cv2.rectangle(
        image,
        (x - half_size, y - half_size),
        (x + half_size, y + half_size),
        color,
        -1,
    )
    cv2.rectangle(
        image,
        (x - half_size, y - half_size),
        (x + half_size, y + half_size),
        (30, 30, 30),
        1,
    )


def _sim_agent_state(sim: Any) -> Any:
    getter = getattr(sim, "get_agent_state", None)
    if callable(getter):
        try:
            return getter(0)
        except TypeError:
            return getter()
        except Exception:
            return None
    return getattr(sim, "agent_state", None)


def _yaw_from_rotation(rotation: Any) -> float:
    if hasattr(rotation, "w") and hasattr(rotation, "x"):
        w = float(rotation.w)
        x = float(rotation.x)
        y = float(rotation.y)
        z = float(rotation.z)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))
    return yaw_from_quaternion_like(rotation)


def _heading_vector_from_sim(
    sim: Any,
    current_pixel: tuple[int, int],
) -> Optional[np.ndarray]:
    state = _sim_agent_state(sim)
    if state is None:
        return None
    rotation = getattr(state, "rotation", None)
    position = getattr(state, "position", None)
    transform = getattr(sim, "transform_from_world_to_pixel", None)
    if rotation is None or position is None or not callable(transform):
        return None
    try:
        yaw = _yaw_from_rotation(rotation)
        position_arr = np.asarray(position, dtype=np.float64).reshape(-1)
        if position_arr.size < 2:
            return None
        probe = position_arr.copy()
        probe[0] += math.cos(yaw) * 1.5
        probe[1] += math.sin(yaw) * 1.5
        probe_pixel = _as_pixel_tuple(transform(probe))
        if probe_pixel is None:
            return None
        vector = np.asarray(
            [
                probe_pixel[0] - current_pixel[0],
                probe_pixel[1] - current_pixel[1],
            ],
            dtype=np.float64,
        )
        return vector if float(np.linalg.norm(vector)) > 0.5 else None
    except Exception:
        return None


def _fallback_heading_vector(
    trajectory_pixels: Sequence[tuple[int, int]],
) -> np.ndarray:
    if trajectory_pixels:
        current = np.asarray(trajectory_pixels[-1], dtype=np.float64)
        for previous_pixel in reversed(trajectory_pixels[:-1]):
            previous = np.asarray(previous_pixel, dtype=np.float64)
            vector = current - previous
            if float(np.linalg.norm(vector)) > 0.5:
                return vector
    return np.asarray([0.0, -1.0], dtype=np.float64)


def _draw_current_pose_arrow(
    image: np.ndarray,
    sim: Any,
    trajectory_pixels: Sequence[tuple[int, int]],
) -> None:
    if not trajectory_pixels:
        return
    center = np.asarray(trajectory_pixels[-1], dtype=np.float64)
    vector = _heading_vector_from_sim(sim, trajectory_pixels[-1])
    if vector is None:
        vector = _fallback_heading_vector(trajectory_pixels)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        vector = np.asarray([0.0, -1.0], dtype=np.float64)
        norm = 1.0
    direction = vector / norm
    perp = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    size = max(14, int(round(min(image.shape[:2]) * 0.045)))
    tip = center + direction * size
    left = center - direction * (size * 0.55) + perp * (size * 0.48)
    right = center - direction * (size * 0.55) - perp * (size * 0.48)
    contour = np.asarray([[tip, left, right]], dtype=np.int32)
    cv2.drawContours(image, contour, 0, (255, 255, 255), 3)
    cv2.drawContours(image, contour, 0, AGENT_ARROW_COLOR_RGB, -1)


def build_navgbench_topdown_trajectory(sim: Any) -> Optional[np.ndarray]:
    """Render a NavGBench occupancy trajectory with SpaceVLN's shared style."""
    original = getattr(sim, "occ_map_original", None)
    gt_pixels = [
        pixel_tuple
        for pixel_tuple in (
            _as_pixel_tuple(item)
            for item in (getattr(sim, "gt_trajectory_pixels", []) or [])
        )
        if pixel_tuple is not None
    ]
    if original is None or not gt_pixels:
        return None

    image = np.asarray(original).copy()
    for start, end in zip(gt_pixels, gt_pixels[1:]):
        cv2.line(image, start, end, GT_PATH_COLOR_RGB, 3)

    trajectory_pixels = [
        pixel_tuple
        for pixel_tuple in (
            _as_pixel_tuple(item)
            for item in (getattr(sim, "trajectory_pixels", []) or [])
        )
        if pixel_tuple is not None
    ]
    draw_step = getattr(sim, "draw_step", None)
    for start, end in zip(trajectory_pixels, trajectory_pixels[1:]):
        if callable(draw_step):
            image = draw_step(image, start, end, EXECUTED_PATH_COLOR_RGB)
        else:
            cv2.line(image, start, end, EXECUTED_PATH_COLOR_RGB, 3)
        if image is None:
            return None

    _draw_square_marker(image, gt_pixels[0], START_SQUARE_COLOR_RGB)
    _draw_square_marker(image, gt_pixels[-1], GOAL_SQUARE_COLOR_RGB)
    _draw_current_pose_arrow(image, sim, trajectory_pixels)
    return image


__all__ = ["build_navgbench_topdown_trajectory"]
