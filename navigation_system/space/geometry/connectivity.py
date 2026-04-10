"""Obstacle-aware local connectivity helpers for waypoint/area reasoning."""

import heapq
import math
from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np

from navigation_system.space.geometry.map_projection import RotatedMapProjector


_NEIGHBOR_STEPS: Tuple[Tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def world_to_rotated_cell(
    projector: RotatedMapProjector,
    world_py: float,
    world_px: float,
) -> Optional[Tuple[int, int]]:
    rotated = projector.world_to_rotated_pixel(float(world_py), float(world_px))
    if rotated is None:
        return None
    return int(round(rotated[0])), int(round(rotated[1]))


def find_nearest_free_cell(
    obstacle_mask: np.ndarray,
    row: int,
    col: int,
    max_radius_cells: int = 8,
) -> Optional[Tuple[int, int]]:
    height, width = obstacle_mask.shape
    if not (0 <= row < height and 0 <= col < width):
        return None

    free_mask = ~np.asarray(obstacle_mask, dtype=bool)
    if free_mask[row, col]:
        return int(row), int(col)

    max_radius = max(int(max_radius_cells), 0)
    visited = np.zeros((height, width), dtype=bool)
    queue: deque[Tuple[int, int, int]] = deque([(int(row), int(col), 0)])
    visited[row, col] = True

    while queue:
        cur_row, cur_col, radius = queue.popleft()
        if radius >= max_radius:
            continue
        for d_row, d_col, _cost in _NEIGHBOR_STEPS:
            next_row = cur_row + int(d_row)
            next_col = cur_col + int(d_col)
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if visited[next_row, next_col]:
                continue
            if free_mask[next_row, next_col]:
                return int(next_row), int(next_col)
            visited[next_row, next_col] = True
            queue.append((int(next_row), int(next_col), int(radius + 1)))
    return None


def build_bounded_geodesic_distance_field(
    obstacle_mask: np.ndarray,
    projector: RotatedMapProjector,
    source_world: Tuple[float, float],
    max_distance_m: float,
    resolution_cm: float,
    source_snap_radius_cells: int = 8,
) -> Optional[np.ndarray]:
    if max_distance_m <= 0.0 or resolution_cm <= 0.0:
        return None

    source_cell = world_to_rotated_cell(
        projector=projector,
        world_py=float(source_world[0]),
        world_px=float(source_world[1]),
    )
    if source_cell is None:
        return None

    start_cell = find_nearest_free_cell(
        obstacle_mask=obstacle_mask,
        row=int(source_cell[0]),
        col=int(source_cell[1]),
        max_radius_cells=int(source_snap_radius_cells),
    )
    if start_cell is None:
        return None

    max_distance_px = (float(max_distance_m) * 100.0) / float(resolution_cm)
    free_mask = ~np.asarray(obstacle_mask, dtype=bool)
    distance_field = np.full(obstacle_mask.shape, np.inf, dtype=np.float32)
    start_row, start_col = int(start_cell[0]), int(start_cell[1])
    distance_field[start_row, start_col] = 0.0
    queue: List[Tuple[float, int, int]] = [(0.0, start_row, start_col)]

    while queue:
        dist_px, row, col = heapq.heappop(queue)
        if dist_px > float(distance_field[row, col]) + 1e-6:
            continue
        if dist_px > max_distance_px + 1e-6:
            continue

        for d_row, d_col, step_cost in _NEIGHBOR_STEPS:
            next_row = row + int(d_row)
            next_col = col + int(d_col)
            if not (0 <= next_row < free_mask.shape[0] and 0 <= next_col < free_mask.shape[1]):
                continue
            if not free_mask[next_row, next_col]:
                continue
            next_dist_px = float(dist_px) + float(step_cost)
            if next_dist_px > max_distance_px + 1e-6:
                continue
            if next_dist_px + 1e-6 >= float(distance_field[next_row, next_col]):
                continue
            distance_field[next_row, next_col] = next_dist_px
            heapq.heappush(queue, (next_dist_px, next_row, next_col))

    return distance_field


def query_world_distance_from_field_m(
    distance_field: np.ndarray,
    obstacle_mask: np.ndarray,
    projector: RotatedMapProjector,
    target_world: Tuple[float, float],
    resolution_cm: float,
    target_radius_m: float = 0.0,
    target_samples: int = 0,
) -> Optional[float]:
    if resolution_cm <= 0.0:
        return None

    radius_px = max(0.0, (float(target_radius_m) * 100.0) / float(resolution_cm))
    snap_radius_cells = max(1, int(math.ceil(radius_px))) if radius_px > 0.0 else 1
    candidate_world_points: List[Tuple[float, float]] = [
        (float(target_world[0]), float(target_world[1]))
    ]
    if radius_px > 0.0 and int(target_samples) > 0:
        for sample_idx in range(int(target_samples)):
            theta = (2.0 * math.pi * float(sample_idx)) / float(target_samples)
            candidate_world_points.append((
                float(target_world[0]) + radius_px * math.sin(theta),
                float(target_world[1]) + radius_px * math.cos(theta),
            ))

    candidate_cells = set()
    for world_py, world_px in candidate_world_points:
        rotated = world_to_rotated_cell(
            projector=projector,
            world_py=float(world_py),
            world_px=float(world_px),
        )
        if rotated is None:
            continue
        snapped = find_nearest_free_cell(
            obstacle_mask=obstacle_mask,
            row=int(rotated[0]),
            col=int(rotated[1]),
            max_radius_cells=int(snap_radius_cells),
        )
        if snapped is None:
            continue
        candidate_cells.add((int(snapped[0]), int(snapped[1])))

    if not candidate_cells:
        return None

    best_distance_px: Optional[float] = None
    for row, col in candidate_cells:
        if not (0 <= row < distance_field.shape[0] and 0 <= col < distance_field.shape[1]):
            continue
        dist_px = float(distance_field[row, col])
        if not np.isfinite(dist_px):
            continue
        if best_distance_px is None or dist_px < best_distance_px:
            best_distance_px = dist_px

    if best_distance_px is None:
        return None
    return float(best_distance_px) * float(resolution_cm) / 100.0
