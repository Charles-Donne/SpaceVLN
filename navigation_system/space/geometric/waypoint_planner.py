"""Local geometric waypoint generation and A* planning on SpaceVLN maps."""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from navigation_system.space.description.direction_format import (
    format_relative_direction,
    normalize_relative_bearing,
)
from navigation_system.space.geometry.map_projection import RotatedMapProjector


_NEIGHBORS_8: Tuple[Tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class GeometricPlannerConfig:
    """Tunable parameters for local geometric waypoint execution."""

    enabled: bool = False
    max_candidates: int = 5
    min_candidate_distance_m: float = 0.8
    max_candidate_distance_m: float = 4.0
    candidate_stride_m: float = 0.75
    obstacle_inflation_radius_m: float = 0.30
    unknown_as_obstacle: bool = True
    min_clearance_m: float = 0.20
    path_step_m: float = 0.35
    waypoint_arrival_radius_m: float = 0.35
    max_path_execute_steps: int = 12
    max_turn_per_action_deg: float = 30.0
    stop_on_blocked_front: bool = True


@dataclass(frozen=True)
class GeometricCandidate:
    """One model-selectable local geometric waypoint."""

    candidate_id: int
    row: int
    col: int
    world_row: float
    world_col: float
    world_x_m: float
    world_y_m: float
    distance_m: float
    path_length_m: float
    bearing_deg: float
    direction: str
    clearance_m: float
    is_backtrack: bool

    def to_prompt_line(self) -> str:
        backtrack = " | backtrack" if self.is_backtrack else ""
        return (
            f"{self.candidate_id}. {self.direction}, "
            f"straight {self.distance_m:.2f}m, path {self.path_length_m:.2f}m, "
            f"clearance {self.clearance_m:.2f}m{backtrack}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeometricPlan:
    """A planned path to one selected geometric candidate."""

    candidate: GeometricCandidate
    path_cells: Tuple[Tuple[int, int], ...]
    world_points: Tuple[Tuple[float, float], ...]
    action_points: Tuple[Tuple[float, float], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "path_cells": [[int(r), int(c)] for r, c in self.path_cells],
            "world_points": [[float(r), float(c)] for r, c in self.world_points],
            "action_points": [[float(x), float(y)] for x, y in self.action_points],
        }


class GeometricWaypointPlanner:
    """Generate reachable local waypoint candidates and A* paths."""

    def __init__(
        self,
        config: Optional[GeometricPlannerConfig] = None,
        *,
        resolution_cm: float = 5.0,
    ) -> None:
        self.config = config or GeometricPlannerConfig()
        self.resolution_cm = max(1e-6, float(resolution_cm or 5.0))
        self._last_debug: Dict[str, Any] = {}

    @property
    def last_debug(self) -> Dict[str, Any]:
        return dict(self._last_debug)

    @property
    def resolution_m(self) -> float:
        return self.resolution_cm / 100.0

    @staticmethod
    def _as_full_map_array(full_map: Any) -> Optional[np.ndarray]:
        if full_map is None:
            return None
        array = np.asarray(full_map)
        if array.ndim == 4:
            array = array[0]
        if array.ndim != 3 or array.shape[0] < 2:
            return None
        return array

    @staticmethod
    def _pose_to_world_pixel(
        pose_xytheta: Optional[Sequence[float]],
        resolution_cm: float,
    ) -> Optional[Tuple[float, float]]:
        if pose_xytheta is None or len(pose_xytheta) < 2:
            return None
        try:
            x_m = float(pose_xytheta[0])
            y_m = float(pose_xytheta[1])
        except (TypeError, ValueError):
            return None
        return y_m * 100.0 / float(resolution_cm), x_m * 100.0 / float(resolution_cm)

    def _build_projector(
        self,
        full_map: np.ndarray,
        pose_xytheta: Sequence[float],
        crop_offset: Sequence[int],
    ) -> RotatedMapProjector:
        heading_deg = 0.0
        if pose_xytheta is not None and len(pose_xytheta) >= 3:
            try:
                heading_deg = float(pose_xytheta[2])
            except (TypeError, ValueError):
                heading_deg = 0.0
        return RotatedMapProjector(
            map_h=int(full_map.shape[1]),
            map_w=int(full_map.shape[2]),
            crop_offset=(int(crop_offset[0]), int(crop_offset[1])),
            agent_orientation_deg=heading_deg,
        )

    def _build_masks(
        self,
        full_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obstacle = np.asarray(full_map[0] > 0.5, dtype=np.uint8)
        explored = np.asarray(full_map[1] > 0.5, dtype=np.uint8)
        floor = np.logical_and(explored > 0, obstacle == 0).astype(np.uint8)

        inflation_px = max(
            0,
            int(math.ceil(float(self.config.obstacle_inflation_radius_m) / self.resolution_m)),
        )
        inflated_obstacle = obstacle.astype(bool)
        if inflation_px > 0 and np.any(obstacle):
            kernel_size = inflation_px * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )
            inflated_obstacle = cv2.dilate(obstacle, kernel, iterations=1).astype(bool)

        free = np.logical_and(floor.astype(bool), np.logical_not(inflated_obstacle))
        if bool(self.config.unknown_as_obstacle):
            free = np.logical_and(free, explored.astype(bool))

        if np.any(free):
            free_uint8 = (free.astype(np.uint8) * 255)
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            free = cv2.morphologyEx(free_uint8, cv2.MORPH_OPEN, open_kernel).astype(bool)

        clearance = cv2.distanceTransform(
            free.astype(np.uint8),
            cv2.DIST_L2,
            5,
        ).astype(np.float32) * float(self.resolution_m)
        return free.astype(bool), inflated_obstacle.astype(bool), clearance

    @staticmethod
    def _nearest_free_cell(
        free_mask: np.ndarray,
        row: int,
        col: int,
        max_radius: int,
    ) -> Optional[Tuple[int, int]]:
        height, width = free_mask.shape
        if not (0 <= row < height and 0 <= col < width):
            return None
        if free_mask[row, col]:
            return int(row), int(col)

        visited = np.zeros((height, width), dtype=bool)
        queue: deque[Tuple[int, int, int]] = deque([(int(row), int(col), 0)])
        visited[row, col] = True
        while queue:
            cur_row, cur_col, radius = queue.popleft()
            if radius >= int(max_radius):
                continue
            for d_row, d_col, _cost in _NEIGHBORS_8:
                next_row = cur_row + d_row
                next_col = cur_col + d_col
                if not (0 <= next_row < height and 0 <= next_col < width):
                    continue
                if visited[next_row, next_col]:
                    continue
                if free_mask[next_row, next_col]:
                    return int(next_row), int(next_col)
                visited[next_row, next_col] = True
                queue.append((int(next_row), int(next_col), int(radius + 1)))
        return None

    def _a_star(
        self,
        free_mask: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        height, width = free_mask.shape
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        if not (
            0 <= start[0] < height
            and 0 <= start[1] < width
            and 0 <= goal[0] < height
            and 0 <= goal[1] < width
        ):
            return None
        if not free_mask[start] or not free_mask[goal]:
            return None

        g_score = {start: 0.0}
        parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
        heap: List[Tuple[float, float, Tuple[int, int]]] = []
        start_h = math.hypot(float(goal[0] - start[0]), float(goal[1] - start[1]))
        heapq.heappush(heap, (start_h, 0.0, start))
        closed = set()

        while heap:
            _f_score, dist, cell = heapq.heappop(heap)
            if cell in closed:
                continue
            if cell == goal:
                return self._reconstruct_path(parents, cell)
            closed.add(cell)

            row, col = cell
            for d_row, d_col, step_cost in _NEIGHBORS_8:
                next_cell = (row + d_row, col + d_col)
                nr, nc = next_cell
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if not free_mask[nr, nc] or next_cell in closed:
                    continue
                next_dist = float(dist) + float(step_cost)
                if next_dist + 1e-6 >= float(g_score.get(next_cell, math.inf)):
                    continue
                parents[next_cell] = cell
                g_score[next_cell] = next_dist
                h_score = math.hypot(float(goal[0] - nr), float(goal[1] - nc))
                heapq.heappush(heap, (next_dist + h_score, next_dist, next_cell))
        return None

    @staticmethod
    def _reconstruct_path(
        parents: Dict[Tuple[int, int], Tuple[int, int]],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = [goal]
        cell = goal
        while cell in parents:
            cell = parents[cell]
            path.append(cell)
        path.reverse()
        return path

    @staticmethod
    def _path_length_px(path: Sequence[Tuple[int, int]]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for idx in range(1, len(path)):
            prev_row, prev_col = path[idx - 1]
            row, col = path[idx]
            total += math.hypot(float(row - prev_row), float(col - prev_col))
        return total

    def _line_is_free(
        self,
        free_mask: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> bool:
        dist = max(1.0, math.hypot(float(goal[0] - start[0]), float(goal[1] - start[1])))
        steps = max(2, int(math.ceil(dist)))
        for idx in range(steps + 1):
            alpha = float(idx) / float(steps)
            row = int(round(float(start[0]) + alpha * float(goal[0] - start[0])))
            col = int(round(float(start[1]) + alpha * float(goal[1] - start[1])))
            if not (0 <= row < free_mask.shape[0] and 0 <= col < free_mask.shape[1]):
                return False
            if not free_mask[row, col]:
                return False
        return True

    def _candidate_cells(
        self,
        free_mask: np.ndarray,
        start: Tuple[int, int],
        clearance: np.ndarray,
    ) -> Iterable[Tuple[int, int]]:
        min_dist_px = float(self.config.min_candidate_distance_m) / self.resolution_m
        max_dist_px = float(self.config.max_candidate_distance_m) / self.resolution_m
        stride_px = max(1, int(round(float(self.config.candidate_stride_m) / self.resolution_m)))
        min_clearance_m = max(0.0, float(self.config.min_clearance_m))
        height, width = free_mask.shape
        start_row, start_col = start
        row_min = max(0, int(math.floor(start_row - max_dist_px)))
        row_max = min(height - 1, int(math.ceil(start_row + max_dist_px)))
        col_min = max(0, int(math.floor(start_col - max_dist_px)))
        col_max = min(width - 1, int(math.ceil(start_col + max_dist_px)))

        for row in range(row_min, row_max + 1, stride_px):
            for col in range(col_min, col_max + 1, stride_px):
                if not free_mask[row, col]:
                    continue
                dist_px = math.hypot(float(row - start_row), float(col - start_col))
                if dist_px < min_dist_px or dist_px > max_dist_px:
                    continue
                if float(clearance[row, col]) < min_clearance_m:
                    continue
                yield int(row), int(col)

    def build_candidates(
        self,
        *,
        full_map: Any,
        pose_xytheta: Optional[Sequence[float]],
        crop_offset: Optional[Sequence[int]],
        trajectory_points: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Tuple[List[GeometricCandidate], Dict[int, List[Tuple[int, int]]]]:
        full_map_array = self._as_full_map_array(full_map)
        if full_map_array is None or pose_xytheta is None or crop_offset is None:
            self._last_debug = {"reason": "missing_map_or_pose"}
            return [], {}

        free_mask, inflated_obstacle, clearance = self._build_masks(full_map_array)
        if not np.any(free_mask):
            self._last_debug = {"reason": "no_free_cells"}
            return [], {}

        projector = self._build_projector(full_map_array, pose_xytheta, crop_offset)
        start_world = self._pose_to_world_pixel(pose_xytheta, self.resolution_cm)
        if start_world is None:
            self._last_debug = {"reason": "missing_start_world"}
            return [], {}
        start_rotated = projector.world_to_rotated_pixel(*start_world)
        if start_rotated is None:
            self._last_debug = {"reason": "start_outside_map"}
            return [], {}

        snap_radius = max(2, int(math.ceil(0.5 / self.resolution_m)))
        start = self._nearest_free_cell(
            free_mask=free_mask,
            row=int(round(start_rotated[0])),
            col=int(round(start_rotated[1])),
            max_radius=snap_radius,
        )
        if start is None:
            self._last_debug = {"reason": "start_not_near_free"}
            return [], {}

        history_cells = self._trajectory_world_points_to_cells(
            projector=projector,
            trajectory_points=trajectory_points,
            shape=free_mask.shape,
        )
        raw_candidates = []
        candidate_paths: Dict[int, List[Tuple[int, int]]] = {}
        for row, col in self._candidate_cells(free_mask, start, clearance):
            if not self._line_is_free(free_mask, start, (row, col)):
                # Still allow non-straight A* paths through corners, but rank them lower.
                line_clear_bonus = 0.0
            else:
                line_clear_bonus = 1.0
            path = self._a_star(free_mask, start, (row, col))
            if not path:
                continue
            path_len_px = self._path_length_px(path)
            path_len_m = path_len_px * self.resolution_m
            world = projector.rotated_to_world_pixel(float(row), float(col))
            if world is None:
                continue
            world_row, world_col = world
            rel_col = float(col - start[1])
            rel_forward = float(start[0] - row)
            bearing_deg = normalize_relative_bearing(math.degrees(math.atan2(rel_col, rel_forward)))
            distance_m = math.hypot(float(row - start[0]), float(col - start[1])) * self.resolution_m
            is_backtrack = self._cell_near_history((row, col), history_cells)
            raw_candidates.append((
                self._candidate_rank_key(
                    distance_m=distance_m,
                    path_len_m=path_len_m,
                    bearing_deg=bearing_deg,
                    clearance_m=float(clearance[row, col]),
                    is_backtrack=is_backtrack,
                    line_clear_bonus=line_clear_bonus,
                ),
                row,
                col,
                world_row,
                world_col,
                distance_m,
                path_len_m,
                bearing_deg,
                float(clearance[row, col]),
                is_backtrack,
                path,
            ))

        raw_candidates.sort(key=lambda item: item[0])
        selected: List[GeometricCandidate] = []
        for item in raw_candidates:
            (
                _rank,
                row,
                col,
                world_row,
                world_col,
                distance_m,
                path_len_m,
                bearing_deg,
                clearance_m,
                is_backtrack,
                path,
            ) = item
            if self._too_close_to_selected((row, col), selected):
                continue
            candidate_id = len(selected) + 1
            candidate = GeometricCandidate(
                candidate_id=candidate_id,
                row=int(row),
                col=int(col),
                world_row=float(world_row),
                world_col=float(world_col),
                world_x_m=float(world_col) * self.resolution_m,
                world_y_m=float(world_row) * self.resolution_m,
                distance_m=float(distance_m),
                path_length_m=float(path_len_m),
                bearing_deg=float(bearing_deg),
                direction=format_relative_direction(float(bearing_deg)),
                clearance_m=float(clearance_m),
                is_backtrack=bool(is_backtrack),
            )
            selected.append(candidate)
            candidate_paths[candidate_id] = list(path)
            if len(selected) >= int(self.config.max_candidates):
                break

        self._last_debug = {
            "reason": "ok",
            "free_cells": int(np.count_nonzero(free_mask)),
            "inflated_obstacle_cells": int(np.count_nonzero(inflated_obstacle)),
            "raw_candidate_count": int(len(raw_candidates)),
            "selected_candidate_count": int(len(selected)),
            "start_cell": [int(start[0]), int(start[1])],
        }
        return selected, candidate_paths

    def _candidate_rank_key(
        self,
        *,
        distance_m: float,
        path_len_m: float,
        bearing_deg: float,
        clearance_m: float,
        is_backtrack: bool,
        line_clear_bonus: float,
    ) -> Tuple[float, float, float, float]:
        forward_penalty = abs(float(bearing_deg)) / 180.0
        backtrack_penalty = 0.75 if is_backtrack else 0.0
        clearance_bonus = min(0.4, max(0.0, float(clearance_m)))
        score = (
            float(path_len_m)
            + forward_penalty * 0.8
            + backtrack_penalty
            - clearance_bonus * 0.25
            - float(line_clear_bonus) * 0.2
        )
        return score, abs(float(bearing_deg)), -float(clearance_m), float(distance_m)

    def _too_close_to_selected(
        self,
        cell: Tuple[int, int],
        selected: Sequence[GeometricCandidate],
    ) -> bool:
        min_sep_px = max(1.0, float(self.config.candidate_stride_m) / self.resolution_m * 0.85)
        for candidate in selected:
            if math.hypot(float(cell[0] - candidate.row), float(cell[1] - candidate.col)) < min_sep_px:
                return True
        return False

    @staticmethod
    def _cell_near_history(
        cell: Tuple[int, int],
        history_cells: Sequence[Tuple[int, int]],
        max_dist_cells: int = 6,
    ) -> bool:
        if not history_cells:
            return False
        row, col = cell
        max_dist_sq = float(max_dist_cells * max_dist_cells)
        return any(
            (float(row - hist_row) ** 2 + float(col - hist_col) ** 2) <= max_dist_sq
            for hist_row, hist_col in history_cells
        )

    @staticmethod
    def _trajectory_world_points_to_cells(
        *,
        projector: RotatedMapProjector,
        trajectory_points: Optional[Sequence[Tuple[int, int]]],
        shape: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        for point in list(trajectory_points or []):
            if point is None or len(point) < 2:
                continue
            projected = projector.world_to_rotated_pixel(float(point[0]), float(point[1]))
            if projected is None:
                continue
            row, col = int(round(projected[0])), int(round(projected[1]))
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                cells.append((row, col))
        return cells

    def build_plan(
        self,
        *,
        candidate: GeometricCandidate,
        candidate_paths: Dict[int, List[Tuple[int, int]]],
        full_map: Any,
        pose_xytheta: Optional[Sequence[float]],
        crop_offset: Optional[Sequence[int]],
    ) -> Optional[GeometricPlan]:
        full_map_array = self._as_full_map_array(full_map)
        if full_map_array is None or pose_xytheta is None or crop_offset is None:
            return None
        path = list(candidate_paths.get(int(candidate.candidate_id)) or [])
        if not path:
            free_mask, _inflated, _clearance = self._build_masks(full_map_array)
            start_world = self._pose_to_world_pixel(pose_xytheta, self.resolution_cm)
            if start_world is None:
                return None
            projector = self._build_projector(full_map_array, pose_xytheta, crop_offset)
            start_rotated = projector.world_to_rotated_pixel(*start_world)
            if start_rotated is None:
                return None
            start = self._nearest_free_cell(
                free_mask=free_mask,
                row=int(round(start_rotated[0])),
                col=int(round(start_rotated[1])),
                max_radius=max(2, int(math.ceil(0.5 / self.resolution_m))),
            )
            if start is None:
                return None
            path = self._a_star(free_mask, start, (candidate.row, candidate.col)) or []
        if not path:
            return None

        projector = self._build_projector(full_map_array, pose_xytheta, crop_offset)
        world_points: List[Tuple[float, float]] = []
        action_points: List[Tuple[float, float]] = []
        for row, col in self._downsample_path_cells(path):
            world = projector.rotated_to_world_pixel(float(row), float(col))
            if world is None:
                continue
            world_row, world_col = world
            world_points.append((float(world_row), float(world_col)))
            action_points.append((
                float(world_col) * self.resolution_m,
                float(world_row) * self.resolution_m,
            ))
        if len(action_points) < 2:
            action_points.append((float(candidate.world_x_m), float(candidate.world_y_m)))
        return GeometricPlan(
            candidate=candidate,
            path_cells=tuple((int(r), int(c)) for r, c in path),
            world_points=tuple(world_points),
            action_points=tuple(action_points),
        )

    def _downsample_path_cells(
        self,
        path: Sequence[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        if not path:
            return []
        step_px = max(1, int(round(float(self.config.path_step_m) / self.resolution_m)))
        sampled = [tuple(path[0])]
        last = tuple(path[0])
        accum = 0.0
        for cell in path[1:]:
            cell = tuple(cell)
            accum += math.hypot(float(cell[0] - last[0]), float(cell[1] - last[1]))
            last = cell
            if accum >= float(step_px):
                sampled.append(cell)
                accum = 0.0
        if sampled[-1] != tuple(path[-1]):
            sampled.append(tuple(path[-1]))
        return sampled


def serialize_candidates(candidates: Sequence[GeometricCandidate]) -> List[Dict[str, Any]]:
    return [candidate.to_dict() for candidate in list(candidates or [])]


def render_candidate_map(
    *,
    full_map: Any,
    candidates: Sequence[GeometricCandidate],
    plan: Optional[GeometricPlan] = None,
    output_size_px: int = 512,
) -> Optional[np.ndarray]:
    """Render the rotated local map with model-selectable waypoint labels."""
    full_map_array = GeometricWaypointPlanner._as_full_map_array(full_map)
    if full_map_array is None:
        return None

    obstacle = np.asarray(full_map_array[0] > 0.5, dtype=bool)
    explored = np.asarray(full_map_array[1] > 0.5, dtype=bool)
    floor = np.logical_and(explored, np.logical_not(obstacle))
    height, width = obstacle.shape
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    image[~explored] = (245, 245, 245)
    image[explored] = (226, 232, 224)
    image[floor] = (182, 224, 184)
    image[obstacle] = (28, 28, 28)

    if plan is not None and plan.path_cells:
        path_points = [
            (int(col), int(row))
            for row, col in list(plan.path_cells)
            if 0 <= int(row) < height and 0 <= int(col) < width
        ]
        if len(path_points) >= 2:
            cv2.polylines(
                image,
                [np.asarray(path_points, dtype=np.int32)],
                isClosed=False,
                color=(255, 72, 196),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    center = (int(round(width / 2.0)), int(round(height / 2.0)))
    arrow_tip = (center[0], max(0, center[1] - 18))
    cv2.arrowedLine(image, center, arrow_tip, (45, 45, 255), 3, cv2.LINE_AA, tipLength=0.45)
    cv2.circle(image, center, 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image, center, 6, (45, 45, 255), 2, cv2.LINE_AA)

    for candidate in list(candidates or []):
        point = (int(candidate.col), int(candidate.row))
        color = (0, 188, 255) if not candidate.is_backtrack else (255, 128, 64)
        cv2.circle(image, point, 12, color, -1, cv2.LINE_AA)
        cv2.circle(image, point, 13, (255, 255, 255), 2, cv2.LINE_AA)
        label = str(int(candidate.candidate_id))
        cv2.putText(
            image,
            label,
            (point[0] - 5, point[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    if int(output_size_px) > 0 and (height != int(output_size_px) or width != int(output_size_px)):
        image = cv2.resize(
            image,
            (int(output_size_px), int(output_size_px)),
            interpolation=cv2.INTER_NEAREST,
        )
    return image
