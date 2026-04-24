"""Runtime OVON top-down visualization fixes."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

VISIBLE_GOAL_LEVEL_TOLERANCE_M = 2.5


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _current_agent_height(top_down_map: Any, ref_floor_height: Any = None) -> float:
    if ref_floor_height is not None:
        return _as_float(ref_floor_height)
    try:
        return _as_float(top_down_map._sim.get_agent(0).state.position[1])
    except Exception:
        return 0.0


def _height_within_current_level(
    top_down_map: Any,
    height: Any,
    *,
    ref_floor_height: Any = None,
    tolerance: float,
) -> bool:
    return abs(_as_float(height) - _current_agent_height(top_down_map, ref_floor_height)) <= float(tolerance)


def _goal_view_point_positions(
    top_down_map: Any,
    goal: Any,
    *,
    apply_floor_filter: bool = True,
) -> list:
    positions = []
    for view_point in list(getattr(goal, "view_points", None) or []):
        try:
            position = view_point.agent_state.position
        except AttributeError:
            continue
        if (not apply_floor_filter) or _height_within_current_level(
            top_down_map,
            position[1],
            tolerance=VISIBLE_GOAL_LEVEL_TOLERANCE_M,
        ):
            positions.append(position)
    return positions


def _to_grid(top_down_map: Any, position: Iterable[float]) -> tuple[int, int]:
    from habitat.utils.visualizations import maps

    return maps.to_grid(
        position[2],
        position[0],
        (top_down_map._top_down_map.shape[0], top_down_map._top_down_map.shape[1]),
        sim=top_down_map._sim,
    )


def _draw_goal_region_disk(top_down_map: Any, position: Any, point_type: int, *, fill: bool) -> None:
    import cv2

    t_x, t_y = _to_grid(top_down_map, position)
    radius = max(int(getattr(top_down_map, "point_padding", 2)) + 1, 3)
    thickness = -1 if fill else max(1, int(getattr(top_down_map, "line_thickness", 1)))
    cv2.circle(
        top_down_map._top_down_map,
        (int(t_y), int(t_x)),
        int(radius),
        int(point_type),
        thickness,
    )


def _draw_goal_view_point_markers(
    top_down_map: Any,
    goal: Any,
    point_type: int,
    *,
    apply_floor_filter: bool = True,
) -> bool:
    import cv2

    positions = _goal_view_point_positions(
        top_down_map,
        goal,
        apply_floor_filter=apply_floor_filter,
    )
    if not positions:
        return False

    point_padding = int(getattr(top_down_map, "point_padding", 2) or 2)
    radius = max(1, min(point_padding, 2))
    for position in positions:
        t_x, t_y = _to_grid(top_down_map, position)
        cv2.circle(
            top_down_map._top_down_map,
            (int(t_y), int(t_x)),
            int(radius),
            int(point_type),
            thickness=-1,
        )
    return True


def _draw_goal_view_point_region(top_down_map: Any, goal: Any, point_type: int, *, fill: bool) -> bool:
    import cv2

    positions = _goal_view_point_positions(top_down_map, goal)
    if not positions:
        return False

    map_points = []
    for position in positions:
        t_x, t_y = _to_grid(top_down_map, position)
        map_points.append((int(t_y), int(t_x)))

    if len(map_points) >= 3:
        hull = cv2.convexHull(np.array(map_points, dtype=np.int32))
        if cv2.contourArea(hull) > 1.0:
            if fill:
                cv2.fillConvexPoly(top_down_map._top_down_map, hull, int(point_type))
            else:
                cv2.polylines(
                    top_down_map._top_down_map,
                    [hull],
                    isClosed=True,
                    color=int(point_type),
                    thickness=max(1, int(getattr(top_down_map, "line_thickness", 1))),
                )
            return True

    radius = max(int(getattr(top_down_map, "point_padding", 2)) + 2, 5)
    if len(map_points) >= 2:
        xs = [point[0] for point in map_points]
        ys = [point[1] for point in map_points]
        x_min = max(0, min(xs) - radius)
        x_max = min(top_down_map._top_down_map.shape[1] - 1, max(xs) + radius)
        y_min = max(0, min(ys) - radius)
        y_max = min(top_down_map._top_down_map.shape[0] - 1, max(ys) + radius)
        thickness = -1 if fill else max(1, int(getattr(top_down_map, "line_thickness", 1)))
        cv2.rectangle(
            top_down_map._top_down_map,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            int(point_type),
            thickness,
        )
        return True

    for position in positions:
        _draw_goal_region_disk(top_down_map, position, point_type, fill=fill)
    return True


def _goal_identity(goal: Any) -> tuple:
    object_id = str(getattr(goal, "object_id", "") or "")
    category = str(getattr(goal, "object_category", "") or "")
    position = getattr(goal, "position", None)
    if position is None:
        position_key = ()
    else:
        try:
            position_key = tuple(round(float(value), 4) for value in list(position)[:3])
        except Exception:
            position_key = ()
    return object_id, category, position_key


def _extract_task_from_call(args: tuple, kwargs: dict) -> Any:
    task = kwargs.get("task")
    if task is not None:
        return task
    for value in args:
        if hasattr(value, "_dataset"):
            return value
    return None


def _episode_success_goals(top_down_map: Any, episode: Any) -> list:
    aggregated = []
    seen = set()

    def extend(goals: Any) -> None:
        for goal in list(goals or []):
            identity = _goal_identity(goal)
            if identity in seen:
                continue
            seen.add(identity)
            aggregated.append(goal)

    extend(getattr(episode, "goals", None) or [])

    task = getattr(top_down_map, "_spacevln_task", None)
    dataset = getattr(task, "_dataset", None)
    goals_by_category = getattr(dataset, "goals_by_category", None) or {}
    goals_key = getattr(episode, "goals_key", None)

    if goals_key and goals_key in goals_by_category:
        extend(goals_by_category.get(goals_key))

    scene_leaf = str(getattr(episode, "scene_id", "") or "").split("/")[-1]
    for child_category in list(getattr(episode, "children_object_categories", None) or []):
        child_key = f"{scene_leaf}_{child_category}"
        if child_key in goals_by_category:
            extend(goals_by_category.get(child_key))

    return aggregated


def _patch_maps_module() -> None:
    from habitat.utils.visualizations import maps

    special_indicators = {
        int(maps.MAP_SOURCE_POINT_INDICATOR),
        int(maps.MAP_TARGET_POINT_INDICATOR),
        int(maps.MAP_SHORTEST_PATH_COLOR),
        int(maps.MAP_VIEW_POINT_INDICATOR),
        int(maps.MAP_TARGET_BOUNDING_BOX),
    }
    maps.TOP_DOWN_MAP_COLORS[maps.MAP_TARGET_POINT_INDICATOR] = [200, 0, 0]
    maps.TOP_DOWN_MAP_COLORS[maps.MAP_SHORTEST_PATH_COLOR] = [0, 200, 0]
    maps.TOP_DOWN_MAP_COLORS[maps.MAP_VIEW_POINT_INDICATOR] = [245, 150, 150]
    maps.TOP_DOWN_MAP_COLORS[maps.MAP_TARGET_BOUNDING_BOX] = [245, 150, 150]

    if getattr(maps, "_spacevln_ovon_colorize_patched", False):
        return

    original_colorize = maps.colorize_topdown_map

    def colorize_topdown_map(top_down_map, fog_of_war_mask=None, fog_of_war_desat_amount=0.5):
        colored = original_colorize(
            top_down_map,
            fog_of_war_mask=None,
            fog_of_war_desat_amount=fog_of_war_desat_amount,
        )
        if fog_of_war_mask is None:
            return colored
        desat_values = np.array([[fog_of_war_desat_amount], [1.0]])
        desat_mask = np.logical_and(
            top_down_map != maps.MAP_INVALID_POINT,
            ~np.isin(top_down_map, tuple(special_indicators)),
        )
        colored[desat_mask] = (
            colored * desat_values[fog_of_war_mask]
        ).astype(np.uint8)[desat_mask]
        return colored

    maps.colorize_topdown_map = colorize_topdown_map
    maps._spacevln_ovon_colorize_patched = True


def install_ovon_topdown_visualization_patch() -> None:
    """Patch Habitat's OVON top-down map rendering in the active Python process."""

    import cv2
    from habitat.tasks.nav import nav
    from habitat.utils.visualizations import maps

    _patch_maps_module()
    top_down_cls = nav.TopDownMap
    if getattr(top_down_cls, "_spacevln_ovon_visualization_patched", False):
        return

    original_reset_metric = top_down_cls.reset_metric
    original_draw_shortest_path = top_down_cls._draw_shortest_path

    def cache_initial_shortest_path(self) -> None:
        points = list(getattr(self, "_shortest_path_points", []) or [])
        self._spacevln_initial_shortest_path_points = [tuple(point) for point in points]

    def goal_shortest_path_candidates(self, episode):
        candidates = []
        fallback_positions = []
        for goal in _episode_success_goals(self, episode):
            goal_view_points = _goal_view_point_positions(
                self,
                goal,
                apply_floor_filter=False,
            )
            if goal_view_points:
                candidates.extend(goal_view_points)
            try:
                if not goal_view_points and _height_within_current_level(
                    self,
                    goal.position[1],
                    tolerance=VISIBLE_GOAL_LEVEL_TOLERANCE_M,
                ):
                    fallback_positions.append(goal.position)
            except AttributeError:
                pass
        if not candidates:
            candidates.extend(fallback_positions)
        return candidates

    def draw_goals_view_points(self, episode):
        if not bool(getattr(self._config, "draw_view_points", True)):
            return
        for goal in _episode_success_goals(self, episode):
            try:
                drew_region = _draw_goal_view_point_region(
                    self,
                    goal,
                    maps.MAP_VIEW_POINT_INDICATOR,
                    fill=True,
                )
                if (not drew_region) and _height_within_current_level(
                    self,
                    goal.position[1],
                    tolerance=VISIBLE_GOAL_LEVEL_TOLERANCE_M,
                ):
                    _draw_goal_region_disk(
                        self,
                        goal.position,
                        maps.MAP_VIEW_POINT_INDICATOR,
                        fill=True,
                    )
            except AttributeError:
                continue

    def draw_goal_view_point_markers_overlay(self, episode):
        if not bool(getattr(self._config, "draw_view_points", True)):
            return
        for goal in _episode_success_goals(self, episode):
            try:
                _draw_goal_view_point_markers(
                    self,
                    goal,
                    maps.MAP_VIEW_POINT_INDICATOR,
                    apply_floor_filter=True,
                )
            except AttributeError:
                continue

    def draw_goals_positions(self, episode):
        if not bool(getattr(self._config, "draw_goal_positions", True)):
            return
        for goal in _episode_success_goals(self, episode):
            try:
                if not _height_within_current_level(
                    self,
                    goal.position[1],
                    tolerance=VISIBLE_GOAL_LEVEL_TOLERANCE_M,
                ):
                    continue
                _draw_goal_region_disk(
                    self,
                    goal.position,
                    maps.MAP_TARGET_POINT_INDICATOR,
                    fill=True,
                )
            except AttributeError:
                continue

    def draw_goals_aabb(self, episode):
        if not bool(getattr(self._config, "draw_goal_aabbs", True)):
            return
        try:
            sem_scene = self._sim.semantic_annotations()
            sem_objects = list(getattr(sem_scene, "objects", []) or [])
        except Exception:
            sem_objects = []

        for goal in _episode_success_goals(self, episode):
            sem_obj = None
            goal_object_id = str(getattr(goal, "object_id", ""))
            try:
                object_index = int(goal_object_id)
            except (TypeError, ValueError):
                object_index = None

            if object_index is not None and 0 <= object_index < len(sem_objects):
                indexed_obj = sem_objects[object_index]
                indexed_obj_id = str(getattr(indexed_obj, "id", ""))
                if indexed_obj_id == goal_object_id or indexed_obj_id.split("_")[-1] == goal_object_id:
                    sem_obj = indexed_obj

            if sem_obj is None:
                for candidate_obj in sem_objects:
                    candidate_obj_id = str(getattr(candidate_obj, "id", ""))
                    if candidate_obj_id == goal_object_id or candidate_obj_id.split("_")[-1] == goal_object_id:
                        sem_obj = candidate_obj
                        break

            if sem_obj is None or getattr(sem_obj, "aabb", None) is None:
                _draw_goal_view_point_region(
                    self,
                    goal,
                    maps.MAP_TARGET_BOUNDING_BOX,
                    fill=False,
                )
                continue

            center = sem_obj.aabb.center
            if not _height_within_current_level(
                self,
                center[1],
                tolerance=VISIBLE_GOAL_LEVEL_TOLERANCE_M,
            ):
                _draw_goal_view_point_region(
                    self,
                    goal,
                    maps.MAP_TARGET_BOUNDING_BOX,
                    fill=False,
                )
                continue

            x_len, _, z_len = sem_obj.aabb.sizes / 2.0
            corners = [
                center + np.array([x, 0, z])
                for x, z in [
                    (-x_len, -z_len),
                    (-x_len, z_len),
                    (x_len, z_len),
                    (x_len, -z_len),
                    (-x_len, -z_len),
                ]
            ]
            map_corners = [_to_grid(self, point) for point in corners]
            if len(map_corners) < 2:
                _draw_goal_view_point_region(
                    self,
                    goal,
                    maps.MAP_TARGET_BOUNDING_BOX,
                    fill=False,
                )
                continue
            polygon = np.array([(int(col), int(row)) for row, col in map_corners[:-1]], dtype=np.int32)
            if polygon.shape[0] >= 3:
                cv2.polylines(
                    self._top_down_map,
                    [polygon],
                    isClosed=True,
                    color=int(maps.MAP_TARGET_BOUNDING_BOX),
                    thickness=max(1, int(getattr(self, "line_thickness", 1))),
                )
            else:
                _draw_goal_view_point_region(
                    self,
                    goal,
                    maps.MAP_TARGET_BOUNDING_BOX,
                    fill=False,
                )

    def draw_static_goal_overlays(self, episode, agent_position):
        if not hasattr(episode, "goals"):
            return
        draw_goals_aabb(self, episode)
        draw_goals_view_points(self, episode)
        original_draw_shortest_path(self, episode, agent_position)
        cache_initial_shortest_path(self)
        draw_goals_positions(self, episode)
        draw_goal_view_point_markers_overlay(self, episode)

    def reset_metric(self, episode, *args, **kwargs):
        self._spacevln_task = _extract_task_from_call(args, kwargs)
        self._spacevln_static_top_down_map = None
        self._spacevln_path_history = []
        self._spacevln_initial_shortest_path_points = []
        original_reset_metric(self, episode, *args, **kwargs)
        self._spacevln_static_top_down_map = self._top_down_map.copy()
        self._spacevln_path_history = [tuple(self._previous_xy_location)] if self._previous_xy_location is not None else []

    def update_metric(self, episode, action, *args, **kwargs):
        if getattr(self, "_spacevln_task", None) is None:
            self._spacevln_task = _extract_task_from_call(args, kwargs)
        self._step_count += 1
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = _to_grid(self, agent_position)
        if getattr(self, "_spacevln_static_top_down_map", None) is not None:
            self._top_down_map = self._spacevln_static_top_down_map.copy()
        history = list(getattr(self, "_spacevln_path_history", []) or [])
        current_xy = (int(a_y), int(a_x))
        if not history and self._previous_xy_location is not None:
            history.append(tuple(self._previous_xy_location))
        if not history or history[-1] != current_xy:
            history.append(current_xy)
        max_steps = max(int(getattr(self._config, "max_episode_steps", 1) or 1), 1)
        for idx in range(1, len(history)):
            color = 10 + min(idx * 245 // max_steps, 245)
            cv2.line(
                self._top_down_map,
                history[idx - 1],
                history[idx],
                int(color),
                thickness=max(1, int(getattr(self, "line_thickness", 1))),
            )
        self._spacevln_path_history = history
        self.update_fog_of_war_mask(np.array([a_x, a_y]))
        self._previous_xy_location = current_xy
        self._metric = {
            "map": self._top_down_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": (a_x, a_y),
            "agent_angle": self.get_polar_angle(),
        }

    def is_on_same_floor(self, height, ref_floor_height=None, ceiling_height=0.75):
        return _height_within_current_level(
            self,
            height,
            ref_floor_height=ref_floor_height,
            tolerance=float(ceiling_height),
        )

    top_down_cls._goal_shortest_path_candidates = goal_shortest_path_candidates
    top_down_cls._draw_goals_aabb = draw_goals_aabb
    top_down_cls._draw_goals_view_points = draw_goals_view_points
    top_down_cls._draw_goals_positions = draw_goals_positions
    top_down_cls._draw_static_goal_overlays = draw_static_goal_overlays
    top_down_cls._is_on_same_floor = is_on_same_floor
    top_down_cls.reset_metric = reset_metric
    top_down_cls.update_metric = update_metric
    top_down_cls._spacevln_ovon_visualization_patched = True
