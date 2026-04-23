"""Runtime OVON top-down visualization fixes."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


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


def _goal_view_point_positions(top_down_map: Any, goal: Any) -> list:
    positions = []
    for view_point in list(getattr(goal, "view_points", None) or []):
        try:
            position = view_point.agent_state.position
        except AttributeError:
            continue
        if _height_within_current_level(top_down_map, position[1], tolerance=0.75):
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
    radius = max(int(getattr(top_down_map, "point_padding", 2)) + 2, 5)
    thickness = -1 if fill else max(2, int(getattr(top_down_map, "line_thickness", 1)))
    cv2.circle(
        top_down_map._top_down_map,
        (int(t_y), int(t_x)),
        int(radius),
        int(point_type),
        thickness,
    )


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
                    thickness=max(2, int(getattr(top_down_map, "line_thickness", 1))),
                )
            return True

    radius = max(int(getattr(top_down_map, "point_padding", 2)) + 4, 8)
    if not fill and map_points:
        xs = [point[0] for point in map_points]
        ys = [point[1] for point in map_points]
        x_min = max(0, min(xs) - radius)
        x_max = min(top_down_map._top_down_map.shape[1] - 1, max(xs) + radius)
        y_min = max(0, min(ys) - radius)
        y_max = min(top_down_map._top_down_map.shape[0] - 1, max(ys) + radius)
        cv2.rectangle(
            top_down_map._top_down_map,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            int(point_type),
            max(2, int(getattr(top_down_map, "line_thickness", 1))),
        )
        return True

    for position in positions:
        _draw_goal_region_disk(top_down_map, position, point_type, fill=fill)
    return True


def _patch_maps_module() -> None:
    from habitat.utils.visualizations import maps

    special_indicators = {
        int(maps.MAP_SOURCE_POINT_INDICATOR),
        int(maps.MAP_TARGET_POINT_INDICATOR),
        int(maps.MAP_SHORTEST_PATH_COLOR),
        int(maps.MAP_VIEW_POINT_INDICATOR),
        int(maps.MAP_TARGET_BOUNDING_BOX),
    }
    maps.TOP_DOWN_MAP_COLORS[maps.MAP_TARGET_POINT_INDICATOR] = [255, 64, 64]
    maps.TOP_DOWN_MAP_COLORS[maps.MAP_SHORTEST_PATH_COLOR] = [64, 210, 128]
    maps.TOP_DOWN_MAP_COLORS[maps.MAP_VIEW_POINT_INDICATOR] = [255, 170, 0]
    maps.TOP_DOWN_MAP_COLORS[maps.MAP_TARGET_BOUNDING_BOX] = [170, 95, 255]

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
    original_update_map = top_down_cls.update_map
    original_draw_shortest_path = top_down_cls._draw_shortest_path

    def draw_goal_view_point_markers(self, goal):
        positions = _goal_view_point_positions(self, goal)
        if not positions:
            return False
        radius = max(int(getattr(self, "point_padding", 2)) + 1, 3)
        for position in positions:
            t_x, t_y = _to_grid(self, position)
            cv2.circle(
                self._top_down_map,
                (int(t_y), int(t_x)),
                int(radius),
                int(maps.MAP_VIEW_POINT_INDICATOR),
                thickness=-1,
            )
        return True

    def draw_goals_view_points(self, episode):
        if not bool(getattr(self._config, "draw_view_points", True)):
            return
        for goal in list(getattr(episode, "goals", []) or []):
            if draw_goal_view_point_markers(self, goal):
                continue
            try:
                if _height_within_current_level(self, goal.position[1], tolerance=2.5):
                    _draw_goal_region_disk(
                        self,
                        goal.position,
                        maps.MAP_VIEW_POINT_INDICATOR,
                        fill=True,
                    )
            except AttributeError:
                continue

    def draw_goals_positions(self, episode):
        if not bool(getattr(self._config, "draw_goal_positions", True)):
            return
        for goal in list(getattr(episode, "goals", []) or []):
            try:
                if not _height_within_current_level(self, goal.position[1], tolerance=2.5):
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

        for goal in list(getattr(episode, "goals", []) or []):
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
            if not _height_within_current_level(self, center[1], tolerance=2.5):
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
            maps.draw_path(
                self._top_down_map,
                map_corners,
                maps.MAP_TARGET_BOUNDING_BOX,
                max(2, int(getattr(self, "line_thickness", 1))),
            )

    def draw_static_goal_overlays(self, episode, agent_position):
        if not hasattr(episode, "goals"):
            return
        draw_goals_aabb(self, episode)
        original_draw_shortest_path(self, episode, agent_position)
        draw_goals_view_points(self, episode)
        draw_goals_positions(self, episode)

    def reset_metric(self, episode, *args, **kwargs):
        original_reset_metric(self, episode, *args, **kwargs)
        draw_static_goal_overlays(self, episode, self._sim.get_agent_state().position)

    def update_metric(self, episode, action, *args, **kwargs):
        self._step_count += 1
        agent_position = self._sim.get_agent_state().position
        house_map, map_agent_x, map_agent_y = original_update_map(self, agent_position)
        draw_static_goal_overlays(self, episode, agent_position)
        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": (map_agent_x, map_agent_y),
            "agent_angle": self.get_polar_angle(),
        }

    def is_on_same_floor(self, height, ref_floor_height=None, ceiling_height=0.75):
        return _height_within_current_level(
            self,
            height,
            ref_floor_height=ref_floor_height,
            tolerance=float(ceiling_height),
        )

    top_down_cls._draw_goals_aabb = draw_goals_aabb
    top_down_cls._draw_goals_view_points = draw_goals_view_points
    top_down_cls._draw_goals_positions = draw_goals_positions
    top_down_cls._draw_static_goal_overlays = draw_static_goal_overlays
    top_down_cls._is_on_same_floor = is_on_same_floor
    top_down_cls.reset_metric = reset_metric
    top_down_cls.update_metric = update_metric
    top_down_cls._spacevln_ovon_visualization_patched = True
