"""Waypoint state manager for world-map navigation history."""

from typing import Any, Dict, List, Optional, Tuple


class WaypointManager:
    """Maintain the ordered waypoint chain in world-map pixels."""

    def __init__(self, resolution: int = 5):
        self.resolution = resolution
        self.reset()

    def reset(self) -> None:
        self.positions: List[Tuple[int, int]] = []
        self.ids: List[int] = []
        self.descriptions: List[str] = []
        self.area_labels: List[str] = []
        self.floor_ids: List[int] = []
        self.initial_neighborhood_flags: List[bool] = []
        self.counter = 0
        self.initial_waypoint_index: Optional[int] = 0

    def add_waypoint(
        self,
        pixel_y: int,
        pixel_x: int,
        description: str = "",
        area_label: str = "",
        floor_id: Optional[int] = None,
        waypoint_id: Optional[int] = None,
        near_initial_neighborhood: bool = False,
    ) -> int:
        """Append a waypoint while preserving the full chronological history."""
        if waypoint_id is None:
            self.counter += 1
            waypoint_id = self.counter
        else:
            waypoint_id = int(waypoint_id)
            self.counter = max(int(self.counter), int(waypoint_id))
        self.positions.append((int(pixel_y), int(pixel_x)))
        self.ids.append(int(waypoint_id))
        self.descriptions.append(description)
        self.area_labels.append(area_label)
        self.floor_ids.append(int(floor_id) if floor_id is not None else 0)
        self.initial_neighborhood_flags.append(bool(near_initial_neighborhood))
        return int(waypoint_id)

    def get_waypoints(self) -> Tuple[List[Tuple[int, int]], List[int], List[str]]:
        return self.positions, self.ids, self.descriptions

    def get_area_labels(self) -> List[str]:
        return list(self.area_labels)

    def get_floor_ids(self) -> List[int]:
        return list(self.floor_ids)

    def get_initial_neighborhood_flags(self) -> List[bool]:
        return [bool(flag) for flag in self.initial_neighborhood_flags]

    def clear(self) -> None:
        self.reset()

    def count(self) -> int:
        return len(self.ids)

    def export_state(self) -> Dict[str, Any]:
        return {
            "positions": [(int(y), int(x)) for y, x in self.positions],
            "ids": [int(wp_id) for wp_id in self.ids],
            "descriptions": [str(text) for text in self.descriptions],
            "area_labels": [str(label) for label in self.area_labels],
            "floor_ids": [int(floor_id) for floor_id in self.floor_ids],
            "initial_neighborhood_flags": [bool(flag) for flag in self.initial_neighborhood_flags],
            "counter": int(self.counter),
            "initial_waypoint_index": (
                int(self.initial_waypoint_index)
                if self.initial_waypoint_index is not None
                else None
            ),
        }

    def import_state(self, state: Optional[Dict[str, Any]]) -> None:
        self.reset()
        if not state:
            return
        self.positions = [
            (int(item[0]), int(item[1]))
            for item in list(state.get("positions", []) or [])
            if item is not None and len(item) >= 2
        ]
        self.ids = [int(item) for item in list(state.get("ids", []) or [])]
        self.descriptions = [str(text) for text in list(state.get("descriptions", []) or [])]
        self.area_labels = [str(text) for text in list(state.get("area_labels", []) or [])]
        raw_floor_ids = list(state.get("floor_ids", []) or [])
        if raw_floor_ids:
            self.floor_ids = [int(item) for item in raw_floor_ids[:len(self.ids)]]
        else:
            self.floor_ids = [0 for _ in self.ids]
        if len(self.floor_ids) < len(self.ids):
            self.floor_ids.extend([0 for _ in range(len(self.ids) - len(self.floor_ids))])
        raw_initial_flags = list(state.get("initial_neighborhood_flags", []) or [])
        self.initial_neighborhood_flags = [
            bool(raw_initial_flags[index])
            for index in range(min(len(raw_initial_flags), len(self.ids)))
        ]
        if len(self.initial_neighborhood_flags) < len(self.ids):
            self.initial_neighborhood_flags.extend(
                [False for _ in range(len(self.ids) - len(self.initial_neighborhood_flags))]
            )
        self.counter = int(state.get("counter", len(self.ids)) or len(self.ids))
        initial_index = state.get("initial_waypoint_index", 0)
        self.initial_waypoint_index = (
            int(initial_index)
            if initial_index is not None
            else None
        )
