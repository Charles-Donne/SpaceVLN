"""
Waypoint state manager for world-map navigation history.
"""

from typing import List, Tuple

import numpy as np


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
        self.counter = 0

    def add_waypoint(
        self,
        pixel_y: int,
        pixel_x: int,
        description: str = "",
        area_label: str = "",
    ) -> int:
        """Append a waypoint, replacing only the immediately previous one if <2m away."""
        distance_threshold_pixels = 200.0 / float(self.resolution)
        if self.positions:
            prev_y, prev_x = self.positions[-1]
            distance = float(np.hypot(pixel_y - prev_y, pixel_x - prev_x))
            if distance < distance_threshold_pixels:
                self.positions.pop()
                self.ids.pop()
                self.descriptions.pop()
                self.area_labels.pop()

        self.counter += 1
        waypoint_id = self.counter
        self.positions.append((int(pixel_y), int(pixel_x)))
        self.ids.append(waypoint_id)
        self.descriptions.append(description)
        self.area_labels.append(area_label)
        return waypoint_id

    def get_waypoints(self) -> Tuple[List[Tuple[int, int]], List[int], List[str]]:
        return self.positions, self.ids, self.descriptions

    def get_area_labels(self) -> List[str]:
        return list(self.area_labels)

    def clear(self) -> None:
        self.reset()

    def count(self) -> int:
        return len(self.ids)
