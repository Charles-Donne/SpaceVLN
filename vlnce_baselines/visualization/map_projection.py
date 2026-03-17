import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class RotatedMapProjector:
    """Project world-map pixels into the rotated full-map/global/local display frames."""

    map_h: int
    map_w: int
    crop_offset: Tuple[int, int]
    agent_orientation_deg: float
    global_display_size: int = 480
    local_crop_size: int = 240
    local_display_size: int = 480

    def __post_init__(self) -> None:
        rotation_angle_deg = self.agent_orientation_deg - 90.0
        object.__setattr__(self, "_rotation_angle_deg", rotation_angle_deg)
        object.__setattr__(self, "_cos_theta", math.cos(math.radians(rotation_angle_deg)))
        object.__setattr__(self, "_sin_theta", math.sin(math.radians(rotation_angle_deg)))

    @property
    def crop_row_px(self) -> int:
        return int(self.crop_offset[0])

    @property
    def crop_col_px(self) -> int:
        return int(self.crop_offset[1])

    @property
    def local_crop_margin(self) -> float:
        return (self.global_display_size - self.local_crop_size) / 2.0

    def world_to_rotated_pixel(
        self,
        world_row_px: float,
        world_col_px: float,
    ) -> Optional[Tuple[float, float]]:
        rel_row = float(world_row_px) - self.crop_row_px
        rel_col = float(world_col_px) - self.crop_col_px
        if not (0.0 <= rel_row < self.map_h and 0.0 <= rel_col < self.map_w):
            return None

        norm_y = (rel_row / self.map_h) * 2.0 - 1.0
        norm_x = (rel_col / self.map_w) * 2.0 - 1.0

        rotated_norm_x = self._cos_theta * norm_x + self._sin_theta * norm_y
        rotated_norm_y = -self._sin_theta * norm_x + self._cos_theta * norm_y

        rotated_row = (rotated_norm_y + 1.0) * self.map_h / 2.0
        rotated_col = (rotated_norm_x + 1.0) * self.map_w / 2.0
        return rotated_row, rotated_col

    def rotated_to_global_display(
        self,
        rotated_row: float,
        rotated_col: float,
    ) -> Tuple[float, float]:
        display_x = rotated_col * self.global_display_size / self.map_w
        display_y = rotated_row * self.global_display_size / self.map_h
        display_y = self.global_display_size - 1 - display_y
        return display_x, display_y

    def world_to_global_display(
        self,
        world_row_px: float,
        world_col_px: float,
    ) -> Optional[Tuple[float, float]]:
        rotated = self.world_to_rotated_pixel(world_row_px, world_col_px)
        if rotated is None:
            return None
        return self.rotated_to_global_display(*rotated)

    def global_to_local_display(
        self,
        global_x: float,
        global_y: float,
    ) -> Optional[Tuple[float, float]]:
        crop_rel_x = global_x - self.local_crop_margin
        crop_rel_y = global_y - self.local_crop_margin
        if not (0.0 <= crop_rel_x < self.local_crop_size and 0.0 <= crop_rel_y < self.local_crop_size):
            return None

        display_x = crop_rel_x * self.local_display_size / self.local_crop_size
        display_y = crop_rel_y * self.local_display_size / self.local_crop_size
        return display_x, display_y

    def rotated_to_local_display(
        self,
        rotated_row: float,
        rotated_col: float,
    ) -> Optional[Tuple[float, float]]:
        global_display = self.rotated_to_global_display(rotated_row, rotated_col)
        return self.global_to_local_display(*global_display)

    def world_to_local_display(
        self,
        world_row_px: float,
        world_col_px: float,
    ) -> Optional[Tuple[float, float]]:
        global_display = self.world_to_global_display(world_row_px, world_col_px)
        if global_display is None:
            return None
        return self.global_to_local_display(*global_display)

    def world_points_to_global_display(
        self,
        points: Iterable[Tuple[float, float]],
    ) -> List[Tuple[int, int]]:
        display_points: List[Tuple[int, int]] = []
        for world_row_px, world_col_px in points:
            projected = self.world_to_global_display(world_row_px, world_col_px)
            if projected is None:
                continue
            display_x, display_y = projected
            display_points.append((int(display_x), int(display_y)))
        return display_points

    def world_points_to_local_display(
        self,
        points: Iterable[Tuple[float, float]],
    ) -> List[Tuple[int, int]]:
        display_points: List[Tuple[int, int]] = []
        for world_row_px, world_col_px in points:
            projected = self.world_to_local_display(world_row_px, world_col_px)
            if projected is None:
                continue
            display_x, display_y = projected
            display_points.append((int(display_x), int(display_y)))
        return display_points
