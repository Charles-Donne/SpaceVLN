"""
语义建图器 - SemanticMapper
============================
职责：
1. 语义地图更新逻辑
2. Floor区域提取
3. 轨迹管理
4. 地图状态查询
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import torch

from skimage.morphology import disk, remove_small_objects
from skimage.morphology import binary_closing as _binary_closing_compat

from vlnce_baselines.config.core.constants import mapping_classes
from vlnce_baselines.mapping.space_area_manager import SpaceAreaManager
from vlnce_baselines.mapping.waypoint_manager import WaypointManager


class SemanticMapper:
    """语义建图器 - 管理地图构建和更新逻辑"""

    def __init__(self,
                 mapping_module,
                 map_shape: Tuple[int, int],
                 resolution: int = 5):
        """
        Args:
            mapping_module: Semantic_Mapping实例
            map_shape: 地图尺寸 (H, W)
            resolution: 地图分辨率 (cm/pixel)
        """
        self.mapping_module = mapping_module
        self.map_shape = map_shape
        self.resolution = resolution

        # 轨迹开关（轨迹实际存储在mapping_module中）
        self.enable_trajectory = True

        # Waypoint / space area 状态拆到独立模块，mapper 只负责协调 world-map 数据流。
        self.waypoint_manager = WaypointManager(resolution=resolution)
        self.space_area_manager = SpaceAreaManager(map_shape=map_shape, resolution=resolution)

        # 地图缓存
        self.floor = np.zeros(map_shape)
        self.full_map = None
        self.full_pose = None
        self._cached_space_area_crop_offset: Optional[Tuple[int, int]] = None
        self._cached_space_area_layer = np.zeros(map_shape, dtype=np.int32)
        self._cached_space_area_records: List[Dict[str, Any]] = []

    @property
    def waypoint_positions(self) -> List[Tuple[int, int]]:
        return self.waypoint_manager.positions

    @property
    def waypoint_ids(self) -> List[int]:
        return self.waypoint_manager.ids

    @property
    def waypoint_descriptions(self) -> List[str]:
        return self.waypoint_manager.descriptions

    @property
    def waypoint_area_labels(self) -> List[str]:
        return self.waypoint_manager.area_labels

    @property
    def waypoint_area_display_labels(self) -> List[str]:
        return [
            self.space_area_manager.get_display_label(label)
            for label in self.waypoint_manager.area_labels
        ]

    @property
    def space_area_records(self) -> List[Dict[str, Any]]:
        return self.space_area_manager.space_area_records

    @property
    def current_space_area_label(self) -> str:
        return self.space_area_manager.current_space_area_label

    @property
    def current_space_area_display_label(self) -> str:
        return self.space_area_manager.get_display_label(self.space_area_manager.current_space_area_label)

    @property
    def current_space_area_type(self) -> str:
        return self.space_area_manager.current_space_area_type

    def reset(self):
        """重置建图器状态"""
        self.waypoint_manager.reset()
        self.space_area_manager.reset()
        self.floor = np.zeros(self.map_shape)
        self.full_map = None
        self.full_pose = None
        self._invalidate_space_area_cache()
        self.mapping_module.reset()

    def init_map_and_pose(self, num_detected_classes: int):
        """初始化地图和位姿"""
        self.mapping_module.init_map_and_pose(num_detected_classes=num_detected_classes)

    def update_map(self,
                  batch_obs: torch.Tensor,
                  poses: torch.Tensor,
                  step: int,
                  detected_classes: List[str],
                  episode_id: int) -> Dict[str, Any]:
        """
        更新语义地图

        Args:
            batch_obs: 批量观察 [B, C, H, W]
            poses: 位姿变化 [B, 3] [Δx, Δy, Δθ]
            step: 当前步数
            detected_classes: 已检测类别列表
            episode_id: episode ID

        Returns:
            map_state: 地图状态字典
                - full_map: [C, H, W]
                - full_pose: [3] (x, y, orientation)
                - floor: [H, W]
                - visited_vis: [H, W]
        """
        self.mapping_module(batch_obs, poses, self.mapping_module.local_map, self.mapping_module.local_pose)

        full_map, full_pose, _ = self.mapping_module.update_map(
            step, detected_classes, episode_id
        )

        if torch.is_tensor(full_map):
            self.full_map = full_map[0].cpu().numpy()
        else:
            self.full_map = full_map[0]

        if torch.is_tensor(full_pose):
            self.full_pose = full_pose[0].cpu().numpy()
        else:
            self.full_pose = full_pose[0]

        self._invalidate_space_area_cache()
        self.floor = self._compute_floor_mask(self.full_map)
        self.mapping_module.clear_one_step_buffers()

        crop_offset = self.mapping_module.full_map_crop_offset
        global_traj = self.mapping_module.global_trajectory_points
        subtask_traj = self.mapping_module.subtask_trajectory_points
        space_area_layer, space_area_metadata = self._get_space_area_state(crop_offset)

        return {
            'full_map': self.full_map,
            'full_pose': self.full_pose,
            'floor': self.floor,
            'crop_offset': crop_offset,
            'global_trajectory_points': global_traj,
            'subtask_trajectory_points': subtask_traj,
            'waypoint_positions': self.waypoint_positions,
            'waypoint_ids': self.waypoint_ids,
            'waypoint_area_labels': list(self.waypoint_area_display_labels),
            'space_area_layer': space_area_layer,
            'space_area_records': space_area_metadata,
            'current_space_area_label': self.current_space_area_display_label,
            'current_space_area_type': self.current_space_area_type,
        }

    def _compute_floor_mask(self, full_map: np.ndarray) -> np.ndarray:
        """Build floor directly from explored minus obstacle for the current world map."""
        if full_map is None or full_map.shape[0] < 2:
            return np.zeros(self.map_shape, dtype=np.uint8)

        obstacle_mask = remove_small_objects(full_map[0, ...] > 0.5, min_size=16).astype(bool)
        explored_mask = remove_small_objects(full_map[1, ...] > 0.5, min_size=16).astype(bool)
        floor = np.logical_and(explored_mask, np.logical_not(obstacle_mask))
        floor = remove_small_objects(floor, min_size=100).astype(bool)
        floor = _binary_closing_compat(floor, disk(3))
        return floor.astype(np.uint8)

    def toggle_trajectory(self):
        """切换轨迹绘制开关"""
        self.enable_trajectory = not self.enable_trajectory
        return "启用" if self.enable_trajectory else "禁用"

    def clear_trajectory(self):
        """清空当前子任务轨迹。"""
        self.mapping_module.clear_trajectory()

    def clear_custom_landmarks(self):
        """清空地图中累计的自定义 landmark 通道。"""
        if hasattr(self.mapping_module, 'clear_landmark_channels'):
            self.mapping_module.clear_landmark_channels(n_mapping=len(mapping_classes))

    def add_waypoint(self, description: str = "") -> int:
        """添加 waypoint 到当前位置，并同步更新 space area。"""
        agent_x_m = self.full_pose[0]
        agent_y_m = self.full_pose[1]
        pixel_y = int(round(float(agent_y_m) * 100.0 / float(self.resolution)))
        pixel_x = int(round(float(agent_x_m) * 100.0 / float(self.resolution)))
        area_label = self.space_area_manager.update_from_waypoint(
            description=description,
            pixel_y=pixel_y,
            pixel_x=pixel_x,
            full_map=self.full_map,
            full_pose=self.full_pose,
            crop_offset=getattr(self.mapping_module, 'full_map_crop_offset', None),
        )
        self._invalidate_space_area_cache()
        return self.waypoint_manager.add_waypoint(
            pixel_y=pixel_y,
            pixel_x=pixel_x,
            description=description,
            area_label=area_label,
        )

    def get_waypoints(self) -> Tuple[List[Tuple[int, int]], List[int], List[str]]:
        """获取 waypoint 位置、ID 和描述。"""
        return self.waypoint_manager.get_waypoints()

    def get_waypoint_area_labels(self) -> List[str]:
        """获取每个 waypoint 对应的空间区域标签。"""
        return list(self.waypoint_area_display_labels)

    def clear_waypoints(self):
        """清空所有 waypoint。"""
        self.waypoint_manager.clear()
        self._invalidate_space_area_cache()

    def get_waypoint_count(self) -> int:
        """获取 waypoint 总数。"""
        return self.waypoint_manager.count()

    def get_map_state(self) -> Dict[str, Any]:
        """获取当前地图状态。"""
        crop_offset = self.mapping_module.full_map_crop_offset
        global_traj = self.mapping_module.global_trajectory_points
        subtask_traj = self.mapping_module.subtask_trajectory_points
        space_area_layer, space_area_records = self._get_space_area_state(crop_offset)
        return {
            'full_map': self.full_map,
            'full_pose': self.full_pose,
            'floor': self.floor,
            'waypoint_positions': self.waypoint_positions,
            'waypoint_ids': self.waypoint_ids,
            'waypoint_area_labels': list(self.waypoint_area_display_labels),
            'map_shape': self.map_shape,
            'resolution': self.resolution,
            'crop_offset': crop_offset,
            'global_trajectory_points': global_traj,
            'subtask_trajectory_points': subtask_traj,
            'space_area_layer': space_area_layer,
            'space_area_records': space_area_records,
            'current_space_area_label': self.current_space_area_display_label,
            'current_space_area_type': self.current_space_area_type,
        }

    def get_current_pose(self) -> Optional[Tuple[float, float, float]]:
        """获取当前位姿。"""
        if self.full_pose is None:
            return None
        return tuple(self.full_pose)

    def _invalidate_space_area_cache(self) -> None:
        self._cached_space_area_crop_offset = None
        self._cached_space_area_layer = np.zeros(self.map_shape, dtype=np.int32)
        self._cached_space_area_records = []

    def _get_space_area_state(self, crop_offset: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if self.full_map is None:
            self._invalidate_space_area_cache()
            return self._cached_space_area_layer, self._cached_space_area_records

        if (
            self._cached_space_area_crop_offset is not None and
            crop_offset == self._cached_space_area_crop_offset
        ):
            return self._cached_space_area_layer, list(self._cached_space_area_records)

        layer, records = self._build_space_area_layer(crop_offset)
        self._cached_space_area_crop_offset = crop_offset
        self._cached_space_area_layer = layer
        self._cached_space_area_records = list(records)
        return layer, list(records)

    def _build_space_area_layer(self, crop_offset: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        return self.space_area_manager.build_layer(
            full_map=self.full_map,
            full_pose=self.full_pose,
            crop_offset=crop_offset,
            waypoint_positions=self.waypoint_positions,
            waypoint_area_labels=self.waypoint_manager.area_labels,
        )


# ========== 便捷函数 ==========

def create_mapper(mapping_module,
                 map_shape: Tuple[int, int],
                 resolution: int = 5) -> SemanticMapper:
    """创建SemanticMapper实例"""
    return SemanticMapper(mapping_module, map_shape, resolution)
