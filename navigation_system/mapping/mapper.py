"""
语义建图器 - SemanticMapper
============================
职责：
1. 语义地图更新逻辑
2. Floor区域提取
3. 轨迹管理
4. 地图状态查询
"""

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from skimage.morphology import binary_closing as _binary_closing_compat
from skimage.morphology import disk, remove_small_objects

from navigation_system.config.core.constants import mapping_classes
from navigation_system.config.core.params.thresholds import (
    FLOOR_SAME_Z_M,
    FLOOR_SWITCH_STABLE_STEPS,
    FLOOR_SWITCH_Z_M,
)
from navigation_system.mapping.space_area_manager import SpaceAreaManager
from navigation_system.mapping.space_types import normalize_space_type
from navigation_system.mapping.waypoint_manager import WaypointManager
from navigation_system.visualization.map_projection import RotatedMapProjector


class SemanticMapper:
    """语义建图器 - 管理地图构建和更新逻辑"""

    def __init__(self, mapping_module, map_shape: Tuple[int, int], resolution: int = 5):
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
        self.global_waypoint_manager = WaypointManager(resolution=resolution)
        self.space_area_manager = SpaceAreaManager(map_shape=map_shape, resolution=resolution)

        # 楼层感知配置：只增强底层拓扑，不改上层推理主流程。
        args = getattr(self.mapping_module, "args", None)
        self.enable_multi_floor_topology = bool(
            getattr(args, "ENABLE_MULTI_FLOOR_TOPOLOGY", True)
        )
        self.floor_z_tolerance_m = float(getattr(args, "FLOOR_Z_TOLERANCE_M", FLOOR_SAME_Z_M))
        self.floor_z_switch_threshold_m = float(
            getattr(args, "FLOOR_Z_SWITCH_THRESHOLD_M", FLOOR_SWITCH_Z_M)
        )
        self.floor_switch_stable_steps = max(
            1, int(getattr(args, "FLOOR_SWITCH_STABLE_STEPS", FLOOR_SWITCH_STABLE_STEPS))
        )
        self.stair_clear_radius_m = max(
            0.0, float(getattr(args, "STAIR_CLEAR_RADIUS_M", 0.45))
        )

        # 地图缓存
        self.floor = np.zeros(map_shape)
        self.full_map = None
        self.full_pose = None
        self._cached_space_area_crop_offset: Optional[Tuple[int, int]] = None
        self._cached_space_area_layer = np.zeros(map_shape, dtype=np.int32)
        self._cached_space_area_records: List[Dict[str, Any]] = []
        self._floor_switched_this_step = False

        self._reset_floor_topology_state()

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
    def global_waypoint_positions(self) -> List[Tuple[int, int]]:
        return self.global_waypoint_manager.positions

    @property
    def global_waypoint_ids(self) -> List[int]:
        return self.global_waypoint_manager.ids

    @property
    def global_waypoint_descriptions(self) -> List[str]:
        return self.global_waypoint_manager.descriptions

    @property
    def global_waypoint_area_labels(self) -> List[str]:
        return self.global_waypoint_manager.get_area_labels()

    @property
    def global_waypoint_floor_ids(self) -> List[int]:
        return self.global_waypoint_manager.get_floor_ids()

    @property
    def space_area_records(self) -> List[Dict[str, Any]]:
        return self.space_area_manager.space_area_records

    @property
    def current_space_area_label(self) -> str:
        return self.space_area_manager.current_space_area_label

    @property
    def current_space_area_display_label(self) -> str:
        return self.space_area_manager.get_display_label(
            self.space_area_manager.current_space_area_label
        )

    @property
    def current_space_area_type(self) -> str:
        return self.space_area_manager.current_space_area_type

    @property
    def current_floor_label(self) -> str:
        return self._format_floor_label(self.current_floor_id)

    def _reset_floor_topology_state(self) -> None:
        self.current_world_z: Optional[float] = None
        self.current_floor_id = 0
        self.multi_floor_active = False
        self.on_stairs_connector = False
        self.floor_z_anchors: Dict[int, float] = {}
        self.floor_contexts: Dict[int, Dict[str, Any]] = {}
        self.pending_floor_candidate_z: Optional[float] = None
        self.pending_floor_candidate_count = 0
        self.pending_floor_candidate_floor_id: Optional[int] = None
        self._active_stair_connector: Optional[Dict[str, Any]] = None
        self.stair_connectors: List[Dict[str, Any]] = []
        self._stair_clear_pixels_by_floor: Dict[int, Set[Tuple[int, int]]] = {}
        self._floor_switched_this_step = False

    def reset(self):
        """重置建图器状态"""
        self.waypoint_manager.reset()
        self.global_waypoint_manager.reset()
        self.space_area_manager.reset()
        self.floor = np.zeros(self.map_shape)
        self.full_map = None
        self.full_pose = None
        self._invalidate_space_area_cache()
        self.mapping_module.reset()
        self._reset_floor_topology_state()

    def init_map_and_pose(self, num_detected_classes: int):
        """初始化地图和位姿"""
        self.mapping_module.init_map_and_pose(num_detected_classes=num_detected_classes)
        self.waypoint_manager.reset()
        self.global_waypoint_manager.reset()
        self.space_area_manager.reset()
        self._invalidate_space_area_cache()
        self.full_pose = np.asarray([6.0, 6.0, 0.0], dtype=np.float32)
        self.current_floor_id = 0
        self.floor_contexts = {}
        self.floor_z_anchors = {}
        self.multi_floor_active = False
        self.on_stairs_connector = False
        self.pending_floor_candidate_z = None
        self.pending_floor_candidate_count = 0
        self.pending_floor_candidate_floor_id = None

    def update_map(
        self,
        batch_obs: torch.Tensor,
        poses: torch.Tensor,
        step: int,
        detected_classes: List[str],
        episode_id: int,
        observations: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        更新语义地图

        Args:
            batch_obs: 批量观察 [B, C, H, W]
            poses: 位姿变化 [B, 3] [Δx, Δy, Δθ]
            step: 当前步数
            detected_classes: 已检测类别列表
            episode_id: episode ID
            observations: 原始obs（读取position传感器里的高度）
        """
        world_z = self._extract_world_z(observations)
        predicted_pose = self._predict_pose_after_delta(poses)
        self._update_floor_topology_before_step(
            world_z=world_z,
            predicted_pose=predicted_pose,
            num_detected_classes=len(detected_classes),
        )

        poses_for_mapping = poses
        if self._floor_switched_this_step:
            if torch.is_tensor(poses):
                poses_for_mapping = torch.zeros_like(poses)
            elif poses is not None:
                poses_for_mapping = np.zeros_like(poses)

        self.mapping_module(
            batch_obs,
            poses_for_mapping,
            self.mapping_module.local_map,
            self.mapping_module.local_pose,
        )

        full_map, full_pose, _ = self.mapping_module.update_map(
            step, detected_classes, episode_id
        )

        if torch.is_tensor(full_map):
            current_full_map = full_map[0].cpu().numpy()
        else:
            current_full_map = full_map[0]

        if torch.is_tensor(full_pose):
            current_full_pose = full_pose[0].cpu().numpy()
        else:
            current_full_pose = full_pose[0]

        self.full_pose = np.asarray(current_full_pose, dtype=np.float32)
        if world_z is not None:
            self.current_world_z = float(world_z)

        if (
            self.on_stairs_connector
            or self._active_stair_connector is not None
            or self._is_current_area_stairs()
        ):
            self._register_stair_step_pixels(
                floor_id=self.current_floor_id,
                pose=self.full_pose,
            )

        crop_offset = tuple(getattr(self.mapping_module, "full_map_crop_offset", (0, 0)) or (0, 0))
        self.full_map = self._apply_stair_clearance(
            full_map=np.asarray(current_full_map),
            full_pose=self.full_pose,
            crop_offset=crop_offset,
            floor_id=self.current_floor_id,
        )

        self._maybe_finalize_active_stair_connector()
        self._invalidate_space_area_cache()
        self.floor = self._compute_floor_mask(self.full_map)
        self.mapping_module.clear_one_step_buffers()

        global_traj = list(self.mapping_module.global_trajectory_points)
        subtask_traj = list(self.mapping_module.subtask_trajectory_points)
        space_area_layer, space_area_metadata = self._get_space_area_state(crop_offset)

        self.multi_floor_active = len(self.floor_z_anchors) > 1
        return self._compose_map_state(
            crop_offset=crop_offset,
            global_traj=global_traj,
            subtask_traj=subtask_traj,
            space_area_layer=space_area_layer,
            space_area_metadata=space_area_metadata,
        )

    def _compose_map_state(
        self,
        crop_offset: Optional[Tuple[int, int]],
        global_traj: List[Tuple[int, int]],
        subtask_traj: List[Tuple[int, int]],
        space_area_layer: np.ndarray,
        space_area_metadata: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "full_map": self.full_map,
            "full_pose": self.full_pose,
            "floor": self.floor,
            "map_shape": self.map_shape,
            "resolution": self.resolution,
            "crop_offset": crop_offset,
            "global_trajectory_points": global_traj,
            "subtask_trajectory_points": subtask_traj,
            "waypoint_positions": self.waypoint_positions,
            "waypoint_ids": self.waypoint_ids,
            "waypoint_floor_ids": self.waypoint_manager.get_floor_ids(),
            "waypoint_area_labels": list(self.waypoint_area_display_labels),
            "waypoint_initial_index": self.waypoint_manager.initial_waypoint_index,
            "space_area_layer": space_area_layer,
            "space_area_records": space_area_metadata,
            "current_space_area_label": self.current_space_area_display_label,
            "current_space_area_type": self.current_space_area_type,
            "current_world_z": self.current_world_z,
            "current_floor_id": self.current_floor_id,
            "current_floor_label": self.current_floor_label,
            "multi_floor_active": self.multi_floor_active,
            "on_stairs_connector": self.on_stairs_connector,
            "stair_connectors": self._serialize_stair_connectors(),
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
        if hasattr(self.mapping_module, "clear_landmark_channels"):
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
            crop_offset=getattr(self.mapping_module, "full_map_crop_offset", None),
        )
        self._invalidate_space_area_cache()
        display_area_label = self.space_area_manager.get_display_label(area_label)
        waypoint_id = self.global_waypoint_manager.add_waypoint(
            pixel_y=pixel_y,
            pixel_x=pixel_x,
            description=description,
            area_label=display_area_label,
            floor_id=self.current_floor_id,
        )
        self.waypoint_manager.add_waypoint(
            pixel_y=pixel_y,
            pixel_x=pixel_x,
            description=description,
            area_label=area_label,
            floor_id=self.current_floor_id,
            waypoint_id=waypoint_id,
        )
        return waypoint_id

    def get_waypoints(self) -> Tuple[List[Tuple[int, int]], List[int], List[str]]:
        """获取 waypoint 位置、ID 和描述。"""
        return self.waypoint_manager.get_waypoints()

    def get_global_waypoints(self) -> Tuple[List[Tuple[int, int]], List[int], List[str]]:
        """获取跨楼层的全局 waypoint 历史。"""
        return self.global_waypoint_manager.get_waypoints()

    def get_waypoint_area_labels(self) -> List[str]:
        """获取每个 waypoint 对应的空间区域标签。"""
        return list(self.waypoint_area_display_labels)

    def get_global_waypoint_area_labels(self) -> List[str]:
        """获取跨楼层 waypoint 历史的区域标签。"""
        return list(self.global_waypoint_area_labels)

    def get_waypoint_floor_ids(self) -> List[int]:
        """获取当前楼层 waypoint 的 floor_id 列表。"""
        return self.waypoint_manager.get_floor_ids()

    def get_global_waypoint_floor_ids(self) -> List[int]:
        """获取跨楼层 waypoint 历史的 floor_id 列表。"""
        return self.global_waypoint_manager.get_floor_ids()

    def clear_waypoints(self):
        """清空当前楼层的所有 waypoint。"""
        self.waypoint_manager.clear()
        self._invalidate_space_area_cache()

    def get_waypoint_count(self) -> int:
        """获取 waypoint 总数。"""
        return self.waypoint_manager.count()

    def get_map_state(self) -> Dict[str, Any]:
        """获取当前地图状态。"""
        crop_offset = tuple(getattr(self.mapping_module, "full_map_crop_offset", (0, 0)) or (0, 0))
        global_traj = list(getattr(self.mapping_module, "global_trajectory_points", []) or [])
        subtask_traj = list(getattr(self.mapping_module, "subtask_trajectory_points", []) or [])
        space_area_layer, space_area_records = self._get_space_area_state(crop_offset)
        return self._compose_map_state(
            crop_offset=crop_offset,
            global_traj=global_traj,
            subtask_traj=subtask_traj,
            space_area_layer=space_area_layer,
            space_area_metadata=space_area_records,
        )

    def get_current_pose(self) -> Optional[Tuple[float, float, float]]:
        """获取当前位姿。"""
        if self.full_pose is None:
            return None
        return tuple(float(value) for value in self.full_pose[:3])

    def _invalidate_space_area_cache(self) -> None:
        self._cached_space_area_crop_offset = None
        self._cached_space_area_layer = np.zeros(self.map_shape, dtype=np.int32)
        self._cached_space_area_records = []

    def _get_space_area_state(
        self, crop_offset: Optional[Tuple[int, int]]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if self.full_map is None:
            self._invalidate_space_area_cache()
            return self._cached_space_area_layer, self._cached_space_area_records

        if (
            self._cached_space_area_crop_offset is not None
            and crop_offset == self._cached_space_area_crop_offset
        ):
            return self._cached_space_area_layer, list(self._cached_space_area_records)

        layer, records = self._build_space_area_layer(crop_offset)
        for record in records:
            record["floor_id"] = int(self.current_floor_id)
        self._cached_space_area_crop_offset = crop_offset
        self._cached_space_area_layer = layer
        self._cached_space_area_records = list(records)
        return layer, list(records)

    def _build_space_area_layer(
        self, crop_offset: Optional[Tuple[int, int]]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        return self.space_area_manager.build_layer(
            full_map=self.full_map,
            full_pose=self.full_pose,
            crop_offset=crop_offset,
            waypoint_positions=self.waypoint_positions,
            waypoint_area_labels=self.waypoint_manager.area_labels,
        )

    def _extract_world_z(
        self, observations: Optional[Sequence[Dict[str, Any]]]
    ) -> Optional[float]:
        if not observations:
            return None
        obs0 = observations[0] if isinstance(observations, (list, tuple)) else observations
        if not isinstance(obs0, dict):
            return None
        position = obs0.get("position")
        if position is None or len(position) < 2:
            return None
        try:
            z_value = float(position[1])
        except (TypeError, ValueError):
            return None
        return z_value if np.isfinite(z_value) else None

    def _predict_pose_after_delta(self, poses: Optional[torch.Tensor]) -> Tuple[float, float, float]:
        if self.full_pose is None:
            base_pose = np.asarray([6.0, 6.0, 0.0], dtype=np.float32)
        else:
            base_pose = np.asarray(self.full_pose[:3], dtype=np.float32)
        if poses is None or len(poses) == 0:
            return tuple(float(value) for value in base_pose)

        pose_delta = poses[0]
        if torch.is_tensor(pose_delta):
            rel_pose = pose_delta.detach().cpu().numpy().astype(np.float32)
        else:
            rel_pose = np.asarray(pose_delta, dtype=np.float32)
        if rel_pose.shape[0] < 3:
            return tuple(float(value) for value in base_pose)

        pose_x, pose_y, pose_o = [float(value) for value in base_pose[:3]]
        delta_x, delta_y, delta_o = [float(value) for value in rel_pose[:3]]
        orientation_rad = math.radians(pose_o)
        pose_y += delta_x * math.sin(orientation_rad) + delta_y * math.cos(orientation_rad)
        pose_x += delta_x * math.cos(orientation_rad) - delta_y * math.sin(orientation_rad)
        pose_o += delta_o * 57.29577951308232
        pose_o = (pose_o + 180.0) % 360.0 - 180.0
        return float(pose_x), float(pose_y), float(pose_o)

    def _format_floor_label(self, floor_id: int) -> str:
        return f"F{int(floor_id) + 1}"

    def _find_matching_floor_id(self, world_z: float) -> Optional[int]:
        best_floor_id = None
        best_distance = None
        for floor_id, anchor_z in self.floor_z_anchors.items():
            distance = abs(float(world_z) - float(anchor_z))
            if distance > self.floor_z_tolerance_m:
                continue
            if best_distance is None or distance < best_distance:
                best_floor_id = int(floor_id)
                best_distance = float(distance)
        return best_floor_id

    def _register_floor_anchor(self, world_z: float, floor_id: Optional[int] = None) -> int:
        if floor_id is None:
            floor_id = max(self.floor_z_anchors.keys(), default=-1) + 1
        self.floor_z_anchors[int(floor_id)] = float(world_z)
        self.multi_floor_active = len(self.floor_z_anchors) > 1
        return int(floor_id)

    def _clear_pending_floor_candidate(self) -> None:
        self.pending_floor_candidate_z = None
        self.pending_floor_candidate_count = 0
        self.pending_floor_candidate_floor_id = None

    def _update_pending_floor_candidate(
        self,
        candidate_z: float,
        candidate_floor_id: Optional[int],
    ) -> None:
        candidate_z = float(candidate_z)
        if (
            self.pending_floor_candidate_z is not None
            and abs(candidate_z - float(self.pending_floor_candidate_z)) <= self.floor_z_tolerance_m
            and self.pending_floor_candidate_floor_id == candidate_floor_id
        ):
            self.pending_floor_candidate_count += 1
            self.pending_floor_candidate_z = (
                float(self.pending_floor_candidate_z) * float(self.pending_floor_candidate_count - 1)
                + candidate_z
            ) / float(self.pending_floor_candidate_count)
            return
        self.pending_floor_candidate_z = candidate_z
        self.pending_floor_candidate_count = 1
        self.pending_floor_candidate_floor_id = candidate_floor_id

    def _capture_floor_context(self) -> Dict[str, Any]:
        return {
            "mapping_state": self.mapping_module.export_state(),
            "waypoint_state": self.waypoint_manager.export_state(),
            "space_area_state": self.space_area_manager.export_state(),
            "full_map": None if self.full_map is None else np.array(self.full_map, copy=True),
            "floor": np.array(self.floor, copy=True),
            "full_pose": None if self.full_pose is None else np.array(self.full_pose, copy=True),
        }

    def _save_current_floor_context(self) -> None:
        self.floor_contexts[int(self.current_floor_id)] = self._capture_floor_context()

    def _load_floor_context(
        self,
        floor_id: int,
        pose_override: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        context = self.floor_contexts.get(int(floor_id))
        if context is None:
            return
        self.mapping_module.import_state(context.get("mapping_state"))
        self.waypoint_manager.import_state(context.get("waypoint_state"))
        self.space_area_manager.import_state(context.get("space_area_state"))
        if pose_override is not None:
            self.mapping_module.recenter_to_world_pose(pose_override, clear_one_step=True)
            self.full_map = None
            self.floor = np.zeros(self.map_shape, dtype=np.uint8)
            self.full_pose = np.asarray(pose_override, dtype=np.float32)
        else:
            self.full_map = (
                None
                if context.get("full_map") is None
                else np.array(context.get("full_map"), copy=True)
            )
            self.floor = np.array(context.get("floor", np.zeros(self.map_shape)), copy=True)
            self.full_pose = (
                None
                if context.get("full_pose") is None
                else np.array(context.get("full_pose"), copy=True)
            )
        self._invalidate_space_area_cache()

    def _prepare_new_floor_context(
        self,
        floor_id: int,
        initial_pose: Tuple[float, float, float],
        num_detected_classes: int,
    ) -> None:
        self.mapping_module.reset()
        self.mapping_module.init_map_and_pose(
            num_detected_classes=num_detected_classes,
            initial_pose=initial_pose,
        )
        self.waypoint_manager.reset()
        if int(floor_id) != 0:
            self.waypoint_manager.initial_waypoint_index = None
        self.space_area_manager.reset()
        self.full_map = None
        self.floor = np.zeros(self.map_shape, dtype=np.uint8)
        self.full_pose = np.asarray(initial_pose, dtype=np.float32)
        self._invalidate_space_area_cache()

    def _start_stair_connector_if_needed(
        self, predicted_pose: Tuple[float, float, float]
    ) -> None:
        if self._active_stair_connector is not None:
            return
        self._active_stair_connector = {
            "id": len(self.stair_connectors) + 1,
            "start_floor_id": int(self.current_floor_id),
            "start_world_pose": tuple(float(value) for value in predicted_pose[:3]),
            "world_pixels_by_floor": {},
            "target_floor_id": None,
        }
        self._register_stair_step_pixels(
            floor_id=self.current_floor_id,
            pose=np.asarray(predicted_pose, dtype=np.float32),
        )

    def _pose_to_world_pixel(self, pose: Sequence[float]) -> Tuple[int, int]:
        pose_x_m, pose_y_m = float(pose[0]), float(pose[1])
        pixel_y = int(round(pose_y_m * 100.0 / float(self.resolution)))
        pixel_x = int(round(pose_x_m * 100.0 / float(self.resolution)))
        return pixel_y, pixel_x

    def _build_disk_pixels(
        self,
        center_pixel: Tuple[int, int],
        radius_m: float,
    ) -> Set[Tuple[int, int]]:
        radius_px = max(1, int(round(float(radius_m) * 100.0 / float(self.resolution))))
        center_y, center_x = int(center_pixel[0]), int(center_pixel[1])
        pixels: Set[Tuple[int, int]] = set()
        for delta_y in range(-radius_px, radius_px + 1):
            for delta_x in range(-radius_px, radius_px + 1):
                if delta_y * delta_y + delta_x * delta_x > radius_px * radius_px:
                    continue
                pixels.add((center_y + delta_y, center_x + delta_x))
        return pixels

    def _register_stair_step_pixels(self, floor_id: int, pose: Sequence[float]) -> None:
        if self.stair_clear_radius_m <= 0.0:
            return
        floor_key = int(floor_id)
        pixels = self._build_disk_pixels(
            center_pixel=self._pose_to_world_pixel(pose),
            radius_m=self.stair_clear_radius_m,
        )
        if not pixels:
            return
        self._stair_clear_pixels_by_floor.setdefault(floor_key, set()).update(pixels)
        if self._active_stair_connector is not None:
            world_pixels = self._active_stair_connector.setdefault("world_pixels_by_floor", {})
            world_pixels.setdefault(floor_key, set()).update(pixels)

    def _is_current_area_stairs(self) -> bool:
        space_type = normalize_space_type(getattr(self.space_area_manager, "current_space_area_type", ""))
        if space_type == "stairs":
            return True
        space_label = normalize_space_type(getattr(self.space_area_manager, "current_space_area_label", ""))
        return space_label == "stairs"

    def _serialize_stair_connectors(self) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        for connector in self.stair_connectors:
            serialized.append({
                "id": int(connector.get("id", 0) or 0),
                "label": str(connector.get("label", "stairs")),
                "from_floor_id": int(connector.get("from_floor_id", 0) or 0),
                "to_floor_id": int(connector.get("to_floor_id", 0) or 0),
                "lower_landing": dict(connector.get("lower_landing", {}) or {}),
                "upper_landing": dict(connector.get("upper_landing", {}) or {}),
            })
        return serialized

    def _switch_to_floor(
        self,
        target_floor_id: int,
        target_anchor_z: float,
        predicted_pose: Tuple[float, float, float],
        num_detected_classes: int,
    ) -> None:
        target_floor_id = int(target_floor_id)
        previous_floor_id = int(self.current_floor_id)
        if target_floor_id == previous_floor_id:
            self.floor_z_anchors[target_floor_id] = float(target_anchor_z)
            self._clear_pending_floor_candidate()
            self.on_stairs_connector = False
            return

        self._save_current_floor_context()
        self.floor_z_anchors[target_floor_id] = float(target_anchor_z)
        if target_floor_id in self.floor_contexts:
            self._load_floor_context(target_floor_id, pose_override=predicted_pose)
        else:
            self._prepare_new_floor_context(
                floor_id=target_floor_id,
                initial_pose=predicted_pose,
                num_detected_classes=num_detected_classes,
            )

        self.current_floor_id = target_floor_id
        self.current_world_z = float(target_anchor_z)
        self.multi_floor_active = len(self.floor_z_anchors) > 1
        self.on_stairs_connector = False
        self._floor_switched_this_step = True
        self._clear_pending_floor_candidate()
        if self._active_stair_connector is not None:
            self._active_stair_connector["target_floor_id"] = target_floor_id
            self._active_stair_connector["target_world_pose"] = tuple(
                float(value) for value in predicted_pose[:3]
            )

    def _maybe_finalize_active_stair_connector(self) -> None:
        if self._active_stair_connector is None:
            return
        target_floor_id = self._active_stair_connector.get("target_floor_id")
        if target_floor_id is None:
            return

        start_floor_id = int(self._active_stair_connector.get("start_floor_id", 0) or 0)
        end_floor_id = int(target_floor_id)
        start_pose = tuple(self._active_stair_connector.get("start_world_pose", (0.0, 0.0, 0.0)))
        end_pose = tuple(float(value) for value in (self.full_pose[:3] if self.full_pose is not None else (0.0, 0.0, 0.0)))
        start_anchor_z = float(self.floor_z_anchors.get(start_floor_id, 0.0))
        end_anchor_z = float(self.floor_z_anchors.get(end_floor_id, 0.0))
        if start_anchor_z <= end_anchor_z:
            lower_floor_id, lower_pose = start_floor_id, start_pose
            upper_floor_id, upper_pose = end_floor_id, end_pose
        else:
            lower_floor_id, lower_pose = end_floor_id, end_pose
            upper_floor_id, upper_pose = start_floor_id, start_pose

        self.stair_connectors.append({
            "id": int(self._active_stair_connector.get("id", len(self.stair_connectors) + 1) or len(self.stair_connectors) + 1),
            "label": f"Stairs#{len(self.stair_connectors) + 1}",
            "from_floor_id": start_floor_id,
            "to_floor_id": end_floor_id,
            "lower_landing": {
                "floor_id": int(lower_floor_id),
                "floor_label": self._format_floor_label(lower_floor_id),
                "pose": [float(value) for value in lower_pose[:3]],
            },
            "upper_landing": {
                "floor_id": int(upper_floor_id),
                "floor_label": self._format_floor_label(upper_floor_id),
                "pose": [float(value) for value in upper_pose[:3]],
            },
            "world_pixels_by_floor": {
                int(floor_id): [
                    (int(pixel_y), int(pixel_x))
                    for pixel_y, pixel_x in sorted(list(pixels))
                ]
                for floor_id, pixels in dict(
                    self._active_stair_connector.get("world_pixels_by_floor", {}) or {}
                ).items()
            },
        })
        self._active_stair_connector = None

    def _update_floor_topology_before_step(
        self,
        world_z: Optional[float],
        predicted_pose: Tuple[float, float, float],
        num_detected_classes: int,
    ) -> None:
        self._floor_switched_this_step = False
        if not self.enable_multi_floor_topology or world_z is None:
            return

        self.current_world_z = float(world_z)
        if int(self.current_floor_id) not in self.floor_z_anchors:
            self._register_floor_anchor(world_z, floor_id=self.current_floor_id)

        current_anchor_z = float(self.floor_z_anchors.get(self.current_floor_id, world_z))
        matched_floor_id = self._find_matching_floor_id(world_z)
        current_distance = abs(float(world_z) - current_anchor_z)

        if matched_floor_id == self.current_floor_id and current_distance <= self.floor_z_tolerance_m:
            if self._active_stair_connector is not None and self.pending_floor_candidate_count == 0:
                self._active_stair_connector = None
            self.on_stairs_connector = False
            self._clear_pending_floor_candidate()
            return

        if matched_floor_id is not None and matched_floor_id != self.current_floor_id:
            self.on_stairs_connector = True
            self._start_stair_connector_if_needed(predicted_pose)
            self._update_pending_floor_candidate(
                candidate_z=float(self.floor_z_anchors[matched_floor_id]),
                candidate_floor_id=matched_floor_id,
            )
            if self.pending_floor_candidate_count >= self.floor_switch_stable_steps:
                self._switch_to_floor(
                    target_floor_id=matched_floor_id,
                    target_anchor_z=float(self.floor_z_anchors[matched_floor_id]),
                    predicted_pose=predicted_pose,
                    num_detected_classes=num_detected_classes,
                )
            return

        if current_distance > self.floor_z_tolerance_m:
            self.on_stairs_connector = True
            self._start_stair_connector_if_needed(predicted_pose)
        else:
            self.on_stairs_connector = False

        if current_distance > self.floor_z_switch_threshold_m:
            self._update_pending_floor_candidate(
                candidate_z=float(world_z),
                candidate_floor_id=None,
            )
            if self.pending_floor_candidate_count >= self.floor_switch_stable_steps:
                target_floor_id = self._find_matching_floor_id(
                    float(self.pending_floor_candidate_z)
                )
                if target_floor_id is None:
                    target_floor_id = self._register_floor_anchor(
                        float(self.pending_floor_candidate_z)
                    )
                self._switch_to_floor(
                    target_floor_id=target_floor_id,
                    target_anchor_z=float(self.pending_floor_candidate_z),
                    predicted_pose=predicted_pose,
                    num_detected_classes=num_detected_classes,
                )
            return

        self._clear_pending_floor_candidate()

    def _apply_stair_clearance(
        self,
        full_map: np.ndarray,
        full_pose: Optional[Sequence[float]],
        crop_offset: Optional[Tuple[int, int]],
        floor_id: int,
    ) -> np.ndarray:
        if full_map is None:
            return full_map
        corrected_map = np.array(full_map, copy=True)
        stair_pixels = set(self._stair_clear_pixels_by_floor.get(int(floor_id), set()) or set())
        if self._active_stair_connector is not None:
            stair_pixels.update(
                set(
                    self._active_stair_connector.get("world_pixels_by_floor", {}).get(
                        int(floor_id),
                        set(),
                    )
                    or set()
                )
            )
        if not stair_pixels or full_pose is None or crop_offset is None:
            return corrected_map

        projector = RotatedMapProjector(
            map_h=corrected_map.shape[1],
            map_w=corrected_map.shape[2],
            crop_offset=crop_offset,
            agent_orientation_deg=float(full_pose[2]),
        )
        for pixel_y, pixel_x in stair_pixels:
            rotated = projector.world_to_rotated_pixel(pixel_y, pixel_x)
            if rotated is None:
                continue
            row = int(round(rotated[0]))
            col = int(round(rotated[1]))
            if not (0 <= row < corrected_map.shape[1] and 0 <= col < corrected_map.shape[2]):
                continue
            corrected_map[0, row, col] = 0.0
            corrected_map[1, row, col] = max(float(corrected_map[1, row, col]), 1.0)
        return corrected_map


# ========== 便捷函数 ==========


def create_mapper(mapping_module, map_shape: Tuple[int, int], resolution: int = 5) -> SemanticMapper:
    """创建SemanticMapper实例"""
    return SemanticMapper(mapping_module, map_shape, resolution)
