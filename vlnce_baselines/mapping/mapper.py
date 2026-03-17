"""
语义建图器 - SemanticMapper
============================
职责：
1. 语义地图更新逻辑
2. Floor区域提取
3. 轨迹管理
4. 地图状态查询
"""

from collections import deque
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import torch

from skimage.morphology import disk, remove_small_objects
from skimage.morphology import binary_closing as _binary_closing_compat

from vlnce_baselines.config_system.constants import (
    mapping_classes,
    navigable_classes,
    map_channels,
)
from vlnce_baselines.visualization.map_projection import RotatedMapProjector


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
        
        # Waypoint管理（与轨迹系统集成）
        self.waypoint_positions = []  # [(map_x, map_y), ...] waypoint的地图坐标
        self.waypoint_ids = []  # [1, 2, 3, ...] 对应的waypoint ID
        self.waypoint_area_labels = []  # ["Bedroom1", "Hallway1", ...]
        # Waypoint管理（新机制：waypoint直接存储在世界地图Channel 2中，作为整数值）
        self.waypoint_counter = 0  # waypoint计数器
        self.waypoint_descriptions = []  # ["desc1", "desc2", ...]

        # 房间区域历史（持久世界坐标）
        self.room_area_records: List[Dict[str, Any]] = []
        self.room_area_counter = 0
        self.current_room_area_label = ""
        self.current_room_area_type = ""
        
        # 地图缓存
        self.floor = np.zeros(map_shape)
        self.full_map = None
        self.full_pose = None
    
    def reset(self):
        """重置建图器状态"""
        self.waypoint_positions.clear()
        self.waypoint_ids.clear()
        self.waypoint_area_labels.clear()
        self.waypoint_descriptions.clear()
        self.waypoint_counter = 0  # 重置计数器
        self.room_area_records.clear()
        self.room_area_counter = 0
        self.current_room_area_label = ""
        self.current_room_area_type = ""
        self.floor = np.zeros(self.map_shape)
        self.full_map = None
        self.full_pose = None
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
        # 1. 调用底层mapping_module更新
        self.mapping_module(batch_obs, poses, self.mapping_module.local_map, self.mapping_module.local_pose)
        
        # 2. 获取更新后的地图
        full_map, full_pose, one_step_full_map = self.mapping_module.update_map(
            step, detected_classes, episode_id
        )
        
        # 转换为numpy（如果是tensor）
        if torch.is_tensor(full_map):
            self.full_map = full_map[0].cpu().numpy()  # [C, H, W]
        else:
            self.full_map = full_map[0]
        
        if torch.is_tensor(full_pose):
            self.full_pose = full_pose[0].cpu().numpy()  # [3]
        else:
            self.full_pose = full_pose[0]
        
        # 3. 提取floor区域
        self.floor = self.extract_floor(self.full_map, detected_classes)
        
        # 4. 清空单步地图缓存（包括 one_step_tiles，避免 recentering 时读回旧残留）
        self.mapping_module.clear_one_step_buffers()
        
        # 5. 获取世界坐标（不旋转！full_map已经旋转过，坐标在visualizer中转换）
        crop_off = self.mapping_module.full_map_crop_offset
        global_traj = self.mapping_module.global_trajectory_points  # 全局轨迹（用于global map）
        subtask_traj = self.mapping_module.subtask_trajectory_points  # 子任务轨迹（用于local map）
        wp_pos_world = self.waypoint_positions  # 保持世界坐标
        room_area_layer, room_area_metadata = self._build_room_area_layer()
        
        # print(f"[Mapper.update_map] 返回轨迹: 全局={len(global_traj)}, 子任务={len(subtask_traj)}")
        
        # 6. 返回完整的地图状态（包含世界坐标，visualizer会根据需要转换）
        return {
            'full_map': self.full_map,
            'full_pose': self.full_pose,
            'floor': self.floor,
            'crop_offset': crop_off,
            'global_trajectory_points': global_traj,  # 全局轨迹（global map用）
            'subtask_trajectory_points': subtask_traj,  # 子任务轨迹（local map用）
            'waypoint_positions': wp_pos_world,   # 世界坐标
            'waypoint_ids': self.waypoint_ids,
            'waypoint_area_labels': list(self.waypoint_area_labels),
            'room_area_layer': room_area_layer,
            'room_area_records': room_area_metadata,
            'current_room_area_label': self.current_room_area_label,
            'current_room_area_type': self.current_room_area_type,
        }
    
    def extract_floor(self, 
                     full_map: np.ndarray,
                     detected_classes: List[str]) -> np.ndarray:
        """
        从full_map提取floor区域（已弃用：floor现在是语义类别）
        
        注意：按照ZS_Evaluator的方式，floor现在是full_map[3+]中的第一个语义类别，
        不再需要通过形态学方法提取。这个方法保留仅用于向后兼容。
        
        Args:
            full_map: [C, H, W] 全局地图
            detected_classes: 已检测类别列表（全局累计的类别，可能多于当前步的检测）
        
        Returns:
            floor: [H, W] floor地图（现在主要用于向后兼容，实际floor在semantic layer）
        """
        full_map_bool = full_map.astype(bool)

        # 按通道单独做小区域过滤，避免把 channel 维误当成空间维导致 floor/obstacle 串扰
        obstacles = remove_small_objects(full_map_bool[0, ...], min_size=16).astype(bool)
        explored_area = remove_small_objects(full_map_bool[1, ...], min_size=16).astype(bool)

        # 提取语义层（从第 map_channels 个通道开始）
        semantic_layers = full_map_bool[map_channels:, ...]
        if semantic_layers.shape[0] > 0:
            semantic_layers = np.stack([
                remove_small_objects(layer, min_size=16).astype(bool)
                for layer in semantic_layers
            ], axis=0)
        num_semantic_channels = semantic_layers.shape[0]
        
        # 如果没有语义通道，直接返回基于explored的简单floor
        if num_semantic_channels == 0:
            # 简单处理：explored且非障碍物的区域
            floor = np.logical_and(explored_area, np.logical_not(obstacles))
            return floor.astype(np.uint8)
        
        # full_map[3:3+len(mapping_classes)] 始终对应固定的 mapping_classes 顺序。
        # 不能用运行时 detected_classes 顺序解释这些通道，否则 floor/door/bed 等类别会错位。
        mapping_semantic_layers = semantic_layers[:len(mapping_classes), ...]

        # 区分可导航和不可导航的类别（只基于固定 mapping_classes，不把 landmark 通道算进 floor）
        navigable_index = []
        not_navigable_index = []

        for i, cls_name in enumerate(mapping_classes[:mapping_semantic_layers.shape[0]]):
            if cls_name in navigable_classes:
                navigable_index.append(i)
            else:
                not_navigable_index.append(i)
        
        # 不可导航物体
        if len(not_navigable_index) > 0:
            objects = np.sum(mapping_semantic_layers[not_navigable_index], axis=0).astype(bool)
        else:
            objects = np.zeros_like(obstacles)
        
        # 可导航区域（如floor, stairs等）
        if len(navigable_index) > 0:
            navigable = np.logical_or.reduce(mapping_semantic_layers[navigable_index])
            navigable = np.logical_and(navigable, np.logical_not(objects))
        else:
            navigable = np.zeros_like(obstacles)
        
        # 计算自由空间
        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        floor = explored_area * free_mask
        
        # 过滤小floor区域并形态学闭运算
        floor = remove_small_objects(floor, min_size=100).astype(bool)
        floor = _binary_closing_compat(floor, disk(3))
        
        # 静默返回，不输出调试信息
        return floor.astype(np.uint8)
    
    # ===== 轨迹管理 =====
    # 轨迹现在直接存储在mapping_module.trajectory_points列表中（世界坐标）
    # 渲染时在visualizer中转换为旋转后的坐标（与full_map一致）
    
    def toggle_trajectory(self):
        """切换轨迹绘制开关"""
        self.enable_trajectory = not self.enable_trajectory
        status = "启用" if self.enable_trajectory else "禁用"
        return status
    
    def clear_trajectory(self):
        """
        清空轨迹（Agent通道中值为0.5的部分）
        
        使用场景：
        - 子任务完成时：清空上一子任务的轨迹，开始记录新子任务轨迹
        - 每个子任务都有独立的轨迹显示，不会累积
        """
        self.mapping_module.clear_trajectory()  # 清空Agent通道中的轨迹

    def clear_custom_landmarks(self):
        """清空地图中累计的自定义 landmark 通道。"""
        if hasattr(self.mapping_module, 'clear_landmark_channels'):
            self.mapping_module.clear_landmark_channels(n_mapping=len(mapping_classes))
    
    # ========== Waypoint管理方法 ==========
    
    def add_waypoint(self, description: str = "") -> int:
        """
        添加waypoint到当前位置
        
        机制：如果当前位置2m之内有上一个waypoint，则仅替换上一个waypoint
        
        Args:
            description: waypoint描述（可选，用于日志）
        
        Returns:
            waypoint_id: 新添加的waypoint ID
        """
        # 使用full_pose获取当前agent位置
        agent_x_m = self.full_pose[0]  # 世界X坐标（米）
        agent_y_m = self.full_pose[1]  # 世界Y坐标（米）
        
        # 转换为世界像素坐标
        pixel_y = int(agent_y_m * 100 / self.resolution)  # Y轴像素 (tensor行)
        pixel_x = int(agent_x_m * 100 / self.resolution)  # X轴像素 (tensor列)
        area_label = self._update_room_area(description, pixel_y, pixel_x)
        
        # ===== 如果与上一个waypoint距离<2m，仅替换上一个，不影响更早历史 =====
        distance_threshold_pixels = 200 / self.resolution  # 2m转换为像素
        if self.waypoint_positions:
            old_py, old_px = self.waypoint_positions[-1]
            distance = np.sqrt((pixel_y - old_py) ** 2 + (pixel_x - old_px) ** 2)
            if distance < distance_threshold_pixels:
                self.waypoint_positions.pop()
                self.waypoint_ids.pop()
                self.waypoint_descriptions.pop()
                self.waypoint_area_labels.pop()
        
        # 分配ID
        self.waypoint_counter += 1
        waypoint_id = self.waypoint_counter
        
        # 保存新waypoint（保存世界像素坐标：(pixel_y, pixel_x) = (行, 列)）
        self.waypoint_positions.append((pixel_y, pixel_x))
        self.waypoint_ids.append(waypoint_id)
        self.waypoint_descriptions.append(description)
        self.waypoint_area_labels.append(area_label)
        
        # print(f"  📍 Waypoint #{waypoint_id} @ (py={pixel_y}, px={pixel_x}) - {description}")
        
        return waypoint_id
    
    def get_waypoints(self) -> Tuple[List[Tuple[int, int]], List[int], List[str]]:
        """
        获取waypoint位置和ID
        
        Returns:
            positions: [(pixel_y, pixel_x), ...] 世界像素坐标列表
            ids: [1, 2, 3, ...] waypoint ID列表
            descriptions: ["desc1", "desc2", ...] waypoint描述列表
        """
        return self.waypoint_positions, self.waypoint_ids, self.waypoint_descriptions

    def get_waypoint_area_labels(self) -> List[str]:
        """获取每个waypoint对应的房间区域标签。"""
        return list(self.waypoint_area_labels)
    
    def clear_waypoints(self):
        """清空所有waypoint"""
        self.waypoint_positions.clear()
        self.waypoint_ids.clear()
        self.waypoint_descriptions.clear()
        self.waypoint_area_labels.clear()
        self.waypoint_counter = 0
    
    def get_waypoint_count(self) -> int:
        """获取waypoint总数"""
        return len(self.waypoint_ids)
    
    # ========== 状态查询方法 ==========
    
    def get_map_state(self) -> Dict[str, Any]:
        """
        获取当前地图状态
        
        注意：
        - floor字段保留用于向后兼容，但实际floor渲染现在从full_map[3+]的
          语义类别中自动获取（floor是第一个mapping_class，索引为0）
        - 返回两个独立的轨迹列表：global_trajectory_points（全局，永不清空）和 subtask_trajectory_points（子任务，可清空）
        - crop_offset用于将世界坐标转换为full_map的局部坐标
        
        Returns:
            state: 地图状态字典
        """
        crop_offset = self.mapping_module.full_map_crop_offset
        global_traj = self.mapping_module.global_trajectory_points  # 全局轨迹
        subtask_traj = self.mapping_module.subtask_trajectory_points  # 子任务轨迹
        room_area_layer, room_area_records = self._build_room_area_layer()
        # print(f"  get_map_state: global_trajectory={len(global_traj)}, subtask_trajectory={len(subtask_traj)}, crop_offset={crop_offset}")
        return {
            'full_map': self.full_map,
            'full_pose': self.full_pose,
            'floor': self.floor,
            'waypoint_positions': self.waypoint_positions,
            'waypoint_ids': self.waypoint_ids,
            'waypoint_area_labels': list(self.waypoint_area_labels),
            'map_shape': self.map_shape,
            'resolution': self.resolution,
            'crop_offset': crop_offset,  # (start_px, start_py) = (world_row_px, world_col_px)
            'global_trajectory_points': global_traj,  # 全局轨迹（global map用）
            'subtask_trajectory_points': subtask_traj,  # 子任务轨迹（local map用）
            'room_area_layer': room_area_layer,
            'room_area_records': room_area_records,
            'current_room_area_label': self.current_room_area_label,
            'current_room_area_type': self.current_room_area_type,
        }
    
    def get_current_pose(self) -> Optional[Tuple[float, float, float]]:
        """获取当前位姿"""
        if self.full_pose is None:
            return None
        return tuple(self.full_pose)

    def _parse_room_type(self, description: str) -> str:
        text = (description or "").strip()
        if not text:
            return "Unknown"

        for sep in ("|", "-", "Nearby", "Connected"):
            if sep in text:
                text = text.split(sep)[0].strip()
        if not text:
            return "Unknown"
        return " ".join(text.split())

    @staticmethod
    def _room_type_key(room_type: str) -> str:
        return "".join(ch.lower() for ch in room_type if ch.isalnum())

    @staticmethod
    def _room_label(room_type: str, variant: int) -> str:
        words = [word.capitalize() for word in room_type.split() if word]
        base = "".join(words) if words else "Unknown"
        return f"{base}{variant}"

    def _build_projector(self) -> Optional[RotatedMapProjector]:
        crop_offset = getattr(self.mapping_module, 'full_map_crop_offset', None)
        if self.full_map is None or self.full_pose is None or crop_offset is None:
            return None
        return RotatedMapProjector(
            map_h=self.full_map.shape[1],
            map_w=self.full_map.shape[2],
            crop_offset=crop_offset,
            agent_orientation_deg=float(self.full_pose[2]),
        )

    def _find_room_area_start(self, traversible: np.ndarray, center_row: int, center_col: int) -> Optional[Tuple[int, int]]:
        h_map, w_map = traversible.shape
        if 0 <= center_row < h_map and 0 <= center_col < w_map and traversible[center_row, center_col]:
            return center_row, center_col

        radius = int(round(100 / self.resolution))
        ys, xs = np.nonzero(traversible)
        if ys.size == 0:
            return None
        d2 = (ys - center_row) ** 2 + (xs - center_col) ** 2
        within = d2 <= radius ** 2
        if not np.any(within):
            return None
        best_idx = int(np.argmin(d2[within]))
        ys_sel = ys[within]
        xs_sel = xs[within]
        return int(ys_sel[best_idx]), int(xs_sel[best_idx])

    def _compute_room_area_world_pixels(self, pixel_y: int, pixel_x: int, max_radius_m: float = 3.0) -> set:
        if self.full_map is None:
            return set()

        projector = self._build_projector()
        if projector is None:
            return set()

        obstacle_mask = self.full_map[0] > 0.5
        explored_mask = self.full_map[1] > 0.5
        traversible = explored_mask & (~obstacle_mask)
        center_rot = projector.world_to_rotated_pixel(pixel_y, pixel_x)
        if center_rot is None:
            return set()

        center_row = int(round(center_rot[0]))
        center_col = int(round(center_rot[1]))
        start = self._find_room_area_start(traversible, center_row, center_col)
        if start is None:
            return {(int(pixel_y), int(pixel_x))}

        max_radius_px = int(round((max_radius_m * 100.0) / self.resolution))
        h_map, w_map = traversible.shape
        visited = np.zeros((h_map, w_map), dtype=bool)
        queue = deque([start])
        visited[start[0], start[1]] = True
        selected_rotated = []

        while queue:
            row, col = queue.popleft()
            if ((row - start[0]) ** 2 + (col - start[1]) ** 2) > max_radius_px ** 2:
                continue
            selected_rotated.append((row, col))

            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    if d_row == 0 and d_col == 0:
                        continue
                    next_row = row + d_row
                    next_col = col + d_col
                    if not (0 <= next_row < h_map and 0 <= next_col < w_map):
                        continue
                    if visited[next_row, next_col] or not traversible[next_row, next_col]:
                        continue
                    if ((next_row - start[0]) ** 2 + (next_col - start[1]) ** 2) > max_radius_px ** 2:
                        continue
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))

        world_pixels = set()
        for row, col in selected_rotated:
            world = projector.rotated_to_world_pixel(row, col)
            if world is None:
                continue
            world_pixels.add((int(round(world[0])), int(round(world[1]))))
        return world_pixels or {(int(pixel_y), int(pixel_x))}

    @staticmethod
    def _pixel_sets_overlap(pixels_a: set, pixels_b: set) -> bool:
        if not pixels_a or not pixels_b:
            return False
        if len(pixels_a) > len(pixels_b):
            pixels_a, pixels_b = pixels_b, pixels_a
        return any(pixel in pixels_b for pixel in pixels_a)

    def _update_room_area(self, description: str, pixel_y: int, pixel_x: int) -> str:
        room_type = self._parse_room_type(description)
        room_key = self._room_type_key(room_type)
        world_pixels = self._compute_room_area_world_pixels(pixel_y, pixel_x)

        overlapping_records = [
            record for record in self.room_area_records
            if record["room_key"] == room_key
            and self._pixel_sets_overlap(record["pixels"], world_pixels)
        ]

        if overlapping_records:
            merged_record = overlapping_records[0]
            merged_record["pixels"].update(world_pixels)
            merged_record["center_world_px"] = (pixel_y, pixel_x)
            merged_record["description"] = description

            if len(overlapping_records) > 1:
                for extra_record in overlapping_records[1:]:
                    merged_record["pixels"].update(extra_record["pixels"])
                    self.room_area_records.remove(extra_record)

            self.current_room_area_label = merged_record["label"]
            self.current_room_area_type = merged_record["room_type"]
            return merged_record["label"]

        existing_variants = [
            int(record["variant"])
            for record in self.room_area_records
            if record["room_key"] == room_key
        ]
        variant = (max(existing_variants) + 1) if existing_variants else 1
        self.room_area_counter += 1
        label = self._room_label(room_type, variant)
        record = {
            "id": self.room_area_counter,
            "label": label,
            "room_type": room_type,
            "room_key": room_key,
            "variant": variant,
            "center_world_px": (pixel_y, pixel_x),
            "pixels": set(world_pixels),
            "description": description,
        }
        self.room_area_records.append(record)
        self.current_room_area_label = label
        self.current_room_area_type = room_type
        return label

    def _set_current_room_area_from_layer(
        self,
        layer: np.ndarray,
        projector: RotatedMapProjector,
    ) -> None:
        if self.full_pose is None or not self.room_area_records:
            return

        curr_py = int(round(float(self.full_pose[1]) * 100.0 / self.resolution))
        curr_px = int(round(float(self.full_pose[0]) * 100.0 / self.resolution))
        current_record = None

        rotated = projector.world_to_rotated_pixel(curr_py, curr_px)
        if rotated is not None:
            row = int(round(rotated[0]))
            col = int(round(rotated[1]))
            if 0 <= row < layer.shape[0] and 0 <= col < layer.shape[1]:
                area_id = int(layer[row, col])
                if area_id > 0:
                    current_record = next(
                        (record for record in self.room_area_records if int(record["id"]) == area_id),
                        None,
                    )

        if current_record is None:
            current_record = min(
                self.room_area_records,
                key=lambda record: float(np.hypot(
                    curr_py - record["center_world_px"][0],
                    curr_px - record["center_world_px"][1],
                )),
            )

        self.current_room_area_label = str(current_record["label"])
        self.current_room_area_type = str(current_record["room_type"])

    def _build_room_area_layer(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if self.full_map is None:
            return np.zeros(self.map_shape, dtype=np.int32), []

        h_map, w_map = self.full_map.shape[1], self.full_map.shape[2]
        layer = np.zeros((h_map, w_map), dtype=np.int32)
        best_distance = np.full((h_map, w_map), np.inf, dtype=np.float32)
        projector = self._build_projector()
        if projector is None:
            return layer, []

        area_records: List[Dict[str, Any]] = []
        for record in self.room_area_records:
            center_py, center_px = record["center_world_px"]
            area_records.append({
                "id": int(record["id"]),
                "label": str(record["label"]),
                "room_type": str(record["room_type"]),
                "variant": int(record["variant"]),
                "center_world_px": (int(center_py), int(center_px)),
            })

            for world_py, world_px in record["pixels"]:
                rotated = projector.world_to_rotated_pixel(world_py, world_px)
                if rotated is None:
                    continue
                row = int(round(rotated[0]))
                col = int(round(rotated[1]))
                if not (0 <= row < h_map and 0 <= col < w_map):
                    continue
                dist = float(np.hypot(world_py - center_py, world_px - center_px))
                if dist < best_distance[row, col]:
                    best_distance[row, col] = dist
                    layer[row, col] = int(record["id"])
        self._set_current_room_area_from_layer(layer, projector)

        return layer, area_records


# ========== 便捷函数 ==========

def create_mapper(mapping_module, 
                 map_shape: Tuple[int, int],
                 resolution: int = 5) -> SemanticMapper:
    """创建SemanticMapper实例"""
    return SemanticMapper(mapping_module, map_shape, resolution)
