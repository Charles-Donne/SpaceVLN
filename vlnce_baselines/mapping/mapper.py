"""
语义建图器 - SemanticMapper
============================
职责：
1. 语义地图更新逻辑
2. Floor区域提取
3. 轨迹管理
4. 地图状态查询

设计原则：
- 单一职责：只负责建图和地图状态管理
- 解耦：独立于可视化和控制器
- 封装：隐藏地图内部实现细节
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import torch

from skimage.morphology import disk, remove_small_objects
from skimage.morphology import binary_closing as _binary_closing_compat

from vlnce_baselines.config_system.constants import navigable_classes, map_channels


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
        self.waypoint_descriptions = []  # ["desc1", "desc2", ...] 对应的waypoint描述
        self.waypoint_counter = 0  # waypoint计数器
        
        # 地图缓存
        self.floor = np.zeros(map_shape)
        self.full_map = None
        self.full_pose = None
    
    def reset(self):
        """重置建图器状态"""
        self.waypoint_positions = []  # 清空waypoint位置
        self.waypoint_ids = []  # 清空waypoint ID
        self.waypoint_descriptions = []  # 清空waypoint描述
        self.waypoint_counter = 0  # 重置计数器
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
        self.mapping_module(batch_obs, poses)
        
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
        
        # 4. 清空单步地图（准备下一步）
        self.mapping_module.one_step_full_map.fill_(0.)
        self.mapping_module.one_step_local_map.fill_(0.)
        
        # 5. 获取轨迹（已经在mapping_module中记录并转换为世界坐标）
        trajectory_points = self.mapping_module.get_trajectory()
        global_trajectory_points = self.mapping_module.get_global_trajectory()
        
        # 6. 转换轨迹坐标：世界坐标 → full_map局部坐标（包含旋转）
        # full_map 是以 agent 为中心的 480×480 裁剪区域，并且已经旋转让agent朝向向上
        # 需要将世界坐标系的 trajectory_points 转换为相对于 full_map 的局部坐标，并应用相同的旋转
        crop_offset = self.mapping_module.full_map_crop_offset
        agent_orientation = self.full_pose[2]  # agent朝向（度数）
        
        local_trajectory_points = self.convert_trajectory_to_local(
            trajectory_points, crop_offset, agent_orientation
        )
        local_global_trajectory_points = self.convert_trajectory_to_local(
            global_trajectory_points, crop_offset, agent_orientation
        )
        
        return {
            'full_map': self.full_map,
            'full_pose': self.full_pose,
            'floor': self.floor,
            'trajectory_points': local_trajectory_points,  # 转换为局部坐标（相对于full_map，已旋转）
            'global_trajectory_points': local_global_trajectory_points  # 转换为局部坐标（相对于full_map，已旋转）
        }
    
    def extract_floor(self, 
                     full_map: np.ndarray,
                     detected_classes: List[str]) -> np.ndarray:
        """
        从full_map提取floor区域（已弃用：floor现在是语义类别）
        
        注意：按照ZS_Evaluator的方式，floor现在是full_map[4+]中的第一个语义类别，
        不再需要通过形态学方法提取。这个方法保留仅用于向后兼容。
        
        Args:
            full_map: [C, H, W] 全局地图
            detected_classes: 已检测类别列表（全局累计的类别，可能多于当前步的检测）
        
        Returns:
            floor: [H, W] floor地图（现在主要用于向后兼容，实际floor在semantic layer）
        """
        # 使用阈值过滤小区域
        full_map_filtered = remove_small_objects(full_map.astype(bool), min_size=16)
        
        # 提取地图通道
        obstacles = full_map_filtered[0, ...].astype(bool)  # 障碍物
        explored_area = full_map_filtered[1, ...].astype(bool)  # 已探索区域
        
        # 提取语义层（从第 map_channels 个通道开始）
        semantic_layers = full_map_filtered[map_channels:, ...]
        
        # 关键修复：使用full_map的实际通道数，而不是detected_classes的长度
        # detected_classes是全局累计的，但每步的full_map只包含当前步检测到的类别
        num_semantic_channels = semantic_layers.shape[0]
        
        # 如果没有语义通道，直接返回基于explored的简单floor
        if num_semantic_channels == 0:
            # 简单处理：explored且非障碍物的区域
            floor = np.logical_and(explored_area, np.logical_not(obstacles))
            return floor.astype(np.uint8)
        
        # 区分可导航和不可导航的类别（只处理当前步实际存在的类别）
        navigable_index = []
        not_navigable_index = []
        
        for i in range(num_semantic_channels):
            # 由于detected_classes可能多于semantic_layers，需要安全索引
            if i < len(detected_classes):
                cls_name = detected_classes[i]
                if cls_name in navigable_classes:
                    navigable_index.append(i)
                else:
                    not_navigable_index.append(i)
        
        # 不可导航物体
        if len(not_navigable_index) > 0:
            objects = np.sum(semantic_layers[not_navigable_index], axis=0).astype(bool)
        else:
            objects = np.zeros_like(obstacles)
        
        # 可导航区域（如floor, stairs等）
        if len(navigable_index) > 0:
            navigable = np.logical_or.reduce(semantic_layers[navigable_index])
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
    
    def convert_trajectory_to_local(self, trajectory_points: List[Tuple[int, int]], crop_offset: Tuple[int, int], agent_orientation: float = None) -> List[Tuple[int, int]]:
        """
        将世界坐标系的轨迹点转换为相对于裁剪地图的局部坐标（可选旋转）
        
        Args:
            trajectory_points: [(world_px, world_py), ...] 世界像素坐标
            crop_offset: (start_px, start_py) 裁剪区域左上角的世界像素坐标
            agent_orientation: agent朝向（度数），如果提供则旋转轨迹点
        
        Returns:
            local_trajectory: [(local_px, local_py), ...] 相对于裁剪地图的局部坐标（已旋转）
        """
        import math
        import numpy as np
        
        start_px, start_py = crop_offset
        local_trajectory = []
        
        # 步骤1: 世界坐标 → 局部坐标（相对于裁剪区域左上角）
        for world_px, world_py in trajectory_points:
            # world坐标: (px, py) = (Y轴, X轴)
            local_px = world_px - start_px  # Y轴方向
            local_py = world_py - start_py  # X轴方向
            local_trajectory.append((local_px, local_py))
        
        # 步骤2: 如果需要，旋转轨迹点（围绕地图中心旋转）
        if agent_orientation is not None:
            # 旋转角度 = 90 - agent_orientation（让agent朝向变成正上方）
            rotation_angle_deg = 90 - agent_orientation
            rotation_angle_rad = math.radians(rotation_angle_deg)
            
            if abs(rotation_angle_deg) >= 0.1:  # 只有显著旋转才处理
                cos_theta = math.cos(rotation_angle_rad)
                sin_theta = math.sin(rotation_angle_rad)
                
                # 地图中心（旋转中心）
                map_size = 480  # full_map 固定480×480
                center = map_size / 2.0  # 240
                
                rotated_trajectory = []
                for local_px, local_py in local_trajectory:
                    # 转换到以中心为原点的坐标系
                    px_centered = local_px - center
                    py_centered = local_py - center
                    
                    # 应用旋转矩阵
                    # 注意：这里的旋转要与_rotate_map_to_agent_heading中的旋转一致
                    px_rotated = cos_theta * px_centered - sin_theta * py_centered
                    py_rotated = sin_theta * px_centered + cos_theta * py_centered
                    
                    # 转换回以左上角为原点的坐标系
                    px_final = px_rotated + center
                    py_final = py_rotated + center
                    
                    rotated_trajectory.append((px_final, py_final))
                
                local_trajectory = rotated_trajectory
        
        return local_trajectory
    
    def toggle_trajectory(self):
        """切换轨迹绘制开关"""
        self.enable_trajectory = not self.enable_trajectory
        status = "启用" if self.enable_trajectory else "禁用"
        return status
    
    def clear_trajectory(self):
        """
        清空当前子任务轨迹（用于local map）
        
        使用场景：
        - 子任务完成时：清空上一子任务的轨迹，开始记录新子任务轨迹
        - 每个子任务都有独立的local map轨迹显示，不会累积
        
        注意：
        - 只清空trajectory_points（local map使用）
        - global_trajectory_points不清空，保持完整历史轨迹（global map使用）
        - 轨迹是动态绘制在地图上的，不会写入底图
        - landmark标注同样是动态绘制，更新landmark_classes即可替换
        """
        self.mapping_module.clear_trajectory()  # 清空mapping_module中的当前子任务轨迹
        # global_trajectory_points保持不变，继续累积
    
    # ========== Waypoint管理方法 ==========
    
    def add_waypoint(self, description: str = "") -> int:
        """
        添加waypoint到当前位置
        
        新增机制：如果当前位置2m之内有之前的waypoint，则删除2m之内的旧waypoint，
        避免同一位置扎堆过多waypoint
        
        Args:
            description: waypoint描述（可选，用于日志）
        
        Returns:
            waypoint_id: 新添加的waypoint ID
        """
        # ===== 关键修复：直接使用trajectory_points[-1]，确保与agent位置完全一致 =====
        if len(self.trajectory_points) == 0:
            print("  ⚠️  Cannot add waypoint: trajectory is empty")
            return -1
        
        # 直接使用轨迹最后一个点作为waypoint位置
        # trajectory_points存储格式：(map_y, map_x)
        last_traj_x, last_traj_y = self.trajectory_points[-1]
        
        # waypoint存储为(map_x, map_y)格式
        map_x = last_traj_x  # 从trajectory的x → map_x
        map_y = last_traj_y  # 从trajectory的y → map_y
        
        # ===== 新增：移除2m范围内的旧waypoint =====
        distance_threshold_pixels = 200 / self.resolution  # 2m转换为像素（200cm / resolution）
        
        # 查找需要保留的waypoint（2m之外的waypoint）
        waypoints_to_keep = []
        for i, (old_x, old_y) in enumerate(self.waypoint_positions):
            # 计算当前位置与旧waypoint的距离
            distance = np.sqrt((map_x - old_x) ** 2 + (map_y - old_y) ** 2)
            
            if distance >= distance_threshold_pixels:
                # 距离>=2m，保留
                waypoints_to_keep.append(i)
            else:
                # 距离<2m，删除（打印日志）
                old_id = self.waypoint_ids[i]
                old_desc = self.waypoint_descriptions[i]
                print(f"  🗑️  Removed nearby Waypoint #{old_id} @ ({old_x}, {old_y}) - {old_desc} (distance: {distance * self.resolution:.1f}cm < 200cm)")
        
        # 更新waypoint列表（只保留2m之外的waypoint）
        if waypoints_to_keep:
            self.waypoint_positions = [self.waypoint_positions[i] for i in waypoints_to_keep]
            self.waypoint_ids = [self.waypoint_ids[i] for i in waypoints_to_keep]
            self.waypoint_descriptions = [self.waypoint_descriptions[i] for i in waypoints_to_keep]
        else:
            # 所有旧waypoint都被删除
            self.waypoint_positions = []
            self.waypoint_ids = []
            self.waypoint_descriptions = []
        
        # 分配ID
        self.waypoint_counter += 1
        waypoint_id = self.waypoint_counter
        
        # 保存新waypoint（只保存位置）
        self.waypoint_positions.append((map_x, map_y))
        self.waypoint_ids.append(waypoint_id)
        self.waypoint_descriptions.append(description)
        
        print(f"  📍 Waypoint #{waypoint_id} @ ({map_x}, {map_y}) - {description}")
        
        return waypoint_id
    
    def get_waypoints(self) -> Tuple[List[Tuple[int, int]], List[int], List[str]]:
        """
        获取所有waypoint的位置、ID和描述
        
        Returns:
            positions: [(map_x, map_y), ...] 地图坐标列表
            ids: [1, 2, 3, ...] waypoint ID列表
            descriptions: ["desc1", "desc2", ...] waypoint描述列表
        """
        return self.waypoint_positions, self.waypoint_ids, self.waypoint_descriptions
    
    def clear_waypoints(self):
        """清空所有waypoint"""
        self.waypoint_positions = []
        self.waypoint_ids = []
        self.waypoint_descriptions = []
        self.waypoint_counter = 0
    
    def get_waypoint_count(self) -> int:
        """获取waypoint总数"""
        return len(self.waypoint_ids)
    
    # ========== 状态查询方法 ==========
    
    def get_map_state(self) -> Dict[str, Any]:
        """
        获取当前地图状态
        
        注意：floor字段保留用于向后兼容，但实际floor渲染现在从full_map[4+]的
        语义类别中自动获取（floor是第一个mapping_class，索引为0）
        
        Returns:
            state: 地图状态字典
        """
        return {
            'full_map': self.full_map,
            'full_pose': self.full_pose,
            'floor': self.floor,
            'trajectory_points': self.trajectory_points,
            'global_trajectory_points': self.global_trajectory_points,
            'waypoint_positions': self.waypoint_positions,
            'waypoint_ids': self.waypoint_ids,
            'map_shape': self.map_shape,
            'resolution': self.resolution
        }
    
    def get_current_pose(self) -> Optional[Tuple[float, float, float]]:
        """获取当前位姿"""
        if self.full_pose is None:
            return None
        return tuple(self.full_pose)


# ========== 便捷函数 ==========

def create_mapper(mapping_module, 
                 map_shape: Tuple[int, int],
                 resolution: int = 5) -> SemanticMapper:
    """创建SemanticMapper实例"""
    return SemanticMapper(mapping_module, map_shape, resolution)
