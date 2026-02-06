"""
地图可视化工具 - MapVisualizer
================================
职责：
1. 地图渲染（全局地图、局部地图）
2. 检测结果可视化
3. 轨迹绘制
4. 文件保存

设计原则：
- 单一职责：只负责可视化和保存，不涉及建图逻辑
- 解耦：独立于Controller和Mapper
- 可复用：支持多种可视化场景
"""

import os
import cv2
import copy
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any

from vlnce_baselines.visualization import rendering as vu
from vlnce_baselines.config_system.constants import (
    color_palette, 
    detection_colors,
    detection_thickness,
    landmark_marker_color,
    landmark_marker_border,
    landmark_marker_radius,
)


class MapVisualizer:
    """地图可视化器 - 统一管理所有可视化和保存逻辑"""
    
    def __init__(self, 
                 results_dir: str,
                 resolution: int = 5,
                 map_shape: Tuple[int, int] = (480, 480),
                 enable_global_map_crop: bool = False,
                 enable_adaptive_zoom: bool = False):
        """
        Args:
            results_dir: 保存根目录（如：data/manual_navigation）
            resolution: 地图分辨率（cm/pixel）
            map_shape: 地图尺寸
            enable_global_map_crop: 是否裁剪global map到440×440（默认False，保持480×480）
            enable_adaptive_zoom: 是否启用自适应缩放（根据轨迹范围自动调整显示区域）
        """
        self.results_dir = results_dir
        self.resolution = resolution
        self.map_shape = map_shape
        self.enable_global_map_crop = enable_global_map_crop
        self.enable_adaptive_zoom = enable_adaptive_zoom
        self.color_palette = [int(x * 255.) for x in color_palette]
        
        # 注意：不在初始化时创建目录，而是在保存时根据episode_id动态创建
    
    def _create_episode_directories(self, episode_id: int):
        """为特定episode创建保存目录"""
        episode_dir = os.path.join(self.results_dir, f'episode_{episode_id}')
        dirs = ['rgb', 'global_map', 'local_map', 'detection']
        for dir_name in dirs:
            os.makedirs(os.path.join(episode_dir, dir_name), exist_ok=True)
        return episode_dir
    
    # ========== 距离计算方法 ==========
    
    def calculate_obstacle_distances_from_rotated_map(
        self,
        obstacle_mask_rotated: np.ndarray,
        center_x: int = 240,
        center_y: int = 240
    ) -> Dict[str, str]:
        """
        在旋转后的obstacle map上计算障碍物距离
        
        ⚠️ 关键优势：
        - 地图已经旋转，箭头朝上（-90°），agent在(240, 240)
        - 直接在像素坐标系中测距，无需复杂的Habitat角度转换
        - 上方 = FRONT, 左上30° = LEFT_30, 右上30° = RIGHT_30
        
        Args:
            obstacle_mask_rotated: [480, 480] 旋转后的障碍物掩码（bool或0/1）
            center_x: Agent中心X坐标（默认240）
            center_y: Agent中心Y坐标（默认240）
            
        Returns:
            距离字典 {
                'front': "X.XXm" | ">2.0m open" | "<0.5m WARNING",
                'left_30': ...,
                'right_30': ...,
                'left_90': ...,
                'right_90': ...
            }
        """
        # 确保是bool mask
        if obstacle_mask_rotated.dtype != bool:
            obstacle_mask_rotated = obstacle_mask_rotated > 127
        
        # 定义7个方向（在旋转后的地图上，箭头朝上=-90°）
        # 用于Action模式（前方扇形视野）
        directions = {
            'front': -90,
            'left_30': -120,
            'left_60': -150,
            'left_90': -180,
            'right_30': -60,
            'right_60': -30,
            'right_90': 0
        }
        
        distances = {}
        
        # 计算5个关键方向
        for key, angle in directions.items():
            # 多光线扫描（5条光线，±5°范围）
            ray_distances = []
            for offset in [-5, -2.5, 0, 2.5, 5]:
                test_angle = angle + offset
                dist_m = self._raycast_on_rotated_map(
                    obstacle_mask_rotated, center_x, center_y, test_angle
                )
                if dist_m is not None:
                    ray_distances.append(dist_m)
            
            # 使用中位数距离
            if ray_distances:
                median_dist = np.median(ray_distances)
                distances[key] = self._format_distance(median_dist)
            else:
                distances[key] = "Unknown"
        
        return distances
    
    def calculate_obstacle_distances_12_directions(
        self,
        obstacle_mask_rotated: np.ndarray,
        center_x: int = 240,
        center_y: int = 240
    ) -> Dict[str, str]:
        """
        在旋转后的obstacle map上计算12个方向的障碍物距离（用于Thinking模式环视）
        
        覆盖完整360°：每30°一个方向，对应12张IMAGE
        
        Args:
            obstacle_mask_rotated: [480, 480] 旋转后的障碍物掩码
            center_x: Agent中心X坐标（默认240）
            center_y: Agent中心Y坐标（默认240）
            
        Returns:
            距离字典 {
                'angle_0': "X.XXm",    # IMAGE 1: Front (0°)
                'angle_30': "X.XXm",   # IMAGE 2: Right (30°)
                'angle_60': "X.XXm",   # IMAGE 3: Right (60°)
                ...
                'angle_330': "X.XXm"   # IMAGE 12: Left (330°)
            }
        """
        # 确保是bool mask
        if obstacle_mask_rotated.dtype != bool:
            obstacle_mask_rotated = obstacle_mask_rotated > 127
        
        # 定义12个方向（在旋转后的地图上，箭头朝上=-90°）
        # 环视是逆时针TURN_LEFT，所以30°是左侧，330°是右侧
        # agent角度 → 旋转后地图角度的映射
        directions = {
            'angle_0':   -90,   # Front (0°)
            'angle_30':  -120,  # Left 30° (逆时针)
            'angle_60':  -150,  # Left 60°
            'angle_90':  -180,  # Left 90°
            'angle_120': 150,   # Left 120° (=-210°)
            'angle_150': 120,   # Left 150° (=-240°)
            'angle_180': 90,    # Back (180°)
            'angle_210': 60,    # Right 150° (顺时针150°)
            'angle_240': 30,    # Right 120°
            'angle_270': 0,     # Right 90°
            'angle_300': -30,   # Right 60°
            'angle_330': -60    # Right 30°
        }
        
        distances = {}
        
        # 计算12个方向
        for key, angle in directions.items():
            # 多光线扫描（5条光线，±5°范围）
            ray_distances = []
            for offset in [-5, -2.5, 0, 2.5, 5]:
                test_angle = angle + offset
                dist_m = self._raycast_on_rotated_map(
                    obstacle_mask_rotated, center_x, center_y, test_angle
                )
                if dist_m is not None:
                    ray_distances.append(dist_m)
            
            # 使用中位数距离
            if ray_distances:
                median_dist = np.median(ray_distances)
                distances[key] = self._format_distance(median_dist)
            else:
                distances[key] = "Unknown"
        
        return distances
    
    def _raycast_on_rotated_map(
        self,
        obstacle_mask: np.ndarray,
        start_x: int,
        start_y: int,
        angle_deg: float
    ) -> Optional[float]:
        """
        在旋转后的地图上进行光线投射
        
        图像坐标系：
        - X向右（列），Y向下（行）
        - 角度从+X轴逆时针：0°=右，90°=下，180°=左，270°=上
        
        Args:
            obstacle_mask: [H, W] bool数组
            start_x, start_y: 起始位置（像素）
            angle_deg: 方向角度（度，图像坐标系）
            
        Returns:
            距离（米），如果超出2.0m返回2.1
        """
        h, w = obstacle_mask.shape
        angle_rad = np.deg2rad(angle_deg)
        
        # 方向向量（图像坐标系）
        dx = np.cos(angle_rad)  # X方向（列）
        dy = np.sin(angle_rad)  # Y方向（行）
        
        max_distance_m = 2.0
        step_size = 0.5  # 0.5像素 = 2.5cm
        resolution_cm = 5  # 5cm/pixel
        
        # 光线步进
        distance_px = 0.0
        max_steps = int(max_distance_m * 100 / resolution_cm / step_size)  # 2m / 0.025m = 80步
        
        for _ in range(max_steps):
            distance_px += step_size
            current_x = start_x + dx * distance_px
            current_y = start_y + dy * distance_px
            
            # 边界检查
            ix, iy = int(round(current_x)), int(round(current_y))
            if not (0 <= ix < w and 0 <= iy < h):
                return 2.1  # 超出地图
            
            # 障碍物检测
            if obstacle_mask[iy, ix]:
                distance_m = distance_px * resolution_cm / 100.0
                return distance_m
        
        return 2.1  # 超过最大范围
    
    def _format_distance(self, distance_m: Optional[float]) -> str:
        """格式化距离字符串"""
        if distance_m is None:
            return "Unknown"
        elif distance_m > 2.0:
            return ">2.0m open"
        elif distance_m < 0.5:
            return f"{distance_m:.2f}m WARNING"
        else:
            return f"{distance_m:.2f}m"
    
    @staticmethod
    def get_distance_summary(distances: Dict[str, str]) -> str:
        """生成距离摘要字符串（供日志打印）"""
        return (f"FRONT={distances.get('front', 'Unknown')}, "
                f"L30={distances.get('left_30', 'Unknown')}, "
                f"R30={distances.get('right_30', 'Unknown')}, "
                f"L90={distances.get('left_90', 'Unknown')}, "
                f"R90={distances.get('right_90', 'Unknown')}")
    
    def _calculate_map_usage(self, 
                            trajectory_points: List[Tuple[int, int]], 
                            h: int, 
                            w: int) -> Dict[str, Any]:
        """
        计算地图实际使用情况统计
        
        Args:
            trajectory_points: 轨迹点列表 [(x, y), ...]（像素坐标）
            h, w: 地图尺寸（像素）
        
        Returns:
            统计字典，包括：
            - x_min, x_max, y_min, y_max: 轨迹在实际世界的范围（米）
            - used_width, used_height: 实际使用的宽度和高度（米）
            - usage_percent: 使用百分比
            - near_boundary: 是否接近边界（距离<10%）
        """
        if not trajectory_points:
            return {
                'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0,
                'used_width': 0, 'used_height': 0,
                'usage_percent': 0,
                'near_boundary': False
            }
        
        # 将像素坐标转换为实际世界坐标（米）
        # trajectory_points 中 (x, y) 是地图坐标系的像素位置
        x_coords = [y * self.resolution / 100.0 for x, y in trajectory_points]  # y是水平方向
        y_coords = [(h - 1 - x) * self.resolution / 100.0 for x, y in trajectory_points]  # x是垂直方向
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        used_width = x_max - x_min
        used_height = y_max - y_min
        
        map_width_m = w * self.resolution / 100.0
        map_height_m = h * self.resolution / 100.0
        
        usage_percent = (used_width * used_height) / (map_width_m * map_height_m) * 100
        
        # 检查是否接近边界（距离<10%）
        boundary_threshold = 0.1
        near_boundary = (
            x_min < map_width_m * boundary_threshold or
            x_max > map_width_m * (1 - boundary_threshold) or
            y_min < map_height_m * boundary_threshold or
            y_max > map_height_m * (1 - boundary_threshold)
        )
        
        return {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'used_width': used_width,
            'used_height': used_height,
            'usage_percent': usage_percent,
            'near_boundary': near_boundary
        }
    
    # ========== 渲染方法 ==========
    
    def render_global_map(self,
                         full_map: np.ndarray,
                         trajectory_points: List[Tuple[int, int]],
                         detected_classes: List[str],
                         floor: Optional[np.ndarray] = None,
                         current_pose: Optional[Tuple[float, float, float]] = None,
                         landmark_classes: Optional[List[str]] = None,
                         landmark_config: Optional[Dict] = None,
                         waypoint_positions: Optional[List[Tuple[int, int]]] = None,
                         waypoint_ids: Optional[List[int]] = None,
                         calculate_distances: bool = False) -> Tuple[np.ndarray, np.ndarray, List, np.ndarray, Dict[str, str], Optional[float]]:
        """
        渲染全局地图（严格按照ZS_Evaluator的渲染逻辑 + 平滑轨迹线）
        
        Args:
            full_map: [C, H, W] 全局地图
                [0] = obstacle map (障碍物)
                [1] = explored map (已探索)
                [2] = current position
                [3] = history position
                [4+] = semantic classes (用于landmark标注，不用于floor渲染)
            trajectory_points: [(x, y), ...] 轨迹坐标列表（像素坐标）
            detected_classes: 已检测类别列表
            floor: [H, W] floor地图（通过形态学方法计算，像ZS_Evaluator）
            current_pose: (x, y, orientation) 当前位姿
            landmark_classes: landmark类别列表
            landmark_config: landmark配置 {min_total_pixels, min_area_threshold}
        
        Returns:
            (sem_map_vis, global_map_with_trajectory, landmarks, global_map_rotated, obstacle_distances, last_waypoint_angle)
            - sem_map_vis: 基础渲染地图 (480×480)
            - global_map_with_trajectory: 带轨迹的旋转地图 (440×440)
            - landmarks: [(x, y, class_name), ...] 标注列表
            - global_map_rotated: 旋转地图（无轨迹，440×440）
            - obstacle_distances: {'front': "X.XXm", 'left_30': ..., ...} 5方向距离
            - last_waypoint_angle: 最后一个waypoint相对于正前方的角度（弧度），None表示无waypoint
        
        渲染层次（严格按照ZS_Evaluator）:
            - 白色(0): 未探索区域
            - 浅灰色(2): 已探索自由空间（先渲染）
            - 黑色(1): 障碍物（覆盖已探索）
            - 浅绿色(5): Floor（通过形态学计算，覆盖障碍物）
            - 橙色(3): Agent轨迹（最后覆盖）
            
        注意：不渲染bed/chair等语义类别的颜色，只用于landmark标注
        """
        obstacle_map = full_map[0, ...]
        explored_map = full_map[1, ...]
        h, w = obstacle_map.shape
        
        # ===== 计算地图使用统计（显示真实探索范围）=====
        map_usage_stats = self._calculate_map_usage(trajectory_points, h, w)
        # 静默处理统计信息，不打印详细数据
        
        # ===== 阶段1: 创建语义地图（严格按照ZS_Evaluator的layer顺序）=====
        semantic_map = np.zeros((h, w), dtype=np.uint8)
        
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        
        # ===== 基础层渲染顺序 =====
        # 底层到中层：已探索自由空间(底层) → Floor(中层)
        # 障碍物层将在绘制轨迹后、箭头前单独叠加
        
        # Layer 1: 已探索自由空间（浅灰色）- 先绘制底层
        explored_free_mask = np.logical_and(explored_mask, ~obstacle_mask)
        semantic_map[explored_free_mask] = 2
        
        # Layer 2: Floor（浅绿色）- 覆盖部分自由空间
        if floor is not None:
            floor_mask = floor.astype(bool)
            floor_display_mask = np.logical_and(floor_mask, explored_mask)
            semantic_map[floor_display_mask] = 5  # 浅绿色
        
        # ===== 阶段2: PIL调色板渲染 =====
        sem_map_vis = Image.new("P", (w, h))
        sem_map_vis.putpalette(self.color_palette)
        sem_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        
        # 坐标系变换：翻转Y轴 + RGB→BGR
        sem_map_vis = np.flipud(sem_map_vis)
        sem_map_vis = np.array(sem_map_vis)
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]  # RGB → BGR
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST)
        
        # ===== 阶段3: 提取Landmark位置（但不绘制）=====
        landmarks = []
        if landmark_classes and landmark_config:
            landmarks = self._extract_landmarks(
                full_map, detected_classes, landmark_classes,
                landmark_config['min_total_pixels'],
                landmark_config['min_area_threshold']
            )
        
        # ===== 阶段4: 旋转调整（箭头朝上，居中240,240）=====
        global_map_rotated = None
        if current_pose is not None:
            current_x, current_y, current_o = current_pose
            
            # ===== 关键：trajectory_points 现在是局部坐标（相对于 full_map）=====
            # full_map 是以 agent 为中心的裁剪区域，所以 agent 应该在 (240, 240)
            # trajectory_points[-1] 应该直接是局部坐标 (px, py) 格式
            if len(trajectory_points) > 0:
                last_traj_px, last_traj_py = trajectory_points[-1]
                # trajectory_points 格式: (px, py) = (Y轴像素, X轴像素)
                # 转换到显示坐标: agent_x = py (水平), agent_y = px (垂直)
                # 由于 full_map 大小就是 480×480，不需要缩放
                agent_y = last_traj_px  # Y轴像素 → 垂直位置
                agent_x = last_traj_py  # X轴像素 → 水平位置
                # 注意：由于坐标系统修复，agent应该在 (240, 240)
            else:
                # 回退：如果没有轨迹，agent 应该在中心
                agent_x = 240
                agent_y = 240
            
            # ===== 变换矩阵数学原理 =====
            # 
            # 变换矩阵结构（2x3仿射变换矩阵）：
            #   rotation_matrix = [
            #       [cos(θ), -sin(θ), tx],   ← 第0行：X方向变换
            #       [sin(θ),  cos(θ), ty]    ← 第1行：Y方向变换
            #   ]
            # 
            # 对地图上任意点(x, y)，计算变换后的新位置(new_x, new_y)：
            #   new_x = cos(θ)*x - sin(θ)*y + tx  ← 旋转部分 + 平移部分
            #   new_y = sin(θ)*x + cos(θ)*y + ty  ← 旋转部分 + 平移部分
            #          └──────旋转──────┘   └─平移─┘
            # 
            # 也就是写成：
            #   new_x = rotation_matrix[0,0]*x + rotation_matrix[0,1]*y + rotation_matrix[0,2]
            #   new_y = rotation_matrix[1,0]*x + rotation_matrix[1,1]*y + rotation_matrix[1,2]
            # 
            # 所以你的理解完全正确：确实是"相加了两个部分"！
            # - 前两项 (rotation_matrix[0,0]*x + rotation_matrix[0,1]*y)：旋转
            # - 最后一项 (rotation_matrix[0,2])：平移
            # 
            # ===== 我们的具体操作 =====
            # 
            # 步骤1: 创建旋转矩阵（围绕agent位置旋转）
            #   rotation_matrix = cv2.getRotationMatrix2D((agent_x, agent_y), rotation_angle, 1.0)
            #   此时 rotation_matrix[0,2] 和 rotation_matrix[1,2] 已经包含了：
            #   - 围绕(agent_x, agent_y)旋转所需的平移分量
            #   - 公式：先平移到原点 → 旋转 → 平移回去
            # 
            # 步骤2: 计算旋转后agent的实际位置
            #   rotated_center = rotation_matrix @ [agent_x, agent_y, 1]
            #   理论上应该还在(agent_x, agent_y)，因为它是旋转中心
            #   但实际有微小数值误差
            # 
            # 步骤3: 添加额外平移，让agent移动到(240, 240)
            #   translation = [240, 240] - rotated_center[:2]
            #   rotation_matrix[0, 2] += translation[0]  ← 在原有tx上叠加新的平移
            #   rotation_matrix[1, 2] += translation[1]  ← 在原有ty上叠加新的平移
            # 
            # 步骤4: 应用最终变换到整个地图
            #   cv2.warpAffine(地图, rotation_matrix, ...)
            #   对地图每个像素都执行上面的公式计算
            # 
            # ===== 最终效果 =====
            # - agent从(agent_x, agent_y)移动到(240, 240) ← 视觉效果
            # - 实际是：整个地图背景移动了，agent相对画布的位置改变了
            # - 轨迹点、landmark等所有元素都跟随地图一起变换
            
            # 旋转使箭头朝正上方
            rotation_angle = 90 - current_o
            rotation_center = (agent_x, agent_y)  # 围绕agent当前位置旋转
            rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1.0)
            
            # 添加平移步骤：将旋转后的agent移动到(240, 240)
            target_center = np.array([240, 240, 1])  # 目标：agent应该在这里
            current_center = np.array([agent_x, agent_y, 1])  # agent当前在这里
            rotated_center = rotation_matrix @ current_center  # 旋转后agent在这里
            
            # 计算平移量：从rotated_center到target_center需要移动多少
            translation = target_center[:2] - rotated_center[:2]
            rotation_matrix[0, 2] += translation[0]  # 添加X方向平移
            rotation_matrix[1, 2] += translation[1]  # 添加Y方向平移

            
            global_map_rotated = cv2.warpAffine(
                sem_map_vis, rotation_matrix, (480, 480),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
            
            # ===== 阶段5: 创建global_map的显示副本（用于绘制轨迹和landmark）=====
            global_map_with_trajectory = global_map_rotated.copy()
            
            # 先在副本上绘制轨迹线（底层）
            if len(trajectory_points) >= 2:
                # 转换轨迹点到旋转后的坐标系
                rotated_trajectory = []
                for px, py in trajectory_points:
                    # trajectory_points 现在是局部坐标: (px, py) = (Y轴像素, X轴像素)
                    # 转换到显示坐标: display_x = py (水平), display_y = px (垂直)
                    # 由于 full_map 大小就是 480×480，不需要缩放
                    display_x = py  # X轴像素 → 水平位置
                    display_y = px  # Y轴像素 → 垂直位置
                    
                    # 应用旋转变换
                    point = np.array([display_x, display_y, 1])
                    rotated_point = rotation_matrix @ point
                    rotated_trajectory.append([int(round(rotated_point[0])), int(round(rotated_point[1]))])
                
                # 绘制实心轨迹线（2像素宽）
                if len(rotated_trajectory) >= 2:
                    trajectory_array = np.array(rotated_trajectory, dtype=np.int32)
                    cv2.polylines(global_map_with_trajectory, [trajectory_array], isClosed=False,
                                 color=(0, 165, 255), thickness=2, lineType=cv2.LINE_8)
            
            # ===== 阶段5.3: 绘制深红色虚线指示正前方（在轨迹之后，箭头之前）=====
            center_x, center_y = 240, 240
            forward_line_length = 120  # 延伸120像素（约3米）
            forward_color = (0, 0, 180)  # 深红色 BGR
            forward_thickness = 2
            
            # 绘制从agent中心向正上方延伸的虚线
            # 虚线：每段10像素，间隙5像素
            dash_length = 10
            gap_length = 5
            num_dashes = int(forward_line_length / (dash_length + gap_length))
            
            for i in range(num_dashes):
                dash_start_y = 240 - i * (dash_length + gap_length)
                dash_end_y = dash_start_y - dash_length
                if dash_end_y < 240 - forward_line_length:
                    dash_end_y = 240 - forward_line_length
                cv2.line(global_map_with_trajectory, (240, int(dash_start_y)), 
                        (240, int(dash_end_y)), forward_color, forward_thickness)
            
            # ===== 阶段5.5: 在中心绘制箭头（在虚线之后）=====
            arrow_angle = np.deg2rad(-90)  # 朝上
            agent_pos = (center_x, center_y, arrow_angle)
            agent_arrow = vu.get_contour_points(agent_pos, origin=(0, 0), size=12)
            cv2.drawContours(global_map_with_trajectory, [agent_arrow], 0, (0, 0, 255), -1)
            
            # ===== 阶段5.6: 叠加黑色障碍物层（覆盖在箭头之上，使障碍物更醒目）=====
            # 创建障碍物掩码并叠加到已渲染的地图上
            obstacle_mask_display = obstacle_map > 0.5
            # ⚠️ 关键修复：障碍物也需要flipud翻转，与semantic_map保持一致
            obstacle_mask_display = np.flipud(obstacle_mask_display)
            # 缩放到480x480
            obstacle_mask_display = cv2.resize(
                obstacle_mask_display.astype(np.uint8) * 255,
                (480, 480),
                interpolation=cv2.INTER_NEAREST
            ) > 127
            # 转换到旋转后的坐标系
            obstacle_mask_rotated = cv2.warpAffine(
                obstacle_mask_display.astype(np.uint8) * 255,
                rotation_matrix, (480, 480),
                flags=cv2.INTER_NEAREST
            ) > 127
            
            # ===== 🎯 可选距离计算（环视时不计算，加快速度）=====
            obstacle_distances = {}
            if calculate_distances:
                obstacle_distances = self.calculate_obstacle_distances_from_rotated_map(
                    obstacle_mask_rotated, 240, 240
                )
            
            # 用黑色覆盖障碍物区域（会覆盖箭头，使障碍物更醒目）
            global_map_with_trajectory[obstacle_mask_rotated] = [0, 0, 0]  # 黑色BGR
            global_map_rotated[obstacle_mask_rotated] = [0, 0, 0]  # 无轨迹版本也叠加
            
            # ===== 阶段6: 在显示副本上绘制Landmark标记 =====
            if len(landmarks) > 0:
                landmark_summary = {}
                for marker_x, marker_y, cls_name in landmarks:
                    # 计算像素数
                    cls_idx = detected_classes.index(cls_name)
                    semantic_channel_idx = 4 + cls_idx
                    if semantic_channel_idx < full_map.shape[0]:
                        cls_mask = full_map[semantic_channel_idx, ...] > 0.5
                        pixel_count = int(cls_mask.sum())
                        if cls_name not in landmark_summary:
                            landmark_summary[cls_name] = {'count': 0, 'total_pixels': 0}
                        landmark_summary[cls_name]['count'] += 1
                        landmark_summary[cls_name]['total_pixels'] += pixel_count
                    
                    # 转换landmark坐标到旋转后的坐标系
                    # centroids返回(cx, cy)格式，cx是列坐标(map_y方向)，cy是行坐标(map_x方向)
                    # 所以 marker_x=cx(列), marker_y=cy(行)
                    display_x = marker_x * 480 / w  # 列坐标 → display_x
                    display_y = (h - 1 - marker_y) * 480 / h  # 行坐标 → display_y（翻转）
                    point = np.array([display_x, display_y, 1])
                    rotated_point = rotation_matrix @ point
                    
                    # 绘制紫色圆球（在显示副本上）
                    cv2.circle(global_map_with_trajectory, 
                              (int(rotated_point[0]), int(rotated_point[1])), 
                              landmark_marker_radius, landmark_marker_color, -1)
                    cv2.circle(global_map_with_trajectory, 
                              (int(rotated_point[0]), int(rotated_point[1])), 
                              landmark_marker_radius, landmark_marker_border, 1)
                
                # 静默处理，不输出标注统计
            
            # ===== 阶段7: 绘制Waypoint标记（蓝色圆圈+白色数字）=====
            last_waypoint_angle = None  # 初始化
            if waypoint_positions and waypoint_ids and len(waypoint_positions) == len(waypoint_ids):
                # 静默处理waypoint渲染，不输出详细坐标
                for idx, ((wp_x, wp_y), wp_id) in enumerate(zip(waypoint_positions, waypoint_ids)):
                    # 转换waypoint坐标到旋转后的坐标系
                    display_x = wp_y * 480 / w
                    display_y = (h - 1 - wp_x) * 480 / h
                    point = np.array([display_x, display_y, 1])
                    rotated_point = rotation_matrix @ point
                    
                    # 计算最后一个waypoint的角度（相对于正前方）
                    if idx == len(waypoint_positions) - 1:
                        dx = rotated_point[0] - 240
                        dy = rotated_point[1] - 240
                        last_waypoint_angle = np.arctan2(dx, -dy)
                    
                    # 绘制蓝色圆圈（BGR=(255, 0, 0)）
                    cv2.circle(global_map_with_trajectory,
                              (int(rotated_point[0]), int(rotated_point[1])),
                              8, (255, 0, 0), -1)  # 蓝色填充
                    cv2.circle(global_map_with_trajectory,
                              (int(rotated_point[0]), int(rotated_point[1])),
                              8, (255, 255, 255), 1)  # 白色边框
                    
                    # 绘制白色数字ID
                    text = str(wp_id)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    text_x = int(rotated_point[0]) - text_width // 2
                    text_y = int(rotated_point[1]) + text_height // 2
                    cv2.putText(global_map_with_trajectory, text, (text_x, text_y),
                               font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # ===== 可选：裁剪到440×440（中心区域）=====
            # 默认关闭裁剪，保持完整的480×480地图
            if self.enable_global_map_crop:
                # 从480x480裁剪中心440x440区域
                crop_offset = (480 - 440) // 2  # = 20
                global_map_with_trajectory = global_map_with_trajectory[crop_offset:crop_offset+440, crop_offset:crop_offset+440].copy()
                global_map_rotated = global_map_rotated[crop_offset:crop_offset+440, crop_offset:crop_offset+440].copy()
                print(f"✂️  Global Map 裁剪: 480×480 → 440×440")
            else:
                print(f"📐 Global Map 尺寸: 480×480 (未裁剪，显示完整地图)")
        
        # 添加方位标签到global map
        global_map_with_trajectory = self.add_orientation_labels(global_map_with_trajectory)
        global_map_rotated = self.add_orientation_labels(global_map_rotated)
        
        # 初始化obstacle_distances和last_waypoint_angle（如果没有current_pose则无法计算）
        if 'obstacle_distances' not in locals():
            obstacle_distances = {
                'front': 'Unknown',
                'left_30': 'Unknown',
                'right_30': 'Unknown',
                'left_90': 'Unknown',
                'right_90': 'Unknown'
            }
        
        if 'last_waypoint_angle' not in locals():
            last_waypoint_angle = None
        
        # 返回：基础地图 + 显示副本（带轨迹和landmark+waypoint） + 无轨迹的旋转地图（供local_map裁剪） + 距离信息 + 最后waypoint角度
        return sem_map_vis, global_map_with_trajectory, landmarks, global_map_rotated, obstacle_distances, last_waypoint_angle
    
    def render_local_map(self, 
                        full_map: np.ndarray,
                        trajectory_points: List[Tuple[int, int]],
                        detected_classes: List[str],
                        current_pose: Tuple[float, float, float],
                        floor: Optional[np.ndarray] = None,
                        landmark_classes: Optional[List[str]] = None,
                        landmark_config: Optional[Dict] = None,
                        hfov: float = 90.0,
                        waypoint_positions: Optional[List[Tuple[int, int]]] = None,
                        waypoint_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        独立渲染局部地图（不继承全局地图，完全独立构建）
        
        注意：Local Map不渲染waypoint标记，因为action模块不需要waypoint信息
        
        Args:
            full_map: [C, H, W] 全局地图数据
            trajectory_points: [(x, y), ...] 原始轨迹坐标列表（地图像素坐标）
            detected_classes: 已检测类别列表
            current_pose: (x, y, orientation) 当前位姿（米）
            floor: [H, W] floor地图
            landmark_classes: landmark类别列表
            landmark_config: landmark配置
            hfov: 水平视野角度（默认90度）
            waypoint_positions: 未使用（保留接口兼容性）
            waypoint_ids: 未使用（保留接口兼容性）
        
        Returns:
            local_map: 局部地图 (400×400)
        """
        if full_map is None:
            return None
        
        # ===== 阶段1: 独立构建局部地图基础层 =====
        obstacle_map = full_map[0, ...]
        explored_map = full_map[1, ...]
        h, w = obstacle_map.shape
        
        # 创建语义地图
        semantic_map = np.zeros((h, w), dtype=np.uint8)
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        
        # Layer 1: 已探索自由空间（浅灰色）
        explored_free_mask = np.logical_and(explored_mask, ~obstacle_mask)
        semantic_map[explored_free_mask] = 2
        
        # Layer 2: Floor（浅绿色）
        if floor is not None:
            floor_mask = floor.astype(bool)
            floor_display_mask = np.logical_and(floor_mask, explored_mask)
            semantic_map[floor_display_mask] = 5
        
        # ===== 阶段2: PIL调色板渲染 =====
        sem_map_vis = Image.new("P", (w, h))
        sem_map_vis.putpalette(self.color_palette)
        sem_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        
        # 坐标系变换
        sem_map_vis = np.flipud(sem_map_vis)
        sem_map_vis = np.array(sem_map_vis)
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]  # RGB → BGR
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST)
        
        # ===== 阶段3: 旋转地图（Agent朝上居中）=====
        current_x, current_y, current_o = current_pose
        
        # ===== 关键：trajectory_points 现在是局部坐标（相对于 full_map）=====
        # full_map 是以 agent 为中心的裁剪区域，所以 agent 应该在 (240, 240)
        if len(trajectory_points) > 0:
            last_traj_px, last_traj_py = trajectory_points[-1]
            # trajectory_points 格式: (px, py) = (Y轴像素, X轴像素)
            # 转换到显示坐标: agent_x = py (水平), agent_y = px (垂直)
            agent_y = last_traj_px  # Y轴像素 → 垂直位置
            agent_x = last_traj_py  # X轴像素 → 水平位置
        else:
            # 回退：如果没有轨迹，agent 应该在中心
            agent_x = 240
            agent_y = 240
        
        rotation_angle = 90 - current_o
        rotation_center = (agent_x, agent_y)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1.0)
        
        # 添加平移到中心
        target_center = np.array([240, 240, 1])
        current_center = np.array([agent_x, agent_y, 1])
        rotated_center = rotation_matrix @ current_center
        translation = target_center[:2] - rotated_center[:2]
        rotation_matrix[0, 2] += translation[0]
        rotation_matrix[1, 2] += translation[1]
        
        local_map = cv2.warpAffine(sem_map_vis, rotation_matrix, (480, 480),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))
        
        # ===== 阶段4: 裁剪中心240×240区域并放大到480×480 =====
        center_x, center_y = 240, 240
        crop_size = 240
        crop_half = crop_size // 2
        
        x1 = center_x - crop_half
        x2 = center_x + crop_half
        y1 = center_y - crop_half
        y2 = center_y + crop_half
        
        local_map = local_map[y1:y2, x1:x2].copy()
        local_map = cv2.resize(local_map, (480, 480), interpolation=cv2.INTER_NEAREST)
        
        # ===== 阶段5: 独立绘制轨迹线 =====
        if len(trajectory_points) >= 2:
            local_trajectory = []
            for px, py in trajectory_points:
                # trajectory_points 现在是局部坐标: (px, py) = (Y轴像素, X轴像素)
                # 转换到显示坐标: display_x = py (水平), display_y = px (垂直)
                display_x = py  # X轴像素 → 水平位置
                display_y = px  # Y轴像素 → 垂直位置
                
                # 应用旋转变换
                point = np.array([display_x, display_y, 1])
                rotated_point = rotation_matrix @ point
                
                # 转换到local_map坐标系（裁剪区域120-360映射到0-480）
                local_x = (rotated_point[0] - 120) * 2
                local_y = (rotated_point[1] - 120) * 2
                
                if 0 <= local_x < 480 and 0 <= local_y < 480:
                    local_trajectory.append([int(round(local_x)), int(round(local_y))])
            
            # 绘制平滑轨迹线（3像素宽）
            if len(local_trajectory) >= 2:
                trajectory_array = np.array(local_trajectory, dtype=np.int32)
                cv2.polylines(local_map, [trajectory_array], isClosed=False,
                             color=(0, 165, 255), thickness=3, lineType=cv2.LINE_8)
        
        # ===== 阶段6: 绘制FOV可见区域（考虑障碍物遮挡）=====
        # 480像素 = 12m，所以1像素 = 2.5cm
        # 5米 = 500cm ÷ 2.5cm/pixel = 200像素
        fov_center_x, fov_center_y = 240, 240
        fov_radius = 200  # 5米视野半径
        
        # Agent朝上（-90度），FOV扇形中心线也朝上
        fov_center_angle = -90
        fov_start_angle = fov_center_angle - hfov / 2
        fov_end_angle = fov_center_angle + hfov / 2
        
        import math
        
        # 先获取旋转后的障碍物掩码（用于raycasting）
        obstacle_mask_flipped = np.flipud(obstacle_map > 0.5)
        obstacle_mask_resized = cv2.resize(
            obstacle_mask_flipped.astype(np.uint8) * 255,
            (480, 480),
            interpolation=cv2.INTER_NEAREST
        ) > 127
        obstacle_mask_rotated = cv2.warpAffine(
            obstacle_mask_resized.astype(np.uint8) * 255,
            rotation_matrix, (480, 480),
            flags=cv2.INTER_NEAREST
        ) > 127
        obstacle_crop = obstacle_mask_rotated[y1:y2, x1:x2]
        obstacle_local = cv2.resize(obstacle_crop.astype(np.uint8) * 255, 
                                   (480, 480), 
                                   interpolation=cv2.INTER_NEAREST) > 127
        
        # 对障碍物掩码进行形态学膨胀，填补小缺口，减少突出的射线
        kernel = np.ones((3, 3), np.uint8)
        obstacle_local_dilated = cv2.dilate(obstacle_local.astype(np.uint8), kernel, iterations=1).astype(bool)
        
        # 使用raycasting计算可见多边形
        num_rays = 180  # 每度2条射线，确保精细度
        angle_step = (fov_end_angle - fov_start_angle) / num_rays
        
        visible_points = [(fov_center_x, fov_center_y)]  # 起始点是agent位置
        
        for i in range(num_rays + 1):
            angle = fov_start_angle + i * angle_step
            angle_rad = math.radians(angle)
            
            # 沿射线方向逐步检测
            max_distance = fov_radius
            hit_obstacle = False
            ray_end_x, ray_end_y = fov_center_x, fov_center_y
            
            # 使用0.5像素步长提高检测精度
            step_size = 0.5
            num_steps = int(max_distance / step_size)
            
            for step in range(num_steps):
                distance = step * step_size
                test_x = fov_center_x + distance * math.cos(angle_rad)
                test_y = fov_center_y + distance * math.sin(angle_rad)
                
                # 检查是否越界
                if test_x < 0 or test_x >= 480 or test_y < 0 or test_y >= 480:
                    ray_end_x, ray_end_y = test_x, test_y
                    break
                
                # 检查是否碰到障碍物（使用膨胀后的障碍物掩码）
                if obstacle_local_dilated[int(test_y), int(test_x)]:
                    hit_obstacle = True
                    ray_end_x, ray_end_y = test_x, test_y
                    break
                
                # 未碰到障碍物，继续延伸
                ray_end_x, ray_end_y = test_x, test_y
            
            visible_points.append((int(ray_end_x), int(ray_end_y)))
        
        # 绘制可见区域多边形（蓝色填充，不透明）
        if len(visible_points) > 2:
            visible_polygon = np.array(visible_points, dtype=np.int32)
            
            # 直接填充蓝色（不需要透明度，因为后续会叠加障碍物、轨迹等）
            fill_color = (255, 200, 100)  # 蓝色 BGR格式，明显但不刺眼
            cv2.fillPoly(local_map, [visible_polygon], color=fill_color)
            
            # 绘制可见区域边框（深蓝色实线）
            border_color = (180, 100, 0)  # 深蓝色 BGR
            border_thickness = 2
            cv2.polylines(local_map, [visible_polygon], isClosed=True, 
                         color=border_color, thickness=border_thickness)
        
        # ===== 绘制轨迹（在FOV之上，箭头之下）=====
        if len(trajectory_points) >= 2:
            rotated_trajectory = []
            for x, y in trajectory_points:
                display_x = y * 480 / w
                display_y = (h - 1 - x) * 480 / h
                point = np.array([display_x, display_y, 1])
                rotated_point = rotation_matrix @ point
                
                # 转换到local坐标系（裁剪区域是120-360，映射到0-480）
                local_x = (rotated_point[0] - 120) * 2
                local_y = (rotated_point[1] - 120) * 2
                rotated_trajectory.append([int(round(local_x)), int(round(local_y))])
            
            if len(rotated_trajectory) >= 2:
                trajectory_array = np.array(rotated_trajectory, dtype=np.int32)
                cv2.polylines(local_map, [trajectory_array], isClosed=False,
                            color=(0, 140, 255), thickness=3, lineType=cv2.LINE_AA)  # 橙色轨迹
        
        # ===== 绘制深红色虚线指示正前方（在箭头下层）=====
        forward_line_length = 120  # 延伸120像素（约3米）
        forward_color = (0, 0, 180)  # 深红色 BGR
        forward_thickness = 2
        
        # 绘制从agent中心向正上方延伸的虚线
        start_point = (fov_center_x, fov_center_y)
        end_point = (fov_center_x, fov_center_y - forward_line_length)  # 朝上是Y减小
        
        # 虚线：每段10像素，间隙5像素
        dash_length = 10
        gap_length = 5
        total_length = forward_line_length
        num_dashes = int(total_length / (dash_length + gap_length))
        
        for i in range(num_dashes):
            dash_start_y = fov_center_y - i * (dash_length + gap_length)
            dash_end_y = dash_start_y - dash_length
            if dash_end_y < fov_center_y - forward_line_length:
                dash_end_y = fov_center_y - forward_line_length
            cv2.line(local_map, (fov_center_x, int(dash_start_y)), 
                    (fov_center_x, int(dash_end_y)), forward_color, forward_thickness)
        
        # ===== 绘制0.5m半径圆圈（深绿色，标识当前位置附近区域）=====
        # 480像素 = 12m，所以1m = 40像素，0.5m = 20像素
        nearby_radius = 20  # 0.5m半径
        nearby_color = (0, 100, 0)  # 深绿色BGR
        nearby_thickness = 2  # 2像素线宽
        cv2.circle(local_map, (fov_center_x, fov_center_y), nearby_radius, nearby_color, nearby_thickness)
        
        # ===== 阶段7: 绘制朝上的大箭头（在轨迹和虚线之上）=====
        arrow_color = (0, 0, 255)  # 亮红色BGR
        arrow_angle = np.deg2rad(-90)  # 朝上
        agent_pos = (fov_center_x, fov_center_y, arrow_angle)
        agent_arrow = vu.get_contour_points(agent_pos, origin=(0, 0), size=24)
        cv2.drawContours(local_map, [agent_arrow], 0, arrow_color, -1)
        
        # ===== 阶段8: 叠加黑色障碍物层（覆盖在箭头之上，使障碍物更醒目）=====
        # 用黑色覆盖障碍物区域（obstacle_local已在阶段6计算）
        local_map[obstacle_local] = [0, 0, 0]  # 黑色BGR
        
        # ===== 阶段9: 绘制Landmark标记（紫色圆球，最上层，不被遮挡）=====
        if landmark_classes and landmark_config:
            landmarks = self._extract_landmarks(
                full_map, detected_classes, landmark_classes,
                landmark_config['min_total_pixels'],
                landmark_config['min_area_threshold']
            )
            
            for marker_x, marker_y, cls_name in landmarks:
                # 转换landmark坐标（与全局地图坐标变换一致）
                # centroids返回(cx, cy)格式，cx是列坐标(map_y方向)，cy是行坐标(map_x方向)
                display_x = marker_x * 480 / w  # 列坐标 → display_x
                display_y = (h - 1 - marker_y) * 480 / h  # 行坐标 → display_y（翻转）
                point = np.array([display_x, display_y, 1])
                rotated_point = rotation_matrix @ point
                
                # 转换到local坐标系（裁剪区域是120-360，映射到0-480）
                local_x = (rotated_point[0] - 120) * 2
                local_y = (rotated_point[1] - 120) * 2
                
                # 只绘制在可见范围内的landmark
                if 0 <= local_x < 480 and 0 <= local_y < 480:
                    # Local map使用更大的landmark标记（10像素）
                    local_landmark_radius = 10
                    cv2.circle(local_map, 
                              (int(local_x), int(local_y)), 
                              local_landmark_radius, 
                              landmark_marker_color, -1)
                    cv2.circle(local_map, 
                              (int(local_x), int(local_y)), 
                              local_landmark_radius, 
                              landmark_marker_border, 1)
        
        # ===== 阶段10: 最终裁剪到440×440（中心区域）=====
        # 从480x480裁剪中心440x440区域
        crop_offset = (480 - 440) // 2  # = 20
        local_map_cropped = local_map[crop_offset:crop_offset+440, crop_offset:crop_offset+440].copy()
        
        # 添加方位标签
        local_map_cropped = self.add_orientation_labels(local_map_cropped)
        
        return local_map_cropped
    
    def add_orientation_labels(self, map_image: np.ndarray) -> np.ndarray:
        """
        在地图四周添加方位标签（俯视图）- 深红字+白底
        地图尺寸：440x440
        
        Args:
            map_image: 地图图像 (440, 440, 3) BGR格式
        
        Returns:
            带方位标签的地图
        """
        h, w = map_image.shape[:2]
        labeled_map = map_image.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # 加粗字体
        text_thickness = 2  # 加粗
        text_color = (0, 0, 139)  # 深红色BGR
        bg_color = (255, 255, 255)  # 白色背景
        
        # 定义方位标签
        labels = {
            'FRONT': (w // 2, 20),  # 上方
            'BACK': (w // 2, h - 8),  # 下方
            'LEFT': (45, h // 2),  # 左侧（更靠近中心，避免被遮挡）
            'RIGHT': (w - 45, h // 2)  # 右侧（更靠近中心，避免被遮挡）
        }
        
        for text, (x, y) in labels.items():
            # 计算文字大小
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
            
            # 调整位置使文字居中
            if text in ['FRONT', 'BACK']:
                text_x = x - text_width // 2
                text_y = y
            else:  # LEFT, RIGHT
                text_x = x - text_width // 2
                text_y = y + text_height // 2
            
            # 绘制白色背景矩形（底部间距更小）
            padding_top = 3
            padding_side = 3
            padding_bottom = 1
            cv2.rectangle(labeled_map,
                         (text_x - padding_side, text_y - text_height - padding_top),
                         (text_x + text_width + padding_side, text_y + baseline + padding_bottom),
                         bg_color, -1)
            
            # 绘制深红色文字
            cv2.putText(labeled_map, text, (text_x, text_y),
                       font, font_scale, text_color, text_thickness, cv2.LINE_AA)
        
        return labeled_map
    
    def render_detection_bbox(self, 
                              rgb: np.ndarray,
                              detections,  # sv.Detections object
                              labels: List[str],
                              landmark_classes: Optional[List[str]] = None,
                              mapping_classes: Optional[List[str]] = None) -> np.ndarray:
        """
        直接在RGB上渲染边界框（只标注Landmark类别）
        
        Args:
            rgb: RGB图像 (H, W, 3) BGR格式
            detections: supervision Detections对象
            labels: 标签列表 (例如: ["chair 0.85", "table 0.92"])
            landmark_classes: Landmark类别列表（只标注这些类别）
            mapping_classes: Mapping类别列表（不标注，仅用于建图）
        
        Returns:
            detection_vis: 检测可视化图像（只显示Landmark边界框）
        """
        detection_vis = rgb.copy()
        
        if detections is None or len(detections.xyxy) == 0:
            return detection_vis, []
        
        # 统计检测到的landmark
        detected_landmarks = []
        
        for i in range(len(detections.xyxy)):
            bbox = detections.xyxy[i]
            label = labels[i] if i < len(labels) else f"object_{i}"
            
            # 提取类别名和置信度
            parts = label.split()
            label_name = parts[0] if len(parts) > 0 else "unknown"
            confidence = float(parts[-1]) if len(parts) > 1 else 0.0
            
            # 只标注在landmark_classes中的类别
            is_landmark = landmark_classes and label_name in landmark_classes
            if not is_landmark:
                continue  # 跳过非Landmark类别
            
            detected_landmarks.append((label_name, confidence))
            
            # 使用醒目的黄色粗框标注Landmark
            color = detection_colors["landmark"]
            thickness = detection_thickness["landmark"]
            
            # 画边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(detection_vis, (x1, y1), (x2, y2), color, thickness)
            
            # 准备标签文本（在框内部上方显示）
            text = f"{label_name} {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # 计算文本尺寸
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # 文本位置：框内部顶端
            text_x = x1 + 5
            text_y = y1 + text_h + 5
            
            # 画黄色背景（填充）
            cv2.rectangle(detection_vis, 
                         (text_x - 2, text_y - text_h - 2), 
                         (text_x + text_w + 2, text_y + baseline + 2), 
                         color, -1)
            
            # 画黑色文字（在黄色背景上清晰可见）
            cv2.putText(detection_vis, text, 
                       (text_x, text_y),
                       font, font_scale, (0, 0, 0), font_thickness)
        
        # 返回检测可视化和检测到的landmark列表
        return detection_vis, detected_landmarks
    
    # ========== 保存方法 ==========
    
    def save_rgb(self, step: int, episode_id: int, rgb: np.ndarray, phase: str = "action", controller = None) -> str:
        """
        保存原始RGB帧（添加距离线）
        
        Args:
            step: 步数
            episode_id: episode ID
            rgb: RGB图像 (H, W, 3) BGR格式
            phase: 阶段标识 ("initial", "action1a", "verify1a" 等)
            controller: VLMNavigationController实例（用于访问_draw_distance_rays_on_first_person_view）
        
        Returns:
            save_path: 保存路径
        """
        # 如果是action阶段且提供了controller，绘制距离线
        if phase.startswith('action') and controller is not None:
            if hasattr(controller, '_draw_distance_rays_on_first_person_view') and hasattr(controller, 'latest_obstacle_distances'):
                rgb = controller._draw_distance_rays_on_first_person_view(rgb.copy(), controller.latest_obstacle_distances)
        
        episode_dir = self._create_episode_directories(episode_id)
        save_path = os.path.join(episode_dir, 'rgb', f'step_{step:04d}_{phase}.png')
        cv2.imwrite(save_path, rgb)
        return save_path
    
    def draw_floor_from_saved_mask(self, image: np.ndarray, mask_path: str, classes: List[str]) -> np.ndarray:
        """
        使用保存的semantic mask绘制地面分割（直接使用原始检测的floor mask）
        
        Args:
            image: 图像 (H, W, 3) BGR格式
            mask_path: semantic mask的numpy文件路径
            classes: 类别列表（用于查找floor索引）
            
        Returns:
            绘制了地面分割的图像
        """
        try:
            if not os.path.exists(mask_path):
                print(f"  ⚠️  Mask file not found: {mask_path}")
                return image
                
            masks = np.load(mask_path)
            floor_idx = None
            for i, cls in enumerate(classes):
                if cls.lower() == 'floor':
                    floor_idx = i
                    break
            
            if floor_idx is None:
                print(f"  ⚠️  'floor' not found in classes: {classes}")
                return image
            
            if floor_idx >= masks.shape[0]:
                print(f"  ⚠️  floor_idx {floor_idx} >= masks.shape[0] {masks.shape[0]}")
                return image
            
            floor_mask = masks[floor_idx]
            
            # 增强可见性：更明显的绿色覆盖
            overlay = image.copy()
            green_color = np.array([0, 255, 0], dtype=np.uint8)  # 纯绿色
            floor_bool = floor_mask > 0.1
            
            # 如果mask有效像素太少，打印警告
            if np.sum(floor_bool) < 100:
                print(f"  ⚠️  Floor mask has too few pixels: {np.sum(floor_bool)}")
                return image
            
            # 绘制半透明绿色覆盖
            overlay[floor_bool] = overlay[floor_bool] * 0.6 + green_color * 0.4
            alpha = 0.7  # 增加透明度，让绿色更明显
            result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            # Floor mask应用成功（静默）
            return result
        except Exception as e:
            # 静默失败处理
            return image
    
    def draw_distance_on_view(self, image: np.ndarray, distance_str: str) -> np.ndarray:
        """
        在视图上绘制距离信息（梯形线条 - 用于thinking模式12个方向view）
        
        Args:
            image: 图像 (H, W, 3) BGR格式
            distance_str: 距离字符串
        """
        h, w = image.shape[:2]
        center_x = w // 2
        bottom_y = h - 5
        side_offset = int(w * 0.25)  # 增大两侧宽度：0.15 → 0.25
        
        if "WARNING" in distance_str or "<0.5" in distance_str:
            color, line_ratio, top_shrink = (0, 0, 255), 0.15, 0.8  # 红色：只延伸一点点，顶部收缩到0.8
        elif ">2.0" in distance_str or "open" in distance_str:
            color, line_ratio, top_shrink = (0, 255, 0), 0.65, 0.3  # 绿色：降到之前黄色位置，顶部收缩到0.3（最窄）
        else:
            color, line_ratio, top_shrink = (0, 255, 255), 0.4, 0.5  # 黄色：再低一点，顶部收缩到0.5（中等）
        
        max_length = bottom_y - h // 2
        end_y = bottom_y - int(max_length * line_ratio)
        
        cv2.line(image, (center_x, bottom_y), (center_x, end_y), color, 3)
        cv2.line(image, (center_x - side_offset, bottom_y), (center_x - int(side_offset * top_shrink), end_y), color, 2)
        cv2.line(image, (center_x + side_offset, bottom_y), (center_x + int(side_offset * top_shrink), end_y), color, 2)
        
        text_x = center_x + 10
        text_y = (bottom_y + h // 2) // 2
        font_scale, thickness = 0.6, 2
        text_size = cv2.getTextSize(distance_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cv2.rectangle(image, (text_x - 2, text_y - text_size[1] - 1),
                     (text_x + text_size[0] + 2, text_y + 2), (0, 0, 0), -1)
        cv2.putText(image, distance_str, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return image
    
    def draw_distance_on_action_view(self, image: np.ndarray, distance_dict: Dict[str, str]) -> np.ndarray:
        """
        在Action模式视图上绘制7个方向的距离信息（从底部中心引出）
        
        Args:
            image: 图像 (H, W, 3) BGR格式
            distance_dict: 距离字典，key为方向（'front', 'left_30', 'right_30', 'left_90', 'right_90'等）
        """
        h, w = image.shape[:2]
        center_x = w // 2
        bottom_y = h - 10
        
        # 7个方向：左90, 左60, 左30, 前, 右30, 右60, 右90
        # 对应的像素角度（从底部向上，-90°是正上方）
        direction_configs = [
            {'key': 'left_90', 'angle': -180, 'label': 'Left 90'},
            {'key': 'left_60', 'angle': -150, 'label': 'Left 60'},
            {'key': 'left_30', 'angle': -120, 'label': 'Left 30'},
            {'key': 'front', 'angle': -90, 'label': 'FRONT'},
            {'key': 'right_30', 'angle': -60, 'label': 'Right 30'},
            {'key': 'right_60', 'angle': -30, 'label': 'Right 60'},
            {'key': 'right_90', 'angle': 0, 'label': 'Right 90'}
        ]
        
        for config in direction_configs:
            dist_str = distance_dict.get(config['key'], 'Unknown')
            if dist_str == 'Unknown':
                continue
            
            # 根据距离确定颜色和长度（FRONT线条更长）
            if "WARNING" in dist_str or "<0.5" in dist_str:
                color = (0, 0, 255)  # 红色
                line_length = 65 if config['key'] == 'front' else 60
            elif ">2.0" in dist_str or "open" in dist_str:
                color = (0, 255, 0)  # 绿色
                line_length = 140 if config['key'] == 'front' else 120
            else:
                color = (0, 255, 255)  # 黄色
                line_length = 105 if config['key'] == 'front' else 90
            
            # 计算终点
            angle_rad = np.deg2rad(config['angle'])
            end_x = int(center_x + line_length * np.cos(angle_rad))
            end_y = int(bottom_y + line_length * np.sin(angle_rad))
            
            # 绘制线条（中心线粗一点）
            thickness = 3 if config['key'] == 'front' else 2
            cv2.line(image, (center_x, bottom_y), (end_x, end_y), color, thickness)
            
            # FRONT用大字号，其他用稍大字号（0.4→0.5）
            font_scale = 0.6 if config['key'] == 'front' else 0.5
            font_thickness = 2 if config['key'] == 'front' else 1
            
            # 合并标签为单行："Left 90 1.3m" 或 "FRONT 0.70m"
            combined_label = f"{config['label']} {dist_str}"
            label_size = cv2.getTextSize(combined_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # 标签位置：从线条终点沿方向延伸
            label_offset = 25
            base_x = int(end_x + label_offset * np.cos(angle_rad))
            base_y = int(end_y + label_offset * np.sin(angle_rad))
            
            # 根据方向调整标签位置，使其向两侧延伸，远离中心
            if config['key'] == 'front':
                # FRONT标签居中
                text_x = base_x - label_size[0] // 2
                text_y = base_y + label_size[1] // 2
            elif config['key'] in ['left_30', 'left_60', 'left_90']:
                # 左侧标签：向左延伸，右对齐（文字在线条左侧）
                side_offset = 15 if config['key'] in ['left_30', 'left_60'] else 0
                text_x = base_x - label_size[0] - side_offset
                text_y = base_y + label_size[1] // 2
            else:  # right_30, right_60, right_90
                # 右侧标签：向右延伸，左对齐（文字在线条右侧）
                side_offset = 15 if config['key'] in ['right_30', 'right_60'] else 0
                text_x = base_x + side_offset
                text_y = base_y + label_size[1] // 2
            
            # 绘制黑色背景和文字
            cv2.rectangle(image, (text_x - 2, text_y - label_size[1] - 2),
                         (text_x + label_size[0] + 2, text_y + 2), (0, 0, 0), -1)
            cv2.putText(image, combined_label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        
        return image
    
    def prepare_action_image_with_enhancements(self, image_path: str, mask_path: str = None, 
                                               distance_dict: Dict[str, str] = None, classes: List[str] = None,
                                               use_floor: bool = True, use_distance: bool = True) -> str:
        """
        为action模式准备增强图像：添加地面分割（绿色）和7方向距离辅助线
        
        Args:
            image_path: 原始图像路径
            mask_path: semantic mask路径
            distance_dict: 距离字典 {'front': 'X.XXm', 'left_30': 'X.XXm', ...}
            classes: 类别列表
            use_floor: 是否绘制地面分割
            use_distance: 是否绘制距离辅助线
            
        Returns:
            增强后的图像路径
        """
        if not os.path.exists(image_path):
            return image_path
        
        image = cv2.imread(image_path)
        if image is None:
            return image_path
        
        if use_floor and mask_path and os.path.exists(mask_path) and classes:
            image = self.draw_floor_from_saved_mask(image, mask_path, classes)
        
        if use_distance and distance_dict:
            image = self.draw_distance_on_action_view(image, distance_dict)
        
        base_path = os.path.splitext(image_path)[0]
        enhanced_path = f"{base_path}_enhanced.png"
        cv2.imwrite(enhanced_path, image)
        return enhanced_path
    
    def save_global_map(self, 
                       step: int,
                       episode_id: int,
                       global_map: np.ndarray,
                       phase: str = "action") -> str:
        """
        保存全局地图（添加标签）
        
        Args:
            step: 步数
            episode_id: episode ID
            global_map: 旋转后的全局地图 (480×480)
            phase: 阶段标识 ("initial", "action1a", "verify1a" 等)
        
        Returns:
            save_path: 保存路径
        """
        if global_map is None:
            return None
        
        # 添加Global Map标签（不显示IMAGE编号）
        label_text = "Global Map"
        
        # 创建白色标签背景（高度40像素）
        label_height = 40
        label_bg = np.ones((label_height, global_map.shape[1], 3), dtype=np.uint8) * 255
        
        # 绘制红色文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0  # 增大字体
        font_thickness = 2  # 保持加粗
        text_color = (0, 0, 255)  # BGR: 红色
        
        # 计算文字位置（居中）
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        text_x = (label_bg.shape[1] - text_size[0]) // 2
        text_y = (label_height + text_size[1]) // 2
        
        # 在标签背景上绘制文字
        cv2.putText(label_bg, label_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
        
        # 垂直拼接：地图在上，标签在下
        labeled_map = np.vstack([global_map, label_bg])
        
        # 保存带标签的地图
        episode_dir = self._create_episode_directories(episode_id)
        save_path = os.path.join(episode_dir, 'global_map', f'step_{step:04d}_{phase}.png')
        cv2.imwrite(save_path, labeled_map, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return save_path
    
    def save_local_map(self,
                      step: int,
                      episode_id: int,
                      local_map: np.ndarray,
                      phase: str = "action") -> str:
        """
        保存局部地图（添加标签）
        
        Args:
            step: 步数
            episode_id: episode ID
            local_map: 局部地图 (400×400)
            phase: 阶段标识 ("initial", "action1a", "verify1a" 等)
        
        Returns:
            save_path: 保存路径
        """
        if local_map is None:
            return None
        
        # 添加Local Map标签（不显示IMAGE编号）
        label_text = "Local Map"
        
        # 创建白色标签背景（高度40像素）
        label_height = 40
        label_bg = np.ones((label_height, local_map.shape[1], 3), dtype=np.uint8) * 255
        
        # 绘制红色文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # 增大字体
        font_thickness = 2  # 加粗
        text_color = (0, 0, 255)  # BGR: 红色
        
        # 计算文字位置（居中）
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        text_x = (label_bg.shape[1] - text_size[0]) // 2
        text_y = (label_height + text_size[1]) // 2
        
        # 在标签背景上绘制文字
        cv2.putText(label_bg, label_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
        
        # 垂直拼接：地图在上，标签在下
        labeled_map = np.vstack([local_map, label_bg])
        
        # 保存带标签的地图
        episode_dir = self._create_episode_directories(episode_id)
        save_path = os.path.join(episode_dir, 'local_map', f'step_{step:04d}_{phase}.png')
        cv2.imwrite(save_path, labeled_map, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return save_path
    
    def save_detection(self,
                      step: int,
                      episode_id: int,
                      detection_vis: np.ndarray,
                      phase: str = "action") -> str:
        """
        保存检测可视化
        
        Args:
            step: 步数
            episode_id: episode ID
            detection_vis: 检测可视化图像
            phase: 阶段标识 ("initial", "action1a", "verify1a" 等)
        
        Returns:
            save_path: 保存路径
        """
        if detection_vis is None:
            return None
        
        # 简化路径：data/manual_navigation/episode_X/detection/step_XXXX.png
        episode_dir = self._create_episode_directories(episode_id)
        save_path = os.path.join(episode_dir, 'detection', f'step_{step:04d}_{phase}.png')
        cv2.imwrite(save_path, detection_vis)
        return save_path
    
    # ========== 一键保存方法 ==========
    
    def save_step_visualization(self, 
                               step: int,
                               episode_id: int,
                               rgb: np.ndarray,
                               full_map: np.ndarray,
                               trajectory_points: List[Tuple[int, int]],
                               detected_classes: List[str],
                               current_pose: Tuple[float, float, float],
                               floor: Optional[np.ndarray] = None,
                               hfov: float = 90.0,
                               detections=None,  # sv.Detections对象（新）
                               labels: Optional[List[str]] = None,
                               landmark_classes: Optional[List[str]] = None,
                               mapping_classes: Optional[List[str]] = None,  # 新增
                               landmark_config: Optional[Dict] = None,
                               waypoint_positions: Optional[List[Tuple[int, int]]] = None,
                               waypoint_ids: Optional[List[int]] = None,
                               masks: Optional[np.ndarray] = None,
                               phase: str = "action",
                               global_trajectory_points: Optional[List[Tuple[int, int]]] = None,
                               controller = None,
                               calculate_distances: bool = False) -> Tuple[Dict[str, str], List, Dict[str, str], Optional[float]]:  # 兼容旧参数
        """
        一键保存当前步骤的所有可视化（支持新detection渲染 + 平滑轨迹线 + waypoint标记）
        
        Args:
            trajectory_points: [(x, y), ...] 当前子任务轨迹（用于local map）
            global_trajectory_points: [(x, y), ...] 完整导航历史轨迹（用于global map，可选）
                - 如果提供，global map显示此轨迹
                - 如果未提供，global map回退使用trajectory_points（向后兼容）
            floor: [H, W] floor地图（通过形态学方法计算，像ZS_Evaluator）
            detections: supervision Detections对象（优先使用）
            masks: 检测掩码（向后兼容，已废弃）
            mapping_classes: Mapping类别列表
            landmark_classes: Landmark类别列表
            waypoint_positions: [(map_x, map_y), ...] waypoint位置列表（可选，从mapper.get_waypoints()获取）
            waypoint_ids: [1, 2, 3, ...] waypoint ID列表（可选，从mapper.get_waypoints()获取）
            phase: 阶段标识 ("initial", "action1a", "verify1a" 等)
            controller: VLMNavigationController实例（用于绘制距离线）
        
        Returns:
            (paths, landmarks, obstacle_distances, last_waypoint_angle)
            - paths: 保存路径字典 {'rgb', 'global_map', 'local_map', 'detection'}
            - landmarks: Landmark列表
            - obstacle_distances: {'front': "X.XXm", 'left_30': ..., ...} 5方向距离
            - last_waypoint_angle: 最后一个waypoint相对于正前方的角度（弧度），None表示无waypoint
            
        注意:
        1. floor通过形态学方法计算（像ZS_Evaluator._process_map）
        2. waypoint数据建议直接从mapper.get_waypoints()传入，无需手动管理
        """
        paths = {}
        
        # 1. 保存RGB（传入controller用于绘制距离线）
        paths['rgb'] = self.save_rgb(step, episode_id, rgb, phase, controller)
        
        # 2. 渲染并保存全局地图（使用global_trajectory_points或回退到trajectory_points）
        global_traj_to_use = global_trajectory_points if global_trajectory_points is not None else trajectory_points
        _, global_map_with_trajectory, landmarks, global_map_clean, obstacle_distances, last_waypoint_angle = self.render_global_map(
            full_map, global_traj_to_use, detected_classes, floor,
            current_pose, landmark_classes, landmark_config,
            waypoint_positions, waypoint_ids, calculate_distances
        )
        paths['global_map'] = self.save_global_map(step, episode_id, global_map_with_trajectory, phase)
        
        # 3. 渲染并保存局部地图（使用trajectory_points）
        local_map = self.render_local_map(
            full_map, trajectory_points, detected_classes, current_pose,
            floor, landmark_classes, landmark_config, hfov,
            waypoint_positions, waypoint_ids
        )
        paths['local_map'] = self.save_local_map(step, episode_id, local_map, phase)        # 4. 渲染并保存检测结果
        detected_landmarks_step = []
        if detections is not None and labels is not None:
            detection_vis, detected_landmarks_step = self.render_detection_bbox(
                rgb, detections, labels, 
                landmark_classes, mapping_classes
            )
            paths['detection'] = self.save_detection(step, episode_id, detection_vis, phase)
        
        # 5. 保存semantic masks（用于action模式的地面分割）
        if masks is not None:
            paths['masks'] = self.save_semantic_masks(step, episode_id, masks, phase)
        
        return paths, detected_landmarks_step, obstacle_distances, last_waypoint_angle
    
    # ========== 辅助方法 ==========
    
    def _extract_landmarks(self,
                          full_map: np.ndarray,
                          detected_classes: List[str],
                          landmark_classes: List[str],
                          min_total_pixels: int,
                          min_area_threshold: int) -> List[Tuple[int, int, str]]:
        """提取landmark标记位置
        
        流程：
        1. 遍历landmark_classes（如cabinet）
        2. 检查是否在detected_classes中
        3. 计算语义通道索引：semantic_channel_idx = 4 + detected_classes.index(cls_name)
        4. 从full_map[semantic_channel_idx]提取mask
        5. 形态学闭运算：填补间隙，合并相近区域
        6. 连通域分析，过滤面积 < min_area_threshold
        7. 空间合并（距离 < landmark_merge_distance）
        
        Args:
            min_total_pixels: 总像素数阈值（已弃用，为兼容保留参数）
            min_area_threshold: 单个连通域最小面积
        
        Returns:
            List of (cx, cy, class_name)
        """
        if not landmark_classes or len(detected_classes) == 0:
            return []
        
        spatial_regions = {}
        landmark_found = False
        
        for cls_name in landmark_classes:
            if cls_name not in detected_classes:
                continue
            
            cls_idx = detected_classes.index(cls_name)
            semantic_channel_idx = 4 + cls_idx
            
            if semantic_channel_idx >= full_map.shape[0]:
                continue
            
            cls_mask = full_map[semantic_channel_idx, ...] > 0.5
            num_pixels = cls_mask.sum()
            
            # 移除min_total_pixels检查，允许标记远处小像素区域
            if num_pixels == 0:
                continue
            
            # 形态学闭运算：填补间隙，合并相近区域
            # 使用7×7核，可以填补距离3-4像素的间隙，合并被Winner-Takes-All分割的区域
            cls_mask_uint8 = cls_mask.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cls_mask_closed = cv2.morphologyEx(cls_mask_uint8, cv2.MORPH_CLOSE, kernel)
            
            # 连通性分析（在闭运算后的mask上）
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cls_mask_closed, connectivity=8)
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                
                if area < min_area_threshold:
                    continue
                
                # 检查空间合并（使用constant.py配置的距离）
                from vlnce_baselines.config_system.constants import landmark_merge_distance
                merged = False
                for existing_pos in list(spatial_regions.keys()):
                    ex_cx, ex_cy = existing_pos
                    dist = np.sqrt((cx - ex_cx)**2 + (cy - ex_cy)**2)
                    if dist < landmark_merge_distance:
                        spatial_regions[existing_pos].append((area, cls_name))
                        merged = True
                        break
                
                if not merged:
                    spatial_regions[(cx, cy)] = [(area, cls_name)]
                    if not landmark_found:
                        landmark_found = True
        landmarks = []
        for (cx, cy), candidates in spatial_regions.items():
            candidates.sort(key=lambda x: x[0], reverse=True)
            dominant_class = candidates[0][1]
            area = candidates[0][0]
            landmarks.append((cx, cy, dominant_class))
            print(f"  📍 {dominant_class} @({cx},{cy}) - {area}px")
        
        return landmarks
    
    def save_semantic_masks(self, step: int, episode_id: int, masks: np.ndarray, phase: str = "action") -> str:
        """
        保存semantic masks到numpy文件
        
        Args:
            step: 当前步数
            episode_id: episode ID
            masks: semantic masks [num_classes, H, W]
            phase: 阶段标识
            
        Returns:
            保存路径
        """
        episode_dir = self._create_episode_directories(episode_id)
        masks_dir = os.path.join(episode_dir, 'semantic_masks')
        os.makedirs(masks_dir, exist_ok=True)
        
        save_path = os.path.join(masks_dir, f'step_{step:04d}_{phase}.npy')
        np.save(save_path, masks)
        
        return save_path


# ========== 便捷函数 ==========

def create_visualizer(results_dir: str, 
                     resolution: int = 5,
                     map_shape: Tuple[int, int] = (480, 480),
                     enable_global_map_crop: bool = False,
                     enable_adaptive_zoom: bool = False) -> MapVisualizer:
    """创建MapVisualizer实例"""
    return MapVisualizer(results_dir, resolution, map_shape, 
                        enable_global_map_crop, enable_adaptive_zoom)
