"""
VLM Navigation Controller
=========================
基于VLM的自动导航控制器

继承InteractiveNavigationController的核心功能：
- 语义建图（GroundedSAM + Semantic Mapping）
- 可视化（MapVisualizer）
- 12步×30°环视建图

新增VLM功能：
- LLM高层规划（生成子任务）
- VLM低层动作执行（基于RGB+地图决策）
- 4方向观察收集（前/右/后/左）
- RGB+俯视图拼接可视化（使用环境提供的top_down_map_vlnce）
- 结果保存供后续测评
"""
import os
import cv2
import json
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from habitat import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from vlnce_baselines.interactive_navigation_controller import InteractiveNavigationController
from vlnce_baselines.vlm import (
    LLMPlanner, ActionExecutor, SaveManager, NavigationVisualizer
)
from vlnce_baselines.visualization import PanoramaGenerator
from vlnce_baselines.vlm.navigation_config import (
    DIRECTION_STEPS, DIRECTION_NAMES, DIRECTION_CONFIG, ACTION_MAPPING
)
from habitat_extensions.pose_utils import get_sim_location


class VLMNavigationController(InteractiveNavigationController):
    """
    VLM导航控制器
    
    继承自InteractiveNavigationController，添加VLM规划和执行功能
    
    工作流程：
    1. 初始环视建图（12步×30°）→ 收集4方向图像
    2. LLM规划 → 生成初始子任务
    3. VLM执行 → 循环执行动作直到子任务完成
    4. 验证环视建图（12步×30°）→ 更新地图和4方向图像
    5. 验证重规划 → 检查完成状态，生成下一子任务
    6. 重复3-5直到导航完成
    
    注意：每次验证重规划前都会执行360°环视，以更新语义地图和当前位置的4方向观察
    """
    
    def __init__(self, config: Config,
                 llm_config_path: str = "vlnce_baselines/vlm/llm_config.yaml",
                 vlm_config_path: str = "vlnce_baselines/vlm/vlm_config.yaml"):
        """
        初始化VLM导航控制器
        
        Args:
            config: Habitat配置
            llm_config_path: LLM配置文件路径
            vlm_config_path: VLM配置文件路径
        """
        # 调用父类初始化（初始化环境、检测、建图、可视化）
        super().__init__(config)
        
        # 初始化VLM模块
        print("\n[Init] 初始化VLM模块...")
        
        # 获取动作参数
        self.turn_angle = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE  # 30°
        self.move_distance = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE  # 0.25m
        
        # 动作空间描述
        self.action_space = f"MOVE_FORWARD ({self.move_distance}m), TURN_LEFT ({self.turn_angle}°), TURN_RIGHT ({self.turn_angle}°), STOP"
        
        # 初始化LLM规划器
        try:
            self.planner = LLMPlanner(llm_config_path, self.action_space)
        except Exception as e:
            print(f"⚠️  LLM Planner初始化失败: {e}")
            self.planner = None
        
        # 初始化VLM执行器
        try:
            self.action_executor = ActionExecutor(vlm_config_path, self.turn_angle, self.move_distance)
        except Exception as e:
            print(f"⚠️  Action Executor初始化失败: {e}")
            self.action_executor = None
        
        # VLM状态
        self.current_subtask = None
        self.subtask_count = 0
        self.subtask_attempt = 0  # 当前子任务的尝试次数（a, b, c...）
        self.progress_summary = ""
        self.previous_action_reason = ""  # 上一步的action_analysis
        self.subtask_history = []
        self.current_subtask_file = None
        
        # 初始化管理器
        self.save_manager = None  # 在reset_episode时初始化
        # waypoint_manager已废弃，直接使用mapper.add_waypoint()
        
        # 观察缓存
        self.latest_obs = None  # 缓存最新的观察
        self.latest_info = None  # 缓存最新的info（包含top_down_map_vlnce）
        self.pose_before_action = None  # 记录动作前的pose (x, y, orientation)
        
        # 观察缓存（环视时收集的4方向图像）
        self.direction_images = {}  # {direction_name: image_path}
        self.latest_map_image = None
        
        # 障碍物距离缓存
        # Thinking模式（环视）：12个方向（360°每30°）
        self.latest_obstacle_distances_12 = {
            f'angle_{i}': 'Unknown' for i in range(0, 360, 30)
        }
        # Action模式：7个方向（前方扇形）
        self.latest_obstacle_distances = {
            'front': 'Unknown',
            'left_30': 'Unknown',
            'left_60': 'Unknown',
            'left_90': 'Unknown',
            'right_30': 'Unknown',
            'right_60': 'Unknown',
            'right_90': 'Unknown'
        }
        
        # NavigationVisualizer（用于RGB+俯视图拼接和GIF生成）
        self.nav_visualizer = None
        
        # PanoramaGenerator（用于全景图拼接和标注）
        self.panorama_generator = PanoramaGenerator()
        
        print("[Init] VLM模块初始化完成\n")
    
    def reset_episode(self, episode_id: int = None):
        """重置Episode，包括VLM状态"""
        # 清理之前episode的输出目录
        if episode_id is not None:
            import shutil
            old_episode_dir = os.path.join(self.config.RESULTS_DIR, f'episode_{episode_id}')
            if os.path.exists(old_episode_dir):
                print(f"[Reset] 清理旧数据: {old_episode_dir}")
                shutil.rmtree(old_episode_dir)
        
        # 调用父类重置
        super().reset_episode(episode_id)
        
        # 初始化SaveManager（使用RESULTS_DIR作为输出根目录）
        self.save_manager = SaveManager(self.config.RESULTS_DIR, self.current_episode_id)
        
        # 重置VLM状态
        self.current_subtask = None
        self.subtask_count = 0
        self.subtask_attempt = 0  # 重置尝试计数
        self.progress_summary = ""
        self.previous_action_reason = ""  # 重置上一步action reason
        self.subtask_history = []
        self.current_subtask_file = None
        self.direction_images = {}
        self.latest_map_image = None
        self.pose_before_action = None  # 重置pose追踪
        self.last_planned_degrees = 0  # 记录计划转向角度
        self.last_planned_meters = 0   # 记录计划移动距离
        self.last_action_name = ""      # 记录上次动作名称
        
        # waypoint已集成到mapper中，mapper.reset()会自动清空
        
        print(f"[Reset] Episode {self.current_episode_id} 重置完成")
        
        # 初始化NavigationVisualizer（用于RGB+俯视图拼接和GIF生成）
        visualization_dir = os.path.join(self.episode_dir, 'visualization')
        self.nav_visualizer = NavigationVisualizer(visualization_dir)
        self.nav_visualizer.setup_maps_dir(self.episode_dir)
        
        # 初始化输出记录列表
        self.thinking_outputs = []  # 记录LLM(thinking)的所有输出
        self.action_outputs = []    # 记录VLM(action)的所有输出
    
    @property
    def episode_dir(self) -> str:
        """获取当前episode的输出目录（动态属性，自动根据current_episode_id生成）"""
        return os.path.join(self.config.RESULTS_DIR, f'episode_{self.current_episode_id}')
    
    def _get_agent_pose(self) -> tuple:
        """获取agent当前pose (x, y, orientation)
        
        Returns:
            tuple: (x, y, o) where x, y are coordinates and o is orientation in radians
        """
        # 通过call_at调用environment 0的get_agent_pose方法
        return self.envs.call_at(0, "get_agent_pose")
    
    def _draw_navigable_area_on_view(self, image: np.ndarray, view_angle: float) -> np.ndarray:
        """
        在方向视图上绘制可导航区域（绿色覆盖层）
        
        Args:
            image: 当前方向的图像 (H, W, 3) BGR格式
            view_angle: 当前视角的角度 (0-330, 30度递增)
            
        Returns:
            绘制了导航区域后的图像
        """
        if not hasattr(self, 'mapper') or self.mapper is None:
            return image
        
        # 获取当前agent位置和朝向
        agent_x, agent_y, agent_o = self._get_agent_pose()
        
        # 计算当前视角的绝对方向
        view_offset_rad = np.deg2rad(view_angle)
        view_direction = agent_o + view_offset_rad
        
        # 相机FOV是79度
        camera_fov_half = np.deg2rad(79 / 2)
        
        h, w = image.shape[:2]
        
        # 创建半透明绿色覆盖层
        overlay = image.copy()
        
        # 从地图获取可导航区域信息
        # 使用mapper的occupancy grid判断可导航性
        if hasattr(self.mapper, 'occupancy_grid'):
            # 投影可导航点到图像上
            max_distance = 5.0  # 最远检测5米
            num_rays = 40  # 检测射线数量（覆盖79度FOV）
            
            # 生成检测射线（在79度FOV内均匀分布）
            for i in range(num_rays):
                # 射线角度：从view_direction - fov_half 到 view_direction + fov_half
                ray_ratio = i / (num_rays - 1) if num_rays > 1 else 0.5
                ray_angle = view_direction - camera_fov_half + ray_ratio * 2 * camera_fov_half
                
                # 沿射线检测可导航区域
                navigable_points = []
                for dist in np.linspace(0.5, max_distance, 20):
                    # 计算世界坐标
                    world_x = agent_x + dist * np.cos(ray_angle)
                    world_y = agent_y + dist * np.sin(ray_angle)
                    
                    # 转换到地图坐标
                    resolution = self.mapper.resolution / 100.0
                    map_shape = self.mapper.map_shape
                    map_min_x = - (map_shape[1] // 2) * resolution
                    map_min_y = - (map_shape[0] // 2) * resolution
                    map_x = int((world_x - map_min_x) / resolution)
                    map_y = int((world_y - map_min_y) / resolution)
                    
                    # 检查是否在地图范围内
                    if 0 <= map_x < self.mapper.occupancy_grid.shape[1] and \
                       0 <= map_y < self.mapper.occupancy_grid.shape[0]:
                        # 0=未知, 1=可导航, 2=障碍物
                        if self.mapper.occupancy_grid[map_y, map_x] == 1:
                            navigable_points.append((dist, ray_angle))
                        elif self.mapper.occupancy_grid[map_y, map_x] == 2:
                            # 遇到障碍物，停止这条射线
                            break
                    else:
                        break
                
                # 在图像上绘制可导航点
                for dist, ray_angle in navigable_points:
                    # 计算相对于视角中心的角度差
                    angle_diff = ray_angle - view_direction
                    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                    
                    # X坐标映射
                    x_ratio = (angle_diff + camera_fov_half) / (2 * camera_fov_half)
                    x_pos = int(x_ratio * w)
                    x_pos = max(0, min(w - 1, x_pos))
                    
                    # Y坐标映射（距离 → 垂直位置）
                    if dist < 1.0:
                        y_pos = int(h * 0.85)  # 近处，下方
                    elif dist < 2.0:
                        y_pos = int(h * 0.75)
                    elif dist < 3.0:
                        y_pos = int(h * 0.65)
                    elif dist < 4.0:
                        y_pos = int(h * 0.55)
                    else:
                        y_pos = int(h * 0.45)  # 远处，上方
                    
                    y_pos = max(0, min(h - 1, y_pos))
                    
                    # 绘制绿色半透明点（表示可导航区域）
                    cv2.circle(overlay, (x_pos, y_pos), 3, (0, 255, 0), -1)
        
        # 混合原图和覆盖层（30%透明度）
        alpha = 0.3
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        return result
    
    def _draw_waypoints_on_view(self, image: np.ndarray, view_angle: float, waypoint_info: tuple) -> np.ndarray:
        """
        在环视方向视图上绘制waypoint标记
        
        简化版本：调用此函数时已经确认waypoint在当前视图的±15°内
        直接在图像中心绘制waypoint标记即可
        
        Args:
            image: 当前方向的图像 (H, W, 3) BGR格式
            view_angle: 当前视角的角度 (0-330, 30度递增，逆时针)
            waypoint_info: (waypoint_positions, waypoint_ids, descriptions) from mapper.get_waypoints()
            
        Returns:
            绘制了waypoint标记后的图像
        """
        if not waypoint_info or len(waypoint_info[0]) == 0:
            return image
        
        waypoint_positions, waypoint_ids, waypoint_descriptions = waypoint_info
        
        # 获取最后一个waypoint
        if len(waypoint_positions) > 0:
            last_idx = len(waypoint_positions) - 1
            wp_id = waypoint_ids[last_idx]
            wp_desc = waypoint_descriptions[last_idx] if len(waypoint_descriptions) > last_idx else ""
            
            # 绘制在图像中心位置
            h, w = image.shape[:2]
            x_pos = w // 2
            y_pos = h // 2
            
            # 绘制waypoint标记（蓝色外圈 + 白色填充）
            cv2.circle(image, (x_pos, y_pos), 20, (255, 0, 0), 3)  # 蓝色边框，半径20
            cv2.circle(image, (x_pos, y_pos), 17, (255, 255, 255), -1)  # 白色填充
            
            # 绘制waypoint ID（红色粗体）
            text = f"{wp_id}"
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = x_pos - text_size[0] // 2
            text_y = y_pos + text_size[1] // 2
            cv2.putText(image, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
            
            # 在waypoint下方绘制描述（如果有）
            if wp_desc:
                desc_font_scale = 0.5
                desc_thickness = 1
                desc_y = y_pos + 35
                desc_size = cv2.getTextSize(wp_desc, cv2.FONT_HERSHEY_SIMPLEX, desc_font_scale, desc_thickness)[0]
                desc_x = x_pos - desc_size[0] // 2
                cv2.putText(image, wp_desc, (desc_x, desc_y),
                           cv2.FONT_HERSHEY_SIMPLEX, desc_font_scale, (0, 0, 255), desc_thickness)
        
        return image
        
        # 获取当前agent位置和朝向（世界坐标系）
        agent_x, agent_y, agent_o = self._get_agent_pose()
        
        # 计算当前视角的绝对方向（世界坐标系）
        # view_angle: 相对于agent朝向的角度（度），逆时针为正
        view_offset_rad = np.deg2rad(view_angle)
        view_direction = agent_o + view_offset_rad  # 当前视图在世界坐标系中的绝对方向
        
        # 实际相机FOV是79度，但只显示±15度内的waypoint（确保唯一性）
        display_fov_half = np.deg2rad(15)  # 只显示30度范围内的waypoint
        camera_fov_half = np.deg2rad(79 / 2)  # 79度用于映射坐标
        
        h, w = image.shape[:2]
        
        # ========== 1. 绘制历史轨迹投影 ==========
        if hasattr(self, 'mapper') and self.mapper:
            trajectory_points = self.mapper.trajectory_points  # List[(map_x, map_y)] 地图像素坐标
            if len(trajectory_points) > 1:
                projected_points = []
                
                # 获取mapper的转换参数
                resolution = self.mapper.resolution / 100.0  # cm/像素 → 米/像素
                map_shape = self.mapper.map_shape  # (H, W)
                map_min_x = - (map_shape[1] // 2) * resolution  # 地图左边缘的世界X坐标（米）
                map_min_y = - (map_shape[0] // 2) * resolution  # 地图上边缘的世界Y坐标（米）
                
                # 投影轨迹点到当前视角
                for traj_map_x, traj_map_y in trajectory_points:
                    # 🔑 关键修复：将地图像素坐标转换为世界坐标（米）
                    # trajectory_points格式：(map_x, map_y) 对应地图的(y, x)
                    traj_world_x = map_min_x + traj_map_y * resolution  # map_y → world_x
                    traj_world_y = map_min_y + traj_map_x * resolution  # map_x → world_y
                    
                    # 计算相对于agent的向量
                    dx = traj_world_x - agent_x
                    dy = traj_world_y - agent_y
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance < 0.1:
                        continue
                    
                    # 🔑 关键修复：arctan2返回相对于正东的角度
                    traj_angle_from_east = np.arctan2(dy, dx)
                    
                    # 计算相对于当前视图方向的角度差
                    # view_direction = agent_o + view_offset_rad
                    angle_diff = traj_angle_from_east - view_direction
                    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                    
                    # 只绘制在79度FOV内的轨迹点
                    if abs(angle_diff) <= camera_fov_half:
                        # X坐标映射（使用79度FOV）
                        x_ratio = (angle_diff + camera_fov_half) / (2 * camera_fov_half)
                        x_pos = int(x_ratio * w)
                        
                        # Y坐标映射（距离）
                        if distance < 1.0:
                            y_pos = int(h * 0.75)
                        elif distance < 2.0:
                            y_pos = int(h * 0.65)
                        elif distance < 3.0:
                            y_pos = int(h * 0.55)
                        elif distance < 5.0:
                            y_pos = int(h * 0.45)
                        else:
                            y_pos = int(h * 0.35)
                        
                        projected_points.append((x_pos, y_pos))
                
                # 绘制轨迹线（橙色虚线）
                for i in range(len(projected_points) - 1):
                    pt1 = projected_points[i]
                    pt2 = projected_points[i + 1]
                    cv2.line(image, pt1, pt2, (0, 165, 255), 2)  # 橙色线条
                
                # 在轨迹点上绘制小圆点
                for pt in projected_points:
                    cv2.circle(image, pt, 3, (0, 140, 255), -1)  # 深橙色点
        
        # ========== 2. 绘制last waypoint标记（使用渲染坐标系直接计算） ==========
        # 显示最后一个waypoint（last waypoint）
        if len(waypoint_positions) > 0:
            # 获取最后一个waypoint（列表最后一个元素）
            last_idx = len(waypoint_positions) - 1
            wp_map_x, wp_map_y = waypoint_positions[last_idx]  # 地图坐标 (map_x, map_y)
            wp_id = waypoint_ids[last_idx]
            wp_desc = waypoint_info[2][last_idx] if len(waypoint_info[2]) > last_idx else ""
            
            # 🔑 新方案：直接使用global map渲染坐标计算角度
            # 原理：在visualizer的render_global_map()中，waypoint已经经过旋转+平移变换
            # 变换后：图像中心(240, 240) = Agent位置，图像上方 = Agent前方
            # 直接计算rotated_point相对于图像中心的角度即可
            
            if not hasattr(self, 'mapper') or not self.mapper:
                return image
            
            # 1. 地图坐标 → 显示坐标（与visualizer.py完全一致）
            map_shape = self.mapper.map_shape
            h_map, w_map = map_shape
            display_x = wp_map_y * 480 / w_map
            display_y = (h_map - 1 - wp_map_x) * 480 / h_map
            
            # 2. 应用旋转+平移变换（需要从visualizer获取rotation_matrix）
            # 旋转角度: 90 - current_o（让agent朝向对准正上方）
            # 旋转中心: agent当前位置
            # 平移: 将agent移到图像中心(240, 240)
            current_o = np.rad2deg(agent_o)  # 弧度→度
            rotation_angle = 90 - current_o
            
            # agent在显示坐标系中的位置（trajectory最后一点）
            trajectory_points = self.mapper.trajectory_points
            if len(trajectory_points) == 0:
                return image
            last_traj_x, last_traj_y = trajectory_points[-1]
            agent_display_x = last_traj_y * 480 / w_map
            agent_display_y = (h_map - 1 - last_traj_x) * 480 / h_map
            
            # 构造旋转矩阵（与visualizer.py逻辑一致）
            rotation_center = (agent_display_x, agent_display_y)
            rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1.0)
            
            # 添加平移：将旋转后的agent移到(240, 240)
            target_center = np.array([240, 240, 1])
            current_center = np.array([agent_display_x, agent_display_y, 1])
            rotated_center = rotation_matrix @ current_center
            translation = target_center[:2] - rotated_center[:2]
            rotation_matrix[0, 2] += translation[0]
            rotation_matrix[1, 2] += translation[1]
            
            # 应用变换到waypoint
            point = np.array([display_x, display_y, 1])
            rotated_point = rotation_matrix @ point
            
            # 3. 计算相对于图像中心(240, 240)的角度
            # 图像坐标系：X右，Y下
            # Agent位置：(240, 240)
            # Agent前方：图像上方（Y = 0方向）
            dx = rotated_point[0] - 240  # X方向：右为正
            dy = rotated_point[1] - 240  # Y方向：下为正
            
            # arctan2(dx, -dy) = 相对于正上方（前方）的角度
            # 解释：
            # - arctan2(y, x) 返回从+X轴逆时针到(x,y)的角度
            # - 这里用 arctan2(dx, -dy)，相当于：
            #   * -dy是向上的分量（图像上方=前方）
            #   * dx是向右的分量
            #   * 返回从正上方（前方）顺时针旋转的角度
            #   * 正值=右侧，负值=左侧
            angle_from_front = np.arctan2(dx, -dy)  # 相对于前方的角度，范围[-π, π]
            
            # 计算相对于当前视图的角度差
            # view_angle: 当前视图相对于agent朝向的偏移（度），逆时针为正
            view_offset_rad = np.deg2rad(view_angle)
            angle_diff = angle_from_front - view_offset_rad
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # 归一化到[-π, π]
            
            # 只显示±15度内的waypoint（确保每个waypoint只出现在一个视图中）
            if abs(angle_diff) <= display_fov_half:
                    # X坐标映射：使用79度FOV范围映射到图像宽度
                    x_ratio = (angle_diff + camera_fov_half) / (2 * camera_fov_half)
                    x_pos = int(x_ratio * w)
                    x_pos = max(0, min(w - 1, x_pos))  # 边界检查
                    
                    # Y坐标统一在中间位置
                    y_pos = int(h * 0.5)  # 固定在图像垂直中心
                    
                    # 绘制waypoint标记（蓝色外圈 + 白色填充，更小的圆圈）
                    cv2.circle(image, (x_pos, y_pos), 15, (255, 0, 0), 2)  # 蓝色边框，半径15
                    cv2.circle(image, (x_pos, y_pos), 13, (255, 255, 255), -1)  # 白色填充
                    
                    # 绘制waypoint ID（红色粗体）
                    text = f"{wp_id}"
                    font_scale = 0.7
                    thickness = 2
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = x_pos - text_size[0] // 2
                    text_y = y_pos + text_size[1] // 2
                    cv2.putText(image, text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                    
                    # 在waypoint上方显示area type（如果有）
                    if wp_desc:
                        # 提取waypoint描述中的area type（第一部分，以" - "分隔）
                        area_type = wp_desc.split(' - ')[0] if ' - ' in wp_desc else wp_desc
                        area_font_scale = 0.5
                        area_thickness = 1
                        area_text_size = cv2.getTextSize(area_type, cv2.FONT_HERSHEY_SIMPLEX, 
                                                          area_font_scale, area_thickness)[0]
                        area_x = x_pos - area_text_size[0] // 2
                        area_y = y_pos - 25  # 圆圈上方25像素（圆圈变小了）
                        
                        # Area type背景框（蓝色边框 + 白色填充）
                        padding = 3
                        cv2.rectangle(image,
                                    (area_x - padding, area_y - area_text_size[1] - padding),
                                    (area_x + area_text_size[0] + padding, area_y + padding),
                                    (255, 0, 0), 1)  # 蓝色边框
                        cv2.rectangle(image,
                                    (area_x - padding + 1, area_y - area_text_size[1] - padding + 1),
                                    (area_x + area_text_size[0] + padding - 1, area_y + padding - 1),
                                    (255, 255, 255), -1)  # 白色填充
                        cv2.putText(image, area_type, (area_x, area_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, area_font_scale, (0, 0, 0), area_thickness)  # 黑色文字
        
        return image
    
    def _draw_floor_segmentation_on_view(self, image: np.ndarray, view_angle: float) -> np.ndarray:
        """
        在图像上绘制地面分割（绿色半透明覆盖）
        
        Args:
            image: 图像 (H, W, 3) BGR格式
            view_angle: 视角角度（相对于agent朝向，0度=正前方）
            
        Returns:
            绘制了地面分割的图像
        """
        if not hasattr(self, 'mapper') or self.mapper is None or self.mapper.floor is None:
            return image
        
        # 获取agent位姿
        agent_x, agent_y, agent_o = self._get_agent_pose()
        
        # 计算当前视角的绝对方向
        view_offset_rad = np.deg2rad(view_angle)
        view_direction = agent_o + view_offset_rad
        
        # 相机FOV (79度)
        camera_fov_half = np.deg2rad(79 / 2)
        
        h, w = image.shape[:2]
        overlay = image.copy()
        
        # 获取floor地图
        floor_map = self.mapper.floor  # [H_map, W_map]
        
        # 将floor投影到图像上
        max_distance = 5.0  # 最远检测5米
        num_rays = 50  # 增加射线密度
        
        for i in range(num_rays):
            # 射线角度分布
            ray_ratio = i / (num_rays - 1) if num_rays > 1 else 0.5
            ray_angle = view_direction - camera_fov_half + ray_ratio * 2 * camera_fov_half
            
            # 沿射线检测floor
            for dist in np.linspace(0.3, max_distance, 30):
                # 世界坐标
                world_x = agent_x + dist * np.cos(ray_angle)
                world_y = agent_y + dist * np.sin(ray_angle)
                
                # 转换到地图坐标
                resolution = self.mapper.resolution / 100.0
                map_shape = self.mapper.map_shape
                map_min_x = - (map_shape[1] // 2) * resolution
                map_min_y = - (map_shape[0] // 2) * resolution
                map_x = int((world_x - map_min_x) / resolution)
                map_y = int((world_y - map_min_y) / resolution)
                
                # 检查是否在地图范围内且是floor
                if 0 <= map_x < floor_map.shape[1] and 0 <= map_y < floor_map.shape[0]:
                    if floor_map[map_y, map_x] > 0:  # floor区域
                        # 计算图像坐标
                        angle_diff = ray_angle - view_direction
                        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                        
                        # X坐标映射
                        x_ratio = (angle_diff + camera_fov_half) / (2 * camera_fov_half)
                        x_pos = int(x_ratio * w)
                        x_pos = max(0, min(w - 1, x_pos))
                        
                        # Y坐标映射（距离越近越靠下，使用对数映射提高近处分辨率）
                        if dist < 1.0:
                            y_pos = int(h * (0.95 - 0.2 * (1.0 - dist)))
                        elif dist < 2.0:
                            y_pos = int(h * (0.75 - 0.15 * (2.0 - dist)))
                        elif dist < 3.5:
                            y_pos = int(h * (0.6 - 0.1 * (3.5 - dist) / 1.5))
                        else:
                            y_pos = int(h * 0.5)
                        
                        y_pos = max(0, min(h - 1, y_pos))
                        
                        # 绘制绿色半透明点（地面）
                        cv2.circle(overlay, (x_pos, y_pos), 2, (0, 200, 0), -1)
                else:
                    break  # 超出地图范围，停止这条射线
        
        # 混合原图和覆盖层（25%透明度，让绿色更明显）
        alpha = 0.25
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        return result
    
    def _draw_distance_rays_on_first_person_view(self, image: np.ndarray, distances: Dict[str, str]) -> np.ndarray:
        """
        在第一人称视图上绘制多条距离射线（复用统一计算的距离数据）
        
        Args:
            image: 第一人称RGB图像 (H, W, 3) BGR格式
            distances: 距离字典，如 {'front': '1.2m', 'left_30': '>2.0m', ...}
        """
        h, w = image.shape[:2]
        center_x, bottom_y = w // 2, h - 20
        fov_half = 39.5
        
        # 方向映射：key -> (相对角度, X位置比例)
        ray_map = {
            'left_90': -90, 'left_60': -60, 'left_30': -30,
            'front': 0,
            'right_30': 30, 'right_60': 60, 'right_90': 90
        }
        
        for key, angle in ray_map.items():
            if key not in distances or abs(angle) > fov_half:
                continue
            
            dist_str = distances[key]
            
            # 解析距离和颜色
            if "WARNING" in dist_str or "<0.5" in dist_str:
                color, y_ratio = (0, 0, 255), 0.7
            elif ">2.0" in dist_str or "open" in dist_str:
                color, y_ratio = (0, 255, 0), 0.1
            else:
                try:
                    dist_val = float(dist_str.replace('m', '').split()[0])
                    color = (0, 255, 255)
                    y_ratio = 0.7 if dist_val < 1.0 else (0.5 if dist_val < 2.0 else 0.3)
                except:
                    color, y_ratio = (0, 255, 255), 0.5
            
            # 计算终点
            x_ratio = (angle + fov_half) / (2 * fov_half)
            end_x, end_y = int(x_ratio * w), int(bottom_y - bottom_y * y_ratio)
            
            # 绘制射线和文字
            cv2.line(image, (center_x, bottom_y), (end_x, end_y), color, 2)
            text_x = end_x - len(dist_str) * 3
            text_y = end_y - 5
            cv2.rectangle(image, (text_x - 2, text_y - 12), (text_x + len(dist_str) * 7, text_y + 2), (0, 0, 0), -1)
            cv2.putText(image, dist_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return image
    
    def look_around_and_collect(self, phase: str = "initial") -> Tuple[List[str], List[str]]:
        """
        360°环视建图 + 生成4方向全景图
        
        执行12次×30°逆时针旋转（TURN_LEFT），每次转完后拍照并更新地图：
        - step 1: 第1次左转30°后拍照
        - step 2: 第2次左转60°后拍照
        - ...
        - step 12: 第12次左转360°后拍照（回到正前方）
        
        合成4个方向的90°视角全景图：
        - 前方：step-11(330°) + step-12(360°=0°) + step-1(30°) = 前方90°
        - 左侧：step-2(60°) + step-3(90°) + step-4(120°) = 左侧90°
        - 后方：step-5(150°) + step-6(180°) + step-7(210°) = 后方90°
        - 右侧：step-8(240°) + step-9(270°) + step-10(300°) = 右侧90°
        
        所有图像和地图统一保存到 vlm/observations/ 目录
        使用柱面投影拼接生成连贯的全景图
        环视过程不影响current_step和trajectory（环视后恢复）
        
        Args:
            phase: 阶段名称（用于文件命名，如 "initial", "verify_1"）
        
        Returns:
            (image_paths, direction_names) - 4个全景图路径和方向名称
        """
        print(f"\n[环视建图] {phase}...")
        
        # 注意：不清空landmark，让VLM能看到旧landmark来判断子任务是否完成
        # 轨迹和landmark的清空会在verify_and_replan中VLM输出后进行
        
        # 不在环视前更新距离（地图还未扫描，数据不准确）
        # 距离计算会在环视完成后进行
        
        # 存储12张环视图像用于合成全景图（step 1-12）
        lookaround_images = []
        total_new_classes = 0
        
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        
        # 直接开始12次旋转，每一步保存rgb、detection、maps
        # 使用累加的self.current_step，避免覆盖之前的数据
        for i in range(1, 13):  # 12次旋转
            self.current_step += 1  # 累加总步数
            look_step = self.current_step
            print(f"  [{i}/12] 第{i}次左转 (30°×{i}={i*30}°)", end="", flush=True)
            
            # 执行旋转
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            
            if dones[0]:
                print(" - Episode提前结束")
                break
            
            # 更新检测和建图（每一步都保存）
            prev_class_count = len(self.detected_classes)
            batch_obs = self._batch_obs(obs, save_object_detection=True)  # 保存检测结果
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            
            map_state = self.mapper.update_map(
                batch_obs, poses, look_step,
                list(self.detected_classes), self.current_episode_id
            )
            
            new_classes = len(self.detected_classes) - prev_class_count
            total_new_classes += new_classes
            
            # 调用visualizer保存所有数据（RGB、检测、全局地图、局部地图、semantic masks）
            # 地图可视化（保存地图+检测landmarks）
            # 环视过程中不传waypoint，不计算角度（环视结束后统一计算）
            rgb_bgr = cv2.cvtColor(obs[0]['rgb'], cv2.COLOR_RGB2BGR)
            paths, detected_landmarks_step, obstacle_distances, _ = self.visualizer.save_step_visualization(
                step=look_step,
                episode_id=self.current_episode_id,
                rgb=rgb_bgr,
                full_map=map_state['full_map'],
                trajectory_points=map_state['trajectory_points'],
                detected_classes=list(self.detected_classes),
                current_pose=map_state['full_pose'],
                floor=map_state['floor'],
                hfov=self.config.MAP.HFOV,
                detections=self.latest_detections_full if hasattr(self, 'latest_detections_full') else None,
                labels=self.latest_labels_full if hasattr(self, 'latest_labels_full') else None,
                masks=self.latest_masks_full if hasattr(self, 'latest_masks_full') else None,
                landmark_classes=self.landmark_classes,
                mapping_classes=self.mapping_classes,
                landmark_config={
                    'min_total_pixels': self.landmark_min_total_pixels,
                    'min_area_threshold': self.landmark_min_area_threshold
                },
                waypoint_positions=None,  # 环视过程中不传waypoint
                waypoint_ids=None,
                phase=phase,
                global_trajectory_points=map_state['global_trajectory_points'],
                calculate_distances=False  # 环视时不计算距离，加快速度
            )
            
            # 累积当前step检测到的landmarks
            if detected_landmarks_step:
                if not hasattr(self, 'current_step_landmarks'):
                    self.current_step_landmarks = {}
                self.current_step_landmarks[look_step] = detected_landmarks_step
            
            # 保存最后一次的距离信息（用于后续的planning）
            self.latest_obstacle_distances = obstacle_distances
            
            # 保存导航可视化（RGB+俯视图拼接）
            if self.nav_visualizer:
                subtask_text = self.current_subtask.get('subtask_instruction', '') if self.current_subtask else f"[环视建图 {phase}]"
                distance = 0.0
                if infos and len(infos) > 0:
                    distance = infos[0].get('distance_to_goal', 0.0)
                
                # 环视阶段的subtask_id为phase（如initial, verify_1a）
                self.nav_visualizer.save_step_visualization(
                    observations=obs[0],
                    info=infos[0] if infos and len(infos) > 0 else {},
                    step=look_step,
                    instruction=self.current_instruction,
                    current_subtask=subtask_text,
                    distance=distance,
                    action=f"TURN_LEFT (360°环视 {i}/12)",
                    subtask_id=phase
                )
            
            if new_classes > 0:
                print(f" +{new_classes}类")
            else:
                print()
            
            # 保存所有12张环视图像（用于后续合成全景图）
            lookaround_images.append(rgb_bgr.copy())
        
        # 环视建图完成
        # 注意：不恢复轨迹，轨迹会自然显示在地图上
        # 如需清空轨迹，应在verify_and_replan中的子任务完成时调用mapper.clear_trajectory()
        
        # 缓存最后的观察（step 12，回到正前方）
        self.latest_obs = obs[0]
        
        print(f"  扫描完成: +{total_new_classes}类 | 总计{len(self.detected_classes)}类")
        
        # 360度环视完成，现在更新距离信息（地图已完整扫描）
        print(f"  [更新距离] 360度扫描完成，计算12个方向的障碍物距离...")
        self._update_obstacle_distances_12_directions()
        
        # 检查是否完成了完整的12步环视
        if len(lookaround_images) < 12:
            print(f"\n⚠️ 警告: 环视未完成，只收集到 {len(lookaround_images)}/12 张图像（Episode可能提前结束）")
            print(f"❌ 无法生成观察图片，跳过视觉观察收集")
            # 返回空列表，调用方需要处理这种情况
            return [], []
        
        # 环视结束后，计算waypoint角度（只计算一次，用于显示在12张view上）
        # 注意：initial时不显示waypoint（还没有历史），replan时显示上一个waypoint
        waypoint_info = None
        last_waypoint_angle_deg = None
        if phase != "initial" and hasattr(self, 'mapper') and self.mapper:
            wp_positions, wp_ids, wp_descriptions = self.mapper.get_waypoints()
            if wp_positions:  # 如果有waypoint
                # 获取当前地图状态和RGB
                map_state = self.mapper.get_map_state()
                rgb_bgr = cv2.cvtColor(obs[0]['rgb'], cv2.COLOR_RGB2BGR)
                
                # 调用visualizer渲染地图并计算waypoint角度
                print(f"\n  [Waypoint角度计算] 环视结束，计算最后waypoint的方向...")
                _, _, _, last_waypoint_angle = self.visualizer.save_step_visualization(
                    step=look_step,  # 使用最后一步的timestep
                    episode_id=self.current_episode_id,
                    rgb=rgb_bgr,
                    full_map=map_state['full_map'],
                    trajectory_points=map_state['trajectory_points'],
                    detected_classes=list(self.detected_classes),
                    current_pose=map_state['full_pose'],
                    floor=map_state['floor'],
                    hfov=self.config.MAP.HFOV,
                    detections=None,
                    labels=None,
                    masks=None,
                    landmark_classes=self.landmark_classes,
                    mapping_classes=self.mapping_classes,
                    landmark_config={
                        'min_total_pixels': self.landmark_min_total_pixels,
                        'min_area_threshold': self.landmark_min_area_threshold
                    },
                    waypoint_positions=wp_positions,
                    waypoint_ids=wp_ids,
                    phase=phase,
                    global_trajectory_points=map_state['global_trajectory_points'],
                    calculate_distances=False
                )
                
                # 转换为度数用于view映射
                if last_waypoint_angle is not None:
                    last_waypoint_angle_deg = np.degrees(last_waypoint_angle)
                    print(f"  📍 Last Waypoint角度: {last_waypoint_angle_deg:.1f}° (正=右侧, 负=左侧, 0=正前方)")
                
                # 保存waypoint信息用于绘制在view上（直接保存tuple）
                waypoint_info = (wp_positions, wp_ids, wp_descriptions)
        
        # 保存12张独立图片（不拼接），每张图片添加角度标注 + waypoint标记
        from .vlm.navigation_config import DIRECTION_CONFIG
        
        direction_paths = []
        direction_names = []
        directions_dir = os.path.join(self.config.RESULTS_DIR, f"episode_{self.current_episode_id}", "directions")
        os.makedirs(directions_dir, exist_ok=True)
        
        for config in DIRECTION_CONFIG:
            step_idx = config["step"]  # 1-12
            angle = config["angle"]
            direction_name = config["name"]  # 如 "IMAGE 1: Front (0°)"
            
            # 获取该step的图像（step是1-based，但step 12对应index 11，step 1对应index 0）
            # lookaround_images[0] = step 1 (30°)
            # lookaround_images[11] = step 12 (0°)
            image = lookaround_images[step_idx - 1].copy()
            
            # 不再绘制可导航区域（去掉绿色地面分割，加快速度）
            # image = self._draw_navigable_area_on_view(image, angle)
            
            # 绘制距离信息（使用统一计算的12方向距离数据）
            dist_key = f'angle_{angle}'  # 'angle_0', 'angle_30', ..., 'angle_330'
            dist_str = self.latest_obstacle_distances_12.get(dist_key, 'Unknown')
            image = self.visualizer.draw_distance_on_view(image, dist_str)
            
            # 绘制waypoint标记（使用最终角度判断是否在当前视图）
            if waypoint_info and last_waypoint_angle_deg is not None:
                # 计算waypoint相对于当前视图的角度差
                angle_diff = last_waypoint_angle_deg - angle
                # 归一化到[-180, 180]
                while angle_diff > 180:
                    angle_diff -= 360
                while angle_diff < -180:
                    angle_diff += 360
                
                # 只在±15度范围内显示waypoint
                if abs(angle_diff) <= 15:
                    print(f"    ✓ Waypoint显示在 {direction_name} (角度差={angle_diff:.1f}°)")
                    image = self._draw_waypoints_on_view(image, angle, waypoint_info)
            
            # 在图片顶部添加白色背景的角度标注
            h, w = image.shape[:2]
            label_height = 35  # 减少白边，刚好够用
            label_img = np.ones((label_height, w, 3), dtype=np.uint8) * 255  # 白色背景
            
            # 添加文字标注（使用OpenCV）
            label_text = direction_name  # 如 "IMAGE 1: Front (0°)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8  # 增大字体
            font_thickness = 2  # 加粗
            text_color = (0, 0, 255)  # 红色
            
            # 计算文字位置（居中）
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            text_x = (w - text_width) // 2
            text_y = (label_height + text_height) // 2
            
            # 绘制文字
            cv2.putText(label_img, label_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
            
            # 拼接标注和图像
            labeled_image = np.vstack([label_img, image])
            
            # 保存图像
            direction_filename = f"{phase}_direction_{angle:03d}.png"  # 如 initial_direction_000.png
            direction_path = os.path.join(directions_dir, direction_filename)
            cv2.imwrite(direction_path, labeled_image)
            
            direction_paths.append(direction_path)
            direction_names.append(direction_name)
        
        # 保存全局地图和局部地图到对应目录
        # 使用当前step的地图（环视完成后的最新地图）
        self.latest_global_map = os.path.join(self.episode_dir, 'global_map', f'step_{self.current_step:04d}_{phase}.png')
        self.latest_local_map = os.path.join(self.episode_dir, 'local_map', f'step_{self.current_step:04d}_{phase}.png')
        
        if not os.path.exists(self.latest_global_map):
            print(f"  ⚠️  Global Map not found: {self.latest_global_map}")
            self.latest_global_map = None
        
        if not os.path.exists(self.latest_local_map):
            print(f"  ⚠️  Local Map not found: {self.latest_local_map}")
            self.latest_local_map = None
        
        print(f"  12方向独立视图已保存 (每张30°) | Step={self.current_step}")
        print("="*60 + "\n")
        
        return direction_paths, direction_names
    
    def _get_current_map_path(self) -> str:
        """
        获取当前语义地图路径（使用global_map/目录中的图像，避免重复保存）
        
        Returns:
            global_map目录中上一步保存的地图路径
        """
        # 返回上一步保存的地图（当前步的地图要等step()执行后才会保存）
        last_step = self.current_step - 1
        map_path = os.path.join(self.episode_dir, 'global_map', f'step_{last_step:04d}.png')
        self.latest_map_image = map_path
        return map_path

    def get_observations_and_maps(self, phase: str) -> Tuple[List[str], List[str], str, str]:
        """
        从directions/目录获取12方向独立视图和地图
        
        Args:
            phase: 阶段名称（如 "initial", "verify_1"）
            
        Returns:
            (direction_paths, direction_names, global_map_path, local_map_path)
        """
        from .vlm.navigation_config import DIRECTION_CONFIG
        
        direction_paths = []
        direction_names = []
        
        # 从episode的directions/目录读取12张独立图片
        directions_dir = os.path.join(self.episode_dir, 'directions')
        
        # 获取12个方向的图片
        for config in DIRECTION_CONFIG:
            angle = config["angle"]
            direction_name = config["name"]
            direction_filename = f"{phase}_direction_{angle:03d}.png"  # 如 initial_direction_000.png
            direction_path = os.path.join(directions_dir, direction_filename)
            
            if os.path.exists(direction_path):
                direction_paths.append(direction_path)
                direction_names.append(direction_name)
            else:
                print(f"  ⚠️  {direction_name} 未找到: {direction_filename}")
        
        # 获取地图（使用当前step的地图，每次环视后current_step已更新）
        # current_step是最后一次环视后的step，地图文件名需要加上phase后缀
        global_map_path = os.path.join(self.episode_dir, 'global_map', f'step_{self.current_step:04d}_{phase}.png')
        local_map_path = os.path.join(self.episode_dir, 'local_map', f'step_{self.current_step:04d}_{phase}.png')
        
        if not os.path.exists(global_map_path):
            print(f"  ⚠️  Global Map 未找到")
            global_map_path = None
        
        if not os.path.exists(local_map_path):
            print(f"  ⚠️  Local Map 未找到")
            local_map_path = None
        
        return direction_paths, direction_names, global_map_path, local_map_path
    
    def generate_initial_subtask(self) -> Optional[Dict]:
        """
        生成初始子任务
        
        使用环视收集的4方向全景图 + 全局地图 + 局部地图调用LLM生成子任务
        """
        if not self.planner:
            print("✗ LLM Planner未初始化")
            return None
        
        print(f"\n[LLM规划] 生成初始子任务...")
        
        # 从 vlm/observations/ 获取全景图和地图
        image_paths, direction_names, global_map, local_map = self.get_observations_and_maps("initial")
        
        # 验证地图文件存在
        if not global_map or not os.path.exists(global_map):
            print(f"✗ Global map not found: {global_map}")
            return None
        
        # 地图已包含waypoint标记（在visualizer.save_step_visualization中渲染）
        global_map_for_llm = global_map
        
        # 使用最近的障碍物距离（在look_around_and_collect中由visualizer计算）
        obstacle_distances = getattr(self, 'latest_obstacle_distances', {
            'front': 'Unknown',
            'left_30': 'Unknown',
            'right_30': 'Unknown',
            'left_90': 'Unknown',
            'right_90': 'Unknown'
        })
        
        # 先构建thinking_record（不含response）
        thinking_record = {
            "step": self.current_step,  # 12
            "phase": "initial_planning",
            "subtask_count": 1,  # 初始化总是第1个子任务
            "subtask_attempt": 0,  # 初始规划总是a
            "subtask_id": "1a",  # 初始化总是1a
            "prompt_type": "initial",
            "timestamp": datetime.now().isoformat(),
            # 保存输入图片路径（12张方向图 + 2张地图）
            "input_images": {
                "global_map.png": global_map_for_llm,
                "local_map.png": local_map,
                # 12个方向视图（IMAGE 1-12）
                **{f"IMAGE {i+1}.png": image_paths[i] if len(image_paths) > i else None for i in range(12)}
            }
        }
        
        # 生成prompt并添加到record
        from vlnce_baselines.vlm.prompts import get_initial_planning_prompt
        prompt = get_initial_planning_prompt(
            self.current_instruction, 
            "TURN_LEFT/RIGHT (30°, 60°, 90°, 120°, 150°, 180°), MOVE_FORWARD (0.25m, 0.5m, 0.75m, 1.0m, 1.25m, 1.5m), STOP"
        )
        thinking_record["prompt"] = prompt
        
        # 1️⃣ 先保存输入（图片 + prompt）
        thinking_dir = self.save_manager.save_thinking_input(thinking_record)
        print(f"  💾 已保存输入: {thinking_dir}")
        
        # 2️⃣ 调用LLM生成初始子任务
        response, _ = self.planner.generate_initial_subtask(
            instruction=self.current_instruction,
            observation_images=image_paths,
            direction_names=direction_names,
            global_map_image=global_map_for_llm,  # 使用带waypoint标注的版本
            local_map_image=local_map,
            obstacle_distances=obstacle_distances
        )
        
        if not response:
            print("✗ LLM未返回有效响应")
            return None
        
        # 3️⃣ 保存response
        thinking_record["response"] = response
        self.save_manager.save_thinking_response(thinking_record, thinking_dir)
        print(f"  💾 已保存响应: {thinking_dir}/response.json")
        
        # 记录到内存
        self.thinking_outputs.append(thinking_record)
        
        # 保存子任务并初始化计数
        self.current_subtask = response
        self.subtask_count = 1  # 初始化为第1个子任务
        self.subtask_attempt = 0  # 第a次尝试
        self.progress_summary = ""
        self.pose_before_action = None  # 重置pose追踪（新子任务从当前位置开始）
        
        # 记录当前位置信息（用于后续验证参考）
        self.current_position_info = {
            'waypoint': response.get('current_waypoint', 'Unknown'),
            'observation': response.get('current_observation', ''),
            'step': self.current_step
        }
        
        # 在mapper中添加waypoint（自动计算地图坐标）
        waypoint_desc = response.get('current_waypoint', 'Unknown location')
        waypoint_id = self.mapper.add_waypoint(waypoint_desc)
        
        # 保存waypoint摘要（用于后续LLM提示词）
        waypoint_summary = self._get_waypoint_summary()
        self.save_manager.save_waypoint_memory(
            waypoint_summary,
            self.current_instruction,
            self.current_step
        )
        
        # 动态更新目标landmark（直接使用VLM输出的next_waypoint_landmark）
        next_waypoint_landmark = response.get('next_waypoint_landmark', None)
        
        # 直接使用VLM输出，不自动提取
        if next_waypoint_landmark:
            self.landmark_classes = [next_waypoint_landmark]
            self.target_landmark = next_waypoint_landmark
            print(f"  🎯 Target Landmark: {self.target_landmark}")
        else:
            self.target_landmark = None
            self.landmark_classes = []
            print(f"  ℹ️  No target landmark")
        
        # ⚠️ 重要：self.classes始终保持为所有mapping_classes，用于完整的语义建图
        # landmark_classes只用于可视化标注和导航决策
        
        # 打印子任务信息
        self._print_subtask_info(response, is_initial=True)
        
        return response
    
    def auto_rotate_to_waypoint(self, waypoint_direction: str) -> Tuple[bool, List[Dict]]:
        """
        解析waypoint方向并生成旋转动作序列
        
        Args:
            waypoint_direction: 如 "IMAGE 5 (Left 120deg)"
            
        Returns:
            (success, action_sequence): 
                - success: 是否成功解析
                - action_sequence: 动作序列，每个动作为 {"action": "TURN_LEFT/RIGHT", "degrees": 30}
        """
        import re
        
        # 解析方向和角度
        # 支持格式: "IMAGE 5 (Left 120deg)" 或 "Left 120deg"
        match = re.search(r'Left (\d+)(?:deg|°)|Right (\d+)(?:deg|°)|Back (\d+)(?:deg|°)?|Front', waypoint_direction)
        
        if not match:
            print(f"  ⚠️  无法解析waypoint_direction: {waypoint_direction}")
            return False, []
        
        angle = 0
        direction = None
        
        if 'Left' in waypoint_direction:
            angle = int(match.group(1))
            direction = 'LEFT'
        elif 'Right' in waypoint_direction:
            angle = int(match.group(2))
            direction = 'RIGHT'
        elif 'Back' in waypoint_direction:
            angle = 180
            direction = 'LEFT'  # 向左转180度
        elif 'Front' in waypoint_direction:
            # 已经面向Front，无需旋转
            print(f"  ✓ Waypoint已在Front方向，无需旋转")
            return True, []
        else:
            print(f"  ⚠️  无法识别方向: {waypoint_direction}")
            return False, []
        
        print(f"\n[自动旋转] 解析waypoint方向: {direction} {angle}°")
        
        # 生成动作序列（每次30度）
        num_turns = angle // 30
        action_sequence = []
        
        for i in range(num_turns):
            action_sequence.append({
                "action": f"TURN_{direction}",
                "degrees": 30
            })
        
        print(f"  生成动作序列: {num_turns}次 TURN_{direction} 30°")
        return True, action_sequence
    
    def execute_rotation_sequence(self, action_sequence: List[Dict]) -> bool:
        """
        执行旋转动作序列（使用统一的执行器，确保地图更新、步数记录、可视化保存）
        
        Args:
            action_sequence: 动作序列，格式 [{"action": "TURN_LEFT", "degrees": 30}, ...]
            
        Returns:
            是否全部执行成功
        """
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        
        for i, action_dict in enumerate(action_sequence):
            action_name = action_dict["action"]
            degrees = action_dict["degrees"]
            
            print(f"  旋转 {i+1}/{len(action_sequence)}: {action_name} {degrees}°")
            
            # 转换为habitat action ID
            if action_name == "TURN_LEFT":
                action_id = HabitatSimActions.TURN_LEFT
            elif action_name == "TURN_RIGHT":
                action_id = HabitatSimActions.TURN_RIGHT
            else:
                print(f"    ⚠️  未知动作: {action_name}")
                continue
            
            # 使用统一的执行器（step_with_vlm），确保：
            # - 更新地图
            # - 保存可视化（RGB、detection、maps）
            # - 更新距离信息
            # - 正确记录步数
            result = self.step_with_vlm(action_id, action_name, save_vis=True)
            
            # 检查episode是否结束
            if result.get('done', False):
                print(f"    ⚠️  Episode在旋转过程中结束")
                return False
        
        print(f"  ✓ 旋转完成，已面向waypoint方向")
        return True
    
    def verify_and_replan(self) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        验证当前子任务并重新规划
        
        流程：
        1. 执行360°环视建图（更新语义地图）- 占用12个step
        2. 生成当前位置的4方向全景图
        3. 调用LLM验证子任务完成状态
        4. 如未完成，生成新子任务
        
        注意：重新扫描会占用新的12个step，验证完成后下一个action继续累加
        
        Returns:
            (is_completed, new_subtask, prompt)
        """
        if not self.planner or not self.current_subtask:
            return False, None, None
        
        # 重新执行环视建图并生成全景图（占用12个step）
        # 注意：如果子任务已完成，会在后面清空轨迹；如果未完成，轨迹继续累积
        # 使用attempt字母标识（a=0, b=1, c=2...）
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        phase = f"verify_{self.subtask_count}{attempt_letter}"
        print(f"\n[验证] 子任务#{self.subtask_count}{attempt_letter} - 重新环视（step {self.current_step + 1}-{self.current_step + 12}）...")
        image_paths, direction_names = self.look_around_and_collect(phase)
        
        if not image_paths:
            print("❌ 环视建图失败（Episode可能提前结束），无法进行验证")
            # Episode提前结束，无法继续验证，返回失败
            return False, None, None
        
        # 从 vlm/observations/ 获取地图（已在 look_around_and_collect 中保存）
        _, _, global_map, local_map = self.get_observations_and_maps(phase)
        
        # 验证地图文件存在
        if not global_map or not os.path.exists(global_map):
            print(f"✗ Global map not found: {global_map}")
            return False, None
        
        # 地图已包含waypoint标记（在visualizer.save_step_visualization中渲染）
        global_map_for_llm = global_map
        
        # 获取已检测到的landmark类别 - 汇总环视12步中检测到的所有landmarks
        detected_landmarks = []
        if hasattr(self, 'current_step_landmarks') and self.current_step_landmarks:
            # 汇总12步环视中所有检测到的landmarks
            all_landmarks = set()
            for step_idx, landmarks_list in self.current_step_landmarks.items():
                for name, conf in landmarks_list:
                    all_landmarks.add(name)
            detected_landmarks = sorted(list(all_landmarks))
        else:
            # 退化使用全局detected_classes
            detected_landmarks = sorted(list(self.detected_classes)) if hasattr(self, 'detected_classes') else []
        
        # 获取waypoint摘要
        waypoint_summary = self._get_waypoint_summary()
        
        # 使用最近的障碍物距离（在look_around_and_collect中由visualizer计算）
        obstacle_distances = getattr(self, 'latest_obstacle_distances', {
            'front': 'Unknown',
            'left_30': 'Unknown',
            'right_30': 'Unknown',
            'left_90': 'Unknown',
            'right_90': 'Unknown'
        })
        
        # 先构建thinking_record（不含response）
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        subtask_id = f"{self.subtask_count}{attempt_letter}"  # 当前验证的子任务，如 "1a"
        
        # 计算下一个subtask_id
        # 注意：此时还不知道is_completed，所以先按未完成准备
        next_subtask_count = self.subtask_count
        next_attempt = self.subtask_attempt + 1
        next_attempt_letter = chr(ord('a') + next_attempt)
        
        # 对于verification，使用下一个subtask_count，因为：
        # 1. Verification的目的是验证当前subtask并规划下一个subtask
        # 2. Initial planning保存在subtask_1/，verification应保存在subtask_2/等
        # 3. 即使verification失败也用下一个编号，表示"尝试规划下一个"
        verification_subtask_count = self.subtask_count + 1
        
        thinking_record = {
            "step": self.current_step,  # 验证扫描完成后的step
            "phase": f"verify_{subtask_id}",  # verify_1a, verify_2b, etc.
            "subtask_count": verification_subtask_count,  # 使用下一个编号
            "subtask_attempt": self.subtask_attempt,
            "subtask_id": subtask_id,  # 当前验证的子任务，如 "1a"
            "next_subtask_id": f"{next_subtask_count}{next_attempt_letter}",  # 暂定，后续根据is_completed更新
            "prompt_type": "verification",
            "timestamp": datetime.now().isoformat(),
            "detected_landmarks": detected_landmarks,  # 记录传递给LLM的landmarks
            # 保存输入图片路径（12张方向图 + 2张地图）
            "input_images": {
                "global_map.png": global_map_for_llm,
                "local_map.png": local_map if os.path.exists(local_map) else None,
                # 12个方向视图（IMAGE 1-12）
                **{f"IMAGE {i+1}.png": image_paths[i] if len(image_paths) > i else None for i in range(12)}
            }
        }
        
        # 生成prompt并添加到record
        from vlnce_baselines.vlm.prompts import get_verification_replanning_prompt
        prompt = get_verification_replanning_prompt(
            self.current_instruction,
            self.current_subtask.get('waypoint_sequence', 'Unknown'),
            self.current_subtask.get('next_waypoint_destination', 'Unknown'),
            self.current_subtask.get('subtask_instruction', 'Unknown'),
            str(self.current_subtask.get('completion_criteria', {})),
            "TURN_LEFT/RIGHT (30°, 60°, 90°, 120°, 150°, 180°), MOVE_FORWARD (0.25m, 0.5m, 0.75m, 1.0m, 1.25m, 1.5m), STOP",
            detected_landmarks=", ".join(detected_landmarks) if detected_landmarks else None,
            waypoint_summary=waypoint_summary
        )
        thinking_record["prompt"] = prompt
        
        # 1️⃣ 先保存输入（图片 + prompt）
        thinking_dir = self.save_manager.save_thinking_input(thinking_record)
        print(f"  💾 已保存输入: {thinking_dir}")
        
        # 2️⃣ 调用LLM验证（全局地图必需，局部地图可选，传递实际检测到的类别）
        response, is_completed, _ = self.planner.verify_and_replan(
            instruction=self.current_instruction,
            current_subtask=self.current_subtask,
            observation_images=image_paths,
            direction_names=direction_names,
            global_map_image=global_map_for_llm,  # 使用带waypoint标注的版本
            local_map_image=local_map if os.path.exists(local_map) else None,
            detected_landmarks=detected_landmarks,
            waypoint_summary=waypoint_summary,
            obstacle_distances=obstacle_distances
        )
        
        print(f"  🏷️  Detected landmarks: {detected_landmarks if detected_landmarks else 'None'}")
        
        if not response:
            print("✗ LLM验证未返回有效响应")
            return False, None, None
        
        # 更新next_subtask_id（根据is_completed）
        if is_completed and not response.get('global_task_finish', False):
            thinking_record["next_subtask_id"] = f"{self.subtask_count + 1}a"
        elif response.get('global_task_finish', False):
            thinking_record["next_subtask_id"] = "final"
        # else: 保持原值（未完成，重试）
        
        # 3️⃣ 保存response
        thinking_record["is_completed"] = is_completed
        thinking_record["response"] = response
        self.save_manager.save_thinking_response(thinking_record, thinking_dir)
        print(f"  💾 已保存响应: {thinking_dir}/response.json")
        
        # 记录到内存
        self.thinking_outputs.append(thinking_record)
        
        if is_completed:
            attempt_letter = chr(ord('a') + self.subtask_attempt)
            print(f"\n[子任务完成] #{self.subtask_count}{attempt_letter}")
            
            # 打印LLM的response关键信息
            print(f"  📍 Current Waypoint: {response.get('current_waypoint', 'N/A')}")
            print(f"  📍 Waypoint Sequence: {response.get('waypoint_sequence', 'N/A')}")
            print(f"  🎯 Next Waypoint Destination: {response.get('next_waypoint_destination', 'N/A')}")
            print(f"  📝 Subtask Instruction: {response.get('subtask_instruction', 'N/A')}")
            
            # 检查是否完成全局任务
            task_finished = response.get('global_task_finish', False)
            print(f"  🎯 global_task_finish: {task_finished}")
            
            if task_finished:
                print("\n" + "="*60)
                print("🏆 全局任务完成 - 导航成功！")
                print("="*60)
                print(f"  📍 Current Waypoint: {response.get('current_waypoint', 'N/A')}")
                print(f"  📍 Waypoint Sequence: {response.get('waypoint_sequence', 'N/A')}")
                print(f"  🎯 Next Waypoint Destination: {response.get('next_waypoint_destination', 'N/A')}")
                print(f"  📝 Subtask Instruction: {response.get('subtask_instruction', 'N/A')}")
                print(f"  🔍 Completion Criteria:")
                criteria = response.get('completion_criteria', {})
                if isinstance(criteria, dict):
                    print(f"     - Panoramic_Detection: {criteria.get('Panoramic_Detection', 'N/A')}")
                    print(f"     - Spatial_relationship: {criteria.get('Spatial_relationship', 'N/A')}")
                    print(f"     - Location: {criteria.get('Location', 'N/A')}")
                print(f"  💭 Reasoning: {response.get('reasoning', 'N/A')}")
                print("="*60)
                
                # 在结束前保存最终waypoint
                waypoint_desc = response.get('current_waypoint', 'Final destination')
                waypoint_id = self.mapper.add_waypoint(waypoint_desc)
                final_waypoint_summary = self._get_waypoint_summary()
                self.save_manager.save_waypoint_memory(
                    final_waypoint_summary,
                    self.current_instruction,
                    self.current_step
                )
                
                # 直接结束导航，不再执行action
                return (True, response, final_waypoint_summary)
            else:
                print(f"  ➡️  继续下一个子任务（#{self.subtask_count + 1}a）")
            
            # ===== 关键顺序：先添加waypoint（使用当前trajectory），再清空 =====
            
            # 1. 先创建waypoint（此时trajectory还存在）
            waypoint_desc = response.get('current_waypoint', 'Unknown location')
            waypoint_id = self.mapper.add_waypoint(waypoint_desc)
            
            # 保存waypoint摘要
            waypoint_summary = self._get_waypoint_summary()
            self.save_manager.save_waypoint_memory(
                waypoint_summary,
                self.current_instruction,
                self.current_step
            )
            
            # 2. 然后清空旧状态（为新子任务准备）
            print("\n[状态清理] 添加waypoint后 - 清空旧landmark和轨迹（准备新子任务）")
            self.mapper.clear_trajectory()  # 清空轨迹
            self.landmark_classes = []      # 清空landmark标注（马上会重新设置）
            self.progress_summary = ""      # 重置进度摘要
            self.previous_action_reason = ""  # 重置上一步action reason（新子任务）
            self.pose_before_action = None   # 重置pose追踪（新子任务从当前位置开始）
            self.last_planned_degrees = 0    # 重置计划角度
            self.last_planned_meters = 0     # 重置计划距离
            self.last_action_name = ""        # 重置动作名称
            if hasattr(self, 'current_step_landmarks'):
                self.current_step_landmarks.clear()  # 清空step landmark记录
            
            # 更新到新子任务：递增计数，重置尝试
            self.subtask_count += 1
            self.subtask_attempt = 0  # 新子任务从a开始
            self.current_subtask = response
            
            # 更新当前位置信息（用于后续参考）
            self.current_position_info = {
                'waypoint': response.get('current_waypoint', 'Unknown'),
                'observation': response.get('current_observation', ''),
                'step': self.current_step
            }
            
            # 动态更新目标landmark（直接使用VLM输出的next_waypoint_landmark）
            next_waypoint_landmark = response.get('next_waypoint_landmark', None)
            
            # 直接使用VLM输出，不自动提取
            if next_waypoint_landmark:
                self.landmark_classes = [next_waypoint_landmark]
                self.target_landmark = next_waypoint_landmark
                print(f"  🎯 New Target Landmark: {self.target_landmark}")
            else:
                self.target_landmark = None
                self.landmark_classes = []
                print(f"  ℹ️  No target landmark")
            
            # ⚠️ 重要：self.classes始终保持为所有mapping_classes，用于完整的语义建图
            
            self._print_subtask_info(response)
            
            # 子任务完成后，自动旋转到新的waypoint方向
            next_waypoint_direction = response.get('next_waypoint_direction', '')
            if next_waypoint_direction and 'Front' not in next_waypoint_direction:
                print(f"\n[两阶段导航] 新子任务 - 阶段1: 旋转到waypoint方向")
                success, action_sequence = self.auto_rotate_to_waypoint(next_waypoint_direction)
                
                if success and action_sequence:
                    # 执行旋转动作序列
                    rotation_success = self.execute_rotation_sequence(action_sequence)
                    
                    if rotation_success:
                        print(f"  ✓ 旋转完成，waypoint应该在前方，直接开始Action")
        else:
            attempt_letter = chr(ord('a') + self.subtask_attempt)
            print(f"\n[子任务未完成] #{self.subtask_count}{attempt_letter} - 重新规划")
            
            # ===== 关键顺序：先添加waypoint（使用当前trajectory），再清空 =====
            
            # 1. 先创建waypoint（此时trajectory还存在）
            waypoint_desc = response.get('current_waypoint', 'Replanning location')
            waypoint_id = self.mapper.add_waypoint(waypoint_desc)
            
            # 保存waypoint摘要
            waypoint_summary = self._get_waypoint_summary()
            self.save_manager.save_waypoint_memory(
                waypoint_summary,
                self.current_instruction,
                self.current_step
            )
            
            # 2. 然后清空旧状态（为新规划准备）
            print("\n[状态清理] 添加waypoint后 - 清空旧landmark和轨迹（准备重新规划）")
            self.mapper.clear_trajectory()  # 清空轨迹
            self.landmark_classes = []      # 清空landmark标注（马上会重新设置）
            self.progress_summary = ""      # 重置进度摘要
            self.previous_action_reason = ""  # 重置上一步action reason（重新规划）
            if hasattr(self, 'current_step_landmarks'):
                self.current_step_landmarks.clear()
            
            # 未完成时保持subtask_count不变，递增attempt
            self.subtask_attempt += 1  # 下次验证用b, c, d...
            self.current_subtask = response
            
            # 更新位置观察（用于记录轨迹）
            if 'current_observation' in response:
                self.current_position_info = {
                    'waypoint': response.get('current_waypoint', getattr(self, 'current_position_info', {}).get('waypoint', 'Unknown')),
                    'observation': response.get('current_observation', ''),
                    'step': self.current_step
                }
            
            # 从新的子任务指令中提取landmark（直接使用VLM输出的next_waypoint_landmark）
            next_waypoint_landmark = response.get('next_waypoint_landmark', None)
            
            # 直接使用VLM输出，不自动提取
            if next_waypoint_landmark:
                self.landmark_classes = [next_waypoint_landmark]
                self.target_landmark = next_waypoint_landmark
                print(f"  🎯 Updated Target Landmark: {self.target_landmark}")
            else:
                self.target_landmark = None
                self.landmark_classes = []
                print(f"  ℹ️  No target landmark")
            
            # ⚠️ 重要：self.classes始终保持为所有mapping_classes，用于完整的语义建图
            
            # 输出LLM验证结果和调整后的子任务
            self._print_subtask_info(response, is_initial=False)
        
        return is_completed, response, prompt
    
    def execute_action_with_vlm(self) -> Tuple[Optional[int], Optional[str], bool, int]:
        """
        使用VLM决策并执行动作
        
        Returns:
            (action_id, action_name, should_stop, repeat_count)
        """
        if not self.action_executor or not self.current_subtask:
            return None, None, True
        
        # 获取当前观察：使用缓存的观察或通过旋转获取
        if self.latest_obs is not None:
            obs = self.latest_obs
        else:
            # 如果没有缓存，执行一次右转再左转回来获取观察
            actions = [{"action": HabitatSimActions.TURN_RIGHT}]
            outputs = self.envs.step(actions)
            obs, _, dones, _ = [list(x) for x in zip(*outputs)]
            if dones[0]:
                print("⚠️ Episode结束")
                return None, None, True
            
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            outputs = self.envs.step(actions)
            obs, _, dones, _ = [list(x) for x in zip(*outputs)]
            if dones[0]:
                print("⚠️ Episode结束")
                return None, None, True
            obs = obs[0]
        
        # 获取最新保存的观察信息
        # 上一步已保存的文件（如果current_step=13，则读取step_0012的地图）
        last_step = self.current_step  # execute_action在step执行前调用，所以用current_step
        
        # 生成当前子任务的phase标识
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        action_phase = f"action{self.subtask_count}{attempt_letter}"
        
        # 智能查找可用的图像：优先使用action phase，回退到verify/initial
        # 可能的phase顺序: action2a -> verify_2a -> verify_1a -> initial (注意verify带下划线)
        possible_phases = [action_phase]
        
        # 添加当前子任务的验证phase（验证完成后保存的全景图）
        current_verify_phase = f"verify_{self.subtask_count}a"
        possible_phases.append(current_verify_phase)
        
        if self.subtask_attempt > 0:
            # 如果是1b, 1c等，可能需要回退到上一次尝试的verify
            prev_attempt_verify = f"verify_{self.subtask_count}{chr(ord('a') + self.subtask_attempt - 1)}"
            possible_phases.append(prev_attempt_verify)
        
        if self.subtask_count > 1:
            # 回退到上一个子任务的verify
            prev_verify_phase = f"verify_{self.subtask_count - 1}a"
            possible_phases.append(prev_verify_phase)
        
        # 最后回退到initial
        possible_phases.append("initial")
        
        # 查找RGB图像
        fp_image = None
        for phase in possible_phases:
            candidate = os.path.join(self.episode_dir, 'rgb', f'step_{last_step:04d}_{phase}.png')
            if os.path.exists(candidate):
                fp_image = candidate
                break
        
        # 如果都不存在，用当前观察创建临时文件
        if not fp_image:
            rgb_bgr = cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR)
            temp_image = os.path.join(self.episode_dir, f'temp_fp_step{last_step}.png')
            cv2.imwrite(temp_image, rgb_bgr)
            fp_image = temp_image
        
        # 查找对应的semantic masks
        mask_path = None
        for phase in possible_phases:
            candidate = os.path.join(self.episode_dir, 'semantic_masks', f'step_{last_step:04d}_{phase}.npy')
            if os.path.exists(candidate):
                mask_path = candidate
                break
        
        # 为RGB图像不添加距离辅助线（只有detection才显示距离）
        fp_image = self.visualizer.prepare_action_image_with_enhancements(
            fp_image, mask_path, self.latest_obstacle_distances, self.classes, use_floor=False, use_distance=False)
        
        # 获取当前地图路径和检测图像
        self._get_current_map_path()
        
        # 查找detection图像（使用相同的回退逻辑）
        detection_image = None
        detection_step = None  # 记录找到的detection图像对应的step
        for phase in possible_phases:
            candidate = os.path.join(self.episode_dir, 'detection', f'step_{last_step:04d}_{phase}.png')
            if os.path.exists(candidate):
                detection_image = candidate
                detection_step = last_step
                break
        if not detection_image:
            print(f"  ⚠️  Detection image not found for step {last_step} (tried phases: {possible_phases})")
        else:
            # 为detection图像添加距离辅助线（不使用地面分割）
            detection_image = self.visualizer.prepare_action_image_with_enhancements(
                detection_image, mask_path, self.latest_obstacle_distances, self.classes, use_floor=False, use_distance=True)
        
        # 查找局部地图（使用相同的回退逻辑）
        local_map = None
        for phase in possible_phases:
            candidate = os.path.join(self.episode_dir, 'local_map', f'step_{last_step:04d}_{phase}.png')
            if os.path.exists(candidate):
                local_map = candidate
                break
        if not local_map:
            print(f"  ⚠️  Local map not found for step {last_step} (tried phases: {possible_phases})")
        
        # 获取detection图像对应的landmark类别
        # 使用找到的detection图像对应的step
        detected_landmarks = None
        if detection_step is not None and hasattr(self, 'current_step_landmarks') and detection_step in self.current_step_landmarks:
            # 当前step检测到的landmarks: [(name, confidence), ...]
            step_landmarks = self.current_step_landmarks[detection_step]
            if step_landmarks:
                # 格式化为 "name1 (conf1), name2 (conf2)"
                detected_landmarks = ', '.join([f"{name} ({conf:.2f})" for name, conf in step_landmarks])
        
        # 退化策略：如果没有检测结果，报告"未检测到"
        if not detected_landmarks:
            if hasattr(self, 'target_landmark') and self.target_landmark:
                detected_landmarks = f"No {self.target_landmark} detected in current view"
            else:
                detected_landmarks = "No landmarks detected"
        
        # 使用最新的障碍物距离（在step_with_vlm中已更新）
        obstacle_distances = getattr(self, 'latest_obstacle_distances', {
            'front': 'Unknown',
            'left_30': 'Unknown',
            'right_30': 'Unknown',
            'left_90': 'Unknown',
            'right_90': 'Unknown'
        })
        
        # 准备action输入记录（在调用API之前）
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        action_record = {
            "step": self.current_step + 1,  # 即将执行的action的step
            "subtask_count": self.subtask_count,
            "subtask_attempt": self.subtask_attempt,
            "subtask_id": f"{self.subtask_count}{attempt_letter}",  # 如 "1a"
            "timestamp": datetime.now().isoformat(),
            "detected_landmarks": detected_landmarks,  # 记录传递给VLM的landmarks
            # 保存输入图片路径（action模块只使用3张图：RGB + Detection + Local Map）
            "input_images": {
                "rgb.png": fp_image,
                "detection.png": detection_image,
                "local_map.png": local_map,
            },
        }
        
        # 保存子任务信息
        subtask_info = {
            "subtask_id": self.subtask_count,
            "next_waypoint_destination": self.current_subtask.get('next_waypoint_destination', ''),
            "subtask_instruction": self.current_subtask.get('subtask_instruction', ''),
            "start_step": self.current_step,  # 子任务开始时的step（决策前）
            "timestamp": datetime.now().isoformat()
        }
        
        # 1️⃣ 先保存输入（图片 + prompt稍后在decide_action中生成后保存）
        action_dir = self.save_manager.save_action_input(action_record, subtask_info)
        print(f"  💾 已保存输入: {action_dir}")
        
        # 2️⃣ 调用VLM决策（不传递pose，使用当前progress_summary）
        result = self.action_executor.decide_action(
            next_waypoint_destination=self.current_subtask.get('next_waypoint_destination', ''),
            subtask_instruction=self.current_subtask.get('subtask_instruction', ''),
            first_person_image=fp_image,
            action_mapping=ACTION_MAPPING,
            progress_summary=self.progress_summary,  # 使用上次动作的progress
            detection_image=detection_image,
            local_map_image=local_map,
            detected_landmarks=detected_landmarks,
            previous_action_reason=self.previous_action_reason,
            obstacle_distances=obstacle_distances
        )
        
        if len(result) == 7:
            action_id, action_name, _, response, degrees, meters, prompt = result  # 忽略updated_progress
        elif len(result) == 6:
            action_id, action_name, _, response, degrees, meters = result  # 忽略updated_progress
            prompt = None
        else:
            # 兼容旧版本返回（没有degrees/meters）
            action_id, action_name, _, response = result  # 忽略updated_progress
            degrees, meters = 0, 0
            prompt = None
        
        if action_id is None:
            print("✗ VLM决策失败")
            return None, None, True, 1
        
        # 3️⃣ 保存响应和prompt
        action_record["action_name"] = action_name
        action_record["action_id"] = action_id
        action_record["response"] = response
        action_record["prompt"] = prompt
        
        self.action_outputs.append(action_record)
        self.save_manager.save_action_response(action_record)
        print(f"  ✓ 已保存响应: {action_name}")
        
        # 保存planned action参数，供后续计算actual progress使用
        self.last_planned_degrees = degrees
        self.last_planned_meters = meters
        self.last_action_name = action_name
        
        # 保存当前的action_analysis作为下一次的previous_action_reason
        if response and 'action_analysis' in response:
            self.previous_action_reason = response['action_analysis']
        else:
            self.previous_action_reason = ""
        
        # 检查是否停止
        should_stop = (action_name == "STOP")
        
        # 计算需要重复执行的次数
        repeat_count = 1
        if action_name == 'TURN_LEFT' or action_name == 'TURN_RIGHT':
            # 每次转30度，计算需要转几次
            if degrees > 0:
                repeat_count = max(1, round(degrees / self.action_executor.turn_angle))
        elif action_name == 'MOVE_FORWARD':
            # 每次移动0.25m，计算需要移动几次
            if meters > 0:
                repeat_count = max(1, round(meters / self.action_executor.move_distance))
        
        return action_id, action_name, should_stop, repeat_count
    
    def step_with_vlm(self, action: int, action_name: str = "", save_vis: bool = True) -> Dict[str, Any]:
        """
        执行VLM决策的动作（调用父类step方法）并缓存观察
        
        Args:
            action: 动作ID
            action_name: 动作名称（用于可视化）
            save_vis: 是否保存可视化
            
        Returns:
            步骤结果字典
        """
        # 生成phase标识: action1a, action2b等
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        phase = f"action{self.subtask_count}{attempt_letter}"
        
        result = self.step(action, save_vis, phase)
        # 缓存最新观察和info用于下次VLM决策和可视化
        self.latest_obs = result.get('obs', None)
        self.latest_info = result.get('info', None)
        
        # 地图已更新，立即计算当前位置的障碍物距离
        self._update_obstacle_distances()
        
        # 保存RGB+俯视图拼接可视化
        if save_vis and self.nav_visualizer and self.latest_obs is not None:
            subtask_text = None
            if self.current_subtask:
                subtask_text = self.current_subtask.get('subtask_instruction', '')
            
            distance = 0.0
            if self.latest_info:
                distance = self.latest_info.get('distance_to_goal', 0.0)
            
            attempt_letter = chr(ord('a') + self.subtask_attempt)
            subtask_id = f"{self.subtask_count}{attempt_letter}"
            
            self.nav_visualizer.save_step_visualization(
                observations=self.latest_obs,
                info=self.latest_info or {},
                step=self.current_step,
                instruction=self.current_instruction,
                current_subtask=subtask_text,
                distance=distance,
                action=action_name,
                subtask_id=subtask_id
            )
        
        return result
    
    def _update_obstacle_distances_12_directions(self):
        """更新当前位置的12个方向障碍物距离（用于Thinking模式环视）"""
        try:
            # 检查地图是否已初始化
            if not hasattr(self, 'mapper') or self.mapper is None:
                raise ValueError("Mapper not initialized")
            
            if self.mapper.full_map is None:
                raise ValueError("Map not initialized yet")
            
            if self.mapper.full_pose is None:
                raise ValueError("Pose not initialized yet")
            
            obstacle_map = self.mapper.full_map[0, ...]
            h, w = obstacle_map.shape
            current_x, current_y, current_o = self.mapper.full_pose
            
            # 预处理obstacle mask
            obstacle_mask = np.flipud(obstacle_map > 0.5)
            obstacle_mask = cv2.resize(
                obstacle_mask.astype(np.uint8) * 255, (480, 480), 
                interpolation=cv2.INTER_NEAREST
            ) > 127
            
            # 计算agent位置
            if len(self.mapper.trajectory_points) > 0:
                last_x, last_y = self.mapper.trajectory_points[-1]
                agent_x, agent_y = last_y * 480 / w, (h - 1 - last_x) * 480 / h
            else:
                pos = np.array([current_x, current_y]) * 100.0 / self.config.MAP.MAP_RESOLUTION
                agent_x, agent_y = pos[1] * 480 / w, (h - 1 - pos[0]) * 480 / h
            
            # 旋转地图使箭头朝上
            rotation_matrix = cv2.getRotationMatrix2D((agent_x, agent_y), 90 - current_o, 1.0)
            rotated_center = rotation_matrix @ np.array([agent_x, agent_y, 1])
            rotation_matrix[0:2, 2] += [240, 240] - rotated_center[:2]
            
            obstacle_mask_rotated = cv2.warpAffine(
                obstacle_mask.astype(np.uint8) * 255,
                rotation_matrix, (480, 480), flags=cv2.INTER_NEAREST
            ) > 127
            
            # 计算12个方向的距离
            self.latest_obstacle_distances_12 = self.visualizer.calculate_obstacle_distances_12_directions(
                obstacle_mask_rotated, 240, 240
            )
        except Exception as e:
            print(f"  ⚠️  12方向距离更新失败: {e}")
            self.latest_obstacle_distances_12 = {
                f'angle_{i}': 'Unknown' for i in range(0, 360, 30)
            }
    
    def _update_obstacle_distances(self):
        """更新当前位置的障碍物距离（用于Action模式，7个方向）"""
        try:
            # 检查地图是否已初始化
            if not hasattr(self, 'mapper') or self.mapper is None:
                raise ValueError("Mapper not initialized")
            
            if self.mapper.full_map is None:
                raise ValueError("Map not initialized yet")
            
            if self.mapper.full_pose is None:
                raise ValueError("Pose not initialized yet")
            
            obstacle_map = self.mapper.full_map[0, ...]
            h, w = obstacle_map.shape
            current_x, current_y, current_o = self.mapper.full_pose
            
            # 预处理obstacle mask
            obstacle_mask = np.flipud(obstacle_map > 0.5)
            obstacle_mask = cv2.resize(
                obstacle_mask.astype(np.uint8) * 255, (480, 480), 
                interpolation=cv2.INTER_NEAREST
            ) > 127
            
            # 计算agent位置
            if len(self.mapper.trajectory_points) > 0:
                last_x, last_y = self.mapper.trajectory_points[-1]
                agent_x, agent_y = last_y * 480 / w, (h - 1 - last_x) * 480 / h
            else:
                pos = np.array([current_x, current_y]) * 100.0 / self.config.MAP.MAP_RESOLUTION
                agent_x, agent_y = pos[1] * 480 / w, (h - 1 - pos[0]) * 480 / h
            
            # 旋转地图使箭头朝上
            rotation_matrix = cv2.getRotationMatrix2D((agent_x, agent_y), 90 - current_o, 1.0)
            rotated_center = rotation_matrix @ np.array([agent_x, agent_y, 1])
            rotation_matrix[0:2, 2] += [240, 240] - rotated_center[:2]
            
            obstacle_mask_rotated = cv2.warpAffine(
                obstacle_mask.astype(np.uint8) * 255,
                rotation_matrix, (480, 480), flags=cv2.INTER_NEAREST
            ) > 127
            
            # 计算距离
            self.latest_obstacle_distances = self.visualizer.calculate_obstacle_distances_from_rotated_map(
                obstacle_mask_rotated, 240, 240
            )
        except Exception as e:
            print(f"  ⚠️  距离更新失败: {e}")
            self.latest_obstacle_distances = {
                'front': 'Unknown',
                'left_30': 'Unknown',
                'left_60': 'Unknown', 
                'left_90': 'Unknown',
                'right_30': 'Unknown',
                'right_60': 'Unknown',
                'right_90': 'Unknown'
            }
    
    def run_vlm_navigation(self, max_steps: int = 500, 
                          max_subtask_steps: int = 5) -> Dict[str, Any]:
        """
        运行完整的VLM导航流程
        
        Args:
            max_steps: 最大总步数
            max_subtask_steps: 每个子任务最大步数（达到后强制触发验证，默认5步）
            
        Returns:
            导航结果字典
        """
        print("\n" + "="*60)
        print("启动VLM自动导航")
        print("="*60)
        print(f"指令: {self.current_instruction}")
        print(f"最大步数: {max_steps} | 子任务最大步数: {max_subtask_steps}")
        print("="*60 + "\n")
        
        # 1. 环视建图 + 收集观察（占用step 1-12）
        image_paths, direction_names = self.look_around_and_collect()
        
        if not image_paths:
            print("❌ 初始环视建图失败（Episode可能在启动时就结束），无法开始导航")
            return {
                'success': False,
                'total_steps': self.current_step,
                'subtask_count': 0,
                'detected_classes': list(self.detected_classes) if hasattr(self, 'detected_classes') else [],
                'gif_path': None,
                'result_file': None,
                'reason': 'initial_lookaround_failed'
            }
        
        # 2. 生成初始子任务（在step 12完成，下一个action从step 13开始）
        subtask = self.generate_initial_subtask()
        if not subtask:
            print("✗ 初始子任务生成失败")
            return {
                'success': False,
                'total_steps': self.current_step,  # 12
                'subtask_count': 0,
                'detected_classes': list(self.detected_classes) if hasattr(self, 'detected_classes') else [],
                'gif_path': None,
                'result_file': None,
                'reason': 'initial_subtask_failed'
            }
        
        # 2.5 自动旋转到waypoint方向
        next_waypoint_direction = subtask.get('next_waypoint_direction', '')
        if next_waypoint_direction and 'Front' not in next_waypoint_direction:
            print(f"\n[两阶段导航] 阶段1: 旋转到waypoint方向")
            success, action_sequence = self.auto_rotate_to_waypoint(next_waypoint_direction)
            
            if not success or not action_sequence:
                print("  ✗ 自动旋转解析失败")
                # 继续执行，让ActionVLM处理
            else:
                # 执行旋转动作序列
                rotation_success = self.execute_rotation_sequence(action_sequence)
                
                if not rotation_success:
                    print("  ✗ 旋转执行失败")
                else:
                    print(f"  ✓ 旋转完成，waypoint应该在前方，直接开始Action")
        
        # 3. 主导航循环
        total_steps = self.current_step
        subtask_steps = 0
        navigation_complete = False
        
        while total_steps < max_steps:
            # VLM决策动作（失败则重试）
            max_retries = 3
            action_id = None
            
            for retry in range(max_retries):
                action_id, action_name, should_stop, repeat_count = self.execute_action_with_vlm()
                
                if action_id is not None:
                    break
                
                if retry < max_retries - 1:
                    print(f"VLM决策失败，重试 ({retry + 1}/{max_retries - 1})...")
                    import time
                    time.sleep(1)
            
            # 所有重试都失败，跳过此步
            if action_id is None:
                print("✗ VLM决策失败，已达最大重试次数，跳过此步")
                continue
            
            # 如果VLM决定停止 → 验证子任务
            if should_stop:
                print("\n[VLM输出STOP] 开始验证子任务...")
                
                # verify_and_replan会调用thinking模型检查任务是否完成
                # 返回值：(is_completed, new_subtask, waypoint_summary)
                is_completed, new_subtask, _ = self.verify_and_replan()
                
                # 检查是否完成全局任务
                if is_completed and new_subtask:
                    task_finished = new_subtask.get('global_task_finish', False)
                    if task_finished:
                        print("\n🛑 全局任务完成 - 立即终止导航循环")
                        navigation_complete = True
                        break
                
                # 子任务完成或重新规划，重置步数计数
                subtask_steps = 0
                continue
            
            # VLM决策计数（每次调用action模型算1步）
            subtask_steps += 1
            print(f"  [步数统计] 当前子任务已执行 {subtask_steps}/{max_subtask_steps} 步")
            
            # 🔑 关键修复：在执行action后检查步数限制
            # 如果达到最大步数（例如5步），执行完当前动作后立即强制replan
            if subtask_steps >= max_subtask_steps:
                print(f"\n⚠️  [强制重规划] 子任务已达到最大步数 ({max_subtask_steps}步)，执行完当前动作后将触发验证")
                # 继续执行当前动作，但标记下一轮要replan
                force_replan_after_action = True
            else:
                force_replan_after_action = False
            
            # 执行动作前记录pose（用于后续计算实际变化）
            if self.pose_before_action is None:
                self.pose_before_action = self._get_agent_pose()
                print(f"[Pose] 初始化pose_before: {self.pose_before_action}")
            pose_before_action_batch = self._get_agent_pose()
            
            # 执行动作（可能需要重复多次）
            for i in range(repeat_count):
                result = self.step_with_vlm(action_id, action_name=action_name, save_vis=True)
                total_steps = self.current_step
                
                if i == 0 and repeat_count > 1:
                    print(f"[Step {total_steps}] {action_name} (1/{repeat_count}) | VLM决策次数: {subtask_steps}")
                elif repeat_count > 1:
                    print(f"[Step {total_steps}] {action_name} ({i+1}/{repeat_count}) | VLM决策次数: {subtask_steps}")
                else:
                    print(f"[Step {total_steps}] {action_name} | VLM决策次数: {subtask_steps}")
                
                if result['done']:
                    print("\nEpisode自动完成")
                    navigation_complete = True
                    break
            
            # 所有重复执行完成后，计算总的progress（一次性）
            if hasattr(self, 'last_action_name') and self.last_action_name and not navigation_complete:
                pose_after_action_batch = self._get_agent_pose()
                
                # 计算实际位姿变化
                x_before, y_before, ori_before = pose_before_action_batch
                x_after, y_after, ori_after = pose_after_action_batch
                
                # 计算实际转向角度变化（保留符号）
                import math
                angle_diff = ori_after - ori_before
                # 归一化到 [-pi, pi]
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                # 判断实际转向方向（正=左转，负=右转）
                actual_degrees = abs(math.degrees(angle_diff))
                
                # 根据实际方向校正action_name（处理转过头的情况）
                actual_action_name = self.last_action_name
                if self.last_action_name == 'TURN_LEFT' and angle_diff < -0.1:  # 计划左转但实际右转
                    actual_action_name = 'TURN_RIGHT'
                    print(f"[Warning] Planned TURN_LEFT but actually turned RIGHT by {actual_degrees:.1f}°")
                elif self.last_action_name == 'TURN_RIGHT' and angle_diff > 0.1:  # 计划右转但实际左转
                    actual_action_name = 'TURN_LEFT'
                    print(f"[Warning] Planned TURN_RIGHT but actually turned LEFT by {actual_degrees:.1f}°")
                
                # 计算实际移动距离（2D欧氏距离）
                actual_meters = math.sqrt((x_after - x_before)**2 + (y_after - y_before)**2)
                
                # 调用_generate_progress_update更新progress
                self.progress_summary = self.action_executor._generate_progress_update(
                    current_progress=self.progress_summary,
                    action_name=actual_action_name,  # 使用校正后的方向
                    degrees=self.last_planned_degrees,
                    meters=self.last_planned_meters,
                    actual_degrees=actual_degrees,
                    actual_meters=actual_meters
                )
                
                print(f"[Progress] {self.last_action_name} planned:{self.last_planned_degrees}°/{self.last_planned_meters}m, actual:{actual_degrees:.1f}°/{actual_meters:.2f}m → {self.progress_summary}")
                
                # 更新pose_before为当前pose（供下次计算使用）
                self.pose_before_action = pose_after_action_batch
            
            # 🔑 强制重规划检查：如果达到最大步数，执行完动作后立即触发verify
            if force_replan_after_action:
                print(f"\n🔄 [强制重规划] 已执行完 {max_subtask_steps} 步，立即触发验证和重规划")
                _, _, _ = self.verify_and_replan()
                subtask_steps = 0  # 重置步数
                continue
            
            if navigation_complete:
                break
            
            if result['done']:
                print("\nEpisode自动完成")
                navigation_complete = True
                break
        
        # 4. 生成GIF动画
        gif_path = None
        if self.nav_visualizer:
            gif_path = self.nav_visualizer.save_gif(fps=2)
            print(f"\n🎬 GIF动画: {gif_path if gif_path else '未生成'}")
        
        # 5. 保存结果（供后续测评）
        final_result = self._save_navigation_result(navigation_complete, total_steps)
        
        return {
            'success': navigation_complete,
            'total_steps': total_steps,
            'subtask_count': self.subtask_count,
            'detected_classes': list(self.detected_classes),
            'gif_path': gif_path,
            'result_file': final_result
        }
    
    def _save_thinking_output(self, thinking_record: Dict):
        """保存LLM思考输出（调用save_manager）"""
        self.save_manager.save_thinking(thinking_record)
    
    def _save_action_output(self, action_record: Dict):
        """保存VLM动作输出（调用save_manager）"""
        self.save_manager.save_action(action_record)
    
    def _save_navigation_result(self, success: bool, total_steps: int) -> str:
        """保存导航结果到log/目录"""
        # 获取所有关键指标
        metrics = {}
        if self.latest_info:
            metrics = {
                # 核心指标
                'distance_to_goal': self.latest_info.get('distance_to_goal', -1),
                'success': int(self.latest_info.get('success', 0)),  # 0或1
                'spl': self.latest_info.get('spl', 0.0),
                'path_length': self.latest_info.get('path_length', 0.0),
                
                # Oracle指标
                'oracle_success': int(self.latest_info.get('oracle_success', 0)),  # 0或1
                'oracle_navigation_error': self.latest_info.get('oracle_navigation_error', float('inf')),
                'oracle_spl': self.latest_info.get('oracle_spl', 0.0),
            }
        
        result = {
            'episode_id': self.current_episode_id,
            'instruction': self.current_instruction,
            'total_steps': total_steps,
            'subtask_count': self.subtask_count,
            
            # 核心导航指标
            'success': metrics.get('success', 0),
            'spl': metrics.get('spl', 0.0),
            'distance_to_goal': metrics.get('distance_to_goal', -1),
            'path_length': metrics.get('path_length', 0.0),
            
            # Oracle指标
            'oracle_success': metrics.get('oracle_success', 0),
            'oracle_navigation_error': metrics.get('oracle_navigation_error', float('inf')),
            'oracle_spl': metrics.get('oracle_spl', 0.0),
            
            # 附加信息
            'detected_classes': list(self.detected_classes),
            'subtask_history': self.subtask_history,
            'thinking_count': len(self.thinking_outputs),
            'action_count': len(self.action_outputs),
            'timestamp': datetime.now().isoformat()
        }
        
        return self.save_manager.save_result(result)
    
    def record_action(self, action_name: str, action_id: int, vlm_response: Dict = None):
        """
        记录动作到当前子任务文件（与llm_vlm_control兼容的格式）
        
        Args:
            action_name: 动作名称
            action_id: 动作ID
            vlm_response: VLM响应字典（可选）
        """
        if not self.current_subtask_file or not os.path.exists(self.current_subtask_file):
            return
        
        action_data = {
            "step": self.current_step,
            "action_name": action_name,
            "action_id": action_id,
        }
        
        if self.latest_info:
            action_data["distance_to_goal"] = self.latest_info.get("distance_to_goal", -1)
        
        if vlm_response:
            action_data["vlm_response"] = {
                k: vlm_response.get(k, "") 
                for k in ['observation', 'reasoning', 'action']  # 移除progress_summary（由系统维护）
            }
        
        # 读取并更新文件
        try:
            with open(self.current_subtask_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "actions" not in data:
                data["actions"] = []
            data["actions"].append(action_data)
            
            with open(self.current_subtask_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 记录动作失败: {e}")
    
    def _print_subtask_info(self, response: Dict, is_initial: bool = False):
        """打印子任务信息（JSON格式）"""
        import json
        
        # 根据响应类型确定标题
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        if is_initial:
            title = f"Initial Subtask #{self.subtask_count}{attempt_letter}"
        elif 'is_completed' in response:
            # 验证响应
            if response.get('is_completed', False):
                title = f"Subtask #{self.subtask_count}{attempt_letter} - Completed ✓"
            else:
                title = f"Subtask #{self.subtask_count}{attempt_letter} - Continue (Not Completed)"
        else:
            title = f"Subtask #{self.subtask_count}{attempt_letter}"
        
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"Global Instruction: {self.current_instruction}")
        print(f"{'-'*60}")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print(f"{'='*60}\n")
    
    # ========== Waypoint辅助方法 ==========
    
    def _get_waypoint_summary(self) -> str:
        """获取waypoint摘要（用于LLM提示词）"""
        wp_pos, wp_ids, wp_descs = self.mapper.get_waypoints()
        if len(wp_ids) == 0:
            return ""
        
        # 根据waypoint ID和描述生成摘要
        summary_lines = []
        for wp_id, wp_desc in zip(wp_ids, wp_descs):
            summary_lines.append(f"#{wp_id}: {wp_desc}")
        
        return "\n".join(summary_lines)
    
    # ========== 原有方法 ==========

