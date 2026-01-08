"""
障碍物距离计算工具
==================
计算五个方向到最近障碍物的距离
"""

import numpy as np
import math
from typing import Tuple, Dict


def calculate_obstacle_distances(
    full_map: np.ndarray,
    full_pose: np.ndarray,
    resolution: int = 5,
    max_distance: float = 2.0
) -> Dict[str, str]:
    """
    计算五个方向到最近障碍物的距离
    
    方向定义：
    - FRONT: 0° (正前方)
    - LEFT-30: 30° (左前方)
    - RIGHT-30: -30° (右前方)
    - LEFT-90: 90° (左侧)
    - RIGHT-90: -90° (右侧)
    
    距离规则：
    - >2.0m: 返回 ">2.0m open"
    - <2.0m: 返回具体距离 "X.XXm"
    - <0.5m: 返回 "<0.5m WARNING"
    
    Args:
        full_map: 全局地图 [C, H, W]，full_map[0]是障碍物层
        full_pose: 当前位姿 [x(m), y(m), orientation(°)]
        resolution: 地图分辨率 cm/pixel，默认5
        max_distance: 最大检测距离(m)，默认2.0m
    
    Returns:
        distances: 字典包含5个方向的距离字符串
            {
                'front': ">2.0m open" | "1.25m" | "<0.5m WARNING",
                'left_30': ...,
                'right_30': ...,
                'left_90': ...,
                'right_90': ...
            }
    """
    if full_map is None or full_pose is None:
        print(f"⚠️  [Distance Calculation] Missing data: full_map={'None' if full_map is None else f'shape={full_map.shape}'}, full_pose={'None' if full_pose is None else f'{full_pose}'}")
        return {
            'front': 'Unknown',
            'left_30': 'Unknown',
            'right_30': 'Unknown',
            'left_90': 'Unknown',
            'right_90': 'Unknown'
        }
    
    # 提取障碍物层
    obstacle_map = full_map[0, :, :]  # [H, W]
    
    # 当前位姿：full_pose = [c, r, orientation]
    # 注意：semantic_mapping.py中 r=full_pose[1], c=full_pose[0]
    # r对应x方向（行），c对应y方向（列）
    c_m, r_m, orientation_deg = full_pose
    
    # 转换为像素坐标
    map_h, map_w = obstacle_map.shape
    
    # 地图中心（agent初始位置）
    center_x = map_w // 2
    center_y = map_h // 2
    
    # full_pose单位是米，地图中心是(12m, 12m)
    # c对应列(x方向)，r对应行(y方向)
    pixel_x = int(c_m * 100 / resolution)  # c(米) → 像素列
    pixel_y = int(r_m * 100 / resolution)  # r(米) → 像素行
    
    # 检查边界
    if not (0 <= pixel_x < map_w and 0 <= pixel_y < map_h):
        print(f"⚠️  [Distance] Position out of map: pixel=({pixel_x}, {pixel_y}), map_size=({map_w}, {map_h}), pose=({c_m:.2f}, {r_m:.2f}, {orientation_deg:.1f}°)")
        return {
            'front': 'Out of map',
            'left_30': 'Out of map',
            'right_30': 'Out of map',
            'left_90': 'Out of map',
            'right_90': 'Out of map'
        }
    
    # 定义五个方向（相对于当前朝向）
    directions = {
        'front': 0,       # 正前方
        'left_30': 30,    # 左前方30°
        'right_30': -30,  # 右前方30°
        'left_90': 90,    # 左侧90°
        'right_90': -90   # 右侧90°
    }
    
    distances = {}
    
    for direction_name, angle_offset in directions.items():
        # 计算绝对角度（世界坐标系）
        absolute_angle = orientation_deg + angle_offset
        
        # 转换为弧度
        angle_rad = math.radians(absolute_angle)
        
        # 计算该方向的距离
        distance_m = _raycast_distance(
            obstacle_map,
            pixel_x,
            pixel_y,
            angle_rad,
            resolution,
            max_distance
        )
        
        # 格式化输出
        distances[direction_name] = _format_distance(distance_m)
    
    return distances


def _raycast_distance(
    obstacle_map: np.ndarray,
    start_x: int,
    start_y: int,
    angle_rad: float,
    resolution: int,
    max_distance: float
) -> float:
    """
    使用光线投射计算到障碍物的距离
    
    Args:
        obstacle_map: 障碍物地图 [H, W]
        start_x: 起始x坐标（像素）
        start_y: 起始y坐标（像素）
        angle_rad: 方向角度（弧度）
        resolution: 地图分辨率 cm/pixel
        max_distance: 最大检测距离(m)
    
    Returns:
        distance_m: 到障碍物的距离(米)，如果无障碍则返回max_distance
    """
    map_h, map_w = obstacle_map.shape
    
    # 方向向量（注意：地图坐标系y向下为正）
    # angle_rad=0表示正右，需要转换为地图坐标系
    # 在Habitat中：0°=正东(+x)，90°=正北(+y)，180°=正西(-x)，270°=正南(-y)
    # 在地图中：y轴向下，x轴向右
    # 需要转换：地图dx = sin(angle), 地图dy = -cos(angle)
    dx = math.sin(angle_rad)
    dy = -math.cos(angle_rad)
    
    # 步长（每次前进的像素数，使用0.5像素保证精度）
    step_pixels = 0.5
    
    # 最大步数
    max_pixels = int(max_distance * 100 / resolution)  # 转换为像素
    max_steps = int(max_pixels / step_pixels)
    
    # 光线投射
    for step in range(1, max_steps + 1):
        # 当前位置
        current_x = start_x + dx * step * step_pixels
        current_y = start_y + dy * step * step_pixels
        
        # 转换为整数坐标
        ix = int(round(current_x))
        iy = int(round(current_y))
        
        # 检查边界
        if not (0 <= ix < map_w and 0 <= iy < map_h):
            # 超出地图边界，认为是障碍物
            distance_pixels = step * step_pixels
            return distance_pixels * resolution / 100.0
        
        # 检查是否遇到障碍物（障碍物值>0.5）
        if obstacle_map[iy, ix] > 0.5:
            # 计算距离（米）
            distance_pixels = step * step_pixels
            distance_m = distance_pixels * resolution / 100.0
            return distance_m
    
    # 未遇到障碍物，返回最大距离
    return max_distance


def _format_distance(distance_m: float) -> str:
    """
    格式化距离字符串
    
    Args:
        distance_m: 距离(米)
    
    Returns:
        formatted: 格式化的距离字符串
    """
    if distance_m >= 2.0:
        return ">2.0m open"
    elif distance_m < 0.5:
        return f"<0.5m WARNING"
    else:
        return f"{distance_m:.2f}m"


def get_distance_summary(distances: Dict[str, str]) -> str:
    """
    生成距离摘要字符串（用于日志输出）
    
    Args:
        distances: 距离字典
    
    Returns:
        summary: 例如 "FRONT: 1.25m | L30: >2m | R30: 0.85m | L90: 0.4m | R90: >2m"
    """
    return (
        f"FRONT: {distances['front']} | "
        f"L30: {distances['left_30']} | "
        f"R30: {distances['right_30']} | "
        f"L90: {distances['left_90']} | "
        f"R90: {distances['right_90']}"
    )
