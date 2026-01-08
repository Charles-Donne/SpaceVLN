"""
障碍物距离计算工具
==================
计算五个方向到最近障碍物的距离

坐标系说明：
1. full_pose = [c, r, orientation_deg]
   - c: 列坐标（对应world Y轴）
   - r: 行坐标（对应world X轴）
   - orientation_deg: Habitat朝向角度（0°=东，90°=北，180°=西，270°=南）

2. full_map坐标系：
   - 行（第一维）：对应world X轴
   - 列（第二维）：对应world Y轴
   - 地图顶部 = agent的当前front方向（可视化时旋转对齐）

3. 射线方向定义（相对于agent front）：
   - FRONT: agent正前方
   - LEFT-30: agent左前30°
   - RIGHT-30: agent右前30°
   - LEFT-90: agent左侧90°
   - RIGHT-90: agent右侧90°
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
    
    Args:
        full_map: 全局地图 [C, H, W]，full_map[0]是障碍物层
        full_pose: 当前位姿 [c(m), r(m), orientation(°)]
        resolution: 地图分辨率 cm/pixel，默认5
        max_distance: 最大检测距离(m)，默认2.0m
    
    Returns:
        distances: 字典包含5个方向的距离字符串
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
    # semantic_mapping.py: r=full_pose[1], c=full_pose[0]
    c_m, r_m, orientation_deg = full_pose
    
    # 转换为像素坐标（full_pose单位是米，直接转换）
    map_h, map_w = obstacle_map.shape
    
    # full_pose直接表示在地图中的位置（米），转换为像素
    pixel_col = int(c_m * 100 / resolution)  # c → 列
    pixel_row = int(r_m * 100 / resolution)  # r → 行
    
    # 检查边界
    if not (0 <= pixel_col < map_w and 0 <= pixel_row < map_h):
        print(f"⚠️  [Distance] Position out of map: pixel=({pixel_row}, {pixel_col}), map_size=({map_h}, {map_w}), pose=({r_m:.2f}, {c_m:.2f}, {orientation_deg:.1f}°)")
        return {
            'front': 'Out of map',
            'left_30': 'Out of map',
            'right_30': 'Out of map',
            'left_90': 'Out of map',
            'right_90': 'Out of map'
        }
    
    # 定义五个方向（相对于agent当前朝向）
    # orientation在Habitat中: 0°=东, 90°=北, 180°=西, 270°=南
    # 地图坐标系: 行向下=南, 列向右=东
    # agent front方向 = orientation
    # agent left = orientation + 90° (逆时针)
    # agent right = orientation - 90° (顺时针)
    directions = {
        'front': 0,        # 正前方
        'left_30': 30,     # 左前方30°（逆时针）
        'right_30': -30,   # 右前方30°（顺时针）
        'left_90': 90,     # 左侧90°
        'right_90': -90    # 右侧90°
    }
    
    distances = {}
    
    for direction_name, angle_offset in directions.items():
        # 计算绝对角度（世界坐标系）
        absolute_angle = orientation_deg + angle_offset
        
        # 在±5°范围内扫描多条射线，避免单点误判
        scan_range = 5  # ±5度范围
        num_rays = 5    # 扫描5条射线
        ray_distances = []
        
        for i in range(num_rays):
            # 在范围内均匀分布射线
            scan_offset = -scan_range + (2 * scan_range * i / (num_rays - 1)) if num_rays > 1 else 0
            scan_angle = absolute_angle + scan_offset
            angle_rad = math.radians(scan_angle)
            
            # 计算该方向的距离（使用row/col参数名）
            distance_m = _raycast_distance(
                obstacle_map,
                pixel_row,
                pixel_col,
                angle_rad,
                resolution,
                max_distance
            )
            ray_distances.append(distance_m)
        
        # 取中位数距离（更鲁棒，避免极值影响）
        ray_distances.sort()
        median_distance = ray_distances[len(ray_distances) // 2]
        
        # 格式化输出
        distances[direction_name] = _format_distance(median_distance)
    
    return distances


def _raycast_distance(
    obstacle_map: np.ndarray,
    start_row: int,
    start_col: int,
    angle_rad: float,
    resolution: int,
    max_distance: float
) -> float:
    """
    使用光线投射计算到障碍物的距离
    
    坐标系说明：
    - obstacle_map[row, col]: 行=X轴方向，列=Y轴方向
    - angle_rad: Habitat角度（0°=东, 90°=北, 180°=西, 270°=南）
    - 方向向量计算：
      * 东(0°):   drow=0,  dcol=+1
      * 北(90°):  drow=-1, dcol=0
      * 西(180°): drow=0,  dcol=-1
      * 南(270°): drow=+1, dcol=0
    
    Args:
        obstacle_map: 障碍物地图 [H, W]
        start_row: 起始行坐标（像素）
        start_col: 起始列坐标（像素）
        angle_rad: Habitat角度（弧度）
        resolution: 地图分辨率 cm/pixel
        max_distance: 最大检测距离(m)
    
    Returns:
        distance_m: 到障碍物的距离(米)
    """
    map_h, map_w = obstacle_map.shape
    
    # Habitat坐标系到地图坐标系的转换
    # Habitat: 0°=东(+Y), 90°=北(+X), 180°=西(-Y), 270°=南(-X)
    # 地图: row向下(+row=南=-X), col向右(+col=东=+Y)
    # 方向向量：
    #   dcol = cos(angle)  # Y轴分量
    #   drow = -sin(angle) # X轴分量（负号因为row向下）
    dcol = math.cos(angle_rad)
    drow = -math.sin(angle_rad)
    
    # 步长（每次前进的像素数）
    step_pixels = 0.5
    
    # 最大步数
    max_pixels = int(max_distance * 100 / resolution)
    max_steps = int(max_pixels / step_pixels)
    
    # 光线投射
    for step in range(1, max_steps + 1):
        # 当前位置
        current_row = start_row + drow * step * step_pixels
        current_col = start_col + dcol * step * step_pixels
        
        # 转换为整数坐标
        irow = int(round(current_row))
        icol = int(round(current_col))
        
        # 检查边界
        if not (0 <= irow < map_h and 0 <= icol < map_w):
            distance_pixels = step * step_pixels
            return distance_pixels * resolution / 100.0
        
        # 检查是否遇到障碍物
        if obstacle_map[irow, icol] > 0.5:
            distance_pixels = step * step_pixels
            distance_m = distance_pixels * resolution / 100.0
            return distance_m
    
    # 未遇到障碍物
    return max_distance
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
