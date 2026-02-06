"""
建图系统常量定义
=================
集中管理建图相关的所有常量，避免magic numbers
"""

# ========================================
# 地图通道配置
# ========================================

# 基础地图通道数（不含语义类别）
MAP_CHANNELS = 3

# 通道索引常量
CHANNEL_OBSTACLE = 0    # 障碍物地图
CHANNEL_EXPLORED = 1    # 已探索区域
CHANNEL_AGENT = 2       # Agent通道（合并：轨迹+当前位置）

# Channel 2 的值定义
TRAJ_VALUE = 0.5        # 轨迹线标记值
CURRENT_VALUE = 1.0     # 当前位置标记值
PAST_VALUE = 0.75       # 历史位置（保留，暂未使用）

# 轨迹提取阈值（用于从Channel 2提取轨迹）
TRAJ_THRESHOLD_LOW = 0.4   # 下界
TRAJ_THRESHOLD_HIGH = 0.6  # 上界

# ========================================
# Tile分块配置
# ========================================

TILE_SIZE = 240           # 每个Tile的像素尺寸
TILE_SIZE_M = 12.0       # 每个Tile的物理尺寸（米）

# ========================================
# 坐标转换常量
# ========================================

# 分辨率：5cm/pixel = 20 pixel/meter
MAP_RESOLUTION_CM = 5     # cm/pixel
PIXELS_PER_METER = 20     # pixel/meter

# ========================================
# 轨迹绘制配置
# ========================================

TRAJ_MARK_SIZE = 3        # 轨迹标记的半径（像素）
TRAJ_LINE_THICKNESS = 1   # 轨迹线粗细（用于Bresenham）

# ========================================
# Waypoint配置
# ========================================

WAYPOINT_MIN_DISTANCE_M = 2.0  # Waypoint最小间距（米）
WAYPOINT_MIN_DISTANCE_PX = int(WAYPOINT_MIN_DISTANCE_M * PIXELS_PER_METER)  # 像素

# ========================================
# 地图参数默认值
# ========================================

DEFAULT_MAP_SIZE_CM = 2400  # 默认地图尺寸 24m = 2400cm
DEFAULT_VISION_RANGE = 100  # 视野范围 100cm = 1m
DEFAULT_AGENT_HEIGHT = 88   # Agent高度 88cm

# ========================================
# 投影参数
# ========================================

DEFAULT_MIN_Z = 25          # 最小高度（cm）
DEFAULT_MAX_HEIGHT = 360    # 最大高度（cm）
DEFAULT_MIN_HEIGHT = -40    # 最小高度（cm）

# ========================================
# 阈值配置
# ========================================

CAT_PRED_THRESHOLD = 1.0    # 语义类别预测阈值
EXP_PRED_THRESHOLD = 1.0    # 探索区域预测阈值
MAP_PRED_THRESHOLD = 1.0    # 地图预测阈值

# ========================================
# 渲染配置
# ========================================

# 轨迹颜色（BGR格式）
TRAJ_COLOR_BGR = (0, 165, 255)  # 橙色

# Waypoint颜色（BGR格式）
WAYPOINT_COLOR_BGR = (255, 0, 0)      # 蓝色
WAYPOINT_FILL_BGR = (255, 255, 255)   # 白色填充

# 当前位置标记颜色
CURRENT_POS_COLOR_BGR = (0, 0, 255)   # 红色

# ========================================
# 调试开关
# ========================================

DEBUG_PRINT_TILE_INFO = False      # 打印Tile信息
DEBUG_PRINT_TRAJECTORY = False     # 打印轨迹点
DEBUG_SAVE_INTERMEDIATE = False    # 保存中间结果
