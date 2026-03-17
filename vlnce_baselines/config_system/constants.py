"""
导航系统配置常量
================
包含地图渲染、物体检测、Landmark标注等核心配置
"""

# ========================================
# 地图渲染配置
# ========================================

# PIL调色板 (归一化0-1)
color_palette = [
    1.00, 1.00, 1.00,  # 0: 白色 - 未探索
    0.00, 0.00, 0.00,  # 1: 黑色 - 障碍物
    0.83, 0.83, 0.83,  # 2: 浅灰 - 已探索
    1.00, 0.65, 0.00,  # 3: 橙色 - 轨迹
    0.12, 0.47, 0.71,  # 4: 蓝色 - Goal
    0.77, 0.88, 0.65,  # 5: 浅绿 - Floor
]

# Legend调色板 (0-255整数，用于visualization.py)
legend_color_palette = [
    255, 255, 255,  # 0: 白色 - out of map
    0, 0, 0,        # 1: 黑色 - obstacle
    211, 211, 211,  # 2: 浅灰 - free space
    255, 165, 0,    # 3: 橙色 - agent trajectory
    31, 119, 180,   # 4: 蓝色 - waypoint
    196, 225, 165,  # 5: 浅绿 - Floor
]

# 地图通道定义
# Channel 0: Obstacle (障碍物)
# Channel 1: Explored (已探索区域)
# Channel 2: Current Location (当前位置)
# Channel 0: Obstacle Map (障碍物)
# Channel 1: Explored Area (已探索区域)
# Channel 2: Agent通道（合并当前位置/历史/轨迹，节省空间）
#   - 0.0 = 无标记
#   - 0.5 = Trajectory (轨迹线)
#   - 0.75 = Past Locations (历史位置，保留但不使用)
#   - 1.0 = Current Location (当前位置)
# Channel 3+: Semantic Categories (语义类别，动态扩展)
# 注意：Waypoint不需要独立通道，而是通过轨迹点列表+ID管理
map_channels = 3  # 基础地图通道数（优化后，不含语义类别）


# ========================================
# 物体检测配置
# ========================================

# ========================================
# GroundedSAM检测类别配置
# ========================================
# 
# 检测原理：
#   GroundedSAM = Grounding DINO (开放词汇检测) + SAM (精确分割)
#   - 通过自然语言文本提示词检测物体，无固定类别限制
#   - 理论上可检测任何英文描述的物体（零样本学习）
#   - classes参数直接作为文本提示词传递给Grounding DINO
#
# 工作流程：
#   RGB图像 + mapping_classes文本列表 → GroundedSAM → 检测+分割 → mask → 投影到地图
#
# 建图逻辑：
#   1. 分类：navigable（可通行） vs not_navigable（障碍物）
#   2. 计算：obstacles = depth障碍物 + sum(not_navigable物体)
#   3. 生成：floor = explored区域 & (1 - obstacles)
#
# 渲染顺序：
#   已探索(灰) → Floor(绿) → 障碍物(黑) → 轨迹(橙) → Landmark(紫)
#
# 注意：
#   - 类别越多，检测耗时越长（建议15-30个）
#   - 使用常见英文词汇，支持短语描述（如"coffee table"）
#   - 任何在此列表中的物体都可以动态成为landmark

# 室内导航基础检测类别（15个）
# 注意：只检测这15个类别，其他检测结果会被忽略，避免类别数爆炸
mapping_classes = [
    # 建筑结构 (3个)
    "floor", "wall", "door",
    
    # 大型家具 (6个)
    "bed", "sofa", "chair", "table", 
    "cabinet", "bookshelf",
    
    # 功能设施 (4个)
    "tv", "sink", "toilet", "plant",
    
    # 装饰 (2个)
    "painting", "window",
]

# 固定类别数量（建图系统预分配通道数）
NUM_SEMANTIC_CATEGORIES = len(mapping_classes)  # 15个

# 可扩展的检测类别建议（按需添加）
# 
# 厨房相关：
#   "kitchen island", "counter", "countertop", "toaster", "kettle", "coffee maker"
#
# 办公相关：
#   "computer", "monitor", "keyboard", "printer", "whiteboard", "filing cabinet"
#
# 娱乐相关：
#   "game console", "speaker", "remote control", "clock", "photo frame"
#
# 户外/特殊：
#   "fireplace", "radiator", "air conditioner", "heater", "fan"
#
# 使用说明：
#   根据导航场景选择相关类别，避免添加过多无关物体影响性能

# 可通行类别（用于区分障碍物）
# 建图时：
#   - 在navigable_classes中 → 可通行区域（绿色）
#   - 不在navigable_classes中 → 障碍物（黑色）
navigable_classes = [
    "floor", "ground", "flooring",
    "walkway", "corridor", "hallway",
    "stair", "stairs", "staircase",
]


# ========================================
# Landmark标注配置
# ========================================

# Landmark类别（地图紫色圆球 + Detection黄色边框）
# 初始化为空，在VLM导航中根据LLM输出的subtask_landmark动态更新
# 
# 工作流程：
#   1. 初始状态：landmark_classes = []
#   2. LLM输出 subtask_landmark (e.g., 'cabinet', 'bed', 'table')
#   3. 验证：只要该物体在mapping_classes中（能被GroundedSAM检测），就可作为landmark
#   4. 动态更新：landmark_classes = [subtask_landmark]
#   5. 地图标注：只显示当前子任务的目标landmark紫色圆球
# 
# 注意：任何在mapping_classes中的物体都可以成为landmark，不需要预先配置
landmark_classes = []

# 地图标记样式
landmark_marker_color = (128, 0, 128)      # 紫色(BGR)
landmark_marker_border = (255, 255, 255)   # 白色边框(BGR)
landmark_marker_radius = 6                 # 圆球半径(像素)
local_map_landmark_topk = 3               # Local map仅渲染置信度最高的Top-K landmark实例

# 标注阈值（控制显示条件）
landmark_min_total_pixels = 1       # 总像素数阈值（1=不过滤）
landmark_min_area_threshold = 1     # 单个连通域最小面积(像素)，1=不过滤稀疏投影
landmark_merge_distance = 0         # 合并距离(像素)，0=不再合并相邻landmark连通域
landmark_instance_topk = 0          # 每类landmark保留的检测实例数上限（0=当前子任务内全部保留）
landmark_instance_merge_radius_m = 0.60  # 同类实例世界坐标去重半径（米）


# ========================================
# Detection可视化配置
# ========================================

# 边界框样式（只显示Landmark类别）
detection_colors = {"landmark": (0, 255, 255)}     # 黄色(BGR)
detection_thickness = {"landmark": 3}              # 线宽
detection_visible_topk = 3                         # 当前帧最多显示/记录的自定义landmark检测数（按置信度）
landmark_strip_topk = 3                            # Action底部白条最多显示的vis/off vis总条目数

# 重复landmark合并阈值（调tight/loose时主要改这里）
landmark_duplicate_iou_strict = 0.65               # 同帧几乎重合bbox，直接视为同一物体
landmark_duplicate_iou_loose = 0.25                # 配合距离/角度使用的较松bbox重叠阈值
landmark_duplicate_rel_dist_m = 0.35               # 同帧深度投影相对坐标的最大距离差（米）
landmark_duplicate_angle_diff_deg = 12.0           # 同帧深度投影允许的最大角度差（度）
