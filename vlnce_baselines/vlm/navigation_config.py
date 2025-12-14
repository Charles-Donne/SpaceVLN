"""
VLM导航系统配置常量
===================
统一管理导航系统中的常量配置，避免重复定义
"""

# 4方向配置（从环视中提取）
# 环视是逆时针TURN_LEFT，12步×30°=360°
DIRECTION_STEPS = [0, 3, 6, 9]  # 对应12步中的第0,3,6,9步

DIRECTION_NAMES = [
    "Front (0°)",      # 步骤0: 初始朝向
    "Left (90°)",      # 步骤3: 左转90°
    "Back (180°)",     # 步骤6: 后方
    "Right (270°)"     # 步骤9: 右方（或左转270°）
]

# 全景图配置：每个方向3张图像拼接成90°视角
# step-11(前一圈最后一张) + step-0(初始) + step-1(第1次左转) = 前方90°
# step-2 + step-3 + step-4 = 60° + 90° + 120° = 左侧90°
# step-5 + step-6 + step-7 = 150° + 180° + 210° = 后方90°
# step-8 + step-9 + step-10 = 240° + 270° + 300° = 右侧90°
PANORAMA_CONFIG = [
    {"name": "Front (0°)", "steps": [11, 0, 1]},
    {"name": "Left (90°)", "steps": [2, 3, 4]},
    {"name": "Back (180°)", "steps": [5, 6, 7]},
    {"name": "Right (270°)", "steps": [8, 9, 10]}
]

# 动作映射
ACTION_MAPPING = {
    "STOP": 0,
    "MOVE_FORWARD": 1, 
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3
}
