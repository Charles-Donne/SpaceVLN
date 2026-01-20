"""
VLM导航系统配置常量
===================
统一管理导航系统中的常量配置，避免重复定义
"""

# 12方向独立视图配置（360°全覆盖）
# 环视是逆时针TURN_LEFT，12步×30°=360°
# 每个方向一张独立图片，30°视角
# step 0: 初始朝向 0° (Front)
# step 1: 左转30° = 30°
# step 2: 左转60° = 60°
# step 3: 左转90° = 90° (Left)
# step 4: 左转120° = 120°
# step 5: 左转150° = 150°
# step 6: 左转180° = 180° (Back)
# step 7: 左转210° = 210°
# step 8: 左转240° = 240°
# step 9: 左转270° = 270° (Right)
# step 10: 左转300° = 300°
# step 11: 左转330° = 330°
# step 12: 左转360° = 0° (回到Front)

# 12个方向的配置（每个方向对应一个step）
DIRECTION_CONFIG = [
    {"step": 12, "angle": 0, "name": "IMAGE 1: Front 0°"},           # 初始朝向
    {"step": 1, "angle": 30, "name": "IMAGE 2: Left 30°"},           # 左转30°
    {"step": 2, "angle": 60, "name": "IMAGE 3: Left 60°"},           # 左转60°
    {"step": 3, "angle": 90, "name": "IMAGE 4: Left 90°"},           # 左转90°
    {"step": 4, "angle": 120, "name": "IMAGE 5: Left 120°"},         # 左转120°
    {"step": 5, "angle": 150, "name": "IMAGE 6: Left 150°"},         # 左转150°
    {"step": 6, "angle": 180, "name": "IMAGE 7: Back 180°"},         # 左转180°
    {"step": 7, "angle": 210, "name": "IMAGE 8: Right 150°"},        # 右侧150° (从Back算起右转30°)
    {"step": 8, "angle": 240, "name": "IMAGE 9: Right 120°"},        # 右侧120°
    {"step": 9, "angle": 270, "name": "IMAGE 10: Right 90°"},        # 右侧90°
    {"step": 10, "angle": 300, "name": "IMAGE 11: Right 60°"},       # 右侧60°
    {"step": 11, "angle": 330, "name": "IMAGE 12: Right 30°"}        # 右侧30°
]

# 提取方向名称列表（用于API传递）
DIRECTION_NAMES = [config["name"] for config in DIRECTION_CONFIG]

# 向后兼容：保留旧的4方向配置（如果有其他模块依赖）
DIRECTION_STEPS = [0, 3, 6, 9]  # 对应主要4个方向
PANORAMA_CONFIG = []  # 废弃：不再使用拼接全景图

# 动作映射
ACTION_MAPPING = {
    "STOP": 0,
    "MOVE_FORWARD": 1, 
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3
}
