"""
LLM规划模块
===========
高层规划：分析环境生成子任务
"""
from typing import Dict, List, Tuple, Optional
from vlnce_baselines.vlm.api_client import APIConfig, BaseAPIClient
from vlnce_baselines.vlm.prompts import (
    get_initial_planning_prompt,
    get_verification_replanning_prompt
)
from vlnce_baselines.visualization.visualizer import MapVisualizer


class LLMPlanner(BaseAPIClient):
    """LLM规划器 - 负责子任务生成和验证"""
    
    REQUIRED_FIELDS_INITIAL = ['waypoint_direction', 'subtask_destination', 'subtask_instruction', 'completion_criteria']
    REQUIRED_FIELDS_VERIFY = ['waypoint_direction', 'subtask_destination', 'subtask_instruction', 'completion_criteria']
    
    # completion_criteria 子字段（嵌套结构）
    REQUIRED_CRITERIA_FIELDS = ['Surrounding_Detection', 'Spatial_relationship', 'Location']
    
    def __init__(self, config_path: str = "vlnce_baselines/vlm/llm_config.yaml", 
                 action_space: str = None):
        """
        初始化LLM规划器
        
        Args:
            config_path: LLM配置文件路径
            action_space: 动作空间描述（如 "MOVE_FORWARD (0.25m), TURN_LEFT (30°), ..."）
        """
        config = APIConfig(config_path)
        super().__init__(config)
        
        # 默认动作空间与interactive_navigation一致
        self.action_space = action_space or "MOVE_FORWARD (0.25m), TURN_LEFT (30°), TURN_RIGHT (30°), STOP"
        
        print(f"✓ LLM Planner initialized")
        print(f"  Model: {self.config.model}")
        print(f"  Action space: {self.action_space}")
    
    def validate_response(self, response: Dict, mode: str = 'initial') -> bool:
        """验证响应字段"""
        required = self.REQUIRED_FIELDS_INITIAL if mode == 'initial' else self.REQUIRED_FIELDS_VERIFY
        
        # 先验证基础字段
        if not self.validate_fields(response, required):
            return False
        
        # 验证completion_criteria嵌套字段
        criteria = response.get('completion_criteria')
        if criteria and isinstance(criteria, dict):
            for field in self.REQUIRED_CRITERIA_FIELDS:
                if field not in criteria:
                    print(f"⚠️ Missing completion_criteria field: {field}")
                    return False
        else:
            print(f"⚠️ completion_criteria should be a dict with fields: {self.REQUIRED_CRITERIA_FIELDS}")
            return False
        
        return True
    
    def generate_initial_subtask(self,
                                instruction: str,
                                observation_images: List[str],
                                direction_names: List[str],
                                global_map_image: str,
                                local_map_image: str = None,
                                obstacle_distances: Dict[str, str] = None) -> Tuple[Optional[Dict], str]:
        """
        生成初始子任务
        
        Args:
            instruction: 完整导航指令
            observation_images: 4方向图像路径列表 [前, 左, 后, 右]
            direction_names: 方向名称列表 ['Front (0°)', 'Left (90°)', 'Back (180°)', 'Right (270°)']
            global_map_image: 全局语义地图路径（global_map/step-N.png）- 必需
            local_map_image: 局部语义地图路径（local_map/step-N.png）- 可选
            obstacle_distances: 预计算的障碍物距离字典 {'front': 'X.XXm', 'left_30': ..., ...}
            
        Returns:
            (LLM响应字典或None, prompt字符串)
        """
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None, ""
        
        # 使用预计算的距离（如果没有则设为Unknown）
        if not obstacle_distances:
            obstacle_distances = {
                'front': 'Unknown',
                'left_30': 'Unknown',
                'right_30': 'Unknown',
                'left_90': 'Unknown',
                'right_90': 'Unknown'
            }
        
        distance_summary = MapVisualizer.get_distance_summary(obstacle_distances)
        print(f"📏 [Initial Planning] Obstacle Distances: {distance_summary}")
        
        prompt = get_initial_planning_prompt(
            instruction, 
            self.action_space,
            distance_front=obstacle_distances['front'],
            distance_left_30=obstacle_distances['left_30'],
            distance_right_30=obstacle_distances['right_30'],
            distance_left_90=obstacle_distances['left_90'],
            distance_right_90=obstacle_distances['right_90']
        )
        
        # 组合图像：4方向观察 + 全局地图 + 局部地图（如果有）
        images = observation_images.copy()
        images.append(global_map_image)
        
        if local_map_image:
            images.append(local_map_image)
            print(f"  📍 Images: 4 directions + Global map + Local map")
        else:
            print(f"  📍 Images: 4 directions + Global map")
        
        # 添加重试机制
        max_retries = 3
        for retry in range(max_retries):
            response = self.call_api(prompt, images)
            
            if response and self.validate_response(response, mode='initial'):
                return response, prompt
            
            if retry < max_retries - 1:
                print(f"  ⚠️  初始规划API调用失败，重试 ({retry + 1}/{max_retries - 1})...")
                import time
                time.sleep(2)
        
        print(f"  ✗ 初始规划API调用失败，已达最大重试次数")
        return None, prompt
    
    def verify_and_replan(self,
                         instruction: str,
                         current_subtask: Dict,
                         observation_images: List[str],
                         direction_names: List[str],
                         global_map_image: str,
                         local_map_image: str = None,
                         detected_landmarks: List[str] = None,
                         waypoint_summary: str = None,
                         obstacle_distances: Dict[str, str] = None) -> Tuple[Optional[Dict], bool]:
        """
        验证子任务完成并规划下一步
        
        Args:
            instruction: 完整导航指令
            current_subtask: 当前子任务字典
            observation_images: 4方向图像路径列表（当前位置重新环视获得）
            direction_names: 方向名称列表
            global_map_image: 更新后的全局语义地图路径 - 必需
            local_map_image: 更新后的局部语义地图路径 - 可选
            detected_landmarks: 已检测到的landmark类别列表 - 可选
            waypoint_summary: 路径点历史记录 - 可选
            obstacle_distances: 预计算的障碍物距离字典 {'front': 'X.XXm', 'left_30': ..., ...}
            
        Returns:
            (response字典, is_completed标志)
        """
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None, False
        
        # 获取当前子任务信息
        waypoint_sequence = current_subtask.get('waypoint_sequence', 'Unknown')
        subtask_destination = current_subtask.get('subtask_destination', 'Unknown')
        subtask_instruction = current_subtask.get('subtask_instruction', 'Unknown')
        completion_criteria = current_subtask.get('completion_criteria', 'Unknown')
        
        # 格式化检测到的landmark信息
        landmarks_str = None
        if detected_landmarks:
            landmarks_str = f"Detected landmarks: {', '.join(sorted(detected_landmarks))}"
        
        # 使用预计算的距离（如果没有则设为Unknown）
        if not obstacle_distances:
            obstacle_distances = {
                'front': 'Unknown',
                'left_30': 'Unknown',
                'right_30': 'Unknown',
                'left_90': 'Unknown',
                'right_90': 'Unknown'
            }
        
        distance_summary = MapVisualizer.get_distance_summary(obstacle_distances)
        print(f"📏 [Verification] Obstacle Distances: {distance_summary}")
        
        prompt = get_verification_replanning_prompt(
            instruction,
            waypoint_sequence,
            subtask_destination,
            subtask_instruction,
            completion_criteria,
            self.action_space,
            detected_landmarks=landmarks_str,
            waypoint_summary=waypoint_summary,
            distance_front=obstacle_distances['front'],
            distance_left_30=obstacle_distances['left_30'],
            distance_right_30=obstacle_distances['right_30'],
            distance_left_90=obstacle_distances['left_90'],
            distance_right_90=obstacle_distances['right_90']
        )
        
        # 组合图像：当前位置4方向 + 全局地图 + 局部地图（如果有）
        images = observation_images.copy()
        images.append(global_map_image)
        
        if local_map_image:
            images.append(local_map_image)
            print(f"  📍 Images: 4 directions (updated) + Global map + Local map")
        else:
            print(f"  📍 Images: 4 directions (updated) + Global map")
        
        # 添加重试机制
        max_retries = 3
        for retry in range(max_retries):
            response = self.call_api(prompt, images)
            
            if response and self.validate_response(response, mode='verify'):
                # 连续导航模式：始终推进到下一个waypoint，不需要is_completed判断
                return response, True, prompt
            
            if retry < max_retries - 1:
                print(f"  ⚠️  验证API调用失败，重试 ({retry + 1}/{max_retries - 1})...")
                import time
                time.sleep(2)  # 等待2秒后重试
        
        print(f"  ✗ 验证API调用失败，已达最大重试次数")
        return None, False, None
