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
    
    REQUIRED_FIELDS_INITIAL = ['next_waypoint_direction', 'next_waypoint_destination', 'subtask_instruction', 'completion_criteria']
    REQUIRED_FIELDS_VERIFY = ['next_waypoint_direction', 'next_waypoint_destination', 'subtask_instruction', 'completion_criteria']
    
    def __init__(self, config_path: str = "vlnce_baselines/vlm/llm_config.yaml", 
                 action_space: str = None):
        """
        初始化LLM规划器
        
        Args:
            config_path: LLM配置文件路径
            action_space: 动作空间描述（如 "MOVE_FORWARD (0.25m), TURN_LEFT (30°), ..."）
        """
        config = APIConfig(config_path, role="llm")
        super().__init__(config)
        
        # 默认动作空间与interactive_navigation一致
        self.action_space = action_space or "MOVE_FORWARD (0.25m), TURN_LEFT (30°), TURN_RIGHT (30°), STOP"
        
        # 方向观察图压缩（节省token），global_map保持全分辨率（需要细节）
        self.set_compression_config(enabled=True, max_size=512, quality=75)
        
        # print(f"✓ LLM Planner initialized")
        print(f"  LLMPlanner: {self.config.model}")
    
    def validate_response(self, response: Dict, mode: str = 'initial') -> bool:
        """验证响应字段"""
        required = self.REQUIRED_FIELDS_INITIAL if mode == 'initial' else self.REQUIRED_FIELDS_VERIFY
        
        # 验证基础字段
        if not self.validate_fields(response, required):
            return False
        
        # 验证completion_criteria为字符串即可（描述到达后的状态）
        criteria = response.get('completion_criteria')
        if not criteria or not isinstance(criteria, str):
            print(f"[WARN] completion_criteria should be string")
            return False
        
        return True
    
    def generate_initial_subtask(self,
                                instruction: str,
                                observation_images: List[str],
                                direction_names: List[str],
                                global_map_image: str,
                                local_map_image: str = None,
                                obstacle_distances: Dict[str, str] = None,
                                save_dir: str = None) -> Tuple[Optional[Dict], str]:
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
        
        prompt = get_initial_planning_prompt(
            instruction, 
            self.action_space
        )
        
        # 组合图像：4方向观察 + 全局地图 + 局部地图（如果有）
        images = observation_images.copy()
        images.append(global_map_image)
        
        if local_map_image:
            images.append(local_map_image)
        
        # global_map不压缩（需要全分辨率），其余图片压缩
        no_compress = {len(observation_images)}  # global_map的索引
        
        # 添加重试机制
        max_retries = 3
        for retry in range(max_retries):
            # Only save on first attempt (avoid duplicate saves on retry)
            response = self.call_api(prompt, images, save_dir=save_dir if retry == 0 else None,
                                     no_compress_indices=no_compress)
            
            if response and self.validate_response(response, mode='initial'):
                return response, prompt
            
            if retry < max_retries - 1:
                wait = (retry + 1) * 2  # 2s, 4s 递增等待
                print(f"  [WARN] LLM Planning failed, retry {retry+1}/{max_retries-1} in {wait}s...")
                import time
                time.sleep(wait)
        
        print(f"  [ERR] LLM Planning failed after {max_retries} attempts")
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
                         obstacle_distances: Dict[str, str] = None,
                         save_dir: str = None) -> Tuple[Optional[Dict], bool]:
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
        subtask_destination = current_subtask.get('next_waypoint_destination', 'Unknown')
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
        
        prompt = get_verification_replanning_prompt(
            instruction,
            subtask_destination,
            subtask_instruction,
            completion_criteria,
            self.action_space,
            detected_landmarks=landmarks_str,
            waypoint_summary=waypoint_summary
        )
        
        # 组合图像：当前位置4方向 + 全局地图 + 局部地图（如果有）
        images = observation_images.copy()
        images.append(global_map_image)
        
        if local_map_image:
            images.append(local_map_image)
        
        # global_map不压缩（需要全分辨率），其余图片压缩
        no_compress = {len(observation_images)}  # global_map的索引
        
        # 添加重试机制
        max_retries = 3
        for retry in range(max_retries):
            response = self.call_api(prompt, images, save_dir=save_dir if retry == 0 else None,
                                     no_compress_indices=no_compress)
            
            if response and self.validate_response(response, mode='verify'):
                return response, prompt
            
            if retry < max_retries - 1:
                wait = (retry + 1) * 2
                print(f"  [WARN] LLM Verify failed, retry {retry+1}/{max_retries-1} in {wait}s...")
                import time
                time.sleep(wait)
        
        print(f"  [ERR] LLM Verify failed after {max_retries} attempts")
        return None, None