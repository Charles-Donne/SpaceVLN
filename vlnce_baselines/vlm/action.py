"""
VLM动作执行模块
===============
低层动作决策：基于视觉和地图输出具体动作
"""
from typing import Dict, Tuple, Optional
from vlnce_baselines.vlm.api_client import APIConfig, BaseAPIClient
from vlnce_baselines.vlm.action_prompt import get_action_execution_prompt


class ActionExecutor(BaseAPIClient):
    """VLM动作执行器 - 负责低层动作决策"""
    
    REQUIRED_FIELDS = ['reasoning', 'action']  # 移除progress_summary，由系统自动生成
    
    def __init__(self, config_path: str = "vlnce_baselines/vlm/vlm_config.yaml", 
                 turn_angle: float = 30.0, 
                 move_distance: float = 0.25):
        """
        初始化动作执行器
        
        Args:
            config_path: VLM配置文件路径
            turn_angle: 每次转向角度（度）- 与interactive_navigation一致：30°
            move_distance: 每次前进距离（米）- 与interactive_navigation一致：0.25m
        """
        config = APIConfig(config_path)
        super().__init__(config)
        
        self.turn_angle = turn_angle
        self.move_distance = move_distance
        
        print(f"✓ Action Executor initialized")
        print(f"  Model: {self.config.model}")
        print(f"  Parameters: turn={turn_angle}°, move={move_distance}m")
    
    def validate_response(self, response: Dict) -> bool:
        """验证VLM响应是否包含所有必需字段"""
        return self.validate_fields(response, self.REQUIRED_FIELDS)
    
    def _generate_progress_update(self, current_progress: str, action_name: str, 
                                  degrees: float = 0, meters: float = 0) -> str:
        """
        系统自动生成progress_summary（根据执行的动作累积更新）
        
        Args:
            current_progress: 当前进度字符串
            action_name: 动作名称
            degrees: 转向角度（仅用于TURN）
            meters: 移动距离（仅用于MOVE_FORWARD）
            
        Returns:
            updated_progress: 更新后的进度字符串
        """
        if not current_progress or current_progress == "(Just started - no actions yet)":
            # 第一步
            if action_name == 'TURN_LEFT':
                return f"Turned left {degrees}°"
            elif action_name == 'TURN_RIGHT':
                return f"Turned right {degrees}°"
            elif action_name == 'MOVE_FORWARD':
                return f"Moved forward {meters}m"
            elif action_name == 'STOP':
                return "Stopped at destination"
        else:
            # 累积更新
            if action_name == 'TURN_LEFT':
                return f"{current_progress}, turned left {degrees}°"
            elif action_name == 'TURN_RIGHT':
                return f"{current_progress}, turned right {degrees}°"
            elif action_name == 'MOVE_FORWARD':
                return f"{current_progress}, moved forward {meters}m"
            elif action_name == 'STOP':
                return f"{current_progress}, stopped at destination"
        
        return current_progress

    
    def decide_action(self,
                     subtask_destination: str,
                     subtask_instruction: str,
                     first_person_image: str,
                     action_mapping: Dict[str, int],
                     progress_summary: str = "",
                     detection_image: str = None,
                     local_map_image: str = None,
                     detected_landmarks: str = None) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[Dict]]:
        """
        基于第一人称视角、检测结果和局部地图决策下一步动作
        
        Args:
            subtask_destination: 子任务目的地
            subtask_instruction: 子任务指令
            first_person_image: 第一人称RGB图像路径
            action_mapping: 动作名称到ID的映射
            progress_summary: 当前子任务进度摘要
            detection_image: 目标检测图像路径（可选）
            local_map_image: 局部语义地图路径（可选）
            detected_landmarks: 已检测landmark类别字符串（可选）
            
        Returns:
            (action_id, action_name, updated_progress, full_response)
        """
        # 构建prompt
        prompt = get_action_execution_prompt(
            subtask_destination=subtask_destination,
            subtask_instruction=subtask_instruction,
            progress_summary=progress_summary,
            detected_landmarks=detected_landmarks
        )
        
        # 组合图像：RGB + Detection + Local Map
        images = [first_person_image]
        if detection_image:
            images.append(detection_image)
        if local_map_image:
            images.append(local_map_image)
        
        # 调用API
        response = self.call_api(prompt, images)
        
        if not response:
            print("✗ No response from VLM")
            return None, None, None, None
        
        # 验诈1响应
        if not self.validate_response(response):
            return None, None, None, None
        
        # 提取动作
        action_name = response['action']
        if action_name not in action_mapping:
            print(f"✗ Invalid action: {action_name}")
            print(f"✗ Valid actions: {list(action_mapping.keys())}")
            return None, None, None, None
        
        action_id = action_mapping[action_name]
        
        # 自动生成progress_summary（系统维护，不依赖模型输出）
        updated_progress = self._generate_progress_update(
            current_progress=progress_summary,
            action_name=action_name,
            degrees=response.get('degrees', 0) if action_name in ['TURN_LEFT', 'TURN_RIGHT'] else 0,
            meters=response.get('meters', 0) if action_name == 'MOVE_FORWARD' else 0
        )
        
        # 提取degrees/meters参数（用于计算重复次数）
        degrees = response.get('degrees', 0) if action_name in ['TURN_LEFT', 'TURN_RIGHT'] else 0
        meters = response.get('meters', 0) if action_name == 'MOVE_FORWARD' else 0
        
        # 打印推理过程
        print(f"Reasoning: {response['reasoning']}")
        if action_name == 'TURN_LEFT' or action_name == 'TURN_RIGHT':
            print(f"Action: {action_name} {degrees}°")
        elif action_name == 'MOVE_FORWARD':
            print(f"Action: {action_name} {meters}m")
        else:
            print(f"Action: {action_name}")
        
        return action_id, action_name, updated_progress, response, degrees, meters, prompt
