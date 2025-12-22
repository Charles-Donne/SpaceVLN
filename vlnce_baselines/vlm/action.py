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
        系统自动生成progress_summary（智能合并累积动作）
        
        规则：
        1. 转向累积：左转(+) 右转(-) 相互抵消，只保留净转向
        2. 直行累积：连续MOVE_FORWARD累加距离
        3. 动作分段：转向打断直行，直行打断转向
        
        示例：
        - "Turned left 90°" + TURN_RIGHT(30) → "Turned left 60°"
        - "Turned left 60°" + MOVE_FORWARD(1.5) → "Turned left 60°, moved forward 1.5m"
        - "Moved forward 1.5m" + MOVE_FORWARD(1.0) → "Moved forward 2.5m"
        
        Args:
            current_progress: 当前进度字符串
            action_name: 动作名称
            degrees: 转向角度（仅用于TURN）
            meters: 移动距离（仅用于MOVE_FORWARD）
            
        Returns:
            updated_progress: 更新后的进度字符串
        """
        import re
        
        if not current_progress or current_progress == "(Just started - no actions yet)":
            # 第一步（使用完成时态标识已完成）
            if action_name == 'TURN_LEFT':
                return f"Had turned left {int(degrees)}°"
            elif action_name == 'TURN_RIGHT':
                return f"Had turned right {int(degrees)}°"
            elif action_name == 'MOVE_FORWARD':
                return f"Had moved forward {meters}m"
            elif action_name == 'STOP':
                return "Had stopped at destination"
        
        # 解析当前进度的最后一段动作
        segments = [s.strip() for s in current_progress.split(',')]
        last_segment = segments[-1] if segments else ""
        
        # 检测最后一段是什么类型的动作（支持完成时态）
        turn_left_match = re.search(r'[Hh]ad turned left (\d+)°|[Tt]urned left (\d+)°', last_segment)
        turn_right_match = re.search(r'[Hh]ad turned right (\d+)°|[Tt]urned right (\d+)°', last_segment)
        move_match = re.search(r'[Hh]ad moved forward ([\d.]+)m|[Mm]oved forward ([\d.]+)m', last_segment)
        
        # 处理新动作
        if action_name in ['TURN_LEFT', 'TURN_RIGHT']:
            # 新动作是转向
            if turn_left_match or turn_right_match:
                # 最后一段也是转向，合并
                current_net_turn = 0
                if turn_left_match:
                    # 提取数字（可能在group(1)或group(2)）
                    current_net_turn = int(turn_left_match.group(1) or turn_left_match.group(2))  # 左转为正
                elif turn_right_match:
                    current_net_turn = -int(turn_right_match.group(1) or turn_right_match.group(2))  # 右转为负
                
                # 计算新的净转向
                if action_name == 'TURN_LEFT':
                    new_net_turn = current_net_turn + int(degrees)
                else:  # TURN_RIGHT
                    new_net_turn = current_net_turn - int(degrees)
                
                # 更新最后一段(保持完成时态)
                if new_net_turn > 0:
                    new_last_segment = f"had turned left {new_net_turn}°"
                elif new_net_turn < 0:
                    new_last_segment = f"had turned right {abs(new_net_turn)}°"
                else:
                    # 刚好抵消，移除这一段
                    if len(segments) > 1:
                        return ', '.join(segments[:-1])
                    else:
                        return "(Just started - no actions yet)"
                
                segments[-1] = new_last_segment
                return ', '.join(segments)
            else:
                # 最后一段是直行或其他，开始新段(转向打断直行)
                if action_name == 'TURN_LEFT':
                    return f"{current_progress}, then turned left {int(degrees)}°"
                else:
                    return f"{current_progress}, then turned right {int(degrees)}°"
        
        elif action_name == 'MOVE_FORWARD':
            # 新动作是直行
            if move_match:
                # 最后一段也是直行，累加距离(提取数字可能在group(1)或group(2))
                current_distance = float(move_match.group(1) or move_match.group(2))
                new_distance = current_distance + meters
                segments[-1] = f"had moved forward {new_distance}m"
                return ', '.join(segments)
            else:
                # 最后一段是转向或其他，开始新段(直行打断转向)
                return f"{current_progress}, then moved forward {meters}m"
        
        elif action_name == 'STOP':
            return f"{current_progress}, then stopped at destination"
        
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
