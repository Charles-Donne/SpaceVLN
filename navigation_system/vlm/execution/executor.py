"""
VLM动作执行模块
===============
低层动作决策：基于视觉和地图输出具体动作
"""
import os
from typing import Any, Dict, Tuple, Optional, Sequence
from navigation_system.config.core.params.actions import VALID_MOVE_METERS, VALID_TURN_DEGREES
from navigation_system.config.core.params.api import (
    ACTION_IMAGE_COMPRESSION_MAX_SIZE,
    ACTION_IMAGE_COMPRESSION_QUALITY,
)
from navigation_system.vlm.api.api_client import APIConfig, BaseAPIClient
from navigation_system.vlm.prompts.builders import get_action_execution_prompt


class ActionExecutor(BaseAPIClient):
    """VLM动作执行器 - 负责低层动作决策"""
    
    REQUIRED_FIELDS = ['reasoning', 'action_analysis', 'action']  # degrees/meters/progress_summary optional
    VALID_TURN_VALUES = VALID_TURN_DEGREES
    VALID_MOVE_VALUES = VALID_MOVE_METERS
    
    def __init__(self, config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml", 
                 turn_angle: float = 30.0, 
                 move_distance: float = 0.25):
        """
        初始化动作执行器
        
        Args:
            config_path: VLM配置文件路径
            turn_angle: 每次转向角度（度）- 与interactive_navigation一致：30°
            move_distance: 每次前进距离（米）- 与interactive_navigation一致：0.25m
        """
        config = APIConfig(config_path, role="vlm")
        super().__init__(config)
        
        self.turn_angle = turn_angle
        self.move_distance = move_distance
        
        # 图片压缩配置（节省token）
        self.enable_compression = True
        self.compression_resolution = ACTION_IMAGE_COMPRESSION_MAX_SIZE
        self.compression_quality = ACTION_IMAGE_COMPRESSION_QUALITY
        
        print(f"  ActionVLM: {self.config.model} | {self.compression_resolution}px Q{self.compression_quality}")
        
        # 配置父类的压缩参数（父类的encode_image_base64会自动使用）
        self.set_compression_config(
            enabled=self.enable_compression,
            max_size=self.compression_resolution,
            quality=self.compression_quality
        )
    
    def validate_response(self, response: Dict) -> bool:
        """验证VLM响应是否包含所有必需字段和受限动作空间。"""
        if not self.validate_fields(response, self.REQUIRED_FIELDS):
            return False

        parsed = self._parse_action_command(response)
        return parsed is not None

    @staticmethod
    def _normalize_allowed_value(value, allowed_values, tol: float = 1e-6):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

        for allowed in allowed_values:
            if abs(numeric - float(allowed)) <= tol:
                return float(allowed)
        return None

    def _parse_action_command(self, response: Dict) -> Optional[Tuple[str, float]]:
        """Parse normalized action command strings from the VLM response."""
        action_raw = str(response.get('action', '')).strip()
        if not action_raw:
            return None

        command = action_raw.upper().replace("DEGREES", "DEG").strip()

        if command == 'STOP':
            return 'STOP', 0.0

        if command.startswith('TURN_LEFT'):
            suffix = command[len('TURN_LEFT'):].strip().replace('DEG', '').strip()
            if not suffix and 'value' in response:
                normalized = self._normalize_allowed_value(response.get('value'), self.VALID_TURN_VALUES)
            else:
                normalized = self._normalize_allowed_value(suffix, self.VALID_TURN_VALUES)
            if normalized is None:
                return None
            return 'TURN_LEFT', float(normalized)

        if command.startswith('TURN_RIGHT'):
            suffix = command[len('TURN_RIGHT'):].strip().replace('DEG', '').strip()
            if not suffix and 'value' in response:
                normalized = self._normalize_allowed_value(response.get('value'), self.VALID_TURN_VALUES)
            else:
                normalized = self._normalize_allowed_value(suffix, self.VALID_TURN_VALUES)
            if normalized is None:
                return None
            return 'TURN_RIGHT', float(normalized)

        if command.startswith('MOVE_FORWARD'):
            suffix = command[len('MOVE_FORWARD'):].strip().replace('M', '').strip()
            if not suffix and 'value' in response:
                normalized = self._normalize_allowed_value(response.get('value'), self.VALID_MOVE_VALUES)
            else:
                normalized = self._normalize_allowed_value(suffix, self.VALID_MOVE_VALUES)
            if normalized is None:
                return None
            return 'MOVE_FORWARD', float(normalized)

        return None
    
    def _generate_progress_update(self, current_progress: str, action_name: str, 
                                  degrees: float = 0, meters: float = 0,
                                  actual_degrees: float = None, actual_meters: float = None) -> str:
        """
        系统自动生成progress_summary（只记录实际值）
        
        规则：
        1. 只记录实际值 "Turned left 88°", "Moved forward 0.47m"
        2. 转向累积：左转(+) 右转(-) 相互抵消，只保留净转向
        3. 直行累积：连续MOVE_FORWARD累加距离
        4. 动作分段：转向打断直行，直行打断转向
        
        示例：
        - "Turned left 88°, moved forward 0.47m"
        - "Moved 0.5m, then moved 0.3m" → "Moved 0.8m"
        
        Args:
            current_progress: 当前进度字符串
            action_name: 动作名称
            degrees: 计划转向角度（不使用）
            meters: 计划移动距离（不使用）
            actual_degrees: 实际转向角度（必需，记录实际值）
            actual_meters: 实际移动距离（必需，记录实际值）
            
        Returns:
            updated_progress: 更新后的进度字符串
        """
        import re
        
        if not current_progress or current_progress == "(Just started - no actions yet)":
            # 第一步（使用完成时态 Had）
            if action_name == 'TURN_LEFT':
                return f"Had turned left {int(actual_degrees if actual_degrees is not None else degrees)}°"
            elif action_name == 'TURN_RIGHT':
                return f"Had turned right {int(actual_degrees if actual_degrees is not None else degrees)}°"
            elif action_name == 'MOVE_FORWARD':
                moved_m = actual_meters if actual_meters is not None else meters
                if float(moved_m or 0.0) <= 0.0:
                    return "(Just started - no actions yet)"
                return f"Had moved forward {moved_m:.2f}m"
            elif action_name == 'STOP':
                return "Had stopped at destination"
        
        # 解析当前进度的最后一段动作
        segments = [s.strip() for s in current_progress.split(',')]
        last_segment = segments[-1] if segments else ""
        
        # 检测最后一段是什么类型的动作
        turn_left_match = re.search(r'[Tt]urned left (\d+)°', last_segment)
        turn_right_match = re.search(r'[Tt]urned right (\d+)°', last_segment)
        move_match = re.search(r'[Mm]oved forward ([\d.]+)m', last_segment)
        
        # 处理新动作
        if action_name in ['TURN_LEFT', 'TURN_RIGHT']:
            # 新动作是转向
            if (turn_left_match or turn_right_match):
                # 最后一段也是转向，可以合并
                current_net_turn = 0
                if turn_left_match:
                    current_net_turn = int(turn_left_match.group(1))  # 左转为正
                elif turn_right_match:
                    current_net_turn = -int(turn_right_match.group(1))  # 右转为负
                
                # 计算新的净转向（使用实际值）
                if action_name == 'TURN_LEFT':
                    if actual_degrees is not None:
                        new_net_turn = current_net_turn + int(actual_degrees)
                    else:
                        new_net_turn = current_net_turn + int(degrees)
                else:  # TURN_RIGHT
                    if actual_degrees is not None:
                        new_net_turn = current_net_turn - int(actual_degrees)
                    else:
                        new_net_turn = current_net_turn - int(degrees)
                
                # 更新最后一段
                if new_net_turn > 0:
                    segments[-1] = f"had turned left {new_net_turn}°"
                elif new_net_turn < 0:
                    segments[-1] = f"had turned right {abs(new_net_turn)}°"
                else:
                    # 刚好抵消，移除这一段
                    if len(segments) > 1:
                        return ', '.join(segments[:-1])
                    else:
                        return "(Just started - no actions yet)"
                
                return ', '.join(segments)
            else:
                # 最后一段是直行，开始新段
                if actual_degrees is not None:
                    if action_name == 'TURN_LEFT':
                        new_segment = f"then turned left {int(actual_degrees)}°"
                    else:
                        new_segment = f"then turned right {int(actual_degrees)}°"
                else:
                    if action_name == 'TURN_LEFT':
                        new_segment = f"then turned left {int(degrees)}°"
                    else:
                        new_segment = f"then turned right {int(degrees)}°"
                return f"{current_progress}, {new_segment}"
        
        elif action_name == 'MOVE_FORWARD':
            # 新动作是直行
            moved_m = actual_meters if actual_meters is not None else meters
            if float(moved_m or 0.0) <= 0.0:
                return current_progress
            if move_match:
                # 最后一段也是直行，累加距离
                current_distance = float(move_match.group(1))
                
                # 使用实际值累加
                new_distance = current_distance + moved_m
                
                # 合并
                segments[-1] = f"had moved forward {new_distance:.2f}m"
                return ', '.join(segments)
            else:
                # 最后一段是转向，开始新段
                new_segment = f"then moved forward {moved_m:.2f}m"
                return f"{current_progress}, {new_segment}"
        
        elif action_name == 'STOP':
            return f"{current_progress}, then stopped at destination"
        
        return current_progress

    
    def decide_action(self,
                     next_waypoint: str,
                     subtask_instruction: str,
                     first_person_image: Any,
                     action_mapping: Dict[str, int],
                     progress_summary: str = "",
                     waypoint_summary: str = "",
                     detection_image: Any = None,
                     detected_landmarks: str = None,
                     previous_action_reason: str = "",
                     controller_action_notice: str = "",
                     obstacle_distances: Dict[str, str] = None,
                     landmark_map_info: str = None,
                     allowed_action_names: Optional[Sequence[str]] = None,
                     save_dir: str = None) -> Tuple[Optional[int], Optional[str], Optional[Dict], int, float, str]:
        """
        基于第一人称视角、检测结果和局部地图决策下一步动作
        
        Args:
            next_waypoint: 下一个waypoint目的地
            subtask_instruction: 子任务指令
            first_person_image: 第一人称RGB图像路径
            action_mapping: 动作名称到ID的映射
            progress_summary: 当前子任务进度摘要
            waypoint_summary: 兼容保留字段，action prompt当前不再使用
            detection_image: 目标检测图像路径（可选）
            detected_landmarks: 已检测landmark类别字符串（可选）
            previous_action_reason: 上一步的action_analysis（可选）
            controller_action_notice: 当前这一步必须 obey 的控制器约束（可选）
            obstacle_distances: 预计算的障碍物距离字典 {'front': 'X.XXm', 'left_30': ..., ...}
            
        Returns:
            (action_id, action_name, full_response, degrees, meters, prompt)
        """
        # 使用预计算的距离（如果没有则设为Unknown）
        if not obstacle_distances:
            obstacle_distances = {
                'front': 'Unknown',
                'left_30': 'Unknown',
                'right_30': 'Unknown',
            }
        
        # 构建prompt（精简版）
        prompt = get_action_execution_prompt(
            next_waypoint=next_waypoint,
            subtask_instruction=subtask_instruction,
            progress_summary=progress_summary,
            waypoint_summary=waypoint_summary,
            detected_landmarks=detected_landmarks,
            previous_action_reason=previous_action_reason,
            controller_action_notice=controller_action_notice,
            obstacle_distances=obstacle_distances,
            landmark_map_info=landmark_map_info,
            allowed_action_names=allowed_action_names,
        )
        
        # 只发Detection图（节省token，local map已移除）
        images = []
        action_image_input = detection_image if detection_image is not None else first_person_image
        if isinstance(action_image_input, str):
            if action_image_input and os.path.exists(action_image_input):
                images.append(action_image_input)
            else:
                print(f"  [WARN] No detection image found")
        elif action_image_input is not None:
            images.append(action_image_input)
        else:
            print(f"  [WARN] No detection image found")
        
        # 调用API（父类call_api → build_message_content → encode_image_base64 → compress_image）
        # save_dir: 在发送时同步保存压缩后的图片+prompt
        response = self.call_api(prompt, images, save_dir=save_dir)
        
        if not response:
            print("✗ No response from VLM")
            return None, None, None, 0, 0.0, ""
        
        # 验诈1响应
        if not self.validate_response(response):
            return None, None, None, 0, 0.0, ""
        
        parsed_action = self._parse_action_command(response)
        if parsed_action is None:
            print(f"✗ Invalid action command: {response.get('action')}")
            return None, None, None, 0, 0.0, ""

        action_name, value = parsed_action
        normalized_allowed_actions = None
        if allowed_action_names:
            normalized_allowed_actions = {
                str(name or "").strip().upper()
                for name in allowed_action_names
                if str(name or "").strip()
            }
        if normalized_allowed_actions and action_name not in normalized_allowed_actions:
            print(
                f"✗ Forbidden action under current constraint: {action_name} | "
                f"Allowed: {sorted(normalized_allowed_actions)}"
            )
            response["_forbidden_action_name"] = action_name
            response["_allowed_action_names"] = sorted(normalized_allowed_actions)
            return None, action_name, response, 0, 0.0, prompt

        if action_name not in action_mapping:
            print(f"✗ Invalid action: {action_name}")
            print(f"✗ Valid actions: {list(action_mapping.keys())}")
            return None, None, None, 0, 0.0, ""

        action_id = action_mapping[action_name]

        degrees = 0
        meters = 0
        if action_name in ['TURN_LEFT', 'TURN_RIGHT']:
            degrees = int(value)
            response['action'] = f"{action_name} {degrees}deg"
        elif action_name == 'MOVE_FORWARD':
            meters = float(value)
            response['action'] = f"{action_name} {meters:g}m"
        elif action_name == 'STOP':
            response['action'] = 'STOP'
        
        # 简洁输出：动作 + 执行结果
        if action_name in ('TURN_LEFT', 'TURN_RIGHT'):
            info = f"{action_name} {degrees}°"
        elif action_name == 'MOVE_FORWARD':
            info = f"{action_name} {meters}m"
        else:
            info = action_name
        print(f"  Action: {info} | {response.get('reasoning', '')[:60]}")

        return action_id, action_name, response, degrees, meters, prompt
