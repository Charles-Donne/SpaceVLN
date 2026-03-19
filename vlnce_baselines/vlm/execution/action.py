"""
VLM动作执行模块
===============
低层动作决策：基于视觉和地图输出具体动作
"""
import os
from typing import Dict, Tuple, Optional
from vlnce_baselines.config.core.params.actions import VALID_MOVE_METERS, VALID_TURN_DEGREES
from vlnce_baselines.config.core.params.api import (
    ACTION_IMAGE_COMPRESSION_MAX_SIZE,
    ACTION_IMAGE_COMPRESSION_QUALITY,
)
from vlnce_baselines.vlm.api.api_client import APIConfig, BaseAPIClient
from vlnce_baselines.vlm.prompts.action_prompt import get_action_execution_prompt
from vlnce_baselines.visualization.visualizer import MapVisualizer


class ActionExecutor(BaseAPIClient):
    """VLM动作执行器 - 负责低层动作决策"""
    
    REQUIRED_FIELDS = ['reasoning', 'action_analysis', 'action']  # degrees/meters/progress_summary optional
    VALID_TURN_VALUES = VALID_TURN_DEGREES
    VALID_MOVE_VALUES = VALID_MOVE_METERS
    
    def __init__(self, config_path: str = "vlnce_baselines/config/api/vlm_api_config.yaml", 
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
        """Parse merged action command strings while keeping backward compatibility."""
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
                return f"Had moved forward {actual_meters if actual_meters is not None else meters:.2f}m"
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
            if move_match:
                # 最后一段也是直行，累加距离
                current_distance = float(move_match.group(1))
                
                # 使用实际值累加
                if actual_meters is not None:
                    new_distance = current_distance + actual_meters
                else:
                    new_distance = current_distance + meters
                
                # 合并
                segments[-1] = f"had moved forward {new_distance:.2f}m"
                return ', '.join(segments)
            else:
                # 最后一段是转向，开始新段
                if actual_meters is not None:
                    new_segment = f"then moved forward {actual_meters:.2f}m"
                else:
                    new_segment = f"then moved forward {meters}m"
                return f"{current_progress}, {new_segment}"
        
        elif action_name == 'STOP':
            return f"{current_progress}, then stopped at destination"
        
        return current_progress

    
    def decide_action(self,
                     next_waypoint_destination: str,
                     subtask_instruction: str,
                     first_person_image: str,
                     action_mapping: Dict[str, int],
                     progress_summary: str = "",
                     waypoint_summary: str = "",
                     detection_image: str = None,
                     local_map_image: str = None,
                     detected_landmarks: str = None,
                     previous_action_reason: str = "",
                     pose_before: tuple = None,
                     pose_after: tuple = None,
                     obstacle_distances: Dict[str, str] = None,
                     landmark_map_info: str = None,
                     save_dir: str = None) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[Dict]]:
        """
        基于第一人称视角、检测结果和局部地图决策下一步动作
        
        Args:
            next_waypoint_destination: 下一个waypoint目的地
            subtask_instruction: 子任务指令
            first_person_image: 第一人称RGB图像路径
            action_mapping: 动作名称到ID的映射
            progress_summary: 当前子任务进度摘要
            waypoint_summary: waypoint历史摘要（含相对当前pose的方向和距离）
            detection_image: 目标检测图像路径（可选）
            local_map_image: 局部语义地图路径（可选）
            detected_landmarks: 已检测landmark类别字符串（可选）
            previous_action_reason: 上一步的action_analysis（可选）
            pose_before: 上一个动作执行前的位姿 (x, y, orientation) 单位：米和度（可选）
            pose_after: 上一个动作执行后的位姿 (x, y, orientation) 单位：米和度（可选）
            obstacle_distances: 预计算的障碍物距离字典 {'front': 'X.XXm', 'left_30': ..., ...}
            
        Returns:
            (action_id, action_name, updated_progress, full_response, degrees, meters, prompt)
        """
        # 使用预计算的距离（如果没有则设为Unknown）
        if not obstacle_distances:
            obstacle_distances = {
                'front': 'Unknown',
                'left_30': 'Unknown',
                'right_30': 'Unknown',
            }
        
        distance_summary = MapVisualizer.get_distance_summary(obstacle_distances)
        
        # 构建prompt（精简版）
        prompt = get_action_execution_prompt(
            next_waypoint_destination=next_waypoint_destination,
            subtask_instruction=subtask_instruction,
            progress_summary=progress_summary,
            waypoint_summary=waypoint_summary,
            detected_landmarks=detected_landmarks,
            previous_action_reason=previous_action_reason,
            landmark_map_info=landmark_map_info
        )
        
        # 只发Detection图（节省token，local map已移除）
        images = []
        
        if detection_image and os.path.exists(detection_image):
            images.append(detection_image)
        else:
            print(f"  [WARN] No detection image found")
        
        # 调用API（父类call_api → build_message_content → encode_image_base64 → compress_image）
        # save_dir: 在发送时同步保存压缩后的图片+prompt
        response = self.call_api(prompt, images, save_dir=save_dir)
        
        if not response:
            print("✗ No response from VLM")
            return None, None, None, None
        
        # 验诈1响应
        if not self.validate_response(response):
            return None, None, None, None
        
        parsed_action = self._parse_action_command(response)
        if parsed_action is None:
            print(f"✗ Invalid action command: {response.get('action')}")
            return None, None, None, None

        action_name, value = parsed_action
        if action_name not in action_mapping:
            print(f"✗ Invalid action: {action_name}")
            print(f"✗ Valid actions: {list(action_mapping.keys())}")
            return None, None, None, None

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
        
        # 计算实际位姿变化（如果提供了pose信息）
        actual_degrees = None
        actual_meters = None
        if pose_before is not None and pose_after is not None:
            x_before, y_before, ori_before = pose_before
            x_after, y_after, ori_after = pose_after
            
            # 计算实际转向角度变化
            # ori_before和ori_after都是角度制（degree），范围[-180, 180]
            angle_diff = ori_after - ori_before
            # 归一化到 [-180, 180]，处理跨越±180°边界的情况
            # 例如：从170°转到-170° = -340° → 归一化为20°
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360
            actual_degrees = abs(angle_diff)  # 取绝对值，因为记录的是转向幅度
            
            # 计算实际移动距离（2D欧氏距离）
            # x, y都是米制（meter）
            import math
            actual_meters = math.sqrt((x_after - x_before)**2 + (y_after - y_before)**2)
        
        # 自动生成progress_summary（系统维护，包含坐标验证）
        updated_progress = self._generate_progress_update(
            current_progress=progress_summary,
            action_name=action_name,
            degrees=degrees,
            meters=meters,
            actual_degrees=actual_degrees,
            actual_meters=actual_meters
        )
        
        # 简洁输出：动作 + 执行结果
        if action_name in ('TURN_LEFT', 'TURN_RIGHT'):
            info = f"{action_name} {degrees}°"
            if actual_degrees is not None:
                diff = abs(degrees - actual_degrees)
                ratio = actual_degrees / degrees if degrees > 0 else 0
                tag = "✓" if diff < 5 else (f"⚠{ratio*100:.0f}%" if diff < 15 else f"✗{ratio*100:.0f}%")
                info += f" → {actual_degrees:.1f}° [{tag}]"
        elif action_name == 'MOVE_FORWARD':
            info = f"{action_name} {meters}m"
            if actual_meters is not None:
                ratio = actual_meters / meters if meters > 0 else 0
                tag = "✓" if ratio > 0.9 else (f"⚠{ratio*100:.0f}%" if ratio > 0.5 else f"✗COLL {ratio*100:.0f}%")
                info += f" → {actual_meters:.2f}m [{tag}]"
        else:
            info = action_name
        print(f"  Action: {info} | {response.get('reasoning', '')[:60]}")
        
        return action_id, action_name, updated_progress, response, degrees, meters, prompt
