"""
VLM动作执行模块
===============
低层动作决策：基于视觉和地图输出具体动作
"""
import os
import re
from typing import Any, Dict, Tuple, Optional, Sequence
from navigation_system.config.core.params.actions import VALID_MOVE_METERS, VALID_TURN_DEGREES
from navigation_system.config.core.params.api import (
    ACTION_IMAGE_COMPRESSION_MAX_SIZE,
    ACTION_IMAGE_COMPRESSION_QUALITY,
)
from navigation_system.vlm.api.api_client import APIConfig, BaseAPIClient
from navigation_system.vlm.prompts.common import PromptBundle, compose_full_prompt
from navigation_system.vlm.prompts.vlnce.builders import build_executor_prompt_bundle


class Executor(BaseAPIClient):
    """VLM动作执行器 - 负责低层动作决策"""
    
    REQUIRED_FIELDS = ['reasoning', 'action']  # degrees/meters/progress_summary optional
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
        self.continuous_turn_targets = str(
            os.getenv("SPACEVLN_CONTINUOUS_TURN_TARGETS", "")
            or os.getenv("SPACEVLN_REAL_CONTINUOUS_TURN_TARGETS", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.VALID_TURN_VALUES = self._resolve_valid_turn_values(turn_angle)
        self.VALID_MOVE_VALUES = self._resolve_valid_move_values(move_distance)
        
        # 图片压缩配置（节省token）
        self.enable_compression = True
        self.compression_resolution = ACTION_IMAGE_COMPRESSION_MAX_SIZE
        self.compression_quality = ACTION_IMAGE_COMPRESSION_QUALITY
        
        print(f"  Executor: {self.config.model} | {self.compression_resolution}px Q{self.compression_quality}")
        
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
    def _normalize_waypoint_action(value: Any) -> Optional[str]:
        text = str(value or "").strip().upper()
        if not text:
            return None
        if text in {"L", "LEFT", "TURN_LEFT"}:
            return "L"
        if text in {"R", "RIGHT", "TURN_RIGHT"}:
            return "R"
        if text in {"B", "BACK", "TURN_BACK"}:
            return "B"
        if text in {"STOP", "-1"}:
            return "STOP"
        match = re.search(r"-?\d+", text)
        if match:
            return str(int(match.group(0)))
        return None

    def decide_waypoint(
        self,
        *,
        next_waypoint: str,
        subtask_instruction: str,
        subtask_landmark: str = "",
        progress_summary: str = "",
        candidates_text: str,
        first_person_image: Any = None,
        candidate_map_image: Any = None,
        obstacle_distances: Dict[str, str] = None,
        save_dir: str = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
        """Ask the VLM to select one numbered geometric waypoint or a turn/stop command."""
        if not obstacle_distances:
            obstacle_distances = {
                "front": "Unknown",
                "left_30": "Unknown",
                "right_30": "Unknown",
            }
        if not progress_summary:
            progress_summary = "Just started"

        obstacle_summary = (
            f"FRONT={obstacle_distances.get('front', 'Unknown')}; "
            f"Left 30deg={obstacle_distances.get('left_30', 'Unknown')}; "
            f"Right 30deg={obstacle_distances.get('right_30', 'Unknown')}"
        )
        system_prompt = (
            "You are the waypoint selector for Vision-Language Navigation. "
            "Choose one safe local geometric waypoint from the numbered candidates, "
            "or choose L/R/B to rotate for a better view, or STOP only when the current "
            "subtask destination is already reached. The geometry layer will plan and "
            "execute the path; do not output movement distances or coordinates. "
            "Return exactly one JSON object with keys reasoning and action."
        )
        user_prompt = (
            "# Current Subtask\n"
            f"Destination: {next_waypoint or 'unknown'}\n"
            f"Tracked Landmark: {subtask_landmark or 'none'}\n"
            f"Instruction: {subtask_instruction or 'none'}\n"
            f"Subtask Progress: {progress_summary}\n\n"
            "# Local Geometry\n"
            "The top-down candidate map is centered on the robot. The red arrow is the current pose and points forward. "
            "Green is explored free floor, black is obstacle, white is unexplored. "
            "Numbered markers are reachable local waypoint candidates from the current map.\n\n"
            f"Obstacle: {obstacle_summary}\n\n"
            "# Candidate Waypoints\n"
            f"{candidates_text or 'No valid candidates'}\n\n"
            "# Decision Rules\n"
            "- Prefer a numbered candidate that advances the current subtask and is not a backtrack.\n"
            "- Prefer forward or mildly side-front candidates when they match the destination.\n"
            "- Use L/R/B only when no listed waypoint supports the destination or the view needs reorientation.\n"
            "- Use STOP only when the current subtask destination is truly reached.\n"
            "- The action must be one listed number, L, R, B, or STOP.\n\n"
            "Return JSON only, for example: {\"reasoning\":\"candidate 2 follows the corridor toward the target\", \"action\":\"2\"}"
        )
        prompt = PromptBundle(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            full_prompt=compose_full_prompt(system_prompt, user_prompt),
        )
        images = []
        if first_person_image is not None:
            images.append(first_person_image)
        if candidate_map_image is not None:
            images.append(candidate_map_image)

        response = self.call_api(prompt, images, save_dir=save_dir)
        if not response:
            print("✗ No waypoint response from VLM")
            return None, None, prompt.full_prompt
        if not self.validate_fields(response, ("reasoning", "action")):
            return None, response, prompt.full_prompt

        action = self._normalize_waypoint_action(response.get("action"))
        if action is None:
            response["_invalid_waypoint_action"] = response.get("action")
            print(f"✗ Invalid waypoint action: {response.get('action')}")
            return None, response, prompt.full_prompt
        response["action"] = action
        print(f"  Waypoint action: {action} | {response.get('reasoning', '')[:60]}")
        return action, response, prompt.full_prompt

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

    @staticmethod
    def _resolve_valid_move_values(move_distance: float):
        try:
            base_distance = float(move_distance)
        except (TypeError, ValueError):
            base_distance = 0.25
        if base_distance >= 0.5:
            return (0.5, 0.75, 1.0, 1.25, 1.5)
        return VALID_MOVE_METERS

    @staticmethod
    def _resolve_valid_turn_values(turn_angle: float):
        try:
            angle = float(turn_angle)
        except (TypeError, ValueError):
            angle = 30.0
        if angle > 0.0 and not any(abs(angle - float(value)) <= 1e-6 for value in VALID_TURN_DEGREES):
            return (angle,)
        return VALID_TURN_DEGREES

    def _normalize_turn_value(self, value):
        if self.continuous_turn_targets:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not (1.0 <= numeric <= 180.0):
                return None
            return float(numeric)
        return self._normalize_allowed_value(value, self.VALID_TURN_VALUES)

    @staticmethod
    def _extract_action_variant(action_raw: Any) -> str:
        command = str(action_raw or "").strip().upper().replace("DEGREES", "DEG")
        if command.startswith("TURN_LEFT_AVOID") or command.startswith("TURN_RIGHT_AVOID"):
            return "avoid_obstacle"
        if command.startswith("TURN_LEFT_ALIGN") or command.startswith("TURN_RIGHT_ALIGN"):
            return "align_destination"
        return ""

    @staticmethod
    def _format_turn_action_label(action_name: str, action_variant: str) -> str:
        action_name_upper = str(action_name or "").strip().upper()
        if action_variant == "avoid_obstacle":
            return f"{action_name_upper}_AVOID"
        if action_variant == "align_destination":
            return f"{action_name_upper}_ALIGN"
        return action_name_upper

    def _parse_action_command(self, response: Dict) -> Optional[Tuple[str, float]]:
        """Parse normalized action command strings from the VLM response."""
        action_raw = str(response.get('action', '')).strip()
        if not action_raw:
            return None

        command = action_raw.upper().replace("DEGREES", "DEG").strip()

        if command == 'STOP':
            return 'STOP', 0.0

        if command.startswith('TURN_LEFT_AVOID'):
            suffix = command[len('TURN_LEFT_AVOID'):].strip().replace('DEG', '').strip()
            normalized = self._normalize_turn_value(suffix)
            if normalized is None:
                return None
            return 'TURN_LEFT', float(normalized)

        if command.startswith('TURN_LEFT_ALIGN'):
            suffix = command[len('TURN_LEFT_ALIGN'):].strip().replace('DEG', '').strip()
            normalized = self._normalize_turn_value(suffix)
            if normalized is None:
                return None
            return 'TURN_LEFT', float(normalized)

        if command.startswith('TURN_LEFT'):
            suffix = command[len('TURN_LEFT'):].strip().replace('DEG', '').strip()
            if not suffix and 'value' in response:
                normalized = self._normalize_turn_value(response.get('value'))
            else:
                normalized = self._normalize_turn_value(suffix)
            if normalized is None:
                return None
            return 'TURN_LEFT', float(normalized)

        if command.startswith('TURN_RIGHT_AVOID'):
            suffix = command[len('TURN_RIGHT_AVOID'):].strip().replace('DEG', '').strip()
            normalized = self._normalize_turn_value(suffix)
            if normalized is None:
                return None
            return 'TURN_RIGHT', float(normalized)

        if command.startswith('TURN_RIGHT_ALIGN'):
            suffix = command[len('TURN_RIGHT_ALIGN'):].strip().replace('DEG', '').strip()
            normalized = self._normalize_turn_value(suffix)
            if normalized is None:
                return None
            return 'TURN_RIGHT', float(normalized)

        if command.startswith('TURN_RIGHT'):
            suffix = command[len('TURN_RIGHT'):].strip().replace('DEG', '').strip()
            if not suffix and 'value' in response:
                normalized = self._normalize_turn_value(response.get('value'))
            else:
                normalized = self._normalize_turn_value(suffix)
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
                     subtask_landmark: str = "",
                     detection_image: Any = None,
                     detected_landmarks: str = None,
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
            waypoint_summary: 兼容保留字段，executor prompt 当前不再使用
            detection_image: 目标检测图像路径（可选）
            detected_landmarks: 已检测landmark类别字符串（可选）
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
        
        # 构建 system/user prompt bundle（standard 与 context-cache 共享同一提示结构）
        prompt = build_executor_prompt_bundle(
            next_waypoint=next_waypoint,
            subtask_instruction=subtask_instruction,
            subtask_landmark=subtask_landmark,
            progress_summary=progress_summary,
            waypoint_summary=waypoint_summary,
            detected_landmarks=detected_landmarks,
            obstacle_distances=obstacle_distances,
            landmark_map_info=landmark_map_info,
            allowed_action_names=allowed_action_names,
            move_distance=float(self.move_distance),
            turn_angle=int(self.turn_angle),
            model_name=self.config.model,
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
        action_variant = self._extract_action_variant(response.get("action"))
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
            return None, action_name, response, 0, 0.0, prompt.full_prompt

        if action_name not in action_mapping:
            print(f"✗ Invalid action: {action_name}")
            print(f"✗ Valid actions: {list(action_mapping.keys())}")
            return None, None, None, 0, 0.0, ""

        action_id = action_mapping[action_name]

        degrees = 0.0
        meters = 0.0
        if action_name in ['TURN_LEFT', 'TURN_RIGHT']:
            degrees = float(value)
            response['action'] = f"{self._format_turn_action_label(action_name, action_variant)} {degrees:g}deg"
        elif action_name == 'MOVE_FORWARD':
            meters = float(value)
            response['action'] = f"{action_name} {meters:g}m"
        elif action_name == 'STOP':
            response['action'] = 'STOP'
        
        # 简洁输出：动作 + 执行结果
        if action_name in ('TURN_LEFT', 'TURN_RIGHT'):
            info = f"{action_name} {degrees:g}°"
        elif action_name == 'MOVE_FORWARD':
            info = f"{action_name} {meters}m"
        else:
            info = action_name
        print(f"  Action: {info} | {response.get('reasoning', '')[:60]}")

        return action_id, action_name, response, degrees, meters, prompt.full_prompt

    def prepare_action_request_artifacts(
        self,
        next_waypoint: str,
        subtask_instruction: str,
        first_person_image: Any,
        progress_summary: str = "",
        waypoint_summary: str = "",
        subtask_landmark: str = "",
        detection_image: Any = None,
        detected_landmarks: str = None,
        obstacle_distances: Dict[str, str] = None,
        landmark_map_info: str = None,
        allowed_action_names: Optional[Sequence[str]] = None,
        save_dir: str = None,
    ) -> Dict[str, Any]:
        """Build and save the action VLM request without calling the VLM API."""
        if not obstacle_distances:
            obstacle_distances = {
                'front': 'Unknown',
                'left_30': 'Unknown',
                'right_30': 'Unknown',
            }

        prompt = build_executor_prompt_bundle(
            next_waypoint=next_waypoint,
            subtask_instruction=subtask_instruction,
            subtask_landmark=subtask_landmark,
            progress_summary=progress_summary,
            waypoint_summary=waypoint_summary,
            detected_landmarks=detected_landmarks,
            obstacle_distances=obstacle_distances,
            landmark_map_info=landmark_map_info,
            allowed_action_names=allowed_action_names,
            move_distance=float(self.move_distance),
            turn_angle=int(self.turn_angle),
            model_name=self.config.model,
        )

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

        artifact_records = []
        self._set_last_response_artifacts(response_text="", parsed_payload=None)
        if save_dir and bool(getattr(self, "save_request_artifacts", False)):
            if getattr(self.config, "wire_api", "chat") == "responses":
                self.build_responses_input_content(
                    prompt.user_prompt,
                    images,
                    save_dir=save_dir,
                    no_compress_indices=None,
                    prompt_artifact_filename="user_prompt.md",
                    artifact_records=artifact_records,
                )
            else:
                self.build_message_content(
                    prompt.user_prompt,
                    images,
                    save_dir=save_dir,
                    no_compress_indices=None,
                    prompt_artifact_filename="user_prompt.md",
                    artifact_records=artifact_records,
                )
            if getattr(prompt, "system_prompt", ""):
                self._save_text_artifact(save_dir, "system_prompt.md", prompt.system_prompt)
            self._save_vlm_info_artifact(
                save_dir,
                self._build_vlm_info_payload(
                    usage={},
                    latency_s=0.0,
                    success=True,
                    extra={
                        "manual_prompt_only": True,
                        "request_artifacts": artifact_records,
                    },
                ),
            )

        return {
            "prompt": prompt.full_prompt,
            "system_prompt": prompt.system_prompt,
            "user_prompt": prompt.user_prompt,
            "images": images,
            "save_dir": save_dir,
            "artifact_records": artifact_records,
        }

    def _decide_action_from_prompt(
        self,
        *,
        prompt: Any,
        first_person_image: Any,
        detection_image: Any,
        action_mapping: Dict[str, int],
        allowed_action_names: Optional[Sequence[str]],
        save_dir: str,
    ) -> Tuple[Optional[int], Optional[str], Optional[Dict], int, float, str]:
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

        response = self.call_api(prompt, images, save_dir=save_dir)

        if not response:
            print("✗ No response from VLM")
            return None, None, None, 0, 0.0, ""

        if not self.validate_response(response):
            return None, None, None, 0, 0.0, ""

        parsed_action = self._parse_action_command(response)
        if parsed_action is None:
            print(f"✗ Invalid action command: {response.get('action')}")
            return None, None, None, 0, 0.0, ""

        action_name, value = parsed_action
        action_variant = self._extract_action_variant(response.get("action"))
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
            return None, action_name, response, 0, 0.0, getattr(prompt, "full_prompt", prompt)

        if action_name not in action_mapping:
            print(f"✗ Invalid action: {action_name}")
            print(f"✗ Valid actions: {list(action_mapping.keys())}")
            return None, None, None, 0, 0.0, ""

        action_id = action_mapping[action_name]

        degrees = 0
        meters = 0
        if action_name in ['TURN_LEFT', 'TURN_RIGHT']:
            degrees = int(value)
            response['action'] = f"{self._format_turn_action_label(action_name, action_variant)} {degrees}deg"
        elif action_name == 'MOVE_FORWARD':
            meters = float(value)
            response['action'] = f"{action_name} {meters:g}m"
        elif action_name == 'STOP':
            response['action'] = 'STOP'

        if action_name in ('TURN_LEFT', 'TURN_RIGHT'):
            info = f"{action_name} {degrees}°"
        elif action_name == 'MOVE_FORWARD':
            info = f"{action_name} {meters}m"
        else:
            info = action_name
        print(f"  Action: {info} | {response.get('reasoning', '')[:60]}")

        return action_id, action_name, response, degrees, meters, getattr(prompt, "full_prompt", prompt)
