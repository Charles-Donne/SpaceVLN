"""
LLM规划模块
===========
高层规划：分析环境生成子任务
"""
import os
import re
import time
from typing import Any, Dict, List, Tuple, Optional
from navigation_system.config.core.params.api import THINKING_IMAGE_COMPRESSION_MAX_SIZE
from navigation_system.vlm.api.api_client import APIConfig, BaseAPIClient
from navigation_system.vlm.prompts.vlnce.builders import (
    build_initial_planner_prompt_bundle,
    build_verify_planner_prompt_bundle,
)
from navigation_system.vlm.prompts.common import (
    PromptLike,
    extract_prompt_debug_text,
)
from navigation_system.vlm.contracts.schema import (
    REQUIRED_SUBTASK_FIELDS,
    get_next_waypoint,
    normalize_subtask_payload,
)

DEFAULT_PLANNER_MAX_RETRIES = 3
DEFAULT_INITIAL_PLANNER_MAX_RETRIES = 5
DEFAULT_VERIFY_PLANNER_MAX_RETRIES = 5
API_NO_RESPONSE_FAILURE_KINDS = {
    "api_error",
    "empty_response",
    "http_error",
    "json_decode_error",
    "json_parse_failed",
    "pending",
    "timeout",
    "unknown",
    "uninitialized",
}


class LLMPlanner(BaseAPIClient):
    """LLM规划器 - 负责子任务生成和验证"""

    def _normalize_response_payload(
        self,
        payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return normalize_subtask_payload(payload)

    def __init__(self, config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml", 
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
        
        # 方向观察图压缩，global_map 保持全分辨率。
        self.set_compression_config(
            enabled=True,
            max_size=THINKING_IMAGE_COMPRESSION_MAX_SIZE,
            quality=75,
        )
        self.last_call_timing_info = {
            "records": [],
            "failed_retry_wait_duration_s": 0.0,
            "final_failure_reason": "",
        }
        
        # print(f"✓ LLM Planner initialized")
        print(f"  LLMPlanner: {self.config.model}")

    def _reset_last_call_timing_info(self) -> None:
        self.last_call_timing_info = {
            "records": [],
            "failed_retry_wait_duration_s": 0.0,
            "final_failure_reason": "",
        }

    @staticmethod
    def _classify_attempt_failure(
        *,
        response: Optional[Dict[str, Any]],
        normalized_response: Optional[Dict[str, Any]],
        is_valid: bool,
        direction_is_available: bool,
        api_status: str,
    ) -> str:
        normalized_status = str(api_status or "").strip() or "unknown"
        if response is None:
            return normalized_status
        if not normalized_response:
            return "invalid_payload"
        if not is_valid:
            return "invalid_response"
        if not direction_is_available:
            return "invalid_direction"
        return "unknown"

    def _summarize_final_failure_reason(self) -> str:
        failed_records = [
            dict(record)
            for record in list(self.last_call_timing_info.get("records", []) or [])
            if not bool(record.get("success", False))
        ]
        if not failed_records:
            return ""
        failure_kinds = [
            str(record.get("failure_kind") or "").strip()
            for record in failed_records
            if str(record.get("failure_kind") or "").strip()
        ]
        if failure_kinds and all(kind == "timeout" for kind in failure_kinds):
            return "planner_timeout"
        if failure_kinds and all(kind in API_NO_RESPONSE_FAILURE_KINDS for kind in failure_kinds):
            return "planner_no_response"
        return "planner_failed"

    @staticmethod
    def _read_positive_int_env(name: str, default: int) -> int:
        raw_value = str(os.getenv(name, "") or "").strip()
        if not raw_value:
            return int(default)
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            return int(default)
        return max(1, parsed)

    def _resolve_planner_max_retries(self, mode: str) -> int:
        mode_key = str(mode or "").strip().lower()
        default = (
            DEFAULT_INITIAL_PLANNER_MAX_RETRIES
            if mode_key == "initial"
            else DEFAULT_VERIFY_PLANNER_MAX_RETRIES
            if mode_key == "verify"
            else DEFAULT_PLANNER_MAX_RETRIES
        )
        mode_env_name = f"SPACEVLN_{mode_key.upper()}_PLANNER_MAX_RETRIES" if mode_key else ""
        if mode_env_name and str(os.getenv(mode_env_name, "") or "").strip():
            return self._read_positive_int_env(mode_env_name, default)
        return self._read_positive_int_env("SPACEVLN_PLANNER_MAX_RETRIES", default)

    def _finalize_vlm_info_retry_summary(self, save_dir: Optional[str]) -> None:
        """Patch the saved vlm_info.json with aggregate retry stats for this call."""
        if not save_dir:
            return

        records = list(self.last_call_timing_info.get("records", []) or [])
        if not records:
            return

        failed_attempts = [record for record in records if not bool(record.get("success", False))]
        failed_api_duration_s = sum(
            max(0.0, float(record.get("duration_s", 0.0) or 0.0))
            for record in failed_attempts
        )
        failed_retry_wait_duration_s = max(
            0.0,
            float(self.last_call_timing_info.get("failed_retry_wait_duration_s", 0.0) or 0.0),
        )
        self._update_vlm_info_artifact(
            save_dir,
            {
                "attempts": len(records),
                "failed_attempts": len(failed_attempts),
                "failed_retry_wait_time_s": round(failed_retry_wait_duration_s, 4),
                "failed_wasted_time_s": round(
                    failed_api_duration_s + failed_retry_wait_duration_s,
                    4,
                ),
            },
        )
    
    def validate_response(self, response: Dict, mode: str = 'initial') -> bool:
        """验证响应字段"""
        self._last_response_rejection_notice = ""
        response = self._normalize_response_payload(response)
        if not response:
            self._last_response_rejection_notice = "Your previous response was rejected because it was empty or invalid JSON."
            return False

        # 验证基础字段
        if not self.validate_fields(response, REQUIRED_SUBTASK_FIELDS):
            self._last_response_rejection_notice = (
                "Your previous response was rejected because required JSON fields were missing. "
                "Return the exact required schema only."
            )
            return False

        current_waypoint = str(response.get('current_waypoint', '') or '').strip()
        if not current_waypoint:
            self._last_response_rejection_notice = (
                "Your previous response was rejected because `current_waypoint` was empty. "
                "Infer the current space from the current observations and provide a concrete `space - landmarks` anchor."
            )
            return False

        for field_name in ("current_waypoint", "next_waypoint", "waypoint_chain", "waypoint_sequence"):
            field_text = str(response.get(field_name, "") or "").strip()
            if re.search(r"(?i)(^|[^a-z0-9])unknown(?:'s|\b)", field_text):
                print(f"  [WARN] Planner returned unresolved `{field_name}` with Unknown; reject and retry")
                self._last_response_rejection_notice = (
                    f"Your previous response was rejected because `{field_name}` contained `Unknown`. "
                    "Do not output `Unknown`, `Unknown's ...`, or any unresolved current area. "
                    "Infer the current area from the nearby current scene first. Previous Space Waypoints only show visited history, not where you are now."
                )
                return False
            if re.search(r"(?i)(^|->\s*)(?:area|room|space)(?:'s|\s*-)", field_text):
                print(f"  [WARN] Planner returned placeholder `{field_name}` with generic area label; reject and retry")
                self._last_response_rejection_notice = (
                    f"Your previous response was rejected because `{field_name}` used a generic placeholder such as "
                    "`area`, `room`, or `space` as the space name. Infer one concrete space type from "
                    "the current views, nearby landmarks, openings, map, and trajectory. Do not output bare "
                    "`area - ...`, `room - ...`, or `space's ...`."
                )
                return False

        if str(mode).strip().lower() == 'initial' and bool(response.get('global_task_finish', False)):
            print("  [WARN] Initial planning returned global_task_finish=true at task start; reject and retry")
            self._last_response_rejection_notice = (
                "Your previous response was rejected because initial planning cannot finish the global task. "
                "Keep the finish flag false and output the first executable subtask."
            )
            return False

        return True

    def _call_planner_with_retry(
        self,
        prompt: PromptLike,
        images: List[Any],
        direction_names: Optional[List[str]],
        mode: str,
        save_dir: Optional[str],
        no_compress: Optional[set] = None,
        failure_label: str = "LLM Planning",
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        max_retries = self._resolve_planner_max_retries(mode)
        self._reset_last_call_timing_info()
        prompt_debug_text = extract_prompt_debug_text(prompt)

        for retry in range(max_retries):
            self._last_response_rejection_notice = ""
            attempt_start_time = time.perf_counter()
            response = self.call_api(
                prompt,
                images,
                save_dir=save_dir,
                no_compress_indices=no_compress,
            )
            attempt_duration_s = time.perf_counter() - attempt_start_time
            request_duration_s = float(
                getattr(self, "last_request_latency_s", 0.0) or 0.0
            )
            if request_duration_s <= 0.0:
                request_duration_s = attempt_duration_s

            normalized_response = self._normalize_response_payload(response)
            is_valid = bool(normalized_response and self.validate_response(normalized_response, mode=mode))
            direction_is_available = False
            if is_valid:
                direction_is_available = self._direction_is_available(
                    normalized_response.get('next_waypoint_direction'),
                    direction_names,
                )
                if not direction_is_available:
                    print(
                        f"  [WARN] {failure_label} chose an unavailable direction: "
                        f"{normalized_response.get('next_waypoint_direction', '')}"
                    )
                    self._last_response_rejection_notice = (
                        "Your previous response was rejected because `next_waypoint_direction` did not match "
                        "one of the provided IMAGE labels. Choose only from the current provided IMAGE labels."
                    )

            attempt_success = bool(is_valid and direction_is_available)
            failure_kind = ""
            if not attempt_success:
                failure_kind = self._classify_attempt_failure(
                    response=response,
                    normalized_response=normalized_response,
                    is_valid=is_valid,
                    direction_is_available=direction_is_available,
                    api_status=str(getattr(self, "last_call_status", "") or ""),
                )
            self.last_call_timing_info["records"].append({
                "attempt": retry + 1,
                "success": attempt_success,
                "duration_s": max(0.0, float(request_duration_s)),
                "total_call_duration_s": max(0.0, float(attempt_duration_s)),
                "failure_kind": failure_kind,
            })
            if attempt_success:
                self.last_call_timing_info["final_failure_reason"] = ""
                self._finalize_vlm_info_retry_summary(save_dir)
                return normalized_response, prompt_debug_text

            if retry < max_retries - 1:
                wait = (retry + 1) * 2
                self.last_call_timing_info["failed_retry_wait_duration_s"] += float(wait)
                print(
                    f"  [WARN] {failure_label} failed, retry {retry + 1}/{max_retries - 1} "
                    f"in {wait}s..."
                )
                time.sleep(wait)

        print(f"  [ERR] {failure_label} failed after {max_retries} attempts")
        self.last_call_timing_info["final_failure_reason"] = self._summarize_final_failure_reason()
        self._finalize_vlm_info_retry_summary(save_dir)
        return None, prompt_debug_text

    @staticmethod
    def _extract_image_index(direction_text: Optional[str]) -> Optional[int]:
        match = re.search(r'IMAGE\s*(\d+)', str(direction_text or ''), re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _direction_is_available(
        cls,
        chosen_direction: Optional[str],
        direction_names: Optional[List[str]],
    ) -> bool:
        chosen_index = cls._extract_image_index(chosen_direction)
        if chosen_index is None:
            return True

        available_indices = {
            image_idx
            for image_idx in (
                cls._extract_image_index(direction_name)
                for direction_name in list(direction_names or [])
            )
            if image_idx is not None
        }
        if not available_indices:
            return True
        return chosen_index in available_indices
    
    def generate_initial_subtask(self,
                                instruction: str,
                                observation_images: List[Any],
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
            local_map_image: 局部语义地图路径（仅调试保留，不传给thinking模型）
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
        
        prompt = build_initial_planner_prompt_bundle(
            instruction=instruction,
            action_space=self.action_space,
        )
        
        # 组合图像：12方向观察 + 全局地图
        images = observation_images.copy()
        images.append(global_map_image)
        no_compress = {len(observation_images)}

        return self._call_planner_with_retry(
            prompt=prompt,
            images=images,
            direction_names=direction_names,
            mode='initial',
            save_dir=save_dir,
            no_compress=no_compress,
            failure_label="LLM Planning",
        )
    
    def verify_and_replan(self,
                         instruction: str,
                         current_subtask: Dict,
                         observation_images: List[Any],
                         direction_names: List[str],
                         global_map_image: str,
                         local_map_image: str = None,
                         detected_landmarks: List[str] = None,
                         waypoint_summary: str = None,
                         previous_subtask_landmark_summary: str = None,
                         obstacle_distances: Dict[str, str] = None,
                         verify_replan_prompt_notice: str = None,
                         save_dir: str = None) -> Tuple[Optional[Dict], str]:
        """
        验证子任务完成并规划下一步
        
        Args:
            instruction: 完整导航指令
            current_subtask: 当前子任务字典
            observation_images: 4方向图像路径列表（当前位置重新环视获得）
            direction_names: 方向名称列表
            global_map_image: 更新后的全局语义地图路径 - 必需
            local_map_image: 更新后的局部语义地图路径（仅调试保留，不传给thinking模型）
            detected_landmarks: 已检测到的landmark类别列表 - 可选
            waypoint_summary: 路径点历史记录 - 可选
            previous_subtask_landmark_summary: 上一子任务landmark最终观测摘要 - 可选
            obstacle_distances: 预计算的障碍物距离字典 {'front': 'X.XXm', 'left_30': ..., ...}
            verify_replan_prompt_notice: 本次verify/replan的顶部附加提示 - 可选
            
        Returns:
            (response字典, prompt字符串)
        """
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None, ""
        
        # 获取当前子任务信息
        subtask_destination = get_next_waypoint(current_subtask) or 'not set'
        subtask_instruction = current_subtask.get('subtask_instruction', 'not set')
        # 使用预计算的距离（如果没有则设为Unknown）
        if not obstacle_distances:
            obstacle_distances = {
                'front': 'Unknown',
                'left_30': 'Unknown',
                'right_30': 'Unknown',
                'left_90': 'Unknown',
                'right_90': 'Unknown'
            }
        
        prompt = build_verify_planner_prompt_bundle(
            instruction=instruction,
            subtask_destination=subtask_destination,
            subtask_instruction=subtask_instruction,
            action_space=self.action_space,
            detected_landmarks=None,
            waypoint_summary=waypoint_summary,
            previous_subtask_landmark_summary=previous_subtask_landmark_summary,
            verify_replan_prompt_notice=verify_replan_prompt_notice,
            direction_names=direction_names,
        )
        
        # 组合图像：当前位置12方向 + 全局地图
        images = observation_images.copy()
        images.append(global_map_image)
        no_compress = {len(observation_images)}

        return self._call_planner_with_retry(
            prompt=prompt,
            images=images,
            direction_names=direction_names,
            mode='verify',
            save_dir=save_dir,
            no_compress=no_compress,
            failure_label="LLM Verify",
        )
