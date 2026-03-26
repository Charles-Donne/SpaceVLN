"""
VLM Navigation Controller
=========================
基于VLM的自动导航控制器

继承BaseNavigationController的核心功能：
- 语义建图（GroundedSAM + Semantic Mapping）
- 可视化（MapVisualizer）
- 12步×30°环视建图

新增VLM功能：
- LLM高层规划（生成子任务）
- VLM低层动作执行（基于RGB+地图决策）
- 4方向观察收集（前/右/后/左）
- RGB+俯视图拼接可视化（使用环境提供的top_down_map_vlnce）
- 结果保存供后续测评
"""
import os
import re
import cv2
import json
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional, Sequence
from datetime import datetime

from habitat import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from vlnce_baselines.utils.spatial_formatter import (
    build_action_landmark_map_info,
    build_waypoint_summary,
)
from vlnce_baselines.controllers.base_navigation_controller import BaseNavigationController
from vlnce_baselines.mapping.space_types import strip_space_type_variant_suffixes
from vlnce_baselines.vlm import (
    LLMPlanner, ActionExecutor, SaveManager, NavigationVisualizer
)
from vlnce_baselines.vlm.support.thinking_view_renderer import ThinkingViewRenderer
from vlnce_baselines.config.core.constants import landmark_edge_depth_keywords
from vlnce_baselines.config.core.params.actions import (
    ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M as CFG_AUTOCOMPLETE_OPEN_DISTANCE_M,
    ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M as CFG_AUTOCOMPLETE_SOLID_DISTANCE_M,
    ACTION_SUBTASK_AUTOCOMPLETE_TOPK as CFG_AUTOCOMPLETE_TOPK,
)
from vlnce_baselines.vlm.support.navigation_config import ACTION_MAPPING
from habitat_extensions.pose_utils import get_sim_location


class VLMNavigationController(BaseNavigationController):
    """
    VLM导航控制器
    
    继承自BaseNavigationController，添加VLM规划和执行功能
    
    工作流程：
    1. 初始环视建图（12步×30°）→ 收集4方向图像
    2. LLM规划 → 生成初始子任务
    3. VLM执行 → 循环执行动作直到子任务完成
    4. 验证环视建图（12步×30°）→ 更新地图和4方向图像
    5. 验证重规划 → 检查完成状态，生成下一子任务
    6. 重复3-5直到导航完成
    
    注意：每次验证重规划前都会执行360°环视，以更新语义地图和当前位置的4方向观察
    """

    ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M = CFG_AUTOCOMPLETE_OPEN_DISTANCE_M
    ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M = CFG_AUTOCOMPLETE_SOLID_DISTANCE_M
    ACTION_SUBTASK_AUTOCOMPLETE_TOPK = CFG_AUTOCOMPLETE_TOPK
    AUTO_RETREAT_STOP_EARLY_IF_REVERSE_BLOCKED = False
    THINKING_LOOKAROUND_STEPS = 12
    FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK = 3
    FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M = 1.0
    ACTION_STAGNATION_REPLAN_STREAK = 3
    ACTION_STAGNATION_MAX_MOVEMENT_M = 0.25
    
    def __init__(self, config: Config,
                 config_path: str = "vlnce_baselines/config/api/vlm_api_config.yaml"):
        """
        初始化VLM导航控制器
        
        Args:
            config: Habitat配置
            config_path: 统一API配置文件路径（同时设置LLM和VLM）
        """
        # 调用父类初始化（初始化环境、检测、建图、可视化）
        super().__init__(config)
        
        # 初始化VLM模块
# print("\n[Init] 初始化VLM模块...")
        
        # 获取动作参数
        self.turn_angle = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE  # 30°
        self.move_distance = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE  # 0.25m
        
        # 动作空间描述
        self.action_space = f"MOVE_FORWARD ({self.move_distance}m), TURN_LEFT ({self.turn_angle}°), TURN_RIGHT ({self.turn_angle}°), STOP"
        self.enable_auto_retreat = bool(getattr(config.MAP, "ENABLE_AUTO_RETREAT", False))
        self.auto_retreat_stop_early_if_reverse_blocked = bool(
            getattr(
                config.MAP,
                "AUTO_RETREAT_STOP_EARLY_IF_REVERSE_BLOCKED",
                self.AUTO_RETREAT_STOP_EARLY_IF_REVERSE_BLOCKED,
            )
        )
        self.final_destination_match_autostop_streak = max(
            1,
            int(
                getattr(
                    config.MAP,
                    "FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK",
                    self.FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK,
                ) or self.FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK
            ),
        )
        self.final_destination_match_autostop_radius_m = max(
            0.0,
            float(
                getattr(
                    config.MAP,
                    "FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M",
                    self.FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M,
                ) or self.FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M
            ),
        )
        self.action_stagnation_replan_streak = max(
            1,
            int(
                getattr(
                    config.MAP,
                    "ACTION_STAGNATION_REPLAN_STREAK",
                    self.ACTION_STAGNATION_REPLAN_STREAK,
                ) or self.ACTION_STAGNATION_REPLAN_STREAK
            ),
        )
        self.action_stagnation_max_movement_m = max(
            0.0,
            float(
                getattr(
                    config.MAP,
                    "ACTION_STAGNATION_MAX_MOVEMENT_M",
                    self.ACTION_STAGNATION_MAX_MOVEMENT_M,
                ) or self.ACTION_STAGNATION_MAX_MOVEMENT_M
            ),
        )
        
        # 初始化LLM规划器
        try:
            self.planner = LLMPlanner(config_path, self.action_space)
        except Exception as e:
            print(f"[WARN] LLM Planner init failed: {e}")
            self.planner = None
        
        # 初始化VLM执行器
        try:
            self.action_executor = ActionExecutor(config_path, self.turn_angle, self.move_distance)
        except Exception as e:
            print(f"[WARN] Action Executor init failed: {e}")
            self.action_executor = None
        
        # VLM状态
        self.current_subtask = None
        self.subtask_count = 0
        self.subtask_attempt = 0  # 当前子任务的尝试次数（a, b, c...）
        self.progress_summary = ""
        self.previous_action_reason = ""  # 上一步的action_analysis
        self.subtask_history = []
        self.latest_thinking_cycle_info = {}
        self.thinking_view_renderer = ThinkingViewRenderer()
        self.final_goal_destination_match_streak = 0
        self.final_goal_destination_match_anchor_xy = None
        self.action_stagnation_streak = 0
        self.verify_replan_prompt_notice = ""
        
        # 初始化管理器
        self.save_manager = None  # 在reset_episode时初始化
        # waypoint_manager已废弃，直接使用mapper.add_waypoint()
        
        self.pose_before_action = None  # 记录动作前的pose (x, y, orientation)
        
        # 当前子任务跟踪的landmark类别（每个子任务重置）
        self.tracked_landmark_classes = set()
        
        # NavigationVisualizer（用于RGB+俯视图拼接和GIF生成）
        self.nav_visualizer = None

        # print("[Init] VLM模块初始化完成\n")

    @staticmethod
    def _parse_distance_text_m(distance_text: Optional[str]) -> Optional[float]:
        """Extract numeric distance in meters from strings like '0.34m WARNING'."""
        if distance_text is None:
            return None
        match = re.search(r'([0-9]+(?:\.[0-9]+)?)m', str(distance_text))
        if not match:
            return None
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            return None

    def _is_obstacle_distance_blocked(self, distance_text: Optional[str], threshold_m: float = 0.5) -> bool:
        distance_m = self._parse_distance_text_m(distance_text)
        return distance_m is not None and distance_m < float(threshold_m)

    def _all_action_directions_blocked(self, threshold_m: float = 0.5) -> bool:
        distances = getattr(self, 'latest_obstacle_distances', {}) or {}
        keys = ('left_30', 'front', 'right_30')
        return all(self._is_obstacle_distance_blocked(distances.get(key), threshold_m=threshold_m) for key in keys)

    def _append_progress_note(self, note: str) -> None:
        note = (note or "").strip()
        if not note:
            return
        if not self.progress_summary or self.progress_summary == "(Just started - no actions yet)":
            self.progress_summary = note
        else:
            self.progress_summary = f"{self.progress_summary}, {note}"

    def _get_low_level_stagnation_threshold_m(self) -> float:
        """Use a strict no-move threshold for low-level forward steps."""
        configured_threshold_m = max(0.0, float(self.action_stagnation_max_movement_m or 0.0))
        step_scaled_threshold_m = max(0.0, float(self.move_distance or 0.0) * 0.2)
        if step_scaled_threshold_m <= 0.0:
            return configured_threshold_m
        if configured_threshold_m <= 0.0:
            return step_scaled_threshold_m
        return min(configured_threshold_m, step_scaled_threshold_m)

    @staticmethod
    def _build_stagnation_verify_notice() -> str:
        return (
            "You just tried to go straight three low-level MOVE_FORWARD steps without actually moving, "
            "so an obstacle is likely blocking the front route. In the next plan, do not choose the "
            "front-facing sector (Left 30deg / Front / Right 30deg); choose a clearer obstacle-free "
            "direction that still advances toward the destination."
        )

    def _update_action_stagnation_streak(
        self,
        action_name: Optional[str],
        actual_meters: float,
    ) -> bool:
        action_name_upper = str(action_name or "").upper()
        if action_name_upper != "MOVE_FORWARD":
            if self.action_stagnation_streak > 0:
                print(
                    "[ActionStagnation] "
                    f"{action_name_upper or 'NON_FORWARD'} broke the forward-stall streak; "
                    "reset stagnation streak"
                )
            self.action_stagnation_streak = 0
            return False

        stagnation_threshold_m = self._get_low_level_stagnation_threshold_m()
        if float(actual_meters) <= stagnation_threshold_m:
            self.action_stagnation_streak += 1
            print(
                "[ActionStagnation] "
                f"low-level MOVE_FORWARD moved only {float(actual_meters):.2f}m "
                f"(no-move threshold {stagnation_threshold_m:.2f}m) | "
                f"streak {self.action_stagnation_streak}/{self.action_stagnation_replan_streak}"
            )
        else:
            if self.action_stagnation_streak > 0:
                print(
                    "[ActionStagnation] "
                    f"Low-level MOVE_FORWARD recovered with {float(actual_meters):.2f}m movement; "
                    "reset stagnation streak"
                )
            self.action_stagnation_streak = 0

        return self.action_stagnation_streak >= self.action_stagnation_replan_streak

    @staticmethod
    def _get_next_waypoint_field(payload: Optional[Dict[str, Any]]) -> str:
        if not payload:
            return ""
        return str(
            payload.get('next_waypoint', payload.get('next_waypoint_destination', '')) or ''
        ).strip()

    @staticmethod
    def _get_subtask_landmark_field(payload: Optional[Dict[str, Any]]) -> str:
        if not payload:
            return ""
        return str(
            payload.get('subtask_landmark', payload.get('next_waypoint_landmark', '')) or ''
        ).strip()

    def _get_episode_max_steps(self) -> int:
        return int(getattr(self.config.TASK_CONFIG.ENVIRONMENT, 'MAX_EPISODE_STEPS', 0) or 0)

    def _get_remaining_episode_steps(self) -> int:
        return max(0, self._get_episode_max_steps() - int(getattr(self, 'current_step', 0) or 0))

    def _has_budget_for_thinking_cycle(self) -> bool:
        # Reserve one extra step so the controller can still call STOP after thinking if needed.
        return self._get_remaining_episode_steps() > int(self.THINKING_LOOKAROUND_STEPS)

    def _parse_subtask_destination(self) -> Tuple[Optional[str], Optional[str]]:
        destination = ""
        if getattr(self, 'current_subtask', None):
            destination = self._get_next_waypoint_field(self.current_subtask)
        if not destination or "'s " not in destination:
            return None, None

        room_text, object_text = destination.split("'s ", 1)
        room_norm = strip_space_type_variant_suffixes(room_text.strip().lower())
        object_norm = self._normalize_landmark_candidate(object_text)
        return room_norm or None, object_norm or None

    def _get_current_subtask_autocomplete_candidates(self) -> List[str]:
        """Only allow proximity auto-stop when subtask_landmark is the destination landmark itself."""
        _dest_room, dest_object = self._parse_subtask_destination()
        dest_object_norm = self._normalize_landmark_candidate(dest_object)
        subtask_landmark_norm = self._normalize_landmark_candidate(
            self._get_subtask_landmark_field(getattr(self, 'current_subtask', None))
        )
        if not dest_object_norm or not subtask_landmark_norm:
            return []

        destination_aligned = (
            subtask_landmark_norm == dest_object_norm or
            subtask_landmark_norm in dest_object_norm or
            dest_object_norm in subtask_landmark_norm
        )
        if not destination_aligned:
            return []

        candidates: List[str] = []
        for raw_candidate in (subtask_landmark_norm, dest_object_norm):
            normalized = self._normalize_landmark_candidate(raw_candidate)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        return candidates

    def _should_autocomplete_subtask_during_action_step(
        self,
        step_landmark_entries: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        candidate_names = self._get_current_subtask_autocomplete_candidates()
        if not candidate_names:
            return None

        matches: List[Dict[str, Any]] = []
        for entry in list(step_landmark_entries or [])[: int(self.ACTION_SUBTASK_AUTOCOMPLETE_TOPK)]:
            if self._landmark_matches_current_subtask_destination(
                entry.get('name'),
                candidate_names=candidate_names,
            ):
                try:
                    distance_m = float(entry.get('distance_m'))
                except (TypeError, ValueError):
                    continue
                is_opening_like = self._is_opening_like_landmark_entry(entry)
                stop_distance_m = self._autocomplete_stop_distance_m(entry, is_opening_like=is_opening_like)
                if distance_m > stop_distance_m:
                    continue
                matches.append({
                    "name": str(entry.get('name') or candidate_names[0]),
                    "distance_m": distance_m,
                    "confidence": float(entry.get('confidence', 0.0) or 0.0),
                    "angle_deg": entry.get('angle_deg'),
                    "is_opening_like": bool(is_opening_like),
                    "stop_distance_m": float(stop_distance_m),
                })

        if not matches:
            return None

        matches.sort(
            key=lambda item: (
                float(item.get("distance_m", 1e9)),
                -float(item.get("confidence", 0.0)),
                str(item.get("name", "")),
            )
        )
        return matches[0]

    def _is_opening_like_landmark_entry(self, entry: Optional[Dict[str, Any]]) -> bool:
        if not entry:
            return False
        if bool(entry.get("is_opening_like", False)):
            return True
        name_text = str(entry.get("name", "") or "").strip().lower()
        return bool(name_text and any(keyword in name_text for keyword in landmark_edge_depth_keywords))

    def _autocomplete_stop_distance_m(
        self,
        entry: Optional[Dict[str, Any]],
        is_opening_like: Optional[bool] = None,
    ) -> float:
        if entry:
            raw_threshold = entry.get("stop_distance_m")
            try:
                threshold_m = float(raw_threshold)
                if threshold_m > 0.0:
                    return threshold_m
            except (TypeError, ValueError):
                pass

        opening_like = self._is_opening_like_landmark_entry(entry) if is_opening_like is None else bool(is_opening_like)
        if opening_like:
            return float(self.ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M)
        return float(self.ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M)

    def _get_current_subtask_landmark_candidates(self) -> List[str]:
        _dest_room, dest_object = self._parse_subtask_destination()
        candidates: List[str] = []
        for raw_candidate in (
            dest_object,
            getattr(self, 'target_landmark', None),
        ):
            normalized = self._normalize_landmark_candidate(raw_candidate)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        return candidates

    def _landmark_matches_current_subtask_destination(
        self,
        name: Optional[str],
        candidate_names: Optional[Sequence[str]] = None,
    ) -> bool:
        normalized = self._normalize_landmark_candidate(name)
        if not normalized:
            return False

        candidates = list(candidate_names or self._get_current_subtask_landmark_candidates())
        if not candidates:
            return False

        return any(
            normalized == candidate or normalized in candidate or candidate in normalized
            for candidate in candidates
        )

    def _get_current_action_step_landmark_entries(self) -> List[Dict[str, Any]]:
        if hasattr(self, 'current_step_action_landmark_topk_entries'):
            entries = self.current_step_action_landmark_topk_entries.get(self.current_step, []) or []
            if entries:
                return entries
        if not hasattr(self, 'current_step_landmark_entries'):
            return []
        return self.current_step_landmark_entries.get(self.current_step, []) or []

    def _trigger_verify_replan(self, reason_tag: str, total_steps: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        new_subtask, _ = self.verify_and_replan()
        if new_subtask and new_subtask.get('global_task_finish', False):
            print(f"[DONE] Task complete ({reason_tag}) | steps={total_steps}")
            return True, new_subtask
        return False, new_subtask

    def _save_waypoint_area_memory_snapshot(self) -> None:
        """Persist waypoint/space-area state for debugging after each planning update."""
        if self.save_manager is None or self.mapper is None:
            return

        map_state = self.mapper.get_map_state()
        waypoint_positions = [
            [int(pos[0]), int(pos[1])]
            for pos in map_state.get('waypoint_positions', []) or []
        ]
        waypoint_ids = [int(wp_id) for wp_id in map_state.get('waypoint_ids', []) or []]
        waypoint_descriptions = [str(desc) for desc in getattr(self.mapper, 'waypoint_descriptions', []) or []]
        waypoint_area_labels = [
            str(label or "Unknown")
            for label in map_state.get('waypoint_area_labels', []) or []
        ]
        space_area_records = []
        for record in map_state.get('space_area_records', []) or []:
            center_world_px = record.get("center_world_px", (0, 0))
            space_area_records.append({
                "id": int(record.get("id", 0) or 0),
                "label": str(record.get("label", "")),
                "display_label": str(record.get("display_label", record.get("label", ""))),
                "space_type": str(record.get("space_type", "")),
                "variant": int(record.get("variant", 0) or 0),
                "center_world_px": [int(center_world_px[0]), int(center_world_px[1])],
                "connected_area_labels": [str(item) for item in record.get("connected_area_labels", []) or []],
            })

        waypoint_memory = {
            "current_space_area_label": str(map_state.get('current_space_area_label', 'Unknown') or 'Unknown'),
            "current_space_area_type": str(map_state.get('current_space_area_type', 'Unknown') or 'Unknown'),
            "waypoint_positions": waypoint_positions,
            "waypoint_ids": waypoint_ids,
            "waypoint_descriptions": waypoint_descriptions,
            "waypoint_area_labels": waypoint_area_labels,
            "space_area_records": space_area_records,
            "waypoint_summary": self._get_waypoint_summary(include_area_chain=True),
        }
        self.save_manager.save_waypoint_memory(
            waypoint_memory=waypoint_memory,
            instruction=self.current_instruction,
            current_step=self.current_step,
        )

    def _execute_auto_retreat(self, retreat_distance_m: float = 1.0) -> Tuple[float, bool]:
        """Turn around, move 1m away, then hand off directly to thinking/lookaround."""
        turn_steps = max(1, round(180.0 / float(self.turn_angle)))
        move_steps = max(1, round(float(retreat_distance_m) / float(self.move_distance)))
        retreated_m = 0.0
        episode_done = False
        reverse_path_blocked = False
        reverse_block_warning_printed = False
        stop_early_if_reverse_blocked = bool(self.auto_retreat_stop_early_if_reverse_blocked)

        print(
            f"\n[AutoRetreat] FRONT/LEFT30/RIGHT30 all blocked (<0.5m). "
            f"Turn around and move {retreat_distance_m:.2f}m, then start thinking/lookaround."
        )

        for _ in range(turn_steps):
            result = self.step_with_vlm(
                HabitatSimActions.TURN_RIGHT,
                action_name="AUTO_RETREAT_TURN_RIGHT",
                save_vis=True,
                enable_landmark_detection=True,
            )
            print(f"  [Step {self.current_step}] AUTO_RETREAT_TURN_RIGHT")
            if result.get('done', False):
                episode_done = True
                break

        for _ in range(move_steps):
            if episode_done:
                break
            if self._is_obstacle_distance_blocked(
                getattr(self, 'latest_obstacle_distances', {}).get('front'),
                threshold_m=0.5,
            ):
                reverse_path_blocked = True
                if stop_early_if_reverse_blocked:
                    print("  [AutoRetreat] Reverse path is also blocked after turning around; stop retreat early.")
                    break
                if not reverse_block_warning_printed:
                    print(
                        "  [AutoRetreat] Reverse path is also blocked after turning around, "
                        "but early-stop is disabled; keep retreat attempts."
                    )
                    reverse_block_warning_printed = True

            pose_before_step = self._get_agent_pose()
            result = self.step_with_vlm(
                HabitatSimActions.MOVE_FORWARD,
                action_name="AUTO_RETREAT_FORWARD",
                save_vis=True,
                enable_landmark_detection=True,
            )
            pose_after_step = self._get_agent_pose()

            actual_m = float(np.hypot(
                pose_after_step[0] - pose_before_step[0],
                pose_after_step[1] - pose_before_step[1],
            ))
            retreated_m += actual_m
            print(
                f"  [Step {self.current_step}] AUTO_RETREAT_FORWARD "
                f"({retreated_m:.2f}/{retreat_distance_m:.2f}m, actual {actual_m:.2f}m)"
            )
            if result.get('done', False):
                episode_done = True
                break

        if retreated_m > 0:
            self._append_progress_note(
                f"Had encountered obstacles on front/left30/right30 (<0.5m), then turned around, moved {retreated_m:.2f}m, and triggered replan"
            )
            self.previous_action_reason = (
                f"Front, Left 30deg, and Right 30deg were all blocked under 0.5m, "
                f"so the system turned around, moved {retreated_m:.2f}m, and started rethinking from the new heading."
            )
        elif reverse_path_blocked and stop_early_if_reverse_blocked:
            self._append_progress_note(
                "Had encountered obstacles on front/left30/right30 (<0.5m), then turned around but the reverse path was also blocked, and triggered replan"
            )
            self.previous_action_reason = (
                "Front, Left 30deg, and Right 30deg were all blocked under 0.5m, "
                "so the system turned around, found the reverse path blocked too, and triggered rethinking."
            )
        elif reverse_path_blocked:
            self._append_progress_note(
                "Had encountered obstacles on front/left30/right30 (<0.5m), then turned around, kept retreat attempts even though the reverse path still looked blocked, and triggered replan"
            )
            self.previous_action_reason = (
                "Front, Left 30deg, and Right 30deg were all blocked under 0.5m, "
                "so the system turned around, kept retreat attempts with reverse-block early-stop disabled, and then triggered rethinking."
            )
        else:
            self._append_progress_note(
                "Had encountered obstacles on front/left30/right30 (<0.5m), then turned around and triggered replan"
            )
            self.previous_action_reason = (
                "Front, Left 30deg, and Right 30deg were all blocked under 0.5m, "
                "so the system turned around and triggered rethinking."
            )

        self.pose_before_action = self._get_agent_pose()
        return retreated_m, episode_done

    def reset_episode(self, episode_id: int = None):
        """重置Episode，包括VLM状态"""
        # 清理之前episode的输出目录
        if episode_id is not None:
            import shutil
            old_episode_dir = os.path.join(self.config.RESULTS_DIR, f'episode_{episode_id}')
            if os.path.exists(old_episode_dir):
                print(f"[Reset] 清理旧数据: {old_episode_dir}")
                shutil.rmtree(old_episode_dir)
        
        # 调用父类重置
        super().reset_episode(episode_id)
        
        # 初始化SaveManager（使用RESULTS_DIR作为输出根目录）
        self.save_manager = SaveManager(self.config.RESULTS_DIR, self.current_episode_id)
        
        # 重置VLM状态
        self.current_subtask = None
        self.subtask_count = 0
        self.subtask_attempt = 0  # 重置尝试计数
        self.progress_summary = ""
        self.previous_action_reason = ""  # 重置上一步action reason
        self.subtask_history = []
        self.latest_thinking_cycle_info = {}
        self.tracked_landmark_classes = set()
        self.final_goal_destination_match_streak = 0
        self.final_goal_destination_match_anchor_xy = None
        self.action_stagnation_streak = 0
        self.verify_replan_prompt_notice = ""
        self.pose_before_action = None  # 重置pose追踪
        self.last_planned_degrees = 0  # 记录计划转向角度
        self.last_planned_meters = 0   # 记录计划移动距离
        self.last_action_name = ""      # 记录上次动作名称
        
        # waypoint已集成到mapper中，mapper.reset()会自动清空
        
        # print(f"[Reset] Episode {self.current_episode_id} 重置完成")
        
        # 初始化NavigationVisualizer（用于RGB+俯视图拼接和GIF生成）
        visualization_dir = os.path.join(self.episode_dir, 'visualization')
        self.nav_visualizer = NavigationVisualizer(visualization_dir)
        self.nav_visualizer.setup_maps_dir(self.episode_dir)
        
    @property
    def episode_dir(self) -> str:
        """获取当前episode的输出目录（动态属性，自动根据current_episode_id生成）"""
        return os.path.join(self.config.RESULTS_DIR, f'episode_{self.current_episode_id}')

    @classmethod
    def _normalize_landmark_candidate(cls, text: Optional[str]) -> Optional[str]:
        """清洗 landmark 文本，但不再限制短语长度。"""
        if not text:
            return None

        cleaned = str(text).strip().lower()
        for token in ["|", "/", "\\", "->", "=>", ":", ";", ",", ".", "!", "?", "(", ")", "[", "]", "{", "}", "\"", "'"]:
            cleaned = cleaned.replace(token, " ")
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return None

        return cleaned

    @classmethod
    def _normalize_waypoint_endpoint_label(cls, text: Optional[str]) -> Optional[str]:
        """Normalize waypoint-chain endpoints / destinations for robust matching."""
        if not text:
            return None

        cleaned = strip_space_type_variant_suffixes(str(text))
        cleaned = cleaned.replace("’", "'").replace("`", "'")
        cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
        cleaned = cleaned.replace("→", " ").replace("->", " ").replace("|", " ")
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return None
        return cls._normalize_landmark_candidate(cleaned)

    @classmethod
    def _extract_last_waypoint_chain_node(cls, waypoint_chain: Optional[str]) -> Optional[str]:
        """Extract the last semantic node from a planner waypoint chain string."""
        if not waypoint_chain:
            return None

        chain_text = strip_space_type_variant_suffixes(str(waypoint_chain)).replace("’", "'").strip()
        if not chain_text:
            return None

        raw_nodes = re.split(r"\s*(?:→|->)\s*", chain_text)
        raw_nodes = [node.strip() for node in raw_nodes if node and node.strip()]
        if not raw_nodes:
            return None

        last_node = raw_nodes[-1]
        last_node = re.sub(r"\([^)]*\)", "", last_node).strip()
        return " ".join(last_node.split()) or None

    @staticmethod
    def _extract_pose_xy(pose: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
        if pose is None or len(pose) < 2:
            return None
        try:
            return float(pose[0]), float(pose[1])
        except (TypeError, ValueError):
            return None

    def _reset_final_goal_destination_match_state(self) -> None:
        self.final_goal_destination_match_streak = 0
        self.final_goal_destination_match_anchor_xy = None

    def _update_final_goal_destination_match_streak(
        self,
        response: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[str], int, Optional[float], bool, bool]:
        """Track goal-tail matches plus whether they stay inside the same finish region."""
        waypoint_chain = response.get('waypoint_chain') or response.get('waypoint_sequence') or ''
        next_destination = self._get_next_waypoint_field(response)
        last_chain_node = self._extract_last_waypoint_chain_node(waypoint_chain)

        normalized_last = self._normalize_waypoint_endpoint_label(last_chain_node)
        normalized_destination = self._normalize_waypoint_endpoint_label(next_destination)
        matched = bool(
            normalized_last and
            normalized_destination and
            normalized_last == normalized_destination
        )

        anchor_distance_m: Optional[float] = None
        stayed_inside_anchor_region = False
        restarted_by_anchor_drift = False

        if matched:
            current_xy = self._extract_pose_xy(self._get_agent_pose())
            if current_xy is None:
                self.final_goal_destination_match_streak = 1
                self.final_goal_destination_match_anchor_xy = None
            elif (
                self.final_goal_destination_match_streak <= 0 or
                self.final_goal_destination_match_anchor_xy is None
            ):
                self.final_goal_destination_match_streak = 1
                self.final_goal_destination_match_anchor_xy = current_xy
                anchor_distance_m = 0.0
                stayed_inside_anchor_region = True
            else:
                anchor_x, anchor_y = self.final_goal_destination_match_anchor_xy
                anchor_distance_m = float(np.hypot(current_xy[0] - anchor_x, current_xy[1] - anchor_y))
                if anchor_distance_m <= self.final_destination_match_autostop_radius_m:
                    self.final_goal_destination_match_streak += 1
                    stayed_inside_anchor_region = True
                else:
                    restarted_by_anchor_drift = True
                    self.final_goal_destination_match_streak = 1
                    self.final_goal_destination_match_anchor_xy = current_xy
        else:
            self._reset_final_goal_destination_match_state()

        response['final_waypoint_chain_goal'] = last_chain_node or ""
        response['final_waypoint_destination_match_streak'] = self.final_goal_destination_match_streak
        response['final_waypoint_destination_anchor_distance_m'] = anchor_distance_m
        response['final_waypoint_destination_anchor_radius_m'] = self.final_destination_match_autostop_radius_m
        response['final_waypoint_destination_anchor_region_stable'] = stayed_inside_anchor_region
        return (
            matched,
            last_chain_node,
            str(next_destination).strip() or None,
            self.final_goal_destination_match_streak,
            anchor_distance_m,
            stayed_inside_anchor_region,
            restarted_by_anchor_drift,
        )

    @classmethod
    def _iter_landmark_source_candidates(cls, source: Optional[str]) -> List[str]:
        """从结构化字段里拆出更像物体短语的候选片段。"""
        if not source:
            return []

        text = str(source).strip()
        if not text:
            return []

        candidates = [text]

        if "Detection:" in text:
            detection_part = text.split("Detection:", 1)[1].split("|", 1)[0].strip()
            if detection_part:
                candidates.append(detection_part)

        if "'s" in text:
            tail = text.split("'s", 1)[1].strip()
            if tail:
                candidates.append(tail)

        for sep in ["|", ",", ";"]:
            if sep in text:
                candidates.extend(
                    part.strip() for part in text.split(sep) if part.strip()
                )

        unique = []
        seen = set()
        for item in candidates:
            if item not in seen:
                unique.append(item)
                seen.add(item)
        return unique

    @classmethod
    def _resolve_landmark_name(
        cls,
        next_waypoint_landmark: Optional[str],
        fallback_sources: Optional[List[Optional[str]]] = None,
    ) -> Optional[str]:
        """优先保留LLM原始输出；为空时再从结构化字段回退。"""
        primary_candidate = cls._normalize_landmark_candidate(next_waypoint_landmark)
        if primary_candidate:
            return primary_candidate

        for source in fallback_sources or []:
            for piece in cls._iter_landmark_source_candidates(source):
                candidate = cls._normalize_landmark_candidate(piece)
                if candidate:
                    print(f"  [INFO] Fallback landmark: {candidate}")
                    return candidate

        return None

    def _set_current_landmark_tracking(
        self,
        next_waypoint_landmark: Optional[str],
        fallback_sources: Optional[List[Optional[str]]] = None,
    ) -> None:
        """每个子任务只保留当前目标landmark，不跨子任务累积。"""
        self.tracked_landmark_classes.clear()
        clean_landmark = self._resolve_landmark_name(next_waypoint_landmark, fallback_sources)

        if clean_landmark:
            self.tracked_landmark_classes.add(clean_landmark)
            self.target_landmark = clean_landmark
        else:
            self.target_landmark = None

        self.landmark_classes = sorted(list(self.tracked_landmark_classes))
        self.classes = list(self.landmark_classes)

    def _reset_custom_landmark_state(self) -> None:
        """在新子任务开始前清空旧自定义 landmark 的类别、记录和地图通道。"""
        old_landmarks = list(getattr(self, 'landmark_classes', []) or [])

        if hasattr(self, 'mapper') and self.mapper is not None:
            self.mapper.clear_custom_landmarks()

        self.landmark_classes = []
        self.tracked_landmark_classes.clear()
        self.target_landmark = None
        self.classes = []

        if hasattr(self, 'current_step_landmarks'):
            self.current_step_landmarks.clear()
        if hasattr(self, 'current_step_landmark_entries'):
            self.current_step_landmark_entries.clear()
        if hasattr(self, 'current_step_action_landmark_topk_entries'):
            self.current_step_action_landmark_topk_entries.clear()
        if hasattr(self, 'latest_landmark_instances_world'):
            self.latest_landmark_instances_world = []
        self._clear_landmark_detection_cache()

        detected = getattr(self.category_config, '_detected_classes', None)
        if detected is not None and hasattr(detected, '_dict'):
            for lm_name in old_landmarks:
                detected._dict.pop(lm_name, None)

    @staticmethod
    def _sanitize_subtask_instruction_text(
        text: Optional[str],
        destination: Optional[str] = None,
        next_waypoint_direction: Optional[str] = None,
        keep_view_prefix: bool = True,
    ) -> str:
        """Normalize planner subtask text to one sentence, optionally keeping the view prefix."""
        text = (text or "").strip()
        destination = (destination or "").strip()
        next_waypoint_direction = (next_waypoint_direction or "").strip()

        cleaned = text.replace("\n", " ").strip() if text else ""
        cleaned = re.sub(r"\s+", " ", cleaned)

        # 去掉显式的自动转向/视角前缀，只保留动作主体。
        prefix_patterns = [
            r"^\s*image\s*\d+\s*(?:\([^)]*\))?\s*[:,-]?\s*",
            r"^\s*from\s+image\s*\d+\s*(?:\([^)]*\))?\s*(?:view)?\s*[:,-]?\s*",
            r"^\s*from\s+[-+]?\d{1,3}\s*(?:deg(?:ree)?s?|°)\s+view\s*[:,-]?\s*",
            r"^\s*from\s+(?:front|left|right|back)\s*[-+]?\d{0,3}\s*(?:deg(?:ree)?s?|°)?\s+view\s*[:,-]?\s*",
            r"^\s*from\s+image\s*\d+\s*view\s*,?\s*start\s*,?\s*",
            r"^\s*from\s+[^,.;:]+view\s*,?\s*start\s*,?\s*",
            r"^\s*(?:after|once)\s+(?:auto-)?rotat(?:e|ing)[^,.;:]*[,.;:]\s*",
            r"^\s*(?:after|once)\s+(?:turn(?:ing)?|rotat(?:e|ing)|facing)[^,.;:]*[,.;:]\s*",
            r"^\s*(?:turn|rotate|face|look)\b[^,.;:]*[,.;:]\s*",
            r"^\s*(?:from|via|toward|towards|to)\s+(?:image\s*\d+|the\s+)?(?:left|right|front|back)\b[^,.;:]*[,.;:]\s*",
            r"^\s*(?:on|from)\s+the\s+(?:left|right|front|back)\b[^,.;:]*[,.;:]\s*",
        ]
        changed = True
        while changed:
            changed = False
            for pattern in prefix_patterns:
                updated = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
                if updated != cleaned:
                    cleaned = updated.strip()
                    changed = True

        # 若前面是转向说明，截断到真正的动作动词
        action_match = re.search(
            r"\b(move|go|walk|enter|pass|follow|cross|approach|continue|stop|head|climb|ascend|descend)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if action_match and action_match.start() > 0:
            prefix = cleaned[:action_match.start()]
            if re.search(r"turn|rotate|face|image|left|right|front|back", prefix, flags=re.IGNORECASE):
                cleaned = cleaned[action_match.start():].strip()

        # 去掉动作开头紧跟的方向模板，保留动作本身
        cleaned = re.sub(
            r"^\s*(move|go|walk|head|continue)\s+(?:to\s+)?(?:the\s+)?(?:left|right|forward|back(?:ward)?)\b\s*",
            r"\1 ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^\s*(move|go|walk|enter|head|continue)\s+from\s+(?:the\s+)?(?:left|right|front|back)\b\s*",
            r"\1 ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^\s*start\s*[,;:-]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*start\s+by\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*(?:then|and then)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")

        # 保留为一句话；多句时只取第一句，避免 action 侧得到冗长指令。
        if cleaned:
            parts = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
            cleaned = parts[0].strip()

        if not cleaned:
            cleaned = f"Move toward {destination}" if destination else "Move toward the target"

        dest_aliases = [destination] if destination else []
        if destination and "'s" in destination:
            room_part, obj_part = destination.split("'s", 1)
            if room_part.strip():
                dest_aliases.append(room_part.strip())
            if obj_part.strip():
                dest_aliases.append(obj_part.strip())

        has_destination_ref = any(
            alias and re.search(re.escape(alias), cleaned, flags=re.IGNORECASE)
            for alias in dest_aliases
        )

        if destination and not has_destination_ref:
            if re.match(r"^\s*stop\b", cleaned, flags=re.IGNORECASE):
                cleaned = f"Stop at {destination}"
            elif re.search(
                r"\btoward\b|\bthrough\b|\binto\b|\balong\b|\bpast\b|\baround\b|"
                r"\bupstairs\b|\bdownstairs\b|\bascend\b|\bdescend\b|\bclimb\b|"
                r"\bup stairs\b|\bdown stairs\b",
                cleaned,
                flags=re.IGNORECASE,
            ):
                cleaned = f"{cleaned.rstrip('.')} toward {destination}"
            else:
                cleaned = f"{cleaned.rstrip('.')} toward {destination}"

        cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned
        cleaned = cleaned.rstrip(" ,;:-")
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."

        if not keep_view_prefix:
            return cleaned

        if next_waypoint_direction:
            body = cleaned
            if body and body[0].isalpha():
                body = body[0].lower() + body[1:]
            return f"From {next_waypoint_direction} view, start, {body}"
        return cleaned

    def _sanitize_current_waypoint_text(self, waypoint_text: Optional[str]) -> Optional[str]:
        """Keep current_waypoint in compact `Space Type - nearby objects` form."""
        if waypoint_text is None:
            return None

        cleaned = strip_space_type_variant_suffixes(str(waypoint_text)).strip()
        if not cleaned:
            return cleaned

        parts = [part.strip() for part in cleaned.split("|") if part.strip()]
        if not parts:
            return cleaned

        base = parts[0]
        if " - " in base:
            return base.strip()

        nearby_part = ""
        for part in parts[1:]:
            if re.match(r"(?i)^connected\b", part):
                continue
            nearby_match = re.match(r"(?i)^nearby(?:\s*\([^)]*\))?\s*:\s*(.+)$", part)
            if nearby_match:
                nearby_part = nearby_match.group(1).strip(" ,;:-")
                break

        if nearby_part:
            return f"{base} - {nearby_part}".strip()
        return base.strip()

    def _sanitize_planner_response(self, response: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Normalize planner outputs while keeping the full view-prefixed instruction."""
        if not response:
            return response
        response = dict(response)
        if 'next_waypoint' not in response and response.get('next_waypoint_destination') is not None:
            response['next_waypoint'] = response.pop('next_waypoint_destination')
        elif 'next_waypoint_destination' in response:
            response.pop('next_waypoint_destination', None)
        if 'subtask_landmark' not in response and response.get('next_waypoint_landmark') is not None:
            response['subtask_landmark'] = response.pop('next_waypoint_landmark')
        elif 'next_waypoint_landmark' in response:
            response.pop('next_waypoint_landmark', None)
        for key in (
            "current_waypoint",
            "waypoint_sequence",
            "waypoint_chain",
            "task_progress",
            "next_waypoint",
            "subtask_instruction",
            "subtask_landmark",
        ):
            if isinstance(response.get(key), str):
                response[key] = strip_space_type_variant_suffixes(response.get(key))
        response["current_waypoint"] = self._sanitize_current_waypoint_text(
            response.get("current_waypoint")
        )
        response["subtask_instruction"] = self._sanitize_subtask_instruction_text(
            response.get("subtask_instruction"),
            response.get("next_waypoint"),
            response.get("next_waypoint_direction"),
            keep_view_prefix=True,
        )
        return response

    def _on_lookaround_step(
        self,
        *,
        phase: str,
        look_index: int,
        look_step: int,
        obs: Dict[str, Any],
        info: Dict[str, Any],
    ) -> None:
        """VLM-only hook: save the RGB + top-down navigation panel during lookaround."""
        if not self.nav_visualizer:
            return

        subtask_text = self.current_subtask.get('subtask_instruction', '') if self.current_subtask else f"[Lookaround {phase}]"
        distance = 0.0
        if info:
            distance = info.get('distance_to_goal', 0.0)

        self.nav_visualizer.save_step_visualization(
            observations=obs,
            info=info or {},
            step=look_step,
            instruction=self.current_instruction,
            current_subtask=subtask_text,
            distance=distance,
            action=f"TURN_LEFT (360 scan {look_index}/12)",
            subtask_id=phase
        )
    
    def _collect_lookaround_direction_views(self, phase: str = "initial") -> Tuple[List[str], List[str]]:
        """Run the shared lookaround scan, then render the 12 thinking views for the VLM."""
        scan_state = self._capture_lookaround_scan(
            phase=phase,
            enable_landmark_detection=False,
        )
        if not scan_state:
            return [], []

        lookaround_images = scan_state.get("lookaround_images", []) or []
        lookaround_depths = scan_state.get("lookaround_depths", []) or []
        final_map_state = scan_state.get("final_map_state")
        final_last_waypoint_angle = scan_state.get("final_last_waypoint_angle")

        waypoint_info = None
        last_waypoint_angle_deg = None
        if phase != "initial" and final_map_state is not None:
            wp_positions = final_map_state.get('waypoint_positions', [])
            wp_ids = final_map_state.get('waypoint_ids', [])
            if wp_positions:
                if final_last_waypoint_angle is not None:
                    last_waypoint_angle_deg = np.degrees(final_last_waypoint_angle)
                _, orig_wp_ids, wp_descriptions = self.mapper.get_waypoints()
                waypoint_info = (wp_positions, wp_ids, wp_descriptions)
        
        directions_dir = os.path.join(self.config.RESULTS_DIR, f"episode_{self.current_episode_id}", "directions")
        def _render_thinking_detection(
            image: np.ndarray,
            detections,
            labels: List[str],
            depth_meters: Optional[np.ndarray],
        ):
            return self.visualizer.render_detection_bbox(
                image,
                detections,
                labels,
                landmark_classes=self.landmark_classes,
                depth_meters=depth_meters,
                hfov=self.config.MAP.HFOV,
                landmark_dist_map=None,
                landmark_dist_map_multi=None,
                show_action_partitions=False,
                append_bottom_strip=False,
                controller=None,
                return_visible_entries=True,
            )

        direction_paths, direction_names = self.thinking_view_renderer.save_direction_views(
            directions_dir=directions_dir,
            phase=phase,
            lookaround_images=lookaround_images,
            lookaround_depths=lookaround_depths,
            landmark_classes=self.landmark_classes,
            detect_landmarks_fn=self._detect_landmarks_for_visualization,
            render_detection_fn=_render_thinking_detection,
            draw_distance_fn=self.visualizer.draw_distance_on_view,
            distance_lookup=self.latest_obstacle_distances_12,
            waypoint_info=waypoint_info,
            waypoint_area_labels=(final_map_state or {}).get('waypoint_area_labels', []),
            current_pose=(final_map_state or {}).get('full_pose'),
            resolution_cm=float(getattr(self.mapper, 'resolution', 5)),
            current_space_area_label=str((final_map_state or {}).get('current_space_area_label', 'Unknown') or 'Unknown'),
            full_map=(final_map_state or {}).get('full_map'),
            crop_offset=(final_map_state or {}).get('crop_offset'),
            waypoint_angle_deg=last_waypoint_angle_deg,
            draw_waypoints_fn=self._draw_waypoints_on_view,
        )
        
        # print(f"  12方向独立视图已保存")
        
        return direction_paths, direction_names

    def run_lookaround_and_update_state(self, phase: str) -> Dict[str, Any]:
        """Unified lookaround entry used by initial / verify thinking cycles."""
        direction_paths, direction_names = self._collect_lookaround_direction_views(phase)
        return {
            "phase": str(phase),
            "direction_paths": direction_paths,
            "direction_names": direction_names,
            "global_map_path": getattr(self, 'latest_global_map', None),
            "local_map_path": getattr(self, 'latest_local_map', None),
        }

    def _collect_thinking_detected_landmarks(self) -> List[str]:
        """Collect landmark names seen during the latest lookaround for thinking verification."""
        if hasattr(self, 'current_step_landmarks') and self.current_step_landmarks:
            all_landmarks = set()
            for landmarks_list in self.current_step_landmarks.values():
                for name, _conf in landmarks_list:
                    all_landmarks.add(name)
            return sorted(list(all_landmarks))
        return sorted(list(self.detected_classes)) if hasattr(self, 'detected_classes') else []

    def _run_thinking_cycle(
        self,
        mode: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
        """Run the shared lookaround -> planner -> save pipeline for initial and verify."""
        if not self.planner:
            print("[ERR] LLM Planner not initialized")
            self.latest_thinking_cycle_info = {"mode": mode, "reason": "planner_not_initialized"}
            return None, None, None

        mode_key = str(mode).strip().lower()
        if mode_key not in {"initial", "verify"}:
            print(f"[ERR] Unsupported thinking mode: {mode}")
            self.latest_thinking_cycle_info = {"mode": mode_key, "reason": "unsupported_mode"}
            return None, None, None

        if mode_key == "initial":
            phase = "initial"
            thinking_dir = os.path.join(self.save_manager.episode_dir, "thinking", "subtask_1")
            print(f"\n[LLM] Planning...")
        else:
            if not self.current_subtask:
                self.latest_thinking_cycle_info = {"mode": mode_key, "reason": "missing_current_subtask"}
                return None, None, None
            attempt_letter = chr(ord('a') + self.subtask_attempt)
            phase = f"verify_{self.subtask_count}{attempt_letter}"
            thinking_dir = os.path.join(
                self.save_manager.episode_dir,
                "thinking",
                f"subtask_{self.subtask_count + 1}",
            )
            print(f"\n[Verify] #{self.subtask_count}{attempt_letter} (lookaround step {self.current_step + 1}-{self.current_step + 12})")

        lookaround_state = self.run_lookaround_and_update_state(phase)
        image_paths = lookaround_state.get("direction_paths", []) or []
        direction_names = lookaround_state.get("direction_names", []) or []
        if not image_paths:
            failure_reason = "lookaround_failed"
            if str(getattr(self, 'latest_lookaround_end_reason', '') or '') == "episode_done" or bool(getattr(self, 'latest_done', False)):
                failure_reason = "episode_done_during_lookaround"
            self.latest_thinking_cycle_info = {
                "mode": mode_key,
                "phase": phase,
                "thinking_dir": thinking_dir,
                "reason": failure_reason,
            }
            if failure_reason == "episode_done_during_lookaround":
                print("[WARN] Episode budget ended during lookaround; stop thinking and finalize")
            elif mode_key == "initial":
                print("[ERR] Initial lookaround failed, cannot start planning")
            else:
                print("[ERR] Lookaround failed, cannot verify")
            return None, None, dict(self.latest_thinking_cycle_info)

        global_map = lookaround_state.get("global_map_path")
        if not global_map or not os.path.exists(global_map):
            print(f"[ERR] Global map not found: {global_map}")
            self.latest_thinking_cycle_info = {
                "mode": mode_key,
                "phase": phase,
                "thinking_dir": thinking_dir,
                "reason": "global_map_missing",
            }
            return None, None, dict(self.latest_thinking_cycle_info)

        os.makedirs(thinking_dir, exist_ok=True)
        obstacle_distances = getattr(self, 'latest_obstacle_distances', {
            'front': 'Unknown',
            'left_30': 'Unknown',
            'right_30': 'Unknown',
        })

        detected_landmarks: List[str] = []
        waypoint_summary: Optional[str] = None
        if mode_key == "initial":
            response, prompt = self.planner.generate_initial_subtask(
                instruction=self.current_instruction,
                observation_images=image_paths,
                direction_names=direction_names,
                global_map_image=global_map,
                local_map_image=None,
                obstacle_distances=obstacle_distances,
                save_dir=thinking_dir,
            )
        else:
            detected_landmarks = self._collect_thinking_detected_landmarks()
            waypoint_summary = self._get_waypoint_summary(include_area_chain=True)
            verify_replan_prompt_notice = str(getattr(self, 'verify_replan_prompt_notice', '') or '').strip()
            response, prompt = self.planner.verify_and_replan(
                instruction=self.current_instruction,
                current_subtask=self.current_subtask,
                observation_images=image_paths,
                direction_names=direction_names,
                global_map_image=global_map,
                local_map_image=None,
                detected_landmarks=detected_landmarks,
                waypoint_summary=waypoint_summary,
                obstacle_distances=obstacle_distances,
                verify_replan_prompt_notice=verify_replan_prompt_notice,
                save_dir=thinking_dir,
            )
            self.verify_replan_prompt_notice = ""

        if not response:
            self.latest_thinking_cycle_info = {
                "mode": mode_key,
                "phase": phase,
                "thinking_dir": thinking_dir,
                "reason": "planner_failed",
                "detected_landmarks": detected_landmarks,
                "waypoint_summary": waypoint_summary,
            }
            if mode_key == "initial":
                print("[ERR] LLM Planning failed")
            else:
                print("[ERR] LLM Verify failed")
            return None, prompt, dict(self.latest_thinking_cycle_info)

        response = self._sanitize_planner_response(response)
        with open(os.path.join(thinking_dir, "response.json"), 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        cycle_info = {
            "mode": mode_key,
            "phase": phase,
            "thinking_dir": thinking_dir,
            "detected_landmarks": detected_landmarks,
            "waypoint_summary": waypoint_summary,
        }
        self.latest_thinking_cycle_info = dict(cycle_info)
        return response, prompt, cycle_info

    def _reset_post_thinking_action_state(self) -> None:
        """Reset per-subtask action state before handing control back to action."""
        if self.mapper is not None:
            self.mapper.clear_trajectory()
        self._reset_custom_landmark_state()
        self.progress_summary = ""
        self.previous_action_reason = ""
        self.action_stagnation_streak = 0
        self.verify_replan_prompt_notice = ""
        self.pose_before_action = None
        self.last_planned_degrees = 0
        self.last_planned_meters = 0
        self.last_action_name = ""

    def _record_current_position_from_thinking_response(self, response: Dict[str, Any]) -> None:
        """Store the planner-localized position for later verification/debug use."""
        self.current_position_info = {
            'waypoint': response.get('current_waypoint', 'Unknown'),
            'observation': response.get('current_observation', ''),
            'step': self.current_step,
        }

    def _auto_rotate_to_current_subtask_waypoint(self) -> bool:
        """Rotate to the planned waypoint heading before the action controller starts."""
        if not self.current_subtask:
            return True

        next_waypoint_direction = self.current_subtask.get('next_waypoint_direction', '')
        if next_waypoint_direction and 'Front' not in next_waypoint_direction:
            success, action_sequence = self.auto_rotate_to_waypoint(next_waypoint_direction)
            if success and action_sequence:
                rotation_ok = self.execute_rotation_sequence(action_sequence)
                print()
                return rotation_ok
        return True

    def _apply_thinking_cycle_result(
        self,
        response: Dict[str, Any],
        cycle_info: Dict[str, Any],
        mode: str,
    ) -> bool:
        """Apply a thinking result using the shared replan-style area update flow."""
        mode_key = str(mode).strip().lower()
        is_initial = mode_key == 'initial'
        task_finished = bool(response.get('global_task_finish', False))
        phase_default = 'initial' if is_initial else ''
        previous_match_streak = int(getattr(self, 'final_goal_destination_match_streak', 0) or 0)
        (
            match_hit,
            last_chain_node,
            next_destination,
            match_streak,
            anchor_distance_m,
            stayed_inside_anchor_region,
            restarted_by_anchor_drift,
        ) = self._update_final_goal_destination_match_streak(response)

        if match_hit and stayed_inside_anchor_region:
            print(
                "[GoalRegionMatch] "
                f"waypoint_chain tail='{last_chain_node}' matches destination='{next_destination}' "
                f"| streak={match_streak}/{self.final_destination_match_autostop_streak} "
                f"| anchor_distance={float(anchor_distance_m or 0.0):.2f}/{self.final_destination_match_autostop_radius_m:.2f}m"
            )
        elif match_hit and restarted_by_anchor_drift:
            print(
                "[GoalRegionMatch] "
                f"waypoint_chain tail='{last_chain_node}' still matches destination='{next_destination}', "
                f"but the agent moved {float(anchor_distance_m or 0.0):.2f}m away from the first matched pose "
                f"(limit {self.final_destination_match_autostop_radius_m:.2f}m); restart stable-goal streak from 1"
            )
        elif match_hit:
            print(
                "[GoalRegionMatch] "
                f"waypoint_chain tail='{last_chain_node}' matches destination='{next_destination}', "
                "but current pose was unavailable so the spatial stable-goal window could not be verified yet"
            )
        elif previous_match_streak > 0 and self.final_goal_destination_match_streak == 0:
            print("[GoalRegionMatch] streak reset (final waypoint tail no longer matches destination)")

        auto_finish_by_streak = (
            not task_finished and
            match_hit and
            stayed_inside_anchor_region and
            match_streak >= int(self.final_destination_match_autostop_streak)
        )
        if auto_finish_by_streak:
            task_finished = True
            response['global_task_finish'] = True
            response['auto_task_finish_by_destination_streak'] = True
            response['auto_task_finish_by_goal_region_stability'] = True
            print(
                "[AutoTaskComplete] "
                f"final waypoint tail matched next destination for {match_streak} consecutive thinking cycles "
                f"and all matched poses stayed within {self.final_destination_match_autostop_radius_m:.2f}m of the first matched pose; "
                "stop the task even though the planner did not set global_task_finish."
            )

        if not is_initial:
            attempt_letter = chr(ord('a') + self.subtask_attempt)
            print(f"  #{self.subtask_count}{attempt_letter} -> {self._get_next_waypoint_field(response) or 'N/A'} | finish={task_finished}")

        self._apply_postplanning_space_area_update(
            response=response,
            phase=str(cycle_info.get('phase', phase_default)),
            thinking_dir=cycle_info.get('thinking_dir'),
            refresh_direction_views=True,
        )
        self._record_current_position_from_thinking_response(response)

        if task_finished:
            self.current_subtask = response
            if is_initial:
                self.subtask_count = 1
                self.subtask_attempt = 0
                self._print_subtask_info(response, is_initial=True)
                if auto_finish_by_streak:
                    print('[DONE] Global task complete by final-goal destination streak at initial planning')
                else:
                    print('[DONE] Global task complete at initial planning')
            else:
                if auto_finish_by_streak:
                    print('[DONE] Global task complete by final-goal destination streak')
                else:
                    print('[DONE] Global task complete')
            return True

        self._reset_post_thinking_action_state()

        if is_initial:
            self.subtask_count = 1
            self.subtask_attempt = 0
        else:
            self.subtask_count += 1
            self.subtask_attempt = 0
            print(f"  Next #{self.subtask_count}a: {response.get('subtask_instruction', 'N/A')[:60]}")

        self.current_subtask = response

        next_waypoint_landmark = self._get_subtask_landmark_field(response)
        self._set_current_landmark_tracking(
            next_waypoint_landmark,
            fallback_sources=[
                self._get_next_waypoint_field(response),
                response.get('subtask_instruction'),
                response.get('current_waypoint'),
            ]
        )

        self._print_subtask_info(response, is_initial=is_initial)
        rotation_ok = self._auto_rotate_to_current_subtask_waypoint()
        if not rotation_ok and self._episode_done_cached():
            print('[WARN] Episode ended while rotating toward the next waypoint; finalize current episode.')
            return True
        return False

    def _run_thinking_controller(
        self,
        mode: str,
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        """Shared thinking controller for both initial planning and verify/replan."""
        response, prompt, cycle_info = self._run_thinking_cycle(mode=mode)
        if not response or cycle_info is None:
            return 'failed', None, prompt

        task_finished = self._apply_thinking_cycle_result(
            response=response,
            cycle_info=cycle_info,
            mode=mode,
        )
        if task_finished:
            return 'complete', response, None
        return 'action', response, prompt

    def generate_initial_subtask(self) -> Optional[Dict]:
        """Backward-compatible wrapper around the shared thinking controller."""
        next_mode, response, _prompt = self._run_thinking_controller(mode='initial')
        if next_mode == 'failed':
            return None
        return response

    def _apply_postplanning_space_area_update(
        self,
        response: Dict[str, Any],
        phase: str,
        thinking_dir: Optional[str] = None,
        refresh_direction_views: bool = True,
    ) -> Optional[int]:
        """Persist planner space-area output into the world map and refresh debug renders."""
        if self.mapper is None:
            return None

        waypoint_desc = response.get('current_waypoint', 'Unknown location')
        waypoint_id = self.mapper.add_waypoint(waypoint_desc)

        refreshed_maps = self._refresh_postplanning_map_snapshots(phase=phase)
        if not refreshed_maps:
            self._refresh_step_visualization_snapshot(
                phase=phase,
                enable_landmark_detection=False,
                force=True,
            )
        if refresh_direction_views:
            self._refresh_cached_lookaround_direction_views(phase=phase)
        self._save_waypoint_area_memory_snapshot()
        return waypoint_id

    def _refresh_postplanning_map_snapshots(self, phase: str) -> bool:
        """Force-refresh global/local map images after planner-created space areas are written."""
        return self._refresh_current_map_snapshots(
            phase=phase,
            landmark_classes=list(self.landmark_classes),
        )

    def _refresh_cached_lookaround_direction_views(self, phase: str) -> bool:
        """Re-render cached 12 views after planning updates so current area/waypoint area stay in sync."""
        if (
            not self.latest_lookaround_images or
            not self.latest_lookaround_depths or
            self.latest_lookaround_phase != phase or
            len(self.latest_lookaround_images) < 12 or
            len(self.latest_lookaround_depths) < 12
        ):
            return False

        if not hasattr(self, 'mapper') or self.mapper is None:
            return False

        map_state = self.mapper.get_map_state()
        waypoint_info = None
        wp_positions, wp_ids, wp_descriptions = self.mapper.get_waypoints()
        if wp_positions and wp_ids:
            waypoint_info = (wp_positions, wp_ids, wp_descriptions)

        directions_dir = os.path.join(self.config.RESULTS_DIR, f"episode_{self.current_episode_id}", 'directions')

        def _render_thinking_detection(
            image: np.ndarray,
            detections,
            labels: List[str],
            depth_meters: Optional[np.ndarray],
        ):
            return self.visualizer.render_detection_bbox(
                image,
                detections,
                labels,
                landmark_classes=self.landmark_classes,
                depth_meters=depth_meters,
                hfov=self.config.MAP.HFOV,
                landmark_dist_map=None,
                landmark_dist_map_multi=None,
                show_action_partitions=False,
                append_bottom_strip=False,
                controller=None,
                return_visible_entries=True,
            )

        self.thinking_view_renderer.save_direction_views(
            directions_dir=directions_dir,
            phase=phase,
            lookaround_images=[img.copy() for img in self.latest_lookaround_images],
            lookaround_depths=[depth.copy() if depth is not None else None for depth in self.latest_lookaround_depths],
            landmark_classes=self.landmark_classes,
            detect_landmarks_fn=self._detect_landmarks_for_visualization,
            render_detection_fn=_render_thinking_detection,
            draw_distance_fn=self.visualizer.draw_distance_on_view,
            distance_lookup=self.latest_obstacle_distances_12,
            waypoint_info=waypoint_info,
            waypoint_area_labels=map_state.get('waypoint_area_labels', []),
            current_pose=map_state.get('full_pose'),
            resolution_cm=float(getattr(self.mapper, 'resolution', 5)),
            current_space_area_label=str(map_state.get('current_space_area_label', 'Unknown') or 'Unknown'),
            full_map=map_state.get('full_map'),
            crop_offset=map_state.get('crop_offset'),
            waypoint_angle_deg=None,
            draw_waypoints_fn=self._draw_waypoints_on_view,
        )
        return True

    def auto_rotate_to_waypoint(self, waypoint_direction: str) -> Tuple[bool, List[Dict]]:
        """
        解析waypoint方向并生成旋转动作序列
        
        Args:
            waypoint_direction: 如 "IMAGE 5 (Left 120deg)"
            
        Returns:
            (success, action_sequence): 
                - success: 是否成功解析
                - action_sequence: 动作序列，每个动作为 {"action": "TURN_LEFT/RIGHT", "degrees": 30}
        """
        import re
        
        match = re.search(r'Left (\d+)(?:deg|°)|Right (\d+)(?:deg|°)|Back (\d+)(?:deg|°)?|Front', waypoint_direction)
        
        if not match:
            print(f"  [WARN] Cannot parse waypoint_direction: {waypoint_direction}")
            return False, []
        
        angle = 0
        direction = None
        
        if 'Left' in waypoint_direction:
            angle = int(match.group(1))
            direction = 'LEFT'
        elif 'Right' in waypoint_direction:
            angle = int(match.group(2))
            direction = 'RIGHT'
        elif 'Back' in waypoint_direction:
            angle = 180
            direction = 'LEFT'
        elif 'Front' in waypoint_direction:
            return True, []
        else:
            print(f"  [WARN] Unrecognized direction: {waypoint_direction}")
            return False, []
        
        num_turns = angle // 30
        action_sequence = []
        
        for _ in range(num_turns):
            action_sequence.append({
                'action': f'TURN_{direction}',
                'degrees': 30
            })
        
        return True, action_sequence
    
    def execute_rotation_sequence(self, action_sequence: List[Dict]) -> bool:
        """
        执行旋转动作序列（使用统一的执行器，确保地图更新、步数记录、可视化保存）
        """
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        
        for i, action_dict in enumerate(action_sequence):
            action_name = action_dict['action']
            
            if action_name == 'TURN_LEFT':
                action_id = HabitatSimActions.TURN_LEFT
            elif action_name == 'TURN_RIGHT':
                action_id = HabitatSimActions.TURN_RIGHT
            else:
                print(f"    [WARN] Unknown action: {action_name}")
                continue
            
            is_last_turn = (i == len(action_sequence) - 1)
            result = self.step_with_vlm(
                action_id,
                action_name,
                save_vis=True,
                enable_landmark_detection=is_last_turn,
            )
            
            if result.get('done', False):
                print(f"    [WARN] Episode ended during rotation")
                return False
        
        return True
    
    def verify_and_replan(self) -> Tuple[Optional[Dict], Optional[str]]:
        """Backward-compatible wrapper around the shared thinking controller."""
        if not self.current_subtask:
            return None, None
        next_mode, response, prompt = self._run_thinking_controller(mode='verify')
        if next_mode == 'failed':
            return None, None
        if next_mode == 'complete':
            return response, None
        return response, prompt

    def execute_action_with_vlm(self) -> Tuple[Optional[int], Optional[str], bool, int, Optional[Dict]]:
        """
        使用VLM决策并执行动作
        
        Returns:
            (action_id, action_name, should_stop, repeat_count, response)
        """
        if not self.action_executor or not self.current_subtask:
            return None, None, True, 1, None

        if self._episode_done_cached():
            print("[WARN] Episode already done, skip action decision")
            return None, None, True, 1, None
        
        # 获取当前观察：使用缓存的观察或通过旋转获取
        if self.latest_obs is not None:
            obs = self.latest_obs
        else:
            # 如果没有缓存，执行一次右转再左转回来获取观察
            actions = [{"action": HabitatSimActions.TURN_RIGHT}]
            step_data = self._safe_env_step(actions, context="action observation refresh turn-right")
            if step_data is None:
                return None, None, True, 1, None
            obs, _, dones, _ = step_data
            if dones[0]:
                print("[WARN] Episode ended")
                return None, None, True, 1, None
            
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            step_data = self._safe_env_step(actions, context="action observation refresh turn-left")
            if step_data is None:
                return None, None, True, 1, None
            obs, _, dones, _ = step_data
            if dones[0]:
                print("[WARN] Episode ended")
                return None, None, True, 1, None
            obs = obs[0]
        
        # 获取最新保存的观察信息
        # 上一步已保存的文件（如果current_step=13，则读取step_0012的地图）
        last_step = self.current_step  # execute_action在step执行前调用，所以用current_step
        
        # 生成当前子任务的phase标识
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        action_phase = f"action{self.subtask_count}{attempt_letter}"

        # 首次Action前检测：不在旋转过程中检测，等旋转结束后在当前朝向做一次
        self._run_pre_action_detection_snapshot(action_phase)
        
        # 智能查找可用的图像：优先使用action phase，回退到verify/initial
        # 可能的phase顺序: action2a -> verify_2a -> verify_1a -> initial (注意verify带下划线)
        possible_phases = [action_phase]
        
        # 添加当前子任务的验证phase（验证完成后保存的全景图）
        current_verify_phase = f"verify_{self.subtask_count}a"
        possible_phases.append(current_verify_phase)
        
        if self.subtask_attempt > 0:
            # 如果是1b, 1c等，可能需要回退到上一次尝试的verify
            prev_attempt_verify = f"verify_{self.subtask_count}{chr(ord('a') + self.subtask_attempt - 1)}"
            possible_phases.append(prev_attempt_verify)
        
        if self.subtask_count > 1:
            # 回退到上一个子任务的verify
            prev_verify_phase = f"verify_{self.subtask_count - 1}a"
            possible_phases.append(prev_verify_phase)
        
        # 最后回退到initial
        possible_phases.append("initial")
        
        # 查找RGB图像
        fp_image = None
        for phase in possible_phases:
            candidate = os.path.join(self.episode_dir, 'rgb', f'step_{last_step:04d}_{phase}.png')
            if os.path.exists(candidate):
                fp_image = candidate
                break
        
        # 如果都不存在，用当前观察创建临时文件
        if not fp_image:
            rgb_bgr = cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR)
            temp_image = os.path.join(self.episode_dir, f'temp_fp_step{last_step}.png')
            cv2.imwrite(temp_image, rgb_bgr)
            fp_image = temp_image
        
        # 查找对应的semantic masks
        mask_path = None
        for phase in possible_phases:
            candidate = os.path.join(self.episode_dir, 'semantic_masks', f'step_{last_step:04d}_{phase}.npy')
            if os.path.exists(candidate):
                mask_path = candidate
                break
        
        # 为RGB图像不添加距离辅助线（只有detection才显示距离）
        fp_image = self.visualizer.prepare_action_image_with_enhancements(
            fp_image, mask_path, self.latest_obstacle_distances, self.classes, use_floor=False, use_distance=False)
        
        # 查找detection图像（使用相同的回退逻辑）
        detection_image = None
        detection_step = None  # 记录找到的detection图像对应的step
        for phase in possible_phases:
            candidate = os.path.join(self.episode_dir, 'detection', f'step_{last_step:04d}_{phase}.png')
            if os.path.exists(candidate):
                detection_image = candidate
                detection_step = last_step
                break
        if not detection_image:
            print(f"  [WARN] Detection image not found for step {last_step}")
        else:
            # 距离线已在save_step_visualization内直接从full_map计算并画入detection图
            # 不需要再叠加
            detection_image = self.visualizer.prepare_action_image_with_enhancements(
                detection_image, mask_path, self.latest_obstacle_distances, self.classes, use_floor=False, use_distance=False)
        
        # 获取detection图像对应的landmark类别
        # 使用找到的detection图像对应的step
        detected_landmarks = None
        step_landmark_entries = []
        if detection_step is not None and hasattr(self, 'current_step_action_landmark_topk_entries'):
            step_landmark_entries = self.current_step_action_landmark_topk_entries.get(detection_step, []) or []

        if detection_step is not None and hasattr(self, 'current_step_landmarks') and detection_step in self.current_step_landmarks:
            # 当前step检测到的landmarks: [(name, confidence), ...]
            step_landmarks = self.current_step_landmarks[detection_step]
            if step_landmarks:
                detected_landmarks = ', '.join([name for name, _ in step_landmarks])
        
        # 退化策略：如果没有检测结果，报告"未检测到"
        if not detected_landmarks:
            if hasattr(self, 'target_landmark') and self.target_landmark:
                detected_landmarks = f"No {self.target_landmark} detected in current view"
            else:
                detected_landmarks = "No landmarks detected"
        
        # 使用最新的障碍物距离（在step_with_vlm中已更新）
        obstacle_distances = getattr(self, 'latest_obstacle_distances', {
            'front': 'Unknown',
            'left_30': 'Unknown',
            'right_30': 'Unknown',
        })
        
        # 准备action记录
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        subtask_id = f"{self.subtask_count}{attempt_letter}"
        
        # 保存子任务信息
        subtask_info = {
            "subtask_id": self.subtask_count,
            "next_waypoint": self._get_next_waypoint_field(self.current_subtask),
            "subtask_instruction": self.current_subtask.get('subtask_instruction', ''),
            "start_step": self.current_step,
            "timestamp": datetime.now().isoformat()
        }
        
        # 计算save_dir: API发送时同步保存压缩图片+prompt
        subtask_dir = os.path.join(self.save_manager.episode_dir, "action", f"subtask_{subtask_id}")
        action_save_dir = os.path.join(subtask_dir, f"step_{self.current_step + 1}")
        os.makedirs(action_save_dir, exist_ok=True)

        waypoint_summary = self._get_waypoint_summary(include_area_chain=True, include_path=True)
        
        # 保存子任务信息（首次创建时）
        info_file = os.path.join(subtask_dir, "info.json")
        if not os.path.exists(info_file):
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(subtask_info, f, ensure_ascii=False, indent=2)

        # 构建 landmark_map_info：可见 + 地图离屏两类（按距离升序）
        # action VLM 只有第一人称图，需要文字告知离屏的已映射 landmark 方向距离
        landmark_dist_map = getattr(self, 'latest_landmark_dist_map', {})
        landmark_dist_map_multi = getattr(self, 'latest_landmark_dist_map_multi', {})
        action_landmark_map_info = build_action_landmark_map_info(
            step_landmark_entries=step_landmark_entries,
            landmark_dist_map=landmark_dist_map,
            landmark_dist_map_multi=landmark_dist_map_multi,
            landmark_instances_world=getattr(self, 'latest_landmark_instances_world', []),
        )

        action_subtask_instruction = self._sanitize_subtask_instruction_text(
            self.current_subtask.get('subtask_instruction', ''),
            self._get_next_waypoint_field(self.current_subtask),
            self.current_subtask.get('next_waypoint_direction', ''),
            keep_view_prefix=False,
        )

        # 调用VLM决策（save_dir使call_api在发送时保存压缩图片+prompt）
        result = self.action_executor.decide_action(
            next_waypoint_destination=self._get_next_waypoint_field(self.current_subtask),
            subtask_instruction=action_subtask_instruction,
            first_person_image=fp_image,
            action_mapping=ACTION_MAPPING,
            progress_summary=self.progress_summary,
            waypoint_summary=waypoint_summary,
            detection_image=detection_image,
            local_map_image=None,
            detected_landmarks=detected_landmarks,
            previous_action_reason=self.previous_action_reason,
            obstacle_distances=obstacle_distances,
            landmark_map_info=action_landmark_map_info,
            save_dir=action_save_dir
        )
        
        if len(result) == 7:
            action_id, action_name, _, response, degrees, meters, prompt = result  # 忽略updated_progress
        elif len(result) == 6:
            action_id, action_name, _, response, degrees, meters = result  # 忽略updated_progress
            prompt = None
        else:
            # 兼容旧版本返回（没有degrees/meters）
            action_id, action_name, _, response = result  # 忽略updated_progress
            degrees, meters = 0, 0
            prompt = None
        
        if action_id is None:
            print("[ERR] VLM decision failed")
            return None, None, True, 1, None
        
        # 保存response（API返回后，到同一个save_dir）
        with open(os.path.join(action_save_dir, "response.json"), 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        # 保存planned action参数，供后续计算actual progress使用
        self.last_planned_degrees = degrees
        self.last_planned_meters = meters
        self.last_action_name = action_name
        
        # 保存当前的action_analysis作为下一次的previous_action_reason
        if response and 'action_analysis' in response:
            self.previous_action_reason = response['action_analysis']
        else:
            self.previous_action_reason = ""
        
        # 检查是否停止
        should_stop = (action_name == "STOP")
        
        # 计算需要重复执行的次数
        repeat_count = 1
        if action_name == 'TURN_LEFT' or action_name == 'TURN_RIGHT':
            # 每次转30度，计算需要转几次
            if degrees > 0:
                repeat_count = max(1, round(degrees / self.action_executor.turn_angle))
        elif action_name == 'MOVE_FORWARD':
            # 每次移动0.25m，计算需要移动几次
            if meters > 0:
                repeat_count = max(1, round(meters / self.action_executor.move_distance))
        
        return action_id, action_name, should_stop, repeat_count, response
    
    def step_with_vlm(self, action: int, action_name: str = "", save_vis: bool = True,
                      enable_landmark_detection: bool = True) -> Dict[str, Any]:
        """
        执行VLM决策的动作（调用父类step方法）并缓存观察
        
        Args:
            action: 动作ID
            action_name: 动作名称（用于可视化）
            save_vis: 是否保存可视化
            enable_landmark_detection: 是否启用landmark检测（旋转阶段可关闭节省算力）
            
        Returns:
            步骤结果字典
        """
        # 生成phase标识: action1a, action2b等
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        phase = f"action{self.subtask_count}{attempt_letter}"
        
        result = self.step(
            action,
            save_vis,
            phase,
            enable_landmark_detection=enable_landmark_detection,
        )
        # 缓存最新观察和info用于下次VLM决策和可视化
        self.latest_obs = result.get('obs', None)
        self.latest_info = result.get('info', None)
        
        # 地图已更新，立即计算当前位置的障碍物距离
        self._update_obstacle_distances()
        
        # 保存RGB+俯视图拼接可视化
        if save_vis and self.nav_visualizer and self.latest_obs is not None:
            subtask_text = None
            if self.current_subtask:
                subtask_text = self.current_subtask.get('subtask_instruction', '')
            
            distance = 0.0
            if self.latest_info:
                distance = self.latest_info.get('distance_to_goal', 0.0)
            
            attempt_letter = chr(ord('a') + self.subtask_attempt)
            subtask_id = f"{self.subtask_count}{attempt_letter}"
            
            self.nav_visualizer.save_step_visualization(
                observations=self.latest_obs,
                info=self.latest_info or {},
                step=self.current_step,
                instruction=self.current_instruction,
                current_subtask=subtask_text,
                distance=distance,
                action=action_name,
                subtask_id=subtask_id
            )
        
        return result
    
    def _run_action_controller(self, max_subtask_steps: int = 5) -> str:
        """Run action decisions until control should return to thinking or the episode ends."""
        subtask_steps = 0

        while True:
            if self._episode_done_cached():
                print('[WARN] Episode already done before action execution')
                return 'complete'

            max_retries = 3
            action_id = None
            vlm_response = None

            for retry in range(max_retries):
                action_id, action_name, should_stop, repeat_count, vlm_response = self.execute_action_with_vlm()

                if action_id is not None:
                    break

                if self._episode_done_cached():
                    print('[WARN] Episode already done while preparing the next action')
                    return 'complete'

                if retry < max_retries - 1:
                    wait = (retry + 1) * 2
                    print(f"  [WARN] VLM Action failed, retry in {wait}s ({retry + 1}/{max_retries - 1})...")
                    import time
                    time.sleep(wait)

            if action_id is None:
                print('[ERR] VLM Action failed after all retries, skipping step')
                continue

            if vlm_response and vlm_response.get('global_task_finish', False):
                print(f"[DONE] Task complete (action) | steps={self.current_step}")
                return 'complete'

            if should_stop:
                print('\n[STOP] -> Thinking controller...')
                return 'thinking'

            subtask_steps += 1
            force_replan_after_action = subtask_steps >= max_subtask_steps
            replan_for_stagnation = False
            stagnation_actual_meters = None
            stagnation_threshold_m = self._get_low_level_stagnation_threshold_m()

            if self.pose_before_action is None:
                self.pose_before_action = self._get_agent_pose()
            pose_before_action_batch = self._get_agent_pose()
            auto_completed_subtask = None

            for i in range(repeat_count):
                pose_before_low_level = self._get_agent_pose()
                result = self.step_with_vlm(action_id, action_name=action_name, save_vis=True)
                pose_after_low_level = self._get_agent_pose()
                low_level_actual_meters = float(np.hypot(
                    pose_after_low_level[0] - pose_before_low_level[0],
                    pose_after_low_level[1] - pose_before_low_level[1],
                ))

                if repeat_count > 1:
                    print(f"  [Step {self.current_step}] {action_name} ({i + 1}/{repeat_count})")
                else:
                    print(f"  [Step {self.current_step}] {action_name} | subtask {subtask_steps}/{max_subtask_steps}")

                if self.latest_info:
                    dtg = self.latest_info.get('distance_to_goal', -1)
                    if not hasattr(self, 'dtg_history'):
                        self.dtg_history = []
                    self.dtg_history.append(dtg)

                if result['done']:
                    print('[WARN] Episode done (Habitat)')
                    return 'complete'

                current_step_landmark_entries = self._get_current_action_step_landmark_entries()
                auto_completed_subtask = self._should_autocomplete_subtask_during_action_step(
                    current_step_landmark_entries
                )
                if auto_completed_subtask is not None:
                    landmark_kind = 'opening-like' if auto_completed_subtask.get('is_opening_like') else 'solid'
                    print(
                        '[AutoSubtaskComplete] '
                        f"{auto_completed_subtask['name']} ({landmark_kind}) reached within "
                        f"{auto_completed_subtask['distance_m']:.2f}m "
                        f"(threshold {float(auto_completed_subtask.get('stop_distance_m', self.ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M)):.2f}m) "
                        f"on action step {self.current_step}; "
                        'return control to the thinking controller.'
                    )
                    break

                replan_for_stagnation = self._update_action_stagnation_streak(
                    action_name,
                    low_level_actual_meters,
                )
                if str(action_name or "").upper() == 'MOVE_FORWARD':
                    stagnation_actual_meters = low_level_actual_meters
                if replan_for_stagnation:
                    break

            if hasattr(self, 'last_action_name') and self.last_action_name:
                pose_after_action_batch = self._get_agent_pose()

                x_before, y_before, ori_before = pose_before_action_batch
                x_after, y_after, ori_after = pose_after_action_batch

                import math
                angle_diff = ori_after - ori_before
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi

                actual_degrees = abs(math.degrees(angle_diff))

                actual_action_name = self.last_action_name
                if self.last_action_name == 'TURN_LEFT' and angle_diff < -0.1:
                    actual_action_name = 'TURN_RIGHT'
                    print(f"[Warning] Planned TURN_LEFT but actually turned RIGHT by {actual_degrees:.1f}°")
                elif self.last_action_name == 'TURN_RIGHT' and angle_diff > 0.1:
                    actual_action_name = 'TURN_LEFT'
                    print(f"[Warning] Planned TURN_RIGHT but actually turned LEFT by {actual_degrees:.1f}°")

                actual_meters = math.sqrt((x_after - x_before) ** 2 + (y_after - y_before) ** 2)

                self.progress_summary = self.action_executor._generate_progress_update(
                    current_progress=self.progress_summary,
                    action_name=actual_action_name,
                    degrees=self.last_planned_degrees,
                    meters=self.last_planned_meters,
                    actual_degrees=actual_degrees,
                    actual_meters=actual_meters
                )

                self.pose_before_action = pose_after_action_batch

            if auto_completed_subtask is not None:
                landmark_kind = 'opening-like' if auto_completed_subtask.get('is_opening_like') else 'solid'
                self._append_progress_note(
                    f"had reached {auto_completed_subtask['name']} ({landmark_kind}) within "
                    f"{auto_completed_subtask['distance_m']:.2f}m "
                    f"(auto-stop threshold {float(auto_completed_subtask.get('stop_distance_m', self.ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M)):.2f}m), "
                    'so ended the current subtask and triggered replan'
                )
                self.previous_action_reason = (
                    f"Displayed destination landmark {auto_completed_subtask['name']} ({landmark_kind}) was within "
                    f"{auto_completed_subtask['distance_m']:.2f}m "
                    f"(threshold {float(auto_completed_subtask.get('stop_distance_m', self.ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M)):.2f}m), "
                    'so the system ended the current subtask and started thinking'
                )
                return 'thinking'

            if replan_for_stagnation:
                latest_actual_meters = (
                    float(stagnation_actual_meters)
                    if stagnation_actual_meters is not None
                    else 0.0
                )
                self.verify_replan_prompt_notice = self._build_stagnation_verify_notice()
                self._append_progress_note(
                    f"tried low-level MOVE_FORWARD {self.action_stagnation_streak} consecutive times but the latest move was only "
                    f"{latest_actual_meters:.2f}m "
                    f"(no-move threshold {stagnation_threshold_m:.2f}m), so triggered replan"
                )
                self.previous_action_reason = (
                    f"The agent made {self.action_stagnation_streak} consecutive low-level MOVE_FORWARD steps with actual movement <= "
                    f"{stagnation_threshold_m:.2f}m "
                    f"(latest actual movement {latest_actual_meters:.2f}m), so the system treated the front route as blocked and started thinking."
                )
                print(
                    "[Replan] Triggered by action stagnation: "
                    f"{self.action_stagnation_streak} consecutive low-level MOVE_FORWARD steps moved <= "
                    f"{stagnation_threshold_m:.2f}m"
                )
                return 'thinking'

            if force_replan_after_action:
                print(f'\n[Replan] Force replan after {max_subtask_steps} steps')
                return 'thinking'

    def run_vlm_navigation(self, max_subtask_steps: int = 5) -> Dict[str, Any]:
        """Run the top-level scheduler over the thinking controller and action controller."""
        max_steps = self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS

        print("\n" + "=" * 60)
        print(f"VLM Navigation | max_steps={max_steps} | subtask_steps={max_subtask_steps}")
        print(f"Instruction: {self.current_instruction}")
        print(f"{'=' * 60}")

        controller_mode = 'thinking'
        thinking_mode = 'initial'
        navigation_complete = False

        while not navigation_complete:
            if controller_mode == 'thinking':
                if not self._has_budget_for_thinking_cycle():
                    remaining_steps = self._get_remaining_episode_steps()
                    print(
                        f"[WARN] Only {remaining_steps} step(s) remain; "
                        f"skip {thinking_mode} lookaround and stop the episode gracefully."
                    )
                    self.latest_thinking_cycle_info = {
                        'mode': thinking_mode,
                        'reason': 'insufficient_steps_for_lookaround',
                        'remaining_steps': remaining_steps,
                    }
                    break

                controller_mode, _response, _prompt = self._run_thinking_controller(mode=thinking_mode)
                if controller_mode == 'failed':
                    cycle_reason = str(getattr(self, 'latest_thinking_cycle_info', {}).get('reason', '') or '')
                    if cycle_reason in {'insufficient_steps_for_lookaround', 'episode_done_during_lookaround'}:
                        print(
                            f"[WARN] Stop {thinking_mode} because the episode budget ended during lookaround; "
                            "finalize with current metrics instead of treating it as a planner error."
                        )
                        break
                    if thinking_mode == 'initial':
                        failure_reason = 'initial_lookaround_failed' if cycle_reason == 'lookaround_failed' else 'initial_subtask_failed'
                    else:
                        failure_reason = 'verify_lookaround_failed' if cycle_reason == 'lookaround_failed' else 'verify_replan_failed'
                    print(f"[ERR] Thinking controller failed ({thinking_mode})")
                    return {
                        'success': False,
                        'total_steps': self.current_step,
                        'subtask_count': self.subtask_count,
                        'detected_classes': list(self.detected_classes) if hasattr(self, 'detected_classes') else [],
                        'gif_path': None,
                        'result_file': None,
                        'reason': failure_reason,
                    }

                if controller_mode == 'complete':
                    navigation_complete = True
                    break

                thinking_mode = 'verify'
                continue

            controller_mode = self._run_action_controller(max_subtask_steps=max_subtask_steps)
            if controller_mode == 'complete':
                navigation_complete = True
                break
            thinking_mode = 'verify'

        total_steps = self.current_step

        if hasattr(self, 'dtg_history') and self.dtg_history:
            valid_dtgs = [d for d in self.dtg_history if d >= 0]
            if valid_dtgs:
                print(f'\nDTG: min={min(valid_dtgs):.2f}m final={valid_dtgs[-1]:.2f}m')

        gif_path = None
        if self.nav_visualizer:
            gif_path = self.nav_visualizer.save_gif(fps=2)

        final_metrics = self.finish_episode(
            success=navigation_complete,
            stop_action=True
        )

        env_metrics = final_metrics if final_metrics else {}
        if not env_metrics:
            try:
                if hasattr(self.envs, 'call_at'):
                    env_metrics = self.envs.call_at(0, 'get_metrics')
            except Exception:
                env_metrics = {}

        env_success = env_metrics.get('success') if isinstance(env_metrics, dict) else None
        final_success = bool(env_success) if env_success is not None else bool(navigation_complete)
        final_result = self._save_navigation_result(final_success, total_steps, env_metrics)

        print("\n" + "=" * 60)
        print(f"{'OK' if final_success else 'FAIL'} | steps={total_steps} | subtasks={self.subtask_count}")
        print(f"{'=' * 60}")

        return {
            'success': final_success,
            'total_steps': total_steps,
            'subtask_count': self.subtask_count,
            'detected_classes': list(self.detected_classes),
            'gif_path': gif_path,
            'result_file': final_result
        }

    def _save_navigation_result(self, success: bool, total_steps: int, env_metrics: Dict = None) -> str:
        """
        保存导航结果到log/目录
        
        VLN-CE关键评估指标说明：
        - NE: 停止时智能体与目标点的距离(米)，对应 distance_to_goal，越小越好
        - SR: 成功率，智能体是否在3米内停止(0或1)，对应 success
        - SPL: Success weighted by Path Length，成功率与路径效率的综合指标
        - OSR: Oracle Success Rate，轨迹中是否曾到达过目标3米内，对应 oracle_success
        - nDTW: 轨迹与GT路径的一致性，范围[0,1]，越高越好
        
        Args:
            success: 是否完成任务
            total_steps: 总步数
            env_metrics: 从环境获取的metrics字典
        """
        import math
        
        def check_inf_nan(value):
            """检查并修正无效值（参考Sub-VLM-VLN）"""
            if isinstance(value, (int, float)):
                if math.isinf(value) or math.isnan(value):
                    return 0
            return value
        
        # 优先使用env_metrics，回退到latest_info
        metrics_source = env_metrics if env_metrics else (self.latest_info if self.latest_info else {})
        
        # 提取并验证核心指标
        result = {
            'episode_id': self.current_episode_id,
            'instruction': self.current_instruction,
            'total_steps': total_steps,
            'subtask_count': self.subtask_count,
            
            # 核心导航指标（带数据验证）
            'success': int(check_inf_nan(metrics_source.get('success', 0))),
            'spl': float(check_inf_nan(metrics_source.get('spl', 0.0))),
            'distance_to_goal': float(check_inf_nan(metrics_source.get('distance_to_goal', -1.0))),
            'ndtw': float(check_inf_nan(metrics_source.get('ndtw', metrics_source.get('nDTW', 0.0)))),
            'path_length': float(check_inf_nan(metrics_source.get('path_length', 0.0))),
            
            # Oracle指标（带数据验证）
            'oracle_success': int(check_inf_nan(metrics_source.get('oracle_success', 0))),
            'oracle_navigation_error': float(check_inf_nan(metrics_source.get('oracle_navigation_error', float('inf')))),
            'oracle_spl': float(check_inf_nan(metrics_source.get('oracle_spl', 0.0))),
            
            # 语义信息（格式化后的）
            'detected_objects': sorted(list(self.detected_classes)),  # 检测到的物体类别（排序后的列表）
            
            # 导航历史
            'subtask_history': self.subtask_history,
            # thinking/action counts removed - no longer tracking in memory
            'timestamp': datetime.now().isoformat()
        }

        result['sr'] = result['success']
        result['osr'] = result['oracle_success']
        result['ne'] = result['distance_to_goal']

        # 打印关键指标（便于实时监控）
        print(
            f"\n[Eval] Episode {self.current_episode_id}: "
            f"NE={result['ne']:.3f}m OSR={result['osr']} SR={result['sr']} "
            f"SPL={result['spl']:.4f} nDTW={result['ndtw']:.4f}"
        )
        
        return self.save_manager.save_result(result)
    
    def _print_subtask_info(self, response: Dict, is_initial: bool = False):
        """打印子任务信息（JSON格式）"""
        # 根据响应类型确定标题
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        if is_initial:
            title = f"Initial Subtask #{self.subtask_count}{attempt_letter}"
        elif 'is_completed' in response:
            # 验证响应
            if response.get('is_completed', False):
                title = f"Subtask #{self.subtask_count}{attempt_letter} - Completed ✓"
            else:
                title = f"Subtask #{self.subtask_count}{attempt_letter} - Continue (Not Completed)"
        else:
            title = f"Subtask #{self.subtask_count}{attempt_letter}"
        
        dest = self._get_next_waypoint_field(response) or 'N/A'
        instr = response.get('subtask_instruction', 'N/A')[:80]
        print(f"  {title}: {dest} | {instr}")
    
    # ========== Waypoint辅助方法 ==========

    def _get_waypoint_summary(self, include_area_chain: bool = True, include_path: bool = True) -> str:
        """
        获取waypoint摘要（用于LLM提示词）
        包含每个waypoint相对当前pose的距离和方向，以及顺序拓扑chain。
        """
        wp_pos, wp_ids, wp_descs = self.mapper.get_waypoints()
        return build_waypoint_summary(
            waypoint_positions=wp_pos,
            waypoint_ids=wp_ids,
            waypoint_descriptions=wp_descs,
            waypoint_area_labels=self.mapper.get_waypoint_area_labels(),
            current_pose=self.mapper.full_pose,
            resolution_cm=self.mapper.resolution,
            current_space_area_label=getattr(self.mapper, 'current_space_area_display_label', ""),
            current_space_area_type=getattr(self.mapper, 'current_space_area_type', ""),
            full_map=getattr(self.mapper, 'full_map', None),
            crop_offset=getattr(getattr(self.mapper, 'mapping_module', None), 'full_map_crop_offset', None),
            include_area_chain=include_area_chain,
            include_path=include_path,
        )

    # ========== 原有方法 ==========
