"""
VLM Navigation Controller
=========================
基于VLM的自动导航控制器

继承InteractiveNavigationController的核心功能：
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

from vlnce_baselines.common.spatial_formatter import (
    build_action_landmark_map_info,
    build_waypoint_summary,
)
from vlnce_baselines.interactive_navigation_controller import InteractiveNavigationController
from vlnce_baselines.mapping.space_types import strip_space_type_variant_suffixes
from vlnce_baselines.vlm import (
    LLMPlanner, ActionExecutor, SaveManager, NavigationVisualizer
)
from vlnce_baselines.vlm.thinking_view_renderer import ThinkingViewRenderer
from vlnce_baselines.config_system.constants import landmark_edge_depth_keywords
from vlnce_baselines.visualization.obstacle_analysis import (
    calculate_obstacle_distances_from_depth,
)
from vlnce_baselines.vlm.navigation_config import ACTION_MAPPING, DIRECTION_CONFIG
from habitat_extensions.pose_utils import get_sim_location


class VLMNavigationController(InteractiveNavigationController):
    """
    VLM导航控制器
    
    继承自InteractiveNavigationController，添加VLM规划和执行功能
    
    工作流程：
    1. 初始环视建图（12步×30°）→ 收集4方向图像
    2. LLM规划 → 生成初始子任务
    3. VLM执行 → 循环执行动作直到子任务完成
    4. 验证环视建图（12步×30°）→ 更新地图和4方向图像
    5. 验证重规划 → 检查完成状态，生成下一子任务
    6. 重复3-5直到导航完成
    
    注意：每次验证重规划前都会执行360°环视，以更新语义地图和当前位置的4方向观察
    """

    ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M = 0.5
    ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M = 1.0
    ACTION_SUBTASK_AUTOCOMPLETE_TOPK = 2
    
    def __init__(self, config: Config,
                 config_path: str = None,
                 llm_config_path: str = "vlnce_baselines/vlm/llm_config.yaml",
                 vlm_config_path: str = "vlnce_baselines/vlm/vlm_config.yaml"):
        """
        初始化VLM导航控制器
        
        Args:
            config: Habitat配置
            config_path: 统一API配置文件路径（同时设置LLM和VLM，优先于下面两个参数）
            llm_config_path: LLM配置文件路径（仅当 config_path=None 时生效）
            vlm_config_path: VLM配置文件路径（仅当 config_path=None 时生效）
        """
        # 统一配置文件优先
        if config_path is not None:
            llm_config_path = config_path
            vlm_config_path = config_path
        
        # 调用父类初始化（初始化环境、检测、建图、可视化）
        super().__init__(config)
        
        # 初始化VLM模块
# print("\n[Init] 初始化VLM模块...")
        
        # 获取动作参数
        self.turn_angle = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE  # 30°
        self.move_distance = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE  # 0.25m
        
        # 动作空间描述
        self.action_space = f"MOVE_FORWARD ({self.move_distance}m), TURN_LEFT ({self.turn_angle}°), TURN_RIGHT ({self.turn_angle}°), STOP"
        
        # 初始化LLM规划器
        try:
            self.planner = LLMPlanner(llm_config_path, self.action_space)
        except Exception as e:
            print(f"[WARN] LLM Planner init failed: {e}")
            self.planner = None
        
        # 初始化VLM执行器
        try:
            self.action_executor = ActionExecutor(vlm_config_path, self.turn_angle, self.move_distance)
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
        
        # 初始化管理器
        self.save_manager = None  # 在reset_episode时初始化
        # waypoint_manager已废弃，直接使用mapper.add_waypoint()
        
        # 观察缓存
        self.latest_obs = None  # 缓存最新的观察
        self.latest_info = None  # 缓存最新的info（包含top_down_map_vlnce）
        self.pose_before_action = None  # 记录动作前的pose (x, y, orientation)
        self.latest_lookaround_images: List[np.ndarray] = []
        self.latest_lookaround_depths: List[np.ndarray] = []
        self.latest_lookaround_phase: str = ""
        
        # 当前子任务跟踪的landmark类别（每个子任务重置）
        self.tracked_landmark_classes = set()
        
        # 障碍物距离缓存
        # Thinking模式（环视）：12个方向（360°每30°）
        self.latest_obstacle_distances_12 = {
            f'angle_{i}': 'Unknown' for i in range(0, 360, 30)
        }
        # Action模式：3个方向（Left 30 / Front / Right 30）
        self.latest_obstacle_distances = {
            'front': 'Unknown',
            'left_30': 'Unknown',
            'right_30': 'Unknown',
        }
        
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

    def _parse_subtask_destination(self) -> Tuple[Optional[str], Optional[str]]:
        destination = ""
        if getattr(self, 'current_subtask', None):
            destination = str(self.current_subtask.get('next_waypoint_destination', '') or '').strip()
        if not destination or "'s " not in destination:
            return None, None

        room_text, object_text = destination.split("'s ", 1)
        room_norm = strip_space_type_variant_suffixes(room_text.strip().lower())
        object_norm = self._normalize_landmark_candidate(object_text)
        return room_norm or None, object_norm or None

    def _should_autocomplete_subtask_during_action_step(
        self,
        step_landmark_entries: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        candidate_names = self._get_current_subtask_landmark_candidates()
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
        """Persist waypoint/room-area state for debugging after each planning update."""
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
        room_area_records = []
        for record in map_state.get('room_area_records', []) or []:
            center_world_px = record.get("center_world_px", (0, 0))
            room_area_records.append({
                "id": int(record.get("id", 0) or 0),
                "label": str(record.get("label", "")),
                "display_label": str(record.get("display_label", record.get("label", ""))),
                "room_type": str(record.get("room_type", "")),
                "variant": int(record.get("variant", 0) or 0),
                "center_world_px": [int(center_world_px[0]), int(center_world_px[1])],
                "connected_area_labels": [str(item) for item in record.get("connected_area_labels", []) or []],
            })

        waypoint_memory = {
            "current_room_area_label": str(map_state.get('current_room_area_label', 'Unknown') or 'Unknown'),
            "current_room_area_type": str(map_state.get('current_room_area_type', 'Unknown') or 'Unknown'),
            "waypoint_positions": waypoint_positions,
            "waypoint_ids": waypoint_ids,
            "waypoint_descriptions": waypoint_descriptions,
            "waypoint_area_labels": waypoint_area_labels,
            "room_area_records": room_area_records,
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
                print("  [AutoRetreat] Reverse path is also blocked after turning around; stop retreat early.")
                break

            result = self.step_with_vlm(
                HabitatSimActions.MOVE_FORWARD,
                action_name="AUTO_RETREAT_FORWARD",
                save_vis=True,
                enable_landmark_detection=True,
            )
            retreated_m += float(self.move_distance)
            print(
                f"  [Step {self.current_step}] AUTO_RETREAT_FORWARD "
                f"({retreated_m:.2f}/{retreat_distance_m:.2f}m)"
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
        else:
            self._append_progress_note(
                "Had encountered obstacles on front/left30/right30 (<0.5m), then turned around but the reverse path was also blocked, and triggered replan"
            )
            self.previous_action_reason = (
                "Front, Left 30deg, and Right 30deg were all blocked under 0.5m, "
                "so the system turned around, found the reverse path blocked too, and triggered rethinking."
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
        self.pose_before_action = None  # 重置pose追踪
        self.latest_lookaround_images = []
        self.latest_lookaround_depths = []
        self.latest_lookaround_phase = ""
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
    
    def _get_agent_pose(self) -> tuple:
        """获取agent当前pose (x, y, orientation)
        
        Returns:
            tuple: (x, y, o) where x, y are coordinates and o is orientation in radians
        """
        # 通过call_at调用environment 0的get_agent_pose方法
        return self.envs.call_at(0, "get_agent_pose")

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
        for key in (
            "current_waypoint",
            "waypoint_sequence",
            "task_progress",
            "next_waypoint_destination",
            "subtask_instruction",
        ):
            if isinstance(response.get(key), str):
                response[key] = strip_space_type_variant_suffixes(response.get(key))
        response["current_waypoint"] = self._sanitize_current_waypoint_text(
            response.get("current_waypoint")
        )
        response["subtask_instruction"] = self._sanitize_subtask_instruction_text(
            response.get("subtask_instruction"),
            response.get("next_waypoint_destination"),
            response.get("next_waypoint_direction"),
            keep_view_prefix=True,
        )
        return response
    
    def _draw_waypoints_on_view(self, image: np.ndarray, waypoint_entry: Dict[str, Any]) -> np.ndarray:
        """
        在12视角图像上绘制单个 waypoint area 提示。

        这里只显示 area 名称和距离；不再显示 waypoint 编号。
        """
        if not waypoint_entry:
            return image

        label_text = str(
            waypoint_entry.get("label")
            or waypoint_entry.get("area_label")
            or waypoint_entry.get("description")
            or "Unknown"
        ).strip()
        if not label_text:
            return image

        try:
            distance_text = f"{float(waypoint_entry.get('distance_m', 0.0)):.1f}m"
        except (TypeError, ValueError):
            distance_text = "unknown"

        display_text = f"{label_text} {distance_text}".strip()
        h, w = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.62
        thickness = 2
        padding_x = 8
        padding_y = 6
        (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)

        box_w = text_w + padding_x * 2
        box_h = text_h + baseline + padding_y * 2
        box_x1 = max(8, (w - box_w) // 2)
        box_y1 = max(12, h // 2 - box_h - 18)
        box_x2 = min(w - 8, box_x1 + box_w)
        box_y2 = min(h - 8, box_y1 + box_h)

        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 2)
        text_x = box_x1 + padding_x
        text_y = box_y2 - baseline - padding_y
        cv2.putText(
            image,
            display_text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        return image
    
    
    def _draw_distance_rays_on_first_person_view(self, image: np.ndarray, distances: Dict[str, str]) -> np.ndarray:
        """
        在第一人称视图上绘制多条距离射线（复用当前depth采样的距离数据）
        
        Args:
            image: 第一人称RGB图像 (H, W, 3) BGR格式
            distances: 距离字典，如 {'front': '1.2m', 'left_30': '>2.0m', ...}
        """
        h, w = image.shape[:2]
        center_x, bottom_y = w // 2, h - 20
        hfov = float(self.config.MAP.HFOV)
        fov_half = hfov / 2.0

        # 方向映射：只显示 Left 30 / Front / Right 30
        ray_map = {
            'left_30': -30,
            'front': 0,
            'right_30': 30,
        }
        
        for key, angle in ray_map.items():
            if key not in distances or abs(angle) > fov_half:
                continue
            
            dist_str = distances[key]
            
            # 解析距离和颜色
            if "WARNING" in dist_str or "<0.5" in dist_str:
                color, y_ratio = (0, 0, 255), 0.7
            elif ">2.0" in dist_str or "open" in dist_str:
                color, y_ratio = (0, 255, 0), 0.1
            else:
                try:
                    dist_val = float(dist_str.replace('m', '').split()[0])
                    color = (0, 255, 255)
                    y_ratio = 0.7 if dist_val < 1.0 else (0.5 if dist_val < 2.0 else 0.3)
                except:
                    color, y_ratio = (0, 255, 255), 0.5
            
            # 计算终点
            x_ratio = (angle + fov_half) / (2 * fov_half)
            end_x, end_y = int(x_ratio * w), int(bottom_y - bottom_y * y_ratio)
            
            # 绘制射线和文字
            cv2.line(image, (center_x, bottom_y), (end_x, end_y), color, 2)
            text_x = end_x - len(dist_str) * 3
            text_y = end_y - 5
            cv2.rectangle(image, (text_x - 2, text_y - 12), (text_x + len(dist_str) * 7, text_y + 2), (0, 0, 0), -1)
            cv2.putText(image, dist_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return image
    
    def look_around_and_collect(self, phase: str = "initial") -> Tuple[List[str], List[str]]:
        """
        360°环视建图 + 生成4方向全景图
        
        执行12次×30°逆时针旋转（TURN_LEFT），每次转完后拍照并更新地图：
        - step 1: 第1次左转30°后拍照
        - step 2: 第2次左转60°后拍照
        - ...
        - step 12: 第12次左转360°后拍照（回到正前方）
        
        合成4个方向的90°视角全景图：
        - 前方：step-11(330°) + step-12(360°=0°) + step-1(30°) = 前方90°
        - 左侧：step-2(60°) + step-3(90°) + step-4(120°) = 左侧90°
        - 后方：step-5(150°) + step-6(180°) + step-7(210°) = 后方90°
        - 右侧：step-8(240°) + step-9(270°) + step-10(300°) = 右侧90°
        
        所有图像和地图统一保存到 vlm/observations/ 目录
        使用柱面投影拼接生成连贯的全景图
        环视过程不影响current_step和trajectory（环视后恢复）
        
        Args:
            phase: 阶段名称（用于文件命名，如 "initial", "verify_1"）
        
        Returns:
            (image_paths, direction_names) - 4个全景图路径和方向名称
        """
# print(f"\n[环视建图] {phase}...")
        
        # 注意：不清空landmark，让VLM能看到旧landmark来判断子任务是否完成
        # 轨迹和landmark的清空会在verify_and_replan中VLM输出后进行
        
        # 不在环视前更新距离（地图还未扫描，数据不准确）
        # 距离计算会在环视完成后进行
        
        # 存储12张环视图像用于合成全景图（step 1-12）
        lookaround_images = []
        lookaround_depths = []
        total_new_classes = 0
        
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        
        # 直接开始12次旋转，每一步保存rgb、detection、maps
        # 使用累加的self.current_step，避免覆盖之前的数据
        for i in range(1, 13):  # 12次旋转
            self.current_step += 1  # 累加总步数
            look_step = self.current_step
# print(f"  [{i}/12] 第{i}次左转")
            
            # 执行旋转
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            
            # 🔑 关键检查：如果episode已结束，立即停止环视并返回空列表
            if dones[0]:
                print(f"[WARN] Episode ended at lookaround step {i}/12")
                # 返回空列表，调用方需要处理这种情况
                return [], []
            
            # 环视阶段不做自定义landmark检测；子任务后续自动转向/动作前快照再检测
            prev_class_count = len(self.detected_classes)
            batch_obs = self._batch_obs(obs, save_object_detection=False)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            
            map_state = self.mapper.update_map(
                batch_obs, poses, look_step,
                list(self.detected_classes), self.current_episode_id
            )
            
            new_classes = len(self.detected_classes) - prev_class_count
            total_new_classes += new_classes
            
            # 调用visualizer保存所有数据（RGB、检测、全局地图、局部地图、semantic masks）
            # 地图可视化（保存地图+检测landmarks）
            # 环视过程中不传waypoint，不计算角度（环视结束后统一计算）
            rgb_bgr = cv2.cvtColor(obs[0]['rgb'], cv2.COLOR_RGB2BGR)
            vis_detections = None
            vis_labels = None
            vis_masks = None
            vis_landmark_classes = []
            
            paths, detected_landmarks_step, _ = self.visualizer.save_step_visualization(
                step=look_step,
                episode_id=self.current_episode_id,
                rgb=rgb_bgr,
                full_map=map_state['full_map'],
                trajectory_points=map_state.get('subtask_trajectory_points', []),  # 子任务轨迹（local map用）
                detected_classes=list(self.detected_classes),
                current_pose=map_state['full_pose'],
                floor=map_state['floor'],
                hfov=self.config.MAP.HFOV,
                detections=vis_detections,
                labels=vis_labels,
                masks=vis_masks,
                landmark_classes=vis_landmark_classes,
                mapping_classes=self.mapping_classes,
                landmark_config={
                    'min_total_pixels': self.landmark_min_total_pixels,
                    'min_area_threshold': self.landmark_min_area_threshold
                },
                waypoint_positions=map_state.get('waypoint_positions', []),  # 从map_state获取（已旋转）
                waypoint_ids=map_state.get('waypoint_ids', []),  # 从map_state获取
                room_area_layer=map_state.get('room_area_layer'),
                room_area_records=map_state.get('room_area_records', []),
                phase=phase,
                global_trajectory_points=map_state.get('global_trajectory_points', []),  # 全局轨迹（global map用）
                crop_offset=map_state.get('crop_offset'),  # 从map_state获取
                controller=self
            )
            
            # 保存导航可视化（RGB+俯视图拼接）
            if self.nav_visualizer:
                subtask_text = self.current_subtask.get('subtask_instruction', '') if self.current_subtask else f"[环视建图 {phase}]"
                distance = 0.0
                if infos and len(infos) > 0:
                    distance = infos[0].get('distance_to_goal', 0.0)
                
                # 环视阶段的subtask_id为phase（如initial, verify_1a）
                self.nav_visualizer.save_step_visualization(
                    observations=obs[0],
                    info=infos[0] if infos and len(infos) > 0 else {},
                    step=look_step,
                    instruction=self.current_instruction,
                    current_subtask=subtask_text,
                    distance=distance,
                    action=f"TURN_LEFT (360°环视 {i}/12)",
                    subtask_id=phase
                )
            
            # New classes detected (静默处理)
            pass
            
            # 保存所有12张环视图像（用于后续合成全景图）
            lookaround_images.append(rgb_bgr.copy())
            lookaround_depths.append(self._depth_to_meters(obs[0]['depth']))
        
        # 环视建图完成
        # 注意：不恢复轨迹，轨迹会自然显示在地图上
        # 如需清空轨迹，应在verify_and_replan中的子任务完成时调用mapper.clear_trajectory()
        
        # 缓存最后的观察（step 12，回到正前方）
        self.latest_obs = obs[0]
        
        # 扫描完成，更新距离（静默处理）
        self._update_obstacle_distances_12_directions(lookaround_depths)
        
        # 检查是否完成了完整的12步环视
        if len(lookaround_images) < 12:
            print(f"[WARN] Lookaround incomplete: {len(lookaround_images)}/12 images")
            # 返回空列表，调用方需要处理这种情况
            return [], []

        self.latest_lookaround_images = [img.copy() for img in lookaround_images]
        self.latest_lookaround_depths = [
            depth.copy() if depth is not None else None for depth in lookaround_depths
        ]
        self.latest_lookaround_phase = str(phase)
        
        # 环视结束后，计算waypoint角度（只计算一次，用于显示在12张view上）
        # 注意：initial时不显示waypoint（还没有历史），replan时显示上一个waypoint
        waypoint_info = None
        last_waypoint_angle_deg = None
        if phase != "initial" and hasattr(self, 'mapper') and self.mapper:
            # 获取当前地图状态（包含旋转后的waypoint坐标）
            map_state = self.mapper.get_map_state()
            wp_positions = map_state.get('waypoint_positions', [])
            wp_ids = map_state.get('waypoint_ids', [])
            
            if wp_positions:  # 如果有waypoint
                rgb_bgr = cv2.cvtColor(obs[0]['rgb'], cv2.COLOR_RGB2BGR)
                
                # 调用visualizer渲染地图并计算waypoint角度
                # Waypoint角度计算
                _, _, last_waypoint_angle = self.visualizer.save_step_visualization(
                    step=look_step,  # 使用最后一步的timestep
                    episode_id=self.current_episode_id,
                    rgb=rgb_bgr,
                    full_map=map_state['full_map'],
                    trajectory_points=map_state.get('subtask_trajectory_points', []),  # local map用子任务轨迹
                    detected_classes=list(self.detected_classes),
                    current_pose=map_state['full_pose'],
                    floor=map_state['floor'],
                    hfov=self.config.MAP.HFOV,
                    detections=None,
                    labels=None,
                    masks=None,
                    landmark_classes=self.landmark_classes,
                    mapping_classes=self.mapping_classes,
                    landmark_config={
                        'min_total_pixels': self.landmark_min_total_pixels,
                        'min_area_threshold': self.landmark_min_area_threshold
                    },
                    waypoint_positions=wp_positions,  # 旋转后的坐标
                    waypoint_ids=wp_ids,
                    room_area_layer=map_state.get('room_area_layer'),
                    room_area_records=map_state.get('room_area_records', []),
                    phase=phase,
                    global_trajectory_points=map_state.get('global_trajectory_points', []),  # global map用全局轨迹
                    crop_offset=map_state.get('crop_offset')  # 从map_state获取
                )
                
                # 转换为度数用于view映射
                if last_waypoint_angle is not None:
                    last_waypoint_angle_deg = np.degrees(last_waypoint_angle)
                    # print(f"  📍 Last Waypoint角度: {last_waypoint_angle_deg:.1f}°")
                
                # 保存waypoint信息用于绘制在view上
                # 注意：这里保存原始世界坐标（用于waypoint描述）
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
            waypoint_area_labels=map_state.get('waypoint_area_labels', []),
            current_pose=map_state.get('full_pose'),
            resolution_cm=float(getattr(self.mapper, 'resolution', 5)),
            current_room_area_label=str(map_state.get('current_room_area_label', 'Unknown') or 'Unknown'),
            full_map=map_state.get('full_map'),
            crop_offset=map_state.get('crop_offset'),
            waypoint_angle_deg=last_waypoint_angle_deg,
            draw_waypoints_fn=self._draw_waypoints_on_view,
        )
        
        # 保存 Global / Local Map 到对应目录
        # 使用当前step的地图（环视完成后的最新地图）
        self.latest_global_map = os.path.join(self.episode_dir, 'global_map', f'step_{self.current_step:04d}_{phase}.png')
        self.latest_local_map = os.path.join(self.episode_dir, 'local_map', f'step_{self.current_step:04d}_{phase}.png')
        
        if not os.path.exists(self.latest_global_map):
            print(f"  [WARN] Global Map not found: {self.latest_global_map}")
            self.latest_global_map = None
        
        if not os.path.exists(self.latest_local_map):
            print(f"  [WARN] Local Map not found: {self.latest_local_map}")
            self.latest_local_map = None

        # print(f"  12方向独立视图已保存")
        
        return direction_paths, direction_names
    
    def get_observations_and_maps(self, phase: str) -> Tuple[List[str], List[str], str, str]:
        """
        从directions/目录获取12方向独立视图和地图
        
        Args:
            phase: 阶段名称（如 "initial", "verify_1"）
            
        Returns:
            (direction_paths, direction_names, global_map_path, local_map_path)
        """
        from .vlm.navigation_config import DIRECTION_CONFIG
        
        direction_paths = []
        direction_names = []
        
        # 从episode的directions/目录读取12张独立图片
        directions_dir = os.path.join(self.episode_dir, 'directions')
        
        # 获取12个方向的图片
        for config in DIRECTION_CONFIG:
            angle = config["angle"]
            direction_name = config["name"]
            direction_filename = f"{phase}_direction_{angle:03d}.png"  # 如 initial_direction_000.png
            direction_path = os.path.join(directions_dir, direction_filename)
            
            if os.path.exists(direction_path):
                direction_paths.append(direction_path)
                direction_names.append(direction_name)
            else:
                print(f"  [WARN] {direction_name} not found: {direction_filename}")
        
        # 获取地图（使用当前step的地图，每次环视后current_step已更新）
        # current_step是最后一次环视后的step，地图文件名需要加上phase后缀
        global_map_path = os.path.join(self.episode_dir, 'global_map', f'step_{self.current_step:04d}_{phase}.png')
        local_map_path = os.path.join(self.episode_dir, 'local_map', f'step_{self.current_step:04d}_{phase}.png')
        
        if not os.path.exists(global_map_path):
            print(f"  [WARN] Global Map not found")
            global_map_path = None
        
        if not os.path.exists(local_map_path):
            print(f"  [WARN] Local Map not found")
            local_map_path = None
        
        return direction_paths, direction_names, global_map_path, local_map_path

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

        image_paths, direction_names = self.look_around_and_collect(phase)
        if not image_paths:
            self.latest_thinking_cycle_info = {
                "mode": mode_key,
                "phase": phase,
                "thinking_dir": thinking_dir,
                "reason": "lookaround_failed",
            }
            if mode_key == "initial":
                print("[ERR] Initial lookaround failed, cannot start planning")
            else:
                print("[ERR] Lookaround failed, cannot verify")
            return None, None, dict(self.latest_thinking_cycle_info)

        _, _, global_map, _local_map = self.get_observations_and_maps(phase)
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
                save_dir=thinking_dir,
            )

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

    def generate_initial_subtask(self) -> Optional[Dict]:
        """
        生成初始子任务
        
        使用环视收集的4方向全景图 + 全局地图 + 局部地图调用LLM生成子任务
        """
        response, _prompt, cycle_info = self._run_thinking_cycle(mode="initial")
        if not response or cycle_info is None:
            return None

        # 保存子任务并初始化计数
        self.current_subtask = response
        self.subtask_count = 1  # 初始化为第1个子任务
        self.subtask_attempt = 0  # 第a次尝试
        self.progress_summary = ""
        self.pose_before_action = None  # 重置pose追踪（新子任务从当前位置开始）
        
        # 记录当前位置信息（用于后续验证参考）
        self.current_position_info = {
            'waypoint': response.get('current_waypoint', 'Unknown'),
            'observation': response.get('current_observation', ''),
            'step': self.current_step
        }
        
        self._apply_postplanning_room_area_update(
            response=response,
            phase=str(cycle_info.get("phase", "initial")),
            thinking_dir=cycle_info.get("thinking_dir"),
            refresh_direction_views=False,
        )
        
        # 初始子任务开始前也显式清空旧自定义 landmark 状态
        self._reset_custom_landmark_state()

        # 动态更新目标landmark（直接使用VLM输出的next_waypoint_landmark）
        next_waypoint_landmark = response.get('next_waypoint_landmark', None)
        
        # 直接使用VLM输出，不自动提取；新子任务会覆盖旧landmark
        self._set_current_landmark_tracking(
            next_waypoint_landmark,
            fallback_sources=[
                response.get('next_waypoint_destination'),
                response.get('subtask_instruction'),
                response.get('current_waypoint'),
            ]
        )

        # 打印子任务信息
        self._print_subtask_info(response, is_initial=True)
        
        return response

    def _apply_postplanning_room_area_update(
        self,
        response: Dict[str, Any],
        phase: str,
        thinking_dir: Optional[str] = None,
        refresh_direction_views: bool = True,
    ) -> Optional[int]:
        """Persist planner room-area output into the world map and refresh debug renders."""
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
        if self.mapper is None or self.visualizer is None:
            return False

        rgb_bgr = None
        if self.latest_obs is not None and 'rgb' in self.latest_obs:
            rgb_bgr = cv2.cvtColor(self.latest_obs['rgb'], cv2.COLOR_RGB2BGR)
        elif (
            self.latest_lookaround_phase == phase and
            self.latest_lookaround_images and
            len(self.latest_lookaround_images) > 0
        ):
            rgb_bgr = self.latest_lookaround_images[-1].copy()

        if rgb_bgr is None:
            return False

        map_state = self.mapper.get_map_state()
        paths, _detected_landmarks_step, _last_waypoint_angle = self.visualizer.save_step_visualization(
            step=self.current_step,
            episode_id=self.current_episode_id,
            rgb=rgb_bgr,
            full_map=map_state['full_map'],
            trajectory_points=map_state.get('subtask_trajectory_points', []),
            detected_classes=list(self.detected_classes),
            current_pose=map_state['full_pose'],
            floor=map_state['floor'],
            hfov=self.config.MAP.HFOV,
            detections=None,
            labels=None,
            masks=None,
            landmark_classes=list(self.landmark_classes),
            mapping_classes=self.mapping_classes,
            landmark_config={
                'min_total_pixels': self.landmark_min_total_pixels,
                'min_area_threshold': self.landmark_min_area_threshold,
            },
            waypoint_positions=map_state.get('waypoint_positions', []),
            waypoint_ids=map_state.get('waypoint_ids', []),
            room_area_layer=map_state.get('room_area_layer'),
            room_area_records=map_state.get('room_area_records', []),
            phase=phase,
            global_trajectory_points=map_state.get('global_trajectory_points', []),
            crop_offset=map_state.get('crop_offset'),
            controller=self,
        )

        if paths.get('global_map'):
            self.latest_global_map = paths.get('global_map')
        if paths.get('local_map'):
            self.latest_local_map = paths.get('local_map')
        return True

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
            current_room_area_label=str(map_state.get('current_room_area_label', 'Unknown') or 'Unknown'),
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
        
        # 解析方向和角度
        # 支持格式: "IMAGE 5 (Left 120deg)" 或 "Left 120deg"
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
            direction = 'LEFT'  # 向左转180度
        elif 'Front' in waypoint_direction:
            # 已经面向Front，无需旋转
            # Waypoint already at Front, no rotation needed
            return True, []
        else:
            print(f"  [WARN] Unrecognized direction: {waypoint_direction}")
            return False, []
        
        # 生成动作序列（每次30度）
        num_turns = angle // 30
        action_sequence = []
        
        for i in range(num_turns):
            action_sequence.append({
                "action": f"TURN_{direction}",
                "degrees": 30
            })
        
        return True, action_sequence
    
    def execute_rotation_sequence(self, action_sequence: List[Dict]) -> bool:
        """
        执行旋转动作序列（使用统一的执行器，确保地图更新、步数记录、可视化保存）
        
        Args:
            action_sequence: 动作序列，格式 [{"action": "TURN_LEFT", "degrees": 30}, ...]
            
        Returns:
            是否全部执行成功
        """
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        
        for i, action_dict in enumerate(action_sequence):
            action_name = action_dict["action"]
            degrees = action_dict["degrees"]
            
            # 转换为habitat action ID
            if action_name == "TURN_LEFT":
                action_id = HabitatSimActions.TURN_LEFT
            elif action_name == "TURN_RIGHT":
                action_id = HabitatSimActions.TURN_RIGHT
            else:
                print(f"    [WARN] Unknown action: {action_name}")
                continue
            
            # 使用统一的执行器（step_with_vlm），确保：
            # - 更新地图
            # - 保存可视化（RGB、detection、maps）
            # - 更新距离信息
            # - 正确记录步数
            is_last_turn = (i == len(action_sequence) - 1)
            result = self.step_with_vlm(
                action_id,
                action_name,
                save_vis=True,
                enable_landmark_detection=is_last_turn,
            )
            
            # 检查episode是否结束
            if result.get('done', False):
                print(f"    [WARN] Episode ended during rotation")
                return False
        
        return True
    
    def verify_and_replan(self) -> Tuple[Optional[Dict], Optional[str]]:
        """
        验证当前子任务并重新规划
        
        流程：
        1. 执行360°环视建图（更新语义地图）- 占用12个step
        2. 生成当前位置的4方向全景图
        3. 调用LLM验证子任务完成状态
        4. 如未完成，生成新子任务
        
        注意：重新扫描会占用新的12个step，验证完成后下一个action继续累加
        
        Returns:
            (new_subtask_or_finish_response, prompt)
        """
        if not self.current_subtask:
            return None, None
        response, prompt, cycle_info = self._run_thinking_cycle(mode="verify")
        if not response or cycle_info is None:
            return None, None
        
        # 打印关键信息（精简）
        task_finished = response.get('global_task_finish', False)
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        print(f"  #{self.subtask_count}{attempt_letter} -> {response.get('next_waypoint_destination', 'N/A')} | finish={task_finished}")
        
        if task_finished:
            print("[DONE] Global task complete")
            return response, None
        else:
            print(f"  Next #{self.subtask_count + 1}a: {response.get('subtask_instruction', 'N/A')[:60]}")
            
            self._apply_postplanning_room_area_update(
                response=response,
                phase=str(cycle_info.get("phase", "")),
                thinking_dir=cycle_info.get("thinking_dir"),
                refresh_direction_views=True,
            )
            
            # 清空旧状态（为新子任务准备）
            self.mapper.clear_trajectory()
            self._reset_custom_landmark_state()
            self.progress_summary = ""
            self.previous_action_reason = ""
            self.pose_before_action = None
            self.last_planned_degrees = 0
            self.last_planned_meters = 0
            self.last_action_name = ""
            
            # 更新到新子任务：递增计数，重置尝试
            self.subtask_count += 1
            self.subtask_attempt = 0
            self.current_subtask = response
            
            # 更新当前位置信息（用于后续参考）
            self.current_position_info = {
                'waypoint': response.get('current_waypoint', 'Unknown'),
                'observation': response.get('current_observation', ''),
                'step': self.current_step
            }
            
            # 动态更新目标landmark（直接使用VLM输出的next_waypoint_landmark）
            next_waypoint_landmark = response.get('next_waypoint_landmark', None)
            
            # 直接使用VLM输出，不自动提取；新子任务会覆盖旧landmark
            self._set_current_landmark_tracking(
                next_waypoint_landmark,
                fallback_sources=[
                    response.get('next_waypoint_destination'),
                    response.get('subtask_instruction'),
                    response.get('current_waypoint'),
                ]
            )
            
            # ⚠️ 重要：self.classes更新已在上方完成
            
            self._print_subtask_info(response)
            
            # 子任务完成后，自动旋转到新的waypoint方向
            next_waypoint_direction = response.get('next_waypoint_direction', '')
            if next_waypoint_direction and 'Front' not in next_waypoint_direction:
                success, action_sequence = self.auto_rotate_to_waypoint(next_waypoint_direction)
                
                if success and action_sequence:
                    self.execute_rotation_sequence(action_sequence)
                    print()  # newline after rotation steps

        # 返回response（prompt已保存到save_dir）
            return response, prompt
    
    def execute_action_with_vlm(self) -> Tuple[Optional[int], Optional[str], bool, int, Optional[Dict]]:
        """
        使用VLM决策并执行动作
        
        Returns:
            (action_id, action_name, should_stop, repeat_count, response)
        """
        if not self.action_executor or not self.current_subtask:
            return None, None, True
        
        # 获取当前观察：使用缓存的观察或通过旋转获取
        if self.latest_obs is not None:
            obs = self.latest_obs
        else:
            # 如果没有缓存，执行一次右转再左转回来获取观察
            actions = [{"action": HabitatSimActions.TURN_RIGHT}]
            outputs = self.envs.step(actions)
            obs, _, dones, _ = [list(x) for x in zip(*outputs)]
            if dones[0]:
                print("[WARN] Episode ended")
                return None, None, True
            
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            outputs = self.envs.step(actions)
            obs, _, dones, _ = [list(x) for x in zip(*outputs)]
            if dones[0]:
                print("[WARN] Episode ended")
                return None, None, True
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
            "next_waypoint_destination": self.current_subtask.get('next_waypoint_destination', ''),
            "subtask_instruction": self.current_subtask.get('subtask_instruction', ''),
            "start_step": self.current_step,
            "timestamp": datetime.now().isoformat()
        }
        
        # 计算save_dir: API发送时同步保存压缩图片+prompt
        subtask_dir = os.path.join(self.save_manager.episode_dir, "action", f"subtask_{subtask_id}")
        action_save_dir = os.path.join(subtask_dir, f"step_{self.current_step + 1}")
        os.makedirs(action_save_dir, exist_ok=True)

        waypoint_summary = self._get_waypoint_summary(include_area_chain=False, include_path=False)
        
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
            self.current_subtask.get('next_waypoint_destination', ''),
            self.current_subtask.get('next_waypoint_direction', ''),
            keep_view_prefix=False,
        )

        # 调用VLM决策（save_dir使call_api在发送时保存压缩图片+prompt）
        result = self.action_executor.decide_action(
            next_waypoint_destination=self.current_subtask.get('next_waypoint_destination', ''),
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

    def _refresh_step_visualization_snapshot(
        self,
        phase: str,
        enable_landmark_detection: bool = False,
        force: bool = False,
    ) -> bool:
        """用当前 pose/obs 重新渲染当前 step 的可视化，不重复做地图融合。"""
        if self.latest_obs is None:
            return False

        required_paths = [
            os.path.join(self.episode_dir, 'rgb', f'step_{self.current_step:04d}_{phase}.png'),
            os.path.join(self.episode_dir, 'global_map', f'step_{self.current_step:04d}_{phase}.png'),
            os.path.join(self.episode_dir, 'local_map', f'step_{self.current_step:04d}_{phase}.png'),
        ]
        if enable_landmark_detection:
            required_paths.append(
                os.path.join(self.episode_dir, 'detection', f'step_{self.current_step:04d}_{phase}.png')
            )

        if not force and all(os.path.exists(path) for path in required_paths):
            if not enable_landmark_detection:
                return True
            if hasattr(self, 'current_step_landmarks') and self.current_step in self.current_step_landmarks:
                return True

        # 当前观测已经在真实环境 step / 环视阶段写入地图，这里只刷新当前帧检测与渲染，
        # 不能再把同一帧 pose 二次融合进语义地图。
        self._batch_obs([self.latest_obs], save_object_detection=enable_landmark_detection)
        map_state = self.mapper.get_map_state()

        rgb_bgr = cv2.cvtColor(self.latest_obs['rgb'], cv2.COLOR_RGB2BGR)
        landmark_classes = list(self.landmark_classes) if enable_landmark_detection else []
        detections = self.latest_detections_full if enable_landmark_detection and hasattr(self, 'latest_detections_full') else None
        labels = self.latest_labels_full if enable_landmark_detection and hasattr(self, 'latest_labels_full') else None
        masks = self.latest_masks_full if enable_landmark_detection and hasattr(self, 'latest_masks_full') else None

        paths, detected_landmarks_step, _ = self.visualizer.save_step_visualization(
            step=self.current_step,
            episode_id=self.current_episode_id,
            rgb=rgb_bgr,
            full_map=map_state['full_map'],
            trajectory_points=map_state.get('subtask_trajectory_points', []),
            detected_classes=list(self.detected_classes),
            current_pose=map_state['full_pose'],
            floor=map_state['floor'],
            hfov=self.config.MAP.HFOV,
            detections=detections,
            labels=labels,
            masks=masks,
            landmark_classes=landmark_classes,
            mapping_classes=self.mapping_classes,
            landmark_config={
                'min_total_pixels': self.landmark_min_total_pixels,
                'min_area_threshold': self.landmark_min_area_threshold
            },
            waypoint_positions=map_state.get('waypoint_positions', []),
            waypoint_ids=map_state.get('waypoint_ids', []),
            room_area_layer=map_state.get('room_area_layer'),
            room_area_records=map_state.get('room_area_records', []),
            phase=phase,
            global_trajectory_points=map_state.get('global_trajectory_points', []),
            crop_offset=map_state.get('crop_offset'),
            controller=self,
        )

        if paths.get('global_map'):
            self.latest_global_map = paths.get('global_map')
        if paths.get('local_map'):
            self.latest_local_map = paths.get('local_map')

        if enable_landmark_detection:
            self._record_landmark_detection_step(self.current_step, detected_landmarks_step)
        return True

    def _run_pre_action_detection_snapshot(self, action_phase: str) -> bool:
        """在不移动agent的情况下，执行一次动作前landmark检测并保存可视化。"""
        return self._refresh_step_visualization_snapshot(
            phase=action_phase,
            enable_landmark_detection=True,
            force=False,
        )
    
    def _update_obstacle_distances_12_directions(self, lookaround_depths: Optional[List[np.ndarray]] = None):
        """更新12个环视方向的障碍物距离（深度角度带采样，失败时回退到地图障碍物）。"""
        try:
            depth_views = lookaround_depths or []
            if len(depth_views) < 12:
                raise ValueError("Lookaround depths incomplete")

            map_fallback = {}
            if hasattr(self, 'mapper') and self.mapper is not None and self.visualizer is not None:
                map_state = self.mapper.get_map_state()
                map_fallback = self.visualizer.calculate_obstacle_distances_12_directions_from_full_map(
                    map_state.get('full_map'),
                )

            distances = {}
            for config in DIRECTION_CONFIG:
                depth_meters = depth_views[config["step"] - 1]
                front_distance = calculate_obstacle_distances_from_depth(
                    depth_meters,
                    hfov_deg=float(self.config.MAP.HFOV),
                    directions={"front": 0.0},
                    angle_band_deg=5.0,
                    fallback_distances={
                        "front": map_fallback.get(f'angle_{config["angle"]}', ">2.0m open")
                    },
                ).get("front", "Unknown")
                distances[f'angle_{config["angle"]}'] = front_distance
            self.latest_obstacle_distances_12 = distances
        except Exception:
            try:
                map_state = self.mapper.get_map_state() if hasattr(self, 'mapper') and self.mapper is not None else {}
                fallback = self.visualizer.calculate_obstacle_distances_12_directions_from_full_map(
                    map_state.get('full_map'),
                ) if self.visualizer is not None else {}
            except Exception:
                fallback = {}
            self.latest_obstacle_distances_12 = {
                f'angle_{i}': fallback.get(f'angle_{i}', '>2.0m open') for i in range(0, 360, 30)
            }

    def _update_obstacle_distances(self):
        """更新当前位置的障碍物距离（深度角度带采样，失败时回退到地图障碍物）。"""
        try:
            map_fallback = {}
            if hasattr(self, 'mapper') and self.mapper is not None and self.visualizer is not None:
                map_state = self.mapper.get_map_state()
                map_fallback = self.visualizer.calculate_obstacle_distances_from_full_map(
                    map_state.get('full_map'),
                )
            self.latest_obstacle_distances = calculate_obstacle_distances_from_depth(
                getattr(self, 'latest_depth_meters', None),
                hfov_deg=float(self.config.MAP.HFOV),
                angle_band_deg=5.0,
                fallback_distances=map_fallback,
            )
        except Exception:
            try:
                map_state = self.mapper.get_map_state() if hasattr(self, 'mapper') and self.mapper is not None else {}
                fallback = self.visualizer.calculate_obstacle_distances_from_full_map(
                    map_state.get('full_map'),
                ) if self.visualizer is not None else {}
            except Exception:
                fallback = {}
            self.latest_obstacle_distances = {
                'front': fallback.get('front', '>2.0m open'),
                'left_30': fallback.get('left_30', '>2.0m open'),
                'right_30': fallback.get('right_30', '>2.0m open'),
            }
    
    def run_vlm_navigation(self, max_subtask_steps: int = 5) -> Dict[str, Any]:
        """
        运行完整的VLM导航流程
        
        Args:
            max_subtask_steps: 每个子任务最大步数（达到后强制触发验证，默认5步）
            
        Returns:
            导航结果字典
        """
        # 从 Habitat 配置读取最大步数限制
        max_steps = self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        
        print(f"\n{'='*60}")
        print(f"VLM Navigation | max_steps={max_steps} | subtask_steps={max_subtask_steps}")
        print(f"Instruction: {self.current_instruction}")
        print(f"{'='*60}")
        
        # 1. 统一的 initial thinking cycle（lookaround + planning + post-planning update）
        subtask = self.generate_initial_subtask()
        if not subtask:
            cycle_reason = str(getattr(self, 'latest_thinking_cycle_info', {}).get('reason', '') or '')
            failure_reason = 'initial_lookaround_failed' if cycle_reason == 'lookaround_failed' else 'initial_subtask_failed'
            print("[ERR] Initial subtask generation failed")
            return {
                'success': False,
                'total_steps': self.current_step,
                'subtask_count': 0,
                'detected_classes': list(self.detected_classes) if hasattr(self, 'detected_classes') else [],
                'gif_path': None,
                'result_file': None,
                'reason': failure_reason,
            }
        
        # 2. 自动旋转到waypoint方向
        next_waypoint_direction = subtask.get('next_waypoint_direction', '')
        if next_waypoint_direction and 'Front' not in next_waypoint_direction:
            success, action_sequence = self.auto_rotate_to_waypoint(next_waypoint_direction)
            
            if success and action_sequence:
                self.execute_rotation_sequence(action_sequence)
                print()  # newline after rotation steps

        # 3. 主导航循环
        total_steps = self.current_step
        subtask_steps = 0
        navigation_complete = False
        
        while True:
            # 🔑 检查退出条件（执行action之前）
            # 如果任务已完成（VLM判断或Habitat设置done），直接退出
            if navigation_complete:
                break

            if self._all_action_directions_blocked(threshold_m=0.5):
                retreated_m, retreat_done = self._execute_auto_retreat(retreat_distance_m=1.0)
                total_steps = self.current_step

                if retreat_done:
                    print("[WARN] Episode done during automatic retreat")
                    navigation_complete = True
                    break

                print(
                    f"[AutoRetreat] Finished turn-around move ({retreated_m:.2f}m). "
                    f"Skip action VLM and start 12-view thinking/replan."
                )
                navigation_complete, _new_subtask = self._trigger_verify_replan(
                    "auto retreat replan",
                    total_steps,
                )
                if navigation_complete:
                    break

                subtask_steps = 0
                continue

            # VLM决策动作（失败则重试）
            max_retries = 3
            action_id = None
            vlm_response = None
            
            for retry in range(max_retries):
                action_id, action_name, should_stop, repeat_count, vlm_response = self.execute_action_with_vlm()
                
                if action_id is not None:
                    break
                
                if retry < max_retries - 1:
                    wait = (retry + 1) * 2
                    print(f"  [WARN] VLM Action failed, retry in {wait}s ({retry + 1}/{max_retries - 1})...")
                    import time
                    time.sleep(wait)
            
            # 所有重试都失败，跳过此步
            if action_id is None:
                print("[ERR] VLM Action failed after all retries, skipping step")
                continue
            
            # 关键检查：在执行任何action之前，检查VLM响应中的global_task_finish
            if vlm_response and vlm_response.get('global_task_finish', False):
                print(f"[DONE] Task complete (action) | steps={total_steps}")
                navigation_complete = True
                break
            
            # 如果VLM决定停止 → 验证子任务
            if should_stop:
                print("\n[STOP] -> Verify...")
                navigation_complete, _new_subtask = self._trigger_verify_replan(
                    "verify",
                    total_steps,
                )
                if navigation_complete:
                    break
                
                # 子任务完成或重新规划，重置步数计数
                subtask_steps = 0
                continue
            
            # VLM决策计数（每次调用action模型算1步）
            subtask_steps += 1
            
            # 🔑 关键修复：在执行action后检查步数限制
            # 如果达到最大步数（例如5步），执行完当前动作后立即强制replan
            if subtask_steps >= max_subtask_steps:
                # 继续执行当前动作，但标记下一轮要replan
                force_replan_after_action = True
            else:
                force_replan_after_action = False
            
            # 执行动作前记录pose（用于后续计算实际变化）
            if self.pose_before_action is None:
                self.pose_before_action = self._get_agent_pose()
            pose_before_action_batch = self._get_agent_pose()
            auto_completed_subtask = None
            
            # 执行动作（可能需要重复多次）
            for i in range(repeat_count):
                result = self.step_with_vlm(action_id, action_name=action_name, save_vis=True)
                total_steps = self.current_step
                
                if repeat_count > 1:
                    print(f"  [Step {total_steps}] {action_name} ({i+1}/{repeat_count})")
                else:
                    print(f"  [Step {total_steps}] {action_name} | subtask {subtask_steps}/{max_subtask_steps}")
                
                # 🔍 记录DTG轨迹（每步记录）
                if self.latest_info:
                    dtg = self.latest_info.get('distance_to_goal', -1)
                    if not hasattr(self, 'dtg_history'):
                        self.dtg_history = []
                    self.dtg_history.append(dtg)
                
                # 🔑 检查episode是否自动结束（Habitat内部判断，如达到MAX_EPISODE_STEPS）
                if result['done']:
                    print(f"[WARN] Episode done (Habitat)")
                    # 不要尝试调用step(STOP)，因为episode已经done，会触发AssertionError
                    # latest_info已在step_with_vlm中更新，包含最终指标
                    navigation_complete = True
                    break

                current_step_landmark_entries = self._get_current_action_step_landmark_entries()
                auto_completed_subtask = self._should_autocomplete_subtask_during_action_step(
                    current_step_landmark_entries
                )
                if auto_completed_subtask is not None:
                    landmark_kind = "opening-like" if auto_completed_subtask.get("is_opening_like") else "solid"
                    print(
                        "[AutoSubtaskComplete] "
                        f"{auto_completed_subtask['name']} ({landmark_kind}) reached within "
                        f"{auto_completed_subtask['distance_m']:.2f}m "
                        f"(threshold {float(auto_completed_subtask.get('stop_distance_m', 1.0)):.2f}m) "
                        f"on action step {self.current_step}; "
                        "skip remaining action repeats and start thinking/replan."
                    )
                    break
            
            # 所有重复执行完成后，计算总的progress（一次性）
            if hasattr(self, 'last_action_name') and self.last_action_name and not navigation_complete:
                pose_after_action_batch = self._get_agent_pose()
                
                # 计算实际位姿变化
                x_before, y_before, ori_before = pose_before_action_batch
                x_after, y_after, ori_after = pose_after_action_batch
                
                # 计算实际转向角度变化（保留符号）
                import math
                angle_diff = ori_after - ori_before
                # 归一化到 [-pi, pi]
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                # 判断实际转向方向（正=左转，负=右转）
                actual_degrees = abs(math.degrees(angle_diff))
                
                # 根据实际方向校正action_name（处理转过头的情况）
                actual_action_name = self.last_action_name
                if self.last_action_name == 'TURN_LEFT' and angle_diff < -0.1:  # 计划左转但实际右转
                    actual_action_name = 'TURN_RIGHT'
                    print(f"[Warning] Planned TURN_LEFT but actually turned RIGHT by {actual_degrees:.1f}°")
                elif self.last_action_name == 'TURN_RIGHT' and angle_diff > 0.1:  # 计划右转但实际左转
                    actual_action_name = 'TURN_LEFT'
                    print(f"[Warning] Planned TURN_RIGHT but actually turned LEFT by {actual_degrees:.1f}°")
                
                # 计算实际移动距离（2D欧氏距离）
                actual_meters = math.sqrt((x_after - x_before)**2 + (y_after - y_before)**2)
                
                # 调用_generate_progress_update更新progress
                self.progress_summary = self.action_executor._generate_progress_update(
                    current_progress=self.progress_summary,
                    action_name=actual_action_name,  # 使用校正后的方向
                    degrees=self.last_planned_degrees,
                    meters=self.last_planned_meters,
                    actual_degrees=actual_degrees,
                    actual_meters=actual_meters
                )
                
                # Progress tracked internally
                
                # 更新pose_before为当前pose（供下次计算使用）
                self.pose_before_action = pose_after_action_batch

            if auto_completed_subtask is not None and not navigation_complete:
                landmark_kind = "opening-like" if auto_completed_subtask.get("is_opening_like") else "solid"
                self._append_progress_note(
                    f"had reached {auto_completed_subtask['name']} ({landmark_kind}) within "
                    f"{auto_completed_subtask['distance_m']:.2f}m "
                    f"(auto-stop threshold {float(auto_completed_subtask.get('stop_distance_m', 1.0)):.2f}m), "
                    "so ended the current subtask and triggered replan"
                )
                self.previous_action_reason = (
                    f"Displayed destination landmark {auto_completed_subtask['name']} ({landmark_kind}) was within "
                    f"{auto_completed_subtask['distance_m']:.2f}m "
                    f"(threshold {float(auto_completed_subtask.get('stop_distance_m', 1.0)):.2f}m), "
                    "so the system ended the current subtask and started thinking"
                )
                navigation_complete, _new_subtask = self._trigger_verify_replan(
                    "action step autocomplete",
                    total_steps,
                )
                if navigation_complete:
                    break
                subtask_steps = 0
                continue
            
            # 🔑 强制重规划检查：如果达到最大步数，执行完动作后立即触发verify
            if force_replan_after_action:
                print(f"\n[Replan] Force replan after {max_subtask_steps} steps")
                navigation_complete, _new_subtask = self._trigger_verify_replan(
                    "force replan",
                    total_steps,
                )
                if navigation_complete:
                    break
                subtask_steps = 0  # 重置步数
                continue
            
            if navigation_complete:
                break
        
        # 主循环结束 - 记录退出原因和DTG轨迹统计
        # DTG统计
        if hasattr(self, 'dtg_history') and self.dtg_history:
            valid_dtgs = [d for d in self.dtg_history if d >= 0]
            if valid_dtgs:
                print(f"\nDTG: min={min(valid_dtgs):.2f}m final={valid_dtgs[-1]:.2f}m")
        
        # 4. 生成GIF动画
        
        gif_path = None
        if self.nav_visualizer:
            gif_path = self.nav_visualizer.save_gif(fps=2)
        
        # 5. 调用finish_episode()执行STOP并获取最终指标
        final_metrics = self.finish_episode(
            success=navigation_complete, 
            stop_action=True  # 总是调用STOP以获得正确的Success判定
        )
        
        # 使用STOP后的最终指标
        env_metrics = final_metrics if final_metrics else {}
        if not env_metrics:
            try:
                if hasattr(self.envs, 'call_at'):
                    env_metrics = self.envs.call_at(0, "get_metrics")
            except Exception as e:
                env_metrics = {}
        
        final_result = self._save_navigation_result(navigation_complete, total_steps, env_metrics)
        
        print(f"\n{'='*60}")
        print(f"{'OK' if navigation_complete else 'FAIL'} | steps={total_steps} | subtasks={self.subtask_count}")
        print(f"{'='*60}")
        
        return {
            'success': navigation_complete,
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
        - distance_to_goal: 停止时智能体与目标点的距离(米)，越小越好
        - success: 成功率，智能体是否在3米内停止(0或1)
        - spl: Success weighted by Path Length，成功率与路径效率的综合指标
               公式: success * (最短路径长度 / 实际路径长度)
               范围[0,1]，越高表示既成功又高效
        - path_length: 智能体实际行走的路径长度(米)
        - oracle_success: 预言成功率，整个轨迹中是否曾经到达过目标3米内(0或1)
                         用于评估智能体是否找到过目标但错过了停止
        - oracle_navigation_error: 轨迹中与目标点的最小距离
        - oracle_spl: 基于oracle_success的spl指标
        
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
        
        # 打印关键指标（便于实时监控）
        print(f"\nEpisode {self.current_episode_id}: succ={result['success']} spl={result['spl']:.4f} dtg={result['distance_to_goal']:.3f}m pl={result['path_length']:.3f}m oracle={result['oracle_success']}")
        
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
        
        dest = response.get('next_waypoint_destination', 'N/A')
        instr = response.get('subtask_instruction', 'N/A')[:80]
        print(f"  {title}: {dest} | {instr}")
    
    # ========== Waypoint辅助方法 ==========

    def _get_waypoint_summary(self, include_area_chain: bool = True, include_path: bool = True) -> str:
        """
        获取waypoint摘要（用于LLM提示词）
        包含每个waypoint相对当前pose的距离和方向，以及顺序拓扑路径。
        """
        wp_pos, wp_ids, wp_descs = self.mapper.get_waypoints()
        return build_waypoint_summary(
            waypoint_positions=wp_pos,
            waypoint_ids=wp_ids,
            waypoint_descriptions=wp_descs,
            waypoint_area_labels=self.mapper.get_waypoint_area_labels(),
            current_pose=self.mapper.full_pose,
            resolution_cm=self.mapper.resolution,
            current_room_area_label=getattr(self.mapper, 'current_room_area_label', ""),
            current_room_area_type=getattr(self.mapper, 'current_room_area_type', ""),
            full_map=getattr(self.mapper, 'full_map', None),
            crop_offset=getattr(getattr(self.mapper, 'mapping_module', None), 'full_map_crop_offset', None),
            include_area_chain=include_area_chain,
            include_path=include_path,
        )

    # ========== 原有方法 ==========
