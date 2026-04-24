"""
VLM Navigation Controller
=========================
Task-specific VLN-CE controller built on top of the shared navigation stack.

Inherited base capabilities:
- semantic mapping
- visualization
- 12-step lookaround map updates

Additional VLM capabilities:
- high-level LLM planning
- low-level VLM action execution from RGB + map context
- four-direction observation collection
- RGB + top-down visualization rendering
- episode artifact persistence for evaluation
"""
import os
import re
import cv2
import json
import time
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Sequence
from datetime import datetime

try:
    from habitat import Config
except ImportError:  # Habitat 0.2.x no longer exports Config
    from typing import Any as Config

from navigation_system.space.description.direction_format import format_relative_direction
from navigation_system.space.description.spatial_formatter import (
    build_action_landmark_map_info,
    build_waypoint_summary,
)
from navigation_system.space.geometry.connectivity import (
    build_bounded_geodesic_distance_field,
    query_world_distance_from_field_m,
)
from navigation_system.space.geometry.map_projection import RotatedMapProjector
from navigation_system.controller.action_compat import resolve_habitat_action
from navigation_system.controller.base_controller import BaseNavigationController
from navigation_system.controller.state import (
    EpisodeTimingTracker,
    VLMControllerOptions,
)
from navigation_system.space.structure.space_types import (
    strip_space_type_variant_suffixes,
)
from navigation_system.vlm.interfaces import (
    NavigationModelStack,
    NavigationModelStackBuilder,
)
from navigation_system.vlm.vlnce.runtime_factory import (
    build_default_navigation_model_stack,
)
from navigation_system.runtime.storage.artifacts import (
    SaveManager,
    get_episode_detail_dir,
    get_episode_detail_path_candidates,
)
from navigation_system.render.episode_visualization.navigation_visualizer import NavigationVisualizer
from navigation_system.render.image_resize import resize_image_to_width
from navigation_system.render.views.thinking_view_renderer import ThinkingViewRenderer
from navigation_system.vlm.contracts.schema import (
    ACTION_MAPPING,
    get_next_waypoint,
    get_subtask_landmark,
    normalize_subtask_payload,
)
from navigation_system.config.core.params.api import ACTION_VIEW_MODEL_CONTENT_WIDTH
from navigation_system.config.core.constants import landmark_edge_depth_keywords
from navigation_system.config.core.params.actions import (
    ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M as CFG_AUTOCOMPLETE_OPEN_DISTANCE_M,
    ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M as CFG_AUTOCOMPLETE_SOLID_DISTANCE_M,
    ACTION_SUBTASK_AUTOCOMPLETE_TOPK as CFG_AUTOCOMPLETE_TOPK,
)
from navigation_system.config.core.params.landmarks import LANDMARK_STRIP_TOPK
from navigation_system.config.core.params.thresholds import (
    EVAL_SUCCESS_DISTANCE_M,
    LOW_LEVEL_STAGNATION_CAP_M as CFG_LOW_LEVEL_STAGNATION_CAP_M,
    LOW_LEVEL_STAGNATION_RATIO as CFG_LOW_LEVEL_STAGNATION_RATIO,
    OBS_BLOCKED_M,
)
from navigation_system.config.core.params.spatial import (
    SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M,
)


class VLMNavigationController(BaseNavigationController):
    """
    VLN-CE controller with VLM planning and execution.

    Main loop:
    1. Initial 12-step lookaround mapping and four-direction snapshot collection
    2. LLM planning for the next subtask
    3. VLM action execution until the subtask is finished
    4. Verification lookaround to refresh the map and local observations
    5. Replanning / completion check
    6. Repeat until the episode ends

    A 360-degree lookaround always runs before verify-time replanning so the
    planner sees updated semantics and current local context.
    """

    ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M = CFG_AUTOCOMPLETE_OPEN_DISTANCE_M
    ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M = CFG_AUTOCOMPLETE_SOLID_DISTANCE_M
    ACTION_SUBTASK_AUTOCOMPLETE_TOPK = CFG_AUTOCOMPLETE_TOPK
    THINKING_LOOKAROUND_STEPS = 12
    FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK = 3
    FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M = 1.0
    ACTION_CONSECUTIVE_TURN_LIMIT = 3
    LOW_LEVEL_STAGNATION_RATIO = CFG_LOW_LEVEL_STAGNATION_RATIO
    LOW_LEVEL_STAGNATION_CAP_M = CFG_LOW_LEVEL_STAGNATION_CAP_M
    STUCK_RETREAT_DISTANCE_M = 1.0
    STUCK_RETREAT_FORBIDDEN_VIEW_IDS = (7,)
    
    def __init__(
        self,
        config: Config,
        config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml",
        model_stack_builder: Optional[NavigationModelStackBuilder] = None,
        envs=None,
    ):
        """
        Initialize the VLM navigation controller.

        Args:
            config: Habitat config.
            config_path: Unified API config path for both LLM and VLM models.
        """
        super().__init__(config, envs=envs)

        self.turn_angle = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE
        self.move_distance = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
        self.runtime_options = VLMControllerOptions.from_config(
            config,
            default_final_destination_match_autostop_streak=self.FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK,
            default_final_destination_match_autostop_radius_m=self.FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M,
            default_low_level_stagnation_ratio=self.LOW_LEVEL_STAGNATION_RATIO,
            default_low_level_stagnation_cap_m=self.LOW_LEVEL_STAGNATION_CAP_M,
        )
        self.timing_tracker = EpisodeTimingTracker()
        
        self.action_space = (
            f"MOVE_FORWARD ({self.move_distance}m), "
            f"TURN_LEFT ({self.turn_angle}°), TURN_RIGHT ({self.turn_angle}°), STOP"
        )
        self.latest_action_local_map_debug_lines = []

        self.model_stack_builder = (
            model_stack_builder or build_default_navigation_model_stack
        )
        try:
            model_stack = self.model_stack_builder(
                config_path=config_path,
                action_space=self.action_space,
                turn_angle=float(self.turn_angle),
                move_distance=float(self.move_distance),
                save_request_artifacts=self.runtime_options.save_api_request_artifacts,
            )
        except Exception as exc:
            print(f"[WARN] Navigation model stack init failed: {exc}")
            model_stack = NavigationModelStack(planner=None, action_executor=None)
        self.planner = model_stack.planner
        self.action_executor = model_stack.action_executor
        
        self.thinking_view_renderer = ThinkingViewRenderer()
        
        self.save_manager = None
        self.nav_visualizer = None
        self._reset_vlm_episode_state()

    def _reset_vlm_episode_state(self) -> None:
        self.current_subtask = None
        self.subtask_count = 0
        self.subtask_attempt = 0
        self.progress_summary = ""
        self.subtask_history = []
        self.latest_thinking_cycle_info = {}
        self.tracked_landmark_classes = set()
        self.final_goal_destination_match_streak = 0
        self.final_goal_destination_match_anchor_xy = None
        self.action_stagnation_streak = 0
        self.blocked_front_controller_recovery_count = 0
        self.action_stagnation_retry_pending = False
        self.action_stagnation_retry_notice_text = ""
        self.action_stagnation_progress_warning_text = ""
        self.action_consecutive_turn_count = 0
        self.action_force_forward_after_turns_pending = False
        self.action_force_forward_after_turns_notice_text = ""
        self.verify_replan_prompt_notice = ""
        self.pending_verify_view_restriction = None
        self.previous_subtask_landmark_final_info = None
        self.previous_subtask_autocomplete_landmark_info = None
        self.latest_action_local_map_debug_lines = []
        self._latest_action_landmark_entries_cache_key = None
        self._latest_action_landmark_entries_cache_value = []
        self._previous_subtask_landmark_summary_cache_key = None
        self._previous_subtask_landmark_summary_cache_summary = ""
        self._previous_subtask_landmark_summary_cache_info = {}
        self.pose_before_action = None
        self.last_planned_degrees = 0
        self.last_planned_meters = 0
        self.last_action_name = ""
        self.last_action_effective_name = ""
        self.last_action_semantic_variant = ""
        self.last_action_progress_hint = ""
        self.timing_tracker.reset()

    def _invalidate_action_landmark_summary_cache(self) -> None:
        self._latest_action_landmark_entries_cache_key = None
        self._latest_action_landmark_entries_cache_value = []
        self._previous_subtask_landmark_summary_cache_key = None
        self._previous_subtask_landmark_summary_cache_summary = ""
        self._previous_subtask_landmark_summary_cache_info = {}

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

    def _is_obstacle_distance_blocked(self, distance_text: Optional[str], threshold_m: float = OBS_BLOCKED_M) -> bool:
        distance_m = self._parse_distance_text_m(distance_text)
        return distance_m is not None and distance_m < float(threshold_m)

    def _append_progress_note(self, note: str) -> None:
        note = (note or "").strip()
        if not note:
            return
        if not self.progress_summary or self.progress_summary == "(Just started - no actions yet)":
            self.progress_summary = note
        else:
            self.progress_summary = f"{self.progress_summary}, {note}"

    @staticmethod
    def _merge_prompt_notices(*notice_texts: Optional[str]) -> str:
        notices: List[str] = []
        for raw in notice_texts:
            text = str(raw or "").strip()
            if text and text not in notices:
                notices.append(text)
        return "\n".join(notices)

    @staticmethod
    def _merge_planner_timing_infos(*timing_infos: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged_records: List[Dict[str, Any]] = []
        merged_wait_duration_s = 0.0
        for info in timing_infos:
            if not info:
                continue
            merged_records.extend(list(info.get("records", []) or []))
            merged_wait_duration_s += float(info.get("failed_retry_wait_duration_s", 0.0) or 0.0)
        return {
            "records": merged_records,
            "failed_retry_wait_duration_s": merged_wait_duration_s,
        }

    def _is_in_initial_position_neighborhood(self, waypoint_summary: Optional[str] = None) -> bool:
        summary_text = str(waypoint_summary or "")
        if "near INITIAL POSITION Space WP#" in summary_text:
            return True

        mapper = getattr(self, "mapper", None)
        if mapper is None:
            return False

        current_pose = getattr(mapper, "full_pose", None)
        resolution_cm = float(getattr(mapper, "resolution", 0.0) or 0.0)
        global_waypoint_manager = getattr(mapper, "global_waypoint_manager", None)
        initial_waypoint_index = getattr(global_waypoint_manager, "initial_waypoint_index", None)
        if (
            current_pose is None
            or len(current_pose) < 3
            or resolution_cm <= 0.0
            or initial_waypoint_index is None
        ):
            return False

        waypoint_positions, _waypoint_ids, _waypoint_descs = mapper.get_global_waypoints()
        waypoint_floor_ids = mapper.get_global_waypoint_floor_ids()
        try:
            initial_idx = int(initial_waypoint_index)
        except (TypeError, ValueError):
            return False
        if initial_idx < 0 or initial_idx >= len(waypoint_positions):
            return False

        current_floor_id = int(getattr(mapper, "current_floor_id", 0) or 0)
        if initial_idx < len(waypoint_floor_ids):
            try:
                initial_floor_id = int(waypoint_floor_ids[initial_idx] or 0)
            except (TypeError, ValueError):
                initial_floor_id = current_floor_id
            if initial_floor_id != current_floor_id:
                return False

        wp_py, wp_px = waypoint_positions[initial_idx]
        full_map = getattr(mapper, "full_map", None)
        crop_offset = getattr(getattr(mapper, "mapping_module", None), "full_map_crop_offset", None)
        if full_map is not None and crop_offset is not None:
            projector = RotatedMapProjector(
                map_h=full_map.shape[1],
                map_w=full_map.shape[2],
                crop_offset=crop_offset,
                agent_orientation_deg=float(current_pose[2]),
            )
            obstacle_mask = np.asarray(full_map[0] > 0.5, dtype=bool)
            current_distance_field = build_bounded_geodesic_distance_field(
                obstacle_mask=obstacle_mask,
                projector=projector,
                source_world=(
                    float(current_pose[1]) * 100.0 / resolution_cm,
                    float(current_pose[0]) * 100.0 / resolution_cm,
                ),
                max_distance_m=float(SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M),
                resolution_cm=resolution_cm,
            )
            if current_distance_field is not None:
                distance_m = query_world_distance_from_field_m(
                    distance_field=current_distance_field,
                    obstacle_mask=obstacle_mask,
                    projector=projector,
                    target_world=(int(wp_py), int(wp_px)),
                    resolution_cm=resolution_cm,
                )
                if distance_m is not None:
                    return distance_m <= float(SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M) + 1e-6

        curr_x_m, curr_y_m, _curr_heading = current_pose[:3]
        curr_py = int(round(float(curr_y_m) * 100.0 / resolution_cm))
        curr_px = int(round(float(curr_x_m) * 100.0 / resolution_cm))
        distance_m = (
            float(math.hypot(float(curr_py) - float(wp_py), float(curr_px) - float(wp_px)))
            * resolution_cm
            / 100.0
        )
        return distance_m <= float(SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M) + 1e-6

    def _build_initial_position_finish_guard_notice(self) -> str:
        return (
            "Current localization is still at/near INITIAL POSITION "
            f"(within about {float(SPACE_AREA_CURRENT_INITIAL_WAYPOINT_MAX_DISTANCE_M):.2f}m of the initial waypoint). "
            "Still near INITIAL POSITION(Task start): do not set global_task_finish=true. "
            "Execute Task first stage first."
        )

    def _clear_action_stagnation_prompt_state(self) -> None:
        self.action_stagnation_retry_pending = False
        self.action_stagnation_retry_notice_text = ""
        self.action_stagnation_progress_warning_text = ""

    def _clear_action_force_forward_prompt_state(self) -> None:
        self.action_force_forward_after_turns_pending = False
        self.action_force_forward_after_turns_notice_text = ""

    def _get_last_effective_turn_action_name(self) -> str:
        action_name_upper = str(
            getattr(self, "last_action_effective_name", "") or getattr(self, "last_action_name", "") or ""
        ).strip().upper()
        if action_name_upper in ("TURN_LEFT", "TURN_RIGHT"):
            return action_name_upper
        return ""

    def _build_reverse_turn_guard_progress_notice(self) -> str:
        last_turn_action = self._get_last_effective_turn_action_name()
        if last_turn_action == "TURN_LEFT":
            return "(constraint: last step turned left; do not immediately turn back right)"
        if last_turn_action == "TURN_RIGHT":
            return "(constraint: last step turned right; do not immediately turn back left)"
        return ""

    def _apply_immediate_reverse_turn_guard(
        self,
        allowed_action_names: Optional[Sequence[str]],
    ) -> Optional[Tuple[str, ...]]:
        last_turn_action = self._get_last_effective_turn_action_name()
        if not last_turn_action:
            return tuple(allowed_action_names) if allowed_action_names else None

        forbidden_turn = "TURN_RIGHT" if last_turn_action == "TURN_LEFT" else "TURN_LEFT"
        base_actions = list(allowed_action_names) if allowed_action_names else [
            "MOVE_FORWARD",
            "TURN_LEFT",
            "TURN_RIGHT",
            "STOP",
        ]
        filtered_actions = [
            str(name).strip().upper()
            for name in base_actions
            if str(name).strip().upper() != forbidden_turn
        ]
        if not filtered_actions:
            filtered_actions = ["STOP"]
        return tuple(filtered_actions)

    def _reset_blocked_front_controller_recovery_state(self) -> None:
        self.blocked_front_controller_recovery_count = 0

    def _build_action_force_forward_after_turns_notice(self) -> str:
        limit = max(1, int(getattr(self, "ACTION_CONSECUTIVE_TURN_LIMIT", 3) or 3))
        return (
            f"The last {limit} action decisions were consecutive turns. "
            "Do not output `TURN_LEFT 30deg` or `TURN_RIGHT 30deg` on this call. "
            "If arrival is already satisfied, output `STOP`; otherwise output one `MOVE_FORWARD` action only when FRONT is passable."
        )

    def _update_action_consecutive_turn_state(self, action_name: Optional[str]) -> None:
        action_name_upper = str(action_name or "").upper()
        limit = max(1, int(getattr(self, "ACTION_CONSECUTIVE_TURN_LIMIT", 3) or 3))
        if action_name_upper in ("TURN_LEFT", "TURN_RIGHT"):
            self.action_consecutive_turn_count = int(
                getattr(self, "action_consecutive_turn_count", 0) or 0
            ) + 1
            if self.action_consecutive_turn_count >= limit:
                self.action_force_forward_after_turns_pending = True
                self.action_force_forward_after_turns_notice_text = (
                    self._build_action_force_forward_after_turns_notice()
                )
                if self.action_consecutive_turn_count == limit:
                    print(
                        "[ActionTurnLimit] "
                        f"{limit} consecutive turn actions detected; "
                        "the next action call will forbid more turning and require forward progress or valid STOP"
                    )
            return

        self.action_consecutive_turn_count = 0
        self._clear_action_force_forward_prompt_state()

    def _get_action_progress_summary_for_prompt(self) -> str:
        base_summary = str(self.progress_summary or "").strip()
        last_action_hint = str(getattr(self, "last_action_progress_hint", "") or "").strip()
        warning_text = str(self.action_stagnation_progress_warning_text or "").strip()
        reverse_turn_notice = self._build_reverse_turn_guard_progress_notice()
        summary_text = base_summary
        if last_action_hint:
            if summary_text and summary_text != "(Just started - no actions yet)":
                summary_text = f"{summary_text} ({last_action_hint})"
            else:
                summary_text = f"({last_action_hint})"
        if reverse_turn_notice:
            if summary_text and summary_text != "(Just started - no actions yet)":
                summary_text = f"{summary_text} {reverse_turn_notice}"
            else:
                summary_text = reverse_turn_notice
        if not warning_text:
            return summary_text
        if summary_text and summary_text != "(Just started - no actions yet)":
            return f"{summary_text} {warning_text}"
        return warning_text

    @staticmethod
    def _build_last_action_progress_hint(
        *,
        action_name: Optional[str],
        action_variant: Optional[str] = "",
        planned_meters: Optional[float] = None,
        actual_meters: Optional[float] = None,
        actual_degrees: Optional[float] = None,
    ) -> str:
        action_name_upper = str(action_name or "").strip().upper()
        variant = str(action_variant or "").strip()
        if action_name_upper == "MOVE_FORWARD":
            planned = float(planned_meters or 0.0)
            actual = float(actual_meters or 0.0)
            if planned > 0.0 and actual < planned * 0.8:
                return f"prev MOVE_FORWARD {planned:.2f}m -> actual {actual:.2f}m, front blocked, avoid obstacle"
            if actual > 0.0:
                return f"prev MOVE_FORWARD actual {actual:.2f}m"
            return "prev MOVE_FORWARD -> actual 0.00m, front blocked, avoid obstacle"

        if action_name_upper in ("TURN_LEFT", "TURN_RIGHT"):
            side = "left" if action_name_upper == "TURN_LEFT" else "right"
            degree_text = (
                f"{float(actual_degrees):.0f}deg"
                if actual_degrees is not None
                else "30deg"
            )
            if variant == "avoid_obstacle":
                return (
                    f"last step TURN_{side.upper()}_AVOID {degree_text} "
                    "to avoid obstacle"
                )
            return (
                f"last step TURN_{side.upper()}_ALIGN {degree_text} "
                "to align destination landmark"
            )

        if action_name_upper == "STOP":
            return "prev STOP at destination"

        return ""

    def _on_env_step_about_to_run(self, *, actions: Optional[List[Any]] = None, context: str = "") -> None:
        self.timing_tracker.mark_episode_active()

    def _on_env_step_finished(
        self,
        *,
        actions: Optional[List[Any]] = None,
        context: str = "",
        dones: Optional[List[Any]] = None,
    ) -> None:
        action_name = ""
        if actions:
            try:
                action_name = str(self._action_name(actions[0]) or "")
            except Exception:
                action_name = ""
        self.timing_tracker.mark_episode_step_finished(
            action_name=action_name,
            episode_done=bool(dones[0]) if dones else False,
        )

    def _build_episode_timing_summary(self) -> Dict[str, Any]:
        return self.timing_tracker.build_summary()

    def _get_low_level_stagnation_threshold_m(self) -> float:
        return self.runtime_options.low_level_stagnation_threshold_m(self.move_distance)

    @staticmethod
    def _extract_direction_image_id(direction_name: Optional[str]) -> Optional[int]:
        match = re.search(r'IMAGE\s*(\d+)', str(direction_name or ""), re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    def _consume_pending_verify_view_restriction(
        self,
        image_paths: List[Any],
        direction_names: List[str],
    ) -> Tuple[List[Any], List[str], Dict[str, Any]]:
        restriction = dict(getattr(self, "pending_verify_view_restriction", None) or {})
        self.pending_verify_view_restriction = None
        if not restriction:
            return image_paths, direction_names, {}

        forbidden_view_ids = {
            int(view_id)
            for view_id in restriction.get("forbidden_view_ids", ()) or ()
        }
        if not forbidden_view_ids:
            return image_paths, direction_names, {}

        filtered_pairs: List[Tuple[Any, str]] = []
        removed_names: List[str] = []
        for image_path, direction_name in zip(image_paths, direction_names):
            image_id = self._extract_direction_image_id(direction_name)
            if image_id is not None and image_id in forbidden_view_ids:
                removed_names.append(str(direction_name))
                continue
            filtered_pairs.append((image_path, direction_name))

        if not filtered_pairs:
            return image_paths, direction_names, {}

        filtered_paths = [item[0] for item in filtered_pairs]
        filtered_names = [item[1] for item in filtered_pairs]
        if removed_names:
            print(
                "[VerifyViews] Omit forbidden rear view after stuck retreat: "
                + ", ".join(removed_names)
            )
        return filtered_paths, filtered_names, {
            "forbidden_view_ids": sorted(forbidden_view_ids),
            "removed_direction_names": removed_names,
        }

    def _build_stagnation_verify_notice(
        self,
        actual_retreat_m: float = 0.0,
        retreat_distance_m: Optional[float] = None,
    ) -> str:
        target_retreat_m = float(retreat_distance_m or self.STUCK_RETREAT_DISTANCE_M)
        if float(actual_retreat_m) >= max(0.75 * target_retreat_m, target_retreat_m - 0.2):
            retreat_text = (
                f"The system automatically turned around 180deg and moved about {target_retreat_m:.1f}m "
                "away from the blocked route."
            )
        elif float(actual_retreat_m) > 0.0:
            retreat_text = (
                f"The system automatically turned around 180deg and tried to move about {target_retreat_m:.1f}m "
                f"away from the blocked route (actual movement {float(actual_retreat_m):.2f}m)."
            )
        else:
            retreat_text = (
                f"The system automatically turned around 180deg and attempted a {target_retreat_m:.1f}m retreat, "
                "but movement was still very limited."
            )

        return (
            "You got stuck again after another three low-level MOVE_FORWARD steps with almost no movement, "
            "so the previous forward route is treated as blocked. "
            f"{retreat_text} "
            "The rear back view now points back toward that blocked route, so that back direction is forbidden for this replan. "
            "The back view (IMAGE 7: Back 180deg) is intentionally not provided. "
            "Choose only among the provided non-back views, and do not pick any backtracking direction."
        )

    def _build_action_stagnation_retry_notice(
        self,
        latest_actual_meters: float,
        stagnation_threshold_m: float,
    ) -> str:
        return (
            f"The last low-level MOVE_FORWARD {float(self.move_distance):.2f}m step advanced only "
            f"{float(latest_actual_meters):.2f}m (no-move threshold {float(stagnation_threshold_m):.2f}m), "
            "so the current FRONT route is blocked. Stop that forward attempt immediately. "
            "For this call, do not output `MOVE_FORWARD` into the same front route. "
            "Use the current view, obstacle lines, destination, and space structure to choose only "
            "`TURN_LEFT 30deg` or `TURN_RIGHT 30deg` around the obstacle, unless the destination is already reached and `STOP` is valid. "
            "Choose the side that best matches the destination and avoids the obstacle. "
            "After one side turn, if FRONT becomes passable and still points toward the destination, prefer forward progress instead of turning back."
        )

    @staticmethod
    def _build_post_avoidance_turn_notice(action_name: str) -> str:
        action_name_upper = str(action_name or "").upper()
        if action_name_upper == "TURN_LEFT":
            turn_text = "left"
            reverse_text = "right"
        else:
            turn_text = "right"
            reverse_text = "left"
        return (
            f"The last step was a forced obstacle-avoidance turn to the {turn_text} because the previous forward route was blocked. "
            f"Do not immediately turn back {reverse_text}. If FRONT is now passable and still aligned with the destination, move forward."
        )

    @staticmethod
    def _sanitize_action_memory_value(value: Optional[str]) -> str:
        text = re.sub(r"[\r\n|]+", " ", str(value or "")).strip()
        text = re.sub(r"\s{2,}", " ", text)
        return text or "destination"

    @staticmethod
    def _format_action_memory_distance_text(distance_m: Optional[float]) -> str:
        if distance_m is None:
            return "unknown"
        return f"{float(distance_m):.2f}m"

    @staticmethod
    def _raw_action_analysis_suggests_obstacle_avoidance(raw_action_analysis: Optional[str]) -> bool:
        text_norm = str(raw_action_analysis or "").strip().lower()
        if not text_norm:
            return False
        keywords = (
            "obstacle",
            "blocked",
            "unsafe",
            "only open side",
            "blocked-front",
            "front route",
            "recovery",
            "retry",
            "safer side",
        )
        return any(keyword in text_norm for keyword in keywords)

    def _get_previous_action_memory_target_snapshot(
        self,
        step_landmark_entries: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[float]]:
        entries = self._sort_action_landmark_entries(step_landmark_entries or [])
        landmark_match_terms = self._get_current_subtask_landmark_match_terms()
        for entry in entries:
            raw_name = str(entry.get("name") or "").strip()
            if raw_name and self._landmark_matches_current_subtask_landmark(
                raw_name,
                landmark_match_terms=landmark_match_terms,
            ):
                return (
                    self._sanitize_action_memory_value(
                        self._format_previous_subtask_landmark_name(raw_name, entry=entry)
                    ),
                    self._safe_float(entry.get("distance_m")),
                )

        fallback_target = (
            self._get_current_subtask_landmark_display_name()
            or self._get_next_waypoint_field(getattr(self, "current_subtask", None))
            or "destination"
        )
        return self._sanitize_action_memory_value(fallback_target), None

    def _build_previous_action_memory(
        self,
        *,
        action_name: Optional[str],
        raw_action_analysis: Optional[str],
        planned_meters: Optional[float],
        step_landmark_entries: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        action_name_upper = str(action_name or "").strip().upper()
        if not action_name_upper:
            return ""

        target_name, target_distance_m = self._get_previous_action_memory_target_snapshot(
            step_landmark_entries=step_landmark_entries,
        )
        target_distance_text = self._format_action_memory_distance_text(target_distance_m)

        if action_name_upper == "STOP":
            return (
                "LAST_STEP_STOP_AT_TARGET"
                f" | target={target_name}"
                f" | target_distance={target_distance_text}"
            )

        if action_name_upper in ("TURN_LEFT", "TURN_RIGHT"):
            turn_side = "LEFT" if action_name_upper == "TURN_LEFT" else "RIGHT"
            if self._raw_action_analysis_suggests_obstacle_avoidance(raw_action_analysis):
                return (
                    "LAST_STEP_AVOID_OBSTACLE"
                    f" | turn={turn_side}"
                    " | obstacle=FRONT blocked"
                )
            return (
                "LAST_STEP_ALIGN_DESTINATION"
                f" | turn={turn_side}"
                f" | target={target_name}"
                f" | target_distance={target_distance_text}"
            )

        if action_name_upper == "MOVE_FORWARD":
            move_m = self._safe_float(planned_meters)
            if move_m is None or move_m <= 0.0:
                move_m = float(self.move_distance)
            return (
                "LAST_STEP_FORWARD_TO_TARGET"
                f" | move={float(move_m):.2f}m"
                f" | target={target_name}"
                f" | target_distance={target_distance_text}"
            )

        return ""

    @staticmethod
    def _extract_side_hint(text: Optional[str]) -> Optional[str]:
        text_norm = str(text or "").strip().lower()
        if not text_norm:
            return None
        if re.search(r"\bleft\b", text_norm):
            return "LEFT"
        if re.search(r"\bright\b", text_norm):
            return "RIGHT"
        if re.search(r"\bfront\b", text_norm):
            return "FRONT"
        if re.search(r"\bback\b", text_norm):
            return "BACK"
        return None

    def _get_recent_forced_avoidance_turn_side(self) -> Optional[str]:
        if str(getattr(self, "last_action_semantic_variant", "") or "").strip() == "avoid_obstacle":
            action_name = str(getattr(self, "last_action_name", "") or "").strip().upper()
            if action_name == "TURN_LEFT":
                return "LEFT"
            if action_name == "TURN_RIGHT":
                return "RIGHT"
        match = re.search(
            r"forced obstacle-avoidance turn to the (left|right)",
            str(getattr(self, "action_stagnation_retry_notice_text", "") or ""),
            flags=re.IGNORECASE,
        )
        if match:
            return str(match.group(1)).upper()
        return None

    def _choose_blocked_front_recovery_side(self) -> Tuple[Optional[str], Dict[str, Dict[str, Any]]]:
        obstacle_distances = getattr(self, "latest_obstacle_distances", {}) or {}
        side_records: Dict[str, Dict[str, Any]] = {
            "LEFT": {
                "distance_text": obstacle_distances.get("left_30"),
                "distance_m": self._parse_distance_text_m(obstacle_distances.get("left_30")),
            },
            "RIGHT": {
                "distance_text": obstacle_distances.get("right_30"),
                "distance_m": self._parse_distance_text_m(obstacle_distances.get("right_30")),
            },
        }
        for side_name, record in side_records.items():
            distance_m = record.get("distance_m")
            record["blocked"] = bool(distance_m is not None and distance_m < float(OBS_BLOCKED_M))
            record["unknown"] = distance_m is None
            record["status"] = (
                "blocked"
                if record["blocked"]
                else ("unknown" if record["unknown"] else "passable")
            )
            record["side_name"] = side_name

        payload = dict(getattr(self, "current_subtask", None) or {})
        preferred_hint = None
        for raw_text in (
            payload.get("next_waypoint_direction"),
            payload.get("subtask_instruction"),
        ):
            hint = self._extract_side_hint(raw_text)
            if hint in ("LEFT", "RIGHT"):
                preferred_hint = hint
                break

        recent_avoidance_side = self._get_recent_forced_avoidance_turn_side()
        ranked_side_options: List[Tuple[float, str]] = []
        for side_name, record in side_records.items():
            if record["blocked"]:
                continue

            score = 0.0
            if preferred_hint == side_name:
                score += 3.0
            elif preferred_hint in ("LEFT", "RIGHT"):
                score -= 0.5

            if recent_avoidance_side == side_name:
                score += 0.5
            elif recent_avoidance_side in ("LEFT", "RIGHT"):
                score -= 0.25

            distance_m = record.get("distance_m")
            if distance_m is None:
                score -= 0.25
            else:
                score += min(float(distance_m), 5.0)

            ranked_side_options.append((score, side_name))

        ranked_side_options.sort(reverse=True)
        if not ranked_side_options:
            return None, side_records
        return ranked_side_options[0][1], side_records

    def _build_controller_forced_action_response(
        self,
        *,
        action_name: str,
        reasoning: str,
    ) -> Dict[str, Any]:
        action_name_upper = str(action_name or "").upper()
        if action_name_upper in ("TURN_LEFT", "TURN_RIGHT"):
            action_text = f"{action_name_upper}_AVOID {int(self.turn_angle)}deg"
        elif action_name_upper == "MOVE_FORWARD":
            action_text = f"{action_name_upper} {float(self.move_distance):g}m"
        else:
            action_text = "STOP"
        payload = {
            "reasoning": reasoning,
            "action": action_text,
            "controller_forced_recovery": True,
        }
        return payload

    @staticmethod
    def _build_action_response_artifact_payload(
        response: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = dict(response or {})
        allowed_keys = (
            "reasoning",
            "action",
            "controller_forced_recovery",
        )
        return {
            key: payload[key]
            for key in allowed_keys
            if key in payload
        }

    @staticmethod
    def _write_json_artifact(path: str, payload: Optional[Dict[str, Any]]) -> None:
        if not path or not isinstance(payload, dict):
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _copy_response_payload(payload: Optional[Dict[str, Any]], fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return dict(payload)
        return dict(fallback or {})

    def _get_latest_planner_raw_response_payload(
        self,
        fallback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        planner = getattr(self, "planner", None)
        payload = getattr(planner, "last_parsed_response_payload", None)
        return self._copy_response_payload(payload, fallback)

    def _get_latest_action_raw_response_payload(
        self,
        fallback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        executor = getattr(self, "action_executor", None)
        payload = getattr(executor, "last_parsed_response_payload", None)
        return self._copy_response_payload(payload, fallback)

    def _persist_response_artifacts(
        self,
        *,
        save_dir: Optional[str],
        raw_payload: Optional[Dict[str, Any]],
        controller_payload: Optional[Dict[str, Any]] = None,
        controller_filename: str = "response.controller.json",
    ) -> None:
        target_dir = str(save_dir or "").strip()
        if not target_dir:
            return
        self._write_json_artifact(os.path.join(target_dir, "response.json"), self._copy_response_payload(raw_payload))
        if isinstance(controller_payload, dict):
            self._write_json_artifact(os.path.join(target_dir, controller_filename), dict(controller_payload))

    def _build_forced_blocked_front_recovery_action(
        self,
        step_landmark_entries: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        current_entries = list(step_landmark_entries or [])
        auto_completed_subtask = self._should_autocomplete_subtask_during_action_step(current_entries)
        if auto_completed_subtask is not None:
            self._record_previous_subtask_autocomplete_landmark(auto_completed_subtask)
            landmark_kind = "opening-like" if auto_completed_subtask.get("is_opening_like") else "solid"
            distance_m = float(auto_completed_subtask.get("distance_m", 0.0) or 0.0)
            threshold_m = float(
                auto_completed_subtask.get(
                    "stop_distance_m",
                    self.ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M,
                )
            )
            response = self._build_controller_forced_action_response(
                action_name="STOP",
                reasoning=(
                    f"The destination landmark {auto_completed_subtask['name']} is already within "
                    f"{distance_m:.2f}m, which satisfies the auto-stop threshold {threshold_m:.2f}m, "
                    "so stop instead of forcing another obstacle-recovery move."
                ),
            )
            return {
                "action_id": resolve_habitat_action("STOP"),
                "action_name": "STOP",
                "response": response,
                "degrees": 0,
                "meters": 0.0,
                "recovery_kind": "stop",
                "recovery_meta": {
                    "landmark_name": auto_completed_subtask["name"],
                    "landmark_kind": landmark_kind,
                    "distance_m": distance_m,
                    "threshold_m": threshold_m,
                },
            }

        if int(getattr(self, "blocked_front_controller_recovery_count", 0) or 0) >= 1:
            return None

        selected_side, side_records = self._choose_blocked_front_recovery_side()
        if selected_side not in ("LEFT", "RIGHT"):
            return None

        turn_action_name = "TURN_LEFT" if selected_side == "LEFT" else "TURN_RIGHT"
        selected_record = dict(side_records.get(selected_side, {}) or {})
        other_side = "RIGHT" if selected_side == "LEFT" else "LEFT"
        other_record = dict(side_records.get(other_side, {}) or {})
        selected_distance_text = str(selected_record.get("distance_text") or "Unknown")
        other_distance_text = str(other_record.get("distance_text") or "Unknown")

        response = self._build_controller_forced_action_response(
            action_name=turn_action_name,
            reasoning=(
                "A blocked-front retry is active, so straight movement into the current FRONT route is forbidden. "
                f"{selected_side.title()} 30deg is the best controller-side recovery because it is "
                f"{selected_record.get('status', 'passable')} ({selected_distance_text})"
                + (
                    f", while {other_side.title()} 30deg is {other_record.get('status', 'unknown')} ({other_distance_text})"
                    if other_record
                    else ""
                )
                + ". Use one side turn now, then let the next action call continue forward only if the new FRONT route is passable and still task-aligned."
            ),
        )
        return {
            "action_id": resolve_habitat_action("TURN_LEFT")
            if selected_side == "LEFT"
            else resolve_habitat_action("TURN_RIGHT"),
            "action_name": turn_action_name,
            "response": response,
            "degrees": int(self.turn_angle),
            "meters": 0.0,
            "recovery_kind": "turn",
            "recovery_meta": {
                "selected_side": selected_side,
                "selected_distance_text": selected_distance_text,
                "other_side": other_side,
                "other_distance_text": other_distance_text,
            },
        }

    def _build_forced_forward_after_turn_limit_action(
        self,
        step_landmark_entries: Optional[Sequence[Dict[str, Any]]] = None,
        obstacle_distances: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        current_entries = list(step_landmark_entries or [])
        auto_completed_subtask = self._should_autocomplete_subtask_during_action_step(current_entries)
        if auto_completed_subtask is not None:
            self._record_previous_subtask_autocomplete_landmark(auto_completed_subtask)
            landmark_kind = "opening-like" if auto_completed_subtask.get("is_opening_like") else "solid"
            distance_m = float(auto_completed_subtask.get("distance_m", 0.0) or 0.0)
            threshold_m = float(
                auto_completed_subtask.get(
                    "stop_distance_m",
                    self.ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M,
                )
            )
            response = self._build_controller_forced_action_response(
                action_name="STOP",
                reasoning=(
                    f"The destination landmark {auto_completed_subtask['name']} is already within "
                    f"{distance_m:.2f}m, which satisfies the auto-stop threshold {threshold_m:.2f}m, "
                    "so stop instead of forcing another movement."
                ),
            )
            return {
                "action_id": resolve_habitat_action("STOP"),
                "action_name": "STOP",
                "response": response,
                "degrees": 0,
                "meters": 0.0,
                "recovery_kind": "stop",
                "recovery_meta": {
                    "landmark_name": auto_completed_subtask["name"],
                    "landmark_kind": landmark_kind,
                    "distance_m": distance_m,
                    "threshold_m": threshold_m,
                },
            }

        front_distance_text = str((obstacle_distances or {}).get("front") or "Unknown")
        if self._is_obstacle_distance_blocked(front_distance_text):
            response = self._build_controller_forced_action_response(
                action_name="STOP",
                reasoning=(
                    "Three consecutive turn actions have already been used, but the current FRONT route is still "
                    f"blocked ({front_distance_text}). Do not keep spinning in place; end this action stage and "
                    "return to thinking for a new route."
                ),
            )
            return {
                "action_id": resolve_habitat_action("STOP"),
                "action_name": "STOP",
                "response": response,
                "degrees": 0,
                "meters": 0.0,
                "recovery_kind": "replan_handoff",
                "recovery_meta": {
                    "front_distance_text": front_distance_text,
                },
            }

        response = self._build_controller_forced_action_response(
            action_name="MOVE_FORWARD",
            reasoning=(
                "Three consecutive turn actions have already happened. "
                f"The current FRONT route is still passable ({front_distance_text}), so force one short forward step "
                "instead of allowing another in-place turn."
            ),
        )
        return {
            "action_id": resolve_habitat_action("MOVE_FORWARD"),
            "action_name": "MOVE_FORWARD",
            "response": response,
            "degrees": 0,
            "meters": float(self.move_distance),
            "recovery_kind": "forced_forward",
            "recovery_meta": {
                "front_distance_text": front_distance_text,
            },
        }

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
            self.action_stagnation_streak = 1
            print(
                "[ActionStagnation] "
                f"low-level MOVE_FORWARD moved only {float(actual_meters):.2f}m "
                f"(no-move threshold {stagnation_threshold_m:.2f}m) | "
                "treat this forward step as blocked and stop the current forward action"
            )
            return True

        if self.action_stagnation_streak > 0:
            print(
                "[ActionStagnation] "
                f"Low-level MOVE_FORWARD recovered with {float(actual_meters):.2f}m movement; "
                "reset stagnation streak"
            )
        self.action_stagnation_streak = 0
        self._reset_blocked_front_controller_recovery_state()
        return False

    @staticmethod
    def _get_next_waypoint_field(payload: Optional[Dict[str, Any]]) -> str:
        return get_next_waypoint(payload)

    @staticmethod
    def _get_subtask_landmark_field(payload: Optional[Dict[str, Any]]) -> str:
        return get_subtask_landmark(payload)

    @staticmethod
    def _attempt_index_to_letter(attempt_index: int) -> str:
        return chr(ord('a') + max(0, int(attempt_index)))

    def _current_attempt_letter(self) -> str:
        return self._attempt_index_to_letter(int(getattr(self, 'subtask_attempt', 0) or 0))

    def _current_action_phase(self) -> str:
        return f"action{self.subtask_count}{self._current_attempt_letter()}"

    def _verify_phase(self, subtask_count: Optional[int] = None, attempt_index: int = 0) -> str:
        target_subtask_count = self.subtask_count if subtask_count is None else int(subtask_count)
        return f"verify_{target_subtask_count}{self._attempt_index_to_letter(attempt_index)}"

    def _current_verify_phase(self) -> str:
        return self._verify_phase(self.subtask_count, int(getattr(self, 'subtask_attempt', 0) or 0))

    def _current_subtask_run_id(self) -> str:
        return f"{self.subtask_count}{self._current_attempt_letter()}"

    def _get_episode_max_steps(self) -> int:
        return int(getattr(self.config.TASK_CONFIG.ENVIRONMENT, 'MAX_EPISODE_STEPS', 0) or 0)

    def _get_remaining_episode_steps(self) -> int:
        return max(0, self._get_episode_max_steps() - int(getattr(self, 'current_step', 0) or 0))

    def _has_budget_for_thinking_cycle(self) -> bool:
        # Reserve one extra step so the controller can still call STOP after thinking if needed.
        return self._get_remaining_episode_steps() > int(self.THINKING_LOOKAROUND_STEPS)

    def _should_hold_last_episode_step_for_stop(self, action_name: Optional[str] = None) -> bool:
        if str(action_name or "").upper() == "STOP":
            return False
        return self._get_remaining_episode_steps() <= 1

    def _get_success_distance_m(self) -> float:
        try:
            return float(
                getattr(
                    self.config.TASK_CONFIG.TASK,
                    'SUCCESS_DISTANCE',
                    EVAL_SUCCESS_DISTANCE_M,
                ) or EVAL_SUCCESS_DISTANCE_M
            )
        except Exception:
            return float(EVAL_SUCCESS_DISTANCE_M)

    def _parse_subtask_destination(self) -> Tuple[Optional[str], Optional[str]]:
        destination = ""
        if getattr(self, 'current_subtask', None):
            destination = self._get_next_waypoint_field(self.current_subtask)
        if not destination or "'s " not in destination:
            return None, None

        room_text, object_text = destination.split("'s ", 1)
        room_norm = strip_space_type_variant_suffixes(room_text.strip().lower())
        object_norm = self._normalize_landmark_text(object_text)
        return room_norm or None, object_norm or None

    @classmethod
    def _landing_side_from_text(cls, text: Optional[str]) -> Optional[str]:
        normalized = cls._normalize_waypoint_endpoint_label(text)
        if not normalized:
            normalized = cls._normalize_landmark_text(text)
        if not normalized:
            return None

        if any(token in normalized for token in ("top landing", "top of stairs", "upper landing", "upstairs landing")):
            return "top"
        if any(token in normalized for token in ("bottom landing", "bottom of stairs", "lower landing", "downstairs landing")):
            return "bottom"
        if "landing" in normalized:
            return "landing"
        return None

    @classmethod
    def _is_stairs_like_text(cls, text: Optional[str]) -> bool:
        normalized = cls._normalize_waypoint_endpoint_label(text)
        if not normalized:
            normalized = cls._normalize_landmark_text(text)
        if not normalized:
            return False
        return any(token in normalized for token in ("stairs", "stair", "staircase", "stairway"))

    def _current_subtask_uses_stairs_like_landmark(self) -> bool:
        current_subtask = getattr(self, "current_subtask", None) or {}
        subtask_landmark = self._get_subtask_landmark_field(current_subtask)
        _dest_room, dest_object = self._parse_subtask_destination()
        return (
            self._is_stairs_like_text(subtask_landmark)
            or self._is_stairs_like_text(dest_object)
        )

    def _current_area_matches_stair_destination(
        self,
        dest_room: Optional[str],
        dest_object: Optional[str],
        subtask_landmark: Optional[str],
    ) -> bool:
        if not self._is_stairs_like_text(subtask_landmark):
            return False
        if not (self._is_stairs_like_text(dest_room) or self._landing_side_from_text(dest_object) is not None):
            return False
        if getattr(self, "mapper", None) is None:
            return False

        dest_room_norm = strip_space_type_variant_suffixes(str(dest_room or "").strip().lower()) or None
        dest_object_norm = self._normalize_waypoint_endpoint_label(dest_object) or self._normalize_landmark_text(dest_object)
        current_area_text = (
            getattr(self.mapper, "current_space_area_display_label", "")
            or getattr(self.mapper, "current_space_area_label", "")
        )
        current_area_norm = self._normalize_waypoint_endpoint_label(current_area_text)
        current_area_type = self._normalize_landmark_text(
            getattr(self.mapper, "current_space_area_type", "")
        )
        dest_side = self._landing_side_from_text(dest_object_norm)

        area_matches = False
        if current_area_norm:
            room_matches = (
                not dest_room_norm or
                dest_room_norm in current_area_norm or
                current_area_norm in dest_room_norm
            )
            side_matches = (
                dest_side is None or
                self._landing_side_from_text(current_area_norm) in {
                    dest_side,
                    "landing" if dest_side in {"top", "bottom"} else dest_side,
                }
            )
            if room_matches and side_matches:
                area_matches = True

        if area_matches:
            return True

        if current_area_type == "stairs" and self._is_stairs_like_text(current_area_norm):
            if dest_side is None:
                return True
            current_side = self._landing_side_from_text(current_area_norm)
            if dest_side == "landing":
                return current_side in {"landing", "top", "bottom"}
            if current_side == dest_side:
                return True

        current_pose = getattr(self.mapper, "full_pose", None)
        resolution_cm = float(getattr(self.mapper, "resolution", 0.0) or 0.0)
        if current_pose is None or len(current_pose) < 2 or resolution_cm <= 0.0:
            return False

        waypoint_positions, _waypoint_ids, waypoint_descriptions = self.mapper.get_global_waypoints()
        waypoint_floor_ids = self.mapper.get_global_waypoint_floor_ids()
        current_floor_id = int(getattr(self.mapper, "current_floor_id", 0) or 0)

        semantic_match_radius_m = max(
            float(self.ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M),
            float(self.ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M),
        )
        for idx, description in enumerate(waypoint_descriptions or []):
            if idx >= len(waypoint_positions):
                break
            if idx < len(waypoint_floor_ids) and int(waypoint_floor_ids[idx] or 0) != current_floor_id:
                continue
            desc_norm = self._normalize_waypoint_endpoint_label(description)
            if not desc_norm:
                continue

            room_matches = (
                not dest_room_norm or
                dest_room_norm in desc_norm or
                desc_norm in dest_room_norm
            )
            if not room_matches:
                continue

            if dest_object_norm and dest_object_norm not in desc_norm:
                desc_side = self._landing_side_from_text(desc_norm)
                if dest_side is None or desc_side != dest_side:
                    continue

            wp_x, wp_y = waypoint_positions[idx]
            try:
                curr_x = float(current_pose[0])
                curr_y = float(current_pose[1])
                dist_m = float(np.hypot(float(wp_x) - curr_x, float(wp_y) - curr_y) * (resolution_cm / 100.0))
            except (TypeError, ValueError):
                continue
            if dist_m <= semantic_match_radius_m:
                return True

        return False

    def _get_current_subtask_autocomplete_match_terms(self) -> List[str]:
        """Use the current VLM subtask landmark as the only proximity auto-complete match term."""
        if self._current_subtask_uses_stairs_like_landmark():
            return []
        subtask_landmark_norm = self._normalize_landmark_text(
            self._get_subtask_landmark_field(getattr(self, 'current_subtask', None))
        )
        if not subtask_landmark_norm:
            return []
        return [subtask_landmark_norm]

    def _should_autocomplete_subtask_during_action_step(
        self,
        step_landmark_entries: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        landmark_match_terms = self._get_current_subtask_autocomplete_match_terms()
        if not landmark_match_terms:
            return None

        matches: List[Dict[str, Any]] = []
        ordered_entries = [
            dict(entry)
            for entry in self._sort_action_landmark_entries(step_landmark_entries or [])
        ]
        for entry in ordered_entries[: int(self.ACTION_SUBTASK_AUTOCOMPLETE_TOPK)]:
            if not self._entry_reaches_action_arrival_threshold(
                entry,
                landmark_match_terms=landmark_match_terms,
            ):
                continue
            matches.append({
                "name": str(entry.get('name') or landmark_match_terms[0]),
                "distance_m": float(entry.get('distance_m')),
                "confidence": float(entry.get('confidence', 0.0) or 0.0),
                "angle_deg": entry.get('angle_deg'),
                "is_opening_like": bool(self._is_opening_like_landmark_entry(entry)),
                "stop_distance_m": float(self._autocomplete_stop_distance_m(entry)),
                "source": "vis" if str(entry.get("source", "mem") or "mem") == "vis" else "mem",
                "display_id": self._safe_int(entry.get("display_id")),
                "instance_idx": self._safe_int(entry.get("instance_idx")),
                "class_total": self._safe_int(entry.get("class_total")),
                "selection_rank": self._safe_int(entry.get("selection_rank")),
                "instance_uid": self._safe_int(entry.get("instance_uid")),
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

    def _entry_reaches_action_arrival_threshold(
        self,
        entry: Optional[Dict[str, Any]],
        landmark_match_terms: Optional[Sequence[str]] = None,
    ) -> bool:
        if not entry:
            return False
        if self._is_stairs_like_text(entry.get("name")) or self._current_subtask_uses_stairs_like_landmark():
            return False
        if not self._landmark_matches_current_subtask_landmark(
            entry.get("name"),
            landmark_match_terms=landmark_match_terms,
        ):
            return False

        distance_m = self._safe_float(entry.get("distance_m"))
        if distance_m is None:
            return False

        stop_distance_m = self._autocomplete_stop_distance_m(entry)
        return float(distance_m) <= float(stop_distance_m)

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

    def _get_current_subtask_landmark_match_terms(self) -> List[str]:
        subtask_landmark = self._normalize_landmark_text(
            self._get_subtask_landmark_field(getattr(self, 'current_subtask', None))
        )
        return [subtask_landmark] if subtask_landmark else []

    def _get_current_subtask_landmark_display_name(self) -> str:
        payload = getattr(self, 'current_subtask', None) or {}
        for raw_landmark_source in (
            self._get_subtask_landmark_field(payload),
            self._get_next_waypoint_field(payload),
            getattr(self, 'target_landmark', None),
        ):
            cleaned = strip_space_type_variant_suffixes(str(raw_landmark_source or "")).strip()
            if not cleaned:
                continue
            if "'s " in cleaned and raw_landmark_source == self._get_next_waypoint_field(payload):
                cleaned = cleaned.split("'s ", 1)[1].strip()
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
            if cleaned:
                return cleaned
        return "Unknown"

    @staticmethod
    def _safe_float(value: Optional[Any]) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Optional[Any]) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _format_previous_subtask_landmark_name(
        self,
        name: Optional[str],
        entry: Optional[Dict[str, Any]] = None,
    ) -> str:
        clean_name = re.sub(r"\s+", " ", str(name or "").strip()).strip(" ,;:-") or "Unknown"
        if clean_name == "Unknown":
            return "[Unknown]"

        display_id = self._safe_int((entry or {}).get("display_id"))
        instance_idx = self._safe_int((entry or {}).get("instance_idx"))
        class_total = self._safe_int((entry or {}).get("class_total")) or 1

        suffix = ""
        if class_total > 1:
            if display_id is not None and display_id > 0:
                suffix = str(display_id)
            elif instance_idx is not None and instance_idx >= 0:
                suffix = str(instance_idx + 1)

        return f"[{clean_name}]{suffix}"

    def _get_latest_action_local_map_landmark_entries(self) -> List[Dict[str, Any]]:
        latest_prompt_entries = self.landmark_memory.get_latest_prompt_entries()
        history = self.landmark_memory.prompt_entries_by_step
        latest_history_step = max(history.keys(), default=-1)
        history_tail_count = len(history.get(latest_history_step, []) or []) if latest_history_step >= 0 else 0
        tracked_landmark_queries = tuple(
            self._normalize_landmark_text(name)
            for name in list(getattr(self, "landmark_classes", []) or [])
            if self._normalize_landmark_text(name)
        )
        cache_key = (
            int(getattr(self, "current_step", -1) or -1),
            tracked_landmark_queries,
            len(latest_prompt_entries or []),
            len(history),
            latest_history_step,
            history_tail_count,
        )
        if cache_key == getattr(self, "_latest_action_landmark_entries_cache_key", None):
            return [dict(entry) for entry in self._latest_action_landmark_entries_cache_value]

        latest_entries = self._sort_action_landmark_entries(latest_prompt_entries)
        if latest_entries:
            self._latest_action_landmark_entries_cache_key = cache_key
            self._latest_action_landmark_entries_cache_value = [dict(entry) for entry in latest_entries]
            return latest_entries

        for step_idx in sorted(history.keys(), reverse=True):
            entries = self._sort_action_landmark_entries(history.get(step_idx, []) or [])
            if entries:
                self._latest_action_landmark_entries_cache_key = cache_key
                self._latest_action_landmark_entries_cache_value = [dict(entry) for entry in entries]
                return entries
        self._latest_action_landmark_entries_cache_key = cache_key
        self._latest_action_landmark_entries_cache_value = []
        return []

    def _sort_action_landmark_entries(
        self,
        entries: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ordered_entries = [
            dict(entry)
            for entry in list(entries or [])
            if isinstance(entry, dict)
        ]
        ordered_entries.sort(
            key=lambda entry: (
                self._safe_int(entry.get("selection_rank"))
                if self._safe_int(entry.get("selection_rank")) is not None
                else 1e9,
                -float(self._safe_float(entry.get("confidence")) or 0.0),
                self._safe_float(entry.get("distance_m"))
                if self._safe_float(entry.get("distance_m")) is not None
                else float("inf"),
                str(entry.get("name", "")),
            )
        )
        return ordered_entries

    def _build_action_landmark_overlay_lines(
        self,
        entries: Sequence[Dict[str, Any]],
        topk: int = 2,
    ) -> List[str]:
        ordered_entries = self._sort_action_landmark_entries(entries or [])
        lines: List[str] = []
        for entry in ordered_entries[:max(1, int(topk))]:
            name = str(entry.get("name") or "Unknown").strip() or "Unknown"
            display_id = self._safe_int(entry.get("display_id"))
            distance_m = self._safe_float(entry.get("distance_m"))
            angle_deg = self._safe_float(entry.get("angle_deg"))
            confidence = self._safe_float(entry.get("confidence")) or 0.0
            prefix = f"#{display_id}" if display_id is not None and display_id > 0 else "#?"
            distance_text = f"{distance_m:.2f}m" if distance_m is not None else "Unknown"
            direction_text = format_relative_direction(angle_deg) if angle_deg is not None else "Unknown"
            lines.append(
                f"{prefix} {name} | {distance_text} | {direction_text} | conf: {confidence:.3f}"
            )
        return lines

    def _log_action_landmark_debug(
        self,
        tag: str,
        entries: Sequence[Dict[str, Any]],
    ) -> None:
        ordered_entries = self._sort_action_landmark_entries(entries or [])
        if not ordered_entries:
            self.latest_action_local_map_debug_lines = []
            return

        self.latest_action_local_map_debug_lines = self._build_action_landmark_overlay_lines(ordered_entries)

    def _get_action_landmark_prompt_entries(self, detection_step: Optional[int]) -> List[Dict[str, Any]]:
        history = self.landmark_memory.prompt_entries_by_step
        topk = max(1, int(LANDMARK_STRIP_TOPK))
        if detection_step is not None:
            entries = self._sort_action_landmark_entries(history.get(detection_step, []) or [])
            if entries:
                return entries[:topk]

        latest_entries = self._sort_action_landmark_entries(
            self.landmark_memory.get_latest_prompt_entries()
        )
        if latest_entries:
            return latest_entries[:topk]
        return []

    def _record_previous_subtask_autocomplete_landmark(
        self,
        entry: Optional[Dict[str, Any]],
    ) -> None:
        payload = dict(entry or {})
        raw_name = str(payload.get("name") or "").strip() or self._get_current_subtask_landmark_display_name()
        angle_deg = self._safe_float(payload.get("angle_deg"))
        distance_m = self._safe_float(payload.get("distance_m"))
        is_opening_like = bool(payload.get("is_opening_like"))
        stop_distance_m = self._autocomplete_stop_distance_m(
            payload,
            is_opening_like=is_opening_like,
        )
        info = {
            "raw_name": raw_name,
            "name": self._format_previous_subtask_landmark_name(raw_name, entry=payload),
            "display_id": self._safe_int(payload.get("display_id")),
            "instance_idx": self._safe_int(payload.get("instance_idx")),
            "class_total": self._safe_int(payload.get("class_total")),
            "instance_uid": self._safe_int(payload.get("instance_uid")),
            "final_distance_m": distance_m,
            "final_direction": format_relative_direction(angle_deg) if angle_deg is not None else "Unknown",
            "final_angle_deg": angle_deg,
            "selection_rank": self._safe_int(payload.get("selection_rank")),
            "confidence": self._safe_float(payload.get("confidence")),
            "source": "auto_subtask_complete",
            "is_opening_like": is_opening_like,
            "stop_distance_m": float(stop_distance_m),
            "has_arrived": True,
            "matched": True,
        }
        self.previous_subtask_autocomplete_landmark_info = info
        self._previous_subtask_landmark_summary_cache_key = None

    @staticmethod
    def _format_previous_subtask_landmark_summary_item(info: Dict[str, Any]) -> str:
        distance_m = info.get("final_distance_m")
        distance_text = (
            f"{float(distance_m):.2f}m"
            if distance_m is not None
            else "Unknown"
        )
        arrived_suffix = " (you have arrived now)" if bool(info.get("has_arrived")) else ""
        return (
            f"{info.get('name', '[Unknown]')}{arrived_suffix}, "
            f"{distance_text}, {info.get('final_direction', 'Unknown')}"
        )

    def _build_previous_subtask_landmark_summary(self) -> str:
        entries = self._get_latest_action_local_map_landmark_entries()
        autocomplete_info = dict(getattr(self, "previous_subtask_autocomplete_landmark_info", {}) or {})
        landmark_match_terms = tuple(self._get_current_subtask_landmark_match_terms())
        cache_key = (
            int(getattr(self, "current_step", -1) or -1),
            tuple(
                (
                    str(entry.get("name") or ""),
                    self._safe_int(entry.get("display_id")),
                    self._safe_int(entry.get("instance_uid")),
                    self._safe_float(entry.get("distance_m")),
                    self._safe_float(entry.get("angle_deg")),
                    bool(entry.get("has_arrived")),
                )
                for entry in entries[:2]
            ),
            tuple(sorted((str(key), str(value)) for key, value in autocomplete_info.items())),
            landmark_match_terms,
        )
        if cache_key == getattr(self, "_previous_subtask_landmark_summary_cache_key", None):
            self.previous_subtask_landmark_final_info = dict(
                getattr(self, "_previous_subtask_landmark_summary_cache_info", {}) or {}
            )
            return str(getattr(self, "_previous_subtask_landmark_summary_cache_summary", "") or "")

        if entries and not autocomplete_info:
            inferred_autocomplete = self._should_autocomplete_subtask_during_action_step(entries)
            if inferred_autocomplete is not None:
                self._record_previous_subtask_autocomplete_landmark(inferred_autocomplete)
                autocomplete_info = dict(getattr(self, "previous_subtask_autocomplete_landmark_info", {}) or {})
                cache_key = (
                    int(getattr(self, "current_step", -1) or -1),
                    tuple(
                        (
                            str(entry.get("name") or ""),
                            self._safe_int(entry.get("display_id")),
                            self._safe_int(entry.get("instance_uid")),
                            self._safe_float(entry.get("distance_m")),
                            self._safe_float(entry.get("angle_deg")),
                            bool(entry.get("has_arrived")),
                        )
                        for entry in entries[:2]
                    ),
                    tuple(sorted((str(key), str(value)) for key, value in autocomplete_info.items())),
                    landmark_match_terms,
                )
        if not entries:
            fallback_info = dict(autocomplete_info)
            if not fallback_info:
                self.previous_subtask_landmark_final_info = {}
                self._previous_subtask_landmark_summary_cache_key = cache_key
                self._previous_subtask_landmark_summary_cache_summary = ""
                self._previous_subtask_landmark_summary_cache_info = {}
                return ""
            fallback_info["entries"] = [dict(fallback_info)]
            fallback_info["count"] = 1
            self.previous_subtask_landmark_final_info = dict(fallback_info)
            summary_text = "Landmark: " + self._format_previous_subtask_landmark_summary_item(fallback_info)
            self._previous_subtask_landmark_summary_cache_key = cache_key
            self._previous_subtask_landmark_summary_cache_summary = summary_text
            self._previous_subtask_landmark_summary_cache_info = dict(fallback_info)
            return summary_text
        summary_items: List[str] = []
        info_entries: List[Dict[str, Any]] = []
        landmark_arrival_match_terms = list(landmark_match_terms)
        for rank, entry in enumerate(entries[:2], start=1):
            raw_name = str(entry.get("name") or "").strip() or "Unknown"
            distance_m = self._safe_float(entry.get("distance_m"))
            angle_deg = self._safe_float(entry.get("angle_deg"))
            is_opening_like = self._is_opening_like_landmark_entry(entry)
            stop_distance_m = self._autocomplete_stop_distance_m(
                entry,
                is_opening_like=is_opening_like,
            )
            entry_name_norm = self._normalize_landmark_text(raw_name)
            autocomplete_name_norm = self._normalize_landmark_text(
                autocomplete_info.get("raw_name") or autocomplete_info.get("name")
            )
            autocomplete_instance_uid = self._safe_int(autocomplete_info.get("instance_uid"))
            entry_instance_uid = self._safe_int(entry.get("instance_uid"))
            autocomplete_display_id = self._safe_int(autocomplete_info.get("display_id"))
            entry_display_id = self._safe_int(entry.get("display_id"))
            has_arrived = bool(entry.get("has_arrived"))
            matched_autocomplete_anchor = False
            if entry_name_norm and autocomplete_name_norm:
                same_name = (
                    entry_name_norm == autocomplete_name_norm
                    or entry_name_norm in autocomplete_name_norm
                    or autocomplete_name_norm in entry_name_norm
                )
                if same_name:
                    if autocomplete_instance_uid is not None and entry_instance_uid is not None:
                        matched_autocomplete_anchor = entry_instance_uid == autocomplete_instance_uid
                    elif autocomplete_display_id is not None and entry_display_id is not None:
                        matched_autocomplete_anchor = entry_display_id == autocomplete_display_id
                    else:
                        autocomplete_class_total = self._safe_int(autocomplete_info.get("class_total")) or 1
                        entry_class_total = self._safe_int(entry.get("class_total")) or 1
                        matched_autocomplete_anchor = (
                            autocomplete_class_total <= 1 and entry_class_total <= 1
                        )
            if matched_autocomplete_anchor and autocomplete_info:
                autocomplete_distance_m = self._safe_float(autocomplete_info.get("final_distance_m"))
                if autocomplete_distance_m is not None:
                    distance_m = autocomplete_distance_m
                autocomplete_angle_deg = self._safe_float(autocomplete_info.get("final_angle_deg"))
                if autocomplete_angle_deg is not None:
                    angle_deg = autocomplete_angle_deg
                stop_distance_m = float(
                    self._safe_float(autocomplete_info.get("stop_distance_m")) or stop_distance_m
                )
                has_arrived = bool(autocomplete_info.get("has_arrived", True))
            if not has_arrived:
                has_arrived = self._entry_reaches_action_arrival_threshold(
                    entry,
                    landmark_match_terms=landmark_arrival_match_terms,
                )
            info = {
                "raw_name": raw_name,
                "name": self._format_previous_subtask_landmark_name(raw_name, entry=entry),
                "display_id": self._safe_int(entry.get("display_id")),
                "instance_idx": self._safe_int(entry.get("instance_idx")),
                "class_total": self._safe_int(entry.get("class_total")),
                "instance_uid": self._safe_int(entry.get("instance_uid")),
                "final_distance_m": distance_m,
                "final_direction": format_relative_direction(angle_deg) if angle_deg is not None else "Unknown",
                "final_angle_deg": angle_deg,
                "selection_rank": self._safe_int(entry.get("selection_rank")),
                "confidence": self._safe_float(entry.get("confidence")),
                "source": str(entry.get("source", "")),
                "is_opening_like": bool(is_opening_like),
                "stop_distance_m": float(stop_distance_m),
                "has_arrived": bool(has_arrived),
                "matched": True,
            }
            info_entries.append(dict(info))
            summary_items.append(self._format_previous_subtask_landmark_summary_item(info))

        if not info_entries:
            self.previous_subtask_landmark_final_info = {}
            self._previous_subtask_landmark_summary_cache_key = cache_key
            self._previous_subtask_landmark_summary_cache_summary = ""
            self._previous_subtask_landmark_summary_cache_info = {}
            return ""

        primary_info = dict(info_entries[0])
        primary_info["entries"] = [dict(item) for item in info_entries]
        primary_info["count"] = len(info_entries)
        self.previous_subtask_landmark_final_info = primary_info
        summary_text = "Landmark: " + " || ".join(summary_items)
        self._previous_subtask_landmark_summary_cache_key = cache_key
        self._previous_subtask_landmark_summary_cache_summary = summary_text
        self._previous_subtask_landmark_summary_cache_info = dict(primary_info)
        return summary_text

    def _build_previous_subtask_instruction_summary(
        self,
        subtask: Optional[Dict[str, Any]],
    ) -> str:
        payload = dict(subtask or {})
        destination = self._get_next_waypoint_field(payload) or ""
        next_waypoint_direction = str(payload.get("next_waypoint_direction", "") or "").strip()
        return self._sanitize_subtask_instruction_text(
            payload.get("subtask_instruction"),
            destination=destination,
            next_waypoint_direction=next_waypoint_direction,
            keep_view_prefix=False,
        )

    def _landmark_matches_current_subtask_landmark(
        self,
        name: Optional[str],
        landmark_match_terms: Optional[Sequence[str]] = None,
    ) -> bool:
        normalized = self._normalize_landmark_text(name)
        if not normalized:
            return False

        match_terms = list(landmark_match_terms or self._get_current_subtask_landmark_match_terms())
        if not match_terms:
            return False

        return any(
            normalized == match_term or normalized in match_term or match_term in normalized
            for match_term in match_terms
        )

    def _get_current_action_step_landmark_entries(self) -> List[Dict[str, Any]]:
        entries = self.landmark_memory.get_step_prompt_entries(self.current_step)
        if entries:
            return entries
        latest_entries = self._get_latest_action_local_map_landmark_entries()
        if latest_entries:
            return latest_entries
        return self.landmark_memory.get_step_visible_entries(self.current_step)

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
                "floor_id": int(record.get("floor_id", map_state.get('current_floor_id', 0)) or 0),
                "center_world_px": [int(center_world_px[0]), int(center_world_px[1])],
                "connected_area_labels": [str(item) for item in record.get("connected_area_labels", []) or []],
            })

        waypoint_memory = {
            "current_space_area_label": str(map_state.get('current_space_area_label', 'Unknown') or 'Unknown'),
            "current_space_area_type": str(map_state.get('current_space_area_type', 'Unknown') or 'Unknown'),
            "current_floor_id": int(map_state.get('current_floor_id', 0) or 0),
            "current_floor_label": str(map_state.get('current_floor_label', 'F1') or 'F1'),
            "current_world_z": map_state.get('current_world_z'),
            "multi_floor_active": bool(map_state.get('multi_floor_active', False)),
            "on_stairs_connector": bool(map_state.get('on_stairs_connector', False)),
            "stair_connectors": list(map_state.get('stair_connectors', []) or []),
            "waypoint_positions": waypoint_positions,
            "waypoint_ids": waypoint_ids,
            "waypoint_descriptions": waypoint_descriptions,
            "waypoint_area_labels": waypoint_area_labels,
            "waypoint_initial_neighborhood_flags": [
                bool(flag)
                for flag in map_state.get('waypoint_initial_neighborhood_flags', []) or []
            ],
            "global_waypoint_initial_neighborhood_flags": [
                bool(flag)
                for flag in self.mapper.get_global_waypoint_initial_neighborhood_flags()
            ],
            "space_area_records": space_area_records,
            "waypoint_summary": self._get_waypoint_summary(include_area_chain=True),
        }
        self.save_manager.save_waypoint_memory(
            waypoint_memory=waypoint_memory,
            instruction=self.current_instruction,
            current_step=self.current_step,
        )

    def _execute_auto_retreat(self, retreat_distance_m: float = 1.0) -> Tuple[float, bool]:
        """Turn around, move away from a stuck route, then hand off to thinking/lookaround."""
        turn_steps = max(1, round(180.0 / float(self.turn_angle)))
        move_steps = max(1, round(float(retreat_distance_m) / float(self.move_distance)))
        retreated_m = 0.0
        episode_done = False

        print(
            f"\n[AutoRetreat] Action stagnation detected. "
            f"Turn around 180deg and move {retreat_distance_m:.2f}m before replan."
        )

        for _ in range(turn_steps):
            result = self.step_with_vlm(
                resolve_habitat_action("TURN_RIGHT"),
                action_name="AUTO_RETREAT_TURN_RIGHT",
                save_vis=True,
                enable_landmark_detection=False,
            )
            print(f"  [Step {self.current_step}] AUTO_RETREAT_TURN_RIGHT")
            if result.get('done', False):
                episode_done = True
                break

        for _ in range(move_steps):
            if episode_done:
                break
            pose_before_step = self._get_agent_pose()
            result = self.step_with_vlm(
                resolve_habitat_action("MOVE_FORWARD"),
                action_name="AUTO_RETREAT_FORWARD",
                save_vis=True,
                enable_landmark_detection=False,
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

        if retreated_m > 0.0:
            self._append_progress_note(
                f"got stuck after repeated forward steps, then turned around and retreated {retreated_m:.2f}m before replanning"
            )
        else:
            self._append_progress_note(
                "got stuck after repeated forward steps, then turned around and triggered replan"
            )
        self.pose_before_action = self._get_agent_pose()
        return retreated_m, episode_done

    def reset_episode(self, episode_id: int = None):
        """Reset the episode state, including VLM-specific runtime state."""
        # Clean output directories from the previous episode.
        if episode_id is not None:
            import shutil
            for old_episode_dir in get_episode_detail_path_candidates(self.config.PATHS.RESULTS_DIR, episode_id):
                if os.path.exists(old_episode_dir):
                    print(f"[Reset] Removed previous episode data: {old_episode_dir}")
                    try:
                        shutil.rmtree(old_episode_dir)
                    except PermissionError as exc:
                        raise PermissionError(
                            "Cannot remove stale episode outputs before reset: "
                            f"{old_episode_dir}. This is usually caused by files created "
                            "by another user or by a Docker container running as root. "
                            "Fix the ownership or delete the stale directory first."
                        ) from exc
        
        # Reset shared controller state first.
        super().reset_episode(episode_id)
        
        # Initialize the save manager under the configured results root.
        self.save_manager = SaveManager(
            self.config.PATHS.RESULTS_DIR,
            self.current_episode_id,
            save_waypoint_memory=self.runtime_options.save_waypoint_memory,
        )
        
        # Reset VLM-specific runtime state.
        self._reset_vlm_episode_state()
        
        # Waypoints are managed by the mapper and cleared inside `mapper.reset()`.

        # Initialize the navigation visualizer for RGB + top-down outputs.
        visualization_dir = os.path.join(self.episode_dir, 'visualization')
        self.nav_visualizer = NavigationVisualizer(
            visualization_dir,
            save_step_images=self.runtime_options.save_navigation_step_images,
            keep_frames_for_gif=self.runtime_options.save_navigation_gif,
        )
        self.nav_visualizer.setup_maps_dir(self.episode_dir)
        
    @property
    def episode_dir(self) -> str:
        """Return the current episode output directory."""
        return get_episode_detail_dir(self.config.PATHS.RESULTS_DIR, self.current_episode_id)

    @classmethod
    def _normalize_landmark_text(cls, text: Optional[str]) -> Optional[str]:
        """Normalize landmark text without forcing a short phrase length."""
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
        return cls._normalize_landmark_text(cleaned)

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
                if (
                    anchor_distance_m
                    <= self.runtime_options.final_destination_match_autostop_radius_m
                ):
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
        response['final_waypoint_destination_anchor_radius_m'] = (
            self.runtime_options.final_destination_match_autostop_radius_m
        )
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
    def _iter_landmark_source_terms(cls, source: Optional[str]) -> List[str]:
        """Extract object-like landmark phrases from structured fields."""
        if not source:
            return []

        text = str(source).strip()
        if not text:
            return []

        source_terms = [text]

        if "Detection:" in text:
            detection_part = text.split("Detection:", 1)[1].split("|", 1)[0].strip()
            if detection_part:
                source_terms.append(detection_part)

        if "'s" in text:
            tail = text.split("'s", 1)[1].strip()
            if tail:
                source_terms.append(tail)

        for sep in ["|", ",", ";"]:
            if sep in text:
                source_terms.extend(
                    part.strip() for part in text.split(sep) if part.strip()
                )

        unique = []
        seen = set()
        for item in source_terms:
            if item not in seen:
                unique.append(item)
                seen.add(item)
        return unique

    @classmethod
    def _resolve_landmark_name(
        cls,
        subtask_landmark: Optional[str],
        fallback_sources: Optional[List[Optional[str]]] = None,
    ) -> Optional[str]:
        """Prefer the raw LLM output, then fall back to structured fields."""
        primary_landmark = cls._normalize_landmark_text(subtask_landmark)
        if primary_landmark:
            return primary_landmark

        for source in fallback_sources or []:
            for piece in cls._iter_landmark_source_terms(source):
                fallback_landmark = cls._normalize_landmark_text(piece)
                if fallback_landmark:
                    print(f"  [INFO] Fallback landmark: {fallback_landmark}")
                    return fallback_landmark

        return None

    def _set_current_landmark_tracking(
        self,
        subtask_landmark: Optional[str],
        fallback_sources: Optional[List[Optional[str]]] = None,
    ) -> None:
        """Keep only the current target landmark for each subtask."""
        previous_landmark = self._normalize_landmark_text(getattr(self, "target_landmark", None))
        self.tracked_landmark_classes.clear()
        clean_landmark = self._resolve_landmark_name(subtask_landmark, fallback_sources)

        if clean_landmark:
            self.tracked_landmark_classes.add(clean_landmark)
            self.target_landmark = clean_landmark
            self.landmark_classes = [clean_landmark]
        else:
            self.target_landmark = None
            self.landmark_classes = []
        self.classes = list(self.landmark_classes)
        if clean_landmark != previous_landmark:
            self.landmark_memory.clear_latest_detection_cache()
            self.latest_action_detection_vis = None
            self._invalidate_action_landmark_summary_cache()

    def _reset_custom_landmark_state(self) -> None:
        """Clear custom landmark classes, records, and map channels before a new subtask."""
        old_landmarks = list(getattr(self, 'landmark_classes', []) or [])

        if hasattr(self, 'mapper') and self.mapper is not None:
            self.mapper.clear_custom_landmarks()

        self.landmark_classes = []
        self.tracked_landmark_classes.clear()
        self.target_landmark = None
        self.classes = []
        self.landmark_memory.reset_subtask()

        detected = getattr(self, 'detected_classes', None)
        if detected is not None:
            for lm_name in old_landmarks:
                detected.discard(lm_name)

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

        # Remove explicit auto-turn / view prefixes and keep the action body.
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

        # If a turn explanation appears first, trim to the actual action verb.
        action_match = re.search(
            r"\b(move|go|walk|enter|pass|follow|cross|approach|continue|stop|head|climb|ascend|descend)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if action_match and action_match.start() > 0:
            prefix = cleaned[:action_match.start()]
            if re.search(r"turn|rotate|face|image|left|right|front|back", prefix, flags=re.IGNORECASE):
                cleaned = cleaned[action_match.start():].strip()

        # Remove leading direction templates while keeping the action itself.
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

        # Keep a single sentence to avoid feeding long action instructions downstream.
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
        """Keep current_waypoint compact in `space - landmark(s)` form."""
        if waypoint_text is None:
            return None

        cleaned = self._strip_visual_brackets_from_text(
            strip_space_type_variant_suffixes(str(waypoint_text)).strip()
        )
        if not cleaned:
            return cleaned

        parts = [part.strip() for part in cleaned.split("|") if part.strip()]
        if not parts:
            return cleaned

        base = parts[0].replace("’", "'").replace("`", "'").strip()
        if "'s " in base and " - " not in base:
            space_part, local_part = base.split("'s ", 1)
            space_part = space_part.strip(" ,;:-")
            local_part = local_part.strip(" ,;:-")
            if space_part and local_part:
                base = f"{space_part} - {local_part}"
        if " - " in base:
            return re.sub(r"\s+", " ", base).strip(" ,;:-")

        nearby_part = ""
        for part in parts[1:]:
            if re.match(r"(?i)^connected\b", part):
                continue
            nearby_match = re.match(r"(?i)^nearby(?:\s*\([^)]*\))?\s*:\s*(.+)$", part)
            if nearby_match:
                nearby_part = nearby_match.group(1).strip(" ,;:-")
                break

        if nearby_part:
            return re.sub(r"\s+", " ", f"{base} - {nearby_part}").strip(" ,;:-")
        return re.sub(r"\s+", " ", base).strip(" ,;:-")

    @staticmethod
    def _normalize_current_waypoint_area_token(text: Optional[str]) -> str:
        cleaned = strip_space_type_variant_suffixes(str(text or "")).strip()
        cleaned = re.sub(r"\s*\[links:[^\]]*\]\s*", " ", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("’", "'").replace("`", "'")
        cleaned = re.sub(r"[^A-Za-z0-9\s]+", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip().lower()

    @classmethod
    def _is_unresolved_current_waypoint_area(cls, text: Optional[str]) -> bool:
        normalized = cls._normalize_current_waypoint_area_token(text)
        if not normalized:
            return True
        if normalized in {
            "unknown",
            "area",
            "room",
            "space",
            "zone",
            "section",
            "place",
            "location",
            "current position",
        }:
            return True
        if re.fullmatch(r"(?:area|room|space|zone|section)\s*\d+", normalized):
            return True
        if normalized.startswith("unknown"):
            return True
        if "infer actual" in normalized or "infer the actual" in normalized:
            return True
        if "infer" in normalized and "view" in normalized:
            return True
        return False

    def _get_runtime_current_area_label(self) -> str:
        mapper = getattr(self, "mapper", None)
        if mapper is None:
            return ""
        current_area = (
            getattr(mapper, "current_space_area_display_label", "")
            or getattr(mapper, "current_space_area_label", "")
            or ""
        )
        current_area = self._strip_visual_brackets_from_text(
            strip_space_type_variant_suffixes(str(current_area).strip())
        )
        current_area = re.sub(r"\s*\[links:[^\]]*\]\s*", " ", current_area, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", current_area).strip(" ,;:-")

    def _get_runtime_current_area_terms(self) -> set:
        mapper = getattr(self, "mapper", None)
        terms = set()
        for raw_text in (
            self._get_runtime_current_area_label(),
            getattr(mapper, "current_space_area_type", "") if mapper is not None else "",
        ):
            normalized = self._normalize_current_waypoint_area_token(raw_text)
            if normalized:
                terms.add(normalized)
        return terms

    def _split_current_waypoint_local_cues(self, local_text: Optional[str]) -> List[str]:
        cues: List[str] = []
        for raw_chunk in str(local_text or "").split("/"):
            chunk = self._strip_visual_brackets_from_text(
                strip_space_type_variant_suffixes(raw_chunk)
            ).strip(" ,;:-")
            if not chunk:
                continue
            lowered = chunk.lower()
            split_for_near = False
            for prefix in ("near ", "by ", "at "):
                if lowered.startswith(prefix):
                    chunk = chunk[len(prefix):].strip(" ,;:-")
                    lowered = chunk.lower()
                    split_for_near = True
                    break
            if not chunk:
                continue
            if split_for_near:
                subparts = [
                    part.strip(" ,;:-")
                    for part in re.split(r"\s*(?:&| and |,)\s*", chunk)
                    if part.strip(" ,;:-")
                ]
                if subparts:
                    cues.extend(subparts)
                    continue
            cues.append(chunk)
        return cues

    def _align_current_waypoint_with_runtime_area(
        self,
        waypoint_text: Optional[str],
    ) -> Optional[str]:
        cleaned = self._sanitize_current_waypoint_text(waypoint_text)
        if cleaned is None:
            return None

        runtime_area = self._get_runtime_current_area_label()
        runtime_area_norm = self._normalize_current_waypoint_area_token(runtime_area)
        if not runtime_area_norm or self._is_unresolved_current_waypoint_area(runtime_area):
            return cleaned

        if " - " in cleaned:
            _space_part, local_part = cleaned.split(" - ", 1)
        else:
            local_part = ""

        area_terms = self._get_runtime_current_area_terms()
        deduped_cues: List[str] = []
        seen_norms = set()
        for cue in self._split_current_waypoint_local_cues(local_part):
            cue_norm = self._normalize_current_waypoint_area_token(cue)
            if not cue_norm or cue_norm in area_terms or cue_norm in seen_norms:
                continue
            deduped_cues.append(re.sub(r"\s+", " ", cue).strip(" ,;:-"))
            seen_norms.add(cue_norm)

        if deduped_cues:
            return f"{runtime_area} - {' / '.join(deduped_cues[:3])}"
        return runtime_area

    @staticmethod
    def _sanitize_next_waypoint_text(next_waypoint_text: Optional[str]) -> Optional[str]:
        """Keep next_waypoint in single `space's landmark` form."""
        if next_waypoint_text is None:
            return None

        cleaned = VLMNavigationController._strip_visual_brackets_from_text(
            strip_space_type_variant_suffixes(str(next_waypoint_text)).strip()
        )
        if not cleaned:
            return cleaned

        cleaned = cleaned.replace("’", "'").replace("`", "'")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
        if "'s" not in cleaned:
            return cleaned

        space_part, local_part = cleaned.split("'s", 1)
        space_part = space_part.strip(" ,;:-")
        local_part = local_part.strip(" ,;:-")
        if not space_part or not local_part:
            return cleaned

        raw_local_parts = [
            part.strip(" ,;:-")
            for part in re.split(r"\s*[\/|]\s*", local_part)
            if part.strip(" ,;:-")
        ]
        if not raw_local_parts:
            return f"{space_part}'s {local_part}".strip()

        generic_tokens = {
            "area",
            "corner",
            "end",
            "part",
            "place",
            "section",
            "side",
            "spot",
            "zone",
        }

        def _local_part_score(local_text: str) -> Tuple[int, int, int]:
            normalized = re.sub(r"[^a-z0-9\s-]", " ", local_text.lower())
            words = [word for word in normalized.split() if word]
            informative_words = [word for word in words if word not in generic_tokens]
            return (
                len(informative_words),
                len(words),
                len(local_text),
            )

        chosen_local_part = max(raw_local_parts, key=_local_part_score).strip(" ,;:-")
        chosen_local_part = re.sub(r"\s+", " ", chosen_local_part).strip()
        if not chosen_local_part:
            chosen_local_part = local_part

        return f"{space_part}'s {chosen_local_part}".strip()

    @staticmethod
    def _strip_visual_brackets_from_text(text: Optional[str]) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return cleaned
        cleaned = cleaned.replace("【", "[").replace("】", "]")
        cleaned = re.sub(r"\[\s*([^\[\]]+?)\s*\]", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _normalize_controller_response_payload(
        self,
        response: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return normalize_subtask_payload(response)

    def _sanitize_planner_response(self, response: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Normalize planner outputs while keeping the full view-prefixed instruction."""
        if not response:
            return response
        response = self._normalize_controller_response_payload(response)
        if not response:
            return response
        response = dict(response)
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
                response[key] = self._strip_visual_brackets_from_text(
                    strip_space_type_variant_suffixes(response.get(key))
                )
        response["current_waypoint"] = self._sanitize_current_waypoint_text(
            response.get("current_waypoint")
        )
        response["current_waypoint"] = self._align_current_waypoint_with_runtime_area(
            response.get("current_waypoint")
        )
        response["next_waypoint"] = self._sanitize_next_waypoint_text(
            response.get("next_waypoint")
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
            info=self._build_nav_visualizer_info(info),
            step=look_step,
            instruction=self.current_instruction,
            current_subtask=subtask_text,
            distance=distance,
            action=f"TURN_LEFT (360 scan {look_index}/12)",
            subtask_id=phase
        )
    
    def _collect_lookaround_direction_views(self, phase: str = "initial") -> Tuple[List[Any], List[str]]:
        """Run the shared lookaround scan, then render the 12 thinking views for the VLM."""
        scan_state = self._capture_lookaround_scan(
            phase=phase,
            enable_landmark_detection=False,
            prepare_thinking_detection=True,
        )
        if not scan_state:
            return [], []

        lookaround_images = scan_state.get("lookaround_images", []) or []
        lookaround_depths = scan_state.get("lookaround_depths", []) or []
        lookaround_detection_payloads = scan_state.get("lookaround_detection_payloads", []) or []
        final_map_state = scan_state.get("final_map_state")
        final_last_waypoint_angle = scan_state.get("final_last_waypoint_angle")

        waypoint_info = None
        last_waypoint_angle_deg = None
        waypoint_initial_index = None
        if phase != "initial" and final_map_state is not None:
            wp_positions = final_map_state.get('waypoint_positions', [])
            wp_ids = final_map_state.get('waypoint_ids', [])
            if wp_positions:
                if final_last_waypoint_angle is not None:
                    last_waypoint_angle_deg = np.degrees(final_last_waypoint_angle)
                _, orig_wp_ids, wp_descriptions = self.mapper.get_waypoints()
                waypoint_info = (wp_positions, wp_ids, wp_descriptions)
                waypoint_initial_index = final_map_state.get('waypoint_initial_index')
        
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
                hfov=self.config.SPACE.SENSOR.HFOV_DEG,
                landmark_dist_map=None,
                landmark_dist_map_multi=None,
                show_action_partitions=False,
                append_bottom_strip=False,
                controller=None,
                return_visible_entries=True,
            )

        rendered_views = self.thinking_view_renderer.render_direction_views(
            phase=phase,
            lookaround_images=lookaround_images,
            lookaround_depths=lookaround_depths,
            lookaround_detection_payloads=lookaround_detection_payloads,
            landmark_classes=self.landmark_classes,
            detect_landmarks_fn=self._detect_landmarks_for_visualization,
            render_detection_fn=_render_thinking_detection,
            draw_distance_fn=self.visualizer.draw_distance_on_view,
            distance_lookup=self.latest_obstacle_distances_12,
            waypoint_info=waypoint_info,
            waypoint_area_labels=(final_map_state or {}).get('waypoint_area_labels', []),
            waypoint_floor_ids=(final_map_state or {}).get('waypoint_floor_ids', []),
            current_pose=(final_map_state or {}).get('full_pose'),
            resolution_cm=float(getattr(self.mapper, 'resolution', 5)),
            current_space_area_label=str((final_map_state or {}).get('current_space_area_label', 'Unknown') or 'Unknown'),
            full_map=(final_map_state or {}).get('full_map'),
            crop_offset=(final_map_state or {}).get('crop_offset'),
            waypoint_angle_deg=last_waypoint_angle_deg,
            draw_waypoints_fn=self._draw_waypoints_on_view,
            current_floor_id=int((final_map_state or {}).get('current_floor_id', 0) or 0),
            initial_waypoint_index=waypoint_initial_index,
        )

        direction_inputs: List[Any] = []
        direction_names: List[str] = []
        for view in rendered_views:
            angle = int(view.get("angle", 0))
            direction_names.append(str(view.get("direction_name", "")))
            direction_inputs.append({
                "image_array": view.get("image"),
                "color_space": "bgr",
                "artifact_name": f"direction_{angle:03d}.jpg",
                "name": f"direction_{angle:03d}",
            })

        return direction_inputs, direction_names

    def run_lookaround_and_update_state(self, phase: str) -> Dict[str, Any]:
        """Unified lookaround entry used by initial / verify thinking cycles."""
        direction_paths, direction_names = self._collect_lookaround_direction_views(phase)
        return {
            "phase": str(phase),
            "direction_paths": direction_paths,
            "direction_names": direction_names,
            "global_map_input": getattr(self, 'latest_global_map_input', None),
            "local_map_path": getattr(self, 'latest_local_map', None),
        }

    def _collect_thinking_detected_landmarks(self) -> List[str]:
        """Keep thinking-side landmark text aligned with the action-side locked top-k entries."""
        latest_entries = self._get_latest_action_local_map_landmark_entries()
        if latest_entries:
            names = []
            seen = set()
            for entry in latest_entries:
                name = str(entry.get("name") or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                names.append(name)
            if names:
                return names
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
            thinking_dir = self.save_manager.thinking_subtask_dir(1)
            print(f"\n[LLM] Planning...")
        else:
            if not self.current_subtask:
                self.latest_thinking_cycle_info = {"mode": mode_key, "reason": "missing_current_subtask"}
                return None, None, None
            attempt_letter = self._current_attempt_letter()
            phase = self._current_verify_phase()
            thinking_dir = self.save_manager.thinking_subtask_dir(self.subtask_count + 1)
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

        global_map = lookaround_state.get("global_map_input")
        if isinstance(global_map, str):
            global_map_missing = (not global_map or not os.path.exists(global_map))
        else:
            global_map_missing = global_map is None
        if global_map_missing:
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
        verify_view_restriction_info: Dict[str, Any] = {}
        if mode_key == "verify":
            image_paths, direction_names, verify_view_restriction_info = self._consume_pending_verify_view_restriction(
                image_paths,
                direction_names,
            )

        detected_landmarks: List[str] = []
        waypoint_summary: Optional[str] = None
        previous_subtask_landmark_summary: Optional[str] = None
        planner_timing_info_chunks: List[Dict[str, Any]] = []

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
            planner_timing_info_chunks.append(
                dict(getattr(self.planner, "last_call_timing_info", {}) or {})
            )
        else:
            detected_landmarks = self._collect_thinking_detected_landmarks()
            waypoint_summary = self._get_waypoint_summary(include_area_chain=True)
            previous_subtask_landmark_summary = self._build_previous_subtask_landmark_summary()
            verify_subtask = dict(self.current_subtask or {})
            verify_subtask["subtask_instruction"] = self._build_previous_subtask_instruction_summary(
                self.current_subtask
            )
            verify_replan_prompt_notice = str(getattr(self, 'verify_replan_prompt_notice', '') or '').strip()
            response, prompt = self.planner.verify_and_replan(
                instruction=self.current_instruction,
                current_subtask=verify_subtask,
                observation_images=image_paths,
                direction_names=direction_names,
                global_map_image=global_map,
                local_map_image=None,
                detected_landmarks=detected_landmarks,
                waypoint_summary=waypoint_summary,
                previous_subtask_landmark_summary=previous_subtask_landmark_summary,
                obstacle_distances=obstacle_distances,
                verify_replan_prompt_notice=verify_replan_prompt_notice,
                save_dir=thinking_dir,
            )
            planner_timing_info_chunks.append(
                dict(getattr(self.planner, "last_call_timing_info", {}) or {})
            )
            if (
                response
                and bool(response.get("global_task_finish", False))
                and self._is_in_initial_position_neighborhood(waypoint_summary)
            ):
                print(
                    "  [WARN] Verify returned global_task_finish=true while still at/near INITIAL POSITION; "
                    "reject and re-query once"
                )
                fallback_response = dict(response)
                retry_notice = self._merge_prompt_notices(
                    verify_replan_prompt_notice,
                    self._build_initial_position_finish_guard_notice(),
                )
                retry_response, retry_prompt = self.planner.verify_and_replan(
                    instruction=self.current_instruction,
                    current_subtask=verify_subtask,
                    observation_images=image_paths,
                    direction_names=direction_names,
                    global_map_image=global_map,
                    local_map_image=None,
                    detected_landmarks=detected_landmarks,
                    waypoint_summary=waypoint_summary,
                    previous_subtask_landmark_summary=previous_subtask_landmark_summary,
                    obstacle_distances=obstacle_distances,
                    verify_replan_prompt_notice=retry_notice,
                    save_dir=thinking_dir,
                )
                planner_timing_info_chunks.append(
                    dict(getattr(self.planner, "last_call_timing_info", {}) or {})
                )
                if retry_response:
                    response = retry_response
                    prompt = retry_prompt
                else:
                    response = fallback_response

                if (
                    response
                    and bool(response.get("global_task_finish", False))
                    and self._is_in_initial_position_neighborhood(waypoint_summary)
                ):
                    print(
                        "  [WARN] Still at/near INITIAL POSITION after re-query; "
                        "force global_task_finish=false"
                    )
                    response = dict(response)
                    response["global_task_finish"] = False
            self.verify_replan_prompt_notice = ""
        planner_timing_info = self._merge_planner_timing_infos(*planner_timing_info_chunks)
        planner_timing_records = list(planner_timing_info.get("records", []) or [])
        for timing_record in planner_timing_records:
            is_success = bool(timing_record.get("success", False))
            self.timing_tracker.record_thinking_call(
                mode=mode_key,
                phase=phase,
                step=int(getattr(self, "current_step", 0) or 0),
                subtask_count=int(getattr(self, "subtask_count", 0) or 0),
                subtask_attempt=int(getattr(self, "subtask_attempt", 0) or 0),
                duration_s=float(timing_record.get("duration_s", 0.0) or 0.0),
                success=is_success,
                next_waypoint=self._get_next_waypoint_field(response) if is_success and response else "",
            )
        if planner_timing_records:
            self.timing_tracker.add_failed_retry_wait(
                float(planner_timing_info.get("failed_retry_wait_duration_s", 0.0) or 0.0)
            )

        if not response:
            planner_failure_reason = str(
                planner_timing_info.get("final_failure_reason", "") or "planner_failed"
            ).strip() or "planner_failed"
            self.latest_thinking_cycle_info = {
                "mode": mode_key,
                "phase": phase,
                "thinking_dir": thinking_dir,
                "reason": planner_failure_reason,
                "detected_landmarks": detected_landmarks,
                "waypoint_summary": waypoint_summary,
                "previous_subtask_landmark_summary": previous_subtask_landmark_summary,
                "previous_subtask_landmark_final_info": dict(getattr(self, "previous_subtask_landmark_final_info", {}) or {}),
                "planner_failure_reason": planner_failure_reason,
                "planner_timing_info": dict(planner_timing_info or {}),
            }
            if mode_key == "initial":
                print("[ERR] LLM Planning failed")
            else:
                print("[ERR] LLM Verify failed")
            return None, prompt, dict(self.latest_thinking_cycle_info)

        raw_response = self._get_latest_planner_raw_response_payload(response)
        response = self._sanitize_planner_response(response)
        self._persist_response_artifacts(
            save_dir=thinking_dir,
            raw_payload=raw_response,
        )

        cycle_info = {
            "mode": mode_key,
            "phase": phase,
            "thinking_dir": thinking_dir,
            "detected_landmarks": detected_landmarks,
            "waypoint_summary": waypoint_summary,
            "previous_subtask_landmark_summary": previous_subtask_landmark_summary,
            "previous_subtask_landmark_final_info": dict(getattr(self, "previous_subtask_landmark_final_info", {}) or {}),
            "verify_view_restriction": verify_view_restriction_info,
            "image_paths": list(image_paths or []),
            "direction_names": list(direction_names or []),
            "global_map_image": global_map,
        }
        self.latest_thinking_cycle_info = dict(cycle_info)
        return response, prompt, cycle_info

    def _reset_post_thinking_action_state(self) -> None:
        """Reset per-subtask action state before handing control back to action."""
        if self.mapper is not None:
            self.mapper.clear_trajectory()
        self._reset_custom_landmark_state()
        self.progress_summary = ""
        self.action_stagnation_streak = 0
        self._reset_blocked_front_controller_recovery_state()
        self._clear_action_stagnation_prompt_state()
        self.action_consecutive_turn_count = 0
        self._clear_action_force_forward_prompt_state()
        self.verify_replan_prompt_notice = ""
        self.pose_before_action = None
        self.last_planned_degrees = 0
        self.last_planned_meters = 0
        self.last_action_name = ""
        self.last_action_effective_name = ""
        self.previous_subtask_landmark_final_info = None
        self.previous_subtask_autocomplete_landmark_info = None

    def _record_current_position_from_thinking_response(self, response: Dict[str, Any]) -> None:
        """Store the planner-localized position for later verification/debug use."""
        self.current_position_info = {
            'waypoint': response.get('current_waypoint', 'Unknown'),
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
        if is_initial and task_finished:
            print(
                "  [WARN] Initial planning returned global_task_finish=true; "
                "force global_task_finish=false"
            )
            response['global_task_finish'] = False
            task_finished = False
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
                f"| streak={match_streak}/{self.runtime_options.final_destination_match_autostop_streak} "
                f"| anchor_distance={float(anchor_distance_m or 0.0):.2f}/{self.runtime_options.final_destination_match_autostop_radius_m:.2f}m"
            )
        elif match_hit and restarted_by_anchor_drift:
            print(
                "[GoalRegionMatch] "
                f"waypoint_chain tail='{last_chain_node}' still matches destination='{next_destination}', "
                f"but the agent moved {float(anchor_distance_m or 0.0):.2f}m away from the first matched pose "
                f"(limit {self.runtime_options.final_destination_match_autostop_radius_m:.2f}m); restart stable-goal streak from 1"
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
            not is_initial and
            not task_finished and
            match_hit and
            stayed_inside_anchor_region and
            match_streak >= int(self.runtime_options.final_destination_match_autostop_streak)
        )
        if auto_finish_by_streak:
            task_finished = True
            response['global_task_finish'] = True
            response['auto_task_finish_by_destination_streak'] = True
            response['auto_task_finish_by_goal_region_stability'] = True
            print(
                "[AutoTaskComplete] "
                f"final waypoint tail matched next destination for {match_streak} consecutive thinking cycles "
                f"and all matched poses stayed within {self.runtime_options.final_destination_match_autostop_radius_m:.2f}m of the first matched pose; "
                "stop the task even though the planner did not set global_task_finish."
            )

        if not is_initial:
            attempt_letter = self._current_attempt_letter()
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

        subtask_landmark = self._get_subtask_landmark_field(response)
        self._set_current_landmark_tracking(subtask_landmark)

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

    def _apply_postplanning_space_area_update(
        self,
        response: Dict[str, Any],
        phase: str,
        thinking_dir: Optional[str] = None,
        refresh_direction_views: bool = True,
    ) -> Optional[int]:
        """Persist planner space-area output into the world map only."""
        _ = phase
        _ = thinking_dir
        _ = refresh_direction_views
        if self.mapper is None:
            return None

        waypoint_desc = response.get('current_waypoint', 'Unknown location')
        waypoint_id = self.mapper.add_waypoint(waypoint_desc)
        self._save_waypoint_area_memory_snapshot()
        return waypoint_id

    def auto_rotate_to_waypoint(self, waypoint_direction: str) -> Tuple[bool, List[Dict]]:
        """
        Parse a waypoint direction string into a rotation-action sequence.

        Args:
            waypoint_direction: Example: `IMAGE 5 (Left 120deg)`.

        Returns:
            `(success, action_sequence)` where each action entry looks like
            `{"action": "TURN_LEFT/RIGHT", "degrees": 30}`.
        """
        import re

        waypoint_direction = str(waypoint_direction or "").strip()
        match = re.search(
            r'Left (\d+)(?:deg|°)|Right (\d+)(?:deg|°)|Back (\d+)(?:deg|°)?|Front',
            waypoint_direction,
        )

        angle = 0
        direction = None

        if match:
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
            image_match = re.search(r'IMAGE\s*(\d+)', waypoint_direction, re.IGNORECASE)
            image_index = None
            if image_match:
                try:
                    image_index = int(image_match.group(1))
                except (TypeError, ValueError):
                    image_index = None
            if image_index is not None:
                image_to_angle = {
                    1: 0,
                    2: 30,
                    3: 60,
                    4: 90,
                    5: 120,
                    6: 150,
                    7: 180,
                    8: -150,
                    9: -120,
                    10: -90,
                    11: -60,
                    12: -30,
                }
                signed_angle = image_to_angle.get(image_index)
                if signed_angle is not None:
                    if signed_angle == 0:
                        return True, []
                    direction = 'LEFT' if signed_angle > 0 else 'RIGHT'
                    angle = abs(int(signed_angle))

        if direction is None or angle < 0:
            print(f"  [WARN] Cannot parse waypoint_direction: {waypoint_direction}")
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
        """Execute a rotation sequence via the shared executor pipeline."""
        for i, action_dict in enumerate(action_sequence):
            action_name = action_dict['action']
            
            if action_name == 'TURN_LEFT':
                action_id = resolve_habitat_action("TURN_LEFT")
            elif action_name == 'TURN_RIGHT':
                action_id = resolve_habitat_action("TURN_RIGHT")
            else:
                print(f"    [WARN] Unknown action: {action_name}")
                continue
            
            result = self.step_with_vlm(
                action_id,
                action_name,
                save_vis=True,
                enable_landmark_detection=False,
            )
            
            if result.get('done', False):
                print(f"    [WARN] Episode ended during rotation")
                return False
        
        return True

    def _ensure_action_observation_ready(self) -> bool:
        """Ensure action planning has a current observation without duplicating refresh logic."""
        if self.latest_obs is not None:
            return True

        refresh_steps = (
            (resolve_habitat_action("TURN_RIGHT"), "OBS_REFRESH_TURN_RIGHT", "turn-right"),
            (resolve_habitat_action("TURN_LEFT"), "OBS_REFRESH_TURN_LEFT", "turn-left"),
        )
        for action_id, action_name, warning_suffix in refresh_steps:
            refresh_result = self.step_with_vlm(
                action_id,
                action_name=action_name,
                save_vis=True,
                enable_landmark_detection=False,
            )
            if refresh_result.get("done", False):
                print(f"[WARN] Episode ended during observation refresh {warning_suffix}")
                return False
        return self.latest_obs is not None

    def _build_action_detected_landmarks_text(self, detection_step: Optional[int]) -> str:
        """Build the compact action-prompt landmark text from the current detection step."""
        if detection_step is not None:
            step_landmarks = self.landmark_memory.get_step_detected(detection_step)
            if step_landmarks:
                return ", ".join([name for name, _ in step_landmarks])

        if getattr(self, "target_landmark", None):
            return f"No {self.target_landmark} detected in current view"
        return "No landmarks detected"

    def _build_action_detection_image_input(self, last_step: int) -> Optional[Dict[str, Any]]:
        """Prefer the detection render for action input and fall back to raw RGB only if needed."""
        detection_vis = getattr(self, "latest_action_detection_vis", None)
        if detection_vis is not None:
            return {
                "image_array": detection_vis,
                "color_space": "bgr",
                "artifact_name": "action_view.jpg",
                "name": f"action_view_step_{last_step:04d}",
            }

        if self.latest_obs is not None and "rgb" in self.latest_obs:
            rgb_bgr = cv2.cvtColor(self.latest_obs["rgb"], cv2.COLOR_RGB2BGR)
            rgb_bgr = resize_image_to_width(rgb_bgr, ACTION_VIEW_MODEL_CONTENT_WIDTH)
            return {
                "image_array": rgb_bgr,
                "color_space": "bgr",
                "artifact_name": "action_view.jpg",
                "name": f"action_view_step_{last_step:04d}",
            }

        print(f"  [WARN] Action input image not available for step {last_step}")
        return None

    def _write_action_subtask_info_if_needed(self, subtask_id: str) -> None:
        """Persist one action-subtask metadata file when a subtask starts running."""
        info_file = self.save_manager.action_info_path(subtask_id)
        if os.path.exists(info_file):
            return

        subtask_info = {
            "subtask_id": self.subtask_count,
            "next_waypoint": self._get_next_waypoint_field(self.current_subtask),
            "subtask_instruction": self.current_subtask.get("subtask_instruction", ""),
            "subtask_landmark": self._get_subtask_landmark_field(self.current_subtask),
            "start_step": self.current_step,
            "timestamp": datetime.now().isoformat(),
        }
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(subtask_info, f, ensure_ascii=False, indent=2)

    def _build_action_decision_context(self) -> Optional[Dict[str, Any]]:
        """Collect the action-LLM inputs in one place so the execution path stays modular."""
        if not self._ensure_action_observation_ready():
            return None

        last_step = self.current_step
        action_phase = self._current_action_phase()
        self._run_pre_action_detection_snapshot(action_phase)

        detection_step = last_step
        step_landmark_entries = self._get_action_landmark_prompt_entries(detection_step)
        detected_landmarks = self._build_action_detected_landmarks_text(detection_step)
        obstacle_distances = getattr(
            self,
            "latest_obstacle_distances",
            {"front": "Unknown", "left_30": "Unknown", "right_30": "Unknown"},
        )

        subtask_id = self._current_subtask_run_id()
        action_save_dir = self.save_manager.action_step_dir(
            subtask_id,
            self.current_step + 1,
            create=True,
        )
        self._write_action_subtask_info_if_needed(subtask_id)

        detection_image = self._build_action_detection_image_input(last_step)
        action_landmark_map_info = build_action_landmark_map_info(
            step_landmark_entries=step_landmark_entries,
            landmark_dist_map=self.landmark_memory.get_latest_dist_map(),
            landmark_dist_map_multi=self.landmark_memory.get_latest_dist_map_multi(),
            landmark_instances_world=self.landmark_memory.get_world_instances(),
        )
        self._log_action_landmark_debug("pre-action", step_landmark_entries)

        return {
            "step_landmark_entries": step_landmark_entries,
            "detected_landmarks": detected_landmarks,
            "obstacle_distances": obstacle_distances,
            "action_save_dir": action_save_dir,
            "detection_image": detection_image,
            "waypoint_summary": "",
            "action_landmark_map_info": action_landmark_map_info,
        }

    def _request_vlm_action(
        self,
        action_context: Dict[str, Any],
        action_subtask_instruction: str,
        progress_summary_for_prompt: str,
        allowed_action_names: Optional[Sequence[str]],
    ) -> Tuple[Optional[int], Optional[str], Optional[Dict[str, Any]], int, float]:
        """Run the action VLM once, including blocked-front controller recovery."""
        step_landmark_entries = action_context["step_landmark_entries"]
        max_blocked_forward_requeries = 1

        action_id: Optional[int] = None
        action_name: Optional[str] = None
        response: Optional[Dict[str, Any]] = None
        degrees = 0
        meters = 0.0

        for blocked_retry_idx in range(max_blocked_forward_requeries):
            action_api_start_time = time.perf_counter()
            (
                action_id,
                action_name,
                response,
                degrees,
                meters,
                _prompt,
            ) = self.action_executor.decide_action(
                next_waypoint=self._get_next_waypoint_field(self.current_subtask),
                subtask_instruction=action_subtask_instruction,
                subtask_landmark=self._get_subtask_landmark_field(self.current_subtask),
                first_person_image=action_context["detection_image"] or "",
                action_mapping=ACTION_MAPPING,
                progress_summary=progress_summary_for_prompt,
                waypoint_summary=action_context["waypoint_summary"],
                detection_image=action_context["detection_image"],
                detected_landmarks=action_context["detected_landmarks"],
                obstacle_distances=action_context["obstacle_distances"],
                landmark_map_info=action_context["action_landmark_map_info"],
                allowed_action_names=allowed_action_names,
                save_dir=action_context["action_save_dir"],
            )
            self.timing_tracker.record_action_call(
                step=int(getattr(self, "current_step", 0) or 0),
                subtask_count=int(getattr(self, "subtask_count", 0) or 0),
                subtask_attempt=int(getattr(self, "subtask_attempt", 0) or 0),
                duration_s=time.perf_counter() - action_api_start_time,
                success=bool(action_id is not None),
                action_name=action_name,
            )

            if (
                self.action_force_forward_after_turns_pending and
                (not self.action_stagnation_retry_pending) and
                str(action_name or "").upper() in ("TURN_LEFT", "TURN_RIGHT")
            ):
                print(
                    "[ActionTurnLimit] Rejected extra turn after consecutive-turn limit; "
                    "controller-side forward recovery will take over from the same current view"
                )
                forced_forward = self._build_forced_forward_after_turn_limit_action(
                    step_landmark_entries=step_landmark_entries,
                    obstacle_distances=action_context.get("obstacle_distances"),
                )
                if forced_forward is None:
                    return None, None, None, 0, 0.0
                action_id = forced_forward["action_id"]
                action_name = forced_forward["action_name"]
                response = forced_forward["response"]
                degrees = int(forced_forward.get("degrees", 0) or 0)
                meters = float(forced_forward.get("meters", 0.0) or 0.0)
                if str(action_name or "").upper() == "MOVE_FORWARD":
                    print("[ActionTurnLimit] Controller forced one short MOVE_FORWARD after consecutive turns")
                else:
                    print(
                        "[ActionTurnLimit] FRONT stayed blocked after consecutive turns; "
                        "end the current action stage and trigger replan"
                    )
                break

            if not (
                self.action_stagnation_retry_pending and
                str(action_name or "").upper() == "MOVE_FORWARD"
            ):
                break

            print(
                "[ActionStagnation] Rejected MOVE_FORWARD after blocked-front warning; "
                "controller-side recovery will take over from the same current view"
            )
            action_id = None
            action_name = None
            if blocked_retry_idx < max_blocked_forward_requeries - 1:
                continue

            print("[ERR] Action VLM kept choosing forbidden MOVE_FORWARD after a blocked-front warning")
            forced_recovery = self._build_forced_blocked_front_recovery_action(
                step_landmark_entries=step_landmark_entries,
            )
            if forced_recovery is None:
                return None, None, None, 0, 0.0

            action_id = forced_recovery["action_id"]
            action_name = forced_recovery["action_name"]
            response = forced_recovery["response"]
            degrees = int(forced_recovery.get("degrees", 0) or 0)
            meters = float(forced_recovery.get("meters", 0.0) or 0.0)
            if str(action_name or "").upper() in ("TURN_LEFT", "TURN_RIGHT"):
                self.blocked_front_controller_recovery_count = int(
                    getattr(self, "blocked_front_controller_recovery_count", 0) or 0
                ) + 1
                print(
                    "[ActionStagnation] Controller forced "
                    f"{action_name} after forbidden blocked-front retries "
                    f"(recovery #{self.blocked_front_controller_recovery_count})"
                )
            else:
                self._reset_blocked_front_controller_recovery_state()
                print("[ActionStagnation] Controller forced STOP because the destination is already reached")
            break

        return action_id, action_name, response, degrees, meters

    def _check_post_action_landmark_autocomplete(
        self,
        action_phase: str,
    ) -> Optional[Dict[str, Any]]:
        """Refresh post-step landmark state once and run auto-stop on that same state."""
        self._refresh_post_action_landmark_detection_state(action_phase)
        step_landmark_entries = self._get_current_action_step_landmark_entries()
        auto_completed_subtask = self._should_autocomplete_subtask_during_action_step(
            step_landmark_entries
        )
        self._log_action_landmark_debug("post-action", step_landmark_entries)
        if auto_completed_subtask is None:
            return None

        self._record_previous_subtask_autocomplete_landmark(auto_completed_subtask)
        landmark_kind = "opening-like" if auto_completed_subtask.get("is_opening_like") else "solid"
        landmark_source = "vis" if str(auto_completed_subtask.get("source", "mem") or "mem") == "vis" else "mem"
        landmark_display_id = self._safe_int(auto_completed_subtask.get("display_id"))
        landmark_id_text = f" #{landmark_display_id}" if landmark_display_id is not None and landmark_display_id > 0 else ""
        print(
            "[AutoSubtaskComplete] "
            f"{landmark_source} {auto_completed_subtask['name']}{landmark_id_text} ({landmark_kind}) reached within "
            f"{auto_completed_subtask['distance_m']:.2f}m "
            f"(threshold {float(auto_completed_subtask.get('stop_distance_m', self.ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M)):.2f}m) "
            f"on action step {self.current_step}; "
            "return control to the thinking controller."
        )
        return auto_completed_subtask
    
    def execute_action_with_vlm(self) -> Tuple[Optional[int], Optional[str], bool, int, Optional[Dict]]:
        """Query the VLM for the next action and execute it."""
        if not self.action_executor or not self.current_subtask:
            return None, None, True, 1, None

        if self._episode_done_cached():
            print("[WARN] Episode already done, skip action decision")
            return None, None, True, 1, None

        action_context = self._build_action_decision_context()
        if action_context is None:
            return None, None, True, 1, None

        action_subtask_instruction = self._sanitize_subtask_instruction_text(
            self.current_subtask.get('subtask_instruction', ''),
            self._get_next_waypoint_field(self.current_subtask),
            self.current_subtask.get('next_waypoint_direction', ''),
            keep_view_prefix=False,
        )

        progress_summary_for_prompt = self._get_action_progress_summary_for_prompt()
        force_forward_after_turns_pending = bool(
            getattr(self, "action_force_forward_after_turns_pending", False)
        ) and (not self.action_stagnation_retry_pending)
        allowed_action_names = (
            ("TURN_LEFT", "TURN_RIGHT", "STOP")
            if self.action_stagnation_retry_pending
            else ("MOVE_FORWARD", "STOP")
            if force_forward_after_turns_pending
            else None
        )
        allowed_action_names = self._apply_immediate_reverse_turn_guard(allowed_action_names)
        if (
            force_forward_after_turns_pending and
            self._is_obstacle_distance_blocked((action_context.get("obstacle_distances") or {}).get("front"))
        ):
            forced_forward = self._build_forced_forward_after_turn_limit_action(
                step_landmark_entries=action_context.get("step_landmark_entries"),
                obstacle_distances=action_context.get("obstacle_distances"),
            )
            if forced_forward is None:
                print("[ERR] Forced forward-after-turn-limit recovery failed")
                return None, None, True, 1, None
            action_id = forced_forward["action_id"]
            action_name = forced_forward["action_name"]
            response = forced_forward["response"]
            degrees = int(forced_forward.get("degrees", 0) or 0)
            meters = float(forced_forward.get("meters", 0.0) or 0.0)
            if str(action_name or "").upper() == "STOP":
                print(
                    "[ActionTurnLimit] "
                    "Consecutive-turn limit is active and FRONT is blocked, so controller ended the current action stage"
                )
            else:
                print(
                    "[ActionTurnLimit] "
                    "Controller forced one short MOVE_FORWARD after consecutive turns"
                )
        else:
            action_id, action_name, response, degrees, meters = self._request_vlm_action(
                action_context=action_context,
                action_subtask_instruction=action_subtask_instruction,
                progress_summary_for_prompt=progress_summary_for_prompt,
                allowed_action_names=allowed_action_names,
            )

        if action_id is None:
            print("[ERR] VLM decision failed")
            return None, None, True, 1, None
        
        self._persist_response_artifacts(
            save_dir=action_context["action_save_dir"],
            raw_payload=self._get_latest_action_raw_response_payload(response),
        )
        
        # Save planned action parameters for later actual-progress estimation.
        self.last_planned_degrees = degrees
        self.last_planned_meters = meters
        self.last_action_name = action_name
        self.last_action_semantic_variant = str(
            self.action_executor._extract_action_variant((response or {}).get("action"))
            or ""
        )
        self._update_action_consecutive_turn_state(action_name)

        if self.action_stagnation_retry_pending and str(action_name or "").upper() in ("TURN_LEFT", "TURN_RIGHT"):
            self._clear_action_stagnation_prompt_state()
        if str(action_name or "").upper() == "STOP":
            self._reset_blocked_front_controller_recovery_state()
            self._clear_action_stagnation_prompt_state()
        
        # Check whether the planner requested stop.
        should_stop = (action_name == "STOP")
        if should_stop:
                self.last_action_progress_hint = self._build_last_action_progress_hint(
                    action_name=action_name,
                )
                self.last_action_effective_name = str(action_name or "").upper()
        
        # Convert the planned macro-action into repeated low-level actions.
        repeat_count = 1
        if action_name == 'TURN_LEFT' or action_name == 'TURN_RIGHT':
            # Each turn is 30 degrees.
            if degrees > 0:
                repeat_count = max(1, round(degrees / self.action_executor.turn_angle))
        elif action_name == 'MOVE_FORWARD':
            # Each forward step is 0.25 m.
            if meters > 0:
                repeat_count = max(1, round(meters / self.action_executor.move_distance))
        
        return action_id, action_name, should_stop, repeat_count, response
    
    def step_with_vlm(self, action: int, action_name: str = "", save_vis: bool = True,
                      enable_landmark_detection: bool = False) -> Dict[str, Any]:
        """
        Execute one VLM action, call the shared `step`, and cache observations.

        Args:
            action: Habitat action id.
            action_name: Human-readable action name for visualization.
            save_vis: Whether to save visualization artifacts.
            enable_landmark_detection: Whether to detect landmarks on the current
                frame and write them into the action-local map. This stays off by
                default and is enabled only for the action-LLM-facing snapshot.

        Returns:
            Step result dictionary.
        """
        # Build a phase tag such as `action1a` or `action2b`.
        phase = self._current_action_phase()
        
        result = self.step(
            action,
            save_vis,
            phase,
            enable_landmark_detection=enable_landmark_detection,
        )
        # Cache latest observations / info for the next VLM decision and visualization.
        self.latest_obs = result.get('obs', None)
        self.latest_info = result.get('info', None)
        
        # The map is updated; recompute obstacle distance immediately.
        self._update_obstacle_distances()
        
        # Save RGB + top-down composite visualization.
        if save_vis and self.nav_visualizer and self.latest_obs is not None:
            subtask_text = None
            if self.current_subtask:
                subtask_text = self.current_subtask.get('subtask_instruction', '')
            
            distance = 0.0
            if self.latest_info:
                distance = self.latest_info.get('distance_to_goal', 0.0)
            
            subtask_id = self._current_subtask_run_id()
            
            self.nav_visualizer.save_step_visualization(
                observations=self.latest_obs,
                info=self._build_nav_visualizer_info(self.latest_info),
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
                    time.sleep(wait)

            if action_id is None:
                if self.action_stagnation_retry_pending:
                    print(
                        "[ActionStagnation] Blocked-front recovery could not get a valid side action; "
                        "end the current action stage and trigger replan"
                    )
                    self.action_stagnation_streak = 0
                    self._reset_blocked_front_controller_recovery_state()
                    if self.runtime_options.enable_auto_retreat:
                        actual_retreat_m, episode_done = self._execute_auto_retreat(
                            retreat_distance_m=self.STUCK_RETREAT_DISTANCE_M
                        )
                        self.verify_replan_prompt_notice = self._build_stagnation_verify_notice(
                            actual_retreat_m=actual_retreat_m,
                            retreat_distance_m=self.STUCK_RETREAT_DISTANCE_M,
                        )
                        self.pending_verify_view_restriction = {
                            "forbidden_view_ids": list(self.STUCK_RETREAT_FORBIDDEN_VIEW_IDS),
                        }
                        self._clear_action_stagnation_prompt_state()
                        if episode_done:
                            return 'complete'
                    else:
                        self._append_progress_note(
                            "front route stayed blocked after repeated blocked-front retries, so ended the current action stage and triggered replan"
                        )
                        self._clear_action_stagnation_prompt_state()
                    return 'thinking'

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
            action_phase = self._current_action_phase()

            for i in range(repeat_count):
                if self._should_hold_last_episode_step_for_stop(action_name):
                    print(
                        "[Budget] Hold the final remaining env step for STOP evaluation; "
                        "stop executing more low-level actions and finalize."
                    )
                    return 'thinking'

                pose_before_low_level = self._get_agent_pose()
                result = self.step_with_vlm(
                    action_id,
                    action_name=action_name,
                    save_vis=True,
                    enable_landmark_detection=False,
                )
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

                if self._attempt_goal_distance_autostop():
                    return 'complete'

                if result['done']:
                    print('[WARN] Episode done (Habitat)')
                    return 'complete'

                auto_completed_subtask = self._check_post_action_landmark_autocomplete(action_phase)
                if auto_completed_subtask is not None:
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

                should_record_progress = not (
                    str(actual_action_name or "").upper() == 'MOVE_FORWARD'
                    and float(actual_meters) <= float(stagnation_threshold_m)
                )
                if should_record_progress:
                    self.progress_summary = self.action_executor._generate_progress_update(
                        current_progress=self.progress_summary,
                        action_name=actual_action_name,
                        degrees=self.last_planned_degrees,
                        meters=self.last_planned_meters,
                        actual_degrees=actual_degrees,
                            actual_meters=actual_meters
                    )

                self.last_action_progress_hint = self._build_last_action_progress_hint(
                    action_name=actual_action_name,
                    action_variant=getattr(self, "last_action_semantic_variant", ""),
                    planned_meters=self.last_planned_meters,
                    actual_meters=actual_meters,
                    actual_degrees=actual_degrees,
                )
                self.last_action_effective_name = str(actual_action_name or "").upper()

                self.pose_before_action = pose_after_action_batch

            if auto_completed_subtask is not None:
                landmark_kind = 'opening-like' if auto_completed_subtask.get('is_opening_like') else 'solid'
                self._append_progress_note(
                    f"had reached {auto_completed_subtask['name']} ({landmark_kind}) within "
                    f"{auto_completed_subtask['distance_m']:.2f}m "
                    f"(auto-stop threshold {float(auto_completed_subtask.get('stop_distance_m', self.ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M)):.2f}m), "
                    'so ended the current subtask and triggered replan'
                )
                return 'thinking'

            if replan_for_stagnation:
                latest_actual_meters = (
                    float(stagnation_actual_meters)
                    if stagnation_actual_meters is not None
                    else 0.0
                )
                self.action_stagnation_streak = 0
                self.action_stagnation_retry_pending = True
                self.action_stagnation_progress_warning_text = "(warning: front route blocked; forced stop)"
                self.action_stagnation_retry_notice_text = self._build_action_stagnation_retry_notice(
                    latest_actual_meters=latest_actual_meters,
                    stagnation_threshold_m=stagnation_threshold_m,
                )
                print(
                    "[ActionStagnation] Blocked low-level forward step: "
                    f"latest actual movement {latest_actual_meters:.2f}m <= {stagnation_threshold_m:.2f}m | "
                    "stop the current forward action and re-query action prompt with a forced side-turn warning"
                )
                continue

            if force_replan_after_action:
                print(f'\n[Replan] Force replan after {max_subtask_steps} steps')
                return 'thinking'

    def run_vlm_navigation(self, max_subtask_steps: int = 5) -> Dict[str, Any]:
        """Run the top-level scheduler over the thinking controller and action controller."""
        max_steps = self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.timing_tracker.reset()

        print("\n" + "=" * 60)
        print(f"VLM Navigation | max_steps={max_steps} | subtask_steps={max_subtask_steps}")
        print(f"Instruction: {self.current_instruction}")
        print(f"{'=' * 60}")

        controller_mode = 'thinking'
        thinking_mode = 'initial'
        navigation_complete = False
        failure_reason = ""

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
                        if cycle_reason == 'lookaround_failed':
                            failure_reason = 'initial_lookaround_failed'
                        elif cycle_reason == 'planner_timeout':
                            failure_reason = 'initial_planner_timeout'
                        else:
                            failure_reason = 'initial_subtask_failed'
                    else:
                        if cycle_reason == 'lookaround_failed':
                            failure_reason = 'verify_lookaround_failed'
                        elif cycle_reason == 'planner_timeout':
                            failure_reason = 'verify_replan_timeout'
                        else:
                            failure_reason = 'verify_replan_failed'
                    print(f"[ERR] Thinking controller failed ({thinking_mode})")
                    print(
                        "[WARN] Finalize the episode with current trajectory/metrics "
                        "instead of discarding evaluation."
                    )
                    break

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

        if hasattr(self, 'dtg_history') and self.dtg_history:
            valid_dtgs = [d for d in self.dtg_history if d >= 0]
            if valid_dtgs:
                print(f'\nDTG: min={min(valid_dtgs):.2f}m final={valid_dtgs[-1]:.2f}m')

        final_metrics = self.finish_episode(
            success=navigation_complete,
            stop_action=True
        )

        if self.final_stop_was_executed and self.nav_visualizer and self.latest_obs is not None:
            subtask_text = None
            if self.current_subtask:
                subtask_text = self.current_subtask.get('subtask_instruction', '')

            distance = 0.0
            if self.latest_info:
                distance = self.latest_info.get('distance_to_goal', 0.0)

            subtask_id = self._current_subtask_run_id()
            self.nav_visualizer.save_step_visualization(
                observations=self.latest_obs,
                info=self._build_nav_visualizer_info(self.latest_info),
                step=self.current_step,
                instruction=self.current_instruction,
                current_subtask=subtask_text,
                distance=distance,
                action="STOP",
                subtask_id=subtask_id,
            )

        gif_path = None
        topdown_path = None
        if self.nav_visualizer:
            topdown_path = self.nav_visualizer.save_final_top_down_map()
        if self.nav_visualizer and self.runtime_options.save_navigation_gif:
            gif_path = self.nav_visualizer.save_gif(fps=2)
            if (
                gif_path
                and self.runtime_options.cleanup_navigation_step_images_after_gif
            ):
                removed_count = self.nav_visualizer.cleanup_step_images()
                if removed_count > 0:
                    print(f"[Visualization] Removed {removed_count} step images after GIF generation")
            self.nav_visualizer.clear_frames()

        total_steps = self.current_step

        env_metrics = final_metrics if final_metrics else {}
        if not env_metrics:
            try:
                if hasattr(self.envs, 'call_at'):
                    env_metrics = self.envs.call_at(0, 'get_metrics')
            except Exception:
                env_metrics = {}

        normalized_env_metrics = self._normalize_final_env_metrics(env_metrics)
        final_success = bool((normalized_env_metrics or {}).get('success', 0))
        final_failure_reason = self._infer_navigation_failure_reason(
            explicit_reason=failure_reason,
            env_metrics=normalized_env_metrics,
            total_steps=total_steps,
            max_steps=max_steps,
            final_success=final_success,
        )
        final_result = self._save_navigation_result(total_steps, normalized_env_metrics)
        episode_timing_summary = self._build_episode_timing_summary()

        return {
            'success': final_success,
            'total_steps': total_steps,
            'subtask_count': self.subtask_count,
            'distance_to_goal': float(
                (normalized_env_metrics or {}).get('distance_to_goal', -1.0) or -1.0
            ),
            'spl': float((normalized_env_metrics or {}).get('spl', 0.0) or 0.0),
            'soft_spl': float(
                (normalized_env_metrics or {}).get('soft_spl', 0.0)
                or (normalized_env_metrics or {}).get('oracle_spl', 0.0)
                or 0.0
            ),
            'detected_classes': list(self.detected_classes),
            'episode_duration_s': episode_timing_summary['episode_duration_s'],
            'failed_api_total_duration_s': episode_timing_summary['failed_api_total_duration_s'],
            'failed_retry_wait_duration_s': episode_timing_summary['failed_retry_wait_duration_s'],
            'failed_wasted_duration_s': episode_timing_summary['failed_wasted_duration_s'],
            'thinking_api_summary': episode_timing_summary['thinking_api_summary'],
            'action_api_summary': episode_timing_summary['action_api_summary'],
            'gif_path': gif_path,
            'topdown_path': topdown_path,
            'result_file': final_result,
            'reason': final_failure_reason,
        }

    def _build_nav_visualizer_info(self, info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        payload = dict(info or {})
        if (
            "global_map_input" not in payload and
            getattr(self, "latest_global_map_input", None) is not None
        ):
            payload["global_map_input"] = self.latest_global_map_input
        return payload

    def _should_autostop_from_goal_distance(self) -> bool:
        return False

    def _attempt_goal_distance_autostop(self) -> bool:
        if not self._should_autostop_from_goal_distance():
            return False
        print(
            "[AutoGoalStop] distance_to_goal is already within the success radius; "
            "issuing STOP immediately."
        )
        result = self.step_with_vlm(
            resolve_habitat_action("STOP"),
            action_name="AUTO_GOAL_STOP",
            save_vis=True,
            enable_landmark_detection=False,
        )
        if result.get("done", False):
            print("[AutoGoalStop] STOP executed and episode finished.")
            return True
        print("[AutoGoalStop] STOP was issued but episode did not finish; continue.")
        return False

    def _estimate_fallback_success_spl(self, path_length: float) -> float:
        shortest_path_distance = float(getattr(self, 'initial_distance_to_goal', 0.0) or 0.0)
        if shortest_path_distance <= 0.0:
            shortest_path_distance = float(getattr(self, 'reference_path_length', 0.0) or 0.0)
        if shortest_path_distance <= 0.0 or path_length <= 0.0:
            return 0.0
        return float(shortest_path_distance / max(float(path_length), float(shortest_path_distance)))

    def _normalize_final_env_metrics(self, env_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        metrics = dict(env_metrics or self.latest_info or {})
        success_distance_m = self._get_success_distance_m()
        distance_to_goal = metrics.get('distance_to_goal', -1.0)
        try:
            distance_to_goal = float(distance_to_goal)
        except (TypeError, ValueError):
            distance_to_goal = -1.0

        if (
            bool(getattr(self, 'final_stop_action_requested', False)) and
            bool(getattr(self, 'final_stop_skipped_due_to_done', False)) and
            0.0 <= distance_to_goal <= success_distance_m
        ):
            metrics['success'] = 1
            metrics['oracle_success'] = max(int(metrics.get('oracle_success', 0) or 0), 1)

            current_spl = float(metrics.get('spl', 0.0) or 0.0)
            if current_spl <= 0.0:
                metrics['spl'] = self._estimate_fallback_success_spl(
                    float(metrics.get('path_length', 0.0) or 0.0)
                )

            current_oracle_spl = float(metrics.get('oracle_spl', 0.0) or 0.0)
            if current_oracle_spl <= 0.0:
                metrics['oracle_spl'] = max(
                    current_oracle_spl,
                    float(metrics.get('spl', 0.0) or 0.0),
                )

            metrics['_final_success_inferred'] = True

        return metrics

    def _infer_navigation_failure_reason(
        self,
        *,
        explicit_reason: str,
        env_metrics: Optional[Dict[str, Any]],
        total_steps: int,
        max_steps: int,
        final_success: bool,
    ) -> str:
        reason_text = str(explicit_reason or "").strip()
        if final_success:
            return ""
        if reason_text:
            return reason_text

        cycle_reason = str(
            getattr(self, "latest_thinking_cycle_info", {}).get("reason", "") or ""
        ).strip()
        if cycle_reason == "insufficient_steps_for_lookaround":
            return "episode_budget_exhausted_before_replan"
        if cycle_reason:
            return cycle_reason

        metrics = dict(env_metrics or self.latest_info or {})
        success_distance_m = self._get_success_distance_m()
        distance_to_goal = metrics.get("distance_to_goal", -1.0)
        try:
            distance_to_goal = float(distance_to_goal)
        except (TypeError, ValueError):
            distance_to_goal = -1.0

        if int(total_steps or 0) >= int(max_steps or 0) > 0:
            return "max_episode_steps_reached"

        episode_done = bool(metrics.get("done", False) or self._episode_done_cached())
        if episode_done:
            return "episode_done_without_success"

        if distance_to_goal >= 0.0:
            return "goal_not_reached"

        if int(total_steps or 0) <= 0:
            return "failed_before_first_step"
        return "navigation_failed"

    def _save_navigation_result(self, total_steps: int, env_metrics: Dict = None) -> str:
        """
        Save the final navigation result under `log/`.

        Key VLN-CE metrics:
        - NE: distance to goal at stop time, lower is better
        - SR: success rate under the success-distance threshold
        - SPL: success weighted by path length
        - OSR: oracle success rate across the full trajectory
        - nDTW: normalized DTW against the reference path, in [0, 1]

        Args:
            total_steps: Total episode steps.
            env_metrics: Metrics dictionary returned by the environment.
        """
        def check_inf_nan(value):
            """Clamp invalid values using the same policy as the baseline code."""
            if isinstance(value, (int, float)):
                if math.isinf(value) or math.isnan(value):
                    return 0
            return value
        
        metrics_source = dict(env_metrics if env_metrics else (self.latest_info if self.latest_info else {}))
        episode_timing_summary = self._build_episode_timing_summary()
        episode_duration_s = episode_timing_summary['episode_duration_s']
        failed_api_total_duration_s = episode_timing_summary['failed_api_total_duration_s']
        failed_retry_wait_duration_s = episode_timing_summary['failed_retry_wait_duration_s']
        failed_wasted_duration_s = episode_timing_summary['failed_wasted_duration_s']
        thinking_api_summary = episode_timing_summary['thinking_api_summary']
        action_api_summary = episode_timing_summary['action_api_summary']
        
        # Extract and validate core metrics.
        result = {
            'episode_id': self.current_episode_id,
            'instruction': self.current_instruction,
            'total_steps': total_steps,
            'subtask_count': self.subtask_count,
            'episode_duration_s': episode_duration_s,
            'failed_api_total_duration_s': failed_api_total_duration_s,
            'failed_retry_wait_duration_s': failed_retry_wait_duration_s,
            'failed_wasted_duration_s': failed_wasted_duration_s,
            
            # Core navigation metrics with validation.
            'success': int(check_inf_nan(metrics_source.get('success', 0))),
            'spl': float(check_inf_nan(metrics_source.get('spl', 0.0))),
            'distance_to_goal': float(check_inf_nan(metrics_source.get('distance_to_goal', -1.0))),
            'ndtw': float(check_inf_nan(metrics_source.get('ndtw', metrics_source.get('nDTW', 0.0)))),
            'path_length': float(check_inf_nan(metrics_source.get('path_length', 0.0))),
            
            # Oracle metrics with validation.
            'oracle_success': int(check_inf_nan(metrics_source.get('oracle_success', 0))),
            'oracle_navigation_error': float(check_inf_nan(metrics_source.get('oracle_navigation_error', float('inf')))),
            'oracle_spl': float(check_inf_nan(metrics_source.get('oracle_spl', 0.0))),

            'thinking_api_summary': thinking_api_summary,
            'action_api_summary': action_api_summary,
            'timestamp': datetime.now().isoformat()
        }

        result['sr'] = result['success']
        result['osr'] = result['oracle_success']
        result['ne'] = result['distance_to_goal']
        return self.save_manager.save_result(result)
    
    def _print_subtask_info(self, response: Dict, is_initial: bool = False):
        """Print planner / verifier subtask payloads in JSON form."""
        # Choose the log title from the response type.
        attempt_letter = self._current_attempt_letter()
        if is_initial:
            title = f"Initial Subtask #{self.subtask_count}{attempt_letter}"
        elif 'is_completed' in response:
            # Verify response
            if response.get('is_completed', False):
                title = f"Subtask #{self.subtask_count}{attempt_letter} - Completed ✓"
            else:
                title = f"Subtask #{self.subtask_count}{attempt_letter} - Continue (Not Completed)"
        else:
            title = f"Subtask #{self.subtask_count}{attempt_letter}"
        
        dest = self._get_next_waypoint_field(response) or 'N/A'
        instr = response.get('subtask_instruction', 'N/A')[:80]
        print(f"  {title}: {dest} | {instr}")
    
    # ========== Waypoint helpers ==========

    def _get_waypoint_summary(self, include_area_chain: bool = True, include_path: bool = True) -> str:
        """
        Build the waypoint summary used in LLM prompts.

        The summary includes waypoint distance / direction relative to the
        current pose and the ordered topology chain.
        """
        wp_pos, wp_ids, wp_descs = self.mapper.get_global_waypoints()
        return build_waypoint_summary(
            waypoint_positions=wp_pos,
            waypoint_ids=wp_ids,
            waypoint_descriptions=wp_descs,
            waypoint_area_labels=self.mapper.get_global_waypoint_area_labels(),
            waypoint_initial_neighborhood_flags=self.mapper.get_global_waypoint_initial_neighborhood_flags(),
            waypoint_floor_ids=self.mapper.get_global_waypoint_floor_ids(),
            current_pose=self.mapper.full_pose,
            resolution_cm=self.mapper.resolution,
            current_space_area_label=getattr(self.mapper, 'current_space_area_display_label', ""),
            current_space_area_type=getattr(self.mapper, 'current_space_area_type', ""),
            full_map=getattr(self.mapper, 'full_map', None),
            crop_offset=getattr(getattr(self.mapper, 'mapping_module', None), 'full_map_crop_offset', None),
            initial_waypoint_index=getattr(getattr(self.mapper, 'global_waypoint_manager', None), 'initial_waypoint_index', 0),
            current_world_z=getattr(self.mapper, 'current_world_z', None),
            current_floor_id=int(getattr(self.mapper, 'current_floor_id', 0) or 0),
            multi_floor_active=bool(getattr(self.mapper, 'multi_floor_active', False)),
            on_stairs_connector=bool(getattr(self.mapper, 'on_stairs_connector', False)),
            stair_connectors=getattr(self.mapper, 'stair_connectors', []),
            include_area_chain=include_area_chain,
            include_path=include_path,
        )

    # ========== Shared / legacy hooks ==========
