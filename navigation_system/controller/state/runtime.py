"""Shared runtime state helpers for navigation controllers."""

from dataclasses import dataclass, field
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from navigation_system.runtime.storage.naming import build_subtask_name
from navigation_system.vlm.reporting.usage import (
    compact_vlm_info_payload,
    merge_vlm_usage_summaries,
    summarize_vlm_usage,
)


@dataclass(frozen=True)
class VLMControllerOptions:
    """Normalized controller runtime options loaded from structured config panels."""

    enable_auto_retreat: bool
    save_api_request_artifacts: bool
    save_navigation_step_images: bool
    save_navigation_gif: bool
    cleanup_navigation_step_images_after_gif: bool
    navigation_gif_fps: int
    navigation_gif_max_width: int
    save_episode_stdout_log: bool
    save_waypoint_memory: bool
    enable_final_destination_match_autostop: bool
    final_destination_match_autostop_streak: int
    final_destination_match_autostop_radius_m: float
    low_level_stagnation_ratio: float
    low_level_stagnation_cap_m: float
    geometric_waypoint_enabled: bool
    geometric_waypoint_max_candidates: int
    geometric_waypoint_min_distance_m: float
    geometric_waypoint_max_distance_m: float
    geometric_waypoint_stride_m: float
    geometric_waypoint_obstacle_inflation_radius_m: float
    geometric_waypoint_unknown_as_obstacle: bool
    geometric_waypoint_min_clearance_m: float
    geometric_waypoint_path_step_m: float
    geometric_waypoint_arrival_radius_m: float
    geometric_waypoint_max_path_execute_steps: int
    geometric_waypoint_stop_on_blocked_front: bool

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        default_final_destination_match_autostop_streak: int,
        default_final_destination_match_autostop_radius_m: float,
        default_low_level_stagnation_ratio: float,
        default_low_level_stagnation_cap_m: float,
    ) -> "VLMControllerOptions":
        output_cfg = getattr(config, "OUTPUT", None)
        control_cfg = getattr(config, "CONTROL", None)
        request_cfg = getattr(output_cfg, "REQUESTS", None)
        replay_cfg = getattr(output_cfg, "REPLAY", None)
        log_cfg = getattr(output_cfg, "LOGS", None)
        state_cfg = getattr(output_cfg, "STATE", None)
        recovery_cfg = getattr(control_cfg, "RECOVERY", None)
        stagnation_cfg = getattr(control_cfg, "STAGNATION", None)
        stopping_cfg = getattr(control_cfg, "STOPPING", None)
        geometric_cfg = getattr(control_cfg, "GEOMETRIC_WAYPOINT", None)
        env_enable_geometric = str(
            os.getenv("SPACEVLN_GEOMETRIC_WAYPOINT", "") or ""
        ).strip().lower()
        geometric_enabled = bool(getattr(geometric_cfg, "ENABLE", False))
        if env_enable_geometric in {"1", "true", "yes", "on"}:
            geometric_enabled = True
        elif env_enable_geometric in {"0", "false", "no", "off"}:
            geometric_enabled = False
        return cls(
            enable_auto_retreat=bool(getattr(recovery_cfg, "ENABLE_AUTO_RETREAT", False)),
            save_api_request_artifacts=bool(getattr(request_cfg, "SAVE_VLM_ARTIFACTS", False)),
            save_navigation_step_images=bool(getattr(replay_cfg, "SAVE_STEP_IMAGES", False)),
            save_navigation_gif=bool(getattr(replay_cfg, "SAVE_GIF", True)),
            cleanup_navigation_step_images_after_gif=bool(
                getattr(replay_cfg, "CLEANUP_STEP_IMAGES_AFTER_GIF", True)
            ),
            navigation_gif_fps=max(1, int(getattr(replay_cfg, "GIF_FPS", 2) or 2)),
            navigation_gif_max_width=max(
                0,
                int(getattr(replay_cfg, "GIF_MAX_WIDTH", 720) or 0),
            ),
            save_episode_stdout_log=bool(getattr(log_cfg, "SAVE_EPISODE_STDOUT", False)),
            save_waypoint_memory=bool(getattr(state_cfg, "SAVE_WAYPOINT_MEMORY", False)),
            enable_final_destination_match_autostop=bool(
                getattr(stopping_cfg, "ENABLE_FINAL_DESTINATION_MATCH_AUTOSTOP", True)
            ),
            final_destination_match_autostop_streak=max(
                1,
                int(
                    getattr(
                        stopping_cfg,
                        "FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK",
                        default_final_destination_match_autostop_streak,
                    )
                    or default_final_destination_match_autostop_streak
                ),
            ),
            final_destination_match_autostop_radius_m=max(
                0.0,
                float(
                    getattr(
                        stopping_cfg,
                        "FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M",
                        default_final_destination_match_autostop_radius_m,
                    )
                    or default_final_destination_match_autostop_radius_m
                ),
            ),
            low_level_stagnation_ratio=max(
                0.0,
                float(
                    getattr(
                        stagnation_cfg,
                        "LOW_LEVEL_RATIO",
                        default_low_level_stagnation_ratio,
                    )
                    or default_low_level_stagnation_ratio
                ),
            ),
            low_level_stagnation_cap_m=max(
                0.0,
                float(
                    getattr(
                        stagnation_cfg,
                        "LOW_LEVEL_CAP_M",
                        default_low_level_stagnation_cap_m,
                    )
                    or default_low_level_stagnation_cap_m
                ),
            ),
            geometric_waypoint_enabled=bool(geometric_enabled),
            geometric_waypoint_max_candidates=max(
                1,
                int(getattr(geometric_cfg, "MAX_CANDIDATES", 5) or 5),
            ),
            geometric_waypoint_min_distance_m=max(
                0.0,
                float(getattr(geometric_cfg, "MIN_DISTANCE_M", 0.8) or 0.8),
            ),
            geometric_waypoint_max_distance_m=max(
                0.1,
                float(getattr(geometric_cfg, "MAX_DISTANCE_M", 4.0) or 4.0),
            ),
            geometric_waypoint_stride_m=max(
                0.1,
                float(getattr(geometric_cfg, "STRIDE_M", 0.75) or 0.75),
            ),
            geometric_waypoint_obstacle_inflation_radius_m=max(
                0.0,
                float(getattr(geometric_cfg, "OBSTACLE_INFLATION_RADIUS_M", 0.30) or 0.30),
            ),
            geometric_waypoint_unknown_as_obstacle=bool(
                getattr(geometric_cfg, "UNKNOWN_AS_OBSTACLE", True)
            ),
            geometric_waypoint_min_clearance_m=max(
                0.0,
                float(getattr(geometric_cfg, "MIN_CLEARANCE_M", 0.20) or 0.20),
            ),
            geometric_waypoint_path_step_m=max(
                0.1,
                float(getattr(geometric_cfg, "PATH_STEP_M", 0.35) or 0.35),
            ),
            geometric_waypoint_arrival_radius_m=max(
                0.05,
                float(getattr(geometric_cfg, "ARRIVAL_RADIUS_M", 0.35) or 0.35),
            ),
            geometric_waypoint_max_path_execute_steps=max(
                1,
                int(getattr(geometric_cfg, "MAX_PATH_EXECUTE_STEPS", 12) or 12),
            ),
            geometric_waypoint_stop_on_blocked_front=bool(
                getattr(geometric_cfg, "STOP_ON_BLOCKED_FRONT", True)
            ),
        )

    def low_level_stagnation_threshold_m(self, move_distance_m: float) -> float:
        configured_threshold_m = max(0.0, float(self.low_level_stagnation_cap_m or 0.0))
        step_scaled_threshold_m = max(
            0.0,
            float(move_distance_m or 0.0) * float(self.low_level_stagnation_ratio or 0.0),
        )
        if step_scaled_threshold_m <= 0.0:
            return configured_threshold_m
        if configured_threshold_m <= 0.0:
            return step_scaled_threshold_m
        return min(configured_threshold_m, step_scaled_threshold_m)


@dataclass
class EpisodeTimingTracker:
    """Owns API timing records and episode wall-clock accounting."""

    episode_wall_start_time: Optional[float] = None
    episode_wall_end_time: Optional[float] = None
    failed_retry_wait_duration_s: float = 0.0
    thinking_records: List[Dict[str, Any]] = field(default_factory=list)
    action_records: List[Dict[str, Any]] = field(default_factory=list)
    local_sections: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def round_duration_s(duration_s: float) -> float:
        try:
            return round(max(0.0, float(duration_s)), 4)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def summarize_records(cls, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        normalized_records = [item for item in list(records or []) if isinstance(item, dict)]
        durations = [float(item.get("duration_s", 0.0)) for item in normalized_records]
        failure_durations = [
            float(item.get("duration_s", 0.0))
            for item in normalized_records
            if not bool(item.get("success", False))
        ]
        count = len(normalized_records)
        failure_count = len(failure_durations)
        total_duration_s = sum(durations)
        failed_total_duration_s = sum(failure_durations)
        avg_duration_s = total_duration_s / count if count > 0 else 0.0
        return {
            "count": int(count),
            "failure_count": int(failure_count),
            "total_duration_s": cls.round_duration_s(total_duration_s),
            "avg_duration_s": cls.round_duration_s(avg_duration_s),
            "failed_total_duration_s": cls.round_duration_s(failed_total_duration_s),
        }

    @classmethod
    def summarize_record_usage(cls, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return summarize_vlm_usage(
            [
                dict(item.get("vlm_info") or {})
                for item in list(records or [])
                if isinstance(item, dict) and isinstance(item.get("vlm_info"), dict)
            ]
        )

    @classmethod
    def attach_usage_summary(
        cls,
        api_summary: Dict[str, Any],
        records: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = dict(api_summary or {})
        usage_summary = cls.summarize_record_usage(records)
        for key, value in usage_summary.items():
            if key == "count":
                payload["token_count"] = int(value or 0)
            else:
                payload[key] = value
        return payload

    def reset(self) -> None:
        self.episode_wall_start_time = None
        self.episode_wall_end_time = None
        self.failed_retry_wait_duration_s = 0.0
        self.thinking_records.clear()
        self.action_records.clear()
        self.local_sections.clear()

    def mark_episode_active(self) -> None:
        if self.episode_wall_start_time is None:
            self.episode_wall_start_time = time.perf_counter()
            self.episode_wall_end_time = None

    def mark_episode_step_finished(self, *, action_name: str = "", episode_done: bool = False) -> None:
        if self.episode_wall_start_time is None:
            return
        if str(action_name or "").upper() == "STOP" or bool(episode_done):
            self.episode_wall_end_time = time.perf_counter()

    def mark_episode_finished(self) -> None:
        """Close the episode timer before post-run artifact/report saving."""
        if self.episode_wall_start_time is None:
            return
        if self.episode_wall_end_time is None:
            self.episode_wall_end_time = time.perf_counter()

    def current_episode_duration_s(self) -> float:
        if self.episode_wall_start_time is None:
            return 0.0
        end_time = self.episode_wall_end_time
        if end_time is None:
            end_time = time.perf_counter()
        return self.round_duration_s(float(end_time) - float(self.episode_wall_start_time))

    def add_failed_retry_wait(self, duration_s: float) -> None:
        self.failed_retry_wait_duration_s = self.round_duration_s(
            float(self.failed_retry_wait_duration_s or 0.0) + float(duration_s or 0.0)
        )

    def record_local_section(self, section: str, duration_s: float) -> None:
        section_name = str(section or "").strip() or "other"
        duration = max(0.0, float(duration_s or 0.0))
        if duration <= 0.0:
            return
        item = self.local_sections.setdefault(
            section_name,
            {
                "count": 0,
                "total_duration_s": 0.0,
                "max_duration_s": 0.0,
            },
        )
        item["count"] = int(item.get("count", 0) or 0) + 1
        item["total_duration_s"] = float(item.get("total_duration_s", 0.0) or 0.0) + duration
        item["max_duration_s"] = max(float(item.get("max_duration_s", 0.0) or 0.0), duration)

    def summarize_local_sections(self) -> Dict[str, Any]:
        sections: Dict[str, Dict[str, Any]] = {}
        total_duration_s = 0.0
        for name, payload in sorted(self.local_sections.items()):
            count = int(payload.get("count", 0) or 0)
            total = float(payload.get("total_duration_s", 0.0) or 0.0)
            max_duration = float(payload.get("max_duration_s", 0.0) or 0.0)
            total_duration_s += total
            sections[name] = {
                "count": count,
                "total_duration_s": self.round_duration_s(total),
                "avg_duration_s": self.round_duration_s(total / count if count > 0 else 0.0),
                "max_duration_s": self.round_duration_s(max_duration),
            }
        return {
            "total_duration_s": self.round_duration_s(total_duration_s),
            "sections": sections,
        }

    def record_thinking_call(
        self,
        *,
        mode: str,
        phase: str,
        step: int,
        subtask_count: int,
        subtask_attempt: int,
        duration_s: float,
        success: bool,
        next_waypoint: Optional[str] = None,
        vlm_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "index": len(self.thinking_records) + 1,
            "mode": str(mode or ""),
            "phase": str(phase or ""),
            "step": int(step or 0),
            "subtask_count": int(subtask_count or 0),
            "subtask_attempt": int(subtask_attempt or 0),
            "success": bool(success),
            "duration_s": self.round_duration_s(duration_s),
            "next_waypoint": str(next_waypoint or ""),
        }
        compact_info = compact_vlm_info_payload(vlm_info)
        if compact_info:
            record["vlm_info"] = compact_info
        self.thinking_records.append(record)

    def record_action_call(
        self,
        *,
        step: int,
        subtask_count: int,
        subtask_attempt: int,
        duration_s: float,
        success: bool,
        action_name: Optional[str] = None,
        vlm_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "index": len(self.action_records) + 1,
            "step": int(step or 0) + 1,
            "subtask_id": build_subtask_name(int(subtask_count or 0) or 1),
            "success": bool(success),
            "duration_s": self.round_duration_s(duration_s),
            "action": str(action_name or ""),
        }
        compact_info = compact_vlm_info_payload(vlm_info)
        if compact_info:
            record["vlm_info"] = compact_info
        self.action_records.append(record)

    def build_summary(self) -> Dict[str, Any]:
        thinking_api_summary = self.attach_usage_summary(
            self.summarize_records(self.thinking_records),
            self.thinking_records,
        )
        action_api_summary = self.attach_usage_summary(
            self.summarize_records(self.action_records),
            self.action_records,
        )
        vlm_usage_summary = {
            "thinking": self.summarize_record_usage(self.thinking_records),
            "action": self.summarize_record_usage(self.action_records),
        }
        vlm_usage_summary["overall"] = merge_vlm_usage_summaries(
            [
                vlm_usage_summary["thinking"],
                vlm_usage_summary["action"],
            ]
        )
        episode_duration_s = self.current_episode_duration_s()
        failed_api_total_duration_s = float(
            thinking_api_summary.get("failed_total_duration_s", 0.0) or 0.0
        ) + float(action_api_summary.get("failed_total_duration_s", 0.0) or 0.0)
        api_total_duration_s = float(
            thinking_api_summary.get("total_duration_s", 0.0) or 0.0
        ) + float(action_api_summary.get("total_duration_s", 0.0) or 0.0)
        failed_retry_wait_duration_s = self.round_duration_s(self.failed_retry_wait_duration_s)
        failed_wasted_duration_s = self.round_duration_s(
            failed_api_total_duration_s + failed_retry_wait_duration_s
        )
        local_timing_summary = self.summarize_local_sections()
        return {
            "episode_duration_s": self.round_duration_s(episode_duration_s),
            "local_non_api_duration_s": self.round_duration_s(
                max(0.0, episode_duration_s - api_total_duration_s)
            ),
            "failed_api_total_duration_s": self.round_duration_s(failed_api_total_duration_s),
            "failed_retry_wait_duration_s": failed_retry_wait_duration_s,
            "failed_wasted_duration_s": failed_wasted_duration_s,
            "thinking_api_summary": thinking_api_summary,
            "action_api_summary": action_api_summary,
            "vlm_usage_summary": vlm_usage_summary,
            "local_timing_summary": local_timing_summary,
        }
