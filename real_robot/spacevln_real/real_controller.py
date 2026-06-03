"""Real-robot controller extensions that stream live progress to result files."""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
import re
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from navigation_system.config.core.params.thresholds import OBS_BLOCKED_M
from navigation_system.controller.action_compat import resolve_habitat_action
from navigation_system.controller.agent.controller import NavigationAgentController
from navigation_system.space.map.obstacle_analysis import sample_depth_distance_from_region


class RealNavigationAgentController(NavigationAgentController):
    """Navigation controller with real-only live result flushing."""

    @staticmethod
    def _safe_artifact_token(text: str, default: str = "step") -> str:
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())
        token = token.strip("._-")
        return token or str(default)

    def _save_real_step_rgb(
        self,
        *,
        event: str,
        step: Optional[int] = None,
        obs: Optional[Dict[str, Any]] = None,
        action: str = "",
    ) -> str:
        save_manager = getattr(self, "save_manager", None)
        if save_manager is None:
            return ""
        obs_payload = obs if isinstance(obs, dict) else getattr(self, "latest_obs", None)
        if not isinstance(obs_payload, dict) or "rgb" not in obs_payload:
            return ""

        try:
            rgb = np.asarray(obs_payload["rgb"], dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] < 3:
                return ""
            rgb = rgb[:, :, :3]
            step_idx = int(self.current_step if step is None else step)
            event_token = self._safe_artifact_token(event, default="step")
            action_token = self._safe_artifact_token(action, default="rgb")
            output_dir = os.path.join(save_manager.records_dir, "step_rgb")
            os.makedirs(output_dir, exist_ok=True)
            if not bool(getattr(self, "_real_step_rgb_dir_reported", False)):
                self._real_step_rgb_dir_reported = True
                print(f"[REAL] step_rgb_dir={output_dir}", flush=True)
            filename = f"step_{step_idx:04d}_{event_token}_{action_token}.jpg"
            path = os.path.join(output_dir, filename)
            if not cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                print(f"[WARN] Failed to save real step RGB: {path}", flush=True)
                return ""
            return path
        except Exception as exc:
            print(f"[WARN] Failed to save real step RGB: {exc}", flush=True)
            return ""

    def _get_latest_real_observation(self) -> Optional[Dict[str, Any]]:
        try:
            call_at = getattr(self.envs, "call_at", None)
            if callable(call_at):
                obs = call_at(0, "get_latest_observation")
            else:
                getter = getattr(self.envs, "get_latest_observation", None)
                obs = getter() if callable(getter) else None
            return obs if isinstance(obs, dict) else None
        except Exception:
            return None

    @staticmethod
    def _real_rgb_transition_sample_count() -> int:
        raw_value = str(os.getenv("SPACEVLN_REAL_RGB_TRANSITION_SAMPLES", "") or "").strip()
        if raw_value:
            try:
                return max(0, min(60, int(raw_value)))
            except (TypeError, ValueError):
                pass
        return 4

    @staticmethod
    def _obs_rgb_stamp(obs: Optional[Dict[str, Any]]) -> Optional[float]:
        if not isinstance(obs, dict):
            return None
        for key in ("rgb_timestamp", "timestamp"):
            try:
                value = float(obs.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0.0:
                return float(value)
        return None

    def _get_real_rgb_samples_between(
        self,
        *,
        start_stamp: float,
        end_stamp: float,
        sample_count: int,
    ) -> Sequence[Dict[str, Any]]:
        try:
            call_at = getattr(self.envs, "call_at", None)
            if callable(call_at):
                samples = call_at(
                    0,
                    "get_rgb_samples_between",
                    start_stamp=float(start_stamp),
                    end_stamp=float(end_stamp),
                    sample_count=int(sample_count),
                )
            else:
                getter = getattr(self.envs, "get_rgb_samples_between", None)
                samples = (
                    getter(
                        start_stamp=float(start_stamp),
                        end_stamp=float(end_stamp),
                        sample_count=int(sample_count),
                    )
                    if callable(getter)
                    else []
                )
            return list(samples) if isinstance(samples, (list, tuple)) else []
        except Exception:
            return []

    def _save_real_transition_rgb_samples(
        self,
        *,
        current_step: int,
        current_obs: Optional[Dict[str, Any]],
        action: str,
    ) -> Sequence[str]:
        previous_step = getattr(self, "_real_last_low_level_rgb_step", None)
        previous_stamp = getattr(self, "_real_last_low_level_rgb_stamp", None)
        current_stamp = self._obs_rgb_stamp(current_obs)
        sample_count = self._real_rgb_transition_sample_count()
        if (
            previous_step is None
            or previous_stamp is None
            or current_stamp is None
            or sample_count <= 0
            or float(current_stamp) <= float(previous_stamp)
        ):
            return []

        samples = self._get_real_rgb_samples_between(
            start_stamp=float(previous_stamp),
            end_stamp=float(current_stamp),
            sample_count=sample_count,
        )
        saved_paths = []
        total = len(samples)
        for index, sample in enumerate(samples, start=1):
            action_token = (
                f"{int(previous_step):04d}_to_{int(current_step):04d}_"
                f"sample_{index:02d}_of_{total:02d}_{action}"
            )
            path = self._save_real_step_rgb(
                event="between_steps",
                step=int(current_step),
                obs=sample if isinstance(sample, dict) else None,
                action=action_token,
            )
            if path:
                saved_paths.append(path)
        return saved_paths

    def _remember_real_low_level_rgb_endpoint(
        self,
        *,
        step: int,
        obs: Optional[Dict[str, Any]],
    ) -> None:
        rgb_stamp = self._obs_rgb_stamp(obs)
        if rgb_stamp is None:
            return
        self._real_last_low_level_rgb_step = int(step)
        self._real_last_low_level_rgb_stamp = float(rgb_stamp)

    def _real_map_alignment_snapshot(
        self,
        *,
        obs: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        mapper = getattr(self, "mapper", None)
        if mapper is None:
            return {}
        try:
            map_state = mapper.get_map_state()
        except Exception:
            map_state = {}

        full_map = map_state.get("full_map") if isinstance(map_state, dict) else None
        full_pose = map_state.get("full_pose") if isinstance(map_state, dict) else None
        global_traj = map_state.get("global_trajectory_points", []) if isinstance(map_state, dict) else []
        subtask_traj = map_state.get("subtask_trajectory_points", []) if isinstance(map_state, dict) else []
        floor = map_state.get("floor") if isinstance(map_state, dict) else None

        pose_delta = None
        if isinstance(info, dict):
            pose_delta = info.get("real_sensor_pose_delta")
        if pose_delta is None and isinstance(obs, dict):
            pose_delta = obs.get("sensor_pose")
        try:
            pose_delta_values = [float(value) for value in list(pose_delta)[:3]]
        except Exception:
            pose_delta_values = []

        pose_delta_m = 0.0
        pose_delta_yaw_deg = 0.0
        if len(pose_delta_values) >= 3:
            pose_delta_m = float(math.hypot(pose_delta_values[0], pose_delta_values[1]))
            pose_delta_yaw_deg = float(math.degrees(pose_delta_values[2]))

        obstacle_cells = explored_cells = 0
        map_shape = None
        if full_map is not None:
            try:
                full_map_array = np.asarray(full_map)
                map_shape = list(full_map_array.shape)
                if full_map_array.ndim >= 3 and full_map_array.shape[0] >= 2:
                    obstacle_cells = int(np.count_nonzero(full_map_array[0] > 0.5))
                    explored_cells = int(np.count_nonzero(full_map_array[1] > 0.5))
            except Exception:
                map_shape = None

        floor_cells = 0
        if floor is not None:
            try:
                floor_cells = int(np.count_nonzero(np.asarray(floor) > 0))
            except Exception:
                floor_cells = 0

        pose_values = []
        try:
            pose_values = [float(value) for value in list(full_pose)[:3]]
        except Exception:
            pose_values = []

        if pose_delta_m > 2.0 or abs(pose_delta_yaw_deg) > 120.0:
            print(
                "[REAL-MAP] large measured pose delta in map update: "
                f"delta_m={pose_delta_m:.2f} yaw_deg={pose_delta_yaw_deg:.1f}",
                flush=True,
            )

        return {
            "pose_delta": pose_delta_values,
            "pose_delta_m": pose_delta_m,
            "pose_delta_yaw_deg": pose_delta_yaw_deg,
            "full_pose": pose_values,
            "global_traj_points": len(global_traj or []),
            "subtask_traj_points": len(subtask_traj or []),
            "obstacle_cells": obstacle_cells,
            "explored_cells": explored_cells,
            "floor_cells": floor_cells,
            "map_shape": map_shape,
            "depth_map_update_disabled": bool(self._depth_map_update_disabled()),
        }

    @staticmethod
    def _real_forward_min_clearance_m() -> float:
        raw_value = str(os.getenv("SPACEVLN_REAL_FORWARD_MIN_CLEARANCE_M", "") or "").strip()
        if raw_value:
            try:
                return max(0.0, float(raw_value))
            except (TypeError, ValueError):
                pass
        return float(OBS_BLOCKED_M)

    def _estimate_real_front_clearance_m(self, obs: Optional[Dict[str, Any]] = None) -> Optional[float]:
        obs_payload = obs if isinstance(obs, dict) else getattr(self, "latest_obs", None)
        if not isinstance(obs_payload, dict) or "depth" not in obs_payload:
            return None
        try:
            return sample_depth_distance_from_region(
                np.asarray(obs_payload["depth"], dtype=np.float32),
                center_x_ratio=0.5,
                width_ratio=0.26,
                row_start_ratio=0.38,
                row_end_ratio=0.92,
                max_distance_m=5.0,
                sensor_min_depth_m=float(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH),
                sample_count=128,
                sample_percentile=20.0,
            )
        except Exception:
            return None

    def _is_real_forward_safety_blocked(
        self,
        action_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[float], float]:
        threshold_m = self._real_forward_min_clearance_m()
        clearance_m = None
        if isinstance(action_context, dict):
            clearance_m = action_context.get("real_front_clearance_m")
        if clearance_m is None:
            clearance_m = self._estimate_real_front_clearance_m()
        try:
            clearance_m = float(clearance_m) if clearance_m is not None else None
        except (TypeError, ValueError):
            clearance_m = None
        blocked = clearance_m is not None and clearance_m < threshold_m
        return blocked, clearance_m, threshold_m

    def _filter_real_forward_safety_actions(
        self,
        allowed_action_names: Optional[Sequence[str]],
        action_context: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[str, ...]]:
        blocked, clearance_m, threshold_m = self._is_real_forward_safety_blocked(action_context)
        if not blocked:
            return tuple(allowed_action_names) if allowed_action_names else None

        base_actions = (
            list(allowed_action_names)
            if allowed_action_names
            else ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
        )
        filtered = [
            str(name).strip().upper()
            for name in base_actions
            if str(name).strip().upper() != "MOVE_FORWARD"
        ]
        if not filtered:
            filtered = ["STOP"]
        print(
            "[REAL-SAFETY] removed MOVE_FORWARD from action space: "
            f"front_clearance={float(clearance_m):.2f}m < {float(threshold_m):.2f}m",
            flush=True,
        )
        return tuple(filtered)

    @staticmethod
    def _real_manual_motion_mode() -> bool:
        mode = str(os.getenv("SPACEVLN_REAL_MOTION_MODE", "") or "").strip().lower()
        executor = str(os.getenv("REAL_ACTION_EXECUTOR", "") or "").strip().lower()
        return mode == "manual" or executor == "manual"

    @staticmethod
    def _real_manual_prompt_only_mode() -> bool:
        return str(os.getenv("SPACEVLN_MANUAL_PROMPT_ONLY", "") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _manual_required_for_action_name(action_name: str) -> bool:
        action = str(action_name or "").strip().upper()
        return action in {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"}

    def _build_env_action_payload(
        self,
        action_id: int,
        action_name: Optional[str],
    ) -> Any:
        payload = super()._build_env_action_payload(action_id, action_name)
        if not isinstance(payload, dict):
            return payload
        payload["phase"] = str(self._current_action_phase())
        payload["manual_required"] = bool(
            self._real_manual_motion_mode()
            and self._manual_required_for_action_name(str(action_name or ""))
        )
        return payload

    def _ensure_real_depth_map_disabled_input(self, phase: str) -> str:
        save_manager = getattr(self, "save_manager", None)
        if save_manager is not None:
            map_dir = os.path.join(save_manager.episode_dir, "map")
        else:
            paths_config = getattr(getattr(self, "config", None), "PATHS", None)
            map_dir = os.path.join(str(getattr(paths_config, "RESULTS_DIR", "") or ""), "map")
        os.makedirs(map_dir, exist_ok=True)
        path = os.path.join(map_dir, f"{str(phase or 'thinking')}_real_depth_map_disabled.png")
        if os.path.exists(path):
            self.latest_global_map = path
            self.latest_global_map_input = path
            return path

        image = np.full((720, 960, 3), 245, dtype=np.uint8)
        cv2.rectangle(image, (28, 28), (932, 692), (80, 80, 80), 2)
        cv2.putText(
            image,
            "REAL ROBOT: DEPTH MAP DISABLED",
            (70, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.35,
            (0, 0, 180),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "Use the stopped RGB views and per-view depth obstacle labels.",
            (70, 345),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (45, 45, 45),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "Do not infer route geometry from this placeholder map.",
            (70, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (45, 45, 45),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(path, image)
        self.latest_global_map = path
        self.latest_global_map_input = path
        return path

    def configure_real_live_status(
        self,
        *,
        session_id: str,
        instruction: str,
    ) -> None:
        self.real_session_id = str(session_id or "").strip()
        self.real_instruction = str(instruction or "").strip()
        self.real_live_status_enabled = True
        self.real_live_status_seq = 0

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value) if math.isfinite(value) else None
        try:
            import numpy as np

            if isinstance(value, np.generic):
                return RealNavigationAgentController._json_safe(value.item())
            if isinstance(value, np.ndarray):
                return {
                    "type": "ndarray",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
        except Exception:
            pass
        if isinstance(value, dict):
            return {
                str(key): RealNavigationAgentController._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            if len(value) > 64:
                return {
                    "type": type(value).__name__,
                    "length": len(value),
                    "items_head": [
                        RealNavigationAgentController._json_safe(item)
                        for item in list(value[:16])
                    ],
                }
            return [RealNavigationAgentController._json_safe(item) for item in value]
        return str(value)

    @staticmethod
    def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_path, path)

    @staticmethod
    def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass

    def _real_live_status_paths(self) -> Dict[str, str]:
        paths_config = getattr(getattr(self, "config", None), "PATHS", None)
        results_dir = str(getattr(paths_config, "RESULTS_DIR", "") or "")
        paths = {
            "latest": os.path.join(results_dir, "real_session_latest.json"),
            "session": os.path.join(
                results_dir,
                "real_sessions",
                f"{getattr(self, 'real_session_id', 'session')}.json",
            ),
            "session_steps": os.path.join(
                results_dir,
                "real_sessions",
                f"{getattr(self, 'real_session_id', 'session')}.jsonl",
            ),
        }
        save_manager = getattr(self, "save_manager", None)
        if save_manager is not None:
            paths["episode_live"] = os.path.join(save_manager.episode_dir, "live_result.json")
            paths["records_live"] = os.path.join(save_manager.records_dir, "live_result.json")
            paths["records_steps"] = os.path.join(save_manager.records_dir, "live_steps.jsonl")
        return paths

    def _compact_metrics(self) -> Dict[str, Any]:
        info = getattr(self, "latest_info", None)
        if not isinstance(info, dict):
            return {}
        keys = (
            "success",
            "distance_to_goal",
            "path_length",
            "steps_taken",
            "done",
            "goal_reached",
            "blocked",
            "collision",
            "message",
            "model_task_finished",
            "success_source",
        )
        metrics = {key: info.get(key) for key in keys if key in info}
        action_status = info.get("action_status")
        if isinstance(action_status, dict):
            metrics["action_status"] = action_status
        return self._json_safe(metrics)

    def _compact_subtask(self) -> Dict[str, Any]:
        subtask = getattr(self, "current_subtask", None)
        if not isinstance(subtask, dict):
            return {}
        keys = (
            "subtask_instruction",
            "current_waypoint",
            "task_progress",
            "waypoint_chain",
            "next_waypoint",
            "next_waypoint_direction",
            "subtask_landmark",
            "global_task_finish",
        )
        return self._json_safe({key: subtask.get(key) for key in keys if key in subtask})

    def _build_action_decision_context(self) -> Optional[Dict[str, Any]]:
        context = super()._build_action_decision_context()
        if not isinstance(context, dict):
            return context

        clearance_m = self._estimate_real_front_clearance_m()
        if clearance_m is None:
            return context

        threshold_m = self._real_forward_min_clearance_m()
        context["real_front_clearance_m"] = float(clearance_m)
        context["real_forward_min_clearance_m"] = float(threshold_m)
        context["real_forward_blocked"] = bool(float(clearance_m) < float(threshold_m))

        obstacle_distances = dict(context.get("obstacle_distances") or {})
        existing_front_m = self._parse_distance_text_m(obstacle_distances.get("front"))
        should_update_front = existing_front_m is None or float(clearance_m) < float(existing_front_m)
        if should_update_front:
            if float(clearance_m) < float(threshold_m):
                obstacle_distances["front"] = (
                    f"{float(clearance_m):.2f}m BLOCKED(real safety <{float(threshold_m):.2f}m)"
                )
            else:
                obstacle_distances["front"] = f"{float(clearance_m):.2f}m"
            context["obstacle_distances"] = obstacle_distances
            self.latest_obstacle_distances = dict(obstacle_distances)
        return context

    def _request_vlm_action(
        self,
        action_context: Dict[str, Any],
        action_subtask_instruction: str,
        progress_summary_for_prompt: str,
        allowed_action_names: Optional[Sequence[str]],
    ):
        allowed_action_names = self._filter_real_forward_safety_actions(
            allowed_action_names,
            action_context,
        )
        return super()._request_vlm_action(
            action_context=action_context,
            action_subtask_instruction=action_subtask_instruction,
            progress_summary_for_prompt=progress_summary_for_prompt,
            allowed_action_names=allowed_action_names,
        )

    @staticmethod
    def _read_real_operator_line(prompt: str) -> str:
        try:
            with open("/dev/tty", "r", encoding="utf-8", errors="ignore") as tty_in, open(
                "/dev/tty", "w", encoding="utf-8", errors="ignore"
            ) as tty_out:
                tty_out.write(prompt)
                tty_out.flush()
                return str(tty_in.readline() or "").strip()
        except Exception:
            try:
                return str(input(prompt) or "").strip()
            except EOFError:
                return ""

    def _prepare_manual_prompt_only_artifacts(
        self,
        action_context: Dict[str, Any],
        *,
        allowed_action_names: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        action_subtask_instruction = self._sanitize_subtask_instruction_text(
            self.current_subtask.get("subtask_instruction", ""),
            self._get_next_waypoint_field(self.current_subtask),
            self.current_subtask.get("next_waypoint_direction", ""),
            keep_view_prefix=False,
        )
        progress_summary_for_prompt = self._get_action_progress_summary_for_prompt()
        allowed_action_names = self._filter_real_forward_safety_actions(
            allowed_action_names,
            action_context,
        )
        preparer = getattr(self.action_executor, "prepare_action_request_artifacts", None)
        if not callable(preparer):
            raise RuntimeError("action executor does not support prompt-only artifact preparation")
        return dict(
            preparer(
                next_waypoint=self._get_next_waypoint_field(self.current_subtask),
                subtask_instruction=action_subtask_instruction,
                subtask_landmark=self._get_subtask_landmark_field(self.current_subtask),
                first_person_image=action_context.get("detection_image") or "",
                progress_summary=progress_summary_for_prompt,
                waypoint_summary=action_context.get("waypoint_summary", ""),
                detection_image=action_context.get("detection_image"),
                detected_landmarks=action_context.get("detected_landmarks"),
                obstacle_distances=action_context.get("obstacle_distances"),
                landmark_map_info=action_context.get("action_landmark_map_info"),
                allowed_action_names=allowed_action_names,
                save_dir=action_context.get("action_save_dir"),
            )
            or {}
        )

    def _print_manual_prompt_only_artifacts(self, prepared: Dict[str, Any]) -> None:
        save_dir = str(prepared.get("save_dir") or "").strip()
        print("\n[ManualPromptOnly] 已保存本步 action VLM 输入，但不会调用 VLM。", flush=True)
        if save_dir:
            print(f"[ManualPromptOnly] dir={save_dir}", flush=True)
            for filename in ("system_prompt.md", "user_prompt.md", "action_view.jpg", "vlm_info.json"):
                path = os.path.join(save_dir, filename)
                if os.path.exists(path):
                    print(f"[ManualPromptOnly] {filename}={path}", flush=True)
        for record in list(prepared.get("artifact_records") or []):
            path = str(record.get("artifact_path") or "").strip()
            if path:
                print(f"[ManualPromptOnly] artifact={path}", flush=True)

    def _print_manual_prompt_only_subtask(self) -> None:
        subtask = self._compact_subtask()
        print("[ManualPromptOnly] 当前 thinking VLM 子任务:", flush=True)
        if not subtask:
            print("[ManualPromptOnly]   <empty>", flush=True)
            return
        for key in (
            "current_waypoint",
            "task_progress",
            "waypoint_chain",
            "next_waypoint",
            "next_waypoint_direction",
            "subtask_landmark",
            "subtask_instruction",
            "global_task_finish",
        ):
            if key in subtask:
                value = subtask.get(key)
                print(f"[ManualPromptOnly]   {key}: {value}", flush=True)

    def _manual_prompt_only_operator_prompt(self) -> str:
        subtask = self._compact_subtask()
        instruction = str(subtask.get("subtask_instruction") or "").strip()
        next_waypoint = str(subtask.get("next_waypoint") or "").strip()
        direction = str(subtask.get("next_waypoint_direction") or "").strip()
        landmark = str(subtask.get("subtask_landmark") or "").strip()
        if not instruction:
            instruction = "<empty>"
        parts = [f"\n[ManualPromptOnly] 当前子任务指令: {instruction}"]
        if next_waypoint:
            parts.append(f"[ManualPromptOnly] next_waypoint: {next_waypoint}")
        if direction:
            parts.append(f"[ManualPromptOnly] direction: {direction}")
        if landmark:
            parts.append(f"[ManualPromptOnly] landmark: {landmark}")
        parts.append(
            "[ManualPromptOnly] 请根据 prompt/image 手动操作机器人；"
            "完成后输入 a 回车继续；输入 f 结束当前 subtask 并回 planner: "
        )
        return "\n".join(parts)

    def _run_manual_prompt_only_action_controller(self, max_subtask_steps: int = 8) -> str:
        subtask_steps = 0
        while subtask_steps < int(max_subtask_steps or 8):
            if self._episode_done_cached():
                print("[WARN] Episode already done before manual prompt-only step", flush=True)
                return "complete"

            action_context = self._build_action_decision_context()
            if action_context is None:
                print("[ERR] Manual prompt-only failed to build action context", flush=True)
                return "thinking"

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
            allowed_action_names = self._apply_subtask_avoidance_side_lock_guard(allowed_action_names)

            prepared = self._prepare_manual_prompt_only_artifacts(
                action_context,
                allowed_action_names=allowed_action_names,
            )
            self._print_manual_prompt_only_artifacts(prepared)
            self._print_manual_prompt_only_subtask()
            self._write_real_live_status(
                event="manual_prompt_ready",
                phase=str(self._current_action_phase()),
                action="MANUAL_PROMPT_ONLY",
                extra={
                    "action_prompt_dir": str(prepared.get("save_dir") or ""),
                    "manual_prompt_only": True,
                },
            )

            while True:
                reply = self._read_real_operator_line(
                    self._manual_prompt_only_operator_prompt()
                ).strip().lower()
                if reply in {"a", "f"}:
                    break
                print("[ManualPromptOnly] 未确认：请输入 a 继续，或 f。", flush=True)

            if reply == "f":
                print("[ManualPromptOnly] operator finished current subtask; return to planner", flush=True)
                return "thinking"

            result = self.step_with_vlm(
                resolve_habitat_action("MOVE_FORWARD"),
                "MANUAL_PROMPT_ONLY",
                save_vis=True,
                enable_landmark_detection=False,
                env_action={
                    "action": resolve_habitat_action("MOVE_FORWARD"),
                    "manual_observation_only": True,
                    "phase": str(self._current_action_phase()),
                },
            )
            if bool((result or {}).get("done", False)):
                return "complete"
            subtask_steps += 1

        print(f"\n[ManualPromptOnly] Force replan after {max_subtask_steps} prompt-only steps", flush=True)
        return "thinking"

    def execute_rotation_sequence(self, action_sequence: Sequence[Dict]) -> bool:
        if not (self._real_manual_motion_mode() and self._real_manual_prompt_only_mode()):
            return super().execute_rotation_sequence(action_sequence)

        sequence = list(action_sequence or [])
        if not sequence:
            return True

        first_action = str(sequence[0].get("action", "") or "").strip().upper()
        if first_action not in {"TURN_LEFT", "TURN_RIGHT"}:
            return super().execute_rotation_sequence(action_sequence)
        if not all(str(item.get("action", "") or "").strip().upper() == first_action for item in sequence):
            return super().execute_rotation_sequence(action_sequence)

        total_degrees = sum(float(item.get("degrees", 0.0) or 0.0) for item in sequence)
        direction_text = "左转" if first_action == "TURN_LEFT" else "右转"
        print(
            "\n[ManualAlignTurn] planner 要先自动对准方向，但手动模式不会发布 /cmd_vel。",
            flush=True,
        )
        print(
            f"[ManualAlignTurn] >>> 请手动{direction_text} {float(total_degrees):.1f} deg",
            flush=True,
        )
        while True:
            reply = self._read_real_operator_line(
                "[ManualAlignTurn] 手动转向完成后输入 a 回车继续；输入 f 跳过本次对准并回到后续流程: "
            ).strip().lower()
            if reply in {"a", "f"}:
                break
            print("[ManualAlignTurn] 未确认：请输入 a 继续，或 f。", flush=True)
        if reply == "f":
            print("[ManualAlignTurn] operator skipped manual alignment turn", flush=True)
            return False

        action_id = (
            resolve_habitat_action("TURN_LEFT")
            if first_action == "TURN_LEFT"
            else resolve_habitat_action("TURN_RIGHT")
        )
        result = self.step_with_vlm(
            action_id,
            f"MANUAL_ALIGN_{first_action}",
            save_vis=True,
            enable_landmark_detection=False,
            env_action={
                "action": action_id,
                "manual_observation_only": True,
                "phase": str(self._current_action_phase()),
            },
        )
        if bool((result or {}).get("done", False)):
            print("[ManualAlignTurn] Episode ended while refreshing observation after manual turn", flush=True)
            return False
        return True

    def _run_action_controller(self, max_subtask_steps: int = 8) -> str:
        if self._real_manual_motion_mode() and self._real_manual_prompt_only_mode():
            return self._run_manual_prompt_only_action_controller(
                max_subtask_steps=max_subtask_steps,
            )
        return super()._run_action_controller(max_subtask_steps=max_subtask_steps)

    def _write_real_live_status(
        self,
        *,
        event: str,
        state: str = "running",
        phase: str = "",
        action: str = "",
        extra: Optional[Dict[str, Any]] = None,
        append_step: bool = True,
    ) -> None:
        if not bool(getattr(self, "real_live_status_enabled", False)):
            return
        try:
            self.real_live_status_seq = int(getattr(self, "real_live_status_seq", 0) or 0) + 1
            save_manager = getattr(self, "save_manager", None)
            paths_config = getattr(getattr(self, "config", None), "PATHS", None)
            results_dir = str(getattr(paths_config, "RESULTS_DIR", "") or "")
            payload = {
                "session_id": str(getattr(self, "real_session_id", "")),
                "state": str(state),
                "event": str(event),
                "seq": int(self.real_live_status_seq),
                "timestamp": datetime.now().isoformat(),
                "instruction": str(getattr(self, "real_instruction", "") or getattr(self, "current_instruction", "") or ""),
                "episode_id": int(getattr(self, "current_episode_id", 0) or 0),
                "step": int(getattr(self, "current_step", 0) or 0),
                "subtask_count": int(getattr(self, "subtask_count", 0) or 0),
                "phase": str(phase or ""),
                "action": str(action or ""),
                "metrics": self._compact_metrics(),
                "current_subtask": self._compact_subtask(),
                "detected_classes": list(getattr(self, "detected_classes", []) or []),
                "results_dir": results_dir,
                "episode_dir": str(getattr(save_manager, "episode_dir", "") or ""),
                "latest_global_map": str(getattr(self, "latest_global_map", "") or ""),
                "latest_local_map": str(getattr(self, "latest_local_map", "") or ""),
                "progress_summary": str(getattr(self, "progress_summary", "") or ""),
            }
            if extra:
                payload["extra"] = self._json_safe(dict(extra))

            paths = self._real_live_status_paths()
            for key in ("latest", "session", "episode_live", "records_live"):
                path = paths.get(key, "")
                if path:
                    self._atomic_write_json(path, payload)
            if append_step:
                for key in ("session_steps", "records_steps"):
                    path = paths.get(key, "")
                    if path:
                        self._append_jsonl(path, payload)

            status_suffix = ""
            action_status = None
            extra_info = payload.get("extra", {}).get("info", {}) if isinstance(payload.get("extra"), dict) else {}
            if isinstance(extra_info, dict):
                action_status = extra_info.get("action_status")
            if action_status is None:
                action_status = payload.get("metrics", {}).get("action_status") if isinstance(payload.get("metrics"), dict) else None
            if isinstance(action_status, dict) and action_status:
                status_suffix = " action_state=%s action_msg=%s" % (
                    str(action_status.get("state") or "unknown"),
                    str(action_status.get("message") or "-"),
                )
            print(
                "[REAL-LIVE] event=%s step=%s action=%s%s status=%s"
                % (event, payload["step"], action or "-", status_suffix, paths.get("latest", "")),
                flush=True,
            )
        except Exception as exc:
            print(f"[REAL-LIVE] failed to write live status: {exc}", flush=True)

    def reset_episode(self, *args, **kwargs):
        result = super().reset_episode(*args, **kwargs)
        self._real_step_rgb_dir_reported = False
        self._real_last_low_level_rgb_step = None
        self._real_last_low_level_rgb_stamp = None
        reset_obs = self._get_latest_real_observation()
        if isinstance(reset_obs, dict):
            self.latest_obs = reset_obs
            reset_rgb = self._save_real_step_rgb(event="episode_reset", step=0, obs=reset_obs)
            if reset_rgb:
                self._remember_real_low_level_rgb_endpoint(step=0, obs=reset_obs)
        self._write_real_live_status(event="episode_reset", append_step=False)
        return result

    def _persist_thinking_controller_response(
        self,
        response: Optional[Dict[str, Any]],
        cycle_info: Optional[Dict[str, Any]],
    ) -> None:
        super()._persist_thinking_controller_response(response, cycle_info)
        compact_response = {}
        if isinstance(response, dict):
            for key in (
                "global_task_finish",
                "current_waypoint",
                "next_waypoint",
                "next_waypoint_direction",
                "subtask_instruction",
            ):
                if key in response:
                    compact_response[key] = response.get(key)
        self._write_real_live_status(
            event="thinking_completed",
            phase=str((cycle_info or {}).get("phase", "") or ""),
            extra={"planner_response": compact_response},
            append_step=False,
        )

    def _on_lookaround_step(
        self,
        *,
        phase: str,
        look_index: int,
        look_step: int,
        obs: Dict[str, Any],
        info: Dict[str, Any],
    ) -> None:
        super()._on_lookaround_step(
            phase=phase,
            look_index=look_index,
            look_step=look_step,
            obs=obs,
            info=info,
        )
        self.latest_info = dict(info or {})
        action_text = (
            f"TURN_LEFT_{int(round(float(getattr(self, 'latest_lookaround_angle_step_deg', 45.0) or 45.0)))}"
            f"[{look_index}/{int(getattr(self, 'latest_lookaround_sample_count', 8) or 8)}]"
        )
        transition_rgb = self._save_real_transition_rgb_samples(
            current_step=look_step,
            current_obs=obs,
            action=action_text,
        )
        step_rgb = self._save_real_step_rgb(
            event="lookaround",
            step=look_step,
            obs=obs,
            action=action_text,
        )
        if step_rgb:
            self._remember_real_low_level_rgb_endpoint(step=look_step, obs=obs)
        extra = {"step_rgb": step_rgb} if step_rgb else {}
        if transition_rgb:
            extra["transition_rgb"] = list(transition_rgb)
        map_alignment = self._real_map_alignment_snapshot(obs=obs, info=info)
        if map_alignment:
            extra["map_alignment"] = map_alignment
        self._write_real_live_status(
            event="lookaround_step_processed",
            phase=phase,
            action=action_text,
            extra=extra or None,
        )

    def step_with_vlm(self, *args, **kwargs) -> Dict[str, Any]:
        action_name = str(kwargs.get("action_name", "") or "")
        if not action_name and len(args) >= 2:
            action_name = str(args[1] or "")
        result = super().step_with_vlm(*args, **kwargs)
        action_step = int(getattr(self, "current_step", 0) or 0)
        result_obs = (result or {}).get("obs") if isinstance(result, dict) else None
        transition_rgb = self._save_real_transition_rgb_samples(
            current_step=action_step,
            current_obs=result_obs if isinstance(result_obs, dict) else None,
            action=action_name,
        )
        step_rgb = self._save_real_step_rgb(
            event="action",
            step=action_step,
            obs=result_obs if isinstance(result_obs, dict) else None,
            action=action_name,
        )
        if step_rgb:
            self._remember_real_low_level_rgb_endpoint(
                step=action_step,
                obs=result_obs if isinstance(result_obs, dict) else None,
            )
        extra = {
            "done": bool((result or {}).get("done", False)),
            "info": (result or {}).get("info", {}),
        }
        if transition_rgb:
            extra["transition_rgb"] = list(transition_rgb)
        if step_rgb:
            extra["step_rgb"] = step_rgb
        map_alignment = self._real_map_alignment_snapshot(
            obs=result_obs if isinstance(result_obs, dict) else None,
            info=(result or {}).get("info", {}) if isinstance(result, dict) else None,
        )
        if map_alignment:
            extra["map_alignment"] = map_alignment
        self._write_real_live_status(
            event="action_step_processed",
            phase=str(self._current_action_phase()),
            action=action_name,
            extra=extra,
        )
        return result
