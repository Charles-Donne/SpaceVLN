"""Real-robot controller extensions that stream live progress to result files."""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np

from navigation_system.controller.agent.controller import NavigationAgentController


class RealNavigationAgentController(NavigationAgentController):
    """Navigation controller with real-only live result flushing."""

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
            "Use the 8 stopped RGB views and per-view depth obstacle labels.",
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
            "next_waypoint",
            "next_waypoint_direction",
            "global_task_finish",
        )
        return self._json_safe({key: subtask.get(key) for key in keys if key in subtask})

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

            print(
                "[REAL-LIVE] event=%s step=%s action=%s status=%s"
                % (event, payload["step"], action or "-", paths.get("latest", "")),
                flush=True,
            )
        except Exception as exc:
            print(f"[REAL-LIVE] failed to write live status: {exc}", flush=True)

    def reset_episode(self, *args, **kwargs):
        result = super().reset_episode(*args, **kwargs)
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
        self._write_real_live_status(
            event="lookaround_step_processed",
            phase=phase,
            action=(
                f"TURN_LEFT_{int(round(float(getattr(self, 'latest_lookaround_angle_step_deg', 45.0) or 45.0)))}"
                f"[{look_index}/{int(getattr(self, 'latest_lookaround_sample_count', 8) or 8)}]"
            ),
        )

    def step_with_vlm(self, *args, **kwargs) -> Dict[str, Any]:
        action_name = str(kwargs.get("action_name", "") or "")
        result = super().step_with_vlm(*args, **kwargs)
        if not action_name and len(args) >= 2:
            action_name = str(args[1] or "")
        self._write_real_live_status(
            event="action_step_processed",
            phase=str(self._current_action_phase()),
            action=action_name,
            extra={
                "done": bool((result or {}).get("done", False)),
                "info": (result or {}).get("info", {}),
            },
        )
        return result
