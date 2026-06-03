"""VectorEnv-like adapter that lets the existing controller drive a real robot."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from spacevln_real.command_bridge import ActionCommandBridge
from spacevln_real.models import ActionCommand, ActionStatus, RealRobotConfig, RobotSnapshot
from spacevln_real.observation_hub import ObservationHub
from spacevln_real.ros_common import relative_pose_delta
from navigation_system.space.map.obstacle_analysis import sample_depth_distance_from_region


ACTION_ID_TO_NAME = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}

ACTION_NAME_ALIASES = {
    "STOP": "STOP",
    "MOVE_FORWARD": "MOVE_FORWARD",
    "FORWARD": "MOVE_FORWARD",
    "TURN_LEFT": "TURN_LEFT",
    "LEFT": "TURN_LEFT",
    "TURN_LEFT_ALIGN": "TURN_LEFT",
    "TURN_LEFT_AVOID": "TURN_LEFT",
    "TURN_RIGHT": "TURN_RIGHT",
    "RIGHT": "TURN_RIGHT",
    "TURN_RIGHT_ALIGN": "TURN_RIGHT",
    "TURN_RIGHT_AVOID": "TURN_RIGHT",
    "LOOK_AROUND_360": "LOOK_AROUND_360",
    "SCAN_360": "LOOK_AROUND_360",
}

REAL_MOVE_TARGETS_M = (0.5, 0.75, 1.0, 1.25, 1.5)
REAL_FORWARD_MIN_CLEARANCE_M = 0.5


@dataclass
class RealInstruction:
    instruction_text: str


@dataclass
class RealEpisode:
    episode_id: str
    instruction: RealInstruction
    reference_path: List[Tuple[float, float, float]] = field(default_factory=list)


class RealRobotVectorEnv:
    """Imitates the small subset of Habitat VectorEnv used by SpaceVLN."""

    def __init__(
        self,
        config: RealRobotConfig,
        observation_hub: ObservationHub,
        command_bridge: ActionCommandBridge,
        ros_runtime,
        *,
        instruction_text: str,
        session_id: str,
        episode_id: int = 0,
        success_distance_m: float = 3.0,
    ):
        self.config = config
        self.observation_hub = observation_hub
        self.command_bridge = command_bridge
        self.ros_runtime = ros_runtime
        self.instruction_text = str(instruction_text or "").strip()
        self.session_id = str(session_id or "spacevln-real-session")
        self.episode_id = int(episode_id or 0)
        self.success_distance_m = float(success_distance_m or 3.0)
        self.number_of_episodes = 1

        self._latest_snapshot: Optional[RobotSnapshot] = None
        self._latest_metrics: Dict[str, Any] = {}
        self._current_episode = RealEpisode(
            episode_id=str(self.episode_id),
            instruction=RealInstruction(instruction_text=self.instruction_text),
        )
        self._steps_taken = 0
        self._path_length_m = 0.0
        self._oracle_success = 0
        self._min_distance_to_goal = float("inf")
        self._goal_seen = False
        self._final_navigation_success: Optional[bool] = None
        self._reported_image_resize_shapes = set()

    def current_episodes(self):
        return [self._current_episode]

    def get_latest_observation(self) -> Optional[Dict[str, Any]]:
        if self._latest_snapshot is None:
            return None
        return self._snapshot_to_obs(self._latest_snapshot, (0.0, 0.0, 0.0))

    def get_rgb_samples_between(
        self,
        start_stamp: float,
        end_stamp: float,
        sample_count: int = 4,
    ) -> List[Dict[str, Any]]:
        frames = self.observation_hub.sample_rgb_frames_between(
            start_stamp=float(start_stamp),
            end_stamp=float(end_stamp),
            sample_count=int(sample_count or 0),
        )
        samples: List[Dict[str, Any]] = []
        for frame in frames:
            rgb = self._resize_frame_to_config(
                np.asarray(frame.image, dtype=np.uint8),
                name="rgb_sample",
                interpolation=cv2.INTER_AREA,
            ).astype(np.uint8, copy=False)
            samples.append(
                {
                    "rgb": rgb,
                    "timestamp": float(frame.stamp),
                    "rgb_timestamp": float(frame.stamp),
                    "frame_id": str(frame.frame_id),
                }
            )
        return samples

    def supports_continuous_action_targets(self) -> bool:
        return True

    def supports_continuous_lookaround_scan(self) -> bool:
        return True

    def get_lookaround_sample_count(self) -> int:
        return max(1, int(getattr(self.config, "lookaround_sample_count", 8) or 8))

    def get_lookaround_angle_step_deg(self) -> float:
        return float(
            getattr(self.config, "lookaround_angle_step_deg", 0.0)
            or getattr(self.config, "turn_angle_deg", 45.0)
            or 45.0
        )

    def _normalize_action(self, raw_action: Any) -> str:
        action = raw_action
        if isinstance(raw_action, dict) and "action" in raw_action:
            action = raw_action["action"]
        if hasattr(action, "value"):
            try:
                action = int(action.value)
            except Exception:
                pass
        if hasattr(action, "name"):
            action_name = str(getattr(action, "name", "") or "").strip().upper()
            if action_name:
                return ACTION_NAME_ALIASES.get(action_name, action_name)
        if isinstance(action, str):
            action_name = action.strip().upper().replace("-", "_")
            if action_name in ACTION_NAME_ALIASES:
                return ACTION_NAME_ALIASES[action_name]
        try:
            action_id = int(action)
        except Exception:
            action_id = -1
        return ACTION_ID_TO_NAME.get(action_id, "STOP")

    @staticmethod
    def _positive_float_or_none(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(parsed) or parsed <= 0.0:
            return None
        return float(parsed)

    @staticmethod
    def _quantize_move_target_meters(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        parsed = float(value)
        return min(REAL_MOVE_TARGETS_M, key=lambda allowed: abs(allowed - parsed))

    def _extract_action_targets(
        self,
        raw_action: Any,
        action_name: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(raw_action, dict):
            return None, None

        target = dict(raw_action.get("target", {}) or {})
        target_meters = self._positive_float_or_none(
            raw_action.get("target_meters", target.get("meters"))
        )
        target_degrees = self._positive_float_or_none(
            raw_action.get("target_degrees", target.get("degrees"))
        )
        if action_name != "MOVE_FORWARD":
            target_meters = None
        if action_name not in {"TURN_LEFT", "TURN_RIGHT", "LOOK_AROUND_360"}:
            target_degrees = None
        if action_name == "MOVE_FORWARD":
            target_meters = self._quantize_move_target_meters(target_meters)
        elif action_name in {"TURN_LEFT", "TURN_RIGHT"} and target_degrees is None:
            target_degrees = float(self.config.turn_angle_deg)
        return target_meters, target_degrees

    def _capture_if_needed(self, reason: str) -> None:
        if str(self.config.capture_mode or "stream").strip().lower() != "trigger":
            return
        self.command_bridge.publish_capture_request(
            session_id=self.session_id,
            reason=reason,
        )

    @staticmethod
    def _forward_min_clearance_m() -> float:
        raw_value = str(os.getenv("SPACEVLN_REAL_FORWARD_MIN_CLEARANCE_M", "") or "").strip()
        if raw_value:
            try:
                return max(0.0, float(raw_value))
            except (TypeError, ValueError):
                pass
        return float(REAL_FORWARD_MIN_CLEARANCE_M)

    def _front_clearance_m_from_depth(self, depth_meters: Any) -> Optional[float]:
        try:
            return sample_depth_distance_from_region(
                np.asarray(depth_meters, dtype=np.float32),
                center_x_ratio=0.5,
                width_ratio=0.26,
                row_start_ratio=0.38,
                row_end_ratio=0.92,
                max_distance_m=5.0,
                sensor_min_depth_m=float(self.config.min_depth_m),
                sample_count=128,
                sample_percentile=20.0,
            )
        except Exception:
            return None

    def _front_clearance_m_from_snapshot(self, snapshot: Optional[RobotSnapshot]) -> Optional[float]:
        if snapshot is None:
            return None
        return self._front_clearance_m_from_depth(getattr(snapshot, "depth", None))

    def _blocked_forward_status(self, clearance_m: float, threshold_m: float) -> ActionStatus:
        return ActionStatus(
            command_id="real_forward_safety_block",
            session_id=self.session_id,
            state="blocked",
            success=False,
            stamp=time.time(),
            message=(
                "real forward safety blocked MOVE_FORWARD: "
                f"front clearance {float(clearance_m):.2f}m < {float(threshold_m):.2f}m"
            ),
            done=True,
            blocked=True,
            collision=False,
            raw_payload={
                "session_id": self.session_id,
                "command_id": "real_forward_safety_block",
                "state": "blocked",
                "success": False,
                "done": True,
                "blocked": True,
                "message": (
                    "real forward safety blocked MOVE_FORWARD: "
                    f"front clearance {float(clearance_m):.2f}m < {float(threshold_m):.2f}m"
                ),
                "front_clearance_m": float(clearance_m),
                "threshold_m": float(threshold_m),
                "executed": {
                    "meters": 0.0,
                    "degrees": 0.0,
                },
            },
        )

    @staticmethod
    def _center_crop_to_aspect(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        if image.ndim < 2:
            return image
        height, width = image.shape[:2]
        if height <= 0 or width <= 0 or target_width <= 0 or target_height <= 0:
            return image

        source_aspect = float(width) / float(height)
        target_aspect = float(target_width) / float(target_height)
        if abs(source_aspect - target_aspect) < 1e-3:
            return image

        if source_aspect > target_aspect:
            crop_width = max(1, int(round(height * target_aspect)))
            x0 = max(0, (width - crop_width) // 2)
            return image[:, x0 : x0 + crop_width, ...]

        crop_height = max(1, int(round(width / target_aspect)))
        y0 = max(0, (height - crop_height) // 2)
        return image[y0 : y0 + crop_height, :, ...]

    def _resize_frame_to_config(
        self,
        image: np.ndarray,
        *,
        name: str,
        interpolation: int,
    ) -> np.ndarray:
        target_width = int(self.config.rgb_width)
        target_height = int(self.config.rgb_height)
        array = np.asarray(image)
        if array.ndim < 2:
            return array

        source_shape = tuple(array.shape)
        if array.shape[0] == target_height and array.shape[1] == target_width:
            return array

        cropped = self._center_crop_to_aspect(array, target_width, target_height)
        resized = cv2.resize(
            cropped,
            (target_width, target_height),
            interpolation=interpolation,
        )
        if array.ndim == 3 and array.shape[2] == 1 and resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        key = (str(name), source_shape, tuple(resized.shape))
        if key not in self._reported_image_resize_shapes:
            self._reported_image_resize_shapes.add(key)
            print(
                "[REAL] resized %s frame from %s to %s"
                % (str(name), source_shape, tuple(resized.shape)),
                flush=True,
            )
        return resized

    def _snapshot_to_obs(
        self,
        snapshot: RobotSnapshot,
        sensor_pose: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        pose = snapshot.pose
        rgb = self._resize_frame_to_config(
            np.asarray(snapshot.rgb, dtype=np.uint8),
            name="rgb",
            interpolation=cv2.INTER_AREA,
        ).astype(np.uint8, copy=False)
        depth = self._resize_frame_to_config(
            np.asarray(snapshot.depth, dtype=np.float32),
            name="depth",
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32, copy=False)
        if depth.ndim == 2:
            depth = depth[:, :, np.newaxis]
        return {
            "rgb": rgb,
            "depth": depth,
            "sensor_pose": np.asarray(sensor_pose, dtype=np.float32),
            "position": np.asarray([pose.x, pose.z, pose.y], dtype=np.float32),
            "heading": np.asarray([pose.yaw_rad], dtype=np.float32),
            "timestamp": float(snapshot.stamp),
            "rgb_timestamp": float(getattr(snapshot, "rgb_stamp", 0.0) or snapshot.stamp),
            "depth_timestamp": float(getattr(snapshot, "depth_stamp", 0.0) or snapshot.stamp),
            "pose_timestamp": float(getattr(snapshot, "pose_stamp", 0.0) or pose.stamp),
        }

    @staticmethod
    def _snapshot_sync_metrics(
        snapshot: RobotSnapshot,
        sensor_pose: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        rgb_stamp = float(getattr(snapshot, "rgb_stamp", 0.0) or snapshot.stamp)
        depth_stamp = float(getattr(snapshot, "depth_stamp", 0.0) or snapshot.stamp)
        pose_stamp = float(getattr(snapshot, "pose_stamp", 0.0) or snapshot.pose.stamp)
        sensor_pose_values = [float(value) for value in sensor_pose]
        return {
            "real_sensor_pose_delta": sensor_pose_values,
            "real_sensor_pose_delta_deg": float(math.degrees(sensor_pose_values[2])),
            "real_snapshot_sync": {
                "sync_stamp": float(snapshot.stamp),
                "rgb_stamp": rgb_stamp,
                "depth_stamp": depth_stamp,
                "pose_stamp": pose_stamp,
                "rgb_depth_dt_s": float(abs(rgb_stamp - depth_stamp)),
                "sync_pose_dt_s": float(abs(float(snapshot.stamp) - pose_stamp)),
            },
        }

    def _build_metrics(
        self,
        *,
        status: Optional[ActionStatus],
        stop_called: bool,
        done: bool,
    ) -> Dict[str, Any]:
        distance_to_goal = -1.0
        if status is not None and status.distance_to_goal_m is not None:
            distance_to_goal = float(status.distance_to_goal_m)
            self._min_distance_to_goal = min(self._min_distance_to_goal, distance_to_goal)

        goal_reached = False
        if status is not None:
            goal_reached = bool(status.goal_reached)
        if distance_to_goal >= 0.0 and distance_to_goal <= self.success_distance_m:
            goal_reached = True

        self._goal_seen = bool(self._goal_seen or goal_reached)
        self._oracle_success = max(self._oracle_success, int(goal_reached))
        success = int(bool(stop_called and goal_reached))
        success_source = "goal_status" if success else ""
        if stop_called and self._final_navigation_success is not None:
            success = int(bool(self._final_navigation_success))
            success_source = "model_global_task_finish" if success else "model_not_finished"

        oracle_navigation_error = (
            float(self._min_distance_to_goal)
            if np.isfinite(self._min_distance_to_goal)
            else float("inf")
        )

        return {
            "distance_to_goal": float(distance_to_goal),
            "success": int(success),
            "oracle_success": int(self._oracle_success),
            "oracle_navigation_error": float(oracle_navigation_error),
            "path_length": float(self._path_length_m),
            "spl": 0.0,
            "oracle_spl": 0.0,
            "ndtw": 0.0,
            "steps_taken": int(self._steps_taken),
            "done": bool(done),
            "top_down_map_vlnce": None,
            "goal_reached": bool(goal_reached),
            "collision": bool(status.collision) if status is not None else False,
            "blocked": bool(status.blocked) if status is not None else False,
            "message": str(status.message or "") if status is not None else "",
            "action_status": dict(status.raw_payload) if status is not None else {},
            "model_task_finished": bool(self._final_navigation_success)
            if self._final_navigation_success is not None
            else False,
            "success_source": success_source,
        }

    def reset(self):
        self._steps_taken = 0
        self._path_length_m = 0.0
        self._oracle_success = 0
        self._goal_seen = False
        self._min_distance_to_goal = float("inf")
        self._final_navigation_success = None
        self._capture_if_needed("episode_reset")
        self._latest_snapshot = self.observation_hub.wait_for_snapshot(
            timeout_s=float(self.config.observation_timeout_s),
        )
        self._latest_metrics = self._build_metrics(
            status=None,
            stop_called=False,
            done=False,
        )
        return [self._snapshot_to_obs(self._latest_snapshot, (0.0, 0.0, 0.0))]

    def _build_command(
        self,
        action_name: str,
        *,
        target_meters: Optional[float] = None,
        target_degrees: Optional[float] = None,
        manual_required: bool = False,
        phase: str = "",
    ) -> ActionCommand:
        command = ActionCommand(
            action=str(action_name),
            timeout_s=float(self.config.action_timeout_s),
            linear_speed_mps=float(self.config.linear_speed_mps),
            angular_speed_deg_s=float(self.config.angular_speed_deg_s),
            session_id=self.session_id,
            step_id=self._steps_taken + 1,
            manual_required=bool(manual_required),
            phase=str(phase or ""),
        )
        if action_name == "MOVE_FORWARD":
            command.forward_m = float(target_meters or self.config.forward_step_m)
        elif action_name == "TURN_LEFT":
            command.turn_deg = float(abs(target_degrees or self.config.turn_angle_deg))
        elif action_name == "TURN_RIGHT":
            command.turn_deg = float(abs(target_degrees or self.config.turn_angle_deg))
        elif action_name == "LOOK_AROUND_360":
            command.turn_deg = float(abs(target_degrees or 360.0))
        return command

    def run_lookaround_scan(
        self,
        *,
        sample_count: int = 12,
        angle_step_deg: Optional[float] = None,
        timeout_s: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], float, bool, Dict[str, Any]]]:
        sample_total = max(1, int(sample_count or self.get_lookaround_sample_count()))
        step_deg = float(angle_step_deg or self.get_lookaround_angle_step_deg())
        command_timeout = float(
            timeout_s
            or max(
                self.config.action_timeout_s,
                step_deg / max(float(self.config.angular_speed_deg_s or 1.0), 1.0) + 5.0,
            )
        )

        before_snapshot = self._latest_snapshot
        if before_snapshot is None:
            self.reset()
            before_snapshot = self._latest_snapshot
        if before_snapshot is None:
            raise RuntimeError("real robot env has no initial observation")

        outputs: List[Tuple[Dict[str, Any], float, bool, Dict[str, Any]]] = []
        previous_snapshot = before_snapshot
        scan_status: Optional[ActionStatus] = None
        for _sample_idx in range(sample_total):
            command = self._build_command(
                "TURN_LEFT",
                target_degrees=step_deg,
                manual_required=False,
                phase="lookaround",
            )
            command.timeout_s = command_timeout
            status = self.command_bridge.send_action(command)
            scan_status = status
            if not status.success:
                break

            self._capture_if_needed("lookaround_step_%02d" % (_sample_idx + 1))
            after_stamp = float(previous_snapshot.stamp)
            if bool(self.config.require_fresh_frame_after_action):
                after_stamp = max(after_stamp, float(status.stamp))
            snapshot = self.observation_hub.wait_for_snapshot(
                after_stamp=after_stamp,
                timeout_s=float(self.config.observation_timeout_s),
            )

            sensor_pose = relative_pose_delta(previous_snapshot.pose, snapshot.pose)
            self._path_length_m += float(math.hypot(sensor_pose[0], sensor_pose[1]))
            self._steps_taken += 1
            self._latest_snapshot = snapshot
            obs = self._snapshot_to_obs(snapshot, sensor_pose)
            metrics = self._build_metrics(
                status=scan_status,
                stop_called=False,
                done=False,
            )
            metrics.update(self._snapshot_sync_metrics(snapshot, sensor_pose))
            self._latest_metrics = dict(metrics)
            outputs.append((obs, 0.0, False, dict(metrics)))
            previous_snapshot = snapshot

        if len(outputs) < sample_total:
            raise TimeoutError(
                "stopped lookaround scan captured %d/%d samples; status=%s message=%s"
                % (
                    len(outputs),
                    sample_total,
                    str(getattr(scan_status, "state", "unknown")),
                    str(getattr(scan_status, "message", "")),
                )
            )

        return outputs

    def step(self, actions):
        if not isinstance(actions, (list, tuple)) or not actions:
            raise ValueError("real robot env expects one action in a list")

        raw_action = actions[0]
        action_name = self._normalize_action(raw_action)
        target_meters, target_degrees = self._extract_action_targets(
            raw_action,
            action_name,
        )
        manual_required = bool(raw_action.get("manual_required", False)) if isinstance(raw_action, dict) else False
        phase = str(raw_action.get("phase", "") or "") if isinstance(raw_action, dict) else ""
        before_snapshot = self._latest_snapshot
        if before_snapshot is None:
            self.reset()
            before_snapshot = self._latest_snapshot
        if before_snapshot is None:
            raise RuntimeError("real robot env has no initial observation")

        safety_blocked_forward = False
        front_clearance_m = None
        if action_name == "MOVE_FORWARD":
            front_clearance_m = self._front_clearance_m_from_snapshot(before_snapshot)
            min_clearance_m = self._forward_min_clearance_m()
            safety_blocked_forward = (
                front_clearance_m is not None
                and float(front_clearance_m) < float(min_clearance_m)
            )

        if safety_blocked_forward:
            status = self._blocked_forward_status(front_clearance_m, self._forward_min_clearance_m())
            after_snapshot = before_snapshot
            sensor_pose = (0.0, 0.0, 0.0)
            print(
                "[REAL-SAFETY] blocked MOVE_FORWARD: front_clearance=%.2fm < %.2fm"
                % (float(front_clearance_m), float(self._forward_min_clearance_m())),
                flush=True,
            )
        else:
            command = self._build_command(
                action_name,
                target_meters=target_meters,
                target_degrees=target_degrees,
                manual_required=manual_required,
                phase=phase,
            )
            status = self.command_bridge.send_action(command)

            self._capture_if_needed("post_%s" % action_name.lower())
            after_stamp = float(before_snapshot.stamp)
            if bool(self.config.require_fresh_frame_after_action):
                after_stamp = max(after_stamp, float(status.stamp))
            after_snapshot = self.observation_hub.wait_for_snapshot(
                after_stamp=after_stamp,
                timeout_s=float(self.config.observation_timeout_s),
            )

            sensor_pose = relative_pose_delta(before_snapshot.pose, after_snapshot.pose)

        self._path_length_m += float(math.hypot(sensor_pose[0], sensor_pose[1]))
        self._steps_taken += 1
        self._latest_snapshot = after_snapshot

        command_failed = str(status.state or "").strip().lower() in {
            "failed",
            "timeout",
            "aborted",
            "emergency_stop",
        }
        done = bool(action_name == "STOP" or command_failed)
        self._latest_metrics = self._build_metrics(
            status=status,
            stop_called=bool(action_name == "STOP"),
            done=done,
        )
        self._latest_metrics.update(self._snapshot_sync_metrics(after_snapshot, sensor_pose))
        obs = self._snapshot_to_obs(after_snapshot, sensor_pose)
        return [(obs, 0.0, done, dict(self._latest_metrics))]

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._latest_metrics)

    def get_agent_pose(self) -> Tuple[float, float, float]:
        if self._latest_snapshot is None:
            raise RuntimeError("no real-robot observation is available yet")
        pose = self._latest_snapshot.pose
        return float(pose.x), float(pose.y), float(pose.yaw_rad)

    def set_final_navigation_success(self, success: bool) -> bool:
        self._final_navigation_success = bool(success)
        return self._final_navigation_success

    def call_at(self, index: int, method_name: str, *args, **kwargs):
        if int(index) != 0:
            raise IndexError("real robot env only exposes index 0")
        target = getattr(self, method_name)
        return target(*args, **kwargs)

    def close(self) -> None:
        try:
            self.ros_runtime.close()
        except Exception:
            pass
