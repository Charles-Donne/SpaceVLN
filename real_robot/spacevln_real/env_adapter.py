"""VectorEnv-like adapter that lets the existing controller drive a real robot."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spacevln_real.command_bridge import ActionCommandBridge
from spacevln_real.models import ActionCommand, ActionStatus, RealRobotConfig, RobotSnapshot
from spacevln_real.observation_hub import ObservationHub
from spacevln_real.ros_common import relative_pose_delta


ACTION_ID_TO_NAME = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}


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

    def current_episodes(self):
        return [self._current_episode]

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
                return action_name
        try:
            action_id = int(action)
        except Exception:
            action_id = -1
        return ACTION_ID_TO_NAME.get(action_id, "STOP")

    def _capture_if_needed(self, reason: str) -> None:
        if str(self.config.capture_mode or "stream").strip().lower() != "trigger":
            return
        self.command_bridge.publish_capture_request(
            session_id=self.session_id,
            reason=reason,
        )

    def _snapshot_to_obs(
        self,
        snapshot: RobotSnapshot,
        sensor_pose: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        pose = snapshot.pose
        return {
            "rgb": np.asarray(snapshot.rgb, dtype=np.uint8),
            "depth": np.asarray(snapshot.depth, dtype=np.float32),
            "sensor_pose": np.asarray(sensor_pose, dtype=np.float32),
            "position": np.asarray([pose.x, pose.z, pose.y], dtype=np.float32),
            "heading": np.asarray([pose.yaw_rad], dtype=np.float32),
            "timestamp": float(snapshot.stamp),
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
        }

    def reset(self):
        self._steps_taken = 0
        self._path_length_m = 0.0
        self._oracle_success = 0
        self._goal_seen = False
        self._min_distance_to_goal = float("inf")
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

    def _build_command(self, action_name: str) -> ActionCommand:
        command = ActionCommand(
            action=str(action_name),
            timeout_s=float(self.config.action_timeout_s),
            linear_speed_mps=float(self.config.linear_speed_mps),
            angular_speed_deg_s=float(self.config.angular_speed_deg_s),
            session_id=self.session_id,
            step_id=self._steps_taken + 1,
        )
        if action_name == "MOVE_FORWARD":
            command.forward_m = float(self.config.forward_step_m)
        elif action_name == "TURN_LEFT":
            command.turn_deg = float(abs(self.config.turn_angle_deg))
        elif action_name == "TURN_RIGHT":
            command.turn_deg = float(abs(self.config.turn_angle_deg))
        return command

    def step(self, actions):
        if not isinstance(actions, (list, tuple)) or not actions:
            raise ValueError("real robot env expects one action in a list")

        action_name = self._normalize_action(actions[0])
        before_snapshot = self._latest_snapshot
        if before_snapshot is None:
            self.reset()
            before_snapshot = self._latest_snapshot
        if before_snapshot is None:
            raise RuntimeError("real robot env has no initial observation")

        command = self._build_command(action_name)
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
        obs = self._snapshot_to_obs(after_snapshot, sensor_pose)
        return [(obs, 0.0, done, dict(self._latest_metrics))]

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._latest_metrics)

    def get_agent_pose(self) -> Tuple[float, float, float]:
        if self._latest_snapshot is None:
            raise RuntimeError("no real-robot observation is available yet")
        pose = self._latest_snapshot.pose
        return float(pose.x), float(pose.y), float(pose.yaw_rad)

    def call_at(self, index: int, method_name: str):
        if int(index) != 0:
            raise IndexError("real robot env only exposes index 0")
        target = getattr(self, method_name)
        return target()

    def close(self) -> None:
        try:
            self.ros_runtime.close()
        except Exception:
            pass
