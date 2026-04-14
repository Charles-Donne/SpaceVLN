"""ROS2 demo bridge for integrating a low-level robot stack with SpaceVLN."""

from __future__ import annotations

import json
from typing import Dict

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


TERMINAL_STATES = {"done", "failed", "timeout", "aborted", "emergency_stop"}
ALLOWED_ACTIONS = {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"}


class SpaceVLNLowLevelBridgeDemo(Node):
    def __init__(self) -> None:
        super().__init__("spacevln_low_level_bridge_demo")
        self.action_status_pub = self.create_publisher(
            String,
            "/spacevln/action_status",
            20,
        )
        self.create_subscription(
            String,
            "/spacevln/action_cmd",
            self.on_action_cmd,
            20,
        )
        self.create_subscription(
            String,
            "/spacevln/capture/request",
            self.on_capture_request,
            20,
        )
        self._active_timers: Dict[str, object] = {}

    def on_capture_request(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            self.get_logger().warning("invalid capture request payload")
            return
        session_id = str(payload.get("session_id", "") or "")
        reason = str(payload.get("reason", "") or "")
        self.get_logger().info(
            f"capture request received: session_id={session_id} reason={reason}"
        )

    def on_action_cmd(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            self.get_logger().error("invalid action command JSON")
            return

        command_id = str(payload.get("command_id", "") or "").strip()
        session_id = str(payload.get("session_id", "") or "").strip()
        action = str(payload.get("action", "") or "").strip().upper()
        target = dict(payload.get("target", {}) or {})
        speed_hint = dict(payload.get("speed_hint", {}) or {})
        timeout_s = float(payload.get("timeout_s", 0.0) or 0.0)
        step_id = int(payload.get("step_id", 0) or 0)

        if not command_id:
            self.get_logger().error("missing command_id in action_cmd")
            return
        if action not in ALLOWED_ACTIONS:
            self.publish_status(
                command_id=command_id,
                session_id=session_id,
                state="failed",
                success=False,
                done=True,
                message=f"unsupported action: {action}",
            )
            return

        self.publish_status(
            command_id=command_id,
            session_id=session_id,
            state="accepted",
            success=False,
            done=False,
            message="command accepted",
        )

        self.publish_status(
            command_id=command_id,
            session_id=session_id,
            state="running",
            success=False,
            done=False,
            message="command executing",
        )

        target_meters = float(target.get("meters", 0.0) or 0.0)
        target_degrees = float(target.get("degrees", 0.0) or 0.0)
        linear_speed_mps = float(speed_hint.get("linear_mps", 0.0) or 0.0)
        angular_speed_deg_s = float(speed_hint.get("angular_deg_s", 0.0) or 0.0)

        self.get_logger().info(
            "action_cmd step_id=%s action=%s meters=%.3f degrees=%.3f linear=%.3f angular=%.3f timeout=%.3f"
            % (
                step_id,
                action,
                target_meters,
                target_degrees,
                linear_speed_mps,
                angular_speed_deg_s,
                timeout_s,
            )
        )

        if action == "STOP":
            self.publish_status(
                command_id=command_id,
                session_id=session_id,
                state="done",
                success=True,
                done=True,
                executed_meters=0.0,
                executed_degrees=0.0,
                message="stop executed",
            )
            return

        simulate_duration_s = self.estimate_duration_s(
            action=action,
            target_meters=target_meters,
            target_degrees=target_degrees,
            linear_speed_mps=linear_speed_mps,
            angular_speed_deg_s=angular_speed_deg_s,
        )
        timer = self.create_timer(
            simulate_duration_s,
            lambda: self.finish_action(
                command_id=command_id,
                session_id=session_id,
                action=action,
                target_meters=target_meters,
                target_degrees=target_degrees,
            ),
        )
        self._active_timers[command_id] = timer

    @staticmethod
    def estimate_duration_s(
        *,
        action: str,
        target_meters: float,
        target_degrees: float,
        linear_speed_mps: float,
        angular_speed_deg_s: float,
    ) -> float:
        if action == "MOVE_FORWARD":
            speed = max(float(linear_speed_mps or 0.0), 0.1)
            return max(float(target_meters or 0.0) / speed, 0.2)
        speed = max(float(angular_speed_deg_s or 0.0), 30.0)
        return max(float(target_degrees or 0.0) / speed, 0.2)

    def finish_action(
        self,
        *,
        command_id: str,
        session_id: str,
        action: str,
        target_meters: float,
        target_degrees: float,
    ) -> None:
        timer = self._active_timers.pop(command_id, None)
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

        executed_meters = target_meters if action == "MOVE_FORWARD" else 0.0
        executed_degrees = target_degrees if action in {"TURN_LEFT", "TURN_RIGHT"} else 0.0

        self.publish_status(
            command_id=command_id,
            session_id=session_id,
            state="done",
            success=True,
            done=True,
            executed_meters=executed_meters,
            executed_degrees=executed_degrees,
            message="demo action completed",
        )

    def publish_status(
        self,
        *,
        command_id: str,
        session_id: str,
        state: str,
        success: bool,
        done: bool,
        executed_meters: float = 0.0,
        executed_degrees: float = 0.0,
        blocked: bool = False,
        collision: bool = False,
        goal_reached: bool = False,
        distance_to_goal_m=None,
        message: str = "",
    ) -> None:
        payload = {
            "session_id": str(session_id or ""),
            "command_id": str(command_id or ""),
            "state": str(state or ""),
            "success": bool(success),
            "done": bool(done or state in TERMINAL_STATES),
            "blocked": bool(blocked),
            "collision": bool(collision),
            "goal_reached": bool(goal_reached),
            "distance_to_goal_m": distance_to_goal_m,
            "executed": {
                "meters": float(executed_meters or 0.0),
                "degrees": float(executed_degrees or 0.0),
            },
            "message": str(message or ""),
            "stamp": self.get_clock().now().nanoseconds / 1e9,
        }
        self.action_status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def main() -> None:
    rclpy.init(args=None)
    node = SpaceVLNLowLevelBridgeDemo()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
