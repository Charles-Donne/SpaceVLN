"""ROS2 closed-loop action executor that drives a base through /cmd_vel."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from spacevln_real.models import PoseFrame, TERMINAL_ACTION_STATES
from spacevln_real.ros_common import normalize_angle_rad, parse_json_text, pose_from_odometry


TURN_ACTIONS = {"TURN_LEFT", "TURN_RIGHT", "LOOK_AROUND_360"}
ALLOWED_ACTIONS = {"MOVE_FORWARD", *TURN_ACTIONS, "STOP"}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(value), float(upper)))


@dataclass
class ExecutorConfig:
    node_name: str = "spacevln_cmd_vel_executor"
    action_cmd_topic: str = "/spacevln/action_cmd"
    action_status_topic: str = "/spacevln/action_status"
    odom_topic: str = "/odom"
    cmd_vel_topic: str = "/cmd_vel"
    control_mode: str = "odom"
    control_rate_hz: float = 10.0
    odom_timeout_s: float = 0.5
    position_tolerance_m: float = 0.10
    angle_tolerance_deg: float = 10.0
    default_linear_speed_mps: float = 0.15
    default_angular_speed_deg_s: float = 45.0
    max_linear_speed_mps: float = 0.25
    max_angular_speed_deg_s: float = 60.0
    slowdown_distance_m: float = 0.08
    slowdown_angle_deg: float = 10.0
    heading_gain: float = 1.8
    max_heading_correction_rad_s: float = 0.35


@dataclass
class ActiveCommand:
    session_id: str
    command_id: str
    step_id: int
    action: str
    target_meters: float
    target_degrees: float
    linear_speed_mps: float
    angular_speed_deg_s: float
    timeout_s: float
    started_at: float
    deadline: float
    start_pose: Optional[PoseFrame]
    last_yaw_rad: float
    control_mode: str
    timed_duration_s: float = 0.0
    turn_progress_rad: float = 0.0


class CmdVelActionExecutor(Node):
    def __init__(self, config: ExecutorConfig) -> None:
        super().__init__(config.node_name)
        self.config = config
        self._latest_pose: Optional[PoseFrame] = None
        self._latest_odom_received_at: float = 0.0
        self._active_command: Optional[ActiveCommand] = None

        qos_sensor = QoSProfile(
            depth=20,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        qos_control = QoSProfile(depth=20)

        self._action_status_pub = self.create_publisher(
            String,
            config.action_status_topic,
            qos_control,
        )
        self._cmd_vel_pub = self.create_publisher(
            Twist,
            config.cmd_vel_topic,
            qos_control,
        )
        self.create_subscription(
            String,
            config.action_cmd_topic,
            self.on_action_cmd,
            qos_control,
        )
        self.create_subscription(
            Odometry,
            config.odom_topic,
            self.on_odom,
            qos_sensor,
        )
        self._control_timer = self.create_timer(
            1.0 / max(float(config.control_rate_hz), 1.0),
            self.on_control_tick,
        )

    def on_odom(self, msg: Odometry) -> None:
        self._latest_pose = pose_from_odometry(
            msg,
            fallback_stamp=self.now_s(),
        )
        self._latest_odom_received_at = self.now_s()

    def on_action_cmd(self, msg: String) -> None:
        try:
            payload = parse_json_text(msg.data)
        except Exception as exc:
            self.get_logger().error(f"invalid action_cmd JSON: {exc}")
            return

        session_id = str(payload.get("session_id", "") or "")
        command_id = str(payload.get("command_id", "") or "").strip()
        action = str(payload.get("action", "") or "").strip().upper()
        step_id = int(payload.get("step_id", 0) or 0)
        timeout_s = max(float(payload.get("timeout_s", 0.0) or 0.0), 0.1)
        target = dict(payload.get("target", {}) or {})
        speed_hint = dict(payload.get("speed_hint", {}) or {})

        if not command_id:
            self.get_logger().warning("ignoring action command without command_id")
            return

        if action not in ALLOWED_ACTIONS:
            self.publish_status(
                session_id=session_id,
                command_id=command_id,
                state="failed",
                success=False,
                done=True,
                message=f"unsupported action: {action}",
            )
            return

        if action == "STOP":
            if self._active_command is not None:
                interrupted = self._active_command
                self.stop_robot()
                self.publish_status(
                    session_id=interrupted.session_id,
                    command_id=interrupted.command_id,
                    state="aborted",
                    success=False,
                    done=True,
                    executed_meters=self.measure_forward_progress(interrupted),
                    executed_degrees=self.measure_turn_progress_deg(interrupted),
                    message="interrupted by STOP command",
                )
                self._active_command = None
            self.stop_robot()
            self.publish_status(
                session_id=session_id,
                command_id=command_id,
                state="done",
                success=True,
                done=True,
                message="stop executed",
            )
            return

        if self._active_command is not None:
            self.publish_status(
                session_id=session_id,
                command_id=command_id,
                state="failed",
                success=False,
                done=True,
                message="executor busy with another command",
            )
            return

        configured_control_mode = self.resolve_control_mode()
        control_mode = configured_control_mode
        if configured_control_mode == "auto":
            control_mode = "odom" if self._latest_pose is not None else "timed"
        if control_mode == "odom" and self._latest_pose is None:
            self.publish_status(
                session_id=session_id,
                command_id=command_id,
                state="failed",
                success=False,
                done=True,
                message="no odometry received yet",
            )
            return

        target_meters = max(float(target.get("meters", 0.0) or 0.0), 0.0)
        target_degrees = max(float(target.get("degrees", 0.0) or 0.0), 0.0)
        linear_speed_mps = float(speed_hint.get("linear_mps", 0.0) or 0.0)
        if linear_speed_mps <= 0.0:
            linear_speed_mps = float(self.config.default_linear_speed_mps)
        angular_speed_deg_s = float(speed_hint.get("angular_deg_s", 0.0) or 0.0)
        if angular_speed_deg_s <= 0.0:
            angular_speed_deg_s = float(self.config.default_angular_speed_deg_s)

        if action == "LOOK_AROUND_360" and target_degrees <= 0.0:
            target_degrees = 360.0
        timed_duration_s = self.estimate_timed_duration_s(
            action=action,
            target_meters=target_meters,
            target_degrees=target_degrees,
            linear_speed_mps=linear_speed_mps,
            angular_speed_deg_s=angular_speed_deg_s,
        )
        start_pose = self._latest_pose

        command = ActiveCommand(
            session_id=session_id,
            command_id=command_id,
            step_id=step_id,
            action=action,
            target_meters=target_meters,
            target_degrees=target_degrees,
            linear_speed_mps=linear_speed_mps,
            angular_speed_deg_s=angular_speed_deg_s,
            timeout_s=timeout_s,
            started_at=self.now_s(),
            deadline=self.now_s() + timeout_s,
            start_pose=start_pose,
            last_yaw_rad=float(start_pose.yaw_rad) if start_pose is not None else 0.0,
            control_mode=control_mode,
            timed_duration_s=timed_duration_s,
        )
        self._active_command = command

        self.publish_status(
            session_id=session_id,
            command_id=command_id,
            state="accepted",
            success=False,
            done=False,
            message="command accepted",
        )
        self.publish_status(
            session_id=session_id,
            command_id=command_id,
            state="running",
            success=False,
            done=False,
            message="command executing",
        )
        self.get_logger().info(
            "accepted step_id=%d action=%s meters=%.3f degrees=%.3f timeout=%.2f mode=%s duration=%.2f"
            % (
                command.step_id,
                command.action,
                command.target_meters,
                command.target_degrees,
                command.timeout_s,
                control_mode,
                command.timed_duration_s,
            )
        )

    def on_control_tick(self) -> None:
        command = self._active_command
        if command is None:
            return
        control_mode = str(command.control_mode or "odom")
        if control_mode == "odom" and self._latest_pose is None:
            self.finish_active_command(
                state="failed",
                success=False,
                message="odometry unavailable during execution",
            )
            return
        if control_mode == "odom" and not self.odom_is_fresh():
            self.finish_active_command(
                state="failed",
                success=False,
                message="odometry stale during execution",
            )
            return
        if self.now_s() >= command.deadline:
            self.finish_active_command(
                state="timeout",
                success=False,
                message="command timeout",
            )
            return

        if control_mode == "timed":
            self.control_timed(command)
            return

        if command.action == "MOVE_FORWARD":
            self.control_move_forward(command)
            return
        if command.action in TURN_ACTIONS:
            self.control_turn(command)
            return

        self.finish_active_command(
            state="failed",
            success=False,
            message=f"unhandled action: {command.action}",
        )

    def resolve_control_mode(self) -> str:
        mode = str(self.config.control_mode or "odom").strip().lower()
        if mode not in {"odom", "timed", "auto"}:
            return "odom"
        return mode

    def odom_is_fresh(self) -> bool:
        if self._latest_pose is None:
            return False
        if float(self.config.odom_timeout_s) <= 0.0:
            return True
        age_s = self.now_s() - float(self._latest_odom_received_at or 0.0)
        return age_s <= float(self.config.odom_timeout_s)

    @staticmethod
    def estimate_timed_duration_s(
        *,
        action: str,
        target_meters: float,
        target_degrees: float,
        linear_speed_mps: float,
        angular_speed_deg_s: float,
    ) -> float:
        if action == "MOVE_FORWARD":
            speed = max(float(linear_speed_mps or 0.0), 1e-3)
            return max(float(target_meters or 0.0) / speed, 0.0)
        if action in TURN_ACTIONS:
            speed = max(float(angular_speed_deg_s or 0.0), 1e-3)
            return max(float(target_degrees or 0.0) / speed, 0.0)
        return 0.0

    def control_timed(self, command: ActiveCommand) -> None:
        elapsed_s = max(0.0, self.now_s() - float(command.started_at))
        if elapsed_s >= float(command.timed_duration_s):
            self.finish_active_command(
                state="done",
                success=True,
                message="timed command complete",
            )
            return

        if command.action == "MOVE_FORWARD":
            linear_speed = min(
                float(command.linear_speed_mps),
                float(self.config.max_linear_speed_mps),
            )
            self.publish_twist(linear_x=max(linear_speed, 0.0), angular_z=0.0)
            return

        if command.action in TURN_ACTIONS:
            angular_speed_rad_s = math.radians(
                min(
                    float(command.angular_speed_deg_s),
                    float(self.config.max_angular_speed_deg_s),
                )
            )
            self.publish_twist(
                linear_x=0.0,
                angular_z=self.turn_direction(command) * angular_speed_rad_s,
            )
            return

        self.finish_active_command(
            state="failed",
            success=False,
            message=f"unhandled timed action: {command.action}",
        )

    def control_move_forward(self, command: ActiveCommand) -> None:
        progress_m = self.measure_forward_progress(command)
        remaining_m = float(command.target_meters) - float(progress_m)
        if remaining_m <= float(self.config.position_tolerance_m):
            self.finish_active_command(
                state="done",
                success=True,
                message="forward motion complete",
            )
            return

        yaw_error = normalize_angle_rad(
            float(self._latest_pose.yaw_rad) - float(command.start_pose.yaw_rad)
        )
        angular_correction = clamp(
            -float(self.config.heading_gain) * float(yaw_error),
            -float(self.config.max_heading_correction_rad_s),
            float(self.config.max_heading_correction_rad_s),
        )

        linear_speed = min(
            float(command.linear_speed_mps),
            float(self.config.max_linear_speed_mps),
        )
        if remaining_m < float(self.config.slowdown_distance_m):
            scale = remaining_m / max(float(self.config.slowdown_distance_m), 1e-6)
            linear_speed = max(0.05, linear_speed * max(scale, 0.2))

        self.publish_twist(linear_x=max(linear_speed, 0.0), angular_z=angular_correction)

    def control_turn(self, command: ActiveCommand) -> None:
        target_rad = math.radians(max(float(command.target_degrees), 0.0))
        progress_rad = self.update_turn_progress(command)
        remaining_rad = max(0.0, target_rad - progress_rad)

        if math.degrees(remaining_rad) <= float(self.config.angle_tolerance_deg):
            self.finish_active_command(
                state="done",
                success=True,
                message="turn complete",
            )
            return

        angular_speed_rad_s = math.radians(
            min(
                float(command.angular_speed_deg_s),
                float(self.config.max_angular_speed_deg_s),
            )
        )
        if math.degrees(remaining_rad) < float(self.config.slowdown_angle_deg):
            scale = math.degrees(remaining_rad) / max(
                float(self.config.slowdown_angle_deg),
                1e-6,
            )
            angular_speed_rad_s = max(
                math.radians(8.0),
                angular_speed_rad_s * max(scale, 0.2),
            )

        self.publish_twist(
            linear_x=0.0,
            angular_z=self.turn_direction(command) * angular_speed_rad_s,
        )

    @staticmethod
    def turn_direction(command: ActiveCommand) -> float:
        return -1.0 if command.action == "TURN_RIGHT" else 1.0

    def update_turn_progress(self, command: ActiveCommand) -> float:
        if self._latest_pose is None:
            return max(0.0, float(command.turn_progress_rad))
        current_yaw = float(self._latest_pose.yaw_rad)
        yaw_delta = normalize_angle_rad(current_yaw - float(command.last_yaw_rad))
        signed_progress_delta = yaw_delta * self.turn_direction(command)
        if signed_progress_delta > 0.0:
            command.turn_progress_rad += signed_progress_delta
        elif abs(math.degrees(signed_progress_delta)) > 1.0:
            command.turn_progress_rad = max(
                0.0,
                float(command.turn_progress_rad) + signed_progress_delta,
            )
        command.last_yaw_rad = current_yaw
        return max(0.0, float(command.turn_progress_rad))

    def measure_forward_progress(self, command: ActiveCommand) -> float:
        if command.control_mode == "timed" or command.start_pose is None:
            elapsed_s = max(0.0, min(self.now_s() - float(command.started_at), float(command.timed_duration_s)))
            return min(
                float(command.target_meters),
                max(0.0, float(command.linear_speed_mps) * elapsed_s),
            )
        if self._latest_pose is None:
            return 0.0
        dx_world = float(self._latest_pose.x) - float(command.start_pose.x)
        dy_world = float(self._latest_pose.y) - float(command.start_pose.y)
        heading = float(command.start_pose.yaw_rad)
        forward = dx_world * math.cos(heading) + dy_world * math.sin(heading)
        return max(0.0, float(forward))

    def measure_turn_progress_deg(self, command: ActiveCommand) -> float:
        if command.control_mode == "timed" or command.start_pose is None:
            elapsed_s = max(0.0, min(self.now_s() - float(command.started_at), float(command.timed_duration_s)))
            return min(
                float(command.target_degrees),
                max(0.0, float(command.angular_speed_deg_s) * elapsed_s),
            )
        if self._latest_pose is None:
            return 0.0
        if command.action not in TURN_ACTIONS:
            return 0.0
        yaw_delta = normalize_angle_rad(
            float(self._latest_pose.yaw_rad) - float(command.last_yaw_rad)
        )
        pending_progress = max(0.0, yaw_delta * self.turn_direction(command))
        return math.degrees(max(0.0, float(command.turn_progress_rad) + pending_progress))

    def finish_active_command(self, *, state: str, success: bool, message: str) -> None:
        command = self._active_command
        if command is None:
            return
        executed_meters = self.measure_forward_progress(command)
        executed_degrees = self.measure_turn_progress_deg(command)
        self.stop_robot()
        self.publish_status(
            session_id=command.session_id,
            command_id=command.command_id,
            state=state,
            success=success,
            done=True,
            executed_meters=executed_meters,
            executed_degrees=executed_degrees,
            message=message,
        )
        self.get_logger().info(
            "finished step_id=%d action=%s state=%s meters=%.3f degrees=%.3f"
            % (
                command.step_id,
                command.action,
                state,
                executed_meters,
                executed_degrees,
            )
        )
        self._active_command = None

    def publish_status(
        self,
        *,
        session_id: str,
        command_id: str,
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
            "done": bool(done or state in TERMINAL_ACTION_STATES),
            "blocked": bool(blocked),
            "collision": bool(collision),
            "goal_reached": bool(goal_reached),
            "distance_to_goal_m": distance_to_goal_m,
            "executed": {
                "meters": float(executed_meters or 0.0),
                "degrees": float(executed_degrees or 0.0),
            },
            "message": str(message or ""),
            "stamp": self.now_s(),
        }
        self._action_status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def publish_twist(self, *, linear_x: float, angular_z: float) -> None:
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(twist)

    def stop_robot(self) -> None:
        self.publish_twist(linear_x=0.0, angular_z=0.0)

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Closed-loop ROS2 action executor for SpaceVLN using /cmd_vel and /odom."
    )
    parser.add_argument("--node-name", default="spacevln_cmd_vel_executor")
    parser.add_argument("--action-cmd-topic", default="/spacevln/action_cmd")
    parser.add_argument("--action-status-topic", default="/spacevln/action_status")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument(
        "--control-mode",
        choices=("odom", "timed", "auto"),
        default="odom",
        help="odom uses feedback; timed publishes speed for target/speed seconds; auto uses odom when available.",
    )
    parser.add_argument("--control-rate-hz", type=float, default=10.0)
    parser.add_argument("--odom-timeout-s", type=float, default=0.5)
    parser.add_argument("--position-tolerance-m", type=float, default=0.10)
    parser.add_argument("--angle-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--default-linear-speed-mps", type=float, default=0.15)
    parser.add_argument("--default-angular-speed-deg-s", type=float, default=45.0)
    parser.add_argument("--max-linear-speed-mps", type=float, default=0.25)
    parser.add_argument("--max-angular-speed-deg-s", type=float, default=60.0)
    return parser


def config_from_args(args: argparse.Namespace) -> ExecutorConfig:
    return ExecutorConfig(
        node_name=str(args.node_name),
        action_cmd_topic=str(args.action_cmd_topic),
        action_status_topic=str(args.action_status_topic),
        odom_topic=str(args.odom_topic),
        cmd_vel_topic=str(args.cmd_vel_topic),
        control_mode=str(args.control_mode),
        control_rate_hz=float(args.control_rate_hz),
        odom_timeout_s=float(args.odom_timeout_s),
        position_tolerance_m=float(args.position_tolerance_m),
        angle_tolerance_deg=float(args.angle_tolerance_deg),
        default_linear_speed_mps=float(args.default_linear_speed_mps),
        default_angular_speed_deg_s=float(args.default_angular_speed_deg_s),
        max_linear_speed_mps=float(args.max_linear_speed_mps),
        max_angular_speed_deg_s=float(args.max_angular_speed_deg_s),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    rclpy.init(args=None)
    node = CmdVelActionExecutor(config)
    try:
        rclpy.spin(node)
    finally:
        try:
            node.stop_robot()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
