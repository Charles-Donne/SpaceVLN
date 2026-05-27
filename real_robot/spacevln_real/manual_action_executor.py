"""Interactive action executor for manually driving the robot."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import sys
import time
from typing import Any, Dict, Optional, TextIO

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String

from spacevln_real.models import TERMINAL_ACTION_STATES
from spacevln_real.ros_common import parse_json_text


@dataclass
class ManualExecutorConfig:
    node_name: str = "spacevln_manual_action_executor"
    action_cmd_topic: str = "/spacevln/action_cmd"
    action_status_topic: str = "/spacevln/action_status"


class ManualActionExecutor(Node):
    def __init__(self, config: ManualExecutorConfig) -> None:
        super().__init__(config.node_name)
        self.config = config
        self._active_command_id: Optional[str] = None
        self._tty_in, self._tty_out = self._open_operator_tty()

        qos_control = QoSProfile(depth=20)
        self._action_status_pub = self.create_publisher(
            String,
            config.action_status_topic,
            qos_control,
        )
        self.create_subscription(
            String,
            config.action_cmd_topic,
            self.on_action_cmd,
            qos_control,
        )
        self._println("")
        self._println("[ManualExecutor] 手摇模式已启动：不会发布 /cmd_vel。")
        self._println("[ManualExecutor] 每次看到动作提示后，手动操作机器人；完成后按 Enter，agent 会继续下一步。")
        self._println("[ManualExecutor] 输入 f 后回车可把当前动作标记为 failed；输入 q 后回车标记 emergency_stop。")
        self._println("")

    @staticmethod
    def _open_operator_tty() -> tuple[TextIO, TextIO]:
        try:
            tty_in = open("/dev/tty", "r", encoding="utf-8")
            tty_out = open("/dev/tty", "w", encoding="utf-8", buffering=1)
            return tty_in, tty_out
        except OSError:
            return sys.stdin, sys.stdout

    def _println(self, text: str = "") -> None:
        print(text, file=self._tty_out, flush=True)

    def _readline(self, prompt: str) -> str:
        print(prompt, file=self._tty_out, end="", flush=True)
        line = self._tty_in.readline()
        return "" if line == "" else line.strip()

    @staticmethod
    def _float_value(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _format_motion(action: str, target: Dict[str, Any]) -> str:
        meters = ManualActionExecutor._float_value(target.get("meters"), 0.0)
        degrees = ManualActionExecutor._float_value(target.get("degrees"), 0.0)
        if action == "MOVE_FORWARD":
            return f"请手动向前走 {meters:.2f} m"
        if action == "TURN_LEFT":
            return f"请手动左转 {degrees:.1f} deg"
        if action == "TURN_RIGHT":
            return f"请手动右转 {degrees:.1f} deg"
        if action == "LOOK_AROUND_360":
            return f"请手动原地环视/旋转 {degrees:.1f} deg"
        if action == "STOP":
            return "请手动停止机器人，并确认当前位置就是本轮任务终点"
        return f"请手动执行动作 {action}"

    def _publish_status(
        self,
        *,
        session_id: str,
        command_id: str,
        state: str,
        success: bool,
        done: bool,
        message: str = "",
        executed_meters: float = 0.0,
        executed_degrees: float = 0.0,
    ) -> None:
        payload = {
            "session_id": str(session_id or ""),
            "command_id": str(command_id or ""),
            "state": str(state or ""),
            "success": bool(success),
            "done": bool(done or state in TERMINAL_ACTION_STATES),
            "blocked": False,
            "collision": False,
            "goal_reached": False,
            "distance_to_goal_m": None,
            "executed": {
                "meters": float(executed_meters or 0.0),
                "degrees": float(executed_degrees or 0.0),
            },
            "message": str(message or ""),
            "stamp": time.time(),
        }
        self._action_status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

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
        target = dict(payload.get("target", {}) or {})
        speed_hint = dict(payload.get("speed_hint", {}) or {})
        meters = self._float_value(target.get("meters"), 0.0)
        degrees = self._float_value(target.get("degrees"), 0.0)

        if not command_id:
            self.get_logger().warning("ignoring action command without command_id")
            return
        if self._active_command_id is not None:
            self._publish_status(
                session_id=session_id,
                command_id=command_id,
                state="failed",
                success=False,
                done=True,
                message="manual executor busy with another command",
            )
            return

        self._active_command_id = command_id
        self._publish_status(
            session_id=session_id,
            command_id=command_id,
            state="accepted",
            success=False,
            done=False,
            message="manual command accepted",
        )
        self._publish_status(
            session_id=session_id,
            command_id=command_id,
            state="running",
            success=False,
            done=False,
            message="waiting for operator confirmation",
        )

        self._println("")
        self._println("=" * 72)
        self._println(f"[ManualExecutor] step_id={step_id} command_id={command_id}")
        self._println(f"[ManualExecutor] action={action}")
        self._println(
            "[ManualExecutor] target: meters=%.2f degrees=%.1f | speed_hint: linear=%.2f angular=%.1f"
            % (
                meters,
                degrees,
                self._float_value(speed_hint.get("linear_mps"), 0.0),
                self._float_value(speed_hint.get("angular_deg_s"), 0.0),
            )
        )
        self._println(f"[ManualExecutor] >>> {self._format_motion(action, target)}")
        reply = self._readline(
            "[ManualExecutor] 手动完成后按 Enter 继续；输入 f=failed, q=emergency_stop: "
        )

        state = "done"
        success = True
        message = "operator confirmed manual action complete"
        if reply.lower() in {"f", "fail", "failed"}:
            state = "failed"
            success = False
            message = "operator marked manual action failed"
        elif reply.lower() in {"q", "quit", "stop", "emergency", "emergency_stop"}:
            state = "emergency_stop"
            success = False
            message = "operator requested emergency stop"

        self._publish_status(
            session_id=session_id,
            command_id=command_id,
            state=state,
            success=success,
            done=True,
            message=message,
            executed_meters=meters if action == "MOVE_FORWARD" and success else 0.0,
            executed_degrees=degrees if action in {"TURN_LEFT", "TURN_RIGHT", "LOOK_AROUND_360"} and success else 0.0,
        )
        self._println(f"[ManualExecutor] status={state}; 已通知 agent 继续。")
        self._println("=" * 72)
        self._println("")
        self._active_command_id = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive manual executor for SpaceVLN real-robot action commands."
    )
    parser.add_argument("--node-name", default="spacevln_manual_action_executor")
    parser.add_argument("--action-cmd-topic", default="/spacevln/action_cmd")
    parser.add_argument("--action-status-topic", default="/spacevln/action_status")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = ManualExecutorConfig(
        node_name=str(args.node_name),
        action_cmd_topic=str(args.action_cmd_topic),
        action_status_topic=str(args.action_status_topic),
    )

    rclpy.init(args=None)
    node = ManualActionExecutor(config)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        if "context is not valid" not in str(exc):
            raise
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
