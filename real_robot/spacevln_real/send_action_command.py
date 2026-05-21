"""CLI helper for publishing one SpaceVLN real-robot action command."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String


TURN_ACTIONS = {"TURN_LEFT", "TURN_RIGHT", "LOOK_AROUND_360"}
ALLOWED_ACTIONS = {"MOVE_FORWARD", *TURN_ACTIONS, "STOP"}


class ActionCommandPublisher(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("spacevln_send_action_command")
        self.publisher = self.create_publisher(String, topic, QoSProfile(depth=10))

    def publish_payload(self, payload: dict) -> None:
        self.publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def build_payload(args: argparse.Namespace) -> dict:
    action = str(args.action or "").strip().upper()
    if action not in ALLOWED_ACTIONS:
        raise ValueError("unsupported action: %s" % action)

    target_meters = float(args.meters or 0.0)
    target_degrees = float(args.degrees or 0.0)
    if action == "MOVE_FORWARD" and target_meters <= 0.0:
        target_meters = 0.5
    if action in {"TURN_LEFT", "TURN_RIGHT"} and target_degrees <= 0.0:
        target_degrees = 30.0
    if action == "LOOK_AROUND_360" and target_degrees <= 0.0:
        target_degrees = 360.0

    return {
        "session_id": str(args.session_id or "manual"),
        "command_id": str(args.command_id or uuid.uuid4()),
        "step_id": int(args.step_id or 1),
        "action": action,
        "target": {
            "meters": max(target_meters, 0.0),
            "degrees": max(target_degrees, 0.0),
        },
        "speed_hint": {
            "linear_mps": float(args.linear_mps),
            "angular_deg_s": float(args.angular_deg_s),
        },
        "timeout_s": float(args.timeout_s),
        "stamp": time.time(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish one SpaceVLN action_cmd JSON message.")
    parser.add_argument("action", choices=sorted(ALLOWED_ACTIONS))
    parser.add_argument("--topic", default="/spacevln/action_cmd")
    parser.add_argument("--meters", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--linear-mps", type=float, default=0.15)
    parser.add_argument("--angular-deg-s", type=float, default=45.0)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--session-id", default="manual")
    parser.add_argument("--command-id", default="")
    parser.add_argument("--step-id", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = build_payload(args)

    rclpy.init(args=None)
    node = ActionCommandPublisher(str(args.topic))
    try:
        for _ in range(max(1, int(args.repeat or 1))):
            node.publish_payload(payload)
            rclpy.spin_once(node, timeout_sec=0.1)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
