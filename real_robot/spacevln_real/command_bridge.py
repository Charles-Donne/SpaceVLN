"""JSON command/status bridge carried over standard ROS String topics."""

from __future__ import annotations

import json
import threading
import time
from typing import Callable, Dict, Optional

from spacevln_real.models import ActionCommand, ActionStatus, RealRobotConfig
from spacevln_real.ros_common import parse_json_text


class ActionCommandBridge:
    """Publishes commands and waits for matching terminal status messages."""

    def __init__(self, config: RealRobotConfig):
        self.config = config
        self._condition = threading.Condition()
        self._status_by_command: Dict[str, ActionStatus] = {}
        self._publish_action: Optional[Callable[[Dict], None]] = None
        self._publish_capture: Optional[Callable[[Dict], None]] = None

    def attach_publishers(
        self,
        *,
        publish_action: Callable[[Dict], None],
        publish_capture: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        self._publish_action = publish_action
        self._publish_capture = publish_capture

    def on_action_status(self, msg) -> None:
        try:
            payload = parse_json_text(getattr(msg, "data", ""))
            status = ActionStatus.from_payload(payload)
        except Exception:
            return

        if not status.command_id:
            return

        with self._condition:
            self._status_by_command[status.command_id] = status
            self._condition.notify_all()

    def publish_capture_request(self, *, session_id: str, reason: str) -> None:
        if self._publish_capture is None:
            return
        payload = {
            "session_id": str(session_id or ""),
            "reason": str(reason or ""),
            "stamp": time.time(),
        }
        self._publish_capture(payload)

    def wait_for_status(self, command_id: str, timeout_s: float) -> ActionStatus:
        deadline = time.time() + max(float(timeout_s or 0.0), 0.1)
        with self._condition:
            while True:
                status = self._status_by_command.get(command_id)
                if status is not None and status.is_terminal():
                    self._status_by_command.pop(command_id, None)
                    return status

                remaining = deadline - time.time()
                if remaining <= 0.0:
                    self._status_by_command.pop(command_id, None)
                    return ActionStatus(
                        command_id=str(command_id),
                        session_id="",
                        state="timeout",
                        success=False,
                        stamp=time.time(),
                        message="action status timeout",
                        done=True,
                    )
                self._condition.wait(timeout=min(0.1, remaining))

    def send_action(self, command: ActionCommand) -> ActionStatus:
        if self._publish_action is None:
            raise RuntimeError("action publisher is not attached")

        payload = command.to_payload()
        self._publish_action(payload)

        if not self.config.action_status_required:
            return ActionStatus(
                command_id=str(payload["command_id"]),
                session_id=str(payload.get("session_id", "")),
                state="done",
                success=True,
                stamp=time.time(),
                done=True,
                raw_payload=dict(payload),
            )

        return self.wait_for_status(
            str(payload["command_id"]),
            timeout_s=float(command.timeout_s or self.config.action_timeout_s),
        )

    @staticmethod
    def encode_json(payload: Dict) -> str:
        return json.dumps(payload, ensure_ascii=False)

