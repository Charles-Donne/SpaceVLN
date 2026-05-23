"""Typed models shared by the real-robot runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import uuid
from typing import Any, Dict, List, Optional


TERMINAL_ACTION_STATES = {
    "done",
    "failed",
    "timeout",
    "aborted",
    "emergency_stop",
}


@dataclass
class TopicConfig:
    rgb: str = "/camera/camera/color/image_raw"
    rgb_camera_info: str = "/camera/camera/color/camera_info"
    depth: str = "/camera/camera/aligned_depth_to_color/image_raw"
    depth_camera_info: str = "/camera/camera/aligned_depth_to_color/camera_info"
    imu: str = "/camera/camera/imu"
    odom: str = "/odom"
    pose: str = ""
    action_cmd: str = "/spacevln/action_cmd"
    action_status: str = "/spacevln/action_status"
    capture_request: str = "/spacevln/capture/request"

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "TopicConfig":
        payload = dict(raw or {})
        return cls(
            rgb=str(payload.get("rgb", cls.rgb)),
            rgb_camera_info=str(payload.get("rgb_camera_info", cls.rgb_camera_info)),
            depth=str(payload.get("depth", cls.depth)),
            depth_camera_info=str(payload.get("depth_camera_info", cls.depth_camera_info)),
            imu=str(payload.get("imu", cls.imu)),
            odom=str(payload.get("odom", cls.odom)),
            pose=str(payload.get("pose", "")),
            action_cmd=str(payload.get("action_cmd", cls.action_cmd)),
            action_status=str(payload.get("action_status", cls.action_status)),
            capture_request=str(payload.get("capture_request", cls.capture_request)),
        )


@dataclass
class RealRobotConfig:
    ros_version: str = "auto"
    node_name: str = "spacevln_real"
    pose_source: str = "odometry"
    capture_mode: str = "stream"
    timestamp_policy: str = "auto"
    max_header_receive_time_delta_s: float = 2.0
    use_imu_orientation: bool = False
    require_fresh_frame_after_action: bool = True
    action_status_required: bool = True
    observation_timeout_s: float = 3.0
    action_timeout_s: float = 20.0
    sync_tolerance_s: float = 0.2
    lookaround_sample_count: int = 8
    lookaround_angle_step_deg: float = 45.0
    disable_depth_map_update: bool = False
    depth_fusion_frames: int = 3
    selective_dynamic_obstacle_update: bool = True
    obstacle_evidence_threshold: float = 0.55
    obstacle_evidence_max_observations: int = 8
    image_queue_size: int = 8
    pose_queue_size: int = 32
    rgb_width: int = 640
    rgb_height: int = 480
    agent_height_m: float = 1.3
    camera_pitch_deg: float = -15.0
    hfov_deg: float = 87.0
    depth_hfov_deg: float = 87.0
    min_depth_m: float = 0.3
    max_depth_m: float = 3.0
    forward_step_m: float = 0.5
    turn_angle_deg: float = 45.0
    linear_speed_mps: float = 0.5
    angular_speed_deg_s: float = 60.0
    topics: TopicConfig = field(default_factory=TopicConfig)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "RealRobotConfig":
        payload = dict(raw or {})
        camera_cfg = dict(payload.get("camera", {}) or {})
        control_cfg = dict(payload.get("control", {}) or {})
        buffer_cfg = dict(payload.get("buffers", {}) or {})
        lookaround_cfg = dict(payload.get("lookaround", {}) or {})
        mapping_cfg = dict(payload.get("mapping", {}) or {})
        return cls(
            ros_version=str(payload.get("ros_version", "auto")).strip() or "auto",
            node_name=str(payload.get("node_name", "spacevln_real")).strip() or "spacevln_real",
            pose_source=str(payload.get("pose_source", "odometry")).strip() or "odometry",
            capture_mode=str(payload.get("capture_mode", "stream")).strip() or "stream",
            timestamp_policy=str(payload.get("timestamp_policy", "auto")).strip() or "auto",
            max_header_receive_time_delta_s=float(
                payload.get("max_header_receive_time_delta_s", 2.0)
            ),
            use_imu_orientation=bool(payload.get("use_imu_orientation", False)),
            require_fresh_frame_after_action=bool(
                payload.get("require_fresh_frame_after_action", True)
            ),
            action_status_required=bool(payload.get("action_status_required", True)),
            observation_timeout_s=float(payload.get("observation_timeout_s", 3.0)),
            action_timeout_s=float(payload.get("action_timeout_s", 20.0)),
            sync_tolerance_s=float(payload.get("sync_tolerance_s", 0.2)),
            lookaround_sample_count=max(
                1,
                int(lookaround_cfg.get("sample_count", payload.get("lookaround_sample_count", 8))),
            ),
            lookaround_angle_step_deg=float(
                lookaround_cfg.get("angle_step_deg", payload.get("lookaround_angle_step_deg", 45.0))
            ),
            disable_depth_map_update=bool(
                mapping_cfg.get(
                    "disable_depth_map_update",
                    payload.get("disable_depth_map_update", False),
                )
            ),
            depth_fusion_frames=max(
                1,
                int(mapping_cfg.get("depth_fusion_frames", payload.get("depth_fusion_frames", 3))),
            ),
            selective_dynamic_obstacle_update=bool(
                mapping_cfg.get(
                    "selective_dynamic_obstacle_update",
                    payload.get("selective_dynamic_obstacle_update", True),
                )
            ),
            obstacle_evidence_threshold=min(
                1.0,
                max(
                    0.0,
                    float(
                        mapping_cfg.get(
                            "obstacle_evidence_threshold",
                            payload.get("obstacle_evidence_threshold", 0.55),
                        )
                    ),
                ),
            ),
            obstacle_evidence_max_observations=max(
                0,
                int(
                    mapping_cfg.get(
                        "obstacle_evidence_max_observations",
                        payload.get("obstacle_evidence_max_observations", 8),
                    )
                ),
            ),
            image_queue_size=max(2, int(buffer_cfg.get("image_queue_size", 8))),
            pose_queue_size=max(8, int(buffer_cfg.get("pose_queue_size", 32))),
            rgb_width=max(1, int(camera_cfg.get("rgb_width", 640))),
            rgb_height=max(1, int(camera_cfg.get("rgb_height", 480))),
            agent_height_m=float(
                camera_cfg.get(
                    "agent_height_m",
                    camera_cfg.get("sensor_height_m", payload.get("agent_height_m", 1.3)),
                )
            ),
            camera_pitch_deg=float(
                camera_cfg.get(
                    "camera_pitch_deg",
                    camera_cfg.get("camera_elevation_deg", payload.get("camera_pitch_deg", -15.0)),
                )
            ),
            hfov_deg=float(camera_cfg.get("hfov_deg", 87.0)),
            depth_hfov_deg=float(
                camera_cfg.get(
                    "depth_hfov_deg",
                    payload.get("depth_hfov_deg", camera_cfg.get("hfov_deg", 87.0)),
                )
            ),
            min_depth_m=float(camera_cfg.get("min_depth_m", 0.3)),
            max_depth_m=float(camera_cfg.get("max_depth_m", 3.0)),
            forward_step_m=float(control_cfg.get("forward_step_m", 0.5)),
            turn_angle_deg=float(control_cfg.get("turn_angle_deg", 45.0)),
            linear_speed_mps=float(control_cfg.get("linear_speed_mps", 0.5)),
            angular_speed_deg_s=float(control_cfg.get("angular_speed_deg_s", 60.0)),
            topics=TopicConfig.from_dict(payload.get("topics")),
        )


@dataclass
class CameraInfoData:
    stamp: float
    frame_id: str
    width: int
    height: int
    k: List[float] = field(default_factory=list)
    d: List[float] = field(default_factory=list)


@dataclass
class ImageFrame:
    stamp: float
    frame_id: str
    encoding: str
    image: Any


@dataclass
class PoseFrame:
    stamp: float
    frame_id: str
    x: float
    y: float
    z: float
    yaw_rad: float


@dataclass
class ImuFrame:
    stamp: float
    frame_id: str
    yaw_rad: Optional[float]
    angular_velocity_z: float


@dataclass
class RobotSnapshot:
    stamp: float
    rgb: Any
    depth: Any
    pose: PoseFrame
    imu: Optional[ImuFrame] = None
    rgb_camera_info: Optional[CameraInfoData] = None
    depth_camera_info: Optional[CameraInfoData] = None


@dataclass
class ActionCommand:
    action: str
    forward_m: float = 0.0
    turn_deg: float = 0.0
    timeout_s: float = 0.0
    linear_speed_mps: float = 0.0
    angular_speed_deg_s: float = 0.0
    session_id: str = ""
    step_id: int = 0
    command_id: str = ""

    def ensure_command_id(self) -> str:
        if not self.command_id:
            self.command_id = str(uuid.uuid4())
        return self.command_id

    def to_payload(self) -> Dict[str, Any]:
        command_id = self.ensure_command_id()
        return {
            "session_id": self.session_id,
            "command_id": command_id,
            "step_id": int(self.step_id),
            "action": str(self.action),
            "target": {
                "meters": float(self.forward_m or 0.0),
                "degrees": float(self.turn_deg or 0.0),
            },
            "speed_hint": {
                "linear_mps": float(self.linear_speed_mps or 0.0),
                "angular_deg_s": float(self.angular_speed_deg_s or 0.0),
            },
            "timeout_s": float(self.timeout_s or 0.0),
            "stamp": time.time(),
        }


@dataclass
class ActionStatus:
    command_id: str
    session_id: str
    state: str
    success: bool
    stamp: float
    message: str = ""
    blocked: bool = False
    collision: bool = False
    goal_reached: bool = False
    done: bool = False
    distance_to_goal_m: Optional[float] = None
    executed_meters: float = 0.0
    executed_degrees: float = 0.0
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, raw: Dict[str, Any]) -> "ActionStatus":
        payload = dict(raw or {})
        executed = dict(payload.get("executed", {}) or {})
        state = str(payload.get("state", "done") or "done")
        done = bool(payload.get("done", state in TERMINAL_ACTION_STATES))
        distance_to_goal = payload.get("distance_to_goal_m")
        if distance_to_goal is not None:
            try:
                distance_to_goal = float(distance_to_goal)
            except (TypeError, ValueError):
                distance_to_goal = None
        return cls(
            command_id=str(payload.get("command_id", "")).strip(),
            session_id=str(payload.get("session_id", "")).strip(),
            state=state,
            success=bool(payload.get("success", state == "done")),
            stamp=float(payload.get("stamp", time.time())),
            message=str(payload.get("message", "") or ""),
            blocked=bool(payload.get("blocked", False)),
            collision=bool(payload.get("collision", False)),
            goal_reached=bool(payload.get("goal_reached", False)),
            done=done,
            distance_to_goal_m=distance_to_goal,
            executed_meters=float(executed.get("meters", 0.0) or 0.0),
            executed_degrees=float(executed.get("degrees", 0.0) or 0.0),
            raw_payload=payload,
        )

    def is_terminal(self) -> bool:
        return bool(self.done or self.state in TERMINAL_ACTION_STATES)
