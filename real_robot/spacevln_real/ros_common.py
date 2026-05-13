"""ROS-version-agnostic helpers for real-robot message parsing."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spacevln_real.models import CameraInfoData, ImuFrame, PoseFrame


def stamp_to_seconds(stamp: Any) -> float:
    if stamp is None:
        return 0.0
    to_sec = getattr(stamp, "to_sec", None)
    if callable(to_sec):
        try:
            return float(to_sec())
        except Exception:
            pass
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is not None:
        try:
            return float(sec) + float(nanosec or 0.0) * 1e-9
        except Exception:
            pass
    secs = getattr(stamp, "secs", None)
    nsecs = getattr(stamp, "nsecs", None)
    if secs is not None:
        try:
            return float(secs) + float(nsecs or 0.0) * 1e-9
        except Exception:
            pass
    return 0.0


def header_stamp(msg: Any, fallback: float) -> float:
    header = getattr(msg, "header", None)
    if header is None:
        return float(fallback)
    stamp = stamp_to_seconds(getattr(header, "stamp", None))
    return float(stamp or fallback)


def header_frame_id(msg: Any) -> str:
    header = getattr(msg, "header", None)
    if header is None:
        return ""
    return str(getattr(header, "frame_id", "") or "")


def normalize_angle_rad(angle_rad: float) -> float:
    angle = float(angle_rad)
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(quaternion_msg: Any) -> float:
    x = float(getattr(quaternion_msg, "x", 0.0) or 0.0)
    y = float(getattr(quaternion_msg, "y", 0.0) or 0.0)
    z = float(getattr(quaternion_msg, "z", 0.0) or 0.0)
    w = float(getattr(quaternion_msg, "w", 1.0) or 1.0)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return normalize_angle_rad(math.atan2(siny_cosp, cosy_cosp))


def pose_from_odometry(msg: Any, *, fallback_stamp: float = 0.0) -> PoseFrame:
    pose_container = getattr(msg, "pose", None)
    pose = getattr(pose_container, "pose", pose_container)
    position = getattr(pose, "position", None)
    orientation = getattr(pose, "orientation", None)
    return PoseFrame(
        stamp=header_stamp(msg, fallback_stamp),
        frame_id=header_frame_id(msg),
        x=float(getattr(position, "x", 0.0) or 0.0),
        y=float(getattr(position, "y", 0.0) or 0.0),
        z=float(getattr(position, "z", 0.0) or 0.0),
        yaw_rad=yaw_from_quaternion(orientation),
    )


def pose_from_pose_stamped(msg: Any, *, fallback_stamp: float = 0.0) -> PoseFrame:
    pose = getattr(msg, "pose", None)
    position = getattr(pose, "position", None)
    orientation = getattr(pose, "orientation", None)
    return PoseFrame(
        stamp=header_stamp(msg, fallback_stamp),
        frame_id=header_frame_id(msg),
        x=float(getattr(position, "x", 0.0) or 0.0),
        y=float(getattr(position, "y", 0.0) or 0.0),
        z=float(getattr(position, "z", 0.0) or 0.0),
        yaw_rad=yaw_from_quaternion(orientation),
    )


def imu_from_message(msg: Any, *, fallback_stamp: float = 0.0) -> ImuFrame:
    orientation = getattr(msg, "orientation", None)
    yaw_rad = None
    if orientation is not None:
        try:
            yaw_rad = yaw_from_quaternion(orientation)
        except Exception:
            yaw_rad = None
    angular_velocity = getattr(msg, "angular_velocity", None)
    return ImuFrame(
        stamp=header_stamp(msg, fallback_stamp),
        frame_id=header_frame_id(msg),
        yaw_rad=yaw_rad,
        angular_velocity_z=float(getattr(angular_velocity, "z", 0.0) or 0.0),
    )


def camera_info_from_message(msg: Any, *, fallback_stamp: float = 0.0) -> CameraInfoData:
    return CameraInfoData(
        stamp=header_stamp(msg, fallback_stamp),
        frame_id=header_frame_id(msg),
        width=int(getattr(msg, "width", 0) or 0),
        height=int(getattr(msg, "height", 0) or 0),
        k=list(getattr(msg, "k", []) or []),
        d=list(getattr(msg, "d", []) or []),
    )


def relative_pose_delta(prev_pose: PoseFrame, next_pose: PoseFrame) -> Tuple[float, float, float]:
    x1, y1, yaw1 = float(prev_pose.x), float(prev_pose.y), float(prev_pose.yaw_rad)
    x2, y2, yaw2 = float(next_pose.x), float(next_pose.y), float(next_pose.yaw_rad)
    delta_x_world = x2 - x1
    delta_y_world = y2 - y1
    theta = math.atan2(delta_y_world, delta_x_world) - yaw1
    distance = math.hypot(delta_x_world, delta_y_world)
    dx = distance * math.cos(theta)
    dy = distance * math.sin(theta)
    dyaw = normalize_angle_rad(yaw2 - yaw1)
    return float(dx), float(dy), float(dyaw)


def image_msg_to_numpy(msg: Any) -> np.ndarray:
    height = int(getattr(msg, "height", 0) or 0)
    width = int(getattr(msg, "width", 0) or 0)
    step = int(getattr(msg, "step", 0) or 0)
    is_bigendian = bool(getattr(msg, "is_bigendian", 0) or 0)
    encoding = str(getattr(msg, "encoding", "") or "").lower()
    raw = np.frombuffer(bytes(getattr(msg, "data", b"")), dtype=np.uint8)
    if not height or not width:
        raise ValueError("invalid image size")

    def rows_for(min_row_bytes: int) -> np.ndarray:
        row_bytes = int(step or min_row_bytes)
        if row_bytes < min_row_bytes:
            raise ValueError("image step is smaller than expected for %s" % encoding)
        needed = int(height) * row_bytes
        if raw.size < needed:
            raise ValueError("image data is shorter than height * step")
        return raw[:needed].reshape(height, row_bytes)

    if encoding in {"rgb8", "bgr8"}:
        row_bytes = width * 3
        array = rows_for(row_bytes)[:, :row_bytes].reshape(height, width, 3)
        if encoding == "bgr8":
            array = array[:, :, ::-1]
        return array.copy()

    if encoding in {"mono8", "8uc1"}:
        row_bytes = width
        return rows_for(row_bytes)[:, :row_bytes].reshape(height, width).copy()

    if encoding in {"mono16", "16uc1"}:
        row_bytes = width * 2
        dtype = np.dtype(">u2" if is_bigendian else "<u2")
        return (
            rows_for(row_bytes)[:, :row_bytes]
            .copy()
            .view(dtype)
            .reshape(height, width)
            .astype(np.uint16, copy=False)
        )

    if encoding in {"32fc1"}:
        row_bytes = width * 4
        dtype = np.dtype(">f4" if is_bigendian else "<f4")
        return (
            rows_for(row_bytes)[:, :row_bytes]
            .copy()
            .view(dtype)
            .reshape(height, width)
            .astype(np.float32, copy=False)
        )

    raise ValueError("unsupported image encoding: %s" % encoding)


def normalize_depth_frame(
    depth_image: np.ndarray,
    *,
    encoding: str,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    encoding_norm = str(encoding or "").lower()
    if encoding_norm in {"mono16", "16uc1"}:
        depth_m = depth_image.astype(np.float32) * 0.001
    elif encoding_norm in {"32fc1"}:
        depth_m = depth_image.astype(np.float32)
    else:
        raise ValueError("unsupported depth encoding: %s" % encoding)

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    clipped = np.where(valid, depth_m, float(max_depth_m))
    clipped = np.clip(clipped, float(min_depth_m), float(max_depth_m))
    normalized = (clipped - float(min_depth_m)) / max(
        float(max_depth_m) - float(min_depth_m),
        1e-6,
    )
    return normalized.astype(np.float32)[..., np.newaxis]


def pick_pose_for_snapshot(
    pose_frame: PoseFrame,
    imu_frame: Optional[ImuFrame],
    *,
    use_imu_orientation: bool,
) -> PoseFrame:
    if not use_imu_orientation or imu_frame is None or imu_frame.yaw_rad is None:
        return pose_frame
    return PoseFrame(
        stamp=pose_frame.stamp,
        frame_id=pose_frame.frame_id,
        x=pose_frame.x,
        y=pose_frame.y,
        z=pose_frame.z,
        yaw_rad=float(imu_frame.yaw_rad),
    )


def parse_json_text(text: str) -> Dict[str, Any]:
    import json

    payload = json.loads(str(text or "{}"))
    if not isinstance(payload, dict):
        raise ValueError("json payload must be an object")
    return payload
