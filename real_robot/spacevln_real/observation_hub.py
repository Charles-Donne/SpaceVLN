"""Observation buffering and approximate synchronization for ROS streams."""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Deque, Optional

from spacevln_real.models import (
    CameraInfoData,
    ImageFrame,
    ImuFrame,
    PoseFrame,
    RealRobotConfig,
    RobotSnapshot,
)
from spacevln_real.ros_common import (
    camera_info_from_message,
    header_frame_id,
    header_stamp,
    image_msg_to_numpy,
    imu_from_message,
    normalize_depth_frame,
    pick_pose_for_snapshot,
    pose_from_odometry,
    pose_from_pose_stamped,
)


class ObservationHub:
    """Keeps a small synchronized window of camera, pose, and IMU messages."""

    def __init__(self, config: RealRobotConfig):
        self.config = config
        self._condition = threading.Condition()
        self._rgb_frames: Deque[ImageFrame] = deque(maxlen=max(2, config.image_queue_size))
        self._depth_frames: Deque[ImageFrame] = deque(maxlen=max(2, config.image_queue_size))
        self._odom_frames: Deque[PoseFrame] = deque(maxlen=max(8, config.pose_queue_size))
        self._pose_frames: Deque[PoseFrame] = deque(maxlen=max(8, config.pose_queue_size))
        self._imu_frames: Deque[ImuFrame] = deque(maxlen=max(8, config.pose_queue_size))
        self._rgb_camera_info: Optional[CameraInfoData] = None
        self._depth_camera_info: Optional[CameraInfoData] = None

    def _store(self, queue: Deque, item) -> None:
        with self._condition:
            queue.append(item)
            self._condition.notify_all()

    def _message_stamp(self, msg, receive_time: float) -> float:
        from spacevln_real.ros_common import header_stamp

        return header_stamp(
            msg,
            receive_time,
            timestamp_policy=str(self.config.timestamp_policy or "header"),
            max_header_receive_time_delta_s=float(
                self.config.max_header_receive_time_delta_s
            ),
        )

    def on_rgb(self, msg) -> None:
        receive_time = time.time()
        rgb = image_msg_to_numpy(msg)
        self._store(
            self._rgb_frames,
            ImageFrame(
                stamp=self._message_stamp(msg, receive_time),
                frame_id=header_frame_id(msg),
                encoding=str(getattr(msg, "encoding", "") or ""),
                image=rgb,
            ),
        )

    def on_depth(self, msg) -> None:
        receive_time = time.time()
        raw_depth = image_msg_to_numpy(msg)
        normalized_depth = normalize_depth_frame(
            raw_depth,
            encoding=str(getattr(msg, "encoding", "") or ""),
            min_depth_m=float(self.config.min_depth_m),
            max_depth_m=float(self.config.max_depth_m),
        )
        self._store(
            self._depth_frames,
            ImageFrame(
                stamp=self._message_stamp(msg, receive_time),
                frame_id=header_frame_id(msg),
                encoding=str(getattr(msg, "encoding", "") or ""),
                image=normalized_depth,
            ),
        )

    def on_rgb_camera_info(self, msg) -> None:
        with self._condition:
            receive_time = time.time()
            self._rgb_camera_info = camera_info_from_message(
                msg,
                fallback_stamp=receive_time,
                timestamp_policy=str(self.config.timestamp_policy or "header"),
                max_header_receive_time_delta_s=float(
                    self.config.max_header_receive_time_delta_s
                ),
            )
            self._condition.notify_all()

    def on_depth_camera_info(self, msg) -> None:
        with self._condition:
            receive_time = time.time()
            self._depth_camera_info = camera_info_from_message(
                msg,
                fallback_stamp=receive_time,
                timestamp_policy=str(self.config.timestamp_policy or "header"),
                max_header_receive_time_delta_s=float(
                    self.config.max_header_receive_time_delta_s
                ),
            )
            self._condition.notify_all()

    def on_odom(self, msg) -> None:
        receive_time = time.time()
        self._store(
            self._odom_frames,
            pose_from_odometry(
                msg,
                fallback_stamp=receive_time,
                timestamp_policy=str(self.config.timestamp_policy or "header"),
                max_header_receive_time_delta_s=float(
                    self.config.max_header_receive_time_delta_s
                ),
            ),
        )

    def on_pose(self, msg) -> None:
        receive_time = time.time()
        self._store(
            self._pose_frames,
            pose_from_pose_stamped(
                msg,
                fallback_stamp=receive_time,
                timestamp_policy=str(self.config.timestamp_policy or "header"),
                max_header_receive_time_delta_s=float(
                    self.config.max_header_receive_time_delta_s
                ),
            ),
        )

    def on_imu(self, msg) -> None:
        receive_time = time.time()
        self._store(
            self._imu_frames,
            imu_from_message(
                msg,
                fallback_stamp=receive_time,
                timestamp_policy=str(self.config.timestamp_policy or "header"),
                max_header_receive_time_delta_s=float(
                    self.config.max_header_receive_time_delta_s
                ),
            ),
        )

    @staticmethod
    def _pick_latest_after(queue: Deque, after_stamp: Optional[float]):
        if not queue:
            return None
        if after_stamp is None:
            return queue[-1]
        for item in reversed(queue):
            if float(item.stamp) > float(after_stamp):
                return item
        return None

    def _pick_closest(self, queue: Deque, target_stamp: Optional[float]):
        if not queue:
            return None
        if target_stamp is None:
            return queue[-1]
        best_item = None
        best_delta = None
        tolerance_s = float(self.config.sync_tolerance_s)
        for item in reversed(queue):
            delta = abs(float(item.stamp) - float(target_stamp))
            if best_delta is None or delta < best_delta:
                best_item = item
                best_delta = delta
            if float(item.stamp) <= float(target_stamp):
                break
        if best_item is None:
            return None
        if best_delta is not None and best_delta > tolerance_s:
            return None
        return best_item

    def _pose_queue(self) -> Deque[PoseFrame]:
        pose_source = str(self.config.pose_source or "odometry").strip().lower()
        if pose_source in {"pose", "pose_stamped"}:
            return self._pose_frames
        return self._odom_frames

    @staticmethod
    def _latest_stamp(queue: Deque) -> Optional[float]:
        if not queue:
            return None
        return float(queue[-1].stamp)

    def _format_sync_debug_locked(
        self,
        *,
        after_stamp: Optional[float],
    ) -> str:
        rgb_stamp = self._latest_stamp(self._rgb_frames)
        depth_stamp = self._latest_stamp(self._depth_frames)
        pose_queue = self._pose_queue()
        pose_stamp = self._latest_stamp(pose_queue)
        parts = [
            f"rgb_count={len(self._rgb_frames)}",
            f"depth_count={len(self._depth_frames)}",
            f"pose_source={str(self.config.pose_source or 'odometry')}",
            f"pose_count={len(pose_queue)}",
            f"imu_count={len(self._imu_frames)}",
            f"sync_tolerance_s={float(self.config.sync_tolerance_s):.3f}",
        ]
        if after_stamp is not None:
            parts.append(f"after_stamp={float(after_stamp):.3f}")
        if rgb_stamp is not None:
            parts.append(f"latest_rgb_stamp={rgb_stamp:.3f}")
        if depth_stamp is not None:
            parts.append(f"latest_depth_stamp={depth_stamp:.3f}")
        if pose_stamp is not None:
            parts.append(f"latest_pose_stamp={pose_stamp:.3f}")
        if rgb_stamp is not None and depth_stamp is not None:
            parts.append(f"rgb_depth_dt_s={abs(rgb_stamp - depth_stamp):.3f}")
        if rgb_stamp is not None and pose_stamp is not None:
            parts.append(f"rgb_pose_dt_s={abs(rgb_stamp - pose_stamp):.3f}")
        return ", ".join(parts)

    def wait_for_snapshot(
        self,
        *,
        after_stamp: Optional[float] = None,
        timeout_s: Optional[float] = None,
    ) -> RobotSnapshot:
        timeout = float(timeout_s or self.config.observation_timeout_s)
        deadline = time.time() + timeout
        with self._condition:
            while True:
                rgb = self._pick_latest_after(self._rgb_frames, after_stamp)
                if rgb is not None:
                    target_stamp = float(rgb.stamp)
                    depth = self._pick_closest(self._depth_frames, target_stamp)
                    pose = self._pick_closest(self._pose_queue(), target_stamp)
                    imu = self._pick_closest(self._imu_frames, target_stamp)
                    if depth is not None and pose is not None:
                        pose = pick_pose_for_snapshot(
                            pose,
                            imu,
                            use_imu_orientation=bool(self.config.use_imu_orientation),
                        )
                        return RobotSnapshot(
                            stamp=target_stamp,
                            rgb=rgb.image,
                            depth=depth.image,
                            pose=pose,
                            imu=imu,
                            rgb_camera_info=self._rgb_camera_info,
                            depth_camera_info=self._depth_camera_info,
                        )

                remaining = deadline - time.time()
                if remaining <= 0.0:
                    debug_state = self._format_sync_debug_locked(after_stamp=after_stamp)
                    raise TimeoutError(
                        "timed out waiting for synchronized real-robot snapshot "
                        f"({debug_state})"
                    )
                self._condition.wait(timeout=min(0.1, remaining))
