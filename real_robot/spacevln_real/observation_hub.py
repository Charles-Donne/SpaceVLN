"""Observation buffering and approximate synchronization for ROS streams."""

from __future__ import annotations

from collections import deque
import os
import threading
import time
from typing import Deque, Optional

import cv2
import numpy as np

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
        self._rgb_record_dir = ""
        self._rgb_record_interval_s = 1.0
        self._rgb_record_last_saved_at = 0.0
        self._rgb_record_index = 0
        self._first_rgb_logged = False
        self._first_depth_logged = False
        self._first_rgb_camera_info_logged = False
        self._first_depth_camera_info_logged = False
        self._first_odom_logged = False
        self._first_pose_logged = False
        self._first_imu_logged = False

    def configure_rgb_recording(
        self,
        *,
        output_dir: str,
        interval_s: float = 1.0,
    ) -> None:
        path = str(output_dir or "").strip()
        if not path:
            return
        os.makedirs(path, exist_ok=True)
        with self._condition:
            self._rgb_record_dir = path
            self._rgb_record_interval_s = max(0.1, float(interval_s or 1.0))
            self._rgb_record_last_saved_at = 0.0
            self._rgb_record_index = 0

    def _record_rgb_frame(self, rgb: np.ndarray, *, receive_time: float, stamp: float) -> None:
        with self._condition:
            output_dir = str(self._rgb_record_dir or "")
            interval_s = float(self._rgb_record_interval_s or 1.0)
            last_saved_at = float(self._rgb_record_last_saved_at or 0.0)
            if not output_dir or receive_time - last_saved_at < interval_s:
                return
            self._rgb_record_last_saved_at = float(receive_time)
            self._rgb_record_index += 1
            frame_index = int(self._rgb_record_index)

        filename = "rgb_%06d_stamp_%.3f.jpg" % (frame_index, float(stamp))
        path = os.path.join(output_dir, filename)
        try:
            cv2.imwrite(path, cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        except Exception as exc:
            print(f"[WARN] Failed to save real RGB frame {path}: {exc}", flush=True)

    def _store(self, queue: Deque, item) -> None:
        with self._condition:
            if queue:
                last = queue[-1]
                if (
                    abs(float(getattr(last, "stamp", -1.0)) - float(getattr(item, "stamp", -2.0))) < 1e-9
                    and str(getattr(last, "frame_id", "")) == str(getattr(item, "frame_id", ""))
                ):
                    return
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
        try:
            receive_time = time.time()
            rgb = image_msg_to_numpy(msg)
            stamp = self._message_stamp(msg, receive_time)
            if not self._first_rgb_logged:
                self._first_rgb_logged = True
                print(
                    "[REAL] first rgb frame stamp=%.3f frame_id=%s encoding=%s shape=%s"
                    % (
                        float(stamp),
                        header_frame_id(msg),
                        str(getattr(msg, "encoding", "") or ""),
                        tuple(np.asarray(rgb).shape),
                    ),
                    flush=True,
                )
            self._record_rgb_frame(rgb, receive_time=receive_time, stamp=stamp)
            self._store(
                self._rgb_frames,
                ImageFrame(
                    stamp=stamp,
                    frame_id=header_frame_id(msg),
                    encoding=str(getattr(msg, "encoding", "") or ""),
                    image=rgb,
                ),
            )
        except Exception as exc:
            print(f"[ERR] rgb callback failed: {exc}", flush=True)

    def on_depth(self, msg) -> None:
        try:
            receive_time = time.time()
            raw_depth = image_msg_to_numpy(msg)
            stamp = self._message_stamp(msg, receive_time)
            if not self._first_depth_logged:
                self._first_depth_logged = True
                print(
                    "[REAL] first depth frame stamp=%.3f frame_id=%s encoding=%s shape=%s"
                    % (
                        float(stamp),
                        header_frame_id(msg),
                        str(getattr(msg, "encoding", "") or ""),
                        tuple(np.asarray(raw_depth).shape),
                    ),
                    flush=True,
                )
            normalized_depth = normalize_depth_frame(
                raw_depth,
                encoding=str(getattr(msg, "encoding", "") or ""),
                min_depth_m=float(self.config.min_depth_m),
                max_depth_m=float(self.config.max_depth_m),
            )
            self._store(
                self._depth_frames,
                ImageFrame(
                    stamp=stamp,
                    frame_id=header_frame_id(msg),
                    encoding=str(getattr(msg, "encoding", "") or ""),
                    image=normalized_depth,
                ),
            )
        except Exception as exc:
            print(f"[ERR] depth callback failed: {exc}", flush=True)

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
            if not self._first_rgb_camera_info_logged:
                self._first_rgb_camera_info_logged = True
                print(
                    "[REAL] first rgb camera_info stamp=%.3f frame_id=%s size=%dx%d"
                    % (
                        float(self._rgb_camera_info.stamp),
                        str(self._rgb_camera_info.frame_id or ""),
                        int(self._rgb_camera_info.width),
                        int(self._rgb_camera_info.height),
                    ),
                    flush=True,
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
            if not self._first_depth_camera_info_logged:
                self._first_depth_camera_info_logged = True
                print(
                    "[REAL] first depth camera_info stamp=%.3f frame_id=%s size=%dx%d"
                    % (
                        float(self._depth_camera_info.stamp),
                        str(self._depth_camera_info.frame_id or ""),
                        int(self._depth_camera_info.width),
                        int(self._depth_camera_info.height),
                    ),
                    flush=True,
                )
            self._condition.notify_all()

    def on_odom(self, msg) -> None:
        try:
            receive_time = time.time()
            pose = pose_from_odometry(
                msg,
                fallback_stamp=receive_time,
                timestamp_policy=str(self.config.timestamp_policy or "header"),
                max_header_receive_time_delta_s=float(
                    self.config.max_header_receive_time_delta_s
                ),
            )
            if not self._first_odom_logged:
                self._first_odom_logged = True
                print(
                    "[REAL] first odom pose stamp=%.3f frame_id=%s pose=(%.3f, %.3f, %.3f) yaw_deg=%.1f"
                    % (
                        float(pose.stamp),
                        str(pose.frame_id or ""),
                        float(pose.x),
                        float(pose.y),
                        float(pose.z),
                        float(np.degrees(pose.yaw_rad)),
                    ),
                    flush=True,
                )
            self._store(self._odom_frames, pose)
        except Exception as exc:
            print(f"[ERR] odom callback failed: {exc}", flush=True)

    def on_pose(self, msg) -> None:
        try:
            receive_time = time.time()
            pose = pose_from_pose_stamped(
                msg,
                fallback_stamp=receive_time,
                timestamp_policy=str(self.config.timestamp_policy or "header"),
                max_header_receive_time_delta_s=float(
                    self.config.max_header_receive_time_delta_s
                ),
            )
            if not self._first_pose_logged:
                self._first_pose_logged = True
                print(
                    "[REAL] first pose frame stamp=%.3f frame_id=%s pose=(%.3f, %.3f, %.3f) yaw_deg=%.1f"
                    % (
                        float(pose.stamp),
                        str(pose.frame_id or ""),
                        float(pose.x),
                        float(pose.y),
                        float(pose.z),
                        float(np.degrees(pose.yaw_rad)),
                    ),
                    flush=True,
                )
            self._store(self._pose_frames, pose)
        except Exception as exc:
            print(f"[ERR] pose callback failed: {exc}", flush=True)

    def on_imu(self, msg) -> None:
        try:
            receive_time = time.time()
            imu = imu_from_message(
                msg,
                fallback_stamp=receive_time,
                timestamp_policy=str(self.config.timestamp_policy or "header"),
                max_header_receive_time_delta_s=float(
                    self.config.max_header_receive_time_delta_s
                ),
            )
            if not self._first_imu_logged:
                self._first_imu_logged = True
                print(
                    "[REAL] first imu frame stamp=%.3f frame_id=%s yaw_deg=%s"
                    % (
                        float(imu.stamp),
                        str(imu.frame_id or ""),
                        "nan" if imu.yaw_rad is None else f"{float(np.degrees(imu.yaw_rad)):.1f}",
                    ),
                    flush=True,
                )
            self._store(self._imu_frames, imu)
        except Exception as exc:
            print(f"[ERR] imu callback failed: {exc}", flush=True)

    @staticmethod
    def _iter_latest_after(queue: Deque, after_stamp: Optional[float]):
        for item in reversed(queue):
            if after_stamp is not None and float(item.stamp) <= float(after_stamp):
                continue
            yield item

    def _pick_closest(
        self,
        queue: Deque,
        target_stamp: Optional[float],
        *,
        after_stamp: Optional[float] = None,
    ):
        if not queue:
            return None
        if target_stamp is None:
            if after_stamp is None:
                return queue[-1]
            for item in reversed(queue):
                if float(item.stamp) > float(after_stamp):
                    return item
            return None
        best_item = None
        best_delta = None
        tolerance_s = float(self.config.sync_tolerance_s)
        for item in queue:
            if after_stamp is not None and float(item.stamp) <= float(after_stamp):
                continue
            delta = abs(float(item.stamp) - float(target_stamp))
            if best_delta is None or delta < best_delta:
                best_item = item
                best_delta = delta
        if best_item is None:
            return None
        if best_delta is not None and best_delta > tolerance_s:
            return None
        return best_item

    @staticmethod
    def _rgb_depth_sync_stamp(rgb: ImageFrame, depth: ImageFrame) -> float:
        return 0.5 * (float(rgb.stamp) + float(depth.stamp))

    def _fuse_depth_window_locked(
        self,
        depth: ImageFrame,
    ) -> Optional[ImageFrame]:
        frame_count = max(1, int(getattr(self.config, "depth_fusion_frames", 1) or 1))
        if frame_count <= 1:
            return depth
        if len(self._depth_frames) < frame_count:
            return None

        ordered = list(self._depth_frames)
        center_idx = next(
            (idx for idx, item in enumerate(ordered) if item is depth),
            None,
        )
        if center_idx is None:
            return depth

        before_count = frame_count // 2
        after_count = frame_count - before_count - 1
        start_idx = max(0, center_idx - before_count)
        end_idx = min(len(ordered), center_idx + after_count + 1)
        selected = ordered[start_idx:end_idx]
        if len(selected) < frame_count:
            return None

        arrays = []
        reference_shape = np.asarray(depth.image).shape
        for item in selected:
            array = np.asarray(item.image, dtype=np.float32)
            if array.shape != reference_shape:
                continue
            arrays.append(array)
        if len(arrays) < frame_count:
            return None

        stacked = np.stack(arrays, axis=0)
        valid = np.isfinite(stacked)
        sums = np.where(valid, stacked, 0.0).sum(axis=0)
        counts = valid.sum(axis=0)
        fused = np.divide(
            sums,
            np.maximum(counts, 1),
            out=np.full(reference_shape, np.nan, dtype=np.float32),
            where=counts > 0,
        )
        return ImageFrame(
            stamp=float(depth.stamp),
            frame_id=str(depth.frame_id),
            encoding=str(depth.encoding),
            image=fused.astype(np.float32, copy=False),
        )

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
                for rgb in self._iter_latest_after(self._rgb_frames, after_stamp):
                    depth = self._pick_closest(
                        self._depth_frames,
                        float(rgb.stamp),
                        after_stamp=after_stamp,
                    )
                    if depth is None:
                        continue

                    target_stamp = self._rgb_depth_sync_stamp(rgb, depth)
                    pose = self._pick_closest(
                        self._pose_queue(),
                        target_stamp,
                        after_stamp=after_stamp,
                    )
                    imu = self._pick_closest(
                        self._imu_frames,
                        target_stamp,
                        after_stamp=after_stamp,
                    )
                    if depth is not None and pose is not None:
                        depth = self._fuse_depth_window_locked(depth)
                        if depth is None:
                            continue
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
                            rgb_stamp=float(rgb.stamp),
                            depth_stamp=float(depth.stamp),
                            pose_stamp=float(pose.stamp),
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
