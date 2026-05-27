"""ROS1/ROS2 runtime wrappers for the SpaceVLN real-robot bridge."""

from __future__ import annotations

import importlib.util
import json
import threading
from typing import Optional

from spacevln_real.command_bridge import ActionCommandBridge
from spacevln_real.models import RealRobotConfig
from spacevln_real.observation_hub import ObservationHub


def detect_ros_version(preferred: str) -> str:
    normalized = str(preferred or "auto").strip().lower()
    if normalized in {"ros1", "ros2"}:
        return normalized
    if importlib.util.find_spec("rclpy") is not None:
        return "ros2"
    if importlib.util.find_spec("rospy") is not None:
        return "ros1"
    raise RuntimeError("neither rclpy nor rospy is available")


class Ros1Runtime:
    def __init__(
        self,
        config: RealRobotConfig,
        observation_hub: ObservationHub,
        command_bridge: ActionCommandBridge,
    ):
        self.config = config
        self.observation_hub = observation_hub
        self.command_bridge = command_bridge
        self._rospy = None
        self._action_pub = None
        self._capture_pub = None
        self._subscribers = []

    def _publish_json(self, publisher, payload) -> None:
        if publisher is None:
            return
        from std_msgs.msg import String

        publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def start(self) -> None:
        import rospy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, Image, Imu
        from std_msgs.msg import String

        self._rospy = rospy
        if not rospy.core.is_initialized():
            rospy.init_node(self.config.node_name, anonymous=False, disable_signals=True)

        self._action_pub = rospy.Publisher(
            self.config.topics.action_cmd,
            String,
            queue_size=10,
        )
        if str(self.config.topics.capture_request or "").strip():
            self._capture_pub = rospy.Publisher(
                self.config.topics.capture_request,
                String,
                queue_size=10,
            )

        self.command_bridge.attach_publishers(
            publish_action=lambda payload: self._publish_json(self._action_pub, payload),
            publish_capture=(
                None
                if self._capture_pub is None
                else lambda payload: self._publish_json(self._capture_pub, payload)
            ),
        )

        if str(self.config.topics.rgb or "").strip():
            self._subscribers.append(
                rospy.Subscriber(
                    self.config.topics.rgb,
                    Image,
                    self.observation_hub.on_rgb,
                    queue_size=1,
                    buff_size=2 ** 24,
                )
            )
        if str(self.config.topics.depth or "").strip():
            self._subscribers.append(
                rospy.Subscriber(
                    self.config.topics.depth,
                    Image,
                    self.observation_hub.on_depth,
                    queue_size=1,
                    buff_size=2 ** 24,
                )
            )
        if str(self.config.topics.rgb_camera_info or "").strip():
            self._subscribers.append(
                rospy.Subscriber(
                    self.config.topics.rgb_camera_info,
                    CameraInfo,
                    self.observation_hub.on_rgb_camera_info,
                    queue_size=1,
                )
            )
        if str(self.config.topics.depth_camera_info or "").strip():
            self._subscribers.append(
                rospy.Subscriber(
                    self.config.topics.depth_camera_info,
                    CameraInfo,
                    self.observation_hub.on_depth_camera_info,
                    queue_size=1,
                )
            )
        if str(self.config.topics.imu or "").strip():
            self._subscribers.append(
                rospy.Subscriber(
                    self.config.topics.imu,
                    Imu,
                    self.observation_hub.on_imu,
                    queue_size=10,
                )
            )

        pose_source = str(self.config.pose_source or "odometry").strip().lower()
        if pose_source in {"pose", "pose_stamped"}:
            if not str(self.config.topics.pose or "").strip():
                raise ValueError("pose_source=pose_stamped requires topics.pose")
            self._subscribers.append(
                rospy.Subscriber(
                    self.config.topics.pose,
                    PoseStamped,
                    self.observation_hub.on_pose,
                    queue_size=10,
                )
            )
        else:
            self._subscribers.append(
                rospy.Subscriber(
                    self.config.topics.odom,
                    Odometry,
                    self.observation_hub.on_odom,
                    queue_size=20,
                )
            )

        if str(self.config.topics.action_status or "").strip():
            self._subscribers.append(
                rospy.Subscriber(
                    self.config.topics.action_status,
                    String,
                    self.command_bridge.on_action_status,
                    queue_size=20,
                )
            )

    def close(self) -> None:
        return None


class Ros2Runtime:
    def __init__(
        self,
        config: RealRobotConfig,
        observation_hub: ObservationHub,
        command_bridge: ActionCommandBridge,
    ):
        self.config = config
        self.observation_hub = observation_hub
        self.command_bridge = command_bridge
        self._rclpy = None
        self._node = None
        self._executor = None
        self._spin_thread: Optional[threading.Thread] = None
        self._owns_init = False
        self._action_pub = None
        self._capture_pub = None
        self._subscriptions = []

    def _publish_json(self, publisher, payload) -> None:
        if publisher is None:
            return
        from std_msgs.msg import String

        publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def start(self) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image, Imu
        from std_msgs.msg import String

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_init = True

        self._node = rclpy.create_node(self.config.node_name)
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self._node)

        qos_best_effort = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        qos_reliable = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_reliable_transient = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        qos_control = QoSProfile(depth=20)
        qos_camera_profiles = (
            qos_best_effort,
            qos_reliable,
            qos_reliable_transient,
        )
        qos_pose_profiles = (
            qos_best_effort,
            qos_reliable,
        )

        def subscribe(message_type, topic, callback, qos_profile):
            subscription = self._node.create_subscription(
                message_type,
                topic,
                callback,
                qos_profile,
            )
            self._subscriptions.append(subscription)
            return subscription

        def subscribe_many(message_type, topic, callback, qos_profiles):
            for qos_profile in qos_profiles:
                subscribe(message_type, topic, callback, qos_profile)

        self._action_pub = self._node.create_publisher(
            String,
            self.config.topics.action_cmd,
            qos_control,
        )
        if str(self.config.topics.capture_request or "").strip():
            self._capture_pub = self._node.create_publisher(
                String,
                self.config.topics.capture_request,
                qos_control,
            )

        self.command_bridge.attach_publishers(
            publish_action=lambda payload: self._publish_json(self._action_pub, payload),
            publish_capture=(
                None
                if self._capture_pub is None
                else lambda payload: self._publish_json(self._capture_pub, payload)
            ),
        )

        if str(self.config.topics.rgb or "").strip():
            subscribe_many(
                Image,
                self.config.topics.rgb,
                self.observation_hub.on_rgb,
                qos_camera_profiles,
            )
        if str(self.config.topics.depth or "").strip():
            subscribe_many(
                Image,
                self.config.topics.depth,
                self.observation_hub.on_depth,
                qos_camera_profiles,
            )
        if str(self.config.topics.rgb_camera_info or "").strip():
            subscribe_many(
                CameraInfo,
                self.config.topics.rgb_camera_info,
                self.observation_hub.on_rgb_camera_info,
                qos_camera_profiles,
            )
        if str(self.config.topics.depth_camera_info or "").strip():
            subscribe_many(
                CameraInfo,
                self.config.topics.depth_camera_info,
                self.observation_hub.on_depth_camera_info,
                qos_camera_profiles,
            )
        if str(self.config.topics.imu or "").strip():
            subscribe_many(
                Imu,
                self.config.topics.imu,
                self.observation_hub.on_imu,
                qos_pose_profiles,
            )

        pose_source = str(self.config.pose_source or "odometry").strip().lower()
        if pose_source in {"pose", "pose_stamped"}:
            if not str(self.config.topics.pose or "").strip():
                raise ValueError("pose_source=pose_stamped requires topics.pose")
            subscribe_many(
                PoseStamped,
                self.config.topics.pose,
                self.observation_hub.on_pose,
                qos_pose_profiles,
            )
        else:
            subscribe_many(
                Odometry,
                self.config.topics.odom,
                self.observation_hub.on_odom,
                qos_pose_profiles,
            )

        if str(self.config.topics.action_status or "").strip():
            subscribe(String, self.config.topics.action_status, self.command_bridge.on_action_status, qos_control)

        pose_topic = self.config.topics.pose if pose_source in {"pose", "pose_stamped"} else self.config.topics.odom
        print(
            "[REAL] ros2 subscriptions rgb=%s depth=%s rgb_info=%s depth_info=%s imu=%s pose_source=%s pose_topic=%s action_cmd=%s action_status=%s"
            % (
                str(self.config.topics.rgb or ""),
                str(self.config.topics.depth or ""),
                str(self.config.topics.rgb_camera_info or ""),
                str(self.config.topics.depth_camera_info or ""),
                str(self.config.topics.imu or ""),
                str(self.config.pose_source or "odometry"),
                str(pose_topic or ""),
                str(self.config.topics.action_cmd or ""),
                str(self.config.topics.action_status or ""),
            ),
            flush=True,
        )

        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        if self._rclpy is not None and self._owns_init and self._rclpy.ok():
            self._rclpy.shutdown()


def build_ros_runtime(
    config: RealRobotConfig,
    observation_hub: ObservationHub,
    command_bridge: ActionCommandBridge,
):
    ros_version = detect_ros_version(config.ros_version)
    if ros_version == "ros1":
        return Ros1Runtime(config, observation_hub, command_bridge)
    return Ros2Runtime(config, observation_hub, command_bridge)
