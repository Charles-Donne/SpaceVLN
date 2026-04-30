# SpaceVLN Real-Robot ROS Interface

This document is the integration contract for the low-level robotics stack.

## 1. Sensor Inputs Consumed by SpaceVLN

### 1.1 Camera Topics

#### RGB image

- Topic: `/oak/rgb/image_raw`
- Type: `sensor_msgs/Image`
- Recommended encoding: `rgb8` or `bgr8`
- Recommended resolution: `640x480`

#### RGB camera info

- Topic: `/oak/rgb/camera_info`
- Type: `sensor_msgs/CameraInfo`

#### Depth image

- Topic: `/oak/stereo/image_raw`
- Type: `sensor_msgs/Image`
- Supported encodings:
  - `16UC1` in millimeters
  - `32FC1` in meters

If the actual OAK-D Lite driver publishes a different aligned-depth topic, such as
`/oak/stereo/depth` or `/oak/rgbd/depth/image`, update
`real_robot/config/real_robot.yaml`.

#### Depth camera info

- Topic: `/oak/stereo/camera_info`
- Type: `sensor_msgs/CameraInfo`

### 1.2 Pose Input

Preferred source:

- Topic: `/odom`
- Type: `nav_msgs/Odometry`

Alternative source:

- Topic: `/spacevln/pose`
- Type: `geometry_msgs/PoseStamped`
- Set `pose_source: pose_stamped` in `real_robot/config/real_robot.yaml`

The runtime extracts:

- planar position: `x`, `y`
- height: `z`
- orientation: `yaw`

It then converts them into the relative motion format required by the existing
mapping pipeline:

- `sensor_pose = [dx, dy, dtheta]`

### 1.3 IMU Input

- Topic: `/oak/imu/data`
- Type: `sensor_msgs/Imu`

Usage:

- cached by default for runtime state and debugging
- can override odometry yaw when `use_imu_orientation: true`

## 2. Action Command Interface Published by SpaceVLN

To keep ROS2 integration lightweight, the runtime uses:

- Topic: `/spacevln/action_cmd`
- Type: `std_msgs/String`
- Payload: JSON string

### 2.1 `action_cmd` JSON schema

```json
{
  "session_id": "2f94b3b5-9c7d-4ef5-89dd-79a18c1b2bdb",
  "command_id": "fd2d1ef4-cf6a-4483-a337-594ab4f0c1ab",
  "step_id": 17,
  "action": "MOVE_FORWARD",
  "target": {
    "meters": 0.25,
    "degrees": 0.0
  },
  "speed_hint": {
    "linear_mps": 0.15,
    "angular_deg_s": 45.0
  },
  "timeout_s": 20.0,
  "stamp": 1713091200.123
}
```

### 2.2 Allowed action values

- `MOVE_FORWARD`
- `TURN_LEFT`
- `TURN_RIGHT`
- `STOP`

### 2.3 Execution expectations

The low-level controller should execute each command as a closed-loop motion:

- `MOVE_FORWARD`: drive toward the requested distance
- `TURN_LEFT` / `TURN_RIGHT`: rotate toward the requested angle
- `STOP`: stop immediately and publish a terminal status

This should not be implemented as a single open-loop velocity pulse.

## 3. Action Status Interface Published by the Low-Level Stack

- Topic: `/spacevln/action_status`
- Type: `std_msgs/String`
- Payload: JSON string

### 3.1 `action_status` JSON schema

```json
{
  "session_id": "2f94b3b5-9c7d-4ef5-89dd-79a18c1b2bdb",
  "command_id": "fd2d1ef4-cf6a-4483-a337-594ab4f0c1ab",
  "state": "done",
  "success": true,
  "done": true,
  "blocked": false,
  "collision": false,
  "goal_reached": false,
  "distance_to_goal_m": null,
  "executed": {
    "meters": 0.24,
    "degrees": 0.0
  },
  "message": "ok",
  "stamp": 1713091201.456
}
```

### 3.2 Recommended `state` values

- `accepted`
- `running`
- `done`
- `failed`
- `timeout`
- `aborted`
- `emergency_stop`

The runtime waits until it receives a terminal status for the matching
`command_id`:

- `done`
- `failed`
- `timeout`
- `aborted`
- `emergency_stop`

### 3.3 Required fields

At minimum, publish:

- `command_id`
- `state`
- `success`
- `done`
- `stamp`

### 3.4 Recommended fields

Strongly recommended:

- `executed.meters`
- `executed.degrees`
- `blocked`
- `collision`
- `goal_reached`
- `distance_to_goal_m`
- `message`

If the low-level stack does not estimate goal distance, use:

- `goal_reached: false`
- `distance_to_goal_m: null`

## 3.5 Recommended Base-Control Pattern for ROS2 Robots

If your robot already exposes the standard ROS2 base-control topic:

- Topic: `/cmd_vel`
- Type: `geometry_msgs/Twist`

do not make `SpaceVLN` publish `/cmd_vel` directly as an open-loop pulse.

Recommended split:

1. `SpaceVLN` publishes one discrete action on `/spacevln/action_cmd`
2. A ROS2 executor node subscribes to `/spacevln/action_cmd`
3. That executor also subscribes to `/odom`
4. The executor publishes `geometry_msgs/Twist` on `/cmd_vel`
5. When the target distance or angle is reached, the executor publishes `/spacevln/action_status`

This keeps the high-level planner discrete and lets the low-level control stay
closed-loop and robot-specific.

Reference implementation in this repository:

- `real_robot/spacevln_real/cmd_vel_executor.py`
- `real_robot/scripts/run_cmd_vel_executor.sh`

Closed-loop behavior expected from the executor:

- `MOVE_FORWARD`: keep publishing forward velocity until odometry shows the target distance was reached
- `TURN_LEFT`: keep publishing positive angular velocity until odometry shows the target rotation was reached
- `TURN_RIGHT`: keep publishing negative angular velocity until odometry shows the target rotation was reached
- `STOP`: publish zero velocity immediately and return a terminal status

Avoid a pure time-based implementation such as "publish 0.15 m/s for 1.7 seconds
and assume it moved 0.25 m". That will drift too much on real hardware.

## 4. Optional Capture Trigger

If the camera stack supports on-demand capture in addition to continuous streaming,
it may also listen to:

- Topic: `/spacevln/capture/request`
- Type: `std_msgs/String`
- Payload: JSON string

Example:

```json
{
  "session_id": "2f94b3b5-9c7d-4ef5-89dd-79a18c1b2bdb",
  "reason": "post_move_forward",
  "stamp": 1713091201.500
}
```

The current runtime defaults to `capture_mode: stream`, which only consumes the
existing camera stream. If needed, switch to `capture_mode: trigger`.

## 5. Recommended OAK-D Lite Parameters

- RGB resolution: `640x480`
- Depth aligned to the RGB frame
- Horizontal FOV: approximately `69°`
- Depth operating range: `0.4m ~ 8.0m`

## 6. Internal Observation Schema Used by SpaceVLN

The real-robot adapter converts synchronized ROS messages into:

```text
obs = {
  rgb: HxWx3 uint8,
  depth: HxWx1 float32,          # normalized depth in [0, 1]
  sensor_pose: [dx, dy, dtheta], # relative motion
  position: [x, z, y],           # compatibility layout, height is the second value
  heading: [yaw_rad],
  timestamp: t
}
```

## 7. Integration Order

Recommended first-stage integration:

1. `/oak/rgb/image_raw`
2. `/oak/stereo/image_raw`
3. `/odom`
4. `/spacevln/action_cmd`
5. `/spacevln/action_status`

Second-stage additions:

- `/oak/imu/data`
- `/spacevln/capture/request`
- `distance_to_goal_m`
- `goal_reached`
