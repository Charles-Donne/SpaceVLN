# SpaceVLN Real-Robot ROS Interface

This document is the integration contract for the low-level robotics stack.

## 1. Sensor Inputs Consumed by SpaceVLN

### 1.1 Camera Topics

D435i default RGB image

- Topic: `/camera/color/image_raw`
- Official realsense-ros default: `/camera/camera/color/image_raw`
- Type: `sensor_msgs/Image`
- Recommended encoding: `rgb8` or `bgr8`
- Recommended resolution: `640x480`

D435i default RGB camera info

- Topic: `/camera/color/camera_info`
- Official realsense-ros default: `/camera/camera/color/camera_info`
- Type: `sensor_msgs/CameraInfo`

D435i default aligned depth image

- Topic: `/camera/aligned_depth_to_color/image_raw`
- Current SpaceVLN real config: `/camera/camera/depth/image_rect_raw`
- Official realsense-ros aligned-depth default: `/camera/camera/aligned_depth_to_color/image_raw`
- Type: `sensor_msgs/Image`
- Supported encodings:
  - `16UC1` in millimeters
  - `32FC1` in meters

If the actual RealSense driver publishes a different namespace, such as
`/camera/camera/color/image_raw`, update `real_robot/config/real_robot.yaml`.

D435i default depth camera info

- Topic: `/camera/aligned_depth_to_color/camera_info`
- Current SpaceVLN real config: `/camera/camera/depth/camera_info`
- Official realsense-ros aligned-depth default: `/camera/camera/aligned_depth_to_color/camera_info`
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

### 1.4 Stream Synchronization

`SpaceVLN` pairs each selected RGB frame with the closest depth frame, then
chooses the closest pose frame to the RGB-D pair's midpoint timestamp within
`sync_tolerance_s` from `real_robot/config/real_robot.yaml`:

- default tolerance: `0.75s`
- if no matching depth or pose frame is inside the tolerance, the runtime keeps
  waiting until `observation_timeout_s`
- if a ROS message has a valid `header.stamp`, that stamp is used
- if `header.stamp` is zero or missing, receive time is used as a fallback

For good alignment, publish RGB, depth, and odometry on the same ROS clock and
with meaningful `header.stamp` values. If the camera and odometry clocks are not
aligned, increase `sync_tolerance_s` only as a temporary bring-up workaround.

### 1.3 IMU Input

- Topic: `/camera/camera/imu`
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
    "meters": 0.5,
    "degrees": 0.0
  },
  "speed_hint": {
    "linear_mps": 0.5,
    "angular_deg_s": 60.0
  },
  "timeout_s": 20.0,
  "stamp": 1713091200.123
}
```

### 2.2 Allowed action values

- `MOVE_FORWARD`
- `TURN_LEFT`
- `TURN_RIGHT`
- `LOOK_AROUND_360`
- `STOP`

### 2.3 Execution expectations

The low-level controller should execute each command as a closed-loop motion:

- `MOVE_FORWARD`: drive toward the requested distance, typically 0.5m to 1.5m from the VLM action output
- `TURN_LEFT` / `TURN_RIGHT`: rotate toward the requested angle, 45 degrees in the current real action space
- `LOOK_AROUND_360`: supported for manual low-level tests; SpaceVLN real lookaround uses eight stopped `TURN_LEFT` commands at 45 degrees each and samples after each turn settles
- `STOP`: stop immediately and publish a terminal status

This should not be implemented as a single open-loop velocity pulse.

The reference executor publishes terminal success only after it has sent zero
velocity and the odometry heading stays stable for a short completion window.
SpaceVLN real capture happens after that terminal action status: once for each
stopped lookaround turn, and once after a normal action command finishes.

Current reference executor defaults:

- control mode: `odom`
- linear speed: `0.5 m/s`
- angular speed: `60 deg/s`
- forward early-stop tolerance: `0.10 m`
- turn early-stop tolerance: `24 deg`
- completion stability window: `0.20 s`
- yaw stability tolerance: `0.50 deg`

The default real RGB-D config leaves depth mapping enabled. The observation hub
averages the synchronized depth frame with immediate neighboring depth frames
when the requested fusion window is present. Real map fusion then updates
obstacle evidence only for observed obstacle or explicit free cells; unknown
depth samples do not clear or reinforce the map.

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
- `message`

Optional offline-evaluation fields:

- `goal_reached`
- `distance_to_goal_m`

The real-robot runtime does not require these fields for stopping or success.
It stops when the model/planner emits `global_task_finish=true` and the final
`STOP` command returns. If the low-level stack does not estimate goal distance,
use:

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
4. The executor publishes `geometry_msgs/Twist` on `/cmd_vel` at a fixed rate, 10Hz by default in the reference launcher
5. When the target distance or angle is reached, the executor publishes `/spacevln/action_status`

This keeps the high-level planner discrete and lets the low-level control stay
closed-loop and robot-specific.

Reference implementation in this repository:

- `real_robot/spacevln_real/cmd_vel_executor.py`
- `real_robot/scripts/run_cmd_vel_executor.sh`

Closed-loop behavior expected from the executor:

- `MOVE_FORWARD`: keep publishing forward velocity until odometry shows the requested target distance was reached, or until the configured early-stop tolerance is reached
- `TURN_LEFT`: keep publishing positive angular velocity until odometry shows the target rotation was reached, or until the configured early-stop tolerance is reached
- `TURN_RIGHT`: keep publishing negative angular velocity until odometry shows the target rotation was reached, or until the configured early-stop tolerance is reached
- `LOOK_AROUND_360`: keep publishing positive angular velocity and report unwrapped accumulated yaw, so a full 360 degree scan does not collapse to zero after angle normalization
- `STOP`: publish zero velocity immediately and return a terminal status

The reference executor defaults to conservative early stopping:

- `--position-tolerance-m 0.10`
- `--angle-tolerance-deg 24`
- `--completion-stability-s 0.20`
- `--completion-yaw-tolerance-deg 0.50`

At 10Hz and 60deg/s, one control tick is about 6 degrees, so a 24 degree
window gives the executor room to stop before communication and base latency
overshoot the target too much. Tune these values on the actual base.

Avoid a pure time-based implementation such as "publish 0.15 m/s for 3.3 seconds
and assume it moved 0.5 m". That will drift too much on real hardware.

For first bring-up, the reference executor also supports:

```bash
bash real_robot/scripts/run_cmd_vel_executor.sh \
  --control-mode timed \
  --control-rate-hz 10
```

In `timed` mode it converts the high-level command to velocity plus duration:

- forward duration: `target.meters / speed_hint.linear_mps`
- turn duration: `target.degrees / speed_hint.angular_deg_s`
- publish rate: `control_rate_hz`, normally 10Hz

This mode is useful to validate topic wiring and basic base motion. Switch back
to the default `--control-mode odom` for real evaluation runs.

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

## 5. Recommended D435i Parameters

- RGB resolution: `640x480`
- Depth aligned to the RGB frame
- RGB horizontal FOV: approximately `69.4°`
- Depth operating range: approximately `0.3m ~ 3.0m`

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

1. `/camera/camera/color/image_raw`
2. `/camera/camera/aligned_depth_to_color/image_raw`
3. `/odom`
4. `/spacevln/action_cmd`
5. `/spacevln/action_status`

Second-stage additions:

- `/camera/camera/imu`
- `/spacevln/capture/request`
- optional offline-evaluation fields such as `distance_to_goal_m` and `goal_reached`
