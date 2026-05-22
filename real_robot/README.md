# SpaceVLN Real-Robot Runtime

This directory contains a standalone real-robot integration layer for `SpaceVLN`.
It is designed to reuse the existing controller, VLM stack, mapping, and artifact
pipeline without changing the simulator workflow.

## Scope

- Upstream modules reused as-is:
  - `NavigationAgentController`
  - `GroundedSAM`
  - `SemanticMapper`
  - artifact saving and visualization
- New real-robot integration modules:
  - ROS2 subscribers and publishers
  - RGB / depth / pose / IMU synchronization
  - action command bridge
  - ROS2 closed-loop `/cmd_vel` action executor
  - real-robot `VectorEnv` adapter

## Layout

- `run_real_navigation.py` — real-robot Python entrypoint
- `run_cmd_vel_executor.py` — ROS2 `/cmd_vel` executor entrypoint
- `config/real_robot.yaml` — Intel RealSense D435i real-robot configuration
- `spacevln_real/` — runtime implementation
- `scripts/run_real_navigation.sh` — shell launcher
- `scripts/run_cmd_vel_executor.sh` — shell launcher for the `/cmd_vel` executor
- `ros_interface.md` — integration contract for the low-level robotics team

## Design Goals

- Keep the simulator runtime untouched
- Restrict controller changes to a minimal environment-injection hook
- Expose a clean real-robot boundary at the repository root
- Use standard English naming throughout the integration layer

## Quick Start

The real-robot runtime uses a real-only config path and does not require
Habitat-Lab or Habitat-Sim. Keep those dependencies for simulator evaluation
only.

Prerequisites:

1. ROS2 is installed and sourced
2. RealSense D435i RGB, aligned depth, IMU, and camera info topics are available
3. The low-level robot controller listens on `/spacevln/action_cmd`
4. The low-level robot controller publishes status on `/spacevln/action_status`
5. Python dependencies for `GroundingDINO` plus optional `SAM`, SpaceVLN mapping,
   VLM API calls, and `rclpy` are available

Environment boundary:

- Required for full real navigation: ROS2/rclpy message packages, PyTorch,
  OpenCV, NumPy, Pillow/image tooling, `yacs`, `requests`, `PyYAML`,
  GroundingDINO and its normal detection helpers such as `supervision`, plus a
  valid VLM API config.
- Optional: Segment Anything / RepViT-SAM. If SAM is not installed, detection
  can fall back to GroundingDINO boxes as coarse masks.
- Not required by the real-robot runtime: Habitat-Lab, Habitat-Sim,
  habitat-baselines, and `numpy-quaternion`. Those remain simulator/evaluation
  dependencies.

Simplest run, with the natural-language task filled directly in the command:

```bash
cd SpaceVLN
bash real_robot/scripts/run_real_robot_lite.sh \
  "Move forward through the doorway and approach the table on the left."
```

The lightweight launcher disables GroundingDINO/SAM so the robot workflow can
run in a minimal ROS2 + SpaceVLN/VLM environment.

Full perception run:

```bash
cd SpaceVLN
bash real_robot/scripts/run_real_robot_full.sh \
  "Move forward through the doorway and approach the table on the left."
```

The full launcher requires GroundingDINO and SAM / RepViT-SAM. Set
`SPACEVLN_REQUIRE_SAM=0` only if you want to allow a GroundingDINO-box-mask
fallback.

You can also edit `TASK_INSTRUCTION` at the top of
`real_robot/scripts/run_real_robot_simple.sh`, or set `SPACEVLN_INSTRUCTION`.
By default this simple launcher starts the reference `/cmd_vel` executor and
then starts SpaceVLN.

Equivalent explicit run:

```bash
cd SpaceVLN
bash real_robot/scripts/run_real_navigation.sh \
  --instruction "Move forward through the doorway and approach the table on the left." \
  --real-config real_robot/config/real_robot.yaml
```

Run the reference ROS2 action executor:

```bash
cd SpaceVLN
bash real_robot/scripts/run_cmd_vel_executor.sh \
  --cmd-vel-topic /cmd_vel \
  --odom-topic /odom \
  --control-mode odom \
  --control-rate-hz 10 \
  --position-tolerance-m 0.10 \
  --angle-tolerance-deg 10
```

This executor subscribes to `/spacevln/action_cmd`, publishes base velocities on
`/cmd_vel` at the configured control rate, and reports terminal results on
`/spacevln/action_status`. By default it uses `/odom` as feedback. For an early
bring-up without reliable odometry, run it with `--control-mode timed`; that
publishes velocity for `target / speed` seconds and then sends zero velocity.

Manual single-action tests:

```bash
bash real_robot/scripts/send_action_command.sh MOVE_FORWARD --meters 0.5
bash real_robot/scripts/send_action_command.sh TURN_LEFT --degrees 30
bash real_robot/scripts/send_action_command.sh LOOK_AROUND_360
bash real_robot/scripts/send_action_command.sh STOP
```

## Recommended Control Split

Use two ROS2 processes:

1. The `SpaceVLN` real-robot runtime
2. The `/cmd_vel` action executor

The high-level runtime should keep producing discrete actions:

- `MOVE_FORWARD`
- `TURN_LEFT`
- `TURN_RIGHT`
- `LOOK_AROUND_360`
- `STOP`

For `MOVE_FORWARD`, the runtime passes the VLM-selected target distance as one
continuous command, usually 0.5m to 1.5m. The executor should translate each
action into a velocity sequence
using:

- input: `/spacevln/action_cmd`
- feedback: `/odom` in closed-loop mode
- output: `/cmd_vel`
- completion: `/spacevln/action_status`

For SpaceVLN lookaround, the real runtime sends eight closed-loop `TURN_LEFT`
commands at 45 degrees each. It waits for each turn to finish, lets the base
report stable completion, then captures the next RGB-D observation before the
following turn. Normal actions capture once after the one requested action
finishes; the runtime does not sample intermediate frames during a long forward
move or an automatic rotation command.

The real RGB-D adapter keeps depth mapping enabled by default. Each synchronized
real snapshot averages the chosen depth frame with its immediate neighboring
depth frames when the configured fusion window is available. Real obstacle
fusion is selective: observed obstacle cells and explicitly observed free cells
update bounded obstacle evidence, while unknown depth cells do not cast votes.
The simulation map path keeps its existing fusion behavior unless a real runtime
config enables this selective update.

For deployment, final success is not judged online from `goal_reached` or
`distance_to_goal_m`. The runtime ends the episode when the model/planner marks
`global_task_finish=true`, sends `STOP`, and records that model-level finish in
the final metrics.

This is safer than sending one open-loop velocity pulse for a fixed duration.

## Default Topics

Sensor topics from `real_robot/config/real_robot.yaml`:

- `/camera/camera/color/image_raw`
- `/camera/camera/color/camera_info`
- `/camera/camera/aligned_depth_to_color/image_raw`
- `/camera/camera/aligned_depth_to_color/camera_info`
- `/camera/camera/imu`
- `/odom`

Action topics:

- `/spacevln/action_cmd`
- `/spacevln/action_status`
- `/cmd_vel` if you use the reference executor

See `real_robot/ros_interface.md` for the full payload specification.
