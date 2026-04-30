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
- `config/real_robot.yaml` — default ROS and OAK-D Lite configuration
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

Prerequisites:

1. ROS2 is installed and sourced
2. OAK-D Lite RGB, depth, and camera info topics are available
3. The low-level robot controller listens on `/spacevln/action_cmd`
4. The low-level robot controller publishes status on `/spacevln/action_status`

Run:

```bash
cd SpaceVLN
bash real_robot/scripts/run_real_navigation.sh \
  --instruction "Move forward through the doorway and approach the table on the left." \
  --real-config real_robot/config/real_robot.yaml \
  --exp-config navigation_system/config/experiments/vlnce/r2r_eval.yaml
```

Run the reference ROS2 action executor:

```bash
cd SpaceVLN
bash real_robot/scripts/run_cmd_vel_executor.sh \
  --cmd-vel-topic /cmd_vel \
  --odom-topic /odom
```

This executor subscribes to `/spacevln/action_cmd`, uses `/odom` as feedback,
publishes base velocities on `/cmd_vel`, and reports terminal results on
`/spacevln/action_status`.

## Recommended Control Split

Use two ROS2 processes:

1. The `SpaceVLN` real-robot runtime
2. The `/cmd_vel` action executor

The high-level runtime should keep producing discrete actions:

- `MOVE_FORWARD`
- `TURN_LEFT`
- `TURN_RIGHT`
- `STOP`

The executor should translate each action into a closed-loop velocity sequence
using:

- input: `/spacevln/action_cmd`
- feedback: `/odom`
- output: `/cmd_vel`
- completion: `/spacevln/action_status`

This is safer than sending one open-loop velocity pulse for a fixed duration.

## Default Topics

Sensor topics:

- `/oak/rgb/image_raw`
- `/oak/rgb/camera_info`
- `/oak/stereo/image_raw`
- `/oak/stereo/camera_info`
- `/oak/imu/data`
- `/odom`

Action topics:

- `/spacevln/action_cmd`
- `/spacevln/action_status`
- `/cmd_vel` if you use the reference executor

See `real_robot/ros_interface.md` for the full payload specification.
