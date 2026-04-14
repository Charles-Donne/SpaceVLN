# SpaceVLN Real-Robot Runtime

This directory contains a standalone real-robot integration layer for `SpaceVLN`.
It is designed to reuse the existing controller, VLM stack, mapping, and artifact
pipeline without changing the simulator workflow.

## Scope

- Upstream modules reused as-is:
  - `VLMNavigationController`
  - `GroundedSAM`
  - `SemanticMapper`
  - artifact saving and visualization
- New real-robot integration modules:
  - ROS1 / ROS2 subscribers and publishers
  - RGB / depth / pose / IMU synchronization
  - action command bridge
  - real-robot `VectorEnv` adapter

## Layout

- `run_real_navigation.py` — real-robot Python entrypoint
- `config/real_robot.yaml` — default ROS and OAK-D Lite configuration
- `spacevln_real/` — runtime implementation
- `scripts/run_real_navigation.sh` — shell launcher
- `ros_interface.md` — integration contract for the low-level robotics team

## Design Goals

- Keep the simulator runtime untouched
- Restrict controller changes to a minimal environment-injection hook
- Expose a clean real-robot boundary at the repository root
- Use standard English naming throughout the integration layer

## Quick Start

Prerequisites:

1. ROS1 or ROS2 is installed and sourced
2. OAK-D Lite RGB, depth, and camera info topics are available
3. The base controller listens on `/spacevln/action_cmd`
4. The base controller publishes status on `/spacevln/action_status`

Run:

```bash
cd SpaceVLN
bash real_robot/scripts/run_real_navigation.sh \
  --instruction "Move forward through the doorway and approach the table on the left." \
  --real-config real_robot/config/real_robot.yaml \
  --exp-config navigation_system/config/experiments/r2r_eval.yaml
```

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

See `real_robot/ros_interface.md` for the full payload specification.

