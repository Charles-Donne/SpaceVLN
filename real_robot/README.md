# SpaceVLN Real-Robot Runtime

This directory contains the real-robot integration layer for SpaceVLN. The
operator-facing runtime now has only two modes:

- `scripts/run_real_robot_auto.sh`: full automatic mode. The agent sends
  `/spacevln/action_cmd` commands and the cmd_vel executor drives `/cmd_vel`.
- `scripts/run_real_robot_manual.sh`: manual motion mode. The agent prints each
  requested action and waits for operator confirmation instead of publishing
  `/cmd_vel`.

`scripts/_run_real_robot_impl.sh` is the shared internal launcher used by both
entrypoints. Perception defaults to full GroundingDINO/SAM for both modes; use
environment variables to override perception, not a separate launcher.

## Deployment Layout

Expected robot workspace:

```text
/ros2_orin/
├── SpaceVLN/
├── GroundingDINO/
├── data/model/grounded_sam/
└── result/
```

Expected model files:

```text
/ros2_orin/data/model/grounded_sam/
├── GroundingDINO_SwinT_OGC.py
├── groundingdino_swint_ogc.pth
└── sam_vit_h_4b8939.pth
```

Results are written to:

```text
/ros2_orin/result/real_robot/
```

Each launcher run automatically chooses the next unused episode id, so separate
instructions are saved as `episode_0`, `episode_1`, `episode_2`, and so on. The
selected id is printed at startup:

```text
[REAL] episode_id=3
```

To intentionally reuse an id:

```bash
sudo -E env EPISODE_ID=12 bash real_robot/scripts/run_real_robot_auto.sh \
  "enter through the door ahead and stop at the table."
```

## Run Auto

```bash
cd /ros2_orin/SpaceVLN

sudo -E bash real_robot/scripts/run_real_robot_auto.sh \
  "enter through the door ahead and stop at the table."
```

## Run Manual

```bash
cd /ros2_orin/SpaceVLN

sudo -E bash real_robot/scripts/run_real_robot_manual.sh \
  "Move forward, then turn right to enter the corridor. Continue to the exhibition room at the end of the corridor, and stop at the cabinet in the exhibition room."
```

Manual mode prints prompts such as:

- `请手动左转 45.0 deg`
- `请手动向前走 0.50 m`
- `请手动停止机器人`

After completing the motion, press Enter. Manual mode defaults
`SPACEVLN_REAL_ACTION_TIMEOUT_S=3600`, so the agent waits for operator
confirmation instead of timing out after the normal autonomous-control timeout.

## RGB Artifacts

There are two RGB outputs:

- Per-step raw RGB:

```text
<episode_dir>/records/step_rgb/
```

The runtime prints the exact directory once per episode:

```text
[REAL] step_rgb_dir=/ros2_orin/result/real_robot/.../records/step_rgb
```

This directory contains `step_0000_episode_reset_*.jpg`, every lookaround view,
and every action step RGB. Use this for agent-step debugging.

- Raw camera stream:

```text
[REAL] rgb_stream_dir=/ros2_orin/result/real_robot/.../real_rgb_stream
```

This is a time-sampled stream from the RGB topic. The default interval is 1s and
can be changed with `SPACEVLN_REAL_RGB_RECORD_INTERVAL_S`. It is not the same as
per-step saving.

## Real-Robot Forward Safety

For real-robot runs only, `MOVE_FORWARD` is disabled when the front depth
clearance is below `0.50m`.

- The action prompt removes `MOVE_FORWARD` from the allowed action space.
- The real environment also hard-blocks any leaked `MOVE_FORWARD` before it can
  publish an action command to the low-level executor.

Override the threshold only when needed:

```bash
sudo -E env SPACEVLN_REAL_FORWARD_MIN_CLEARANCE_M=0.60 \
bash real_robot/scripts/run_real_robot_auto.sh \
  "enter through the door ahead and stop at the table."
```

## Perception Setup

Install Python dependencies:

```bash
cd /ros2_orin/SpaceVLN
bash real_robot/scripts/install_grounded_sam_deps.sh
```

On Jetson, install CUDA compilation pieces once if GroundingDINO `_C` needs to
be built:

```bash
sudo apt install -y \
  cuda-nvcc-12-6 \
  cuda-cudart-dev-12-6 \
  cuda-cccl-12-6 \
  cuda-command-line-tools-12-6 \
  cuda-libraries-dev-12-6 \
  libcusparse-dev-12-6 \
  libcublas-dev-12-6 \
  libcusolver-dev-12-6 \
  libcurand-dev-12-6 \
  ninja-build

sudo ln -sfn /usr/local/cuda-12.6 /usr/local/cuda
```

Build GroundingDINO's CUDA extension:

```bash
cd /ros2_orin/SpaceVLN
sudo -E bash real_robot/scripts/build_groundingdino_cuda_ext.sh
```

The launcher automatically exposes `/usr/local/cuda/lib64` and PyTorch's
`torch/lib` through `scripts/setup_real_accel_env.sh` when run with `sudo -E`.

For no GroundingDINO/SAM bring-up, keep the same auto/manual mode and override
perception:

```bash
sudo -E env \
SPACEVLN_PERCEPTION_MODE=lite \
SPACEVLN_DISABLE_GROUNDED_SAM=1 \
bash real_robot/scripts/run_real_robot_auto.sh \
  "enter through the door ahead and stop at the table."
```

## Sensor Topics

Defaults from `config/real_robot.yaml`:

- RGB image: `/camera/camera/color/image_raw`
- RGB camera info: `/camera/camera/color/camera_info`
- Depth image: `/camera/camera/depth/image_rect_raw`
- Depth camera info: `/camera/camera/depth/camera_info`
- IMU: `/camera/camera/imu`
- Odometry: `/odom`
- Action command: `/spacevln/action_cmd`
- Action status: `/spacevln/action_status`
- Base velocity: `/cmd_vel`

Useful checks:

```bash
ros2 topic echo --once --field encoding /camera/camera/color/image_raw
ros2 topic echo --once --field encoding /camera/camera/depth/image_rect_raw
timeout 5s ros2 topic hz /odom
```

## Grounded-SAM Single Image Test

```bash
cd /ros2_orin/SpaceVLN

sudo -E env \
MODEL_DIR=/ros2_orin/data/model/grounded_sam \
GROUNDINGDINO_DIR=/ros2_orin/GroundingDINO \
TEST_IMAGE=../result/real_robot/grounded_sam_tests/shelf_rgb.jpg \
CLASSES="shelving unit" \
BOX_THRESHOLD=0.25 \
TEXT_THRESHOLD=0.20 \
bash real_robot/scripts/test_grounded_sam.sh
```

Use `CLASSES="single target"` for one class. Use `CAPTION=...` only for
open-vocabulary debugging, because it can return many candidate boxes.
