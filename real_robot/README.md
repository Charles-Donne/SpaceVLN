# SpaceVLN Real-Robot Runtime

This directory contains the real-robot integration layer for SpaceVLN. It reuses
the existing controller, VLM stack, mapping, and artifact writer while keeping
the simulator workflow untouched.

## What Runs On The Robot

- `run_real_robot_lite.sh`: default real-robot launcher. It starts the
  `/cmd_vel` executor, runs navigation, and disables GroundingDINO/SAM.
- `run_real_robot_full.sh`: full perception launcher. It reuses the same
  bring-up as lite and only enables GroundingDINO/SAM.
- `run_real_robot_full_manual.sh`: full perception launcher with manual motion
  handoff. It keeps the agent loop running but waits for operator confirmation
  instead of publishing `/cmd_vel`.
- `run_real_robot_simple.sh`: shared implementation used by both launchers.
- `run_cmd_vel_executor.py`: reference ROS2 executor that converts
  `/spacevln/action_cmd` into closed-loop `/cmd_vel`.
- `run_manual_action_executor.py`: interactive executor that prints each
  requested action and waits for Enter before publishing a successful status.
- `config/real_robot.yaml`: D435i, odometry, sync, motion, and mapping defaults.
- `ros_interface.md`: ROS topic and JSON payload contract.

Older explicit wrappers such as `run_real_navigation.sh`,
`run_cmd_vel_executor.sh`, and `send_action_command.sh` are kept for debugging
and manual bring-up. Normal evaluation should use `run_real_robot_lite.sh`,
`run_real_robot_full.sh`, or `run_real_robot_full_manual.sh` depending on
whether you want autonomous or hand-driven motion.

## Deployment Layout

The scripts assume this workspace shape on the robot:

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

The real runtime writes results to the workspace sibling result directory by
default:

```text
/ros2_orin/result/real_robot/
```

Each launcher run automatically chooses the next unused episode id under that
results directory, so separate instructions are saved as `episode_0`,
`episode_1`, `episode_2`, and so on instead of overwriting `episode_0`.
The selected id is printed at startup:

```text
[REAL] episode_id=3
```

To reproduce or overwrite a specific episode intentionally, set `EPISODE_ID`:

```bash
sudo -E env EPISODE_ID=12 bash real_robot/scripts/run_real_robot_full.sh \
  "enter through the door ahead and stop at the table."
```

During a run, the terminal prints the detailed log directory:

```text
[RealRobot] logs=/ros2_orin/result/real_robot_console_logs/...
```

## Prerequisites

Required:

- ROS2 Humble environment with `rclpy`
- RealSense RGB/depth topics and `/odom`
- Python dependencies used by SpaceVLN, including PyTorch, OpenCV, NumPy,
  Pillow, PyYAML, and the VLM API dependencies
- A valid `navigation_system/config/vlm/vlm_api_config.yaml`

Not required for real-robot runtime:

- Habitat-Lab
- Habitat-Sim
- habitat-baselines
- `numpy-quaternion`

Jetson AGX Orin notes:

- JetPack 6.2 / L4T R36.4.x works with CUDA 12.6.
- `numpy==1.24.4` is recommended with the current SciPy/PyTorch stack.
- If GroundingDINO/SAM is run with `sudo -E`, the scripts automatically expose
  `/usr/local/cuda/lib64` and PyTorch's `torch/lib` through
  `setup_real_accel_env.sh`, so `libc10.so` does not need to be exported by hand.
- The launchers also force UDP transport for ROS2 so same-host `sudo` and
  container runs do not silently lose samples.

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

The RGB and depth images should both be `640x480`. RGB encodings `rgb8` and
`bgr8` are supported. Depth encodings `16UC1` in millimeters and `32FC1` in
meters are supported.

Useful checks:

```bash
ros2 topic echo --once --field height /camera/camera/color/image_raw
ros2 topic echo --once --field width /camera/camera/color/image_raw
ros2 topic echo --once --field encoding /camera/camera/color/image_raw

ros2 topic echo --once --field height /camera/camera/depth/image_rect_raw
ros2 topic echo --once --field width /camera/camera/depth/image_rect_raw
ros2 topic echo --once --field encoding /camera/camera/depth/image_rect_raw

timeout 5s ros2 topic hz /odom
```

## Control Defaults

The reference executor uses odometry feedback by default:

- mode: `odom`
- linear speed: `0.5 m/s`
- angular speed: `60 deg/s`
- forward early-stop tolerance: `0.10 m`
- turn early-stop tolerance: `24 deg`
- completion stability window: `0.20 s`
- yaw stability tolerance: `0.50 deg`
- subtask action limit: `5`

Lookaround is stopped and sampled:

- 8 views total
- 45 degrees per turn
- after each turn's terminal action status, the runtime captures one new RGB-D
  snapshot
- normal navigation actions capture once after the action finishes; no
  intermediate frames are sampled during a long forward move or rotation

## Run Lite Navigation

Lite mode is the default for bring-up and normal runs without GroundingDINO/SAM:

```bash
cd /ros2_orin/SpaceVLN

bash real_robot/scripts/run_real_robot_lite.sh \
  "enter through the door ahead and stop at the table."
```

Terminal output is compact by default. Full logs still go to
`/ros2_orin/result/real_robot_console_logs/...`.

For full terminal output:

```bash
REAL_CONSOLE=full bash real_robot/scripts/run_real_robot_lite.sh \
  "enter through the door ahead and stop at the table."
```

## Run Full Perception

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

The build script applies the small PyTorch 2.8 compatibility patch needed by
the upstream GroundingDINO CUDA source and checks that `groundingdino._C`
imports.

Then run full perception:

```bash
cd /ros2_orin/SpaceVLN

sudo -E bash real_robot/scripts/run_real_robot_full.sh \
  "enter through the door ahead and stop at the table."
```

Use `sudo -E` until ordinary `rosuser` CUDA access is fixed. You can verify
ordinary-user CUDA with:

```bash
python3 - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

`run_real_robot_full.sh` shares the same ROS topics, executor, and observation
sync as lite. The only difference is that it requires the GroundingDINO/SAM
runtime and model files.

## Manual Motion Mode

Use the manual variant when you want to test the agent while driving the robot
yourself:

```bash
cd /ros2_orin/SpaceVLN
sudo -E bash real_robot/scripts/run_real_robot_full_manual.sh \
  "Move forward, then turn right to enter the corridor. Continue to the exhibition room at the end of the corridor, and stop at the cabinet in the exhibition room."
```

The terminal will print the requested action, for example:

- `请手动左转 45.0 deg`
- `请手动向前走 0.50 m`
- `请手动停止机器人`

After you complete the motion, press Enter and the agent will continue with
the next step.

## Test GroundingDINO/SAM On One Image

Capture or reuse a saved RGB frame, then test one class:

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

The concise output includes the device and runtime, for example:

```text
device: cuda (Orin)
dino_runtime: cuda_custom_ops
boxes: 1
```

The annotated image and default log are written beside `TEST_IMAGE`.

For open-vocabulary debugging only, use `CAPTION=...` instead of `CLASSES=...`.
This may return many candidate boxes:

```bash
sudo -E env \
MODEL_DIR=/ros2_orin/data/model/grounded_sam \
GROUNDINGDINO_DIR=/ros2_orin/GroundingDINO \
TEST_IMAGE=../result/real_robot/grounded_sam_tests/shelf_rgb.jpg \
CAPTION="shelving unit . storage rack . rack . shelf . bookcase ." \
BOX_THRESHOLD=0.08 \
TEXT_THRESHOLD=0.08 \
bash real_robot/scripts/test_grounded_sam.sh
```

`BOX_THRESHOLD` and `TEXT_THRESHOLD` are confidence thresholds, not object
sizes. Lower values produce more boxes and more false positives.

For a single target class, prefer `CLASSES="shelf"` or another one-item class
list. Use `CAPTION=...` only when you want open-vocabulary debugging and are
okay with many candidate boxes.

## Manual Action Tests

Start the executor manually in one terminal:

```bash
cd /ros2_orin/SpaceVLN

python3 -u real_robot/run_cmd_vel_executor.py \
  --cmd-vel-topic /cmd_vel \
  --odom-topic /odom \
  --control-mode odom \
  --angle-tolerance-deg 24
```

Watch status in another terminal:

```bash
ros2 topic echo /spacevln/action_status
```

Send a command:

```bash
cd /ros2_orin/SpaceVLN

python3 real_robot/spacevln_real/send_action_command.py \
  TURN_LEFT --degrees 45 --timeout-s 20
```

## Mapping And Synchronization

The observation hub pairs each RGB frame with the nearest depth frame, then uses
the RGB-D midpoint timestamp to select the nearest odometry pose within
`sync_tolerance_s`. The default tolerance is `0.75s`.

Mapping uses the real odometry delta between snapshots, not the nominal
lookaround image angle. D435i projection defaults:

- RGB/depth HFOV: `87 deg`
- camera height: `1.3 m`
- camera pitch: `-15 deg`
- min depth: `0.3 m`
- max depth: `3.0 m`

Depth mapping is enabled by default. The real path averages the selected depth
frame with immediate neighboring frames when available, and uses selective
dynamic obstacle evidence so unknown cells do not vote.

## Artifacts

Per-step prompt/response/images are saved under:

```text
/ros2_orin/result/real_robot/<model_stack>/detail/...
```

The runtime also records a 1Hz raw RGB stream from the camera topic:

```text
[REAL] rgb_stream_dir=...
```

The top-down simulator map is not available on the real robot; use the saved
view images and RGB stream for visual debugging.
