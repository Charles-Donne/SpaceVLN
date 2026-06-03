# SpaceVLN Real-Robot Runtime

This directory contains the real-robot integration layer for SpaceVLN. The
operator-facing runtime now has only two modes:

- `scripts/run_real_robot_auto.sh`: full automatic mode. The agent sends
  `/spacevln/action_cmd` commands and the cmd_vel executor drives `/cmd_vel`.
- `scripts/run_real_robot_manual.sh`: manual prompt-only mode. The agent saves
  the action VLM prompt/image for each step and waits for the operator to move
  the robot manually; it does not call the action VLM or publish `/cmd_vel` for
  action-stage motion.

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

Each subtask defaults to at most `8` low-level action steps before returning to
the thinking/planning stage. Override with `MAX_SUBTASK_STEPS=<n>` if needed.

## Run Manual

```bash
cd /ros2_orin/SpaceVLN

sudo -E bash real_robot/scripts/run_real_robot_manual.sh \
  "Move forward, then turn right to enter the corridor. Continue to the exhibition room at the end of the corridor, and stop at the cabinet in the exhibition room."
```

Manual mode still runs the automatic 8-view lookaround scan. After thinking, it
builds the next action-step context, saves the exact action VLM input artifacts,
prints the current thinking subtask, and waits for the operator:

```text
[ManualPromptOnly] dir=.../action_step_0001
[ManualPromptOnly] user_prompt.md=.../user_prompt.md
[ManualPromptOnly] action_view.jpg=.../action_view.jpg
[ManualPromptOnly] 当前 thinking VLM 子任务:
[ManualPromptOnly]   next_waypoint: ...
[ManualPromptOnly]   subtask_instruction: ...
[ManualPromptOnly] 当前子任务指令: ...
[ManualPromptOnly] 请根据 prompt/image 手动操作机器人；完成后输入 a 回车继续；输入 f 结束当前 subtask 并回 planner:
```

In this mode the action VLM request is not sent, and no VLM result is parsed.
Move the robot manually after inspecting the prompt/image, then input `a` and
press Enter. The runtime captures a fresh synchronized RGB-D/pose observation
and continues from that real pose. Input `f` to finish the current subtask and
return to thinking/replanning; the planner can then choose a new subtask or end
the task.

Manual mode defaults `SPACEVLN_MANUAL_PROMPT_ONLY=1` and
`SPACEVLN_DISABLE_LANDMARK_AUTOSTOP=1`, so landmark proximity will not
automatically finish an action stage.

Real-robot launchers default `SPACEVLN_SPACE_AREA_REGION_RADIUS_M=3.0`, so each
parsed space/area covers a larger local region than the simulation default
`2.0m`. Override this variable if you need a tighter or looser real map.

## Finish Gate

Real-robot runs stop only when the planner returns `global_task_finish=true`.
The simulation-style final-destination streak autostop is disabled for real
runs. The launcher also enables a strict finish guard:

```text
[RealRobot] strict_planner_finish_guard=1 planner_finish_near_m=1.0
```

With this guard, a planner finish is rejected if the final task waypoint does
not match the planner destination. If the current goal landmark has distance
evidence, it must be within `1.0m`; visible-but-far or unknown-distance
landmarks should not finish the task. Override only for debugging:

```bash
SPACEVLN_PLANNER_FINISH_NEAR_M=0.8
SPACEVLN_STRICT_PLANNER_FINISH_GUARD=0
```

## RGB Artifacts

RGB frames are saved inside the current episode directory:

```text
<episode_dir>/records/step_rgb/
```

The runtime prints the exact directory once per episode:

```text
[REAL] step_rgb_dir=/ros2_orin/result/real_robot/.../records/step_rgb
```

This directory contains `step_0000_episode_reset_*.jpg`, every lookaround view,
every low-level action step RGB, and uniformly sampled `between_steps` RGB
frames between adjacent low-level steps. The old global 1Hz `real_rgb_stream`
recording is disabled to avoid writing unnecessary frames.

By default, each step interval saves up to `4` transition samples, chosen evenly
by RGB timestamp. Tune this for video export:

```bash
SPACEVLN_REAL_RGB_TRANSITION_SAMPLES=6
SPACEVLN_REAL_RGB_SAMPLE_BUFFER_SIZE=180
```

`SPACEVLN_REAL_RGB_SAMPLE_BUFFER_SIZE` defaults to `360` frames so the 8-view
lookaround can still be sampled after the scan completes.

## Map Alignment

Real map fusion uses only synchronized RGB-D observations at low-level step
endpoints. The extra `between_steps` RGB samples are for video export only and
are not fed into the mapper, so they cannot shift the accumulated map.

The configured 45-degree value is the stopped lookaround target and fallback
display label, not the map angle. Action-stage turns can use any concrete
VLM-selected angle from 1deg to 180deg, and can switch between left and right
turns instead of staying locked to one direction. The mapper receives
`sensor_pose=[dx, dy, dtheta]` computed from the actual odometry/pose before and
after each settled low-level action, so a commanded 37deg turn that actually
settles at 35deg is fused with the measured 35deg.

Each live status step includes a compact `map_alignment` summary with the latest
pose delta, `full_pose`, global/subtask trajectory counts, and obstacle/explored
cell counts. If a real pose jump is unusually large, the runtime also prints a
`[REAL-MAP] large measured pose delta...` warning.

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

Real-robot launchers also default `SPACEVLN_DISABLE_CORRIDOR_AUTOSTOP=1`.
This prevents hallway/corridor/passage/walkway landmarks from ending an action
stage early. Doorway and solid-object autocomplete are left enabled.

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
