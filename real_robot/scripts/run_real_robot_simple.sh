#!/usr/bin/env bash
set -euo pipefail

# Fill the task instruction here, or pass it as command-line text:
#   bash real_robot/scripts/run_real_robot_simple.sh "Go to the table near the sofa."
TASK_INSTRUCTION="${SPACEVLN_INSTRUCTION:-}"

# Common bring-up knobs. Override any of them from the shell if needed, e.g.
#   START_EXECUTOR=0 REAL_CONFIG=real_robot/config/my_robot.yaml bash ...
START_EXECUTOR="${START_EXECUTOR:-1}"
CONTROL_MODE="${CONTROL_MODE:-odom}"
CONTROL_RATE_HZ="${CONTROL_RATE_HZ:-10}"
POSITION_TOLERANCE_M="${POSITION_TOLERANCE_M:-0.10}"
ANGLE_TOLERANCE_DEG="${ANGLE_TOLERANCE_DEG:-20}"
ODOM_TIMEOUT_S="${ODOM_TIMEOUT_S:-0.5}"
DEFAULT_LINEAR_SPEED_MPS="${DEFAULT_LINEAR_SPEED_MPS:-0.5}"
DEFAULT_ANGULAR_SPEED_DEG_S="${DEFAULT_ANGULAR_SPEED_DEG_S:-60}"
MAX_LINEAR_SPEED_MPS="${MAX_LINEAR_SPEED_MPS:-0.5}"
MAX_ANGULAR_SPEED_DEG_S="${MAX_ANGULAR_SPEED_DEG_S:-60}"
COMPLETION_STABILITY_S="${COMPLETION_STABILITY_S:-0.20}"
COMPLETION_YAW_TOLERANCE_DEG="${COMPLETION_YAW_TOLERANCE_DEG:-0.50}"
CMD_VEL_TOPIC="${CMD_VEL_TOPIC:-/cmd_vel}"
ODOM_TOPIC="${ODOM_TOPIC:-/odom}"

REAL_CONFIG="${REAL_CONFIG:-real_robot/config/real_robot.yaml}"
RUNTIME="${RUNTIME:-context_cache}"
MAX_SUBTASK_STEPS="${MAX_SUBTASK_STEPS:-8}"
MAX_STEPS="${MAX_STEPS:-}"
RESULTS_DIR="${RESULTS_DIR:-}"
VLM_API_CONFIG="${VLM_API_CONFIG:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPACEVLN_DIR="$(cd "${REAL_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${SPACEVLN_DIR}/.." && pwd)"

# Real-robot runs always default to the workspace-level sibling result dir:
#   <workspace>/SpaceVLN
#   <workspace>/result/real_robot/...
REAL_RESULTS_ROOT="${REAL_RESULTS_ROOT:-${WORKSPACE_DIR}/result}"
export SPACEVLN_REAL_RESULTS_ROOT="${REAL_RESULTS_ROOT}"
export SPACEVLN_RESULTS_ROOT="${REAL_RESULTS_ROOT}"
export SPACEVLN_RESULTS_FAMILY="${SPACEVLN_RESULTS_FAMILY:-real_robot}"
export SPACEVLN_OUTPUT_PROFILE="${SPACEVLN_OUTPUT_PROFILE:-debug}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export SPACEVLN_ALLOW_GENERIC_WAYPOINT_LABELS="${SPACEVLN_ALLOW_GENERIC_WAYPOINT_LABELS:-1}"
export SPACEVLN_LOOKAROUND_VIEW_COUNT="${SPACEVLN_LOOKAROUND_VIEW_COUNT:-8}"
export SPACEVLN_LOOKAROUND_STEP_DEG="${SPACEVLN_LOOKAROUND_STEP_DEG:-45}"

if [[ -n "${SPACEVLN_ROS_SETUP:-}" && -f "${SPACEVLN_ROS_SETUP}" ]]; then
  # shellcheck disable=SC1090
  source "${SPACEVLN_ROS_SETUP}"
fi

export PYTHONPATH="${SPACEVLN_DIR}:${REAL_DIR}:${PYTHONPATH:-}"

if [[ "$#" -gt 0 ]]; then
  TASK_INSTRUCTION="$*"
fi

if [[ -z "${TASK_INSTRUCTION// }" ]]; then
  echo "ERROR: no task instruction provided." >&2
  echo "Fill TASK_INSTRUCTION in this script, set SPACEVLN_INSTRUCTION, or pass the instruction as an argument." >&2
  exit 2
fi

cd "${SPACEVLN_DIR}"

echo "[RealRobot] results_root=${SPACEVLN_RESULTS_ROOT}"
echo "[RealRobot] output_profile=${SPACEVLN_OUTPUT_PROFILE}"
echo "[RealRobot] runtime=${RUNTIME}"

EXECUTOR_PID=""
cleanup() {
  if [[ -n "${EXECUTOR_PID}" ]]; then
    kill "${EXECUTOR_PID}" >/dev/null 2>&1 || true
    wait "${EXECUTOR_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${START_EXECUTOR}" == "1" ]]; then
  python3 -u "${REAL_DIR}/run_cmd_vel_executor.py" \
    --cmd-vel-topic "${CMD_VEL_TOPIC}" \
    --odom-topic "${ODOM_TOPIC}" \
    --control-mode "${CONTROL_MODE}" \
    --control-rate-hz "${CONTROL_RATE_HZ}" \
    --position-tolerance-m "${POSITION_TOLERANCE_M}" \
    --angle-tolerance-deg "${ANGLE_TOLERANCE_DEG}" \
    --odom-timeout-s "${ODOM_TIMEOUT_S}" \
    --default-linear-speed-mps "${DEFAULT_LINEAR_SPEED_MPS}" \
    --default-angular-speed-deg-s "${DEFAULT_ANGULAR_SPEED_DEG_S}" \
    --max-linear-speed-mps "${MAX_LINEAR_SPEED_MPS}" \
    --max-angular-speed-deg-s "${MAX_ANGULAR_SPEED_DEG_S}" \
    --completion-stability-s "${COMPLETION_STABILITY_S}" \
    --completion-yaw-tolerance-deg "${COMPLETION_YAW_TOLERANCE_DEG}" &
  EXECUTOR_PID="$!"
  sleep 1
fi

NAV_ARGS=(
  --instruction "${TASK_INSTRUCTION}"
  --real-config "${REAL_CONFIG}"
  --runtime "${RUNTIME}"
  --max-subtask-steps "${MAX_SUBTASK_STEPS}"
)

if [[ -n "${MAX_STEPS}" ]]; then
  NAV_ARGS+=(--max-steps "${MAX_STEPS}")
fi
if [[ -n "${RESULTS_DIR}" && "${SPACEVLN_ALLOW_REAL_RESULTS_DIR_OVERRIDE:-0}" == "1" ]]; then
  NAV_ARGS+=(--results-dir "${RESULTS_DIR}")
elif [[ -n "${RESULTS_DIR}" ]]; then
  echo "[RealRobot] ignoring RESULTS_DIR; set SPACEVLN_ALLOW_REAL_RESULTS_DIR_OVERRIDE=1 to override the sibling result dir" >&2
fi
if [[ -n "${VLM_API_CONFIG}" ]]; then
  NAV_ARGS+=(--vlm-api-config "${VLM_API_CONFIG}")
fi

python3 -u "${REAL_DIR}/run_real_navigation.py" "${NAV_ARGS[@]}"
