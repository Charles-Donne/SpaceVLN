#!/usr/bin/env bash
set -euo pipefail

# Fill the task instruction here, or pass it as command-line text:
#   bash real_robot/scripts/run_real_robot_simple.sh "Go to the table near the sofa."
TASK_INSTRUCTION="${SPACEVLN_INSTRUCTION:-}"

# Common bring-up knobs. Override any of them from the shell if needed, e.g.
#   START_EXECUTOR=0 REAL_CONFIG=real_robot/config/my_robot.yaml bash ...
START_EXECUTOR="${START_EXECUTOR:-1}"
REAL_ACTION_EXECUTOR="${REAL_ACTION_EXECUTOR:-cmd_vel}"
CONTROL_MODE="${CONTROL_MODE:-odom}"
CONTROL_RATE_HZ="${CONTROL_RATE_HZ:-10}"
POSITION_TOLERANCE_M="${POSITION_TOLERANCE_M:-0.10}"
ANGLE_TOLERANCE_DEG="${ANGLE_TOLERANCE_DEG:-24}"
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
MAX_SUBTASK_STEPS="${MAX_SUBTASK_STEPS:-5}"
MAX_STEPS="${MAX_STEPS:-}"
RESULTS_DIR="${RESULTS_DIR:-}"
VLM_API_CONFIG="${VLM_API_CONFIG:-}"
REAL_CONSOLE="${REAL_CONSOLE:-compact}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPACEVLN_DIR="$(cd "${REAL_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${SPACEVLN_DIR}/.." && pwd)"
GROUNDINGDINO_DIR="${GROUNDINGDINO_DIR:-${WORKSPACE_DIR}/GroundingDINO}"

source_setup_bash_safely() {
  local setup_file="$1"
  [[ -f "${setup_file}" ]] || return 0
  local had_nounset=0
  case $- in
    *u*) had_nounset=1 ;;
  esac
  set +u
  # shellcheck disable=SC1090
  source "${setup_file}"
  if [[ "${had_nounset}" -eq 1 ]]; then
    set -u
  fi
}

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

if [[ -n "${SPACEVLN_ROS_SETUP:-}" ]]; then
  source_setup_bash_safely "${SPACEVLN_ROS_SETUP}"
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  source_setup_bash_safely /opt/ros/humble/setup.bash
fi

# Full perception is commonly launched with sudo on Jetson. On same-host ROS2
# graphs, FastDDS shared memory can discover publishers but fail to deliver
# samples across users/containers. Force UDP unless the operator overrides it.
export RMW_FASTRTPS_USE_SHM="${RMW_FASTRTPS_USE_SHM:-0}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_real_accel_env.sh"

if [[ -d "${GROUNDINGDINO_DIR}/groundingdino" ]]; then
  export PYTHONPATH="${GROUNDINGDINO_DIR}:${PYTHONPATH:-}"
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

REAL_LOG_ROOT="${REAL_LOG_ROOT:-${REAL_RESULTS_ROOT}/real_robot_console_logs}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${REAL_LOG_ROOT}/$(date +%Y%m%d_%H%M%S)_$$}"
EXECUTOR_LOG="${RUN_LOG_DIR}/cmd_vel_executor.log"
MANUAL_EXECUTOR_LOG="${RUN_LOG_DIR}/manual_action_executor.log"
NAVIGATION_LOG="${RUN_LOG_DIR}/run_real_navigation.log"
mkdir -p "${RUN_LOG_DIR}"

echo "[RealRobot] results_root=${SPACEVLN_RESULTS_ROOT}"
echo "[RealRobot] output_profile=${SPACEVLN_OUTPUT_PROFILE}"
echo "[RealRobot] runtime=${RUNTIME}"
echo "[RealRobot] console=${REAL_CONSOLE}"
echo "[RealRobot] logs=${RUN_LOG_DIR}"
echo "[RealRobot] perception=${SPACEVLN_PERCEPTION_MODE:-lite}"
echo "[RealRobot] ros user=$(id -un) euid=${EUID} ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-default} RMW_FASTRTPS_USE_SHM=${RMW_FASTRTPS_USE_SHM:-unset} FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-unset}"
if [[ "${SPACEVLN_PERCEPTION_MODE:-lite}" =~ ^(full|grounded_sam|groundingdino|groundedsam)$ ]]; then
  echo "[RealRobot] accel CUDA_HOME=${CUDA_HOME:-none} torch_lib=${SPACEVLN_TORCH_LIB_DIR:-none}"
fi

EXECUTOR_PID=""
cleanup() {
  if [[ -n "${EXECUTOR_PID}" ]]; then
    kill "${EXECUTOR_PID}" >/dev/null 2>&1 || true
    wait "${EXECUTOR_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_for_ros2_topic_count() {
  local topic="$1"
  local field="$2"
  local expected_min="$3"
  local attempts="${4:-20}"
  local count=""

  if ! command -v ros2 >/dev/null 2>&1; then
    return 0
  fi

  for _ in $(seq 1 "${attempts}"); do
    count="$(
      ros2 topic info "${topic}" 2>/dev/null \
        | awk -F': ' -v field="${field}" '$1 == field {print $2; exit}'
    )"
    if [[ -n "${count}" && "${count}" =~ ^[0-9]+$ && "${count}" -ge "${expected_min}" ]]; then
      return 0
    fi
    sleep 0.25
  done

  echo "[RealRobot] waiting for ${topic} ${field} >= ${expected_min} timed out (last=${count:-unknown})" >&2
  return 1
}

if [[ "${START_EXECUTOR}" == "1" ]]; then
  if [[ "${REAL_ACTION_EXECUTOR}" == "manual" ]]; then
    if [[ "${REAL_CONSOLE}" == "full" ]]; then
      python3 -u "${REAL_DIR}/run_manual_action_executor.py" &
    else
      python3 -u "${REAL_DIR}/run_manual_action_executor.py" >"${MANUAL_EXECUTOR_LOG}" 2>&1 &
    fi
  else
    EXECUTOR_ARGS=(
      --cmd-vel-topic "${CMD_VEL_TOPIC}"
      --odom-topic "${ODOM_TOPIC}"
      --control-mode "${CONTROL_MODE}"
      --control-rate-hz "${CONTROL_RATE_HZ}"
      --position-tolerance-m "${POSITION_TOLERANCE_M}"
      --angle-tolerance-deg "${ANGLE_TOLERANCE_DEG}"
      --odom-timeout-s "${ODOM_TIMEOUT_S}"
      --default-linear-speed-mps "${DEFAULT_LINEAR_SPEED_MPS}"
      --default-angular-speed-deg-s "${DEFAULT_ANGULAR_SPEED_DEG_S}"
      --max-linear-speed-mps "${MAX_LINEAR_SPEED_MPS}"
      --max-angular-speed-deg-s "${MAX_ANGULAR_SPEED_DEG_S}"
      --completion-stability-s "${COMPLETION_STABILITY_S}"
      --completion-yaw-tolerance-deg "${COMPLETION_YAW_TOLERANCE_DEG}"
    )
    if [[ "${REAL_CONSOLE}" == "full" ]]; then
      python3 -u "${REAL_DIR}/run_cmd_vel_executor.py" "${EXECUTOR_ARGS[@]}" &
    else
      python3 -u "${REAL_DIR}/run_cmd_vel_executor.py" "${EXECUTOR_ARGS[@]}" >"${EXECUTOR_LOG}" 2>&1 &
    fi
  fi
  EXECUTOR_PID="$!"
  sleep 1
  if ! kill -0 "${EXECUTOR_PID}" >/dev/null 2>&1; then
    echo "[RealRobot] action executor exited before navigation started" >&2
    if [[ "${REAL_ACTION_EXECUTOR}" == "manual" ]]; then
      if [[ -f "${MANUAL_EXECUTOR_LOG}" ]]; then
        echo "[RealRobot] executor log=${MANUAL_EXECUTOR_LOG}" >&2
        tail -n 80 "${MANUAL_EXECUTOR_LOG}" >&2 || true
      fi
    elif [[ -f "${EXECUTOR_LOG}" ]]; then
      echo "[RealRobot] executor log=${EXECUTOR_LOG}" >&2
      tail -n 80 "${EXECUTOR_LOG}" >&2 || true
    fi
    wait "${EXECUTOR_PID}" || true
    exit 1
  fi
  if [[ "${REAL_ACTION_EXECUTOR}" != "manual" ]]; then
    wait_for_ros2_topic_count "${CMD_VEL_TOPIC}" "Subscription count" 1 8 || true
  fi
  wait_for_ros2_topic_count "/spacevln/action_cmd" "Subscription count" 1 20
  wait_for_ros2_topic_count "/spacevln/action_status" "Publisher count" 1 20
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

if [[ "${REAL_CONSOLE}" == "full" ]]; then
  python3 -u "${REAL_DIR}/run_real_navigation.py" "${NAV_ARGS[@]}"
else
  set +e
  python3 -u "${REAL_DIR}/run_real_navigation.py" "${NAV_ARGS[@]}" 2>&1 \
    | tee "${NAVIGATION_LOG}" \
    | awk '
      /^Traceback/ {
        in_traceback = 1;
        print;
        fflush();
        next;
      }
      in_traceback {
        print;
        fflush();
        if ($0 ~ /^[A-Za-z_][A-Za-z0-9_]*(Error|Exception|Warning|Interrupt|Exit):/) {
          in_traceback = 0;
        }
        next;
      }
      /^\[REAL\]/ ||
      /^\[REAL-LIVE\]/ ||
      /^\[ERR\]/ ||
      /^\[WARN\]/ ||
      /^\[LLM\] Planning/ ||
      /^Episode [0-9]+/ ||
      /^Instruction:/ ||
      /^[A-Za-z_][A-Za-z0-9_]*(Error|Exception):/ {
        print;
        fflush();
      }
    '
  nav_status=${PIPESTATUS[0]}
  set -e
  if [[ "${nav_status}" -ne 0 ]]; then
    echo "[RealRobot] navigation failed exit=${nav_status}; log=${NAVIGATION_LOG}" >&2
    if [[ -f "${EXECUTOR_LOG}" ]]; then
      echo "[RealRobot] executor log=${EXECUTOR_LOG}" >&2
    fi
  fi
  exit "${nav_status}"
fi
