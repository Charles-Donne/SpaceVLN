#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPACEVLN_DIR="$(cd "${REAL_DIR}/.." && pwd)"

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

if [[ -n "${SPACEVLN_ROS_SETUP:-}" && -f "${SPACEVLN_ROS_SETUP}" ]]; then
  source_setup_bash_safely "${SPACEVLN_ROS_SETUP}"
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  source_setup_bash_safely /opt/ros/humble/setup.bash
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_real_accel_env.sh"

export PYTHONPATH="${SPACEVLN_DIR}:${REAL_DIR}:${PYTHONPATH:-}"

cd "${SPACEVLN_DIR}"

python3 "${REAL_DIR}/run_cmd_vel_executor.py" "$@"
