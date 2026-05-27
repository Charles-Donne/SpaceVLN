#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPACEVLN_DIR="$(cd "${REAL_DIR}/.." && pwd)"

if [[ -n "${SPACEVLN_ROS_SETUP:-}" && -f "${SPACEVLN_ROS_SETUP}" ]]; then
  # shellcheck disable=SC1090
  source "${SPACEVLN_ROS_SETUP}"
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_real_accel_env.sh"

export PYTHONPATH="${SPACEVLN_DIR}:${REAL_DIR}:${PYTHONPATH:-}"

cd "${SPACEVLN_DIR}"

python3 "${REAL_DIR}/run_cmd_vel_executor.py" "$@"
