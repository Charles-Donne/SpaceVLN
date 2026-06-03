#!/usr/bin/env bash
set -euo pipefail

# Manual mode: lookaround/refresh actions still use the automatic cmd_vel
# executor; action-stage commands marked manual_required wait for operator
# confirmation instead of publishing /cmd_vel.
export REAL_ACTION_EXECUTOR="manual"
export SPACEVLN_REAL_MOTION_MODE="manual"
export SPACEVLN_PERCEPTION_MODE="${SPACEVLN_PERCEPTION_MODE:-full}"
export SPACEVLN_DISABLE_GROUNDED_SAM="${SPACEVLN_DISABLE_GROUNDED_SAM:-0}"
export SPACEVLN_REQUIRE_GROUNDINGDINO="${SPACEVLN_REQUIRE_GROUNDINGDINO:-1}"
export SPACEVLN_REQUIRE_SAM="${SPACEVLN_REQUIRE_SAM:-1}"
export SPACEVLN_DISABLE_LANDMARK_AUTOSTOP="${SPACEVLN_DISABLE_LANDMARK_AUTOSTOP:-1}"
export SPACEVLN_MANUAL_PROMPT_ONLY="${SPACEVLN_MANUAL_PROMPT_ONLY:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_run_real_robot_impl.sh" "$@"
