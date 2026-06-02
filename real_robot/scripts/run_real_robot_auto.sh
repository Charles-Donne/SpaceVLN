#!/usr/bin/env bash
set -euo pipefail

# Full automatic mode: the agent sends actions to the cmd_vel executor.
# Perception defaults to full GroundingDINO/SAM, but can be overridden from env.
export REAL_ACTION_EXECUTOR="cmd_vel"
export SPACEVLN_REAL_MOTION_MODE="auto"
export SPACEVLN_PERCEPTION_MODE="${SPACEVLN_PERCEPTION_MODE:-full}"
export SPACEVLN_DISABLE_GROUNDED_SAM="${SPACEVLN_DISABLE_GROUNDED_SAM:-0}"
export SPACEVLN_REQUIRE_GROUNDINGDINO="${SPACEVLN_REQUIRE_GROUNDINGDINO:-1}"
export SPACEVLN_REQUIRE_SAM="${SPACEVLN_REQUIRE_SAM:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_run_real_robot_impl.sh" "$@"
