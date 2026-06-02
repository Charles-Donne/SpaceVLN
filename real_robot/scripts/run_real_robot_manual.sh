#!/usr/bin/env bash
set -euo pipefail

# Manual mode: the agent prints every requested action and waits for operator
# confirmation instead of publishing /cmd_vel.
export REAL_ACTION_EXECUTOR="manual"
export SPACEVLN_PERCEPTION_MODE="${SPACEVLN_PERCEPTION_MODE:-full}"
export SPACEVLN_DISABLE_GROUNDED_SAM="${SPACEVLN_DISABLE_GROUNDED_SAM:-0}"
export SPACEVLN_REQUIRE_GROUNDINGDINO="${SPACEVLN_REQUIRE_GROUNDINGDINO:-1}"
export SPACEVLN_REQUIRE_SAM="${SPACEVLN_REQUIRE_SAM:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_run_real_robot_impl.sh" "$@"
