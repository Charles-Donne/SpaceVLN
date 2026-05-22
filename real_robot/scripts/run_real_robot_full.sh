#!/usr/bin/env bash
set -euo pipefail

# Full perception launcher: require GroundingDINO and SAM.
# Set SPACEVLN_REQUIRE_SAM=0 if you want to allow GroundingDINO box masks only.
export SPACEVLN_PERCEPTION_MODE="${SPACEVLN_PERCEPTION_MODE:-full}"
export SPACEVLN_DISABLE_GROUNDED_SAM="${SPACEVLN_DISABLE_GROUNDED_SAM:-0}"
export SPACEVLN_REQUIRE_GROUNDINGDINO="${SPACEVLN_REQUIRE_GROUNDINGDINO:-1}"
export SPACEVLN_REQUIRE_SAM="${SPACEVLN_REQUIRE_SAM:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_real_robot_simple.sh" "$@"
