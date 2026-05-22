#!/usr/bin/env bash
set -euo pipefail

# Lightweight real-robot launcher: no GroundingDINO/SAM dependency required.
export SPACEVLN_PERCEPTION_MODE="${SPACEVLN_PERCEPTION_MODE:-lite}"
export SPACEVLN_DISABLE_GROUNDED_SAM="${SPACEVLN_DISABLE_GROUNDED_SAM:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_real_robot_simple.sh" "$@"
