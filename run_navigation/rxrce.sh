#!/bin/bash
# RxR-CE launcher. Reuses the R2R-CE runtime with the RxR task config.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXP_CONFIG="${EXP_CONFIG:-navigation_system/config/experiments/vlnce/rxr_eval.yaml}"
export SPACEVLN_RESULTS_FAMILY="${SPACEVLN_RESULTS_FAMILY:-rxrce}"

exec "$SCRIPT_DIR/r2rce.sh" "$@"
