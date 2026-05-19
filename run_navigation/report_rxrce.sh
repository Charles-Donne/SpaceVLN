#!/bin/bash
# Generate a compact RxR-CE Markdown report from existing sample-index logs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXP_CONFIG="${EXP_CONFIG:-navigation_system/config/experiments/vlnce/rxr_eval.yaml}"
export SPACEVLN_REPORT_FAMILY="${SPACEVLN_REPORT_FAMILY:-rxrce}"
export SPACEVLN_REPORT_TITLE="${SPACEVLN_REPORT_TITLE:-RxR-CE report}"
export SPACEVLN_REPORT_RANGE_KEY="${SPACEVLN_REPORT_RANGE_KEY:-sample_index}"
export SPACEVLN_REPORT_RUNTIME="${SPACEVLN_REPORT_RUNTIME:-standard}"

exec bash "$SCRIPT_DIR/report_r2rce.sh" "$@"
