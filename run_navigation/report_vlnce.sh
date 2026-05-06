#!/bin/bash
# Legacy alias. Use report_r2rce.sh for new commands.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/report_r2rce.sh" "$@"
