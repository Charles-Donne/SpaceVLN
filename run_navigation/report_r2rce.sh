#!/bin/bash
# Preferred R2R-CE report entrypoint. Kept separate from the legacy name so
# existing commands can still call report_vlnce.sh during the transition.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/report_vlnce.sh" "$@"
