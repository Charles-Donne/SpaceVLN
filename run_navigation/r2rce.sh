#!/bin/bash
# Preferred R2R-CE launcher. The legacy vlnce.sh name still works.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/vlnce.sh" "$@"
