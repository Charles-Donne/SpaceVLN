#!/bin/bash
# Generate an OVON partial report from existing sample logs only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

spacevln_select_objectnav_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        printf '%s\n' "$PYTHON_BIN"
        return
    fi
    if [ -x "$HOME/anaconda3/envs/spacevln_ovon/bin/python" ]; then
        printf '%s\n' "$HOME/anaconda3/envs/spacevln_ovon/bin/python"
        return
    fi
    if [ -x "$HOME/anaconda3/envs/ovon/bin/python" ]; then
        printf '%s\n' "$HOME/anaconda3/envs/ovon/bin/python"
        return
    fi
    spacevln_select_python
}

PROJECT_ROOT="$(spacevln_project_root)"
PYTHON_BIN="$(spacevln_select_objectnav_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"
NAV_WS_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
OVON_EXP_CONFIG="$NAV_WS_ROOT/ovon/config/experiments/transformer_dagger.yaml"
OVON_DATA_PATH="$NAV_WS_ROOT/data/datasets/ovon/hm3d/v1/val_unseen/val_unseen_hard.json.gz"

DEFAULT_RESULTS_DIR="$(
    cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY'
from navigation_system.runtime.storage.results_layout import build_default_results_family_root
from pathlib import Path
print(Path(build_default_results_family_root("ovon")).resolve() / "qwen3.5-plus__qwen3.5-flash_cache")
PY
)"
DEFAULT_RESULTS_ROOT="$(dirname "$DEFAULT_RESULTS_DIR")"

START_RAW="${1:-all}"
END_RAW="${2:-all}"
RESULTS_SELECTOR="${3:-$DEFAULT_RESULTS_DIR}"

if [[ "$START_RAW" == "-h" || "$START_RAW" == "--help" || "$START_RAW" == "help" ]]; then
    cat <<'EOF'
Usage:
  bash run_navigation/report_ovon_range.sh [start_sample|all] [end_sample|all] [results_dir]

Examples:
  bash run_navigation/report_ovon_range.sh 1 100
  bash run_navigation/report_ovon_range.sh 501 600 result/ovon/qwen3.5-plus__qwen3.5-flash_cache
  bash run_navigation/report_ovon_range.sh all all

Notes:
  - Reads existing OVON sample logs only; does not rerun episodes.
  - Range uses sample index, not episode id.
EOF
    exit 0
fi

if [[ "$RESULTS_SELECTOR" != /* ]]; then
    if [[ -d "$RESULTS_SELECTOR" ]]; then
        RESULTS_DIR="$(cd "$RESULTS_SELECTOR" && pwd)"
    elif [[ -d "$NAV_WS_ROOT/$RESULTS_SELECTOR" ]]; then
        RESULTS_DIR="$(cd "$NAV_WS_ROOT/$RESULTS_SELECTOR" && pwd)"
    elif [[ -d "$PROJECT_ROOT/$RESULTS_SELECTOR" ]]; then
        RESULTS_DIR="$(cd "$PROJECT_ROOT/$RESULTS_SELECTOR" && pwd)"
    elif [[ -d "$DEFAULT_RESULTS_ROOT/$RESULTS_SELECTOR" ]]; then
        RESULTS_DIR="$(cd "$DEFAULT_RESULTS_ROOT/$RESULTS_SELECTOR" && pwd)"
    else
        RESULTS_DIR="$RESULTS_SELECTOR"
    fi
else
    RESULTS_DIR="$RESULTS_SELECTOR"
fi

START_ARG=()
END_ARG=()
if [[ "$START_RAW" != "all" && -n "$START_RAW" ]]; then
    START_ARG=(--start-index "$START_RAW")
fi
if [[ "$END_RAW" != "all" && -n "$END_RAW" ]]; then
    END_ARG=(--end-index "$END_RAW")
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m navigation_system.runtime.object_navigation.report_range \
    --path "$RESULTS_DIR" \
    --exp-config "$OVON_EXP_CONFIG" \
    --split "val_unseen" \
    --data-path "$OVON_DATA_PATH" \
    "${START_ARG[@]}" \
    "${END_ARG[@]}"
