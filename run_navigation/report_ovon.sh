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
    if [ -x "$HOME/.conda/envs/spacevln_ovon/bin/python" ]; then
        printf '%s\n' "$HOME/.conda/envs/spacevln_ovon/bin/python"
        return
    fi
    if [ -x "$HOME/anaconda3/envs/ovon/bin/python" ]; then
        printf '%s\n' "$HOME/anaconda3/envs/ovon/bin/python"
        return
    fi
    if [ -x "$HOME/.conda/envs/ovon/bin/python" ]; then
        printf '%s\n' "$HOME/.conda/envs/ovon/bin/python"
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
HELP_REQUESTED=0
SUMMARY_ONLY=0
LOAD_WORKERS="${SPACEVLN_REPORT_WORKERS:-}"
POSITIONAL_ARGS=()

is_sample_selector() {
    local raw="${1:-}"
    [[ "$raw" == "all" || "$raw" =~ ^[0-9]+$ ]]
}

while (( $# > 0 )); do
    case "$1" in
        -h|--help|help)
            HELP_REQUESTED=1
            shift
            ;;
        --fast|--summary-only)
            SUMMARY_ONLY=1
            shift
            ;;
        --workers|--load-workers)
            if (( $# < 2 )); then
                echo "❌ Missing value for $1" >&2
                exit 1
            fi
            LOAD_WORKERS="$2"
            shift 2
            ;;
        --)
            shift
            while (( $# > 0 )); do
                POSITIONAL_ARGS+=("$1")
                shift
            done
            ;;
        -*)
            echo "❌ Unknown option: $1" >&2
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"

if (( $# > 3 )); then
    echo "❌ Too many arguments: expected at most 3, got $#." >&2
    exit 1
fi

ARG1="${1:-}"
ARG2="${2:-}"
ARG3="${3:-}"
START_RAW="all"
END_RAW="all"
RESULTS_SELECTOR="$DEFAULT_RESULTS_DIR"

if (( HELP_REQUESTED == 1 )) || [[ "$ARG1" == "-h" || "$ARG1" == "--help" || "$ARG1" == "help" ]]; then
    cat <<'EOF'
Usage:
    bash run_navigation/report_ovon.sh [--fast] [--workers N] [start_sample|all] [end_sample|all] [results_dir]
    bash run_navigation/report_ovon.sh [start_sample|all] [end_sample|all] [results_dir]
    bash run_navigation/report_ovon.sh [start_sample|all] [results_dir]
    bash run_navigation/report_ovon.sh [results_dir]

Examples:
    bash run_navigation/report_ovon.sh 1 100
    bash run_navigation/report_ovon.sh 501 600 result/ovon/qwen3.5-plus__qwen3.5-flash_cache
    bash run_navigation/report_ovon.sh all result/ovon/qwen3.5-plus__qwen3.5-flash_cache
    bash run_navigation/report_ovon.sh --fast all result/ovon/qwen3.5-plus__qwen3.5-flash_cache
    bash run_navigation/report_ovon.sh all all

Notes:
  - Reads existing OVON sample logs only; does not rerun episodes.
  - Range uses sample index, not episode id.
  - If `end_sample` is omitted, it defaults to `all`.
  - Default mode now loads sample logs in parallel.
  - `--fast` saves only `summary.txt` + `metrics.json`.
EOF
    exit 0
fi

if (( $# == 1 )); then
    if is_sample_selector "$ARG1"; then
        START_RAW="$ARG1"
    else
        RESULTS_SELECTOR="$ARG1"
    fi
elif (( $# == 2 )); then
    if is_sample_selector "$ARG1" && is_sample_selector "$ARG2"; then
        START_RAW="$ARG1"
        END_RAW="$ARG2"
    elif is_sample_selector "$ARG1"; then
        START_RAW="$ARG1"
        RESULTS_SELECTOR="$ARG2"
    else
        echo "❌ Invalid arguments: when passing 2 positional args, use either [start end] or [start results_dir]." >&2
        exit 1
    fi
elif (( $# == 3 )); then
    START_RAW="$ARG1"
    END_RAW="$ARG2"
    RESULTS_SELECTOR="$ARG3"
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
SUMMARY_ARG=()
LOAD_WORKERS_ARG=()
if [[ "$START_RAW" != "all" && -n "$START_RAW" ]]; then
    START_ARG=(--start-index "$START_RAW")
fi
if [[ "$END_RAW" != "all" && -n "$END_RAW" ]]; then
    END_ARG=(--end-index "$END_RAW")
fi
if (( SUMMARY_ONLY == 1 )); then
    SUMMARY_ARG=(--summary-only)
fi
if [[ -n "$LOAD_WORKERS" ]]; then
    LOAD_WORKERS_ARG=(--load-workers "$LOAD_WORKERS")
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m navigation_system.runtime.object_navigation.report_range \
    --path "$RESULTS_DIR" \
    --exp-config "$OVON_EXP_CONFIG" \
    --split "val_unseen" \
    --data-path "$OVON_DATA_PATH" \
    "${SUMMARY_ARG[@]}" \
    "${LOAD_WORKERS_ARG[@]}" \
    "${START_ARG[@]}" \
    "${END_ARG[@]}"
