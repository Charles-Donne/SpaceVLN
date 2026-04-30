#!/bin/bash
# Generate NavGBench reports from existing SpaceVLN logs only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

PROJECT_ROOT="$(spacevln_project_root)"
PYTHON_BIN="$(spacevln_select_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"

DEFAULT_RESULTS_ROOT="$(
    cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY'
from navigation_system.runtime.storage.results_layout import build_default_results_family_root
print(build_default_results_family_root("navgbench"))
PY
)"

RESULTS_SELECTOR="all"
REPORT_LOAD_WORKERS="${SPACEVLN_REPORT_WORKERS:-16}"
SUMMARY_ONLY=0

usage() {
    cat <<'EOF'
Usage:
  bash run_navigation/report_navgbench.sh [results_dir|name|all]
  bash run_navigation/report_navgbench.sh --results DIR|NAME|all
  bash run_navigation/report_navgbench.sh --fast all

Examples:
  bash run_navigation/report_navgbench.sh
  bash run_navigation/report_navgbench.sh all
  bash run_navigation/report_navgbench.sh complex/qwen3.5-plus__qwen3.5-flash_cache
  bash run_navigation/report_navgbench.sh /abs/path/to/result/navgbench/complex/qwen3.5-plus__qwen3.5-flash_cache

Notes:
  Reads existing log/episode_*.json files only and regenerates episode_results/metrics reports.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --fast|--summary-only)
            SUMMARY_ONLY=1
            shift
            ;;
        --workers|--load-workers)
            if (( $# < 2 )); then
                echo "Missing value for $1" >&2
                exit 1
            fi
            REPORT_LOAD_WORKERS="$2"
            shift 2
            ;;
        --results)
            if (( $# < 2 )); then
                echo "Missing value for --results" >&2
                exit 1
            fi
            RESULTS_SELECTOR="$2"
            shift 2
            ;;
        --results=*)
            RESULTS_SELECTOR="${1#*=}"
            shift
            ;;
        --*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            RESULTS_SELECTOR="$1"
            shift
            ;;
    esac
done

resolve_one_results_dir() {
    local raw="$1"
    if [[ "$raw" = /* ]]; then
        printf '%s\n' "$raw"
        return
    fi
    if [ -d "$raw" ]; then
        (cd "$raw" && pwd)
        return
    fi
    if [ -d "$DEFAULT_RESULTS_ROOT/$raw" ]; then
        (cd "$DEFAULT_RESULTS_ROOT/$raw" && pwd)
        return
    fi
    printf '%s\n' "$DEFAULT_RESULTS_ROOT/$raw"
}

RESULTS_TARGET_DIRS=()
if [[ -z "$RESULTS_SELECTOR" || "$RESULTS_SELECTOR" == "all" ]]; then
    if [ -d "$DEFAULT_RESULTS_ROOT" ]; then
        while IFS= read -r item; do
            RESULTS_TARGET_DIRS+=("$item")
        done < <(find "$DEFAULT_RESULTS_ROOT" -maxdepth 5 -type d -name log -printf '%h\n' | sort -u)
    fi
else
    IFS=',' read -ra SELECTORS <<< "$RESULTS_SELECTOR"
    for selector in "${SELECTORS[@]}"; do
        selector="${selector#"${selector%%[![:space:]]*}"}"
        selector="${selector%"${selector##*[![:space:]]}"}"
        if [ -n "$selector" ]; then
            RESULTS_TARGET_DIRS+=("$(resolve_one_results_dir "$selector")")
        fi
    done
fi

if [ "${#RESULTS_TARGET_DIRS[@]}" -eq 0 ]; then
    echo "No NavGBench reportable result directories found under: $DEFAULT_RESULTS_ROOT"
    exit 1
fi

overall_rc=0
for RESULTS_DIR in "${RESULTS_TARGET_DIRS[@]}"; do
    if [ ! -d "$RESULTS_DIR/log" ]; then
        echo "Skipping missing log directory: $RESULTS_DIR"
        overall_rc=1
        continue
    fi

    echo ""
    echo "Generating NavGBench report"
    echo "  Source: $RESULTS_DIR"
    echo "  Load workers: $REPORT_LOAD_WORKERS"

    cmd=(
        "$PYTHON_BIN" -m navigation_system.runtime.results_report
        --path "$RESULTS_DIR"
        --save
        --load-workers "$REPORT_LOAD_WORKERS"
    )
    if [ "$SUMMARY_ONLY" -eq 1 ]; then
        cmd+=(--summary-only)
    fi
    (cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" "${cmd[@]}") || overall_rc=1
done

exit "$overall_rc"
