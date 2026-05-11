#!/bin/bash
# Generate a compact NavGBench Markdown report from existing logs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/report_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash run_navigation/report_navgbench.sh [all|start end|start-end] [model] [complex|simple|moving]

Examples:
  bash run_navigation/report_navgbench.sh all
  bash run_navigation/report_navgbench.sh 1 100
  bash run_navigation/report_navgbench.sh 1-100 qwen3.5-plus__qwen3.5-flash_cache complex
  bash run_navigation/report_navgbench.sh all --mode simple --model mimo-v2.5__mimo-v2-omni

Output:
  all:    result/navgbench/<mode>/<model>/episode_results.md
  range:  result/navgbench/<mode>/<model>/reports/<start-end>/episode_results.md
EOF
}

is_navgbench_mode() {
    case "${1,,}" in
        complex|simple|moving)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

PROJECT_ROOT="$(spacevln_project_root)"
PYTHON_BIN="$(spacevln_select_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"

API_CONFIG="${VLM_API_CONFIG:-navigation_system/config/vlm/vlm_api_config.yaml}"
EXP_CONFIG="${EXP_CONFIG:-navigation_system/config/experiments/vlnce/navgbench_eval.yaml}"
WORKERS="${SPACEVLN_REPORT_WORKERS:-$(spacevln_report_default_workers)}"
MODEL=""
MODE="${SPACEVLN_NAVGBENCH_INSTRUCTION_MODE:-complex}"
RESULTS_DIR_OVERRIDE=""
POSITIONAL_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --model)
            MODEL="${2:-}"
            shift 2
            ;;
        --model=*)
            MODEL="${1#*=}"
            shift
            ;;
        --mode|--instruction-mode)
            MODE="${2:-}"
            shift 2
            ;;
        --mode=*|--instruction-mode=*)
            MODE="${1#*=}"
            shift
            ;;
        --results-dir)
            RESULTS_DIR_OVERRIDE="${2:-}"
            shift 2
            ;;
        --results-dir=*)
            RESULTS_DIR_OVERRIDE="${1#*=}"
            shift
            ;;
        --workers|--load-workers)
            WORKERS="${2:-}"
            shift 2
            ;;
        --workers=*|--load-workers=*)
            WORKERS="${1#*=}"
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [ "$WORKERS" -lt 1 ]; then
    echo "workers must be a positive integer: $WORKERS" >&2
    exit 1
fi

spacevln_report_parse_range "${POSITIONAL_ARGS[@]}"
REMAINING=("${REPORT_REMAINING_ARGS[@]}")
for token in "${REMAINING[@]}"; do
    if is_navgbench_mode "$token"; then
        MODE="${token,,}"
    elif [ -z "$MODEL" ]; then
        MODEL="$token"
    else
        echo "Too many arguments after range. Use: [model] [complex|simple|moving]." >&2
        exit 1
    fi
done

MODE="${MODE,,}"
if ! is_navgbench_mode "$MODE"; then
    echo "NavGBench mode must be complex, simple, or moving: $MODE" >&2
    exit 1
fi
if [ -z "$MODEL" ]; then
    MODEL="$(spacevln_report_model_dir_name "$PYTHON_BIN" "$API_CONFIG")"
fi

NAVGBENCH_ROOT="$(spacevln_report_family_root "$PYTHON_BIN" "navgbench")"
DEFAULT_DIR="$NAVGBENCH_ROOT/$MODE/$MODEL"
if [ -n "$RESULTS_DIR_OVERRIDE" ]; then
    RESULTS_DIR="$(spacevln_report_resolve_results_dir "$RESULTS_DIR_OVERRIDE" "$NAVGBENCH_ROOT" "$DEFAULT_DIR")"
else
    RESULTS_DIR="$DEFAULT_DIR"
fi

if [ ! -d "$RESULTS_DIR/log" ]; then
    echo "Missing log directory: $RESULTS_DIR/log" >&2
    exit 1
fi

echo "NavGBench report"
echo "  Range: ${REPORT_RANGE_LABEL}"
echo "  Mode: $MODE"
echo "  Model: $MODEL"
echo "  Source: $RESULTS_DIR"
REPORT_MARKDOWN_TITLE="NavGBench report | mode: $MODE | model: $MODEL | range: $REPORT_RANGE_LABEL"
if [ "$REPORT_RANGE_LABEL" = "all" ]; then
    echo "  Output: $RESULTS_DIR/episode_results.md"
else
    echo "  Output: $RESULTS_DIR/reports/$REPORT_RANGE_LABEL/episode_results.md"
fi
echo "  Workers: $WORKERS"

spacevln_report_run_results_report_md \
    "$PYTHON_BIN" \
    "$PROJECT_ROOT" \
    "$RESULTS_DIR" \
    "$EXP_CONFIG" \
    "$WORKERS" \
    "$REPORT_START" \
    "$REPORT_END" \
    "episode_id" \
    "$REPORT_MARKDOWN_TITLE"
