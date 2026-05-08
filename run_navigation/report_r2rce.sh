#!/bin/bash
# Generate a compact R2R-CE/RxR-CE Markdown report from existing logs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/report_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash run_navigation/report_r2rce.sh [all|start end|start-end] [model] [ablation]

Examples:
  bash run_navigation/report_r2rce.sh all
  bash run_navigation/report_r2rce.sh 1000 1300
  bash run_navigation/report_r2rce.sh 1000-1300 qwen3.5-plus__qwen3.5-flash_cache
  bash run_navigation/report_r2rce.sh all qwen3.5-plus__qwen3.5-flash_cache no-landmark
  bash run_navigation/report_r2rce.sh 1000 1300 --model qwen3.5-plus__qwen3.5-flash_cache --ablation no-landmark

Output:
  all:       result/<family>/<model>/episode_results.md
  range:     result/<family>/<model>/reports/<start-end>/episode_results.md
  ablation:  result/<family>/ablation/<ablation>/<model>/...
EOF
}

PROJECT_ROOT="$(spacevln_project_root)"
PYTHON_BIN="$(spacevln_select_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"

API_CONFIG="${VLM_API_CONFIG:-navigation_system/config/vlm/vlm_api_config.yaml}"
EXP_CONFIG="${EXP_CONFIG:-navigation_system/config/experiments/vlnce/r2r_eval.yaml}"
REPORT_FAMILY="${SPACEVLN_REPORT_FAMILY:-r2rce}"
REPORT_TITLE="${SPACEVLN_REPORT_TITLE:-R2R-CE report}"
REPORT_RANGE_KEY="${SPACEVLN_REPORT_RANGE_KEY:-episode_id}"
WORKERS="${SPACEVLN_REPORT_WORKERS:-$(spacevln_report_default_workers)}"
RUNTIME_MODE="${SPACEVLN_REPORT_RUNTIME:-standard}"
MODEL=""
ABLATION=""
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
        --ablation)
            ABLATION="${2:-}"
            shift 2
            ;;
        --ablation=*)
            ABLATION="${1#*=}"
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
        --runtime)
            RUNTIME_MODE="${2:-}"
            shift 2
            ;;
        --runtime=*)
            RUNTIME_MODE="${1#*=}"
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
case "$RUNTIME_MODE" in
    standard|context_cache)
        ;;
    *)
        echo "runtime must be standard or context_cache: $RUNTIME_MODE" >&2
        exit 1
        ;;
esac

spacevln_report_parse_range "${POSITIONAL_ARGS[@]}"
REMAINING=("${REPORT_REMAINING_ARGS[@]}")
if [ "${#REMAINING[@]}" -gt 0 ] && [ -z "$MODEL" ]; then
    MODEL="${REMAINING[0]}"
fi
if [ "${#REMAINING[@]}" -gt 1 ] && [ -z "$ABLATION" ]; then
    ABLATION="${REMAINING[1]}"
fi
if [ "${#REMAINING[@]}" -gt 2 ]; then
    echo "Too many arguments after range. Use: [model] [ablation]." >&2
    exit 1
fi

if [ -z "$MODEL" ]; then
    MODEL="$(spacevln_report_model_dir_name "$PYTHON_BIN" "$API_CONFIG" "$RUNTIME_MODE")"
fi
ABLATION="${ABLATION#ablation/}"
ABLATION="${ABLATION%/}"

R2RCE_ROOT="$(spacevln_report_family_root "$PYTHON_BIN" "$REPORT_FAMILY")"
if [ -n "$RESULTS_DIR_OVERRIDE" ]; then
    RESULTS_DIR="$(spacevln_report_resolve_results_dir "$RESULTS_DIR_OVERRIDE" "$R2RCE_ROOT" "$R2RCE_ROOT/$MODEL")"
elif [ -n "$ABLATION" ]; then
    RESULTS_DIR="$R2RCE_ROOT/ablation/$ABLATION/$MODEL"
else
    RESULTS_DIR="$R2RCE_ROOT/$MODEL"
fi

if [ ! -d "$RESULTS_DIR/log" ]; then
    echo "Missing log directory: $RESULTS_DIR/log" >&2
    exit 1
fi

echo "$REPORT_TITLE"
echo "  Range: ${REPORT_RANGE_LABEL}"
echo "  Range key: $REPORT_RANGE_KEY"
echo "  Runtime: $RUNTIME_MODE"
echo "  Model: $MODEL"
if [ -n "$ABLATION" ]; then
    echo "  Ablation: $ABLATION"
fi
echo "  Source: $RESULTS_DIR"
if [ "$REPORT_RANGE_LABEL" = "all" ]; then
    echo "  Output: $RESULTS_DIR/episode_results.md"
else
    REPORT_OUTPUT_LABEL="$REPORT_RANGE_LABEL"
    if [ "$REPORT_RANGE_KEY" != "episode_id" ]; then
        REPORT_OUTPUT_LABEL="${REPORT_RANGE_KEY}_${REPORT_RANGE_LABEL}"
    fi
    echo "  Output: $RESULTS_DIR/reports/$REPORT_OUTPUT_LABEL/episode_results.md"
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
    "$REPORT_RANGE_KEY"
