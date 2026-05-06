#!/bin/bash
# Generate a compact OVON Markdown report from existing sample logs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/report_common.sh"

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

usage() {
    cat <<'EOF'
Usage:
  bash run_navigation/report_ovon.sh [all|start end|start-end] [model]

Examples:
  bash run_navigation/report_ovon.sh all
  bash run_navigation/report_ovon.sh 1 100
  bash run_navigation/report_ovon.sh 501-600 qwen3.5-plus__qwen3.5-flash_cache
  bash run_navigation/report_ovon.sh all --model mimo-v2.5__mimo-v2-omni

Output:
  result/ovon/<model>/reports/<all|start-end>/episode_results.md
EOF
}

PROJECT_ROOT="$(spacevln_project_root)"
PYTHON_BIN="$(spacevln_select_objectnav_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"

NAV_WS_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
API_CONFIG="${VLM_API_CONFIG:-navigation_system/config/vlm/vlm_api_config.yaml}"
OVON_EXP_CONFIG="$NAV_WS_ROOT/ovon/config/experiments/transformer_dagger.yaml"
OVON_DATA_PATH="$NAV_WS_ROOT/data/datasets/ovon/hm3d/v1/val_unseen/val_unseen_hard.json.gz"
WORKERS="${SPACEVLN_REPORT_WORKERS:-$(spacevln_report_default_workers)}"
MODEL=""
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
if [ "${#REMAINING[@]}" -gt 0 ] && [ -z "$MODEL" ]; then
    MODEL="${REMAINING[0]}"
fi
if [ "${#REMAINING[@]}" -gt 1 ]; then
    echo "Too many arguments after range. Use: [model]." >&2
    exit 1
fi

if [ -z "$MODEL" ]; then
    MODEL="$(spacevln_report_model_dir_name "$PYTHON_BIN" "$API_CONFIG")"
fi

OVON_ROOT="$(spacevln_report_family_root "$PYTHON_BIN" "ovon")"
if [ -n "$RESULTS_DIR_OVERRIDE" ]; then
    RESULTS_DIR="$(spacevln_report_resolve_results_dir "$RESULTS_DIR_OVERRIDE" "$OVON_ROOT" "$OVON_ROOT/$MODEL")"
else
    RESULTS_DIR="$OVON_ROOT/$MODEL"
fi

if [ ! -d "$RESULTS_DIR/log" ]; then
    echo "Missing log directory: $RESULTS_DIR/log" >&2
    exit 1
fi

START_ARG=()
END_ARG=()
if [ -n "$REPORT_START" ]; then
    START_ARG=(--start-index "$REPORT_START")
fi
if [ -n "$REPORT_END" ]; then
    END_ARG=(--end-index "$REPORT_END")
fi

echo "OVON report"
echo "  Range: ${REPORT_RANGE_LABEL}"
echo "  Model: $MODEL"
echo "  Source: $RESULTS_DIR"
echo "  Output: $RESULTS_DIR/reports/$REPORT_RANGE_LABEL/episode_results.md"
echo "  Workers: $WORKERS"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m navigation_system.runtime.object_navigation.ovon.report_range \
    --path "$RESULTS_DIR" \
    --exp-config "$OVON_EXP_CONFIG" \
    --split "val_unseen" \
    --data-path "$OVON_DATA_PATH" \
    --md-only \
    --load-workers "$WORKERS" \
    "${START_ARG[@]}" \
    "${END_ARG[@]}"
