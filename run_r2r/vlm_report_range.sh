#!/bin/bash
# Generate a partial SpaceVLN report from existing log files only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

PROJECT_ROOT="$(spacevln_project_root)"
CONFIG_FILE="${EXP_CONFIG:-navigation_system/config/experiments/r2r_eval.yaml}"

START_ID=${1:-}
END_ID=${2:-}
RESULTS_DIR_ARG=${3:-}

MIN_EPISODE_ID=1
MAX_EPISODE_ID=1800

PYTHON_BIN="$(spacevln_select_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"
RESULTS_ROOT=""
REPORT_LOAD_WORKERS="${SPACEVLN_REPORT_WORKERS:-8}"

usage() {
    echo "用法:"
    echo "  bash run_r2r/vlm_report_range.sh [start_episode_id] [end_episode_id] [results_dir]"
    echo ""
    echo "示例:"
    echo "  bash run_r2r/vlm_report_range.sh 1500 1799"
    echo "  bash run_r2r/vlm_report_range.sh 1 50 qwen3.5-plus__qwen3.5-flash_cache"
    echo "  bash run_r2r/vlm_report_range.sh 1500 1799 /abs/path/to/result/vlnce/qwen3.5-plus__qwen3.5-flash"
    echo ""
    echo "说明:"
    echo "  只读取已有 log 生成部分汇总，不会重新跑 episode。"
    echo "  第3个参数既可以传完整 results_dir，也可以直接传默认 result/vlnce 下的实验文件夹名。"
}

print_available_results_dirs() {
    if [ -z "$RESULTS_ROOT" ]; then
        RESULTS_ROOT="$(spacevln_default_results_root "$PYTHON_BIN")"
    fi
    echo "可选实验目录:"
    if [ ! -d "$RESULTS_ROOT" ]; then
        echo "  (results 根目录不存在: $RESULTS_ROOT)"
        return
    fi

    local found_any=0
    while IFS= read -r name; do
        found_any=1
        echo "  - $name"
    done < <(find "$RESULTS_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

    if [ "$found_any" -eq 0 ]; then
        echo "  (当前没有实验目录)"
    fi
}

resolve_results_dir() {
    local raw_arg="$1"
    local default_dir="$2"

    if [ -z "$raw_arg" ]; then
        printf '%s\n' "$default_dir"
        return 0
    fi

    if [[ "$raw_arg" = /* ]]; then
        printf '%s\n' "$raw_arg"
        return 0
    fi

    if [ -d "$raw_arg" ]; then
        local abs_dir
        abs_dir="$(cd "$raw_arg" && pwd)"
        printf '%s\n' "$abs_dir"
        return 0
    fi

    if [ -z "$RESULTS_ROOT" ]; then
        RESULTS_ROOT="$(spacevln_default_results_root "$PYTHON_BIN")"
    fi
    if [ -d "$RESULTS_ROOT/$raw_arg" ]; then
        printf '%s\n' "$RESULTS_ROOT/$raw_arg"
        return 0
    fi

    printf '%s\n' "$raw_arg"
    return 0
}

if [ -z "$START_ID" ] || [ -z "$END_ID" ]; then
    usage
    exit 1
fi

if ! [[ "$START_ID" =~ ^[0-9]+$ ]]; then
    echo "❌ start_episode_id 必须是正整数: $START_ID"
    exit 1
fi

if ! [[ "$END_ID" =~ ^[0-9]+$ ]]; then
    echo "❌ end_episode_id 必须是正整数: $END_ID"
    exit 1
fi

if [ "$START_ID" -lt "$MIN_EPISODE_ID" ] || [ "$START_ID" -gt "$MAX_EPISODE_ID" ]; then
    echo "❌ start_episode_id 超出范围 [$MIN_EPISODE_ID, $MAX_EPISODE_ID]: $START_ID"
    exit 1
fi

if [ "$END_ID" -lt "$MIN_EPISODE_ID" ] || [ "$END_ID" -gt "$MAX_EPISODE_ID" ]; then
    echo "❌ end_episode_id 超出范围 [$MIN_EPISODE_ID, $MAX_EPISODE_ID]: $END_ID"
    exit 1
fi

if [ "$END_ID" -lt "$START_ID" ]; then
    echo "❌ end_episode_id 不能小于 start_episode_id: $START_ID -> $END_ID"
    exit 1
fi

if ! [[ "$REPORT_LOAD_WORKERS" =~ ^[0-9]+$ ]] || [ "$REPORT_LOAD_WORKERS" -lt 1 ]; then
    echo "❌ SPACEVLN_REPORT_WORKERS 必须是大于等于 1 的正整数: $REPORT_LOAD_WORKERS"
    exit 1
fi

cd "$PROJECT_ROOT"

DEFAULT_RESULTS_DIR=""
if [ -z "$RESULTS_DIR_ARG" ]; then
    DEFAULT_RESULTS_DIR="$(spacevln_default_results_dir "$PYTHON_BIN" "navigation_system/config/vlm/vlm_api_config.yaml")"
fi

RESULTS_DIR="$(resolve_results_dir "$RESULTS_DIR_ARG" "$DEFAULT_RESULTS_DIR")"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ 目录不存在: $RESULTS_DIR"
    echo ""
    print_available_results_dirs
    echo ""
    echo "示例:"
    echo "  bash run_r2r/vlm_report_range.sh $START_ID $END_ID qwen3.5-plus__qwen3.5-flash_cache"
    exit 1
fi

echo "📊 生成部分结果报告"
echo "   Range: $START_ID-$END_ID"
echo "   Source: $RESULTS_DIR"
echo "   Output: $RESULTS_DIR/reports/$START_ID-$END_ID"
echo "   Load workers: $REPORT_LOAD_WORKERS"

"$PYTHON_BIN" -m navigation_system.runtime.results_report \
    --path "$RESULTS_DIR" \
    --exp-config "$CONFIG_FILE" \
    --save \
    --start-id "$START_ID" \
    --end-id "$END_ID" \
    --load-workers "$REPORT_LOAD_WORKERS"
