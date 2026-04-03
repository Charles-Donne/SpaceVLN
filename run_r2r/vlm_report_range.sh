#!/bin/bash
# Generate a partial SpaceVLN report from existing log files only.
#
# Usage:
#   bash run_r2r/vlm_report_range.sh [start_episode_id] [end_episode_id] [results_dir]
#
# Examples:
#   bash run_r2r/vlm_report_range.sh 1500 1799
#   bash run_r2r/vlm_report_range.sh 1500 1799 ../data/result/spacevln

set -euo pipefail
trap 'echo "❌ 错误：脚本在第 $LINENO 行失败"; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

START_ID=${1:-}
END_ID=${2:-}
RESULTS_DIR_ARG=${3:-}

MIN_EPISODE_ID=1
MAX_EPISODE_ID=1800

if [ -x "$HOME/anaconda3/envs/spatial_agent/bin/python" ]; then
    PYTHON_BIN="$HOME/anaconda3/envs/spatial_agent/bin/python"
    SPATIAL_ENV="$HOME/anaconda3/envs/spatial_agent"
else
    PYTHON_BIN="python"
    SPATIAL_ENV=""
fi

if [ -n "$SPATIAL_ENV" ]; then
    export LD_LIBRARY_PATH="$SPATIAL_ENV/lib:$SPATIAL_ENV/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
fi

usage() {
    echo "用法:"
    echo "  bash run_r2r/vlm_report_range.sh [start_episode_id] [end_episode_id] [results_dir]"
    echo ""
    echo "示例:"
    echo "  bash run_r2r/vlm_report_range.sh 1500 1799"
    echo "  bash run_r2r/vlm_report_range.sh 1500 1799 ../data/result/spacevln"
    echo ""
    echo "说明:"
    echo "  只读取已有 log 生成部分汇总，不会重新跑 episode。"
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

cd "$PROJECT_ROOT"

RESULTS_DIR=${RESULTS_DIR_ARG:-../data/result/spacevln}

echo "📊 生成部分结果报告"
echo "   Range: $START_ID-$END_ID"
echo "   Source: $RESULTS_DIR"
echo "   Output: $RESULTS_DIR/reports/$START_ID-$END_ID"

"$PYTHON_BIN" -m vlnce_baselines.runtime.results_report \
    --path "$RESULTS_DIR" \
    --save \
    --start-id "$START_ID" \
    --end-id "$END_ID"
