#!/bin/bash
# 批量评测快捷脚本
#
# 用法:
#   bash run_r2r/vlm_batch_eval.sh [start_id] [num_episodes] [max_steps] [mode]
#
# mode:
#   all         - 全部运行
#   skip-sr1    - 跳过结果目录中已有最佳结果且 SR=1 的 episode
#
# 示例:
#   bash run_r2r/vlm_batch_eval.sh 1342 300 200 all
#   bash run_r2r/vlm_batch_eval.sh 1342 300 200 skip-sr1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

START_ID="${1:-1342}"
NUM_EPISODES="${2:-300}"
MAX_STEPS="${3:-200}"
MODE="${4:-all}"

EXTRA_ARGS=()
case "$MODE" in
    all)
        ;;
    skip-sr1|skip_sr1|resume)
        EXTRA_ARGS+=(--skip-existing-sr1)
        ;;
    *)
        echo "❌ 不支持的模式: $MODE"
        echo "   可选模式: all | skip-sr1"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════"
echo "批量评测模式"
echo "  Start ID:     $START_ID"
echo "  Num Episodes: $NUM_EPISODES"
echo "  Max Steps:    $MAX_STEPS"
echo "  Mode:         $MODE"
echo "════════════════════════════════════════════════════════════"
echo ""

bash "$SCRIPT_DIR/vlm_navigation.sh" "$START_ID" "$NUM_EPISODES" "$MAX_STEPS" "${EXTRA_ARGS[@]}"
