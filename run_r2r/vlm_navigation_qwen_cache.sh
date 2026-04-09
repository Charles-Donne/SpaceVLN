#!/bin/bash
# Qwen explicit-context-cache navigation entrypoint.
# Supports both:
# 1. old positional shorthand
# 2. standard flag-based CLI

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

PROJECT_ROOT="$(spacevln_project_root)"
PYTHON_BIN="$(spacevln_select_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"

CONFIG_FILE="navigation_system/config/experiments/r2r_eval.yaml"
API_CONFIG="${VLM_API_CONFIG:-navigation_system/config/api/vlm_api_config_qwen_cache.yaml}"

cd "$PROJECT_ROOT"

if [ ! -f "$API_CONFIG" ]; then
    echo "❌ 缓存版 API 配置不存在: $API_CONFIG"
    exit 1
fi

run_python() {
    exec "$PYTHON_BIN" vlm_navigation_qwen_cache.py \
        --exp-config "$CONFIG_FILE" \
        --vlm-api-config "$API_CONFIG" \
        "$@"
}

# If any argument already uses flag style, pass through directly.
for arg in "$@"; do
    if [[ "$arg" == --* ]]; then
        run_python "$@"
    fi
done

FIRST_ARG="${1:-}"
SECOND_ARG="${2:-}"
THIRD_ARG="${3:-}"
FOURTH_ARG="${4:-}"
FIFTH_ARG="${5:-}"

MODE="all"
PARALLEL_WORKERS="1"
MODE_ARG=""

if [[ -n "$FOURTH_ARG" && "$FOURTH_ARG" =~ ^[0-9]+$ ]]; then
    PARALLEL_WORKERS="$FOURTH_ARG"
elif [[ -n "$FOURTH_ARG" ]]; then
    MODE="$FOURTH_ARG"
fi

if [[ -n "$FIFTH_ARG" ]]; then
    if [[ "$FIFTH_ARG" =~ ^[0-9]+$ ]]; then
        PARALLEL_WORKERS="$FIFTH_ARG"
    else
        echo "❌ parallel_workers 必须是正整数: $FIFTH_ARG"
        exit 1
    fi
fi

spacevln_validate_parallel_workers "$PARALLEL_WORKERS"
MODE_ARG="$(spacevln_mode_arg "$MODE")"

if [[ -z "$FIRST_ARG" ]]; then
    run_python
fi

if [[ "$FIRST_ARG" == "random" ]]; then
    NUM_EPISODES="${SECOND_ARG:-1}"
    MAX_STEPS="${THIRD_ARG:-}"
    ARGS=(
        --random
        --num-episodes "$NUM_EPISODES"
        --parallel-workers "$PARALLEL_WORKERS"
        --max-subtask-steps 5
    )
    if [[ -n "$MAX_STEPS" ]]; then
        ARGS+=(--max-steps "$MAX_STEPS")
    fi
    if [[ -n "$MODE_ARG" ]]; then
        ARGS+=("$MODE_ARG")
    fi
    run_python "${ARGS[@]}"
fi

if [[ "$FIRST_ARG" == "list" ]]; then
    EPISODE_IDS="$SECOND_ARG"
    MAX_STEPS="${THIRD_ARG:-}"
    if [[ -z "$EPISODE_IDS" ]]; then
        echo "❌ list 模式需要 episode id 列表"
        exit 1
    fi
    ARGS=(
        --episode-ids "$EPISODE_IDS"
        --parallel-workers "$PARALLEL_WORKERS"
        --max-subtask-steps 5
    )
    if [[ -n "$MAX_STEPS" ]]; then
        ARGS+=(--max-steps "$MAX_STEPS")
    fi
    if [[ -n "$MODE_ARG" ]]; then
        ARGS+=("$MODE_ARG")
    fi
    run_python "${ARGS[@]}"
fi

if ! [[ "$FIRST_ARG" =~ ^[0-9]+$ ]]; then
    echo "❌ 不支持的第一个参数: $FIRST_ARG"
    echo "   可用写法:"
    echo "   bash run_r2r/vlm_navigation_qwen_cache.sh 832"
    echo "   bash run_r2r/vlm_navigation_qwen_cache.sh 832 300"
    echo "   bash run_r2r/vlm_navigation_qwen_cache.sh 1 600 260 5"
    echo "   bash run_r2r/vlm_navigation_qwen_cache.sh random 20 260 all 4"
    echo "   bash run_r2r/vlm_navigation_qwen_cache.sh --episode-id 832 --num-episodes 1"
    exit 1
fi

EPISODE_ID="$FIRST_ARG"

# 2 个位置参数时，默认按“单 episode + max_steps”解释，更符合旧脚本注释示例。
if [[ -n "$SECOND_ARG" && -z "$THIRD_ARG" ]]; then
    run_python \
        --episode-id "$EPISODE_ID" \
        --num-episodes 1 \
        --max-steps "$SECOND_ARG" \
        --parallel-workers "$PARALLEL_WORKERS" \
        --max-subtask-steps 5 \
        ${MODE_ARG:+"$MODE_ARG"}
fi

NUM_EPISODES="${SECOND_ARG:-1}"
MAX_STEPS="${THIRD_ARG:-}"
ARGS=(
    --episode-id "$EPISODE_ID"
    --num-episodes "$NUM_EPISODES"
    --parallel-workers "$PARALLEL_WORKERS"
    --max-subtask-steps 5
)
if [[ -n "$MAX_STEPS" ]]; then
    ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$MODE_ARG" ]]; then
    ARGS+=("$MODE_ARG")
fi
run_python "${ARGS[@]}"
