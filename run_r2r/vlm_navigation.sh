#!/bin/bash
# Unified SpaceVLN navigation entrypoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/ablation_common.sh"

spacevln_navigation_print_usage() {
    cat <<'EOF'
用法:
  bash run_r2r/vlm_navigation.sh [--runtime standard|context_cache] [--ablation PRESET_OR_YAML] [episode_args...]

模式:
  --runtime standard        标准运行（默认）
  --runtime context_cache   显式 context cache 运行

保存:
  --results-root DIR        只覆盖总根目录，仍自动保存到 vlnce/模型名 或 vlnce/ablation/消融项/模型名
  --results-dir DIR         高级：直接覆盖最终目录，不再自动追加结构化子目录

消融:
  --ablation landmark
  --ablation space_structure
  --ablation planning_reasoning
  --ablation action_reasoning
  --ablation planning_action_reasoning
  --ablation both
  --ablation /abs/path/to/config.yaml

episode_args 与原来保持一致，例如:
  832
  832 300
  1 100 260 4
  random 20 260 all 4

示例:
  bash run_r2r/vlm_navigation.sh 1 10 260 4
  bash run_r2r/vlm_navigation.sh --runtime context_cache 1 10 260 4
  bash run_r2r/vlm_navigation.sh --ablation landmark 1 100 260 4
  bash run_r2r/vlm_navigation.sh --runtime context_cache --ablation space_structure 1 100 260 4
EOF
}

PROJECT_ROOT="$(spacevln_project_root)"
PYTHON_BIN="$(spacevln_select_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"

CONFIG_FILE="${EXP_CONFIG:-navigation_system/config/experiments/r2r_eval.yaml}"
RUNTIME_MODE="standard"
ABLATION_RAW=""
RESULT_PATH_ARGS=()
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            spacevln_navigation_print_usage
            exit 0
            ;;
        --runtime)
            if [[ $# -lt 2 ]]; then
                echo "❌ --runtime 后面需要 standard 或 context_cache" >&2
                exit 1
            fi
            RUNTIME_MODE="$2"
            shift 2
            ;;
        --runtime=*)
            RUNTIME_MODE="${1#*=}"
            shift
            ;;
        --ablation|--preset|--ablation-preset|--ablation-config)
            if [[ $# -lt 2 ]]; then
                echo "❌ $1 后面需要 preset 名称或 yaml 路径" >&2
                exit 1
            fi
            ABLATION_RAW="$2"
            shift 2
            ;;
        --ablation=*|--preset=*|--ablation-preset=*|--ablation-config=*)
            ABLATION_RAW="${1#*=}"
            shift
            ;;
        --results-root)
            if [[ $# -lt 2 ]]; then
                echo "❌ --results-root 后面需要目录" >&2
                exit 1
            fi
            RESULT_PATH_ARGS+=(--results-root "$2")
            shift 2
            ;;
        --results-root=*)
            RESULT_PATH_ARGS+=(--results-root "${1#*=}")
            shift
            ;;
        --results-dir)
            if [[ $# -lt 2 ]]; then
                echo "❌ --results-dir 后面需要目录" >&2
                exit 1
            fi
            RESULT_PATH_ARGS+=(--results-dir "$2")
            shift 2
            ;;
        --results-dir=*)
            RESULT_PATH_ARGS+=(--results-dir "${1#*=}")
            shift
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

case "${RUNTIME_MODE}" in
    standard|context_cache)
        ;;
    *)
        echo "❌ 不支持的 runtime: ${RUNTIME_MODE}" >&2
        echo "   可选值: standard | context_cache" >&2
        exit 1
        ;;
esac

if [[ "${RUNTIME_MODE}" == "context_cache" ]]; then
    API_CONFIG="${VLM_API_CONFIG:-$(spacevln_default_context_cache_api_config)}"
    API_MISSING_MESSAGE="context-cache API 配置不存在"
    API_MISSING_HINT="请从 navigation_system/config/vlm/vlm_api_config_context_cache.yaml.template 复制并填写"
else
    API_CONFIG="${VLM_API_CONFIG:-navigation_system/config/vlm/vlm_api_config.yaml}"
    API_MISSING_MESSAGE="API 配置不存在"
    API_MISSING_HINT="请从 navigation_system/config/vlm/vlm_api_config.yaml.template 复制并配置"
fi

EXTRA_ARGS=(--runtime "$RUNTIME_MODE" "${RESULT_PATH_ARGS[@]}")

if [[ -n "$ABLATION_RAW" ]]; then
    if preset_path="$(spacevln_ablation_resolve_preset_path "$ABLATION_RAW" 2>/dev/null)"; then
        ABLATION_CONFIG_PATH="$(spacevln_ablation_resolve_config_path "$preset_path" "$PROJECT_ROOT")" || exit 1
    else
        ABLATION_CONFIG_PATH="$(spacevln_ablation_resolve_config_path "$ABLATION_RAW" "$PROJECT_ROOT")" || exit 1
    fi
    EXTRA_ARGS+=(--ablation-config "$ABLATION_CONFIG_PATH")
fi

spacevln_dispatch_navigation_cli \
    "$PROJECT_ROOT" \
    "$PYTHON_BIN" \
    "vlm_navigation.py" \
    "$CONFIG_FILE" \
    "$API_CONFIG" \
    "$API_MISSING_MESSAGE" \
    "$API_MISSING_HINT" \
    EXTRA_ARGS \
    "${FORWARD_ARGS[@]}"
