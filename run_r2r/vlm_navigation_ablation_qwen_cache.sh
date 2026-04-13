#!/bin/bash
# SpaceVLN isolated ablation entrypoint for Qwen explicit-context-cache runtime.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/ablation_common.sh"

PROJECT_ROOT="$(spacevln_project_root)"
PYTHON_BIN="$(spacevln_select_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"

CONFIG_FILE="${EXP_CONFIG:-navigation_system/config/experiments/r2r_eval.yaml}"
API_CONFIG="${VLM_API_CONFIG:-navigation_system/config/vlm/vlm_api_config_qwen_cache.yaml}"

if spacevln_ablation_parse_cli "$PROJECT_ROOT" "$(basename "${BASH_SOURCE[0]}")" "$@"; then
    :
else
    PARSE_STATUS=$?
    if [ "$PARSE_STATUS" -eq 2 ]; then
        exit 0
    fi
    exit "$PARSE_STATUS"
fi

export SPACEVLN_ABLATION_CONFIG="$SPACEVLN_ABLATION_RESOLVED_CONFIG"
spacevln_ablation_print_selection "$SPACEVLN_ABLATION_CONFIG" "$SPACEVLN_ABLATION_SELECTED_PRESET"

spacevln_dispatch_navigation_cli \
    "$PROJECT_ROOT" \
    "$PYTHON_BIN" \
    "vlm_navigation_ablation_qwen_cache.py" \
    "$CONFIG_FILE" \
    "$API_CONFIG" \
    "Ablation 缓存版 API 配置不存在" \
    "请先准备缓存版 API 配置，再通过 ABLATION_CONFIG 指向消融 yaml" \
    "${SPACEVLN_ABLATION_FORWARD_ARGS[@]}"
