#!/bin/bash
# SpaceVLN standard navigation entrypoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

PROJECT_ROOT="$(spacevln_project_root)"
PYTHON_BIN="$(spacevln_select_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"

CONFIG_FILE="${EXP_CONFIG:-navigation_system/config/experiments/r2r_eval.yaml}"
API_CONFIG="${VLM_API_CONFIG:-navigation_system/config/vlm/vlm_api_config.yaml}"

spacevln_dispatch_navigation_cli \
    "$PROJECT_ROOT" \
    "$PYTHON_BIN" \
    "vlm_navigation.py" \
    "$CONFIG_FILE" \
    "$API_CONFIG" \
    "API 配置不存在" \
    "请从 navigation_system/config/vlm/vlm_api_config.yaml.template 复制并配置" \
    "$@"
