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
Usage:
  bash run_navigation/vlnce.sh [--runtime standard|context_cache] [--ablation PRESET_OR_YAML] [episode_args...]

Runtime:
  --runtime standard        Standard runtime (default)
  --runtime context_cache   Explicit context-cache runtime

Results:
    --results-root DIR        Compatibility override for the results root
    --results-dir DIR         Compatibility override for the final results directory

Path policy:
    Use the unified workspace defaults whenever possible:
        Results: nav_ws/result
        Data:    nav_ws/data
    To place large storage on another disk, create symlinks at the default paths:
        bash run_navigation/setup_storage_symlinks.sh --disk-root /abs/path/to/nav_ws_storage --both --backup-existing

Console output:
    The launcher prints only key progress and failures by default.
    Detailed context-cache statistics are saved under reports/cache instead of being spammed to stdout.

Ablation:
  --ablation landmark
  --ablation space_structure
  --ablation landmark_space_structure
  --ablation planning_reasoning
  --ablation action_reasoning
  --ablation planning_action_reasoning
  --ablation planning_reasoning_no_progress
  --ablation planning_action_reasoning_no_progress
  --ablation /abs/path/to/config.yaml

Episode-arg examples:
  832
  832 300
  1 100 260 4
  random 20 260 all 4

Examples:
  bash run_navigation/vlnce.sh 1 10 260 4
  bash run_navigation/vlnce.sh --runtime context_cache 1 10 260 4
  bash run_navigation/vlnce.sh --ablation landmark 1 100 260 4
  bash run_navigation/vlnce.sh --ablation thinking_reasoning 1 100 260 4
  bash run_navigation/vlnce.sh --ablation planning_reasoning_no_progress 1 100 260 4
  bash run_navigation/vlnce.sh --runtime context_cache --ablation space_structure 1 100 260 4
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
                echo "❌ --runtime requires standard or context_cache" >&2
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
                echo "❌ $1 requires a preset name or YAML path" >&2
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
                echo "❌ --results-root requires a directory" >&2
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
                echo "❌ --results-dir requires a directory" >&2
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
        echo "❌ Unsupported runtime: ${RUNTIME_MODE}" >&2
        echo "   Supported values: standard | context_cache" >&2
        exit 1
        ;;
esac

if [[ "${RUNTIME_MODE}" == "context_cache" ]]; then
    API_CONFIG="${VLM_API_CONFIG:-$(spacevln_default_context_cache_api_config)}"
    API_MISSING_MESSAGE="Context-cache API config does not exist"
    API_MISSING_HINT="Copy and fill navigation_system/config/vlm/vlm_api_config.yaml.template"
else
    API_CONFIG="${VLM_API_CONFIG:-navigation_system/config/vlm/vlm_api_config.yaml}"
    API_MISSING_MESSAGE="API config does not exist"
    API_MISSING_HINT="Copy and fill navigation_system/config/vlm/vlm_api_config.yaml.template"
fi

EXTRA_ARGS=(--runtime "$RUNTIME_MODE" "${RESULT_PATH_ARGS[@]}")

if [[ ${#RESULT_PATH_ARGS[@]} -gt 0 ]]; then
    echo "⚠️  Detected --results-root/--results-dir overrides."
    echo "   The unified nav_ws/result layout is recommended; use the symlink helper if you need another disk."
fi

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
