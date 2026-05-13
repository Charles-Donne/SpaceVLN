#!/bin/bash
# R2R-CE launcher for the shared Navigation Agent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/ablation_common.sh"

spacevln_navigation_print_usage() {
    cat <<'EOF'
Usage:
  bash run_navigation/r2rce.sh [--runtime standard|context_cache] [--prompt-profile fast|original] [--python-bin PATH|--compat-env ENV] [--ablation PRESET_OR_YAML] [episode_args...]

Runtime:
  --runtime standard        Standard runtime (default)
  --runtime context_cache   Explicit context-cache runtime

Python environment:
  --python-bin PATH         Temporarily run R2R-CE with this Python executable
  --compat-env ENV          Temporarily run with a named conda env, e.g. spacevln
                             Default remains the normal SpaceVLN Python selection.
                             PYTHON_BIN or SPACEVLN_R2R_PYTHON can also override it.

Prompt profile:
  --prompt-profile original  Use the original full prompt templates
  --prompt-profile fast      Use the compressed fast prompt templates
                             Default: original unless SPACEVLN_PROMPT_PROFILE=fast

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
    --no-report skips post-run report generation when you only need raw episode logs.

Output speed switches:
    --output-profile metric|debug|config
                             metric keeps only final evaluation artifacts;
                             debug saves VLM artifacts, key local/global map snapshots, and navigation.gif;
                             config preserves YAML defaults.
    --episode-workdir DIR    Write current episode artifacts to fast local cache first,
                             then sync them back to the final results directory
                             in a background transfer thread. If final results
                             resolve under /media or /mnt, this is enabled automatically
                             with /dev/shm/spacevln_episode_cache when available,
                             falling back to nav_ws/.spacevln_episode_cache, and cleaned after sync.
                             Transfer pool defaults per worker: pool=3, batch=2.
                             Tune with SPACEVLN_EPISODE_TRANSFER_POOL/BATCH.
    --no-gif                 Skip final navigation.gif generation for metric-only runs
    --no-vlm-artifacts       Skip prompt/image/debug artifact files for maximum speed
    --save-local-map         Save key local/global map snapshots for thinking/detection debugging
    --save-step-images       Save per-step replay PNGs only when you need detailed visual debugging

Retries:
    --initial-failure-max-attempts N
        Rerun an episode up to N times when initial planning cannot produce a usable subtask.
        Default: 3. Initial planner API retries default to 5 and can be overridden with
        SPACEVLN_INITIAL_PLANNER_MAX_RETRIES.
    --max-initial-planner-api-errors N
        Abort a batch after too many initial planner API failures.
        Default: max(10, 2 * parallel-workers); use 0 to disable.

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
  bash run_navigation/r2rce.sh 1 10 260 4
  bash run_navigation/r2rce.sh --runtime context_cache 1 10 260 4
  bash run_navigation/r2rce.sh --runtime context_cache --prompt-profile fast 1 10 260 4
  bash run_navigation/r2rce.sh --runtime context_cache --prompt-profile original 1 10 260 4
  bash run_navigation/r2rce.sh --ablation landmark 1 100 260 4
  bash run_navigation/r2rce.sh --ablation thinking_reasoning 1 100 260 4
  bash run_navigation/r2rce.sh --ablation planning_reasoning_no_progress 1 100 260 4
  bash run_navigation/r2rce.sh --runtime context_cache --ablation space_structure 1 100 260 4
EOF
}

R2R_PYTHON_BIN_OVERRIDE="${SPACEVLN_R2R_PYTHON:-}"
R2R_COMPAT_ENV=""

_spacevln_scan_args=("$@")
_spacevln_i=0
while [[ $_spacevln_i -lt ${#_spacevln_scan_args[@]} ]]; do
    _spacevln_arg="${_spacevln_scan_args[$_spacevln_i]}"
    case "$_spacevln_arg" in
        --python-bin)
            _spacevln_i=$((_spacevln_i + 1))
            if [[ $_spacevln_i -lt ${#_spacevln_scan_args[@]} ]]; then
                R2R_PYTHON_BIN_OVERRIDE="${_spacevln_scan_args[$_spacevln_i]}"
            fi
            ;;
        --python-bin=*)
            R2R_PYTHON_BIN_OVERRIDE="${_spacevln_arg#*=}"
            ;;
        --compat-env|--conda-env)
            _spacevln_i=$((_spacevln_i + 1))
            if [[ $_spacevln_i -lt ${#_spacevln_scan_args[@]} ]]; then
                R2R_COMPAT_ENV="${_spacevln_scan_args[$_spacevln_i]}"
            fi
            ;;
        --compat-env=*|--conda-env=*)
            R2R_COMPAT_ENV="${_spacevln_arg#*=}"
            ;;
    esac
    _spacevln_i=$((_spacevln_i + 1))
done
unset _spacevln_scan_args _spacevln_i _spacevln_arg

if [[ -n "$R2R_PYTHON_BIN_OVERRIDE" ]]; then
    export PYTHON_BIN="$R2R_PYTHON_BIN_OVERRIDE"
elif [[ -n "$R2R_COMPAT_ENV" ]]; then
    if R2R_PYTHON_BIN_OVERRIDE="$(spacevln_find_conda_env_python "$R2R_COMPAT_ENV" 2>/dev/null)"; then
        export PYTHON_BIN="$R2R_PYTHON_BIN_OVERRIDE"
    else
        echo "❌ Cannot find conda env for --compat-env ${R2R_COMPAT_ENV}" >&2
        exit 1
    fi
fi

PROJECT_ROOT="$(spacevln_project_root)"
PYTHON_BIN="$(spacevln_select_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"

CONFIG_FILE="${EXP_CONFIG:-navigation_system/config/experiments/vlnce/r2r_eval.yaml}"
RUNTIME_MODE="standard"
PROMPT_PROFILE=""
ABLATION_RAW=""
RESULT_PATH_ARGS=()
PASSTHROUGH_ARGS=()
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
        --prompt-profile)
            if [[ $# -lt 2 ]]; then
                echo "❌ --prompt-profile requires fast or original" >&2
                exit 1
            fi
            PROMPT_PROFILE="$2"
            shift 2
            ;;
        --prompt-profile=*)
            PROMPT_PROFILE="${1#*=}"
            shift
            ;;
        --python-bin)
            if [[ $# -lt 2 ]]; then
                echo "❌ --python-bin requires a Python executable path" >&2
                exit 1
            fi
            shift 2
            ;;
        --python-bin=*)
            shift
            ;;
        --compat-env|--conda-env)
            if [[ $# -lt 2 ]]; then
                echo "❌ $1 requires a conda environment name" >&2
                exit 1
            fi
            shift 2
            ;;
        --compat-env=*|--conda-env=*)
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
        --initial-failure-max-attempts)
            if [[ $# -lt 2 ]]; then
                echo "❌ --initial-failure-max-attempts requires a positive integer" >&2
                exit 1
            fi
            PASSTHROUGH_ARGS+=(--initial-failure-max-attempts "$2")
            shift 2
            ;;
        --initial-failure-max-attempts=*)
            PASSTHROUGH_ARGS+=(--initial-failure-max-attempts "${1#*=}")
            shift
            ;;
        --max-initial-planner-api-errors)
            if [[ $# -lt 2 ]]; then
                echo "❌ --max-initial-planner-api-errors requires an integer" >&2
                exit 1
            fi
            PASSTHROUGH_ARGS+=(--max-initial-planner-api-errors "$2")
            shift 2
            ;;
        --max-initial-planner-api-errors=*)
            PASSTHROUGH_ARGS+=(--max-initial-planner-api-errors "${1#*=}")
            shift
            ;;
        --episode-workdir)
            if [[ $# -lt 2 ]]; then
                echo "❌ --episode-workdir requires a directory" >&2
                exit 1
            fi
            PASSTHROUGH_ARGS+=(--episode-workdir "$2")
            shift 2
            ;;
        --episode-workdir=*)
            PASSTHROUGH_ARGS+=(--episode-workdir "${1#*=}")
            shift
            ;;
        --output-profile)
            if [[ $# -lt 2 ]]; then
                echo "❌ --output-profile requires metric, debug, or config" >&2
                exit 1
            fi
            PASSTHROUGH_ARGS+=(--output-profile "$2")
            shift 2
            ;;
        --output-profile=*)
            PASSTHROUGH_ARGS+=(--output-profile "${1#*=}")
            shift
            ;;
        --no-gif|--no-save-gif|--save-gif|--no-vlm-artifacts|--save-vlm-artifacts|--save-map-artifacts|--no-save-map-artifacts|--save-local-map|--no-local-map|--save-step-images|--no-save-step-images|--no-report)
            PASSTHROUGH_ARGS+=("$1")
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

if [[ -n "${PROMPT_PROFILE}" ]]; then
    case "${PROMPT_PROFILE}" in
        fast|compressed|compact)
            export SPACEVLN_PROMPT_PROFILE="fast"
            ;;
        original|default|full|standard)
            export SPACEVLN_PROMPT_PROFILE="original"
            ;;
        *)
            echo "❌ Unsupported prompt profile: ${PROMPT_PROFILE}" >&2
            echo "   Supported values: fast | original" >&2
            exit 1
            ;;
    esac
fi

if [[ "${RUNTIME_MODE}" == "context_cache" ]]; then
    API_CONFIG="${VLM_API_CONFIG:-$(spacevln_default_context_cache_api_config)}"
    API_MISSING_MESSAGE="Context-cache API config does not exist"
    API_MISSING_HINT="Copy and fill navigation_system/config/vlm/vlm_api_config.yaml.template"
else
    API_CONFIG="${VLM_API_CONFIG:-navigation_system/config/vlm/vlm_api_config.yaml}"
    API_MISSING_MESSAGE="API config does not exist"
    API_MISSING_HINT="Copy and fill navigation_system/config/vlm/vlm_api_config.yaml.template"
fi

EXTRA_ARGS=(--runtime "$RUNTIME_MODE" "${RESULT_PATH_ARGS[@]}" "${PASSTHROUGH_ARGS[@]}")

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

spacevln_dispatch_r2r_cli \
    "$PROJECT_ROOT" \
    "$PYTHON_BIN" \
    "$CONFIG_FILE" \
    "$API_CONFIG" \
    "$API_MISSING_MESSAGE" \
    "$API_MISSING_HINT" \
    EXTRA_ARGS \
    "${FORWARD_ARGS[@]}"
