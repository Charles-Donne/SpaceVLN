#!/bin/bash
# Canonical OVON object-navigation launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

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
  bash run_navigation/object_navigation.sh [--runtime standard|context_cache] [--run-config YAML] [episode args...]

Episode-arg examples:
  1                            Run the first sample in the current split
  1 60                         Run one sample with max_steps=60
  1 10 500                     Start from sample 1, run 10 tasks in split/sample order
  1 10 500 skip-sr1 5          Same range, skip episodes with existing SR=1 logs, use 5 workers
  random 10 500 all 1          Sample 10 random episodes, max_steps=500
  list 1074,1081 500           Run explicit episode ids, max_steps=500

Examples:
  bash run_navigation/object_navigation.sh --episode-ids 1074
  bash run_navigation/object_navigation.sh --episode-id 1074
  bash run_navigation/object_navigation.sh --start-index 1 --num-episodes 10
  bash run_navigation/object_navigation.sh --runtime context_cache --num-episodes 10
  bash run_navigation/object_navigation.sh --runtime context_cache 1 100 300 5
  bash run_navigation/object_navigation.sh --run-config navigation_system/config/experiments/object_navigation/ovon_val_unseen_eval.yaml --episode-ids 1074,1081

Notes:
  - Default results root follows the unified workspace layout: nav_ws/result/ovon
  - Default data root follows the unified workspace layout: nav_ws/data
  - The OVON runtime defaults YAML is:
      navigation_system/config/experiments/object_navigation/ovon_val_unseen_eval.yaml
  - Positional numeric launch follows split/sample order semantics.
  - Use --start-index explicitly if you want the same behavior in long-form flags.
  - Use --episode-id for an exact OVON episode id.
  - The workers argument matches the VLNCE launcher style and now enables parallel episode workers.
  - Output profiles are shared with VLNCE:
      --output-profile metric   keep final metrics/results only (default)
      --output-profile debug    save VLM artifacts and navigation.gif
      --output-profile config   preserve YAML/default save switches
EOF
}

PROJECT_ROOT="$(spacevln_project_root)"
PYTHON_BIN="$(spacevln_select_objectnav_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"

RUNTIME_MODE="context_cache"
RUN_CONFIG="navigation_system/config/experiments/object_navigation/ovon_val_unseen_eval.yaml"
PASSTHROUGH_ARGS=()
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            usage
            cd "$PROJECT_ROOT"
            exec "$PYTHON_BIN" navigation_agent.py ovon --help
            ;;
        --runtime)
            RUNTIME_MODE="$2"
            shift 2
            ;;
        --runtime=*)
            RUNTIME_MODE="${1#*=}"
            shift
            ;;
        --run-config)
            RUN_CONFIG="$2"
            shift 2
            ;;
        --run-config=*)
            RUN_CONFIG="${1#*=}"
            shift
            ;;
        --exp-config|--vlm-api-config|--data-path|--split|--episode-id|--episode-ids|--num-episodes|--gpu-id|--max-steps|--max-subtask-steps|--results-root|--results-dir|--seed|--parallel-workers|--output-profile)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 1
            fi
            PASSTHROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
        --exp-config=*|--vlm-api-config=*|--data-path=*|--split=*|--episode-id=*|--episode-ids=*|--num-episodes=*|--gpu-id=*|--max-steps=*|--max-subtask-steps=*|--results-root=*|--results-dir=*|--seed=*|--parallel-workers=*|--output-profile=*)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
        --random|--skip-sr1|--skip-existing-sr1|--save-step-images|--no-save-step-images|--save-gif|--no-gif|--no-save-gif|--save-vlm-artifacts|--no-vlm-artifacts|--no-report)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
        --*)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

case "${RUNTIME_MODE}" in
    standard|context_cache)
        ;;
    *)
        echo "Unsupported runtime: ${RUNTIME_MODE}" >&2
        echo "Supported values: standard | context_cache" >&2
        exit 1
        ;;
esac

TRANSLATED_ARGS=()

if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    first_arg="${POSITIONAL_ARGS[0]:-}"
    second_arg="${POSITIONAL_ARGS[1]:-}"
    third_arg="${POSITIONAL_ARGS[2]:-}"
    fourth_arg="${POSITIONAL_ARGS[3]:-}"
    fifth_arg="${POSITIONAL_ARGS[4]:-}"

    if [[ ${#POSITIONAL_ARGS[@]} -gt 5 ]]; then
        echo "Unsupported OVON positional arguments: ${POSITIONAL_ARGS[*]}" >&2
        usage >&2
        exit 1
    fi

    mode="all"
    parallel_workers="1"
    if [[ -n "$fourth_arg" && "$fourth_arg" =~ ^[0-9]+$ ]]; then
        parallel_workers="$fourth_arg"
    elif [[ -n "$fourth_arg" ]]; then
        mode="$fourth_arg"
    fi
    if [[ -n "$fifth_arg" ]]; then
        if [[ "$fifth_arg" =~ ^[0-9]+$ ]]; then
            parallel_workers="$fifth_arg"
        else
            echo "parallel_workers must be a positive integer: $fifth_arg" >&2
            exit 1
        fi
    fi

    spacevln_validate_parallel_workers "$parallel_workers"
    mode_arg="$(spacevln_mode_arg "$mode")"

    if [[ "$first_arg" == "random" ]]; then
        num_episodes="${second_arg:-1}"
        max_steps="${third_arg:-}"
        TRANSLATED_ARGS=(--random --num-episodes "$num_episodes" --parallel-workers "$parallel_workers")
        if [[ -n "$max_steps" ]]; then
            TRANSLATED_ARGS+=(--max-steps "$max_steps")
        fi
    elif [[ "$first_arg" == "list" ]]; then
        episode_ids="$second_arg"
        max_steps="${third_arg:-}"
        if [[ -z "$episode_ids" ]]; then
            echo "list mode requires an explicit episode-id list" >&2
            exit 1
        fi
        TRANSLATED_ARGS=(--episode-ids "$episode_ids" --parallel-workers "$parallel_workers")
        if [[ -n "$max_steps" ]]; then
            TRANSLATED_ARGS+=(--max-steps "$max_steps")
        fi
    elif [[ "$first_arg" =~ ^[0-9]+$ ]]; then
        if [[ -n "$second_arg" && -z "$third_arg" ]]; then
            TRANSLATED_ARGS=(
                --start-index "$first_arg"
                --num-episodes 1
                --max-steps "$second_arg"
                --parallel-workers "$parallel_workers"
            )
        else
            num_episodes="${second_arg:-1}"
            max_steps="${third_arg:-}"
            TRANSLATED_ARGS=(
                --start-index "$first_arg"
                --num-episodes "$num_episodes"
                --parallel-workers "$parallel_workers"
            )
            if [[ -n "$max_steps" ]]; then
                TRANSLATED_ARGS+=(--max-steps "$max_steps")
            fi
        fi
    else
        echo "Unsupported first OVON positional argument: $first_arg" >&2
        usage >&2
        exit 1
    fi

    if [[ -n "$mode_arg" ]]; then
        TRANSLATED_ARGS+=("$mode_arg")
    fi
fi

API_CONFIG="${VLM_API_CONFIG:-navigation_system/config/vlm/vlm_api_config.yaml}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" navigation_agent.py ovon \
    --run-config "$RUN_CONFIG" \
    --runtime "$RUNTIME_MODE" \
    --vlm-api-config "$API_CONFIG" \
    "${TRANSLATED_ARGS[@]}" \
    "${PASSTHROUGH_ARGS[@]}"
