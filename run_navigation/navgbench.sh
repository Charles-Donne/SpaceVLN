#!/bin/bash
# Run the Navigation Agent on NavGBench/GN-Bench episodes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash run_navigation/navgbench.sh [--runtime standard|context_cache] [episode args...]

Episode-arg examples:
  1                         Run sample 1
  1 10                      Run 10 samples from sample 1
  1 10 300                  Same, with max_steps=300
  1 10 300 2                Same, with 2 parallel episode workers
  1 10 300 skip-sr1 2       Same, skip existing SR=1 best logs, 2 workers
  random 5 300              Random 5 episodes, max_steps=300
  random 5 300 2            Random 5 episodes, max_steps=300, 2 workers
  list 0864_841787_156      Run explicit NavGBench stable id

Long-form examples:
  bash run_navigation/navgbench.sh --dry-run --start-sample 1 --num-episodes 3
  bash run_navigation/navgbench.sh --episode-id 1 --max-steps 300
  bash run_navigation/navgbench.sh --episode-ids 0864_841787_156,0278_840770_12
  bash run_navigation/navgbench.sh --simple-instruction 1 10 300
  bash run_navigation/navgbench.sh --complex-instruction 1 10 300

Defaults:
  Agent config:     navigation_system/config/experiments/vlnce/navgbench_eval.yaml
  NavGBench root:  ../Nav-GBench, or NAVGBENCH_ROOT if set
  NavGBench env:   conda env named gn_bench, or SPACEVLN_NAVGBENCH_PYTHON/PYTHON_BIN if set
  Backend:         auto (in-process when GN_Bench is importable, subprocess fallback)
  Runtime:         context_cache
  Instruction:     complex (landmark-rich route)
  Results:         nav_ws/result/navgbench/<complex|simple|moving>/<llm>__<vlm>
  Console output:  episode internals are quiet by default; summaries/reports stay on stdout
EOF
}

spacevln_select_navgbench_agent_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        printf '%s\n' "$PYTHON_BIN"
        return
    fi
    if [[ -n "${SPACEVLN_NAVGBENCH_PYTHON:-}" ]]; then
        printf '%s\n' "$SPACEVLN_NAVGBENCH_PYTHON"
        return
    fi
    if candidate="$(spacevln_find_conda_env_python gn_bench 2>/dev/null)"; then
        printf '%s\n' "$candidate"
        return
    fi
    spacevln_select_python
}

PROJECT_ROOT="$(spacevln_project_root)"
PYTHON_BIN="$(spacevln_select_navgbench_agent_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"

spacevln_prepend_colon_var PYTHONPATH "$PROJECT_ROOT"
if [ -d "$PROJECT_ROOT/../GroundingDINO" ]; then
    spacevln_prepend_colon_var PYTHONPATH "$PROJECT_ROOT/../GroundingDINO"
fi

RUNTIME_MODE="${SPACEVLN_NAVGBENCH_RUNTIME:-context_cache}"
API_CONFIG="${VLM_API_CONFIG:-navigation_system/config/vlm/vlm_api_config.yaml}"
SPACEVLN_CONFIG="${SPACEVLN_NAVGBENCH_CONFIG:-navigation_system/config/experiments/vlnce/navgbench_eval.yaml}"
PASSTHROUGH_ARGS=()
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            usage
            cd "$PROJECT_ROOT"
            exec "$PYTHON_BIN" navigation_agent.py navgbench --help
            ;;
        --runtime)
            RUNTIME_MODE="$2"
            shift 2
            ;;
        --runtime=*)
            RUNTIME_MODE="${1#*=}"
            shift
            ;;
        --vlm-api-config|--config)
            API_CONFIG="$2"
            shift 2
            ;;
        --vlm-api-config=*|--config=*)
            API_CONFIG="${1#*=}"
            shift
            ;;
        --spacevln-config)
            SPACEVLN_CONFIG="$2"
            shift 2
            ;;
        --spacevln-config=*)
            SPACEVLN_CONFIG="${1#*=}"
            shift
            ;;
        --dry-run|--no-report|--random|--use-raw-instruction|--complex-instruction|--simple-instruction|--skip-sr1|--skip-existing-sr1)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
        --backend|--navgbench-python|--gnbench-root|--gnbench-exp-config|--start-sample|--start-idx|--end-idx|--num-episodes|--episode-id|--episode-ids|--seed|--max-steps|--max-subtask-steps|--parallel-workers|--instruction-mode|--results-root|--results-dir)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 1
            fi
            PASSTHROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
        --backend=*|--navgbench-python=*|--gnbench-root=*|--gnbench-exp-config=*|--start-sample=*|--start-idx=*|--end-idx=*|--num-episodes=*|--episode-id=*|--episode-ids=*|--seed=*|--max-steps=*|--max-subtask-steps=*|--parallel-workers=*|--instruction-mode=*|--results-root=*|--results-dir=*)
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
        echo "Unsupported NavGBench positional arguments: ${POSITIONAL_ARGS[*]}" >&2
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
        TRANSLATED_ARGS=(--random --num-episodes "${second_arg:-1}" --parallel-workers "$parallel_workers")
        if [[ -n "$third_arg" ]]; then
            TRANSLATED_ARGS+=(--max-steps "$third_arg")
        fi
    elif [[ "$first_arg" == "list" ]]; then
        if [[ -z "$second_arg" ]]; then
            echo "list mode requires episode ids" >&2
            exit 1
        fi
        TRANSLATED_ARGS=(--episode-ids "$second_arg" --parallel-workers "$parallel_workers")
        if [[ -n "$third_arg" ]]; then
            TRANSLATED_ARGS+=(--max-steps "$third_arg")
        fi
    elif [[ "$first_arg" =~ ^[0-9]+$ ]]; then
        TRANSLATED_ARGS=(--start-sample "$first_arg" --num-episodes "${second_arg:-1}" --parallel-workers "$parallel_workers")
        if [[ -n "$third_arg" ]]; then
            TRANSLATED_ARGS+=(--max-steps "$third_arg")
        fi
    else
        TRANSLATED_ARGS=(--episode-id "$first_arg" --parallel-workers "$parallel_workers")
        if [[ -n "$second_arg" ]]; then
            TRANSLATED_ARGS+=(--max-steps "$second_arg")
        fi
    fi
    if [[ -n "$mode_arg" ]]; then
        TRANSLATED_ARGS+=("$mode_arg")
    fi
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" navigation_agent.py navgbench \
    --spacevln-config "$SPACEVLN_CONFIG" \
    --vlm-api-config "$API_CONFIG" \
    --runtime "$RUNTIME_MODE" \
    "${TRANSLATED_ARGS[@]}" \
    "${PASSTHROUGH_ARGS[@]}"
