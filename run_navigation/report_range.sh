#!/bin/bash
# Generate a partial SpaceVLN report from existing log files only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

PROJECT_ROOT="$(spacevln_project_root)"
CONFIG_FILE="${EXP_CONFIG:-navigation_system/config/experiments/r2r_eval.yaml}"
PYTHON_BIN="$(spacevln_select_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"

START_ID_RAW=""
END_ID_RAW=""
RESULTS_SELECTOR=""
RESULTS_ROOT_OVERRIDE=""
POSITIONAL_ARGS=()

START_ID=""
END_ID=""

DATASET_MIN_EPISODE_ID=1
DATASET_MAX_EPISODE_ID=1800
DATASET_TOTAL_EPISODES=1800

RESULTS_ROOT=""
RESULTS_TARGET_DIRS=()
DEFAULT_REPORT_WORKERS="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
if ! [[ "$DEFAULT_REPORT_WORKERS" =~ ^[0-9]+$ ]] || [ "$DEFAULT_REPORT_WORKERS" -lt 1 ]; then
    DEFAULT_REPORT_WORKERS=8
fi
DEFAULT_REPORT_WORKERS=$((DEFAULT_REPORT_WORKERS * 4))
if [ "$DEFAULT_REPORT_WORKERS" -lt 8 ]; then
    DEFAULT_REPORT_WORKERS=8
fi
if [ "$DEFAULT_REPORT_WORKERS" -gt 64 ]; then
    DEFAULT_REPORT_WORKERS=64
fi

REPORT_LOAD_WORKERS="${SPACEVLN_REPORT_WORKERS:-$DEFAULT_REPORT_WORKERS}"
REPORT_SUMMARY_ONLY=0

usage() {
    echo "Usage:"
    echo "  bash run_navigation/report_range.sh [start_episode_id|all] [end_episode_id|all] [results_selector]"
    echo "  bash run_navigation/report_range.sh [start_episode_id|all] [results_selector]"
    echo "  bash run_navigation/report_range.sh [results_selector]"
    echo "  bash run_navigation/report_range.sh --start-id ID|all --end-id ID|all --results DIR|NAME|all"
    echo "  bash run_navigation/report_range.sh --fast --start-id all --end-id all --results DIR|NAME"
    echo ""
    echo "Examples:"
    echo "  bash run_navigation/report_range.sh 1500 1799"
    echo "  bash run_navigation/report_range.sh 1 50 qwen3.5-plus__qwen3.5-flash_cache"
    echo "  bash run_navigation/report_range.sh 1500 1799 /abs/path/to/result/vlnce/qwen3.5-plus__qwen3.5-flash"
    echo "  bash run_navigation/report_range.sh all all"
    echo "  bash run_navigation/report_range.sh --start-id 1600 --end-id all --results all"
    echo "  bash run_navigation/report_range.sh --results qwen3.5-plus__qwen3.5-flash,gemini2.5pro__gemini2.5flash"
    echo "  bash run_navigation/report_range.sh --fast all all ablation/no-space-structure/qwen3.5-plus__qwen3.5-flash_cache"
    echo ""
    echo "Notes:"
    echo "  Reads existing logs only and regenerates reports without rerunning episodes."
    echo "  The episode upper bound is inferred from the dataset in exp-config instead of being hard-coded to 1800."
    echo "  start/end support all; when both are omitted, the script summarizes all existing logs."
    echo "  If only start-id is provided, end-id expands to the dataset maximum."
    echo "  If only end-id is provided, start-id expands to the dataset minimum."
    echo "  --summary-only/--fast saves only summary + metrics.json and skips episode-level CSV/MD for speed."
    echo "  results_selector supports:"
    echo "    - absolute paths"
    echo "    - existing relative paths"
    echo "    - experiment names under the default result/vlnce root"
    echo "    - all (scan every experiment directory containing logs)"
    echo "    - comma-separated directory names or paths"
}

print_available_results_dirs() {
    if [ -z "$RESULTS_ROOT" ]; then
        RESULTS_ROOT="$(spacevln_default_results_root "$PYTHON_BIN")"
    fi
    echo "Available experiment directories:"
    if [ ! -d "$RESULTS_ROOT" ]; then
        echo "  (results root does not exist: $RESULTS_ROOT)"
        return
    fi

    local -a candidates=()
    local name=""
    while IFS= read -r name; do
        candidates+=("$name")
    done < <(find "$RESULTS_ROOT" -maxdepth 6 -type d -name log -printf '%h\n' | sort -u)

    local total="${#candidates[@]}"
    if [ "$total" -eq 0 ]; then
        echo "  (no reportable experiment directories found: missing log subdirectories)"
        return
    fi

    local max_show=30
    local idx=0
    for name in "${candidates[@]}"; do
        idx=$((idx + 1))
        if [ "$idx" -gt "$max_show" ]; then
            break
        fi
        echo "  - $name"
    done

    if [ "$total" -gt "$max_show" ]; then
        local omitted=$((total - max_show))
        echo "  ... ($omitted additional directories omitted)"
    fi
}

resolve_results_dir() {
    local raw_arg="$1"
    local default_dir="$2"

    if [ -z "$raw_arg" ]; then
        printf '%s\n' "$default_dir"
        return 0
    fi

    if [[ "$raw_arg" = /* ]]; then
        printf '%s\n' "$raw_arg"
        return 0
    fi

    if [ -d "$raw_arg" ]; then
        local abs_dir
        abs_dir="$(cd "$raw_arg" && pwd)"
        printf '%s\n' "$abs_dir"
        return 0
    fi

    if [ -z "$RESULTS_ROOT" ]; then
        RESULTS_ROOT="$(spacevln_default_results_root "$PYTHON_BIN")"
    fi
    if [ -d "$RESULTS_ROOT/$raw_arg" ]; then
        printf '%s\n' "$RESULTS_ROOT/$raw_arg"
        return 0
    fi

    printf '%s\n' "$raw_arg"
    return 0
}

normalize_episode_token() {
    local raw_value="$1"
    if [ -z "$raw_value" ]; then
        printf '%s\n' ""
        return 0
    fi

    local lowered="${raw_value,,}"
    if [ "$lowered" = "all" ]; then
        printf '%s\n' ""
        return 0
    fi

    if [[ "$raw_value" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$raw_value"
        return 0
    fi

    return 1
}

is_episode_selector_token() {
    local raw_value="${1:-}"
    if [ -z "$raw_value" ]; then
        return 1
    fi

    local lowered="${raw_value,,}"
    if [ "$lowered" = "all" ]; then
        return 0
    fi

    [[ "$raw_value" =~ ^[0-9]+$ ]]
}

resolve_dataset_episode_bounds() {
    local marker="SPACEVLN_EPISODE_BOUNDS="
    local raw_output=""
    raw_output="$({
        cd "$PROJECT_ROOT" || exit 1
        PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" "$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
import sys

from navigation_system.config import get_config
from navigation_system.runtime.vlnce.episode_selection import (
    MAX_EPISODE_ID,
    MIN_EPISODE_ID,
    get_available_episode_ids,
)

config_path = sys.argv[1]

try:
    config = get_config(config_path, [])
    available_ids = get_available_episode_ids(config)
    if available_ids:
        min_id = int(available_ids[0])
        max_id = int(available_ids[-1])
        total = int(len(available_ids))
    else:
        min_id = int(MIN_EPISODE_ID)
        max_id = int(MAX_EPISODE_ID)
        total = int(MAX_EPISODE_ID - MIN_EPISODE_ID + 1)
except Exception:
    min_id = int(MIN_EPISODE_ID)
    max_id = int(MAX_EPISODE_ID)
    total = int(MAX_EPISODE_ID - MIN_EPISODE_ID + 1)

print(f"SPACEVLN_EPISODE_BOUNDS={min_id},{max_id},{total}")
PY
    } 2>/dev/null || true)"

    local bounds_line=""
    bounds_line="$(printf '%s\n' "$raw_output" | grep "^${marker}" | tail -n 1 || true)"
    if [ -z "$bounds_line" ]; then
        DATASET_MIN_EPISODE_ID=1
        DATASET_MAX_EPISODE_ID=1800
        DATASET_TOTAL_EPISODES=1800
        return 0
    fi

    local parsed
    parsed="${bounds_line#${marker}}"
    IFS=',' read -r DATASET_MIN_EPISODE_ID DATASET_MAX_EPISODE_ID DATASET_TOTAL_EPISODES <<<"$parsed"

    if ! [[ "$DATASET_MIN_EPISODE_ID" =~ ^[0-9]+$ ]]; then
        DATASET_MIN_EPISODE_ID=1
    fi
    if ! [[ "$DATASET_MAX_EPISODE_ID" =~ ^[0-9]+$ ]]; then
        DATASET_MAX_EPISODE_ID=1800
    fi
    if ! [[ "$DATASET_TOTAL_EPISODES" =~ ^[0-9]+$ ]]; then
        DATASET_TOTAL_EPISODES=$((DATASET_MAX_EPISODE_ID - DATASET_MIN_EPISODE_ID + 1))
    fi
}

resolve_results_targets() {
    local selector="$1"
    local default_dir="$2"
    local -a raw_targets=()
    local token=""

    RESULTS_TARGET_DIRS=()

    if [ -z "$selector" ]; then
        raw_targets+=("$default_dir")
    else
        local -a tokens=()
        IFS=',' read -r -a tokens <<<"$selector"
        for token in "${tokens[@]}"; do
            token="${token#${token%%[![:space:]]*}}"
            token="${token%${token##*[![:space:]]}}"
            if [ -z "$token" ]; then
                continue
            fi

            if [ "${token,,}" = "all" ]; then
                if [ -z "$RESULTS_ROOT" ]; then
                    RESULTS_ROOT="$(spacevln_default_results_root "$PYTHON_BIN")"
                fi
                if [ ! -d "$RESULTS_ROOT" ]; then
                    echo "❌ results root does not exist: $RESULTS_ROOT" >&2
                    return 1
                fi

                while IFS= read -r found_dir; do
                    raw_targets+=("$found_dir")
                done < <(find "$RESULTS_ROOT" -maxdepth 6 -type d -name log -printf '%h\n' | sort -u)
                continue
            fi

            raw_targets+=("$(resolve_results_dir "$token" "$default_dir")")
        done
    fi

    local -A seen=()
    local target=""
    for target in "${raw_targets[@]}"; do
        if [ -z "$target" ]; then
            continue
        fi
        if [ -n "${seen[$target]+x}" ]; then
            continue
        fi
        seen["$target"]=1

        if [ ! -d "$target" ]; then
            echo "❌ Directory does not exist: $target" >&2
            return 1
        fi
        RESULTS_TARGET_DIRS+=("$target")
    done

    if [ "${#RESULTS_TARGET_DIRS[@]}" -eq 0 ]; then
        echo "❌ No result directories are available for reporting" >&2
        return 1
    fi

    return 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --start-id)
            if [[ $# -lt 2 ]]; then
                echo "❌ --start-id requires an ID or all"
                exit 1
            fi
            START_ID_RAW="$2"
            shift 2
            ;;
        --start-id=*)
            START_ID_RAW="${1#*=}"
            shift
            ;;
        --end-id)
            if [[ $# -lt 2 ]]; then
                echo "❌ --end-id requires an ID or all"
                exit 1
            fi
            END_ID_RAW="$2"
            shift 2
            ;;
        --end-id=*)
            END_ID_RAW="${1#*=}"
            shift
            ;;
        --all)
            START_ID_RAW="all"
            END_ID_RAW="all"
            shift
            ;;
        --results|--results-dir|--model|--models)
            if [[ $# -lt 2 ]]; then
                echo "❌ $1 requires a directory, experiment name, all, or a comma-separated list"
                exit 1
            fi
            RESULTS_SELECTOR="$2"
            shift 2
            ;;
        --results=*|--results-dir=*|--model=*|--models=*)
            RESULTS_SELECTOR="${1#*=}"
            shift
            ;;
        --results-root)
            if [[ $# -lt 2 ]]; then
                echo "❌ --results-root requires a directory"
                exit 1
            fi
            RESULTS_ROOT_OVERRIDE="$2"
            shift 2
            ;;
        --results-root=*)
            RESULTS_ROOT_OVERRIDE="${1#*=}"
            shift
            ;;
        --exp-config)
            if [[ $# -lt 2 ]]; then
                echo "❌ --exp-config requires a config-file path"
                exit 1
            fi
            CONFIG_FILE="$2"
            shift 2
            ;;
        --exp-config=*)
            CONFIG_FILE="${1#*=}"
            shift
            ;;
        --load-workers)
            if [[ $# -lt 2 ]]; then
                echo "❌ --load-workers requires a positive integer"
                exit 1
            fi
            REPORT_LOAD_WORKERS="$2"
            shift 2
            ;;
        --load-workers=*)
            REPORT_LOAD_WORKERS="${1#*=}"
            shift
            ;;
        --summary-only)
            REPORT_SUMMARY_ONLY=1
            shift
            ;;
        --fast)
            REPORT_SUMMARY_ONLY=1
            shift
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                POSITIONAL_ARGS+=("$1")
                shift
            done
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

POSITIONAL_COUNT="${#POSITIONAL_ARGS[@]}"

if [ "$POSITIONAL_COUNT" -gt 3 ]; then
    echo "❌ Too many positional arguments"
    usage
    exit 1
fi

if [ -z "$START_ID_RAW" ] && [ -z "$END_ID_RAW" ] && [ -z "$RESULTS_SELECTOR" ]; then
    case "$POSITIONAL_COUNT" in
        0)
            ;;
        1)
            if is_episode_selector_token "${POSITIONAL_ARGS[0]}"; then
                START_ID_RAW="${POSITIONAL_ARGS[0]}"
            else
                RESULTS_SELECTOR="${POSITIONAL_ARGS[0]}"
            fi
            ;;
        2)
            if is_episode_selector_token "${POSITIONAL_ARGS[0]}" && is_episode_selector_token "${POSITIONAL_ARGS[1]}"; then
                START_ID_RAW="${POSITIONAL_ARGS[0]}"
                END_ID_RAW="${POSITIONAL_ARGS[1]}"
            elif is_episode_selector_token "${POSITIONAL_ARGS[0]}"; then
                START_ID_RAW="${POSITIONAL_ARGS[0]}"
                RESULTS_SELECTOR="${POSITIONAL_ARGS[1]}"
            else
                echo "❌ Invalid positional arguments: when passing 2 args, use [start end] or [start results_selector]"
                exit 1
            fi
            ;;
        3)
            START_ID_RAW="${POSITIONAL_ARGS[0]}"
            END_ID_RAW="${POSITIONAL_ARGS[1]}"
            RESULTS_SELECTOR="${POSITIONAL_ARGS[2]}"
            ;;
    esac
else
    if [ -z "$START_ID_RAW" ] && [ "$POSITIONAL_COUNT" -ge 1 ]; then
        START_ID_RAW="${POSITIONAL_ARGS[0]}"
    fi
    if [ -z "$END_ID_RAW" ] && [ "$POSITIONAL_COUNT" -ge 2 ]; then
        END_ID_RAW="${POSITIONAL_ARGS[1]}"
    fi
    if [ -z "$RESULTS_SELECTOR" ] && [ "$POSITIONAL_COUNT" -ge 3 ]; then
        RESULTS_SELECTOR="${POSITIONAL_ARGS[2]}"
    fi
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Habitat config does not exist: $CONFIG_FILE"
    exit 1
fi

if ! [[ "$REPORT_LOAD_WORKERS" =~ ^[0-9]+$ ]] || [ "$REPORT_LOAD_WORKERS" -lt 1 ]; then
    echo "❌ SPACEVLN_REPORT_WORKERS / --load-workers must be a positive integer >= 1: $REPORT_LOAD_WORKERS"
    exit 1
fi

resolve_dataset_episode_bounds

if ! START_ID="$(normalize_episode_token "$START_ID_RAW")"; then
    echo "❌ start_episode_id only accepts positive integers or all: $START_ID_RAW"
    exit 1
fi

if ! END_ID="$(normalize_episode_token "$END_ID_RAW")"; then
    echo "❌ end_episode_id only accepts positive integers or all: $END_ID_RAW"
    exit 1
fi

if [ -n "$START_ID" ] && [ -z "$END_ID" ]; then
    END_ID="$DATASET_MAX_EPISODE_ID"
fi
if [ -z "$START_ID" ] && [ -n "$END_ID" ]; then
    START_ID="$DATASET_MIN_EPISODE_ID"
fi

if [ -n "$START_ID" ]; then
    if [ "$START_ID" -lt "$DATASET_MIN_EPISODE_ID" ]; then
        echo "⚠️  start_episode_id=$START_ID is below the dataset minimum $DATASET_MIN_EPISODE_ID; clipping automatically"
        START_ID="$DATASET_MIN_EPISODE_ID"
    fi
    if [ "$START_ID" -gt "$DATASET_MAX_EPISODE_ID" ]; then
        echo "❌ start_episode_id exceeds the dataset maximum $DATASET_MAX_EPISODE_ID: $START_ID"
        exit 1
    fi
fi

if [ -n "$END_ID" ]; then
    if [ "$END_ID" -lt "$DATASET_MIN_EPISODE_ID" ]; then
        echo "❌ end_episode_id is below the dataset minimum $DATASET_MIN_EPISODE_ID: $END_ID"
        exit 1
    fi
    if [ "$END_ID" -gt "$DATASET_MAX_EPISODE_ID" ]; then
        echo "⚠️  end_episode_id=$END_ID exceeds the dataset maximum $DATASET_MAX_EPISODE_ID; clipping automatically"
        END_ID="$DATASET_MAX_EPISODE_ID"
    fi
fi

if [ -n "$START_ID" ] && [ -n "$END_ID" ] && [ "$END_ID" -lt "$START_ID" ]; then
    echo "❌ end_episode_id cannot be smaller than start_episode_id: $START_ID -> $END_ID"
    exit 1
fi

if [ -n "$RESULTS_ROOT_OVERRIDE" ]; then
    RESULTS_ROOT="$RESULTS_ROOT_OVERRIDE"
else
    RESULTS_ROOT="$(spacevln_default_results_root "$PYTHON_BIN")"
fi

if [[ "$RESULTS_ROOT" != /* ]] && [ -d "$RESULTS_ROOT" ]; then
    RESULTS_ROOT="$(cd "$RESULTS_ROOT" && pwd)"
fi

cd "$PROJECT_ROOT"

DEFAULT_RESULTS_DIR=""
if [ -z "$RESULTS_SELECTOR" ] || [ "${RESULTS_SELECTOR,,}" = "all" ] || [[ "$RESULTS_SELECTOR" == *,* ]]; then
    DEFAULT_RESULTS_DIR="$(spacevln_default_results_dir "$PYTHON_BIN" "navigation_system/config/vlm/vlm_api_config.yaml")"
fi

if ! resolve_results_targets "$RESULTS_SELECTOR" "$DEFAULT_RESULTS_DIR"; then
    echo ""
    print_available_results_dirs
    exit 1
fi

echo "📦 Dataset episode range: [$DATASET_MIN_EPISODE_ID, $DATASET_MAX_EPISODE_ID] (count=$DATASET_TOTAL_EPISODES)"
if [ "$REPORT_SUMMARY_ONLY" -eq 1 ]; then
    echo "⚡ Fast mode: summary metrics only (summary + metrics.json), skipping episode-level CSV/MD."
fi

overall_rc=0
for RESULTS_DIR in "${RESULTS_TARGET_DIRS[@]}"; do
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "❌ Directory does not exist: $RESULTS_DIR"
        overall_rc=1
        continue
    fi

    if [ ! -d "$RESULTS_DIR/log" ]; then
        echo "⚠️  Skipping directory (missing log subdirectory): $RESULTS_DIR"
        overall_rc=1
        continue
    fi

    local_range="all(existing logs)"
    local_output="$RESULTS_DIR"
    if [ -n "$START_ID" ] || [ -n "$END_ID" ]; then
        local_range="${START_ID:-start}-${END_ID:-end}"
        local_output="$RESULTS_DIR/reports/${START_ID:-start}-${END_ID:-end}"
    fi

    echo ""
    echo "📊 Generating results report"
    echo "   Range: $local_range"
    echo "   Source: $RESULTS_DIR"
    echo "   Output: $local_output"
    echo "   Load workers: $REPORT_LOAD_WORKERS"
    if [ "$REPORT_SUMMARY_ONLY" -eq 1 ]; then
        echo "   Mode: summary-only"
    fi

    cmd=(
        "$PYTHON_BIN" -m navigation_system.runtime.results_report
        --path "$RESULTS_DIR"
        --exp-config "$CONFIG_FILE"
        --save
        --load-workers "$REPORT_LOAD_WORKERS"
    )

    if [ -n "$START_ID" ]; then
        cmd+=(--start-id "$START_ID")
    fi
    if [ -n "$END_ID" ]; then
        cmd+=(--end-id "$END_ID")
    fi
    if [ "$REPORT_SUMMARY_ONLY" -eq 1 ]; then
        cmd+=(--summary-only)
    fi

    if ! "${cmd[@]}"; then
        echo "❌ Report generation failed: $RESULTS_DIR"
        overall_rc=1
    fi
done

exit "$overall_rc"
