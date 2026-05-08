#!/bin/bash
# Small shared helpers for report entrypoints.

spacevln_report_default_workers() {
    local workers
    workers="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
    if ! [[ "$workers" =~ ^[0-9]+$ ]] || [ "$workers" -lt 1 ]; then
        workers=8
    fi
    workers=$((workers * 4))
    if [ "$workers" -lt 8 ]; then
        workers=8
    fi
    if [ "$workers" -gt 64 ]; then
        workers=64
    fi
    printf '%s\n' "$workers"
}

spacevln_report_model_dir_name() {
    local python_bin="$1"
    local api_config="$2"
    local runtime_mode="${3:-standard}"
    local project_root
    project_root="$(spacevln_project_root)"
    (
        cd "$project_root" || exit 1
        PYTHONPATH="$project_root:${PYTHONPATH:-}" "$python_bin" - "$api_config" "$runtime_mode" <<'PY'
import sys
from navigation_system.runtime.storage.results_layout import (
    build_default_context_cache_results_dir,
    build_default_results_family_root,
    build_model_results_dir_name,
)

api_config = sys.argv[1]
runtime_mode = sys.argv[2]
if runtime_mode == "context_cache":
    family_root = build_default_results_family_root("placeholder")
    results_dir = build_default_context_cache_results_dir(api_config, family="placeholder")
    prefix = family_root + "/"
    print(results_dir[len(prefix):] if results_dir.startswith(prefix) else results_dir)
else:
    print(build_model_results_dir_name(api_config))
PY
    )
}

spacevln_report_family_root() {
    local python_bin="$1"
    local family="$2"
    local project_root
    project_root="$(spacevln_project_root)"
    (
        cd "$project_root" || exit 1
        PYTHONPATH="$project_root:${PYTHONPATH:-}" "$python_bin" - "$family" <<'PY'
import sys
from navigation_system.runtime.storage.results_layout import build_default_results_family_root

print(build_default_results_family_root(sys.argv[1]))
PY
    )
}

spacevln_report_is_number() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

spacevln_report_parse_range() {
    REPORT_START=""
    REPORT_END=""
    REPORT_RANGE_LABEL="all"
    REPORT_REMAINING_ARGS=("$@")

    if [ "$#" -eq 0 ]; then
        return 0
    fi

    local first="${1:-}"
    if [[ "${first,,}" == "all" ]]; then
        REPORT_REMAINING_ARGS=("${@:2}")
        return 0
    fi

    if [[ "$first" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        REPORT_START="${BASH_REMATCH[1]}"
        REPORT_END="${BASH_REMATCH[2]}"
        REPORT_RANGE_LABEL="${REPORT_START}-${REPORT_END}"
        REPORT_REMAINING_ARGS=("${@:2}")
        return 0
    fi

    if spacevln_report_is_number "$first"; then
        if [ "$#" -lt 2 ] || ! spacevln_report_is_number "${2:-}"; then
            echo "Range must be either 'all', 'start end', or 'start-end'." >&2
            return 1
        fi
        REPORT_START="$first"
        REPORT_END="$2"
        REPORT_RANGE_LABEL="${REPORT_START}-${REPORT_END}"
        REPORT_REMAINING_ARGS=("${@:3}")
        return 0
    fi

    # No range token was provided; keep all args for model/mode parsing.
    return 0
}

spacevln_report_resolve_results_dir() {
    local raw="$1"
    local default_root="$2"
    local default_dir="$3"
    local project_root
    project_root="$(spacevln_project_root)"

    if [ -z "$raw" ]; then
        printf '%s\n' "$default_dir"
        return 0
    fi
    if [[ "$raw" = /* ]]; then
        printf '%s\n' "$raw"
        return 0
    fi
    if [ -d "$raw" ]; then
        (cd "$raw" && pwd)
        return 0
    fi
    if [ -d "$project_root/$raw" ]; then
        (cd "$project_root/$raw" && pwd)
        return 0
    fi
    if [ -d "$project_root/../$raw" ]; then
        (cd "$project_root/../$raw" && pwd)
        return 0
    fi
    if [ -d "$default_root/$raw" ]; then
        (cd "$default_root/$raw" && pwd)
        return 0
    fi
    printf '%s\n' "$default_root/$raw"
}

spacevln_report_run_results_report_md() {
    local python_bin="$1"
    local project_root="$2"
    local results_dir="$3"
    local exp_config="$4"
    local workers="$5"
    local start_id="$6"
    local end_id="$7"
    local range_key="${8:-episode_id}"

    local cmd=(
        "$python_bin" -m navigation_system.runtime.results_report
        --path "$results_dir"
        --exp-config "$exp_config"
        --save
        --md-only
        --load-workers "$workers"
        --range-key "$range_key"
    )
    if [ -n "$start_id" ]; then
        cmd+=(--start-id "$start_id")
    fi
    if [ -n "$end_id" ]; then
        cmd+=(--end-id "$end_id")
    fi
    (cd "$project_root" && PYTHONPATH="$project_root:${PYTHONPATH:-}" "${cmd[@]}")
}
