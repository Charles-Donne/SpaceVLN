#!/bin/bash

spacevln_shell_name_for_entry() {
    local entry_script="$1"
    local entry_name
    entry_name="$(basename "$entry_script" .py)"
    case "$entry_name" in
        vlm_navigation)
            printf '%s\n' "vlnce.sh"
            ;;
        object_navigation)
            printf '%s\n' "object_navigation.sh"
            ;;
        *)
            printf '%s.sh\n' "$entry_name"
            ;;
    esac
}

spacevln_common_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

spacevln_project_root() {
    cd "$(spacevln_common_dir)/.." && pwd
}

spacevln_select_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        printf '%s\n' "$PYTHON_BIN"
        return
    fi
    if [ -x "$HOME/anaconda3/envs/spacevln/bin/python" ]; then
        printf '%s\n' "$HOME/anaconda3/envs/spacevln/bin/python"
        return
    fi
    if [ -x "$HOME/.conda/envs/spacevln/bin/python" ]; then
        printf '%s\n' "$HOME/.conda/envs/spacevln/bin/python"
        return
    fi
    if [ -x "$HOME/anaconda3/envs/spatial_agent/bin/python" ]; then
        printf '%s\n' "$HOME/anaconda3/envs/spatial_agent/bin/python"
        return
    fi
    if [ -x "$HOME/.conda/envs/spatial_agent/bin/python" ]; then
        printf '%s\n' "$HOME/.conda/envs/spatial_agent/bin/python"
        return
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return
    fi
    printf '%s\n' "python"
}

spacevln_env_root() {
    local python_bin="$1"
    local python_dir
    python_dir="$(cd "$(dirname "$python_bin")" && pwd)"
    cd "$python_dir/.." && pwd
}

spacevln_prepend_colon_var() {
    local var_name="$1"
    local value="$2"
    local current_value="${!var_name:-}"

    if [[ -z "$value" ]]; then
        return
    fi

    if [[ -z "$current_value" ]]; then
        export "$var_name=$value"
        return
    fi

    if [[ ":$current_value:" == *":$value:"* ]]; then
        return
    fi

    export "$var_name=$value:$current_value"
}

spacevln_habitat_sim_ext_dir() {
    local python_bin="$1"
    "$python_bin" - <<'PY'
import glob
import os
import site

candidates = []
for site_dir in site.getsitepackages():
    candidates.extend(
        glob.glob(
            os.path.join(
                site_dir, "habitat_sim-*.egg", "habitat_sim", "_ext"
            )
        )
    )
    candidates.extend(
        glob.glob(os.path.join(site_dir, "habitat_sim", "_ext"))
    )

for path in candidates:
    if os.path.isdir(path):
        print(path)
        break
PY
}

spacevln_setup_runtime_env() {
    local python_bin="$1"
    local resolved_python_bin=""

    if [[ "$python_bin" != /* ]]; then
        resolved_python_bin="$(command -v "$python_bin" 2>/dev/null || true)"
        if [[ -n "$resolved_python_bin" ]]; then
            python_bin="$resolved_python_bin"
        fi
    fi

    export GLOG_minloglevel="${GLOG_minloglevel:-2}"
    export HABITAT_SIM_LOG="${HABITAT_SIM_LOG:-quiet}"
    export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
    export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
    export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-spacevln}"
    mkdir -p "$MPLCONFIGDIR"

    export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
    export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"

    if [[ "$python_bin" == */envs/*/bin/python ]]; then
        local env_root
        local torch_lib
        local preload_libs=()
        env_root="$(spacevln_env_root "$python_bin")"
        torch_lib="$env_root/lib/python3.8/site-packages/torch/lib"

        if [[ -d "$torch_lib" ]]; then
            spacevln_prepend_colon_var LD_LIBRARY_PATH "$torch_lib"
        fi

        local habitat_sim_ext
        habitat_sim_ext="$(spacevln_habitat_sim_ext_dir "$python_bin")"
        if [[ -n "$habitat_sim_ext" && -d "$habitat_sim_ext" ]]; then
            spacevln_prepend_colon_var LD_LIBRARY_PATH "$habitat_sim_ext"
        fi

        if [[ -f "$env_root/lib/libstdc++.so.6" ]]; then
            preload_libs+=("$env_root/lib/libstdc++.so.6")
        fi
        if [[ -f "$env_root/lib/libgcc_s.so.1" ]]; then
            preload_libs+=("$env_root/lib/libgcc_s.so.1")
        fi
        if [[ -f "/lib/x86_64-linux-gnu/libEGL.so.1" ]]; then
            preload_libs+=("/lib/x86_64-linux-gnu/libEGL.so.1")
        elif [[ -f "/usr/lib/x86_64-linux-gnu/libEGL.so.1" ]]; then
            preload_libs+=("/usr/lib/x86_64-linux-gnu/libEGL.so.1")
        fi
        if [[ -f "/lib/x86_64-linux-gnu/libGLdispatch.so.0" ]]; then
            preload_libs+=("/lib/x86_64-linux-gnu/libGLdispatch.so.0")
        elif [[ -f "/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0" ]]; then
            preload_libs+=("/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0")
        fi
        if [[ ${#preload_libs[@]} -gt 0 ]]; then
            local preload_lib
            for preload_lib in "${preload_libs[@]}"; do
                spacevln_prepend_colon_var LD_PRELOAD "$preload_lib"
            done
        fi
    fi
}

spacevln_default_results_dir() {
    local python_bin="$1"
    local api_config="$2"
    local project_root
    project_root="$(spacevln_project_root)"
    (
        cd "$project_root" || exit 1
        PYTHONPATH="$project_root:${PYTHONPATH:-}" "$python_bin" - <<PY
from navigation_system.vlm.api.api_client import build_default_results_dir_from_api_config
print(build_default_results_dir_from_api_config("$api_config"))
PY
    )
}

spacevln_default_results_root() {
    local python_bin="$1"
    local project_root
    project_root="$(spacevln_project_root)"
    (
        cd "$project_root" || exit 1
        PYTHONPATH="$project_root:${PYTHONPATH:-}" "$python_bin" - <<PY
from navigation_system.vlm.api.api_client import build_default_results_family_root
print(build_default_results_family_root("vlnce"))
PY
    )
}

spacevln_default_context_cache_api_config() {
    printf '%s\n' "navigation_system/config/vlm/vlm_api_config.yaml"
}

spacevln_validate_parallel_workers() {
    local workers="$1"
    if ! [[ "$workers" =~ ^[0-9]+$ ]] || [ "$workers" -lt 1 ]; then
        echo "❌ parallel_workers must be a positive integer >= 1: $workers"
        return 1
    fi
}

spacevln_mode_arg() {
    local mode="$1"
    case "$mode" in
        ""|all)
            printf '%s\n' ""
            ;;
        skip-sr1|skip_sr1|resume)
            printf '%s\n' "--skip-sr1"
            ;;
        *)
            echo "❌ Unsupported mode: $mode" >&2
            echo "   Supported modes: all | skip-sr1" >&2
            return 1
            ;;
    esac
}

spacevln_is_help_request() {
    local arg=""
    for arg in "$@"; do
        case "$arg" in
            -h|--help|help)
                return 0
                ;;
        esac
    done
    return 1
}

spacevln_dispatch_navigation_cli() {
    local project_root="$1"
    local python_bin="$2"
    local entry_script="$3"
    local config_file="$4"
    local api_config="$5"
    local api_missing_message="$6"
    local api_missing_hint="$7"
    local extra_args_name="${8:-}"
    shift 8

    local cli_args=("$@")
    local extra_args=()
    if [[ -n "$extra_args_name" ]]; then
        local -n _spacevln_extra_args_ref="$extra_args_name"
        extra_args=("${_spacevln_extra_args_ref[@]}")
    fi

    cd "$project_root" || exit 1

    if [ ! -f "$config_file" ]; then
        echo "❌ Habitat config does not exist: $config_file"
        exit 1
    fi

    if spacevln_is_help_request "${cli_args[@]}"; then
        exec "$python_bin" "$entry_script" \
            --exp-config "$config_file" \
            --vlm-api-config "$api_config" \
            "${extra_args[@]}" \
            "${cli_args[@]}"
    fi

    if [ ! -f "$api_config" ]; then
        echo "❌ $api_missing_message: $api_config"
        if [ -n "$api_missing_hint" ]; then
            echo "   $api_missing_hint"
        fi
        exit 1
    fi

    spacevln_run_python_entry() {
        exec "$python_bin" "$entry_script" \
            --exp-config "$config_file" \
            --vlm-api-config "$api_config" \
            "${extra_args[@]}" \
            "$@"
    }

    for arg in "${cli_args[@]}"; do
        if [[ "$arg" == --* ]]; then
            spacevln_run_python_entry "${cli_args[@]}"
        fi
    done

    local first_arg="${cli_args[0]:-}"
    local second_arg="${cli_args[1]:-}"
    local third_arg="${cli_args[2]:-}"
    local fourth_arg="${cli_args[3]:-}"
    local fifth_arg="${cli_args[4]:-}"
    local mode="all"
    local parallel_workers="1"
    local mode_arg=""

    if [[ -n "$fourth_arg" && "$fourth_arg" =~ ^[0-9]+$ ]]; then
        parallel_workers="$fourth_arg"
    elif [[ -n "$fourth_arg" ]]; then
        mode="$fourth_arg"
    fi

    if [[ -n "$fifth_arg" ]]; then
        if [[ "$fifth_arg" =~ ^[0-9]+$ ]]; then
            parallel_workers="$fifth_arg"
        else
            echo "❌ parallel_workers must be a positive integer: $fifth_arg"
            exit 1
        fi
    fi

    spacevln_validate_parallel_workers "$parallel_workers"
    mode_arg="$(spacevln_mode_arg "$mode")"

    if [[ -z "$first_arg" ]]; then
        spacevln_run_python_entry
    fi

    if [[ "$first_arg" == "random" ]]; then
        local num_episodes="${second_arg:-1}"
        local max_steps="${third_arg:-}"
        local args=(
            --random
            --num-episodes "$num_episodes"
            --parallel-workers "$parallel_workers"
            --max-subtask-steps 5
        )
        if [[ -n "$max_steps" ]]; then
            args+=(--max-steps "$max_steps")
        fi
        if [[ -n "$mode_arg" ]]; then
            args+=("$mode_arg")
        fi
        spacevln_run_python_entry "${args[@]}"
    fi

    if [[ "$first_arg" == "list" ]]; then
        local episode_ids="$second_arg"
        local max_steps="${third_arg:-}"
        if [[ -z "$episode_ids" ]]; then
            echo "❌ list mode requires an explicit episode-id list"
            exit 1
        fi
        local args=(
            --episode-ids "$episode_ids"
            --parallel-workers "$parallel_workers"
            --max-subtask-steps 5
        )
        if [[ -n "$max_steps" ]]; then
            args+=(--max-steps "$max_steps")
        fi
        if [[ -n "$mode_arg" ]]; then
            args+=("$mode_arg")
        fi
        spacevln_run_python_entry "${args[@]}"
    fi

    if ! [[ "$first_arg" =~ ^[0-9]+$ ]]; then
        local shell_name
        shell_name="$(spacevln_shell_name_for_entry "$entry_script")"
        echo "❌ Unsupported first positional argument: $first_arg"
        echo "   Supported examples:"
        echo "   bash run_navigation/${shell_name} 832"
        echo "   bash run_navigation/${shell_name} 832 300"
        echo "   bash run_navigation/${shell_name} 1 600 260 5"
        echo "   bash run_navigation/${shell_name} random 20 260 all 4"
        echo "   bash run_navigation/${shell_name} --episode-id 832 --num-episodes 1"
        exit 1
    fi

    local episode_id="$first_arg"

    if [[ -n "$second_arg" && -z "$third_arg" ]]; then
        spacevln_run_python_entry \
            --episode-id "$episode_id" \
            --num-episodes 1 \
            --max-steps "$second_arg" \
            --parallel-workers "$parallel_workers" \
            --max-subtask-steps 5 \
            ${mode_arg:+"$mode_arg"}
    fi

    local num_episodes="${second_arg:-1}"
    local max_steps="${third_arg:-}"
    local args=(
        --episode-id "$episode_id"
        --num-episodes "$num_episodes"
        --parallel-workers "$parallel_workers"
        --max-subtask-steps 5
    )
    if [[ -n "$max_steps" ]]; then
        args+=(--max-steps "$max_steps")
    fi
    if [[ -n "$mode_arg" ]]; then
        args+=("$mode_arg")
    fi
    spacevln_run_python_entry "${args[@]}"
}
