#!/bin/bash

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
    if [ -x "$HOME/anaconda3/envs/spatial_agent/bin/python" ]; then
        printf '%s\n' "$HOME/anaconda3/envs/spatial_agent/bin/python"
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

spacevln_setup_runtime_env() {
    local python_bin="$1"

    export GLOG_minloglevel="${GLOG_minloglevel:-2}"
    export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
    export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
    export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-spacevln}"
    mkdir -p "$MPLCONFIGDIR"

    export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
    export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"

    if [[ "$python_bin" == "$HOME/anaconda3/envs/"*"/bin/python" ]]; then
        local env_root
        env_root="$(spacevln_env_root "$python_bin")"
        export LD_LIBRARY_PATH="$env_root/lib:$env_root/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
    fi
}

spacevln_default_results_dir() {
    local python_bin="$1"
    local api_config="$2"
    "$python_bin" - <<PY
from navigation_system.vlm.api.client import build_default_results_dir_from_api_config
print(build_default_results_dir_from_api_config("$api_config"))
PY
}

spacevln_validate_parallel_workers() {
    local workers="$1"
    if ! [[ "$workers" =~ ^[0-9]+$ ]] || [ "$workers" -lt 1 ]; then
        echo "❌ parallel_workers 必须是大于等于 1 的正整数: $workers"
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
            echo "❌ 不支持的模式: $mode" >&2
            echo "   可选模式: all | skip-sr1" >&2
            return 1
            ;;
    esac
}
