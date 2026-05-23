#!/usr/bin/env bash
# Source this from real-robot launch/test scripts to make CUDA/PyTorch
# extension libraries visible without hand-writing long env prefixes.

_spacevln_prepend_path() {
  local var_name="$1"
  local entry="$2"
  [[ -n "${entry}" && -d "${entry}" ]] || return 0

  local current="${!var_name:-}"
  case ":${current}:" in
    *":${entry}:"*) ;;
    *) export "${var_name}=${entry}${current:+:${current}}" ;;
  esac
}

spacevln_setup_accel_env() {
  local python_bin="${PYTHON_BIN:-python3}"

  if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -d "/usr/local/cuda" ]]; then
      export CUDA_HOME="/usr/local/cuda"
    elif [[ -d "/usr/local/cuda-12.6" ]]; then
      export CUDA_HOME="/usr/local/cuda-12.6"
    fi
  fi

  if [[ -n "${CUDA_HOME:-}" ]]; then
    _spacevln_prepend_path PATH "${CUDA_HOME}/bin"
    _spacevln_prepend_path LD_LIBRARY_PATH "${CUDA_HOME}/lib64"
  fi

  local torch_lib=""
  torch_lib="$(
    "${python_bin}" - <<'PY' 2>/dev/null || true
from pathlib import Path

try:
    import torch
except Exception:
    raise SystemExit(0)

torch_lib = Path(torch.__file__).resolve().parent / "lib"
if torch_lib.is_dir():
    print(torch_lib)
PY
  )"
  if [[ -n "${torch_lib}" ]]; then
    export SPACEVLN_TORCH_LIB_DIR="${torch_lib}"
    _spacevln_prepend_path LD_LIBRARY_PATH "${torch_lib}"
  fi

  export SPACEVLN_ACCEL_ENV_READY=1
}

spacevln_setup_accel_env "$@"
