#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPACEVLN_DIR="$(cd "${REAL_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${SPACEVLN_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GROUNDINGDINO_DIR="${GROUNDINGDINO_DIR:-${WORKSPACE_DIR}/GroundingDINO}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.7}"

if [[ ! -d "${GROUNDINGDINO_DIR}/groundingdino" ]]; then
  echo "ERROR: GroundingDINO source not found at ${GROUNDINGDINO_DIR}" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_real_accel_env.sh"

cuda_src="${GROUNDINGDINO_DIR}/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu"
if [[ -f "${cuda_src}" ]]; then
  sed -i \
    -e 's/value.type().is_cuda()/value.is_cuda()/g' \
    -e 's/AT_DISPATCH_FLOATING_TYPES(value.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' \
    -e 's/\.data</.data_ptr</g' \
    "${cuda_src}"
fi

echo "[GroundingDINO] dir=${GROUNDINGDINO_DIR}"
echo "[GroundingDINO] CUDA_HOME=${CUDA_HOME:-none}"
echo "[GroundingDINO] torch_lib=${SPACEVLN_TORCH_LIB_DIR:-none}"
echo "[GroundingDINO] arch=${TORCH_CUDA_ARCH_LIST}"

cd "${GROUNDINGDINO_DIR}"
rm -rf build groundingdino/_C*.so 2>/dev/null || sudo rm -rf build groundingdino/_C*.so

TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
"${PYTHON_BIN}" setup.py build_ext --inplace

"${PYTHON_BIN}" - <<'PY'
import groundingdino._C as C

print("GroundingDINO _C OK")
PY
