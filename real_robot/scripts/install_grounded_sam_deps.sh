#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPACEVLN_DIR="$(cd "${REAL_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${SPACEVLN_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GROUNDINGDINO_DIR="${GROUNDINGDINO_DIR:-${WORKSPACE_DIR}/GroundingDINO}"

if [[ ! -d "${GROUNDINGDINO_DIR}/groundingdino" ]]; then
  echo "ERROR: GroundingDINO source not found at ${GROUNDINGDINO_DIR}" >&2
  echo "Set GROUNDINGDINO_DIR=/path/to/GroundingDINO or clone it next to SpaceVLN." >&2
  exit 2
fi

constraints="$(mktemp)"
trap 'rm -f "${constraints}"' EXIT
printf '%s\n' 'numpy==1.24.4' > "${constraints}"

echo "[GroundedSAM] python=$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "[GroundedSAM] GroundingDINO=${GROUNDINGDINO_DIR}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_real_accel_env.sh"

"${PYTHON_BIN}" -m pip install --user --no-cache-dir -c "${constraints}" \
  "numpy==1.24.4" \
  "transformers==4.37.2" \
  "addict" \
  "yapf" \
  "timm==0.9.16" \
  "pycocotools" \
  "nltk"

"${PYTHON_BIN}" -m pip install --user --no-cache-dir --no-deps \
  "supervision==0.6.0"

if ! PYTHONPATH="${GROUNDINGDINO_DIR}:${SPACEVLN_DIR}:${REAL_DIR}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import segment_anything
PY
then
  "${PYTHON_BIN}" -m pip install --user --no-cache-dir --no-deps \
    "git+https://github.com/facebookresearch/segment-anything.git"
fi

PYTHONPATH="${GROUNDINGDINO_DIR}:${SPACEVLN_DIR}:${REAL_DIR}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" - <<'PY'
import numpy as np
import torch
import cv2
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

print("numpy", np.__version__, np.__file__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("cv2", cv2.__version__)
print("GroundingDINO/SAM imports OK")
PY

echo "[GroundedSAM] Optional CUDA op build:"
echo "  bash real_robot/scripts/build_groundingdino_cuda_ext.sh"
