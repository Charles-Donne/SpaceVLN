#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPACEVLN_DIR="$(cd "${REAL_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${SPACEVLN_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GROUNDINGDINO_DIR="${GROUNDINGDINO_DIR:-${WORKSPACE_DIR}/GroundingDINO}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_DIR}/data/model/grounded_sam}"
TEST_IMAGE="${TEST_IMAGE:-/tmp/spacevln_rgb.jpg}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-/tmp/grounded_sam_test.jpg}"
CLASSES="${CLASSES:-table,chair,door,sofa,person,cabinet}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.25}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"

export PYTHONPATH="${GROUNDINGDINO_DIR}:${SPACEVLN_DIR}:${REAL_DIR}:${PYTHONPATH:-}"

"${PYTHON_BIN}" - <<'PY'
import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from navigation_system.detection.grounded_sam import GroundedSAM

model_dir = os.environ["MODEL_DIR"]
test_image = os.environ["TEST_IMAGE"]
output_image = os.environ["OUTPUT_IMAGE"]
classes = [
    item.strip()
    for item in os.environ.get("CLASSES", "").split(",")
    if item.strip()
]
box_threshold = float(os.environ.get("BOX_THRESHOLD", "0.25"))
text_threshold = float(os.environ.get("TEXT_THRESHOLD", "0.25"))

required = [
    "GroundingDINO_SwinT_OGC.py",
    "groundingdino_swint_ogc.pth",
    "sam_vit_h_4b8939.pth",
]
for name in required:
    path = os.path.join(model_dir, name)
    print(("OK " if os.path.exists(path) else "MISS "), path)
    if not os.path.exists(path):
        raise SystemExit(f"missing checkpoint/config: {path}")

image = cv2.imread(test_image)
if image is None:
    raise SystemExit(
        f"missing test image: {test_image}. Capture one first or set TEST_IMAGE=/path/to/rgb.jpg"
    )

cfg = SimpleNamespace(
    DETECTION=SimpleNamespace(
        MODEL=SimpleNamespace(
            GROUNDING_DINO_CONFIG_PATH=os.path.join(model_dir, "GroundingDINO_SwinT_OGC.py"),
            GROUNDING_DINO_CHECKPOINT_PATH=os.path.join(model_dir, "groundingdino_swint_ogc.pth"),
            SAM_CHECKPOINT_PATH=os.path.join(model_dir, "sam_vit_h_4b8939.pth"),
            REPVIT_SAM_CHECKPOINT_PATH=os.path.join(model_dir, "repvit_sam.pt"),
            SAM_ENCODER_VERSION="vit_h",
            USE_REPVIT_SAM=False,
        ),
        THRESHOLDS=SimpleNamespace(BOX=box_threshold, TEXT=text_threshold),
    )
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("classes:", classes)

model = GroundedSAM(cfg, device)
masks, labels, annotated, detections = model.segment(
    image,
    classes=classes,
    box_threshold=box_threshold,
    text_threshold=text_threshold,
)

cv2.imwrite(output_image, annotated)
print("labels:", labels)
print("boxes:", len(getattr(detections, "xyxy", []) or []))
print("masks:", getattr(masks, "shape", None))
print("saved:", output_image)
PY
