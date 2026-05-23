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
if [[ -z "${OUTPUT_IMAGE:-}" ]]; then
  image_dir="$(dirname "${TEST_IMAGE}")"
  image_base="$(basename "${TEST_IMAGE}")"
  OUTPUT_IMAGE="${image_dir}/${image_base%.*}_grounded_sam.jpg"
fi
CLASSES="${CLASSES:-table,chair,door,sofa,person,cabinet}"
CAPTION="${CAPTION:-}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.25}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
QUIET="${QUIET:-1}"
LOG_FILE="${LOG_FILE:-/tmp/grounded_sam_test.log}"
GROUNDINGDINO_DEVICE="${GROUNDINGDINO_DEVICE:-auto}"

export PYTHONPATH="${GROUNDINGDINO_DIR}:${SPACEVLN_DIR}:${REAL_DIR}:${PYTHONPATH:-}"
export MODEL_DIR TEST_IMAGE OUTPUT_IMAGE CLASSES CAPTION BOX_THRESHOLD TEXT_THRESHOLD
export GROUNDINGDINO_DEVICE
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi
export SPACEVLN_GROUNDINGDINO_CPU_FALLBACK="${SPACEVLN_GROUNDINGDINO_CPU_FALLBACK:-1}"

if [[ "${QUIET}" == "1" ]]; then
  exec 3>&1
  exec >"${LOG_FILE}" 2>&1
else
  exec 3>&1
fi

"${PYTHON_BIN}" - <<'PY'
import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import supervision as sv

from navigation_system.detection.grounded_sam import GroundedSAM
from groundingdino.util.inference import Model

model_dir = os.environ["MODEL_DIR"]
test_image = os.environ["TEST_IMAGE"]
output_image = os.environ["OUTPUT_IMAGE"]
classes = [
    item.strip()
    for item in os.environ.get("CLASSES", "").split(",")
    if item.strip()
]
caption = os.environ.get("CAPTION", "").strip()
box_threshold = float(os.environ.get("BOX_THRESHOLD", "0.25"))
text_threshold = float(os.environ.get("TEXT_THRESHOLD", "0.25"))

required = [
    "GroundingDINO_SwinT_OGC.py",
    "groundingdino_swint_ogc.pth",
    "sam_vit_h_4b8939.pth",
]
for name in required:
    path = os.path.join(model_dir, name)
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

device_name = os.environ.get("GROUNDINGDINO_DEVICE", "auto").strip().lower()
if device_name in {"auto", ""}:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif device_name.startswith("cuda"):
    device = torch.device(device_name)
else:
    device = torch.device("cpu")

print(f"device: {device}")
print(f"classes: {classes}")
if caption:
    print(f"caption: {caption}")
print(f"thresholds: box={box_threshold:.3f} text={text_threshold:.3f}")

if caption:
    dino = Model(
        model_config_path=os.path.join(model_dir, "GroundingDINO_SwinT_OGC.py"),
        model_checkpoint_path=os.path.join(model_dir, "groundingdino_swint_ogc.pth"),
        device=str(device),
    )
    detections, phrases = dino.predict_with_caption(
        image=image,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    labels = []
    for phrase, confidence in zip(phrases, detections.confidence):
        labels.append(f"{phrase} {float(confidence):0.2f}")
    annotated = image.copy()
    try:
        annotated = sv.BoxAnnotator().annotate(
            scene=annotated,
            detections=detections,
            labels=labels,
        )
    except TypeError:
        annotated = sv.BoxAnnotator().annotate(
            scene=annotated,
            detections=detections,
        )
    cv2.imwrite(output_image, annotated)
    print("GROUNDING_SAM_RESULT_START")
    print("boxes:", len(detections.xyxy))
    print("labels:", labels)
    print("masks:", None)
    print("saved:", output_image)
    print("GROUNDING_SAM_RESULT_END")
    raise SystemExit(0)

model = GroundedSAM(cfg, device)
masks, labels, annotated, detections = model.segment(
    image,
    classes=classes,
    box_threshold=box_threshold,
    text_threshold=text_threshold,
)

cv2.imwrite(output_image, annotated)
xyxy = getattr(detections, "xyxy", None)
print("GROUNDING_SAM_RESULT_START")
print("boxes:", 0 if xyxy is None else len(xyxy))
print("labels:", labels)
print("masks:", getattr(masks, "shape", None))
print("saved:", output_image)
print("GROUNDING_SAM_RESULT_END")
PY

if [[ "${QUIET}" == "1" ]]; then
  sed -n '/GROUNDING_SAM_RESULT_START/,/GROUNDING_SAM_RESULT_END/p' "${LOG_FILE}" \
    | sed '/GROUNDING_SAM_RESULT_/d' >&3
  echo "log: ${LOG_FILE}" >&3
fi
