# Local Changes (2026-03-23)

This file records local runtime-oriented changes made to run SpaceVLN in `nav_ws`.

## 1) Path migration to relative paths

- Updated AutoDL absolute paths to relative workspace paths.
- Key files:
  - `habitat_extensions/config/spacevln_task.yaml`
  - `vlnce_baselines/config/runtime/default.py`
  - `run_r2r/vlm_navigation.sh`

## 2) Launcher/runtime stability

- `run_r2r/vlm_navigation.sh` now:
  - Prefers `spatial_agent` interpreter
  - Exports NVIDIA EGL vendor variables for headless rendering
  - Exports `LD_LIBRARY_PATH` for conda/torch native libraries
  - Writes outputs to `../data/result/spacevln`

## 3) Detection compatibility fix

- `vlnce_baselines/detection/grounded_sam.py`
  - Added compatibility-safe handling for `supervision==0.4.0`
  - Avoids `AttributeError: 'Detections' object has no attribute 'mask'`

## 4) Dependency alignment

- `requirements.txt` updated for validated local run versions:
  - `gym==0.21.0`
  - `numpy==1.23.5`
  - `supervision==0.4.0`
  - Added `platformdirs`, `tomli`
  - Added editable local dependency: `-e ../habitat-lab`

## 5) Current external blocker

- Runtime now reaches model loading stage.
- Remaining blocker is network/DNS access to `huggingface.co` for model/tokenizer fetch unless local cache/model path is provided.
