# SpaceVLN

SpaceVLN is a modular vision-language navigation system for continuous VLN-CE evaluation in Habitat. The repository integrates hierarchical language-conditioned planning, action selection, explicit spatial structure modeling, landmark-centric perception, grounded visual reasoning, batch evaluation utilities, and an isolated ablation subsystem for controlled analysis.

The implementation is built on ideas and engineering components inherited from prior embodied navigation systems, in particular `NaVid-VLN-CE` and `CA-Nav`, while adapting the full runtime to a Habitat-based continuous evaluation stack with GroundingDINO/SAM-style perception and a reorganized experimental interface.

## Abstract

## Overview

This repository is intended to serve as an experimental research codebase rather than a minimal demo. Relative to the upstream projects it builds upon, the current codebase emphasizes:

- a unified navigation runtime for planner/action VLM inference;
- explicit `space structure` and `landmark` representations for navigation state;
- modular rendering for thinking views, action views, and replay artifacts;
- persistent run artifacts for qualitative analysis and failure diagnosis;
- isolated ablation experiments that do not modify the original system prompts in place.

The repository is **not** a line-by-line reproduction of any single upstream project. Instead, it consolidates and adapts ideas from:

- `NaVid-VLN-CE`: <https://github.com/jzhzhang/NaVid-VLN-CE>
- `CA-Nav-code`: <https://github.com/Chenkehan21/CA-Nav-code>
- `Habitat-Lab`: <https://github.com/facebookresearch/habitat-lab>
- `Habitat-Sim`: <https://github.com/facebookresearch/habitat-sim>
- `GroundingDINO`: <https://github.com/IDEA-Research/GroundingDINO>
- `Grounded-Segment-Anything`: <https://github.com/IDEA-Research/Grounded-Segment-Anything>
- `VLN-CE`: <https://github.com/jacobkrantz/VLN-CE>

## Validated Software Stack

The current SpaceVLN codebase is organized around a **legacy but validated Habitat stack** rather than the latest Habitat releases.

### Core stack used by this repository

- Python: `3.8`
- Habitat-Lab: `0.1.7`
- Habitat-Sim: `0.1.7`
- GroundingDINO: `0.1.0`
- PyTorch: install a CUDA-compatible build for your machine
- Python package snapshot: synchronized from the validated `spatial_agent` conda environment

### Important note

The official current Habitat documentation now targets newer Python versions. However, this repository imports legacy `habitat`, `habitat_baselines`, and `habitat_sim` interfaces that are consistent with the `0.1.7` generation of Habitat-Lab / Habitat-Sim. For reproducibility, we recommend staying on that stack unless you explicitly plan to port the code.

## Repository Layout

```text
SpaceVLN/
├── navigation_system/
│   ├── controller/        # navigation control loop
│   ├── vlm/               # planner, executor, prompts, API clients
│   ├── space/             # map, topology, landmarks, spatial descriptors
│   ├── render/            # thinking/action visual inputs and replay outputs
│   ├── runtime/           # runners, batch execution, reports, storage
│   ├── detection/         # GroundingDINO + SAM / RepViT-SAM integration
│   └── ablation/          # isolated ablation subsystem
├── habitat_extensions/    # Habitat task, sensors, measures, simulator extensions
├── run_r2r/               # shell entrypoints for evaluation and reporting
├── docs/                  # architecture and deployment notes
└── vlm_navigation*.py     # Python entrypoints
```

## Recommended Workspace Layout

The default configuration assumes the following workspace organization:

```text
nav_ws/
├── SpaceVLN/
├── habitat-lab/
├── GroundingDINO/
└── data/
    ├── datasets/
    ├── scene_datasets/
    └── model/
        └── grounded_sam/
```

By default, SpaceVLN expects:

- Habitat/VLN datasets under `../data/datasets/`
- scene assets under `../data/scene_datasets/`
- detection checkpoints under `../data/model/grounded_sam/`
- a local editable `habitat-lab` source tree at `../habitat-lab`
- a local editable `GroundingDINO` source tree at `../GroundingDINO`

If your directory layout differs, update:

- `habitat_extensions/config/spacevln_task.yaml`
- `navigation_system/config/system/00_runtime.yaml`
- `navigation_system/config/system/10_detection_models.yaml`

## Installation

### 1. Create a Python environment

```bash
conda create -n spacevln python=3.8 cmake=3.14.0 -y
conda activate spacevln
```

### 2. Install PyTorch

Install a CUDA-compatible PyTorch build appropriate for your machine. One tested example is:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2+cu121 \
  torchvision==0.16.2+cu121 \
  torchaudio==2.1.2+cu121
```

If you use a different CUDA runtime, replace the wheels accordingly.

### 3. Install Habitat-Sim

For this repository, `habitat-sim==0.1.7` via conda is the recommended installation path.

```bash
conda install habitat-sim=0.1.7 headless withbullet -c conda-forge -c aihabitat
```

Remarks:

- `headless` is recommended for servers and multi-GPU machines.
- `withbullet` is recommended for consistency with Habitat installation guidance.
- If you require a display-attached build, adjust the conda flags accordingly.

### 4. Clone sibling source repositories

Clone the external source repositories expected by the current SpaceVLN workspace layout:

```bash
cd ..
git clone https://github.com/facebookresearch/habitat-lab.git
git clone https://github.com/IDEA-Research/GroundingDINO.git
```

Notes:

- the current `requirements.txt` installs both repositories in editable mode from sibling paths;
- the validated Habitat-Lab source tree used with this system exposes both `habitat` and `habitat_baselines`.

### 5. Install SpaceVLN Python dependencies

Before installation, set `CUDA_HOME` for GroundingDINO compilation:

```bash
export CUDA_HOME=/path/to/your/cuda
```

Then return to SpaceVLN and install the validated dependency snapshot:

```bash
cd ../SpaceVLN
pip install -r requirements.txt
```

This requirements file is intentionally aligned with the currently validated `spatial_agent` environment. The README still recommends creating a fresh environment named `spacevln`, but the version pins are synchronized from that validated runtime snapshot.

## Data and Checkpoints

### Navigation datasets

The default task configuration reads:

- `../data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen.json.gz`
- `../data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen_gt.json.gz`
- `../data/scene_datasets/`

These paths are defined in `habitat_extensions/config/spacevln_task.yaml`.

### Detection checkpoints

Detection model paths are defined in:

- `navigation_system/config/system/10_detection_models.yaml`

The expected layout is:

```text
../data/model/grounded_sam/
├── GroundingDINO_SwinT_OGC.py
├── groundingdino_swint_ogc.pth
├── sam_vit_h_4b8939.pth
└── repvit_sam.pt
```

Notes:

- `GroundingDINO_SwinT_OGC.py` can be copied from the GroundingDINO repository, or the config path can be redirected to the source tree.
- If RepViT-SAM is unavailable, set `USE_REPVIT_SAM: false` in `navigation_system/config/system/10_detection_models.yaml`.

## API Configuration

SpaceVLN does not commit active provider credentials. Create local configuration files from the templates in `navigation_system/config/vlm/`.

### Standard runtime

```bash
cp navigation_system/config/vlm/vlm_api_config.yaml.template \
   navigation_system/config/vlm/vlm_api_config.yaml
```

### Explicit-context-cache runtime

```bash
cp navigation_system/config/vlm/vlm_api_config_context_cache.yaml.template \
   navigation_system/config/vlm/vlm_api_config_context_cache.yaml
```

Both templates are written to prefer environment variables such as:

- `OPENAI_API_KEY`
- `DASHSCOPE_API_KEY`
- `OPENROUTER_API_KEY`

## Running Evaluation

### Standard evaluation

```bash
bash run_r2r/vlm_navigation.sh 1 10 260 4
```

This runs:

- starting episode ID: `1`
- number of episodes: `10`
- max episode steps: `260`
- parallel workers: `4`

### Explicit-context-cache evaluation

```bash
bash run_r2r/vlm_navigation.sh --runtime context_cache 1 10 260 4
```

### Single-episode evaluation

```bash
bash run_r2r/vlm_navigation.sh 832
bash run_r2r/vlm_navigation.sh 832 300
```

### Random evaluation

```bash
bash run_r2r/vlm_navigation.sh random 20 260 all 4
```

### Direct Python entrypoint

```bash
python vlm_navigation.py \
  --exp-config navigation_system/config/experiments/r2r_eval.yaml \
  --runtime standard \
  --episode-id 1 \
  --num-episodes 10 \
  --max-steps 260 \
  --parallel-workers 4 \
  --vlm-api-config navigation_system/config/vlm/vlm_api_config.yaml
```

Help menus:

```bash
bash run_r2r/vlm_navigation.sh --help
```

## Ablation Studies

The ablation subsystem is isolated under `navigation_system/ablation/` and is designed to perform **subtractive ablations** without modifying the original main-system prompt templates in place.

Supported presets include:

- `landmark`
- `space_structure`
- `planning_reasoning` / `thinking_reasoning`
- `action_reasoning`
- `planning_action_reasoning` / `thinking_action_reasoning`
- `both`

Example commands:

```bash
bash run_r2r/vlm_navigation.sh --ablation landmark 1 100 260 4
bash run_r2r/vlm_navigation.sh --ablation planning_action_reasoning 1 100 260 4
bash run_r2r/vlm_navigation.sh --ablation thinking_reasoning 1 100 260 4
bash run_r2r/vlm_navigation.sh --runtime context_cache --ablation space_structure 1 100 260 4
```

Further details are documented in:

- `navigation_system/ablation/README.md`

## Results and Artifacts

The runtime resolves result directories in the following order:

1. `SPACEVLN_RESULTS_ROOT`
2. `PATHS.RESULTS_ROOT` in `navigation_system/config/system/00_runtime.yaml`
3. the default workspace-relative fallback under `result/vlnce/`

Standard runs are stored as:

```text
result/vlnce/<planner>__<executor>/
```

Context-cache runs are stored as:

```text
result/vlnce/<planner>__<executor>_cache/
```

Ablation runs are stored as:

```text
result/vlnce/ablation/<ablation_name>/<model_name>/
```

Stored artifacts may include:

- planner/action prompts and responses
- per-request `vlm_info.json`
- visualization frames and replay GIFs
- per-episode result logs
- ablation manifests for ablation runs

## Configuration Entry Points

For day-to-day experiments, the most important files are:

- `navigation_system/config/experiments/r2r_eval.yaml`
- `navigation_system/config/system/00_runtime.yaml`
- `navigation_system/config/system/10_detection_models.yaml`
- `navigation_system/config/system/20_space_sensor.yaml`
- `navigation_system/config/vlm/vlm_api_config.yaml`
- `navigation_system/config/vlm/vlm_api_config_context_cache.yaml`

See also:

- `navigation_system/config/README.md`
- `docs/ARCHITECTURE.md`

## Docker

The repository now includes a full-workspace Docker build at `../Dockerfile.spacevln`.

- build context: the `nav_ws/` workspace root
- bundled into the image: `data/`, `GroundingDINO/`, `habitat-lab/`, and `SpaceVLN/`
- default result root in-container: `/workspace/result`
- default result root on bare-metal runs from the same workspace: `nav_ws/result`
- API config templates are copied into place during image build; provide real keys via environment variables such as `DASHSCOPE_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`

Typical build command:

```bash
cd ..
docker buildx build --platform linux/amd64 -f Dockerfile.spacevln -t spacevln:v1 .
```

See `docs/dockerhub.md` for a complete build / smoke-test / push workflow.

## Reproducibility Notes

To keep the repository maintainable and sharable:

- no active API keys are committed;
- no user-specific absolute paths are committed;
- large datasets and checkpoints are not committed;
- local VLM configuration files are created from templates;
- `requirements.txt` is synchronized from a validated working environment instead of being reduced to a speculative minimal set.

## Acknowledgements

This codebase builds on the open-source efforts of the Habitat, VLN-CE, GroundingDINO, Grounded-Segment-Anything, NaVid-VLN-CE, and CA-Nav communities. Please consult the original repositories for licensing terms, model usage constraints, and dataset licenses before redistributing derived artifacts or checkpoints.
