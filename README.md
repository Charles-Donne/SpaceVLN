# SpaceVLN

SpaceVLN is a research codebase for vision-and-language navigation in continuous Habitat environments. It integrates VLM-based hierarchical planning, action-level visual decision making, explicit space-structure reasoning, landmark-centric perception, qualitative rendering, batch evaluation, and controlled ablation experiments.

The system is built on the Habitat VLN-CE ecosystem and adapts components and design ideas from prior embodied navigation projects, especially [NaVid-VLN-CE](https://github.com/jzhzhang/NaVid-VLN-CE), [CA-Nav-code](https://github.com/Chenkehan21/CA-Nav-code), Habitat-Lab / Habitat-Sim, GroundingDINO, and Segment Anything.

## Abstract

## Highlights

- VLM planner and VLM action executor with standard and explicit-context-cache runtimes.
- Persistent spatial memory with spatial waypoints, area labels, landmark memory, and map-rendered reasoning evidence.
- Two task families under one Navigation Agent core: VLN-CE-style navigation and object navigation.
- Benchmarks are task plugins: R2R-CE and NavGBench under `vlnce`, OVON under `object_navigation`.
- GroundingDINO + SAM perception interface for open-vocabulary landmark grounding.
- Full run artifacts: prompts, responses, step views, maps, per-episode logs, reports, and optional replay visualizations.
- Isolated ablation subsystem under `navigation_system/ablation/` for subtractive studies without editing the main prompt templates in place.

## Repository Layout

```text
SpaceVLN/
├── navigation_system/
│   ├── controller/              # shared Navigation Agent controller
│   ├── env/                     # shared env adapter contract + task adapters
│   ├── runtime/                 # task/benchmark runners, storage, reports
│   ├── vlm/                     # prompts, API clients, model-stack factories
│   ├── space/                   # semantic maps, topology, landmarks, spatial formatting
│   ├── render/                  # thinking/action view rendering and episode visualization
│   ├── detection/               # GroundingDINO/SAM integration
│   └── ablation/                # ablation configs, prompt variants, wrappers
├── habitat_extensions/          # VLN-CE Habitat task, sensors, measures, config
├── run_navigation/              # canonical bash launchers
├── docs/                        # architecture / deployment notes
├── requirements.txt             # validated Python dependency snapshot
└── navigation_agent.py          # unified Python entrypoint:
                                  #   r2r | navgbench | ovon
```

Recommended workspace layout:

```text
nav_ws/
├── SpaceVLN/
├── vlnce/
│   └── habitat-lab/             # Habitat-Lab v0.1.7 source tree
├── GroundingDINO/               # GroundingDINO source tree
├── data/
│   ├── datasets/
│   ├── scene_datasets/
│   └── model/grounded_sam/
└── result/                      # default output root
```

The repository is configured to prefer workspace-relative paths. If datasets or results must live on another disk, keep the default config paths and create symlinks at `nav_ws/data` and/or `nav_ws/result`.

## Validated Software Stack

The VLN-CE runtime uses the legacy Habitat generation required by VLN-CE-style continuous navigation:

- Python `3.8`
- Habitat-Sim `0.1.7`
- Habitat-Lab `0.1.7`
- GroundingDINO commit `57535c5a79791cb76e36fdb64975271354f10251`
- Segment Anything from `facebookresearch/segment-anything`
- CUDA-enabled PyTorch matching your local CUDA driver/runtime

Newer Habitat releases are not drop-in replacements for this VLN-CE stack. Use the versions above unless you intend to port the code.

## Installation

The commands below assume the workspace root is `nav_ws/` and this repository is cloned as `nav_ws/SpaceVLN`.

### 1. Create the conda environment

```bash
conda create -n spacevln python=3.8 -y
conda activate spacevln
```

Optional but useful build tools:

```bash
conda install -c conda-forge cmake=3.14.0 ninja -y
```

### 2. Install PyTorch

Install a CUDA build that matches your machine. For example, for CUDA 12.1 wheels:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2+cu121 \
  torchvision==0.16.2+cu121 \
  torchaudio==2.1.2+cu121
```

If your CUDA runtime is different, replace the wheel index and versions accordingly.

### 3. Install Habitat-Sim 0.1.7

Recommended conda installation:

```bash
conda install -c aihabitat -c conda-forge \
  habitat-sim=0.1.7=py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7 -y
```

For unstable networks, download the matching `habitat-sim` package from the conda package page and install it locally:

```bash
conda install /path/to/habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2
```

### 4. Install Habitat-Lab 0.1.7

```bash
cd /path/to/nav_ws
mkdir -p vlnce
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git vlnce/habitat-lab
cd vlnce/habitat-lab
```

Install Habitat-Lab and Habitat-Baselines:

```bash
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```

If installation fails because of legacy TensorFlow constraints, remove/comment `tensorflow==1.13.1` from `habitat_baselines/rl/requirements.txt`, then rerun the commands above. Network-related failures are common with this old stack; rerunning the failed command is often sufficient.

### 5. Install GroundingDINO and Segment Anything

```bash
cd /path/to/nav_ws
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
pip install -e .
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
pip install nltk
```

#### Recommended GroundingDINO phrase-to-class refinement

For more stable phrase-to-class mapping, refine `GroundingDINO/groundingdino/util/inference.py` by replacing the original `phrases2classes` implementation with an edit-distance fallback:

```python
from nltk.metrics import edit_distance

@staticmethod
def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
    class_ids = []
    for phrase in phrases:
        if phrase in classes:
            class_ids.append(classes.index(phrase))
        else:
            distances = np.array([edit_distance(phrase, class_id) for class_id in classes])
            idx = np.argmin(distances)
            class_ids.append(idx)
    return np.array(class_ids)
```

The original implementation returns `None` for unmatched phrases; the refined version maps each phrase to the nearest class name and produces more stable open-vocabulary detection outputs.

### 6. Install SpaceVLN dependencies

```bash
cd /path/to/nav_ws/SpaceVLN
pip install -r requirements.txt
```

## Data and Checkpoints

Expected dataset/checkpoint layout:

```text
nav_ws/data/
├── datasets/
│   ├── R2R_VLNCE_v1-3_preprocessed/
│   │   ├── val_unseen.json.gz
│   │   └── val_unseen_gt.json.gz
│   └── ovon/hm3d/v1/val_unseen/val_unseen_hard.json.gz
├── scene_datasets/
└── model/grounded_sam/
    ├── GroundingDINO_SwinT_OGC.py
    ├── groundingdino_swint_ogc.pth
    ├── sam_vit_h_4b8939.pth
    └── repvit_sam.pt
```

Important config files:

- VLN-CE task/data paths: `habitat_extensions/config/spacevln_task.yaml`
- OVON dataset path: `navigation_system/config/experiments/object_navigation/ovon_val_unseen_eval.yaml`
- detection checkpoint paths: `navigation_system/config/system/10_detection_models.yaml`
- default result path policy: `navigation_system/config/system/00_runtime.yaml`

If `repvit_sam.pt` is unavailable, set `USE_REPVIT_SAM: false` in `navigation_system/config/system/10_detection_models.yaml`.

## VLM API Configuration

Runtime credentials are not committed. Create local config files from templates:

```bash
cp navigation_system/config/vlm/vlm_api_config.yaml.template \
   navigation_system/config/vlm/vlm_api_config.yaml
```

Use the unified `vlm_api_config.yaml` for both standard and `context_cache` runtimes.
The same file now contains the `qwen_context_cache` switch.

The templates can read provider keys from environment variables such as:

- `DASHSCOPE_API_KEY`
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`

## Running Experiments

### VLN-CE

Standard runtime:

```bash
bash run_navigation/r2rce.sh 1 10 260 4
```

Explicit-context-cache runtime:

```bash
bash run_navigation/r2rce.sh --runtime context_cache 1 10 260 4
```

Common argument pattern:

```text
start_episode num_episodes max_steps parallel_workers
```

Examples:

```bash
bash run_navigation/r2rce.sh 832
bash run_navigation/r2rce.sh 832 300
bash run_navigation/r2rce.sh random 20 260 all 4
bash run_navigation/r2rce.sh --help
```

### OVON / Object Navigation

```bash
bash run_navigation/object_navigation.sh --runtime context_cache 1 10 500 4
```

Examples:

```bash
bash run_navigation/object_navigation.sh --episode-id 1074
bash run_navigation/object_navigation.sh --episode-ids 1074,1081
bash run_navigation/object_navigation.sh --run-config navigation_system/config/experiments/object_navigation/ovon_val_unseen_eval.yaml --num-episodes 10
bash run_navigation/object_navigation.sh --help
```

### NavGBench / GN-Bench

Run the Navigation Agent on the NavGBench InteriorGS loop:

```bash
bash run_navigation/navgbench.sh --dry-run --start-sample 1 --num-episodes 3
bash run_navigation/navgbench.sh 1 1 300
bash run_navigation/navgbench.sh --simple-instruction 1 10 300
bash run_navigation/navgbench.sh list 0864_841787_156 300
```

The runner expects `../Nav-GBench` by default, or `NAVGBENCH_ROOT` if set. It
prefers a conda environment named `gn_bench` when it can find one; override with
`SPACEVLN_NAVGBENCH_PYTHON` or `PYTHON_BIN` if needed. It wraps GN-Bench
episodes with the same small env adapter contract used by VLNCE/OVON,
enables `RGB_SENSOR` + `DEPTH_SENSOR`, keeps the agent's model-facing turn action
at `30deg`, expands each turn into two GN-Bench `15deg` primitives, and writes
SpaceVLN logs under
`nav_ws/result/navgbench/<complex|simple|moving>/<planner>__<executor>/` plus
NavGBench-style metric JSON under `navgbench_log/`, so different instruction
families keep separate best logs and reports.

NavGBench rendering dependencies live in the NavGBench env. SpaceVLN's normal
`spacevln` env does not need `GN_Bench` installed. For in-process NavGBench
evaluation, install only the SpaceVLN agent-side dependencies needed by the
controller into `gn_bench` and keep Habitat/habitat-sim out of that env.

### Ablation Studies

Ablation presets are isolated under `navigation_system/ablation/`.

```bash
bash run_navigation/r2rce.sh --ablation landmark 1 100 260 4
bash run_navigation/r2rce.sh --ablation space_structure 1 100 260 4
bash run_navigation/r2rce.sh --ablation landmark_space_structure 1 100 260 4
bash run_navigation/r2rce.sh --ablation planning_action_reasoning 1 100 260 4
bash run_navigation/r2rce.sh --ablation planning_reasoning_no_progress 1 100 260 4
bash run_navigation/r2rce.sh --runtime context_cache --ablation space_structure 1 100 260 4
```

Supported preset names include:

- `landmark`
- `space_structure`
- `landmark_space_structure`
- `planning_reasoning`
- `action_reasoning`
- `planning_action_reasoning`
- `planning_reasoning_no_progress`
- `planning_action_reasoning_no_progress`

See `navigation_system/ablation/configs/` and `navigation_system/ablation/templates/` for preset definitions and prompt variants.

### Reporting Existing Results

```bash
bash run_navigation/report_r2rce.sh 1 100 qwen3.5-plus__qwen3.5-flash_cache
bash run_navigation/report_r2rce.sh all all qwen3.5-plus__qwen3.5-flash_cache
bash run_navigation/report_r2rce.sh --start-id all --end-id all --results all
```

## Result Layout

Default output root is `nav_ws/result`.

```text
result/
├── vlnce/
│   ├── <planner>__<executor>/
│   ├── <planner>__<executor>_cache/
│   └── ablation/<preset>/<model_name>/
├── ovon/
│   ├── <planner>__<executor>/
│   └── <planner>__<executor>_cache/
└── navgbench/
    └── <planner>__<executor>/
```

Per-episode artifacts may include:

- planner/executor prompts and responses;
- rendered thinking views and action views;
- global/local maps and visual summaries;
- `vlm_info.json` request metadata;
- per-episode metrics and logs;
- cache reports under `reports/cache/` for context-cache runs.

To move data/results to another disk while keeping config portable, use symlinks:

```bash
bash run_navigation/setup_storage_symlinks.sh --disk-root /abs/path/to/nav_ws_storage --both --backup-existing
```

## Docker

A full-workspace Dockerfile is provided at the workspace root:

```text
nav_ws/Dockerfile.spacevln
```

It is intended to bundle `SpaceVLN/`, `vlnce/habitat-lab/`, `GroundingDINO/`, and optionally `data/` into one reproducible image.

Build from `nav_ws/`:

```bash
cd /path/to/nav_ws
docker buildx build --platform linux/amd64 -f Dockerfile.spacevln -t spacevln:v1 .
```

Run interactively:

```bash
docker run --rm -it --gpus all spacevln:v1 bash
```

If data/results are not baked into the image, mount them explicitly:

```bash
docker run --rm -it --gpus all \
  -v /path/to/nav_ws/data:/workspace/data \
  -v /path/to/nav_ws/result:/workspace/result \
  spacevln:v1 bash
```

Provide API keys via environment variables or create local config files from the templates inside the container.

## Citation and Acknowledgements

This repository builds on open-source embodied navigation and perception projects, including:

- [NaVid-VLN-CE](https://github.com/jzhzhang/NaVid-VLN-CE)
- [CA-Nav-code](https://github.com/Chenkehan21/CA-Nav-code)
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)

Please consult the upstream repositories for licenses, dataset terms, and model usage constraints before redistribution.
