# SpaceVLN: Space-Guided Vision-Language Navigation

<div align="center">

**A ReAct-based VLM navigation system with dynamic semantic map guidance**

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![Habitat](https://img.shields.io/badge/Habitat-v0.1.7-orange.svg)](https://github.com/facebookresearch/habitat-lab)

[Paper](https://github.com/Charles-Donne/SpaceVLN) | [Web](https://github.com/Charles-Donne/SpaceVLN) | [Docs](docs/)

</div>

---

## 🎯 Overview

SpaceVLN implements a **hierarchical navigation framework** for Vision-Language Navigation in Continuous Environments (VLN-CE):

- **THOUGHT**: LLM decomposes tasks into subtasks with map context
- **ACT**: VLM executes actions guided by RGB + Detection + Local Map
- **REFLECT**: 360° scanning verifies progress and replans

**Key Features**:
- 🗺️ Dynamic semantic mapping with landmark tracking
- 🎯 Constraint-aware subtask decomposition
- 🔄 Adaptive replanning with 360° context
- 🤖 Open-vocabulary object detection (GroundedSAM)

---

## 🏗️ System Architecture

```
Instruction → LLM (THOUGHT)
                ↓
            Subtask + Landmark
                ↓
          Update Map Markers
                ↓
            VLM (ACT)
                ↓
    RGB + Detection + Local Map → Action
                ↓
          Execute & Update
                ↓
          Check Completion (REFLECT)
                ↓
        360° Scan → Verify → Replan
                ↓
            Loop until Goal
```

**Data Flow**:
```python
# THOUGHT Phase
llm_input = {
    "instruction": "Walk to the kitchen...",
    "images": [front_0°, left_90°, back_180°, right_270°],
    "maps": [global_map, local_map],
    "detected_objects": ["floor", "wall", "door", "table"]
}
llm_output = {
    "subtask_destination": "kitchen's table",
    "subtask_landmark": "table"  # → landmark_classes = ["table"]
}

# ACT Phase  
vlm_input = {
    "images": [rgb_view, detection_view, local_map],
    "subtask": "Move to doorway"
}
vlm_output = {
    "action": "MOVE_FORWARD",  # or TURN_LEFT/RIGHT/STOP
    "reasoning": "Safe path ahead, door 3m away"
}

# REFLECT Phase
verify_input = {
    "360_scan": [updated_maps, new_detections],
    "current_subtask": {
        "destination": "kitchen's table",
        "landmark": "table"
    }
}
verify_output = {
    "global_task_finish": False,
    "next_subtask": {...}
}
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Assume Habitat-Sim/Lab v0.1.7 already installed on server
conda activate your-habitat-env
cd CA-Nav-code
```

### 1. Configure API Keys
```bash
cd vlnce_baselines/config/api/

# Create one unified API config for both thinking/action models
cp vlm_api_config.yaml.template vlm_api_config.yaml
# Edit: add your provider, API key, and model names

# Example:
# provider: "dashscope"
# dashscope.api_key: "sk-..."
# dashscope.llm_model: "qwen-vl-max-latest"
# dashscope.vlm_model: "qwen-vl-plus-latest"
```

### 2. Run Navigation
```bash
# Single episode test
python run_vlm_navigation.py \
    --episode-id 0 \
    --split val_seen \
    --max-steps 500

# Batch evaluation
bash run_r2r/interactive_navigation.sh
```

### 3. Check Results
```bash
ls results/episode_0/
# ├── rgb/              # First-person views
# ├── detection/        # Object detection
# ├── global_map/       # Semantic maps
# ├── local_map/        # Local maps
# └── vlm/
#     ├── observations/ # Stitched views
#     ├── subtasks/     # Subtask logs
#     └── navigation.gif
```

---

## 📂 Project Structure

```
CA-Nav-code/
├── vlnce_baselines/
│   ├── vlm/                           # Core VLM modules
│   │   ├── thinking.py                # LLM planner (THOUGHT)
│   │   ├── action.py                  # VLM executor (ACT)
│   │   ├── prompts.py                 # LLM prompts
│   │   ├── action_prompt.py           # VLM prompts
│   │   └── api_client.py              # API wrapper
│   ├── vlm_navigation_controller.py   # Main controller
│   ├── detection/grounded_sam.py      # Object detection
│   ├── mapping/mapper.py              # Semantic mapping
│   └── visualization/visualizer.py    # Visualization
├── docs/
│   ├── 工作流程图.md                   # Detailed workflow
│   └── 建图机制说明.md                 # Mapping mechanism
└── requirements.txt
```

---

## ⚙️ Configuration

### Detection Classes
```python
# vlnce_baselines/config_system/constants.py
mapping_classes = [
    'floor', 'wall', 'door', 'bed', 'sofa', 'chair', 
    'table', 'desk', 'cabinet', 'tv', 'toilet', ...
]  # 25 common indoor objects

landmark_classes = []  # Dynamically updated per subtask
```

### Action Parameters
```python
TURN_ANGLE = 30        # degrees (12 steps = 360°)
MOVE_DISTANCE = 0.25   # meters
```

---

## 🔑 Key Mechanisms

### 1. Dynamic Landmark Tracking
```python
# Subtask 1
subtask_1 = {"subtask_landmark": "door"}
→ landmark_classes = ["door"]
→ Map: Purple "door" markers + Orange trajectory

# Subtask completes
mapper.clear_trajectory()

# Subtask 2  
subtask_2 = {"subtask_landmark": "sofa"}
→ landmark_classes = ["sofa"]  # Replaces "door"
→ Map: Purple "sofa" markers + New trajectory
```

### 2. Three-Image Action Guidance
```
VLM receives 3 images:
1. RGB View: First-person perspective
2. Detection View: Bounding boxes with labels
3. Local Map: Top-down semantic map
   • Green: Safe floor
   • Black: Obstacles (AVOID)
   • Orange: Trajectory
   • Blue: Field of view
```

### 3. 360° Contextual Awareness
```python
# Before each REFLECT phase
look_around_and_collect()  # 12 steps × 30° = 360°
# Captures: Front (0°), Left (90°), Back (180°), Right (270°)
# Updates: Semantic map + Detected objects
```

---

## 📊 Performance

| Metric | Value | Note |
|--------|-------|------|
| Token Usage | 60k-120k | per episode (GPT-4o) |
| API Calls | 50-100 | LLM + VLM |
| Cost | $0.01-0.05 | per episode |
| Runtime | 5-10 min | 50-100 actions |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: config.yaml` | Create from `.template` files |
| API call fails | Check API key and network |
| No landmarks on map | Verify `landmark_classes` updated |
| Trajectory accumulates | Confirm `clear_trajectory()` called |

---

## 🛠️ Local Environment Notes (2026-03-23)

This repository was adapted for local execution under `nav_ws` with relative paths and GPU-EGL fixes.

### Dependency versions in use

- `supervision==0.4.0`
- `gym==0.21.0`
- `numpy==1.23.5`

### Why `supervision==0.4.0` can fail by default

In this version, `sv.Detections` may not always expose a `mask` attribute in the same way as newer releases.
To avoid runtime crashes, `grounded_sam.py` now uses compatibility-safe access (`getattr(..., None)`).

### Runtime/path adjustments applied

- Relative dataset/model paths under `../data/...` (from SpaceVLN root)
- Launcher forces `spatial_agent` Python interpreter
- Headless EGL is pinned to NVIDIA vendor to avoid Mesa/dri2 context failures

For exact edited files and rationale, see [docs/LOCAL_CHANGES_2026-03-23.md](docs/LOCAL_CHANGES_2026-03-23.md).

---

## 📚 Documentation

- **[工作流程图.md](docs/工作流程图.md)**: Detailed workflow with diagrams
- **[建图机制说明.md](docs/建图机制说明.md)**: Semantic mapping mechanism
- **[系统说明文档.md](docs/系统说明文档.md)**: System architecture (Chinese)

---

## 🙏 Acknowledgments

Built upon:
- **CA-Nav**: Constraint-aware navigation framework ([Code](https://github.com/Chenkehan21/CA-Nav-code))
- **NaVid**: Video-based VLM navigation baseline ([Code](https://github.com/jzhzhang/NaVid-VLN-CE))
- **Habitat**: Simulation platform ([Code](https://github.com/facebookresearch/habitat-lab))
- **GroundedSAM**: Open-vocabulary detection ([Code](https://github.com/IDEA-Research/Grounded-Segment-Anything))

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

## 📧 Citation

To be added.

---

**System Status**: ✅ Production Ready | **Version**: v1.0.0 | **Updated**: 2025-12-10
