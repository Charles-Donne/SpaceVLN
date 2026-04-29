"""SpaceVLN core package.

Top-level structure:
- `controller/`: shared Navigation Agent control flow
- `vlm/`: shared model-stack helpers plus task-specific planner/executor logic
- `detection/`: detection and segmentation models
- `space/`: world-map, topology, landmarks, geometry, and textual space descriptions
- `render/`: map rendering, model-view rendering, and episode visualization
- `runtime/`: task/benchmark runtime orchestration, storage, and reporting
- `env/`: shared adapter contract plus task/benchmark environment adapters
- `config/`: runtime config plus project-level config helpers/params
"""
