"""SpaceVLN core package.

Top-level structure:
- `controller/`: navigation control flow and controller-local state
- `vlm/`: planner/executor/model-facing logic grouped by role
- `detection/`: detection and segmentation models
- `space/`: world-map, topology, landmarks, geometry, and textual space descriptions
- `render/`: map rendering, model-view rendering, and episode visualization
- `runtime/`: CLI/runtime orchestration, storage, and reporting
- `env/`: Habitat environment registration and construction
- `config/`: runtime config plus project-level config helpers/params
"""
