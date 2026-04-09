"""SpaceVLN core package.

Top-level structure:
- `controllers/`: internal base controller + exported VLM runtime controller
- `runtime/`: CLI/runtime orchestration
- `env/`: Habitat environment registration and construction
- `config/`: runtime config plus project-level config helpers/params
- `mapping/`: semantic/world-map maintenance
- `visualization/`: renderers and overlays
- `vlm/`: planner/executor/model-facing logic grouped by role
- `utils/`: narrow shared helpers
"""
