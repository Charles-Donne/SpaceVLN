"""SpaceVLN baseline package.

Top-level structure:
- `config/`: runtime config plus project-level config helpers/params
- `common/`: cross-cutting shared utilities
- `mapping/`: semantic/world-map maintenance
- `visualization/`: renderers and overlays
- `vlm/`: planner/executor/model-facing logic
"""

from vlnce_baselines.common import environments
