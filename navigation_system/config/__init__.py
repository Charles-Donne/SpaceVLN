"""Configuration package for SpaceVLN.

Layout:
- `runtime/`: runtime Habitat/YACS config assembly
- `core/`: project-level static params, constants, and config helpers
- `experiments/`: experiment YAML overrides
- `api/`: canonical API config templates
"""

from navigation_system.config.runtime.default import get_config
from navigation_system.config.core import ConfigHelper, CategoryConfig, create_category_config

__all__ = [
    "get_config",
    "ConfigHelper",
    "CategoryConfig",
    "create_category_config",
]
