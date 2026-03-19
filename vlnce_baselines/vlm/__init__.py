"""
VLM subsystem.

This package contains model-facing logic:
- `api_client`: shared API/YAML loading layer
- `thinking` / `action`: planner and executor
- `prompts` / `action_prompt`: prompt templates
- `navigation_visualizer` / `save_manager`: model I/O artifacts

Canonical API YAML files now live under `vlnce_baselines/config/api/`.
Old `vlnce_baselines/vlm/*.yaml` paths are resolved for backward compatibility
when users still pass them explicitly.
"""

from vlnce_baselines.vlm.api_client import APIConfig, BaseAPIClient
from vlnce_baselines.vlm.thinking import LLMPlanner
from vlnce_baselines.vlm.action import ActionExecutor
from vlnce_baselines.vlm.navigation_visualizer import NavigationVisualizer
from vlnce_baselines.vlm.action_parser import ActionParser
from vlnce_baselines.vlm.save_manager import SaveManager

__all__ = [
    'APIConfig',
    'BaseAPIClient', 
    'LLMPlanner',
    'ActionExecutor',
    'NavigationVisualizer',
    'ActionParser',
    'SaveManager',
]
