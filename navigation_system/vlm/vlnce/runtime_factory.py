"""Build pluggable planner/executor model stacks for navigation runtime."""

from navigation_system.vlm.execution.vlnce.executor import Executor
from navigation_system.vlm.execution.vlnce.executor_context_cache import (
    ContextCacheExecutor,
)
from navigation_system.vlm.interfaces import NavigationModelStack
from navigation_system.vlm.model_stack_factory import (
    build_executor,
    build_planner,
)
from navigation_system.vlm.planning.vlnce.planner import LLMPlanner
from navigation_system.vlm.planning.vlnce.planner_context_cache import ContextCachePlanner


def build_default_navigation_model_stack(
    *,
    config_path: str,
    action_space: str,
    turn_angle: float,
    move_distance: float,
    save_request_artifacts: bool,
) -> NavigationModelStack:
    """Build the standard planner + executor stack."""
    planner = build_planner(
        LLMPlanner,
        config_path=config_path,
        action_space=action_space,
        save_request_artifacts=save_request_artifacts,
        label="LLM Planner",
    )
    action_executor = build_executor(
        Executor,
        config_path=config_path,
        turn_angle=turn_angle,
        move_distance=move_distance,
        save_request_artifacts=save_request_artifacts,
        label="Executor",
    )
    return NavigationModelStack(
        planner=planner,
        action_executor=action_executor,
    )


def build_context_cache_navigation_model_stack(
    *,
    config_path: str,
    action_space: str,
    turn_angle: float,
    move_distance: float,
    save_request_artifacts: bool,
) -> NavigationModelStack:
    """Build the explicit-cache planner + executor stack."""
    planner = build_planner(
        ContextCachePlanner,
        config_path=config_path,
        action_space=action_space,
        save_request_artifacts=save_request_artifacts,
        label="Cached LLM Planner",
    )
    action_executor = build_executor(
        ContextCacheExecutor,
        config_path=config_path,
        turn_angle=turn_angle,
        move_distance=move_distance,
        save_request_artifacts=save_request_artifacts,
        label="Cached Executor",
    )
    return NavigationModelStack(
        planner=planner,
        action_executor=action_executor,
    )


__all__ = [
    "build_default_navigation_model_stack",
    "build_context_cache_navigation_model_stack",
]
