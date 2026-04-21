"""Build pluggable planner/action model stacks for navigation runtime."""

from navigation_system.vlm.execution.vlnce.executor import ActionExecutor
from navigation_system.vlm.execution.vlnce.executor_context_cache import (
    ContextCacheActionExecutor,
)
from navigation_system.vlm.interfaces import NavigationModelStack
from navigation_system.vlm.planning.vlnce.planner import LLMPlanner
from navigation_system.vlm.planning.vlnce.planner_context_cache import ContextCachePlanner


def _configure_component(component, *, save_request_artifacts: bool):
    if component is None:
        return None
    component.set_request_artifact_saving(bool(save_request_artifacts))
    return component


def _build_planner(
    planner_cls,
    *,
    config_path: str,
    action_space: str,
    save_request_artifacts: bool,
    label: str,
):
    try:
        planner = planner_cls(config_path=config_path, action_space=action_space)
    except Exception as exc:
        print(f"[WARN] {label} init failed: {exc}")
        return None
    return _configure_component(
        planner,
        save_request_artifacts=save_request_artifacts,
    )


def _build_action_executor(
    executor_cls,
    *,
    config_path: str,
    turn_angle: float,
    move_distance: float,
    save_request_artifacts: bool,
    label: str,
):
    try:
        executor = executor_cls(
            config_path=config_path,
            turn_angle=turn_angle,
            move_distance=move_distance,
        )
    except Exception as exc:
        print(f"[WARN] {label} init failed: {exc}")
        return None
    return _configure_component(
        executor,
        save_request_artifacts=save_request_artifacts,
    )


def build_default_navigation_model_stack(
    *,
    config_path: str,
    action_space: str,
    turn_angle: float,
    move_distance: float,
    save_request_artifacts: bool,
) -> NavigationModelStack:
    """Build the standard planner + action executor stack."""
    planner = _build_planner(
        LLMPlanner,
        config_path=config_path,
        action_space=action_space,
        save_request_artifacts=save_request_artifacts,
        label="LLM Planner",
    )
    action_executor = _build_action_executor(
        ActionExecutor,
        config_path=config_path,
        turn_angle=turn_angle,
        move_distance=move_distance,
        save_request_artifacts=save_request_artifacts,
        label="Action Executor",
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
    """Build the explicit-cache planner + action executor stack."""
    planner = _build_planner(
        ContextCachePlanner,
        config_path=config_path,
        action_space=action_space,
        save_request_artifacts=save_request_artifacts,
        label="Cached LLM Planner",
    )
    action_executor = _build_action_executor(
        ContextCacheActionExecutor,
        config_path=config_path,
        turn_angle=turn_angle,
        move_distance=move_distance,
        save_request_artifacts=save_request_artifacts,
        label="Cached Action Executor",
    )
    return NavigationModelStack(
        planner=planner,
        action_executor=action_executor,
    )


__all__ = [
    "build_default_navigation_model_stack",
    "build_context_cache_navigation_model_stack",
]
