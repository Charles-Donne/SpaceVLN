"""Ablation-specific model-stack builders."""

from navigation_system.ablation.models.executor import AblationActionExecutor
from navigation_system.ablation.models.executor_context_cache import (
    AblationContextCacheActionExecutor,
)
from navigation_system.ablation.models.planner import AblationLLMPlanner
from navigation_system.ablation.models.planner_context_cache import (
    AblationContextCachePlanner,
)
from navigation_system.vlm.interfaces import NavigationModelStack
from navigation_system.vlm.model_stack_factory import (
    build_action_executor,
    build_planner,
)


def build_ablation_navigation_model_stack(
    *,
    config_path: str,
    action_space: str,
    turn_angle: float,
    move_distance: float,
    save_request_artifacts: bool,
) -> NavigationModelStack:
    planner = build_planner(
        AblationLLMPlanner,
        config_path=config_path,
        action_space=action_space,
        save_request_artifacts=save_request_artifacts,
        label="Ablation LLM Planner",
    )
    action_executor = build_action_executor(
        AblationActionExecutor,
        config_path=config_path,
        turn_angle=turn_angle,
        move_distance=move_distance,
        save_request_artifacts=save_request_artifacts,
        label="Ablation Action Executor",
    )
    return NavigationModelStack(
        planner=planner,
        action_executor=action_executor,
    )


def build_ablation_context_cache_navigation_model_stack(
    *,
    config_path: str,
    action_space: str,
    turn_angle: float,
    move_distance: float,
    save_request_artifacts: bool,
) -> NavigationModelStack:
    planner = build_planner(
        AblationContextCachePlanner,
        config_path=config_path,
        action_space=action_space,
        save_request_artifacts=save_request_artifacts,
        label="Ablation Cached LLM Planner",
    )
    action_executor = build_action_executor(
        AblationContextCacheActionExecutor,
        config_path=config_path,
        turn_angle=turn_angle,
        move_distance=move_distance,
        save_request_artifacts=save_request_artifacts,
        label="Ablation Cached Action Executor",
    )
    return NavigationModelStack(
        planner=planner,
        action_executor=action_executor,
    )


__all__ = [
    "build_ablation_navigation_model_stack",
    "build_ablation_context_cache_navigation_model_stack",
]
