"""Build OVON-specific planner/action stacks."""

from navigation_system.vlm.execution.object_navigation.executor import OVONActionExecutor
from navigation_system.vlm.execution.object_navigation.executor_context_cache import (
    OVONContextCacheActionExecutor,
)
from navigation_system.vlm.planning.object_navigation.planner import OVONPlanner
from navigation_system.vlm.planning.object_navigation.planner_context_cache import (
    OVONContextCachePlanner,
)
from navigation_system.vlm.interfaces import NavigationModelStack
from navigation_system.vlm.model_stack_factory import (
    build_action_executor,
    build_planner,
)


def build_ovon_navigation_model_stack(
    *,
    config_path: str,
    action_space: str,
    turn_angle: float,
    move_distance: float,
    save_request_artifacts: bool,
) -> NavigationModelStack:
    planner = build_planner(
        OVONPlanner,
        config_path=config_path,
        action_space=action_space,
        save_request_artifacts=save_request_artifacts,
        label="OVON Planner",
    )
    action_executor = build_action_executor(
        OVONActionExecutor,
        config_path=config_path,
        turn_angle=turn_angle,
        move_distance=move_distance,
        save_request_artifacts=save_request_artifacts,
        label="OVON Action Executor",
    )
    return NavigationModelStack(
        planner=planner,
        action_executor=action_executor,
    )


def build_ovon_context_cache_navigation_model_stack(
    *,
    config_path: str,
    action_space: str,
    turn_angle: float,
    move_distance: float,
    save_request_artifacts: bool,
) -> NavigationModelStack:
    planner = build_planner(
        OVONContextCachePlanner,
        config_path=config_path,
        action_space=action_space,
        save_request_artifacts=save_request_artifacts,
        label="OVON Cached Planner",
    )
    action_executor = build_action_executor(
        OVONContextCacheActionExecutor,
        config_path=config_path,
        turn_angle=turn_angle,
        move_distance=move_distance,
        save_request_artifacts=save_request_artifacts,
        label="OVON Cached Action Executor",
    )
    return NavigationModelStack(
        planner=planner,
        action_executor=action_executor,
    )
