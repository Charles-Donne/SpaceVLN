"""Build NavGBench-specific VLNCE model stacks."""

from navigation_system.vlm.execution.vlnce.executor import ActionExecutor
from navigation_system.vlm.execution.vlnce.executor_context_cache import (
    ContextCacheActionExecutor,
)
from navigation_system.vlm.interfaces import NavigationModelStack
from navigation_system.vlm.model_stack_factory import (
    build_action_executor,
    configure_component,
)
from navigation_system.vlm.planning.vlnce.navgbench import (
    NavGBenchContextCachePlanner,
    NavGBenchPlanner,
)


def _build_navgbench_planner(
    planner_cls: type,
    *,
    config_path: str,
    action_space: str,
    instruction_mode: str,
    save_request_artifacts: bool,
    label: str,
):
    try:
        planner = planner_cls(
            config_path=config_path,
            action_space=action_space,
            instruction_mode=instruction_mode,
        )
    except Exception as exc:
        print(f"[WARN] {label} init failed: {exc}")
        return None
    return configure_component(
        planner,
        save_request_artifacts=save_request_artifacts,
    )


def build_navgbench_navigation_model_stack(
    *,
    config_path: str,
    action_space: str,
    turn_angle: float,
    move_distance: float,
    save_request_artifacts: bool,
    instruction_mode: str = "complex",
) -> NavigationModelStack:
    planner = _build_navgbench_planner(
        NavGBenchPlanner,
        config_path=config_path,
        action_space=action_space,
        instruction_mode=instruction_mode,
        save_request_artifacts=save_request_artifacts,
        label="NavGBench Planner",
    )
    action_executor = build_action_executor(
        ActionExecutor,
        config_path=config_path,
        turn_angle=turn_angle,
        move_distance=move_distance,
        save_request_artifacts=save_request_artifacts,
        label="NavGBench Action Executor",
    )
    return NavigationModelStack(
        planner=planner,
        action_executor=action_executor,
    )


def build_navgbench_context_cache_navigation_model_stack(
    *,
    config_path: str,
    action_space: str,
    turn_angle: float,
    move_distance: float,
    save_request_artifacts: bool,
    instruction_mode: str = "complex",
) -> NavigationModelStack:
    planner = _build_navgbench_planner(
        NavGBenchContextCachePlanner,
        config_path=config_path,
        action_space=action_space,
        instruction_mode=instruction_mode,
        save_request_artifacts=save_request_artifacts,
        label="NavGBench Cached Planner",
    )
    action_executor = build_action_executor(
        ContextCacheActionExecutor,
        config_path=config_path,
        turn_angle=turn_angle,
        move_distance=move_distance,
        save_request_artifacts=save_request_artifacts,
        label="NavGBench Cached Action Executor",
    )
    return NavigationModelStack(
        planner=planner,
        action_executor=action_executor,
    )


__all__ = [
    "build_navgbench_context_cache_navigation_model_stack",
    "build_navgbench_navigation_model_stack",
]
