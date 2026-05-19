"""Shared helpers for assembling planner/executor model stacks."""

from __future__ import annotations

from typing import Any


def configure_component(component: Any, *, save_request_artifacts: bool) -> Any:
    if component is None:
        return None
    component.set_request_artifact_saving(bool(save_request_artifacts))
    return component


def build_planner(
    planner_cls: type,
    *,
    config_path: str,
    action_space: str,
    save_request_artifacts: bool,
    label: str,
) -> Any:
    try:
        planner = planner_cls(config_path=config_path, action_space=action_space)
    except Exception as exc:
        raise RuntimeError(f"{label} init failed: {exc}") from exc
    return configure_component(
        planner,
        save_request_artifacts=save_request_artifacts,
    )


def build_executor(
    executor_cls: type,
    *,
    config_path: str,
    turn_angle: float,
    move_distance: float,
    save_request_artifacts: bool,
    label: str,
) -> Any:
    try:
        executor = executor_cls(
            config_path=config_path,
            turn_angle=turn_angle,
            move_distance=move_distance,
        )
    except Exception as exc:
        raise RuntimeError(f"{label} init failed: {exc}") from exc
    return configure_component(
        executor,
        save_request_artifacts=save_request_artifacts,
    )


__all__ = [
    "build_executor",
    "build_planner",
    "configure_component",
]
