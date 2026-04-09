"""Runtime entrypoints and episode orchestration."""

from importlib import import_module
from typing import Any

__all__ = [
    "build_arg_parser",
    "build_results_arg_parser",
    "generate_results_report",
    "run_navigation_from_args",
    "run_results_report_from_args",
]


def __getattr__(name: str) -> Any:
    if name in {"build_arg_parser", "run_navigation_from_args"}:
        return getattr(import_module("navigation_system.runtime.runner"), name)
    if name in {
        "build_results_arg_parser",
        "generate_results_report",
        "run_results_report_from_args",
    }:
        return getattr(import_module("navigation_system.runtime.results_report"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
