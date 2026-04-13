"""Qwen explicit-context-cache ablation runtime wrapper."""

from navigation_system.ablation.runtime.runner import (
    build_arg_parser as build_standard_arg_parser,
    run_navigation_from_args as run_standard_navigation_from_args,
)
from navigation_system.ablation.runtime.profiles import (
    ABLATION_QWEN_CONTEXT_CACHE_RUNTIME_PROFILE,
)


def build_arg_parser():
    return build_standard_arg_parser(profile=ABLATION_QWEN_CONTEXT_CACHE_RUNTIME_PROFILE)


def run_navigation_from_args(args):
    return run_standard_navigation_from_args(
        args,
        profile=ABLATION_QWEN_CONTEXT_CACHE_RUNTIME_PROFILE,
    )


__all__ = [
    "build_arg_parser",
    "run_navigation_from_args",
]
