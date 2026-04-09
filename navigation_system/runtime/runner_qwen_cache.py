"""Qwen explicit-context-cache runtime wrapper built on the standard runner."""

from navigation_system.runtime.profiles import QWEN_CONTEXT_CACHE_RUNTIME_PROFILE
from navigation_system.runtime.runner import (
    build_arg_parser as build_standard_arg_parser,
    run_navigation_from_args as run_standard_navigation_from_args,
)


def build_arg_parser():
    return build_standard_arg_parser(profile=QWEN_CONTEXT_CACHE_RUNTIME_PROFILE)


def run_navigation_from_args(args):
    return run_standard_navigation_from_args(
        args,
        profile=QWEN_CONTEXT_CACHE_RUNTIME_PROFILE,
    )


__all__ = [
    "build_arg_parser",
    "run_navigation_from_args",
]
