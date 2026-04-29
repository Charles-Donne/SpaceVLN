"""R2R-CE benchmark runtime for the VLNCE task family."""

from navigation_system.runtime.vlnce.r2r.runner import (
    build_arg_parser,
    run_navigation_from_args,
)

__all__ = [
    "build_arg_parser",
    "run_navigation_from_args",
]
