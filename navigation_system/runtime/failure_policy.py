"""Shared policy for failures that are not meaningful evaluation results."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


INITIAL_PLANNER_API_ERROR_REASON = "initial_planner_api_error"

NON_BEST_LOG_FAILURE_REASONS = {
    INITIAL_PLANNER_API_ERROR_REASON,
}


def result_failure_reason(result: Optional[Dict[str, Any]]) -> str:
    if not isinstance(result, dict):
        return ""
    return str(result.get("reason") or result.get("failure_reason") or "").strip().lower()


def should_skip_best_log_for_failure(result: Optional[Dict[str, Any]]) -> bool:
    return result_failure_reason(result) in NON_BEST_LOG_FAILURE_REASONS


def is_initial_planner_api_error_result(result: Optional[Dict[str, Any]]) -> bool:
    return result_failure_reason(result) == INITIAL_PLANNER_API_ERROR_REASON


def resolve_max_initial_planner_api_errors(
    value: Any = None,
    *,
    worker_count: int = 1,
) -> int:
    """Resolve the batch-level abort threshold for repeated initial planner API failures.

    A value <= 0 disables the guard. The default scales with worker count so a
    full first wave of parallel workers can fail without immediately aborting.
    """
    default = max(10, int(worker_count or 1) * 2)
    raw = value
    if raw is None:
        env_value = str(os.getenv("SPACEVLN_MAX_INITIAL_PLANNER_API_ERRORS", "") or "").strip()
        raw = env_value if env_value else default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default
