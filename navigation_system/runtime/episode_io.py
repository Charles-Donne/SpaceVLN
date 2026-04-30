"""Episode-level console output, result lookup, and stdout redirection helpers."""

import contextlib
import json
import os
import sys
from typing import Any, Dict, Optional

from navigation_system.runtime.storage.artifacts import (
    get_episode_detail_dir,
)


_NORMAL_FAILURE_REASONS = {
    "episode_done_without_success",
    "max_episode_steps_reached",
    "goal_not_reached",
    "episode_budget_exhausted_before_replan",
}


def get_episode_records_log_path(
    results_dir: str,
    episode_id: int,
    *,
    entry_kind: str = "episode",
) -> str:
    episode_dir = get_episode_detail_dir(results_dir, episode_id, entry_kind=entry_kind)
    records_dir = os.path.join(episode_dir, "records")
    os.makedirs(records_dir, exist_ok=True)
    entry_prefix = str(entry_kind or "episode").strip() or "episode"
    return os.path.join(records_dir, f"{entry_prefix}_{int(episode_id)}.log")


def get_episode_result_path(
    results_dir: str,
    episode_id: int,
    *,
    entry_kind: str = "episode",
) -> str:
    episode_dir = get_episode_detail_dir(results_dir, episode_id, entry_kind=entry_kind)
    records_dir = os.path.join(episode_dir, "records")
    os.makedirs(records_dir, exist_ok=True)
    return os.path.join(records_dir, "result.json")


def save_episode_stdout_log_enabled(config) -> bool:
    output_cfg = getattr(config, "OUTPUT", None)
    log_cfg = getattr(output_cfg, "LOGS", None)
    return bool(getattr(log_cfg, "SAVE_EPISODE_STDOUT", False))


@contextlib.contextmanager
def redirect_process_output_to_file(log_path: str, mode: str = "w"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    sys.stdout.flush()
    sys.stderr.flush()

    log_file = open(log_path, mode, encoding="utf-8")
    try:
        os.dup2(log_file.fileno(), stdout_fd)
        os.dup2(log_file.fileno(), stderr_fd)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        log_file.close()


@contextlib.contextmanager
def redirect_process_output_to_null():
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    sys.stdout.flush()
    sys.stderr.flush()

    with open(os.devnull, "w", encoding="utf-8") as sink:
        try:
            os.dup2(sink.fileno(), stdout_fd)
            os.dup2(sink.fileno(), stderr_fd)
            yield
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def load_json_if_exists(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def should_suppress_normal_failure_reason(
    *,
    status: str,
    reason: str,
    error: str,
) -> bool:
    return bool(
        status == "FAIL"
        and not str(error or "").strip()
        and str(reason or "").strip().lower() in _NORMAL_FAILURE_REASONS
    )


def is_normal_evaluation_failure(result: Dict[str, Any]) -> bool:
    """A completed evaluation episode can fail SR without being a runtime error."""
    if bool((result or {}).get("success", False)):
        return False
    if str((result or {}).get("error") or "").strip():
        return False
    reason = str((result or {}).get("reason") or "").strip().lower()
    return reason in _NORMAL_FAILURE_REASONS


def is_abnormal_episode_failure(result: Dict[str, Any]) -> bool:
    """Return True only for failures that should make the batch look broken."""
    if bool((result or {}).get("success", False)):
        return False
    return not is_normal_evaluation_failure(result or {})


def extract_episode_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    candidate_paths = [
        str(result.get("result_detail_file") or "").strip(),
        str(result.get("result_file") or "").strip(),
    ]
    for path in candidate_paths:
        if not path:
            continue
        metrics = load_json_if_exists(path)
        if metrics:
            break

    if not metrics:
        return {}

    return {
        "sr": metrics.get("sr", metrics.get("success", 0)),
        "osr": metrics.get("osr", metrics.get("oracle_success", 0)),
        "ne": metrics.get("ne", metrics.get("distance_to_goal", -1.0)),
        "spl": metrics.get("spl", 0.0),
        "ndtw": metrics.get("ndtw", 0.0),
    }


def build_episode_console_summary(
    *,
    episode_id: int,
    index: int,
    total: int,
    result: Dict[str, Any],
    metrics: Dict[str, Any],
    worker_index: int = 0,
    worker_count: int = 0,
    sample_index: Optional[int] = None,
) -> str:
    if bool(result.get("success", False)):
        status = "OK"
    elif result.get("recorded") is False:
        status = "UNRECORDED"
    else:
        status = "FAIL"
    steps = int(result.get("steps", result.get("total_steps", 0)) or 0)
    reason = str(result.get("reason") or "").strip()
    error = str(result.get("error") or "").strip()
    if status == "FAIL" and not reason and not error:
        reason = "failed_before_first_step" if steps <= 0 else "episode_failed"

    parts = [
        (
            f"[W{worker_index}/{worker_count} {index}/{total}]"
            if worker_index > 0 and worker_count > 0
            else f"[{index}/{total}]"
        ),
        (f"Sample {int(sample_index)}" if sample_index is not None else None),
        f"Episode {episode_id}",
        status,
        f"steps={steps}",
    ]
    parts = [part for part in parts if part is not None]

    if metrics:
        metric_specs = (
            ("dtg", "DTG", "distance"),
            ("ne", "NE", "distance"),
            ("osr", "OSR", "int"),
            ("sr", "SR", "int"),
            ("spl", "SPL", "float"),
            ("soft_spl", "SoftSPL", "float"),
            ("ndtw", "nDTW", "float"),
        )
        for key, label, value_type in metric_specs:
            if key not in metrics:
                continue
            value = metrics.get(key)
            if value is None:
                continue
            try:
                if value_type == "distance":
                    parts.append(f"{label}={float(value):.3f}m")
                elif value_type == "int":
                    parts.append(f"{label}={int(value)}")
                else:
                    parts.append(f"{label}={float(value):.4f}")
            except Exception:
                continue

    if reason and not should_suppress_normal_failure_reason(
        status=status,
        reason=reason,
        error=error,
    ):
        parts.append(f"reason={reason}")
    if error:
        parts.append(f"error={error}")

    return " | ".join(parts)


def build_episode_start_summary(
    *,
    episode_id: int,
    index: int,
    total: int,
    worker_index: int = 0,
    worker_count: int = 0,
    sample_index: Optional[int] = None,
) -> str:
    parts = [
        (
            f"[W{worker_index}/{worker_count} {index}/{total}]"
            if worker_index > 0 and worker_count > 0
            else f"[{index}/{total}]"
        ),
        (f"Sample {int(sample_index)}" if sample_index is not None else None),
        f"Episode {episode_id}",
        "START",
    ]
    return " | ".join([part for part in parts if part is not None])


__all__ = [
    "build_episode_console_summary",
    "build_episode_start_summary",
    "extract_episode_metrics",
    "get_episode_records_log_path",
    "get_episode_result_path",
    "load_json_if_exists",
    "redirect_process_output_to_file",
    "redirect_process_output_to_null",
    "save_episode_stdout_log_enabled",
    "should_suppress_normal_failure_reason",
]
