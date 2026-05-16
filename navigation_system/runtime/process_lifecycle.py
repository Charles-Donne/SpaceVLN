"""Process and resource cleanup helpers for benchmark launchers."""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
from typing import Callable, Optional


def env_flag_enabled(name: str, default: bool = False) -> bool:
    raw_value = str(os.getenv(name, "") or "").strip().lower()
    if not raw_value:
        return bool(default)
    return raw_value in {"1", "true", "yes", "on", "y"}


def env_float(name: str, default: float) -> float:
    raw_value = str(os.getenv(name, "") or "").strip()
    if not raw_value:
        return float(default)
    try:
        return float(raw_value)
    except ValueError:
        return float(default)


def close_with_timeout(
    close_fn: Callable[[], object],
    *,
    label: str = "resource",
    timeout_s: Optional[float] = None,
) -> bool:
    """Run a close callback without allowing it to block process shutdown forever."""
    timeout = env_float(
        "SPACEVLN_CLOSE_TIMEOUT_S",
        15.0 if timeout_s is None else float(timeout_s),
    )
    if timeout <= 0:
        try:
            close_fn()
            return True
        except Exception as exc:
            print(f"⚠️  Failed to close {label}: {exc}", flush=True)
            return False

    errors: list[BaseException] = []

    def _runner() -> None:
        try:
            close_fn()
        except BaseException as exc:  # pragma: no cover - defensive cleanup path
            errors.append(exc)

    thread_name = "spacevln-close-" + "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "-"
        for ch in str(label or "resource").lower()
    )[:48]
    thread = threading.Thread(target=_runner, name=thread_name, daemon=True)
    thread.start()
    thread.join(timeout=float(timeout))
    if thread.is_alive():
        print(
            f"⚠️  Timed out closing {label} after {float(timeout):.1f}s; "
            "continuing shutdown.",
            flush=True,
        )
        return False
    if errors:
        print(f"⚠️  Failed to close {label}: {errors[0]}", flush=True)
        return False
    return True


def terminate_active_children(timeout_s: Optional[float] = None) -> None:
    """Best-effort cleanup for multiprocessing children before a forced CLI exit."""
    timeout = max(
        0.0,
        env_float(
            "SPACEVLN_CHILD_EXIT_TIMEOUT_S",
            2.0 if timeout_s is None else timeout_s,
        ),
    )
    try:
        children = list(multiprocessing.active_children())
    except Exception:
        return
    if not children:
        return
    for child in children:
        try:
            if child.is_alive():
                child.terminate()
        except Exception:
            pass
    for child in children:
        try:
            child.join(timeout=timeout)
        except Exception:
            pass
    for child in children:
        try:
            if child.is_alive() and hasattr(child, "kill"):
                child.kill()
        except Exception:
            pass


def exit_process(status: int) -> None:
    """Exit the CLI without waiting on third-party atexit hooks that may hang."""
    code = int(status or 0)
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    if env_flag_enabled("SPACEVLN_FORCE_PROCESS_EXIT", default=True):
        terminate_active_children()
        os._exit(code)
    raise SystemExit(code)
