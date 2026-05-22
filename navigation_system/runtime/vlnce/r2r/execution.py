"""Runtime execution helpers: config loading, controller creation, and episode jobs."""

import argparse
import atexit
import contextlib
import copy
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import shutil
import signal
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from navigation_system.config import ConfigHelper, get_config
from navigation_system.config.runtime.default import apply_runtime_derived_fields
from navigation_system.controller.agent.controller import NavigationAgentController
from navigation_system.runtime.episode_io import (
    build_episode_console_summary,
    build_episode_start_summary,
    extract_episode_metrics,
    get_episode_records_log_path,
    get_episode_result_path,
    redirect_process_output_to_file,
    redirect_process_output_to_null,
    save_episode_stdout_log_enabled,
)
from navigation_system.runtime.failure_policy import (
    INITIAL_PLANNER_API_ERROR_REASON,
    is_initial_planner_api_error_result,
    resolve_max_initial_planner_api_errors,
    should_suppress_persistent_record_for_failure,
)
from navigation_system.runtime.output_policy import (
    apply_output_policy_to_config,
    build_output_job_fields,
    build_output_namespace_kwargs,
)
from navigation_system.runtime.process_lifecycle import close_with_timeout, env_float
from navigation_system.runtime.storage.artifacts import (
    SaveManager,
    get_episode_detail_dir,
    get_episode_log_path,
)
from navigation_system.runtime.vlnce.profiles import (
    NavigationRuntimeProfile,
    STANDARD_RUNTIME_PROFILE,
    resolve_runtime_profile,
)
from navigation_system.vlm.api.qwen_context_cache_client import (
    validate_qwen_context_cache_api_config,
)
from navigation_system.vlm.api.api_client import (
    resolve_api_config_path,
    resolve_results_dir_path,
    resolve_results_root_path,
)

INITIAL_FAILURE_RETRY_REASONS = {
    "initial_subtask_failed",
    "initial_lookaround_failed",
    "initial_planner_no_response",
    "initial_planner_timeout",
}

UNRECORDED_INITIAL_FAILURE_REASONS = {
    "initial_subtask_failed",
    "initial_lookaround_failed",
    "initial_planner_no_response",
    "initial_planner_timeout",
    INITIAL_PLANNER_API_ERROR_REASON,
}

DEFAULT_INITIAL_FAILURE_MAX_ATTEMPTS = 3

_EPISODE_TRANSFER_FUTURES: List[concurrent.futures.Future] = []
_EPISODE_TRANSFER_LOCK = threading.Lock()
_EPISODE_TRANSFER_SEMAPHORE: Optional[threading.BoundedSemaphore] = None
_EPISODE_TRANSFER_SEMAPHORE_LIMIT = 0


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(1, parsed)


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    raw_value = str(os.getenv(name, "") or "").strip().lower()
    if not raw_value:
        return bool(default)
    return raw_value in {"1", "true", "yes", "on", "y"}


def _resolve_initial_failure_max_attempts(args: argparse.Namespace) -> int:
    cli_value = getattr(args, "initial_failure_max_attempts", None)
    if cli_value is not None:
        return _positive_int(cli_value, DEFAULT_INITIAL_FAILURE_MAX_ATTEMPTS)
    env_value = str(os.getenv("SPACEVLN_INITIAL_FAILURE_MAX_ATTEMPTS", "") or "").strip()
    if env_value:
        return _positive_int(env_value, DEFAULT_INITIAL_FAILURE_MAX_ATTEMPTS)
    return DEFAULT_INITIAL_FAILURE_MAX_ATTEMPTS


def _snapshot_file(path: str) -> Optional[bytes]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _restore_file(path: str, snapshot: Optional[bytes]) -> None:
    if not path:
        return
    if snapshot is None:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(snapshot)
    except Exception:
        pass


def _snapshot_episode_result_artifacts(
    *,
    result_path: str,
    best_log_path: str,
    stdout_log_path: str,
) -> Dict[str, Optional[bytes]]:
    return {
        "result_path": _snapshot_file(result_path),
        "best_log_path": _snapshot_file(best_log_path),
        "stdout_log_path": _snapshot_file(stdout_log_path),
    }


def _restore_episode_result_artifacts(
    snapshots: Dict[str, Optional[bytes]],
    *,
    result_path: str,
    best_log_path: str,
    stdout_log_path: str,
) -> None:
    _restore_file(result_path, snapshots.get("result_path"))
    _restore_file(best_log_path, snapshots.get("best_log_path"))
    _restore_file(stdout_log_path, snapshots.get("stdout_log_path"))


def _should_suppress_initial_failure_record(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    if bool(result.get("success", False)):
        return False
    if str(result.get("error") or "").strip():
        return False
    reason = str(result.get("reason") or "").strip().lower()
    return reason in UNRECORDED_INITIAL_FAILURE_REASONS


def _mark_result_unrecorded(result: Dict[str, Any]) -> None:
    result["recorded"] = False
    result["record_suppressed_reason"] = str(result.get("reason") or "").strip()
    result["result_file"] = ""
    result["result_detail_file"] = ""


def _format_exception_message(exc: BaseException) -> str:
    exc_type = type(exc).__name__
    exc_text = str(exc).strip()
    return f"{exc_type}: {exc_text}" if exc_text else exc_type


def _parallel_worker_initializer() -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(_positive_int(os.getenv("SPACEVLN_TORCH_NUM_THREADS", "1"), 1))
        torch.set_num_interop_threads(
            _positive_int(os.getenv("SPACEVLN_TORCH_INTEROP_THREADS", "1"), 1)
        )
    except Exception:
        pass


def _shutdown_parallel_executor(
    executor: concurrent.futures.ProcessPoolExecutor,
    *,
    interrupted: bool,
) -> None:
    if interrupted:
        processes = list((getattr(executor, "_processes", None) or {}).values())
        for process in processes:
            try:
                if process is not None and process.is_alive():
                    process.terminate()
            except Exception:
                pass
        for process in processes:
            try:
                if process is not None:
                    process.join(timeout=0.5)
            except Exception:
                pass
    try:
        executor.shutdown(wait=not interrupted)
    except Exception:
        if not interrupted:
            raise


def load_runtime_config(
    args: argparse.Namespace,
    profile: NavigationRuntimeProfile = STANDARD_RUNTIME_PROFILE,
):
    if str(getattr(profile, "name", "") or "") == "context_cache":
        validate_qwen_context_cache_api_config(args.vlm_api_config)

    config = get_config(args.exp_config, [])
    paths_config = config.PATHS
    resolved_results_dir = resolve_results_dir_path(
        str(getattr(args, "results_dir", "") or "").strip()
    )
    configured_results_root = str(getattr(paths_config, "RESULTS_ROOT", "") or "").strip()
    env_results_root = str(os.getenv("SPACEVLN_RESULTS_ROOT", "") or "").strip()
    cli_results_root = str(getattr(args, "results_root", "") or "").strip()
    selected_results_root = cli_results_root or env_results_root or configured_results_root
    if not resolved_results_dir:
        resolved_results_dir = profile.default_results_dir_builder(
            args.vlm_api_config,
            results_root=selected_results_root or None,
        )
    config.defrost()
    config.PATHS.RESULTS_ROOT = resolve_results_root_path(selected_results_root)
    config.PATHS.RESULTS_DIR = resolved_results_dir
    if args.max_steps is not None:
        config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_steps
    apply_output_policy_to_config(config, args)
    apply_runtime_derived_fields(config)
    output_logs = getattr(getattr(config, "OUTPUT", None), "LOGS", None)
    if output_logs is not None:
        args.save_episode_stdout_log = bool(
            getattr(output_logs, "SAVE_EPISODE_STDOUT", False)
        )
    config.freeze()
    return config


def build_episode_config(base_config, args: argparse.Namespace, episode_id: int):
    episode_config = base_config.clone()
    episode_config.defrost()
    episode_config = ConfigHelper.setup_episode_config(
        episode_config,
        [episode_id],
        num_environments=1,
    )
    if args.results_dir:
        episode_config = ConfigHelper.setup_results_dir(episode_config, args.results_dir)
    episode_config = ConfigHelper.setup_navigation_config(episode_config)
    episode_config.freeze()
    return episode_config


def create_navigation_controller(
    episode_config,
    args: argparse.Namespace,
    profile: NavigationRuntimeProfile = STANDARD_RUNTIME_PROFILE,
) -> Tuple[NavigationAgentController, str]:
    from navigation_system.env.vlnce.r2r.adapter import build_vlnce_vector_env

    unified_config = resolve_api_config_path(args.vlm_api_config)
    envs = build_vlnce_vector_env(
        episode_config,
        auto_reset_done=False,
        episodes_allowed=episode_config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,
    )
    controller = NavigationAgentController(
        episode_config,
        config_path=unified_config,
        model_stack_builder=profile.model_stack_builder,
        envs=envs,
    )
    return controller, unified_config


def _workspace_episode_cache_dir() -> str:
    workspace_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../..")
    )
    fast_root = str(os.getenv("SPACEVLN_FAST_EPISODE_CACHE_ROOT", "") or "").strip()
    if not fast_root and os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        fast_root = "/dev/shm"
    if fast_root:
        user_tag = str(os.getenv("USER", "") or os.getuid()).strip()
        workspace_tag = hashlib.sha1(
            os.path.realpath(workspace_root).encode("utf-8")
        ).hexdigest()[:10]
        return os.path.join(
            os.path.abspath(os.path.expanduser(fast_root)),
            "spacevln_episode_cache",
            user_tag,
            workspace_tag,
        )
    return os.path.join(workspace_root, ".spacevln_episode_cache")


def _path_matches_any_prefix(path: str, prefixes: List[str]) -> bool:
    if not path:
        return False
    normalized_path = os.path.realpath(os.path.abspath(path))
    for prefix in prefixes:
        clean_prefix = os.path.realpath(os.path.abspath(prefix))
        try:
            if os.path.commonpath([normalized_path, clean_prefix]) == clean_prefix:
                return True
        except ValueError:
            continue
    return False


def _should_auto_stage_results(final_results_dir: str) -> bool:
    if _env_flag_enabled("SPACEVLN_DISABLE_AUTO_EPISODE_WORKDIR", default=False):
        return False
    raw_prefixes = str(
        os.getenv("SPACEVLN_SLOW_RESULT_PREFIXES", "/media:/mnt:/run/media") or ""
    ).strip()
    prefixes = [item for item in raw_prefixes.split(":") if item.strip()]
    return _path_matches_any_prefix(final_results_dir, prefixes)


def _resolve_episode_workdir(
    args: argparse.Namespace,
    *,
    final_results_dir: str = "",
) -> str:
    raw_value = (
        str(getattr(args, "episode_workdir", "") or "").strip()
        or str(os.getenv("SPACEVLN_EPISODE_WORKDIR", "") or "").strip()
    )
    if raw_value:
        return os.path.abspath(os.path.expanduser(raw_value))
    if final_results_dir and _should_auto_stage_results(final_results_dir):
        return _workspace_episode_cache_dir()
    return ""


def _build_staging_results_dir(
    *,
    episode_workdir: str,
    final_results_dir: str,
    episode_id: int,
    worker_index: int = 0,
) -> str:
    final_leaf = os.path.basename(os.path.abspath(final_results_dir).rstrip(os.sep)) or "results"
    final_digest = hashlib.sha1(
        os.path.realpath(os.path.abspath(final_results_dir)).encode("utf-8")
    ).hexdigest()[:10]
    result_tag = f"{final_leaf}_{final_digest}"
    worker_tag = f"worker_{int(worker_index or 0)}" if int(worker_index or 0) > 0 else f"pid_{os.getpid()}"
    return os.path.join(
        episode_workdir,
        result_tag,
        worker_tag,
        f"episode_{int(episode_id)}",
    )


def _copytree_replace(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_file(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copy2(src, dst)


def _rewrite_staging_paths_in_value(value: Any, staging_prefix: str, final_prefix: str) -> Any:
    if isinstance(value, str):
        if not value.strip():
            return value
        value_path = os.path.abspath(value)
        if value_path.startswith(staging_prefix):
            return final_prefix + value_path[len(staging_prefix):]
        return value
    if isinstance(value, dict):
        return {
            key: _rewrite_staging_paths_in_value(item, staging_prefix, final_prefix)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_staging_paths_in_value(item, staging_prefix, final_prefix)
            for item in value
        ]
    return value


def _rewrite_staging_paths_in_json_file(path: str, staging_results_dir: str, final_results_dir: str) -> None:
    if not os.path.exists(path):
        return
    payload = SaveManager._load_json_if_exists(path)
    if payload is None:
        return
    staging_prefix = os.path.abspath(staging_results_dir).rstrip(os.sep) + os.sep
    final_prefix = os.path.abspath(final_results_dir).rstrip(os.sep) + os.sep
    rewritten = _rewrite_staging_paths_in_value(payload, staging_prefix, final_prefix)
    if rewritten == payload:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rewritten, f, indent=2, ensure_ascii=False)


def _rewrite_staged_episode_json_paths(
    *,
    final_detail_dir: str,
    final_log_path: str,
    staging_results_dir: str,
    final_results_dir: str,
) -> None:
    _rewrite_staging_paths_in_json_file(
        final_log_path,
        staging_results_dir,
        final_results_dir,
    )
    records_result_path = os.path.join(final_detail_dir, "records", "result.json")
    _rewrite_staging_paths_in_json_file(
        records_result_path,
        staging_results_dir,
        final_results_dir,
    )


def _get_episode_result_path_no_create(
    results_dir: str,
    episode_id: int,
    *,
    entry_kind: str = "episode",
) -> str:
    return os.path.join(
        get_episode_detail_dir(results_dir, episode_id, entry_kind=entry_kind),
        "records",
        "result.json",
    )


def _remove_empty_parent_dirs(path: str, *, max_levels: int = 2) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    for _ in range(max(0, int(max_levels))):
        if not parent:
            break
        try:
            os.rmdir(parent)
        except OSError:
            break
        parent = os.path.dirname(parent)


def _sync_episode_staging_outputs(
    *,
    staging_results_dir: str,
    final_results_dir: str,
    episode_id: int,
    storage_entry_id: Optional[int] = None,
    entry_kind: str = "episode",
    save_stdout_log: bool,
) -> None:
    """Move one episode's fast local artifacts back to the final results directory.

    Detail artifacts describe the latest completed run and are replaced whenever
    a complete staged result exists. The compact log is the resumable/reportable
    best result and is updated only when the staged result ranks better.
    """
    if not staging_results_dir or not final_results_dir:
        return
    entry_id = int(storage_entry_id) if storage_entry_id is not None else int(episode_id)
    entry_kind = str(entry_kind or "episode").strip() or "episode"
    staging_detail_dir = get_episode_detail_dir(
        staging_results_dir,
        entry_id,
        entry_kind=entry_kind,
    )
    final_detail_dir = get_episode_detail_dir(
        final_results_dir,
        entry_id,
        entry_kind=entry_kind,
    )

    staging_log_path = get_episode_log_path(
        staging_results_dir,
        entry_id,
        entry_kind=entry_kind,
    )
    final_log_path = get_episode_log_path(
        final_results_dir,
        entry_id,
        entry_kind=entry_kind,
    )
    staging_result_path = _get_episode_result_path_no_create(
        staging_results_dir,
        entry_id,
        entry_kind=entry_kind,
    )
    staging_result = SaveManager._load_json_if_exists(staging_result_path)
    detail_updated = False
    suppress_staging_record = should_suppress_persistent_record_for_failure(staging_result)
    if SaveManager.is_complete_result(staging_result) and not suppress_staging_record:
        _copytree_replace(staging_detail_dir, final_detail_dir)
        detail_updated = True

    staging_log = SaveManager._load_json_if_exists(staging_log_path)
    if staging_log is not None and not should_suppress_persistent_record_for_failure(staging_log):
        final_log = SaveManager._load_json_if_exists(final_log_path)
        final_baseline = final_log if SaveManager.is_complete_result(final_log) else None
        if SaveManager.is_complete_result(staging_log) and (
            final_baseline is None
            or SaveManager.result_rank_key(staging_log) > SaveManager.result_rank_key(final_baseline)
        ):
            _copy_file(staging_log_path, final_log_path)

    if detail_updated or os.path.exists(final_log_path):
        _rewrite_staged_episode_json_paths(
            final_detail_dir=final_detail_dir,
            final_log_path=final_log_path,
            staging_results_dir=staging_results_dir,
            final_results_dir=final_results_dir,
        )

    if detail_updated and save_stdout_log:
        _copy_file(
            get_episode_records_log_path(
                staging_results_dir,
                entry_id,
                entry_kind=entry_kind,
            ),
            get_episode_records_log_path(
                final_results_dir,
                entry_id,
                entry_kind=entry_kind,
            ),
        )


def _transfer_episode_staging_outputs(
    *,
    staging_results_dir: str,
    final_results_dir: str,
    episode_id: int,
    storage_entry_id: Optional[int] = None,
    entry_kind: str = "episode",
    save_stdout_log: bool,
    postprocess_futures: Optional[List[concurrent.futures.Future]] = None,
) -> None:
    try:
        _wait_for_episode_postprocess_futures(
            postprocess_futures,
            episode_id=episode_id,
        )
        _sync_episode_staging_outputs(
            staging_results_dir=staging_results_dir,
            final_results_dir=final_results_dir,
            episode_id=episode_id,
            storage_entry_id=storage_entry_id,
            entry_kind=entry_kind,
            save_stdout_log=save_stdout_log,
        )
    finally:
        if staging_results_dir:
            shutil.rmtree(staging_results_dir, ignore_errors=True)
            _remove_empty_parent_dirs(staging_results_dir, max_levels=2)


def _episode_transfer_worker_count() -> int:
    return min(
        4,
        _positive_int(os.getenv("SPACEVLN_EPISODE_TRANSFER_WORKERS", "2"), 2),
    )


def _get_episode_transfer_semaphore() -> threading.BoundedSemaphore:
    global _EPISODE_TRANSFER_SEMAPHORE, _EPISODE_TRANSFER_SEMAPHORE_LIMIT
    worker_count = _episode_transfer_worker_count()
    if (
        _EPISODE_TRANSFER_SEMAPHORE is None
        or _EPISODE_TRANSFER_SEMAPHORE_LIMIT != worker_count
    ):
        _EPISODE_TRANSFER_SEMAPHORE = threading.BoundedSemaphore(worker_count)
        _EPISODE_TRANSFER_SEMAPHORE_LIMIT = worker_count
    return _EPISODE_TRANSFER_SEMAPHORE


def _submit_daemon_episode_transfer(**kwargs: Any) -> concurrent.futures.Future:
    """Run one transfer on daemon threads so process exit is not held hostage."""
    future: concurrent.futures.Future = concurrent.futures.Future()
    semaphore = _get_episode_transfer_semaphore()

    def _runner() -> None:
        acquired = False
        try:
            semaphore.acquire()
            acquired = True
            if not future.set_running_or_notify_cancel():
                return
            _transfer_episode_staging_outputs(**kwargs)
            future.set_result(None)
        except BaseException as exc:  # pragma: no cover - defensive background path
            future.set_exception(exc)
        finally:
            if acquired:
                try:
                    semaphore.release()
                except ValueError:
                    pass

    thread = threading.Thread(
        target=_runner,
        name="spacevln-transfer",
        daemon=True,
    )
    thread.start()
    return future


def _forget_episode_transfer_future(future: concurrent.futures.Future) -> None:
    with _EPISODE_TRANSFER_LOCK:
        try:
            _EPISODE_TRANSFER_FUTURES.remove(future)
        except ValueError:
            pass


def _handle_episode_transfer_done(future: concurrent.futures.Future) -> None:
    try:
        future.result()
    except BaseException as exc:
        print(f"⚠️  Failed to sync staged episode artifacts: {_format_exception_message(exc)}")
    finally:
        _forget_episode_transfer_future(future)


def _submit_episode_staging_sync(
    *,
    staging_results_dir: str,
    final_results_dir: str,
    episode_id: int,
    storage_entry_id: Optional[int] = None,
    entry_kind: str = "episode",
    save_stdout_log: bool,
    postprocess_futures: Optional[List[concurrent.futures.Future]] = None,
) -> None:
    if not staging_results_dir:
        return
    if _env_flag_enabled("SPACEVLN_SYNC_EPISODE_TRANSFER", default=False):
        _transfer_episode_staging_outputs(
            staging_results_dir=staging_results_dir,
            final_results_dir=final_results_dir,
            episode_id=episode_id,
            storage_entry_id=storage_entry_id,
            entry_kind=entry_kind,
            save_stdout_log=save_stdout_log,
            postprocess_futures=postprocess_futures,
        )
        return
    future = _submit_daemon_episode_transfer(
        staging_results_dir=staging_results_dir,
        final_results_dir=final_results_dir,
        episode_id=episode_id,
        storage_entry_id=storage_entry_id,
        entry_kind=entry_kind,
        save_stdout_log=save_stdout_log,
        postprocess_futures=postprocess_futures,
    )
    with _EPISODE_TRANSFER_LOCK:
        _EPISODE_TRANSFER_FUTURES.append(future)
    future.add_done_callback(_handle_episode_transfer_done)
    _throttle_episode_transfer_backlog()


def _episode_transfer_pool_limit() -> int:
    raw_value = str(os.getenv("SPACEVLN_EPISODE_TRANSFER_POOL", "") or "").strip()
    if raw_value:
        return _positive_int(raw_value, 3)
    legacy_value = str(os.getenv("SPACEVLN_EPISODE_TRANSFER_BACKLOG", "") or "").strip()
    if legacy_value:
        return _positive_int(legacy_value, 3)
    return 3


def _episode_transfer_batch_size(pool_limit: int) -> int:
    default_batch = min(2, max(1, int(pool_limit)))
    batch_size = _positive_int(
        os.getenv("SPACEVLN_EPISODE_TRANSFER_BATCH", str(default_batch)),
        default_batch,
    )
    return max(1, min(int(batch_size), int(pool_limit)))


def _throttle_episode_transfer_backlog() -> None:
    pool_limit = _episode_transfer_pool_limit()
    transfer_batch = _episode_transfer_batch_size(pool_limit)
    throttle_timeout_s = max(
        0.0,
        env_float("SPACEVLN_EPISODE_TRANSFER_THROTTLE_TIMEOUT_S", 60.0),
    )
    while True:
        with _EPISODE_TRANSFER_LOCK:
            pending_futures = list(_EPISODE_TRANSFER_FUTURES)
        if len(pending_futures) < pool_limit:
            return
        target_pending = max(0, pool_limit - transfer_batch)
        while len(pending_futures) > target_pending:
            done, _ = concurrent.futures.wait(
                pending_futures,
                timeout=throttle_timeout_s if throttle_timeout_s > 0 else None,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                print(
                    "⚠️  Episode artifact transfer backlog did not drain within "
                    f"{throttle_timeout_s:.1f}s; continuing without blocking the run.",
                    flush=True,
                )
                return
            for future in done:
                try:
                    future.result()
                except BaseException:
                    pass
                _forget_episode_transfer_future(future)
            with _EPISODE_TRANSFER_LOCK:
                pending_futures = list(_EPISODE_TRANSFER_FUTURES)
        return


def _discard_episode_staging(staging_results_dir: str) -> None:
    if staging_results_dir:
        shutil.rmtree(staging_results_dir, ignore_errors=True)
        _remove_empty_parent_dirs(staging_results_dir, max_levels=2)


def _wait_for_episode_postprocess_futures(
    futures: Optional[List[concurrent.futures.Future]],
    *,
    episode_id: int,
) -> None:
    """Wait for GIF/topdown post-processing before moving or deleting staging dirs."""
    pending = [future for future in list(futures or []) if future is not None]
    if not pending:
        return
    timeout_s = max(0.0, env_float("SPACEVLN_POSTPROCESS_WAIT_TIMEOUT_S", 120.0))
    deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
    while pending:
        wait_timeout = None
        if deadline is not None:
            wait_timeout = max(0.0, deadline - time.monotonic())
            if wait_timeout <= 0:
                break
        done, not_done = concurrent.futures.wait(
            pending,
            timeout=wait_timeout,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        if not done:
            break
        for future in done:
            try:
                future.result()
            except BaseException as exc:
                print(
                    f"⚠️  Episode {int(episode_id)} post-processing failed before artifact sync: "
                    f"{_format_exception_message(exc)}",
                    flush=True,
                )
        pending = list(not_done)
    if pending:
        for future in pending:
            try:
                future.cancel()
            except Exception:
                pass
        print(
            f"⚠️  Episode {int(episode_id)} post-processing did not finish within "
            f"{timeout_s:.1f}s; continuing shutdown.",
            flush=True,
        )


def wait_for_pending_episode_transfers() -> None:
    timeout_s = max(0.0, env_float("SPACEVLN_EPISODE_TRANSFER_WAIT_TIMEOUT_S", 120.0))
    deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
    timed_out = False
    while True:
        with _EPISODE_TRANSFER_LOCK:
            pending_futures = list(_EPISODE_TRANSFER_FUTURES)
        if not pending_futures:
            break
        wait_timeout = None
        if deadline is not None:
            wait_timeout = max(0.0, deadline - time.monotonic())
            if wait_timeout <= 0:
                timed_out = True
                break
        done, _not_done = concurrent.futures.wait(
            pending_futures,
            timeout=wait_timeout,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        if not done:
            timed_out = True
            break
        for future in done:
            try:
                future.result()
            except BaseException:
                # The done callback already prints the concrete error once.
                pass
            _forget_episode_transfer_future(future)
    if timed_out:
        with _EPISODE_TRANSFER_LOCK:
            pending_futures = list(_EPISODE_TRANSFER_FUTURES)
            _EPISODE_TRANSFER_FUTURES.clear()
        for future in pending_futures:
            try:
                future.cancel()
            except Exception:
                pass
        print(
            "⚠️  Episode artifact transfer did not finish within "
            f"{timeout_s:.1f}s; continuing shutdown. "
            "Increase SPACEVLN_EPISODE_TRANSFER_WAIT_TIMEOUT_S or set it to 0 to wait indefinitely.",
            flush=True,
        )


atexit.register(wait_for_pending_episode_transfers)


def _should_retry_initial_failure(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    if bool(result.get("success", False)):
        return False
    if str(result.get("error") or "").strip():
        return False
    reason = str(result.get("reason") or "").strip().lower()
    return reason in INITIAL_FAILURE_RETRY_REASONS


def _run_single_episode_attempt(
    base_config,
    args: argparse.Namespace,
    episode_id: int,
    *,
    profile: NavigationRuntimeProfile,
    episode_log_path: str,
    result_path: str,
    save_stdout_log: bool,
    stdout_log_mode: str,
) -> Dict[str, Any]:
    controller = None
    episode_initialized = False
    sample_index = getattr(args, "sample_index", None)
    storage_entry_id = getattr(args, "storage_entry_id", None)
    entry_kind = str(getattr(args, "entry_kind", "episode") or "episode")
    if storage_entry_id is not None:
        storage_entry_id = int(storage_entry_id)
    if sample_index is not None:
        sample_index = int(sample_index)
    final_results_dir = os.path.abspath(str(base_config.PATHS.RESULTS_DIR or os.getcwd()))
    final_episode_log_path = episode_log_path
    final_result_path = result_path
    staging_results_dir = ""
    episode_workdir = _resolve_episode_workdir(args, final_results_dir=final_results_dir)
    run_args = args
    run_episode_log_path = episode_log_path
    run_result_path = result_path
    if episode_workdir:
        staging_results_dir = _build_staging_results_dir(
            episode_workdir=episode_workdir,
            final_results_dir=final_results_dir,
            episode_id=episode_id,
            worker_index=int(getattr(args, "worker_index", 0) or 0),
        )
        if os.path.exists(staging_results_dir):
            shutil.rmtree(staging_results_dir)
        os.makedirs(staging_results_dir, exist_ok=True)
        run_args = copy.copy(args)
        run_args.results_dir = staging_results_dir
        run_episode_log_path = (
            get_episode_records_log_path(
                staging_results_dir,
                int(storage_entry_id) if storage_entry_id is not None else int(episode_id),
                entry_kind=entry_kind,
            )
            if save_stdout_log
            else ""
        )
        run_result_path = get_episode_result_path(
            staging_results_dir,
            int(storage_entry_id) if storage_entry_id is not None else int(episode_id),
            entry_kind=entry_kind,
        )

    console_result: Dict[str, Any] = {
        "episode_id": int(episode_id),
        "success": False,
        "steps": 0,
        "episode_duration_s": 0.0,
        "api_total_duration_s": 0.0,
        "failed_wasted_duration_s": 0.0,
        "error": None,
    }

    try:
        episode_output_mode = str(
            os.getenv("SPACEVLN_EPISODE_OUTPUT", "") or ""
        ).strip().lower()
        if episode_output_mode in {"console", "stdout", "passthrough"}:
            redirect_context = contextlib.nullcontext()
        else:
            redirect_context = (
                redirect_process_output_to_file(run_episode_log_path, mode=stdout_log_mode)
                if save_stdout_log and run_episode_log_path
                else redirect_process_output_to_null()
            )
        with redirect_context:
            episode_config = build_episode_config(base_config, run_args, episode_id)
            controller, _config_desc = create_navigation_controller(
                episode_config,
                run_args,
                profile=profile,
            )
            controller.reset_episode(
                episode_id=episode_id,
                sample_index=sample_index,
                storage_entry_id=storage_entry_id,
                entry_kind=entry_kind,
            )
            episode_initialized = True

            result = controller.run_navigation(max_subtask_steps=args.max_subtask_steps)
            total_steps = result.get("total_steps", result.get("steps", 0))

            console_result = {
                "episode_id": episode_id,
                "sample_index": sample_index,
                "success": result["success"],
                "steps": total_steps,
                "episode_duration_s": result.get("episode_duration_s", 0.0),
                "api_total_duration_s": (
                    (result.get("thinking_api_summary") or {}).get("total_duration_s", 0.0)
                    + (result.get("action_api_summary") or {}).get("total_duration_s", 0.0)
                ),
                "failed_wasted_duration_s": result.get("failed_wasted_duration_s", 0.0),
                "error": None,
                "reason": result.get("reason", ""),
                "result_file": get_episode_log_path(
                    final_results_dir,
                    int(storage_entry_id) if storage_entry_id is not None else int(episode_id),
                    entry_kind=entry_kind,
                ),
                "result_detail_file": final_result_path,
                "episode_log_path": final_episode_log_path,
            }
    except BaseException as exc:
        import traceback

        if isinstance(exc, KeyboardInterrupt):
            raise

        error_msg = _format_exception_message(exc)
        timing_summary = {}
        finalized_success = False
        finalized_steps = 0
        if controller is not None and hasattr(controller, "_build_episode_timing_summary"):
            try:
                timing_summary = controller._build_episode_timing_summary()
            except Exception:
                timing_summary = {}
        if controller is not None and episode_initialized:
            try:
                finalized_steps = int(getattr(controller, "current_step", 0) or 0)
                final_metrics = controller.finish_episode(success=False, stop_action=True)
                normalized_metrics = controller._normalize_final_env_metrics(final_metrics)
                controller._save_navigation_result(finalized_steps, normalized_metrics)
                finalized_success = bool((normalized_metrics or {}).get("success", 0))
            except Exception:
                pass
        if save_stdout_log and run_episode_log_path:
            with redirect_process_output_to_file(run_episode_log_path, mode="a"):
                print(f"\n❌ Episode {episode_id} failed: {error_msg}")
                print("\nFull traceback:")
                traceback.print_exc()
        else:
            print(f"\n❌ Episode {episode_id} failed: {error_msg}")
            print("\nFull traceback:")
            traceback.print_exc()
        console_result = {
            "episode_id": episode_id,
            "sample_index": sample_index,
            "success": finalized_success,
            "steps": finalized_steps,
            "episode_duration_s": timing_summary.get("episode_duration_s", 0.0),
            "api_total_duration_s": (
                (timing_summary.get("thinking_api_summary") or {}).get("total_duration_s", 0.0)
                + (timing_summary.get("action_api_summary") or {}).get("total_duration_s", 0.0)
            ),
            "failed_wasted_duration_s": timing_summary.get("failed_wasted_duration_s", 0.0),
            "error": error_msg,
            "result_file": "",
            "result_detail_file": final_result_path,
            "episode_log_path": final_episode_log_path if save_stdout_log else "",
        }
    finally:
        if controller is not None:
            close_with_timeout(
                controller.envs.close,
                label=f"episode {int(episode_id)} environment",
            )
            try:
                postprocess_futures = controller.pop_pending_post_episode_futures()
                if postprocess_futures:
                    console_result["_postprocess_futures"] = postprocess_futures
            except Exception:
                pass

    if staging_results_dir:
        console_result["_staging_results_dir"] = staging_results_dir
        console_result["_staging_final_results_dir"] = final_results_dir
        console_result["_staging_save_stdout_log"] = bool(save_stdout_log)
        console_result["_staging_result_file"] = get_episode_log_path(
            staging_results_dir,
            int(storage_entry_id) if storage_entry_id is not None else int(episode_id),
            entry_kind=entry_kind,
        )
        console_result["_staging_result_detail_file"] = get_episode_result_path(
            staging_results_dir,
            int(storage_entry_id) if storage_entry_id is not None else int(episode_id),
            entry_kind=entry_kind,
        )

    return console_result


def run_single_episode(
    base_config,
    args: argparse.Namespace,
    episode_id: int,
    index: int,
    total: int,
    profile: NavigationRuntimeProfile = STANDARD_RUNTIME_PROFILE,
) -> Dict[str, Any]:
    console_result: Dict[str, Any] = {
        "episode_id": int(episode_id),
        "success": False,
        "steps": 0,
        "episode_duration_s": 0.0,
        "api_total_duration_s": 0.0,
        "failed_wasted_duration_s": 0.0,
        "error": None,
    }
    sample_index = getattr(args, "sample_index", None)
    sample_index = int(sample_index) if sample_index is not None else None
    storage_entry_id = getattr(args, "storage_entry_id", None)
    storage_entry_id = (
        int(storage_entry_id)
        if storage_entry_id is not None
        else (int(sample_index) if sample_index is not None else int(episode_id))
    )
    entry_kind = str(getattr(args, "entry_kind", "episode") or "episode")
    results_dir = os.path.abspath(str(base_config.PATHS.RESULTS_DIR or os.getcwd()))
    save_stdout_log = save_episode_stdout_log_enabled(base_config)
    episode_log_path = (
        get_episode_records_log_path(
            results_dir,
            storage_entry_id,
            entry_kind=entry_kind,
        )
        if save_stdout_log
        else ""
    )
    result_path = _get_episode_result_path_no_create(
        results_dir,
        storage_entry_id,
        entry_kind=entry_kind,
    )
    best_log_path = get_episode_log_path(
        results_dir,
        storage_entry_id,
        entry_kind=entry_kind,
    )
    print(
        build_episode_start_summary(
            episode_id=episode_id,
            index=index,
            total=total,
            worker_index=int(getattr(args, "worker_index", 0) or 0),
            worker_count=int(getattr(args, "worker_count", 0) or 0),
            sample_index=sample_index,
        ),
        flush=True,
    )

    max_attempts = _resolve_initial_failure_max_attempts(args)
    attempts_run = 0
    for attempt_index in range(max_attempts):
        attempts_run = attempt_index + 1
        artifact_snapshots = _snapshot_episode_result_artifacts(
            result_path=result_path,
            best_log_path=best_log_path,
            stdout_log_path=episode_log_path if save_stdout_log else "",
        )
        stdout_log_mode = "w"

        run_args = copy.copy(args)
        run_args.sample_index = sample_index
        run_args.storage_entry_id = storage_entry_id
        run_args.entry_kind = entry_kind
        console_result = _run_single_episode_attempt(
            base_config,
            run_args,
            episode_id,
            profile=profile,
            episode_log_path=episode_log_path,
            result_path=result_path,
            save_stdout_log=save_stdout_log,
            stdout_log_mode=stdout_log_mode,
        )
        staging_results_dir = str(console_result.get("_staging_results_dir") or "")
        postprocess_futures = console_result.pop("_postprocess_futures", None)
        staging_final_results_dir = str(
            console_result.get("_staging_final_results_dir") or results_dir
        )
        staging_metrics = {}
        if staging_results_dir:
            staging_metrics = extract_episode_metrics(
                {
                    "result_detail_file": console_result.get(
                        "_staging_result_detail_file",
                        "",
                    ),
                    "result_file": console_result.get("_staging_result_file", ""),
                }
            )
            if staging_metrics:
                console_result["_console_metrics"] = staging_metrics

        should_retry = (
            attempt_index < max_attempts - 1
            and _should_retry_initial_failure(console_result)
        )
        should_suppress_record = _should_suppress_initial_failure_record(console_result)
        if should_retry or should_suppress_record:
            _restore_episode_result_artifacts(
                artifact_snapshots,
                result_path=result_path,
                best_log_path=best_log_path,
                stdout_log_path=episode_log_path if save_stdout_log else "",
            )
            if should_suppress_record:
                _mark_result_unrecorded(console_result)
            _wait_for_episode_postprocess_futures(
                postprocess_futures,
                episode_id=episode_id,
            )
            _discard_episode_staging(staging_results_dir)
        elif staging_results_dir:
            _submit_episode_staging_sync(
                staging_results_dir=staging_results_dir,
                final_results_dir=staging_final_results_dir,
                episode_id=episode_id,
                storage_entry_id=storage_entry_id,
                entry_kind=entry_kind,
                save_stdout_log=bool(
                    console_result.get("_staging_save_stdout_log", save_stdout_log)
                ),
                postprocess_futures=postprocess_futures,
            )
        elif postprocess_futures:
            _wait_for_episode_postprocess_futures(
                postprocess_futures,
                episode_id=episode_id,
            )
        if not should_retry:
            break
        retry_reason = str(console_result.get("reason") or "").strip()
        print(
            f"[Retry] Episode {int(episode_id)} rerun because {retry_reason} "
            f"(attempt {attempt_index + 2}/{max_attempts})",
            flush=True,
        )

    console_result["attempts"] = attempts_run

    metrics = dict(console_result.get("_console_metrics") or {})
    if not metrics:
        metrics = extract_episode_metrics(console_result)
    for internal_key in (
        "_console_metrics",
        "_staging_results_dir",
        "_staging_final_results_dir",
        "_staging_save_stdout_log",
        "_staging_result_file",
        "_staging_result_detail_file",
    ):
        console_result.pop(internal_key, None)
    print(
        build_episode_console_summary(
            episode_id=episode_id,
            index=index,
            total=total,
            result=console_result,
            metrics=metrics,
            worker_index=int(getattr(args, "worker_index", 0) or 0),
            worker_count=int(getattr(args, "worker_count", 0) or 0),
            sample_index=sample_index,
        ),
        flush=True,
    )
    return console_result


def _build_parallel_episode_spec(
    *,
    args: argparse.Namespace,
    episode_id: int,
    index: int,
    total: int,
    worker_index: int,
    worker_count: int,
    profile: NavigationRuntimeProfile,
) -> Dict[str, Any]:
    sample_index_by_episode = dict(getattr(args, "_sample_index_by_episode", {}) or {})
    sample_index = sample_index_by_episode.get(int(episode_id))
    entry_kind = "sample" if sample_index is not None else "episode"
    storage_entry_id = int(sample_index) if sample_index is not None else int(episode_id)
    return {
        "exp_config": args.exp_config,
        "episode_id": int(episode_id),
        "sample_index": int(sample_index) if sample_index is not None else None,
        "storage_entry_id": int(storage_entry_id),
        "entry_kind": entry_kind,
        "index": int(index),
        "total": int(total),
        "results_dir": args.results_dir,
        "episode_workdir": str(getattr(args, "episode_workdir", "") or ""),
        "vlm_api_config": args.vlm_api_config,
        "max_subtask_steps": int(args.max_subtask_steps),
        "max_steps": args.max_steps,
        "initial_failure_max_attempts": _resolve_initial_failure_max_attempts(args),
        **build_output_job_fields(args),
        "worker_index": int(worker_index),
        "worker_count": int(worker_count),
        "runtime_profile_name": profile.name,
    }


def _run_parallel_episode_job(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    args = argparse.Namespace(
        exp_config=job_spec["exp_config"],
        episode_id=int(job_spec["episode_id"]),
        sample_index=job_spec.get("sample_index"),
        storage_entry_id=int(job_spec.get("storage_entry_id", job_spec["episode_id"])),
        entry_kind=str(job_spec.get("entry_kind", "episode") or "episode"),
        episode_ids=None,
        num_episodes=1,
        random=False,
        results_dir=job_spec.get("results_dir"),
        episode_workdir=job_spec.get("episode_workdir", ""),
        vlm_api_config=job_spec["vlm_api_config"],
        max_subtask_steps=job_spec["max_subtask_steps"],
        max_steps=job_spec.get("max_steps"),
        initial_failure_max_attempts=int(job_spec.get("initial_failure_max_attempts", DEFAULT_INITIAL_FAILURE_MAX_ATTEMPTS)),
        skip_sr1=False,
        parallel_workers=1,
        **build_output_namespace_kwargs(job_spec),
        worker_index=int(job_spec["worker_index"]),
        worker_count=int(job_spec["worker_count"]),
        suppress_runtime_prints=True,
        auto=True,
    )
    profile = resolve_runtime_profile(job_spec["runtime_profile_name"])
    config = load_runtime_config(args, profile=profile)
    return {
        "worker_index": int(job_spec["worker_index"]),
        "worker_count": int(job_spec["worker_count"]),
        "episode_id": int(job_spec["episode_id"]),
        "result": run_single_episode(
            config,
            args,
            episode_id=int(job_spec["episode_id"]),
            index=int(job_spec["index"]),
            total=int(job_spec["total"]),
            profile=profile,
        ),
    }


def run_parallel_episodes(
    config,
    args: argparse.Namespace,
    episode_ids: List[int],
    profile: NavigationRuntimeProfile = STANDARD_RUNTIME_PROFILE,
) -> List[Dict[str, Any]]:
    worker_count = max(1, int(args.parallel_workers or 1))
    worker_count = min(worker_count, len(episode_ids))
    if worker_count <= 1 or len(episode_ids) <= 1:
        return []

    total = len(episode_ids)
    results_summary: List[Dict[str, Any]] = []
    next_job_cursor = 0
    pool_broken_error = ""
    initial_planner_api_error_count = 0
    max_initial_planner_api_errors = resolve_max_initial_planner_api_errors(
        getattr(args, "max_initial_planner_api_errors", None),
        worker_count=worker_count,
    )
    abort_for_initial_api_errors = False

    def _build_parallel_failure(episode_id: int, error_text: str) -> Dict[str, Any]:
        return {
            "episode_id": int(episode_id),
            "success": False,
            "steps": 0,
            "episode_duration_s": 0.0,
            "api_total_duration_s": 0.0,
            "failed_wasted_duration_s": 0.0,
            "error": error_text,
        }

    mp_context = multiprocessing.get_context("spawn")
    interrupted = False
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
        initializer=_parallel_worker_initializer,
    )
    try:
        future_to_job: Dict[concurrent.futures.Future, Dict[str, Any]] = {}

        def _submit_next_job(worker_index: int) -> bool:
            nonlocal next_job_cursor, pool_broken_error
            if abort_for_initial_api_errors:
                return False
            if next_job_cursor >= total:
                return False
            job_spec = _build_parallel_episode_spec(
                args=args,
                episode_id=int(episode_ids[next_job_cursor]),
                index=next_job_cursor + 1,
                total=total,
                worker_index=worker_index,
                worker_count=worker_count,
                profile=profile,
            )
            try:
                future = executor.submit(_run_parallel_episode_job, job_spec)
            except concurrent.futures.process.BrokenProcessPool as exc:
                pool_broken_error = str(exc)
                results_summary.append(
                    _build_parallel_failure(
                        int(job_spec["episode_id"]),
                        f"parallel worker pool broken while submitting episode: {exc}",
                    )
                )
                next_job_cursor += 1
                return False
            future_to_job[future] = job_spec
            next_job_cursor += 1
            return True

        for worker_index in range(1, worker_count + 1):
            if not _submit_next_job(worker_index):
                break

        while future_to_job:
            try:
                done, _ = concurrent.futures.wait(
                    list(future_to_job.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
            except KeyboardInterrupt:
                interrupted = True
                for pending_future in list(future_to_job.keys()):
                    pending_future.cancel()
                raise
            for future in done:
                job_spec = future_to_job.pop(future)
                try:
                    job_result = future.result()
                    result_payload = job_result.get("result", {})
                    results_summary.append(result_payload)
                except (KeyboardInterrupt, SystemExit):
                    interrupted = True
                    future.cancel()
                    for pending_future in list(future_to_job.keys()):
                        pending_future.cancel()
                    raise
                except Exception as exc:
                    if isinstance(exc, concurrent.futures.process.BrokenProcessPool):
                        pool_broken_error = str(exc)
                    results_summary.append(
                        _build_parallel_failure(
                            int(job_spec["episode_id"]),
                            f"parallel worker failed: {_format_exception_message(exc)}",
                        )
                    )
                    result_payload = results_summary[-1]
                if is_initial_planner_api_error_result(result_payload):
                    initial_planner_api_error_count += 1
                    if (
                        max_initial_planner_api_errors > 0
                        and initial_planner_api_error_count >= max_initial_planner_api_errors
                    ):
                        abort_for_initial_api_errors = True
                        interrupted = True
                        print(
                            "[Abort] Too many initial planner API errors "
                            f"({initial_planner_api_error_count}/{max_initial_planner_api_errors}); "
                            "stop submitting remaining episodes.",
                            flush=True,
                        )
                        for pending_future in list(future_to_job.keys()):
                            pending_future.cancel()
                        future_to_job.clear()
                        break
                if not pool_broken_error:
                    _submit_next_job(int(job_spec["worker_index"]))

    finally:
        _shutdown_parallel_executor(executor, interrupted=interrupted)

    if pool_broken_error and next_job_cursor < total:
        for episode_id in episode_ids[next_job_cursor:]:
            results_summary.append(
                _build_parallel_failure(
                    int(episode_id),
                    f"parallel worker pool broken, skipped pending episode: {pool_broken_error}",
                )
            )

    results_summary.sort(key=lambda item: int(item.get("episode_id", 0)))
    return results_summary


__all__ = [
    "build_episode_config",
    "create_navigation_controller",
    "load_runtime_config",
    "run_parallel_episodes",
    "run_single_episode",
    "wait_for_pending_episode_transfers",
]
