"""Runtime execution helpers: config loading, controller creation, and episode jobs."""

import argparse
import concurrent.futures
import multiprocessing
import os
import signal
from typing import Any, Dict, List, Optional, Tuple

from navigation_system.config import ConfigHelper, get_config
from navigation_system.config.runtime.default import apply_runtime_derived_fields
from navigation_system.controller.vlnce.controller import VLMNavigationController
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
from navigation_system.runtime.storage.artifacts import get_episode_log_path
from navigation_system.runtime.vlnce.profiles import (
    NavigationRuntimeProfile,
    STANDARD_RUNTIME_PROFILE,
    resolve_runtime_profile,
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
    "initial_planner_no_response",
    "initial_planner_timeout",
}

DEFAULT_INITIAL_FAILURE_MAX_ATTEMPTS = 3


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(1, parsed)


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
) -> Tuple[VLMNavigationController, str]:
    unified_config = resolve_api_config_path(args.vlm_api_config)
    controller = VLMNavigationController(
        episode_config,
        config_path=unified_config,
        model_stack_builder=profile.model_stack_builder,
    )
    return controller, unified_config


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
        redirect_context = (
            redirect_process_output_to_file(episode_log_path, mode=stdout_log_mode)
            if save_stdout_log and episode_log_path
            else redirect_process_output_to_null()
        )
        with redirect_context:
            episode_config = build_episode_config(base_config, args, episode_id)
            controller, _config_desc = create_navigation_controller(
                episode_config,
                args,
                profile=profile,
            )
            controller.reset_episode(episode_id=episode_id)
            episode_initialized = True

            result = controller.run_vlm_navigation(max_subtask_steps=args.max_subtask_steps)
            total_steps = result.get("total_steps", result.get("steps", 0))

            console_result = {
                "episode_id": episode_id,
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
                "result_file": result.get("result_file", ""),
                "result_detail_file": result_path,
                "episode_log_path": episode_log_path,
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
        if save_stdout_log and episode_log_path:
            with redirect_process_output_to_file(episode_log_path, mode="a"):
                print(f"\n❌ Episode {episode_id} failed: {error_msg}")
                print("\nFull traceback:")
                traceback.print_exc()
        else:
            print(f"\n❌ Episode {episode_id} failed: {error_msg}")
            print("\nFull traceback:")
            traceback.print_exc()
        console_result = {
            "episode_id": episode_id,
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
            "result_detail_file": result_path,
            "episode_log_path": episode_log_path if save_stdout_log else "",
        }
    finally:
        if controller is not None:
            try:
                controller.envs.close()
            except Exception as cleanup_error:
                print(f"⚠️  Failed to clean up the environment: {cleanup_error}")

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
    results_dir = os.path.abspath(str(base_config.PATHS.RESULTS_DIR or os.getcwd()))
    save_stdout_log = save_episode_stdout_log_enabled(base_config)
    episode_log_path = (
        get_episode_records_log_path(results_dir, episode_id)
        if save_stdout_log
        else ""
    )
    result_path = get_episode_result_path(results_dir, episode_id)
    best_log_path = get_episode_log_path(results_dir, episode_id)
    print(
        build_episode_start_summary(
            episode_id=episode_id,
            index=index,
            total=total,
            worker_index=int(getattr(args, "worker_index", 0) or 0),
            worker_count=int(getattr(args, "worker_count", 0) or 0),
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

        console_result = _run_single_episode_attempt(
            base_config,
            args,
            episode_id,
            profile=profile,
            episode_log_path=episode_log_path,
            result_path=result_path,
            save_stdout_log=save_stdout_log,
            stdout_log_mode=stdout_log_mode,
        )
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
        if not should_retry:
            break
        retry_reason = str(console_result.get("reason") or "").strip()
        print(
            f"[Retry] Episode {int(episode_id)} rerun because {retry_reason} "
            f"(attempt {attempt_index + 2}/{max_attempts})",
            flush=True,
        )

    console_result["attempts"] = attempts_run

    metrics = extract_episode_metrics(console_result)
    print(
        build_episode_console_summary(
            episode_id=episode_id,
            index=index,
            total=total,
            result=console_result,
            metrics=metrics,
            worker_index=int(getattr(args, "worker_index", 0) or 0),
            worker_count=int(getattr(args, "worker_count", 0) or 0),
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
    return {
        "exp_config": args.exp_config,
        "episode_id": int(episode_id),
        "index": int(index),
        "total": int(total),
        "results_dir": args.results_dir,
        "vlm_api_config": args.vlm_api_config,
        "max_subtask_steps": int(args.max_subtask_steps),
        "max_steps": args.max_steps,
        "initial_failure_max_attempts": _resolve_initial_failure_max_attempts(args),
        "worker_index": int(worker_index),
        "worker_count": int(worker_count),
        "runtime_profile_name": profile.name,
    }


def _run_parallel_episode_job(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    args = argparse.Namespace(
        exp_config=job_spec["exp_config"],
        episode_id=int(job_spec["episode_id"]),
        episode_ids=None,
        num_episodes=1,
        random=False,
        results_dir=job_spec.get("results_dir"),
        vlm_api_config=job_spec["vlm_api_config"],
        max_subtask_steps=job_spec["max_subtask_steps"],
        max_steps=job_spec.get("max_steps"),
        initial_failure_max_attempts=int(job_spec.get("initial_failure_max_attempts", DEFAULT_INITIAL_FAILURE_MAX_ATTEMPTS)),
        skip_sr1=False,
        parallel_workers=1,
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
                    results_summary.append(job_result.get("result", {}))
                except (KeyboardInterrupt, SystemExit):
                    interrupted = True
                    future.cancel()
                    for pending_future in list(future_to_job.keys()):
                        pending_future.cancel()
                    raise
                except Exception as exc:
                    results_summary.append(
                        _build_parallel_failure(
                            int(job_spec["episode_id"]),
                            f"parallel worker failed: {_format_exception_message(exc)}",
                        )
                    )
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
]
