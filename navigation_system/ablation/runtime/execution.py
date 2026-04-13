"""Runtime execution helpers for ablation entrypoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
from typing import Any, Dict, List, Tuple

from navigation_system.ablation.config import ABLATION_CONFIG_ENV
from navigation_system.ablation.models.controller import AblationVLMNavigationController
from navigation_system.ablation.runtime.profiles import (
    ABLATION_STANDARD_RUNTIME_PROFILE,
    AblationRuntimeProfile,
    resolve_ablation_runtime_profile,
)
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
from navigation_system.runtime.execution import (
    build_episode_config,
    load_runtime_config,
)


INITIAL_FAILURE_RETRY_REASONS = {
    "initial_subtask_failed",
    "initial_lookaround_failed",
}


def create_navigation_controller(
    episode_config,
    args: argparse.Namespace,
    profile: AblationRuntimeProfile = ABLATION_STANDARD_RUNTIME_PROFILE,
) -> Tuple[AblationVLMNavigationController, str]:
    unified_config = str(getattr(args, "vlm_api_config", "") or "").strip()
    controller = AblationVLMNavigationController(
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
    profile: AblationRuntimeProfile,
    episode_log_path: str,
    result_path: str,
    save_stdout_log: bool,
    stdout_log_mode: str,
) -> Dict[str, Any]:
    controller = None
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

        error_msg = str(exc)
        timing_summary = {}
        finalized_success = False
        finalized_steps = 0
        if controller is not None and hasattr(controller, "_build_episode_timing_summary"):
            try:
                timing_summary = controller._build_episode_timing_summary()
            except Exception:
                timing_summary = {}
        if controller is not None:
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
                print(f"\n❌ Episode {episode_id} 运行失败: {error_msg}")
                print("\n完整错误堆栈:")
                traceback.print_exc()
        else:
            print(f"\n❌ Episode {episode_id} 运行失败: {error_msg}")
            print("\n完整错误堆栈:")
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
                print(f"⚠️  清理环境时出错: {cleanup_error}")

    return console_result


def run_single_episode(
    base_config,
    args: argparse.Namespace,
    episode_id: int,
    index: int,
    total: int,
    profile: AblationRuntimeProfile = ABLATION_STANDARD_RUNTIME_PROFILE,
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
    results_dir = os.path.abspath(str(base_config.RESULTS_DIR or os.getcwd()))
    save_stdout_log = save_episode_stdout_log_enabled(base_config)
    episode_log_path = (
        get_episode_records_log_path(results_dir, episode_id)
        if save_stdout_log
        else ""
    )
    result_path = get_episode_result_path(results_dir, episode_id)
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

    max_attempts = 2
    attempts_run = 0
    for attempt_index in range(max_attempts):
        attempts_run = attempt_index + 1
        stdout_log_mode = "w" if attempt_index == 0 else "a"
        if save_stdout_log and episode_log_path and attempt_index > 0:
            with redirect_process_output_to_file(episode_log_path, mode="a"):
                print("\n" + "-" * 60)
                print(
                    f"[Retry] Episode {int(episode_id)} rerun attempt "
                    f"{attempt_index + 1}/{max_attempts}"
                )
                print("-" * 60)

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
        if attempt_index >= max_attempts - 1 or not _should_retry_initial_failure(console_result):
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
    profile: AblationRuntimeProfile,
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
        "worker_index": int(worker_index),
        "worker_count": int(worker_count),
        "runtime_profile_name": profile.name,
        "ablation_config": str(getattr(args, "ablation_config", "") or ""),
    }


def _run_parallel_episode_job(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    ablation_config = str(job_spec.get("ablation_config", "") or "").strip()
    if ablation_config:
        os.environ[ABLATION_CONFIG_ENV] = ablation_config

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
        skip_sr1=False,
        parallel_workers=1,
        worker_index=int(job_spec["worker_index"]),
        worker_count=int(job_spec["worker_count"]),
        suppress_runtime_prints=True,
        auto=True,
        ablation_config=ablation_config,
    )
    profile = resolve_ablation_runtime_profile(job_spec["runtime_profile_name"])
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
    profile: AblationRuntimeProfile = ABLATION_STANDARD_RUNTIME_PROFILE,
) -> List[Dict[str, Any]]:
    worker_count = max(1, int(args.parallel_workers or 1))
    worker_count = min(worker_count, len(episode_ids))
    if worker_count <= 1 or len(episode_ids) <= 1:
        return []

    total = len(episode_ids)
    results_summary: List[Dict[str, Any]] = []
    next_job_cursor = 0
    mp_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
    ) as executor:
        future_to_job: Dict[concurrent.futures.Future, Dict[str, Any]] = {}

        def _submit_next_job(worker_index: int) -> bool:
            nonlocal next_job_cursor
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
            future = executor.submit(_run_parallel_episode_job, job_spec)
            future_to_job[future] = job_spec
            next_job_cursor += 1
            return True

        for worker_index in range(1, worker_count + 1):
            if not _submit_next_job(worker_index):
                break

        while future_to_job:
            done, _ = concurrent.futures.wait(
                list(future_to_job.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                job_spec = future_to_job.pop(future)
                try:
                    job_result = future.result()
                    results_summary.append(job_result.get("result", {}))
                except Exception as exc:
                    results_summary.append(
                        {
                            "episode_id": int(job_spec["episode_id"]),
                            "success": False,
                            "steps": 0,
                            "episode_duration_s": 0.0,
                            "api_total_duration_s": 0.0,
                            "failed_wasted_duration_s": 0.0,
                            "error": f"parallel worker failed: {exc}",
                        }
                    )
                _submit_next_job(int(job_spec["worker_index"]))

    results_summary.sort(key=lambda item: int(item.get("episode_id", 0)))
    return results_summary


__all__ = [
    "create_navigation_controller",
    "load_runtime_config",
    "run_parallel_episodes",
    "run_single_episode",
]
