"""Thin wrapper that reuses the standard runner with the Qwen explicit-cache controller."""

import argparse
import concurrent.futures
import multiprocessing
import os
from typing import Any, Dict, List

import navigation_system.runtime.runner as base_runner
from navigation_system.controllers.vlm_navigation_controller_qwen_cache import (
    QwenContextCacheNavigationController,
)
from navigation_system.vlm.api.qwen_cache import (
    build_default_qwen_context_cache_results_dir,
)
from navigation_system.vlm.api.client import resolve_api_config_path
from navigation_system.vlm.reporting.cache_report import (
    build_cache_report,
    render_cache_report,
    write_cache_report_json,
)


DEFAULT_QWEN_CACHE_CONFIG = "navigation_system/config/api/vlm_api_config_qwen_cache.yaml"


def build_arg_parser():
    parser = base_runner.build_arg_parser()
    parser.set_defaults(vlm_api_config=DEFAULT_QWEN_CACHE_CONFIG)
    return parser


def create_controller(
    episode_config,
    args: argparse.Namespace,
):
    unified_config = resolve_api_config_path(args.vlm_api_config)
    controller = QwenContextCacheNavigationController(
        episode_config,
        config_path=unified_config,
    )
    return controller, unified_config


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
        skip_sr1=False,
        parallel_workers=1,
        worker_index=int(job_spec["worker_index"]),
        worker_count=int(job_spec["worker_count"]),
        suppress_runtime_prints=True,
        auto=True,
    )

    config = base_runner.load_runtime_config(args)
    original_create_controller = base_runner.create_controller
    try:
        base_runner.create_controller = create_controller
        result = base_runner.run_single_episode(
            config,
            args,
            episode_id=int(job_spec["episode_id"]),
            index=int(job_spec["index"]),
            total=int(job_spec["total"]),
        )
    finally:
        base_runner.create_controller = original_create_controller

    return {
        "worker_index": int(job_spec["worker_index"]),
        "worker_count": int(job_spec["worker_count"]),
        "episode_id": int(job_spec["episode_id"]),
        "result": result,
    }


def run_parallel_episodes(
    config,
    args: argparse.Namespace,
    episode_ids: List[int],
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
            job_spec = base_runner._build_parallel_episode_spec(
                args=args,
                episode_id=int(episode_ids[next_job_cursor]),
                index=next_job_cursor + 1,
                total=total,
                worker_index=worker_index,
                worker_count=worker_count,
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
                            "episode_duration_including_failed_s": 0.0,
                            "episode_duration_excluding_failed_s": 0.0,
                            "api_total_duration_s": 0.0,
                            "failed_wasted_duration_s": 0.0,
                            "error": f"parallel worker failed: {exc}",
                        }
                    )
                _submit_next_job(int(job_spec["worker_index"]))

    results_summary.sort(key=lambda item: int(item.get("episode_id", 0)))
    return results_summary


def _write_cache_report_text(report_text: str, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        if not report_text.endswith("\n"):
            f.write("\n")
    return output_path


def maybe_generate_cache_report(args: argparse.Namespace, config) -> None:
    results_dir = str(getattr(args, "results_dir", "") or getattr(config, "RESULTS_DIR", "") or "").strip()
    if not results_dir:
        return

    try:
        report = build_cache_report(results_dir)
    except Exception as exc:
        print(f"⚠️  无法生成缓存报告: {exc}")
        return

    if report.overall.request_count <= 0:
        return

    json_path = os.path.join(results_dir, "cache_report_latest.json")
    txt_path = os.path.join(results_dir, "cache_report_latest.txt")
    report_text = render_cache_report(report)
    write_cache_report_json(report, json_path)
    _write_cache_report_text(report_text, txt_path)

    overall = report.overall
    if overall.requests_with_provider_counters > 0:
        print(
            "[CtxCache][overall] "
            f"req={overall.request_count} "
            f"| covered={overall.cache_metric_request_coverage_ratio * 100:.1f}% "
            f"| hit={overall.weighted_cache_hit_ratio * 100:.1f}% "
            f"| input_cost_x={overall.effective_input_cost_multiplier:.3f} "
            f"| speed={overall.end_to_end_tokens_per_second:.1f} tok/s"
        )
    else:
        print(
            "[CtxCache][overall] "
            f"req={overall.request_count} | provider_counters=0 "
            "| cache metrics unavailable in current artifacts"
        )
    print(f"[CtxCache][report] {txt_path}")


def run_navigation_from_args(args: argparse.Namespace) -> int:
    patched_args = argparse.Namespace(**vars(args))
    if not str(getattr(patched_args, "vlm_api_config", "") or "").strip():
        patched_args.vlm_api_config = DEFAULT_QWEN_CACHE_CONFIG
    if not str(getattr(patched_args, "results_dir", "") or "").strip():
        patched_args.results_dir = build_default_qwen_context_cache_results_dir(
            patched_args.vlm_api_config
        )

    original_create_controller = base_runner.create_controller
    original_run_parallel_episodes = base_runner.run_parallel_episodes
    try:
        base_runner.create_controller = create_controller
        base_runner.run_parallel_episodes = run_parallel_episodes
        exit_code = base_runner.run_navigation_from_args(patched_args)
    finally:
        base_runner.create_controller = original_create_controller
        base_runner.run_parallel_episodes = original_run_parallel_episodes
    try:
        config = base_runner.load_runtime_config(patched_args)
        maybe_generate_cache_report(patched_args, config)
    except Exception as exc:
        print(f"⚠️  缓存报告后处理失败: {exc}")
    return exit_code
