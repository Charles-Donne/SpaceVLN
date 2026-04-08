"""
VLM navigation runtime orchestration.

Keep CLI entrypoints thin by moving episode selection, controller bootstrapping,
batch execution, and summary/report generation into reusable helpers.
"""

import argparse
import contextlib
import concurrent.futures
import gzip
import json
import os
import random
import multiprocessing
import sys
import time
from typing import Any, Dict, List, Tuple

from vlnce_baselines.config import ConfigHelper, get_config
from vlnce_baselines.vlm.api.api_client import resolve_api_config_path
from vlnce_baselines.vlm.support.save_manager import (
    SaveManager,
    get_episode_detail_dir,
    get_episode_detail_path_candidates,
    get_episode_log_path_candidates,
)
from vlnce_baselines.controllers.vlm_navigation_controller import VLMNavigationController
from vlnce_baselines.runtime.results_report import generate_results_report


MIN_EPISODE_ID = 1
MAX_EPISODE_ID = 1800
_DATASET_EPISODE_ID_CACHE: Dict[str, List[int]] = {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLM自动导航系统")

    parser.add_argument("--exp-config", type=str, required=True, help="Habitat配置文件")
    parser.add_argument("--episode-id", type=int, default=0, help="起始Episode ID")
    parser.add_argument(
        "--episode-ids",
        type=str,
        default=None,
        help="指定episode ID列表，逗号分隔（如 '832,701,231'）",
    )
    parser.add_argument("--num-episodes", type=int, default=1, help="运行Episode数量（连续或随机）")
    parser.add_argument("--random", action="store_true", help="随机选择episodes而非连续运行")
    parser.add_argument("--results-dir", type=str, default=None, help="结果保存目录")

    parser.add_argument(
        "--vlm-api-config",
        "--config",
        type=str,
        dest="vlm_api_config",
        default="vlnce_baselines/config/api/vlm_api_config.yaml",
        help="统一 API 配置文件路径（LLM/VLM 共用，推荐）",
    )

    parser.add_argument(
        "--max-subtask-steps",
        type=int,
        default=5,
        help="每个子任务最大步数（达到后强制触发验证，默认5步）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode最大总步数（覆盖配置文件，默认使用配置文件值）",
    )
    parser.add_argument(
        "--skip-sr1",
        "--skip-existing-sr1",
        action="store_true",
        dest="skip_sr1",
        help="跳过结果目录中已有 SR=1 最佳结果的 episode",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="并行运行的worker数量（默认1，表示串行）",
    )
    parser.add_argument("--auto", action="store_true", help="全自动运行（无需确认）")
    return parser


def load_runtime_config(args: argparse.Namespace):
    config = get_config(args.exp_config, [])
    if args.max_steps is not None:
        config.defrost()
        config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_steps
        config.freeze()
        if not bool(getattr(args, "suppress_runtime_prints", False)):
            print(f"\n⚙️  覆盖最大步数: {args.max_steps} (命令行参数)")
    return config


def _summarize_episode_ids(episode_ids: List[int], preview_count: int = 12) -> str:
    if not episode_ids:
        return "0 episodes"
    ordered = [int(item) for item in episode_ids]
    if len(ordered) <= preview_count:
        return ",".join(str(item) for item in ordered)
    head = ",".join(str(item) for item in ordered[: preview_count // 2])
    tail = ",".join(str(item) for item in ordered[-(preview_count // 2):])
    return f"{head} ... {tail}"


def _get_episode_records_log_path(results_dir: str, episode_id: int) -> str:
    episode_dir = get_episode_detail_dir(results_dir, episode_id)
    records_dir = os.path.join(episode_dir, "records")
    os.makedirs(records_dir, exist_ok=True)
    return os.path.join(records_dir, f"episode_{int(episode_id)}.log")


def _get_episode_latest_result_path(results_dir: str, episode_id: int) -> str:
    episode_dir = get_episode_detail_dir(results_dir, episode_id)
    records_dir = os.path.join(episode_dir, "records")
    os.makedirs(records_dir, exist_ok=True)
    return os.path.join(records_dir, "result_latest.json")


@contextlib.contextmanager
def _redirect_process_output_to_file(log_path: str, mode: str = "w"):
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


def _extract_episode_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    candidate_paths = [
        str(result.get("latest_result_file") or "").strip(),
        str(result.get("result_file") or "").strip(),
    ]
    for path in candidate_paths:
        if not path:
            continue
        metrics = _load_json_if_exists(path)
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


def _build_episode_console_summary(
    *,
    episode_id: int,
    index: int,
    total: int,
    result: Dict[str, Any],
    metrics: Dict[str, Any],
    worker_index: int = 0,
    worker_count: int = 0,
) -> str:
    status = "OK" if bool(result.get("success", False)) else "FAIL"
    steps = int(result.get("steps", result.get("total_steps", 0)) or 0)
    reason = str(result.get("reason") or "").strip()
    error = str(result.get("error") or "").strip()

    parts = [
        (
            f"[W{worker_index}/{worker_count} {index}/{total}]"
            if worker_index > 0 and worker_count > 0
            else f"[{index}/{total}]"
        ),
        f"Episode {episode_id}",
        status,
        f"steps={steps}",
    ]

    if metrics:
        ne = metrics.get("ne", -1.0)
        sr = metrics.get("sr", 0)
        osr = metrics.get("osr", 0)
        spl = metrics.get("spl", 0.0)
        ndtw = metrics.get("ndtw", 0.0)
        try:
            parts.append(f"NE={float(ne):.3f}m")
        except Exception:
            pass
        parts.append(f"OSR={int(osr)}")
        parts.append(f"SR={int(sr)}")
        try:
            parts.append(f"SPL={float(spl):.4f}")
        except Exception:
            pass
        try:
            parts.append(f"nDTW={float(ndtw):.4f}")
        except Exception:
            pass

    if reason:
        parts.append(f"reason={reason}")
    if error:
        parts.append(f"error={error}")

    return " | ".join(parts)


def _build_episode_start_summary(
    *,
    episode_id: int,
    index: int,
    total: int,
    worker_index: int = 0,
    worker_count: int = 0,
) -> str:
    parts = [
        (
            f"[W{worker_index}/{worker_count} {index}/{total}]"
            if worker_index > 0 and worker_count > 0
            else f"[{index}/{total}]"
        ),
        f"Episode {episode_id}",
        "START",
    ]
    return " | ".join(parts)


def _get_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_dataset_path(data_path: str) -> str:
    if not data_path:
        return ""
    if os.path.isabs(data_path) and os.path.exists(data_path):
        return data_path

    candidates = [
        os.path.abspath(data_path),
        os.path.abspath(os.path.join(os.getcwd(), data_path)),
        os.path.abspath(os.path.join(_get_repo_root(), data_path)),
    ]
    deduped: List[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    for candidate in deduped:
        if os.path.exists(candidate):
            return candidate
    return deduped[0] if deduped else ""


def _load_dataset_episode_ids(config) -> List[int]:
    data_path = str(getattr(config.TASK_CONFIG.DATASET, "DATA_PATH", "") or "").strip()
    resolved_path = _resolve_dataset_path(data_path)
    if not resolved_path:
        return []
    if resolved_path in _DATASET_EPISODE_ID_CACHE:
        return list(_DATASET_EPISODE_ID_CACHE[resolved_path])

    episode_ids: List[int] = []
    try:
        opener = gzip.open if resolved_path.endswith(".gz") else open
        with opener(resolved_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        episodes = payload.get("episodes", []) if isinstance(payload, dict) else payload
        if not isinstance(episodes, list):
            episodes = []
        for item in episodes:
            if not isinstance(item, dict):
                continue
            try:
                episode_ids.append(int(item.get("episode_id")))
            except Exception:
                continue
    except Exception:
        episode_ids = []

    episode_ids = sorted(set(episode_ids))
    _DATASET_EPISODE_ID_CACHE[resolved_path] = list(episode_ids)
    return episode_ids


def _get_available_episode_ids(config) -> List[int]:
    episode_ids = _load_dataset_episode_ids(config)
    if episode_ids:
        return episode_ids
    return list(range(MIN_EPISODE_ID, MAX_EPISODE_ID + 1))


def resolve_episode_ids(args: argparse.Namespace, config) -> List[int]:
    available_episode_ids = _get_available_episode_ids(config)
    available_episode_set = set(available_episode_ids)
    min_episode_id = int(available_episode_ids[0]) if available_episode_ids else MIN_EPISODE_ID
    max_episode_id = int(available_episode_ids[-1]) if available_episode_ids else MAX_EPISODE_ID

    if args.episode_ids:
        episode_ids = [int(x.strip()) for x in args.episode_ids.split(",")]
        invalid_ids = [eid for eid in episode_ids if eid not in available_episode_set]
        if invalid_ids:
            print(
                f"\n❌ 错误: 以下episode ID超出有效范围 "
                f"[{min_episode_id}, {max_episode_id}] 或不在当前数据集中: {invalid_ids}"
            )
            return []
        print(f"\n📝 指定运行 {len(episode_ids)} 个episodes")
        print(f"📊 Preview: {_summarize_episode_ids(episode_ids)}")
        return episode_ids

    if args.random:
        random_seed = int(time.time() * 1000) % (2**32)
        random.seed(random_seed)
        print(f"\n🎲 随机选择模式（从当前数据集有效范围 [{min_episode_id}, {max_episode_id}] 中选择）")
        print(f"   🎯 随机种子: {random_seed}")
        num_to_sample = min(args.num_episodes, len(available_episode_ids))
        if num_to_sample == 0:
            print("\n❌ 错误: 请求的episode数量为0")
            return []

        episode_ids = random.sample(list(available_episode_ids), num_to_sample)
        print(f"📊 随机选择了 {len(episode_ids)} 个episodes")
        print(f"📊 Preview: {_summarize_episode_ids(episode_ids)}")
        return episode_ids

    start_id = args.episode_id
    end_id = args.episode_id + args.num_episodes - 1

    if start_id < min_episode_id:
        print(f"\n❌ 错误: 起始episode ID {start_id} 小于最小值 {min_episode_id}")
        print(f"   建议使用: --episode-id {min_episode_id}")
        return []

    if start_id > max_episode_id:
        print(f"\n❌ 错误: 起始episode ID {start_id} 超过当前数据集最大值 {max_episode_id}")
        return []

    clipped_end_id = min(end_id, max_episode_id)
    if clipped_end_id < end_id:
        print(
            f"\n⚠️  请求区间 {start_id}-{end_id} 超出当前数据集末尾，"
            f"自动截断为 {start_id}-{clipped_end_id}"
        )

    episode_ids = [eid for eid in available_episode_ids if start_id <= eid <= clipped_end_id]
    if not episode_ids:
        print("\n❌ 错误: 请求区间内没有可运行的episodes")
        return []
    print(f"\n📋 连续运行 episodes {episode_ids[0]} 到 {episode_ids[-1]} (共{len(episode_ids)}个)")
    return episode_ids


def _load_json_if_exists(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _result_has_sr1(result: Dict[str, Any]) -> bool:
    return SaveManager.result_has_sr1(result)


def _result_is_complete_sr1(result: Dict[str, Any]) -> bool:
    return _result_has_sr1(result) and SaveManager.is_complete_saved_result(result)


def _episode_has_existing_sr1(results_dir: str, episode_id: int) -> bool:
    candidate_paths = []
    candidate_paths.extend(get_episode_log_path_candidates(results_dir, episode_id))
    for detail_dir in get_episode_detail_path_candidates(results_dir, episode_id):
        candidate_paths.extend([
            os.path.join(detail_dir, "records", "result.json"),
            os.path.join(detail_dir, "records", "result_latest.json"),
        ])

    deduped_paths = []
    for path in candidate_paths:
        if path not in deduped_paths:
            deduped_paths.append(path)

    for path in deduped_paths:
        loaded = _load_json_if_exists(path)
        if _result_is_complete_sr1(loaded):
            return True
    return False


def filter_episode_ids(args: argparse.Namespace, config, episode_ids: List[int]) -> List[int]:
    if not args.skip_sr1:
        return episode_ids

    results_dir = os.path.abspath(args.results_dir or config.RESULTS_DIR or "")
    if not results_dir:
        print("\n⚠️  未提供结果目录，无法启用 skip-sr1，继续运行全部 episodes")
        return episode_ids

    kept_episode_ids: List[int] = []
    skipped_episode_ids: List[int] = []
    for episode_id in episode_ids:
        if _episode_has_existing_sr1(results_dir, episode_id):
            skipped_episode_ids.append(int(episode_id))
        else:
            kept_episode_ids.append(int(episode_id))

    print(
        f"\n🧹 skip-sr1: 跳过 {len(skipped_episode_ids)} 个已存在 SR=1 最佳结果的 episodes，"
        f"保留 {len(kept_episode_ids)} 个待运行 episodes"
    )
    if skipped_episode_ids:
        preview = skipped_episode_ids[:12]
        preview_text = ",".join(str(item) for item in preview)
        suffix = " ..." if len(skipped_episode_ids) > len(preview) else ""
        print(f"⏭️  已跳过: {preview_text}{suffix}")

    return kept_episode_ids


def build_episode_config(base_config, args: argparse.Namespace, episode_id: int):
    episode_config = base_config.clone()
    episode_config.defrost()
    episode_config = ConfigHelper.setup_episode_config(episode_config, [episode_id], num_environments=1)
    if args.results_dir:
        episode_config = ConfigHelper.setup_results_dir(episode_config, args.results_dir)
    episode_config = ConfigHelper.setup_navigation_config(episode_config)
    episode_config.freeze()
    return episode_config


def create_controller(
    episode_config,
    args: argparse.Namespace,
) -> Tuple[VLMNavigationController, str]:
    unified_config = resolve_api_config_path(args.vlm_api_config)
    controller = VLMNavigationController(episode_config, config_path=unified_config)
    return controller, unified_config


def run_single_episode(
    base_config,
    args: argparse.Namespace,
    episode_id: int,
    index: int,
    total: int,
) -> Dict[str, Any]:
    controller = None
    console_result: Dict[str, Any] = {
        "episode_id": int(episode_id),
        "success": False,
        "steps": 0,
        "episode_duration_including_failed_s": 0.0,
        "episode_duration_excluding_failed_s": 0.0,
        "api_total_duration_s": 0.0,
        "failed_wasted_duration_s": 0.0,
        "error": None,
    }
    results_dir = os.path.abspath(args.results_dir or base_config.RESULTS_DIR or os.getcwd())
    episode_log_path = _get_episode_records_log_path(results_dir, episode_id)
    latest_result_path = _get_episode_latest_result_path(results_dir, episode_id)
    print(
        _build_episode_start_summary(
            episode_id=episode_id,
            index=index,
            total=total,
            worker_index=int(getattr(args, "worker_index", 0) or 0),
            worker_count=int(getattr(args, "worker_count", 0) or 0),
        ),
        flush=True,
    )

    try:
        with _redirect_process_output_to_file(episode_log_path):
            print(f"\n{'='*80}")
            print(f"🔄 [{index}/{total}] 开始Episode {episode_id}")
            print(f"{'='*80}")

            episode_config = build_episode_config(base_config, args, episode_id)
            controller, config_desc = create_controller(episode_config, args)
            controller.reset_episode(episode_id=episode_id)

            max_steps = episode_config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
            print(f"\n📝 指令: {controller.current_instruction}")
            print(f"⚙️  配置: Episode {episode_id} | 最大步数 {max_steps} (从 Habitat 配置)")
            print(f"🔧 API config: {config_desc}")

            result = controller.run_vlm_navigation(max_subtask_steps=args.max_subtask_steps)
            total_steps = result.get("total_steps", result.get("steps", 0))

            console_result = {
                "episode_id": episode_id,
                "success": result["success"],
                "steps": total_steps,
                "episode_duration_including_failed_s": result.get(
                    "episode_duration_including_failed_s",
                    result.get("episode_duration_s", 0.0),
                ),
                "episode_duration_excluding_failed_s": result.get(
                    "episode_duration_excluding_failed_s",
                    result.get("episode_duration_s", 0.0),
                ),
                "api_total_duration_s": (result.get("api_summary") or {}).get("total_duration_s", 0.0),
                "failed_wasted_duration_s": result.get("failed_wasted_duration_s", 0.0),
                "error": None,
                "reason": result.get("reason", ""),
                "result_file": result.get("result_file", ""),
                "latest_result_file": latest_result_path,
                "episode_log_path": episode_log_path,
            }
    except Exception as exc:
        import traceback

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
        with _redirect_process_output_to_file(episode_log_path, mode="a"):
            print(f"\n❌ Episode {episode_id} 运行失败: {error_msg}")
            print("\n完整错误堆栈:")
            traceback.print_exc()
        console_result = {
            "episode_id": episode_id,
            "success": finalized_success,
            "steps": finalized_steps,
            "episode_duration_including_failed_s": timing_summary.get("episode_duration_including_failed_s", 0.0),
            "episode_duration_excluding_failed_s": timing_summary.get("episode_duration_excluding_failed_s", 0.0),
            "api_total_duration_s": (timing_summary.get("api_summary") or {}).get("total_duration_s", 0.0),
            "failed_wasted_duration_s": timing_summary.get("failed_wasted_duration_s", 0.0),
            "error": error_msg,
            "result_file": "",
            "latest_result_file": latest_result_path,
            "episode_log_path": episode_log_path,
        }
    finally:
        if controller is not None:
            try:
                controller.envs.close()
            except Exception as cleanup_error:
                print(f"⚠️  清理环境时出错: {cleanup_error}")
    metrics = _extract_episode_metrics(console_result)
    print(
        _build_episode_console_summary(
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
    args: argparse.Namespace,
    episode_id: int,
    index: int,
    total: int,
    worker_index: int,
    worker_count: int,
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
        skip_sr1=False,
        parallel_workers=1,
        worker_index=int(job_spec["worker_index"]),
        worker_count=int(job_spec["worker_count"]),
        suppress_runtime_prints=True,
        auto=True,
    )

    config = load_runtime_config(args)
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
        ),
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
    print(f"\n🚀 并行模式: {worker_count} 个workers（动态补位）")
    print(f"📊 待运行 episodes: {total}")

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


def maybe_generate_report(args: argparse.Namespace, config, verbose: bool = True) -> None:
    results_dir = args.results_dir or config.RESULTS_DIR
    if not results_dir:
        return

    try:
        report = generate_results_report(
            results_dir,
            save=True,
            debug=False,
            verbose=verbose,
        )
        if not verbose:
            summary_path = report.get("saved_paths", {}).get("summary", os.path.join(results_dir, "summary.txt"))
            print(f"📄 评估报告已更新: {summary_path}")
    except FileNotFoundError:
        return
    except Exception as exc:
        print(f"⚠️  无法生成评估报告: {exc}")


def print_batch_summary(results_summary: List[Dict[str, Any]], args: argparse.Namespace, config) -> None:
    print(f"\n\n{'='*80}")
    print("📊 批量运行总结")
    print(f"{'='*80}")

    success_count = sum(1 for result in results_summary if result["success"])
    total_count = len(results_summary)
    results_dir = args.results_dir or config.RESULTS_DIR

    if total_count > 0:
        success_rate = success_count / total_count * 100.0
        avg_steps = sum(result["steps"] for result in results_summary) / total_count
        avg_episode_duration_including_failed_s = (
            sum(float(result.get("episode_duration_including_failed_s", 0.0) or 0.0) for result in results_summary)
            / total_count
        )
        avg_episode_duration_excluding_failed_s = (
            sum(float(result.get("episode_duration_excluding_failed_s", 0.0) or 0.0) for result in results_summary)
            / total_count
        )
        avg_api_total_duration_s = (
            sum(float(result.get("api_total_duration_s", 0.0) or 0.0) for result in results_summary)
            / total_count
        )
        avg_failed_wasted_duration_s = (
            sum(float(result.get("failed_wasted_duration_s", 0.0) or 0.0) for result in results_summary)
            / total_count
        )
        print(f"\n✅ 成功: {success_count}/{total_count} ({success_rate:.1f}%)")
        print(f"❌ 失败: {total_count - success_count}/{total_count}")
    else:
        success_rate = 0.0
        avg_steps = 0.0
        avg_episode_duration_including_failed_s = 0.0
        avg_episode_duration_excluding_failed_s = 0.0
        avg_api_total_duration_s = 0.0
        avg_failed_wasted_duration_s = 0.0
        print("\n⚠️  没有运行任何episodes")

    failed_results = [result for result in results_summary if not bool(result.get("success", False))]
    if failed_results:
        preview = failed_results[:20]
        preview_text = ", ".join(str(int(item.get("episode_id", 0))) for item in preview)
        suffix = " ..." if len(failed_results) > len(preview) else ""
        print(f"\n❌ 失败episodes预览: {preview_text}{suffix}")

    print(f"\n{'='*80}")
    print("\n" + "=" * 60)
    print("🏁 批量评估完成")
    print("=" * 60)
    print(f"✅ 成功率: {success_count}/{total_count} ({success_rate:.1f}%)")
    print(f"📊 平均步数: {avg_steps:.1f}")
    print(
        "⏱️  平均时间: "
        f"API成功={avg_api_total_duration_s:.2f}s | "
        f"Episode(去失败)={avg_episode_duration_excluding_failed_s:.2f}s | "
        f"Episode(含失败)={avg_episode_duration_including_failed_s:.2f}s | "
        f"失败浪费={avg_failed_wasted_duration_s:.2f}s"
    )
    print(f"📁 结果目录: {results_dir}")
    print(f"📄 详细报告: {os.path.join(results_dir, 'summary.txt')}")
    print("=" * 60)


def run_navigation_from_args(args: argparse.Namespace) -> int:
    config = load_runtime_config(args)
    episode_ids = resolve_episode_ids(args, config)
    episode_ids = filter_episode_ids(args, config, episode_ids)

    if not episode_ids:
        if args.skip_sr1:
            print("\n✅ 没有需要运行的 episodes：目标范围内都已有 SR=1 最佳结果")
            maybe_generate_report(args, config, verbose=False)
            return 0
        print("\n❌ 错误: 没有可运行的episodes")
        return 1

    results_summary: List[Dict[str, Any]] = []
    parallel_workers = max(1, int(args.parallel_workers or 1))
    if parallel_workers > 1 and len(episode_ids) > 1:
        results_summary = run_parallel_episodes(config, args, episode_ids)
    else:
        for idx, episode_id in enumerate(episode_ids, 1):
            results_summary.append(
                run_single_episode(
                    config,
                    args,
                    episode_id=episode_id,
                    index=idx,
                    total=len(episode_ids),
                )
            )

    if len(episode_ids) > 1:
        print_batch_summary(results_summary, args, config)
        maybe_generate_report(args, config, verbose=True)
    else:
        maybe_generate_report(args, config, verbose=False)
    return 0
