"""Thin runtime orchestrator for SpaceVLN navigation."""

import argparse
import os
from typing import List

from navigation_system.runtime.episode_selection import (
    filter_episode_ids,
    resolve_episode_ids,
)
from navigation_system.runtime.execution import (
    load_runtime_config,
    run_parallel_episodes,
    run_single_episode,
)
from navigation_system.runtime.profiles import (
    NavigationRuntimeProfile,
    STANDARD_RUNTIME_PROFILE,
)
from navigation_system.runtime.results_report import generate_results_report


def build_arg_parser(
    profile: NavigationRuntimeProfile = STANDARD_RUNTIME_PROFILE,
) -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="结果总根目录（兼容保留；默认建议使用统一路径 nav_ws/result）",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="最终结果目录（兼容保留；默认建议使用统一路径 nav_ws/result）",
    )

    parser.add_argument(
        "--vlm-api-config",
        "--config",
        type=str,
        dest="vlm_api_config",
        default=profile.default_api_config_path,
        help="统一 API 配置文件路径（LLM/VLM 共用）",
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
        help="跳过结果目录中已有 SR=1 完整结果的 episode",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="并行运行的worker数量（默认1，表示串行）",
    )
    parser.add_argument("--auto", action="store_true", help="全自动运行（无需确认）")
    return parser


def maybe_generate_report(args: argparse.Namespace, config, verbose: bool = True) -> None:
    results_dir = args.results_dir or config.PATHS.RESULTS_DIR
    if not results_dir:
        return

    try:
        report_payload = generate_results_report(
            results_dir,
            save=True,
            debug=False,
            verbose=False,
        )
    except FileNotFoundError:
        return
    except Exception as exc:
        print(f"⚠️  无法生成评估报告: {exc}")

    if not verbose:
        return

    metrics = dict(report_payload.get("metrics") or {})
    saved_paths = dict(report_payload.get("saved_paths") or {})
    total_episodes = int(metrics.get("total_episodes", 0) or 0)
    if total_episodes <= 0:
        return

    timing = dict(metrics.get("timing") or {})
    print(
        "\n📊 评测汇总 "
        f"| episodes={total_episodes} "
        f"| NE={float(metrics.get('avg_ne', -1.0)):.3f}m "
        f"| OSR={float(metrics.get('avg_osr', 0.0)):.3f} "
        f"| SR={float(metrics.get('avg_sr', 0.0)):.3f} "
        f"| SPL={float(metrics.get('avg_spl', 0.0)):.3f} "
        f"| nDTW={float(metrics.get('avg_ndtw', 0.0)):.3f}"
    )
    if timing:
        print(
            "⏱️  时延汇总 "
            f"| episode_avg={float(timing.get('episode_duration_s_avg', 0.0)):.2f}s "
            f"| api_total={float(timing.get('api_total_duration_s', 0.0)):.2f}s"
        )
    if saved_paths:
        summary_path = str(saved_paths.get("summary") or "").strip()
        csv_path = str(saved_paths.get("csv") or "").strip()
        if summary_path:
            print(f"📄 汇总文件: {summary_path}")
        if csv_path:
            print(f"📄 Episode表: {csv_path}")


def run_navigation_from_args(
    args: argparse.Namespace,
    profile: NavigationRuntimeProfile = STANDARD_RUNTIME_PROFILE,
) -> int:
    if str(getattr(args, "results_root", "") or "").strip() or str(getattr(args, "results_dir", "") or "").strip():
        print(
            "⚠️  检测到 results 路径覆盖参数（--results-root/--results-dir）。"
            " 当前推荐统一默认路径（nav_ws/result），仅在兼容场景下使用覆盖参数。"
        )

    config = load_runtime_config(args, profile=profile)
    results_dir = str(getattr(config.PATHS, "RESULTS_DIR", "") or "").strip()
    if results_dir:
        try:
            os.makedirs(results_dir, exist_ok=True)
        except Exception as exc:
            print(f"\n❌ 结果目录不可写: {results_dir}")
            print(f"   具体错误: {exc}")
            print("   可通过 --results-dir 指定一个当前用户可写的目录重新运行。")
            return 1
    episode_ids = resolve_episode_ids(args, config)
    episode_ids = filter_episode_ids(args, config, episode_ids)

    if not episode_ids:
        if args.skip_sr1:
            print("\n✅ 没有需要运行的 episodes：目标范围内都已有 SR=1 最佳结果")
            maybe_generate_report(args, config, verbose=True)
            if profile.post_run_hook is not None:
                profile.post_run_hook(args, config)
            return 0
        print("\n❌ 错误: 没有可运行的episodes")
        return 1

    results_summary: List[dict] = []
    parallel_workers = max(1, int(args.parallel_workers or 1))
    if parallel_workers > 1 and len(episode_ids) > 1:
        results_summary = run_parallel_episodes(
            config,
            args,
            episode_ids,
            profile=profile,
        )
    else:
        for idx, episode_id in enumerate(episode_ids, 1):
            results_summary.append(
                run_single_episode(
                    config,
                    args,
                    episode_id=episode_id,
                    index=idx,
                    total=len(episode_ids),
                    profile=profile,
                )
            )

    failed_results = [item for item in results_summary if not bool(item.get("success", False))]
    if failed_results:
        print("\n⚠️ 以下 episodes 运行失败:", flush=True)
        for item in failed_results:
            episode_id = item.get("episode_id", "?")
            error = str(item.get("error") or "").strip()
            reason = str(item.get("reason") or "").strip()
            parts = [f"Episode {episode_id}"]
            if reason:
                parts.append(f"reason={reason}")
            if error:
                parts.append(f"error={error}")
            print("   " + " | ".join(parts), flush=True)

    del results_summary
    maybe_generate_report(args, config, verbose=True)
    if profile.post_run_hook is not None:
        profile.post_run_hook(args, config)
    return 1 if failed_results else 0


__all__ = [
    "build_arg_parser",
    "maybe_generate_report",
    "run_navigation_from_args",
]
