"""Isolated ablation runtime orchestrator."""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

from navigation_system.ablation.config import (
    ABLATION_CONFIG_ENV,
    get_default_ablation_config_path,
    resolve_ablation_config_path,
    save_ablation_manifest,
)
from navigation_system.ablation.runtime.execution import (
    load_runtime_config,
    run_parallel_episodes,
    run_single_episode,
)
from navigation_system.ablation.runtime.profiles import (
    ABLATION_STANDARD_RUNTIME_PROFILE,
    AblationRuntimeProfile,
)
from navigation_system.runtime.episode_selection import (
    filter_episode_ids,
    resolve_episode_ids,
)
from navigation_system.runtime.results_report import generate_results_report


def build_arg_parser(
    profile: AblationRuntimeProfile = ABLATION_STANDARD_RUNTIME_PROFILE,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpaceVLN isolated ablation runtime")

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
        help="结果总根目录；运行时自动追加 vlnce/ablation/消融项/模型名",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="最终结果目录（高级覆盖项；指定后不会再自动追加 vlnce/ablation/消融项/模型名）",
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
        "--ablation-config",
        type=str,
        default=os.environ.get(ABLATION_CONFIG_ENV, get_default_ablation_config_path()),
        help="消融实验配置文件路径",
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
        generate_results_report(
            results_dir,
            save=True,
            debug=False,
            verbose=verbose,
        )
    except FileNotFoundError:
        return
    except Exception as exc:
        print(f"⚠️  无法生成评估报告: {exc}")


def prepare_ablation_runtime(
    args: argparse.Namespace,
    profile: AblationRuntimeProfile = ABLATION_STANDARD_RUNTIME_PROFILE,
):
    resolved_ablation_config = resolve_ablation_config_path(getattr(args, "ablation_config", None))
    args.ablation_config = resolved_ablation_config
    os.environ[ABLATION_CONFIG_ENV] = resolved_ablation_config

    config = load_runtime_config(args, profile=profile)
    return config, resolved_ablation_config


def ensure_results_dir_ready(
    results_dir: str,
    *,
    config_path: str,
) -> Optional[str]:
    target_dir = str(results_dir or "").strip()
    if not target_dir:
        return None
    os.makedirs(target_dir, exist_ok=True)
    return save_ablation_manifest(
        target_dir,
        config_path=config_path,
    )


def run_episode_batch(
    config,
    args: argparse.Namespace,
    episode_ids: List[int],
    *,
    profile: AblationRuntimeProfile,
) -> List[dict]:
    results_summary: List[dict] = []
    parallel_workers = max(1, int(args.parallel_workers or 1))
    if parallel_workers > 1 and len(episode_ids) > 1:
        return run_parallel_episodes(
            config,
            args,
            episode_ids,
            profile=profile,
        )

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
    return results_summary


def run_navigation_from_args(
    args: argparse.Namespace,
    profile: AblationRuntimeProfile = ABLATION_STANDARD_RUNTIME_PROFILE,
) -> int:
    config, resolved_ablation_config = prepare_ablation_runtime(args, profile=profile)
    results_dir = str(getattr(config.PATHS, "RESULTS_DIR", "") or "").strip()
    if results_dir:
        try:
            ensure_results_dir_ready(
                results_dir,
                config_path=resolved_ablation_config,
            )
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
            maybe_generate_report(args, config, verbose=False)
            if profile.post_run_hook is not None:
                profile.post_run_hook(args, config)
            return 0
        print("\n❌ 错误: 没有可运行的episodes")
        return 1

    results_summary = run_episode_batch(
        config,
        args,
        episode_ids,
        profile=profile,
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
    maybe_generate_report(args, config, verbose=False)
    if profile.post_run_hook is not None:
        profile.post_run_hook(args, config)
    return 1 if failed_results else 0


__all__ = [
    "build_arg_parser",
    "ensure_results_dir_ready",
    "maybe_generate_report",
    "prepare_ablation_runtime",
    "run_episode_batch",
    "run_navigation_from_args",
]
