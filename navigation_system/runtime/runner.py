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
    parser = argparse.ArgumentParser(description="SpaceVLN navigation runner")

    parser.add_argument("--exp-config", type=str, required=True, help="Habitat experiment config")
    parser.add_argument("--episode-id", type=int, default=0, help="Starting episode id")
    parser.add_argument(
        "--episode-ids",
        type=str,
        default=None,
        help="Explicit episode id list, comma separated (for example: '832,701,231')",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample random episodes instead of a contiguous range",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Results root override (compatibility option; unified nav_ws/result is preferred)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Final results directory override (compatibility option)",
    )

    parser.add_argument(
        "--vlm-api-config",
        "--config",
        type=str,
        dest="vlm_api_config",
        default=profile.default_api_config_path,
        help="Unified API config path shared by LLM and VLM calls",
    )

    parser.add_argument(
        "--max-subtask-steps",
        type=int,
        default=5,
        help="Maximum low-level steps per subtask before forced verification",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode (overrides the config default)",
    )
    parser.add_argument(
        "--skip-sr1",
        "--skip-existing-sr1",
        action="store_true",
        dest="skip_sr1",
        help="Skip episodes that already have a complete SR=1 result in the results directory",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of parallel workers (1 means serial execution)",
    )
    parser.add_argument("--auto", action="store_true", help="Run without interactive confirmation")
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
        print(f"⚠️  Failed to generate evaluation report: {exc}")

    if not verbose:
        return

    metrics = dict(report_payload.get("metrics") or {})
    saved_paths = dict(report_payload.get("saved_paths") or {})
    total_episodes = int(metrics.get("total_episodes", 0) or 0)
    if total_episodes <= 0:
        return

    timing = dict(metrics.get("timing") or {})
    print(
        "\n📊 Evaluation Summary "
        f"| episodes={total_episodes} "
        f"| NE={float(metrics.get('avg_ne', -1.0)):.3f}m "
        f"| OSR={float(metrics.get('avg_osr', 0.0)):.3f} "
        f"| SR={float(metrics.get('avg_sr', 0.0)):.3f} "
        f"| SPL={float(metrics.get('avg_spl', 0.0)):.3f} "
        f"| nDTW={float(metrics.get('avg_ndtw', 0.0)):.3f}"
    )
    if timing:
        print(
            "⏱️  Timing Summary "
            f"| episode_avg={float(timing.get('episode_duration_s_avg', 0.0)):.2f}s "
            f"| api_total={float(timing.get('api_total_duration_s', 0.0)):.2f}s"
        )
    if saved_paths:
        summary_path = str(saved_paths.get("summary") or "").strip()
        csv_path = str(saved_paths.get("csv") or "").strip()
        if summary_path:
            print(f"📄 Summary file: {summary_path}")
        if csv_path:
            print(f"📄 Episode table: {csv_path}")


def run_navigation_from_args(
    args: argparse.Namespace,
    profile: NavigationRuntimeProfile = STANDARD_RUNTIME_PROFILE,
) -> int:
    if str(getattr(args, "results_root", "") or "").strip() or str(getattr(args, "results_dir", "") or "").strip():
        print(
            "⚠️  Detected results path overrides (--results-root/--results-dir)."
            " The unified nav_ws/result layout is recommended unless compatibility requires an override."
        )

    config = load_runtime_config(args, profile=profile)
    results_dir = str(getattr(config.PATHS, "RESULTS_DIR", "") or "").strip()
    if results_dir:
        try:
            os.makedirs(results_dir, exist_ok=True)
        except Exception as exc:
            print(f"\n❌ Results directory is not writable: {results_dir}")
            print(f"   Error: {exc}")
            print("   Re-run with --results-dir pointing to a writable directory if needed.")
            return 1
    episode_ids = resolve_episode_ids(args, config)
    episode_ids = filter_episode_ids(args, config, episode_ids)

    if not episode_ids:
        if args.skip_sr1:
            print("\n✅ No episodes need to run: the requested range already has SR=1 results")
            maybe_generate_report(args, config, verbose=True)
            if profile.post_run_hook is not None:
                profile.post_run_hook(args, config)
            return 0
        print("\n❌ Error: no runnable episodes were selected")
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
        print("\n⚠️ The following episodes failed:", flush=True)
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
