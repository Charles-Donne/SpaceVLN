"""
VLM navigation runtime orchestration.

Keep CLI entrypoints thin by moving episode selection, controller bootstrapping,
batch execution, and summary/report generation into reusable helpers.
"""

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

from vlnce_baselines.config import ConfigHelper, get_config
from vlnce_baselines.vlm.api.api_client import resolve_api_config_path
from vlnce_baselines.vlm.support.save_manager import SaveManager, get_episode_detail_dir
from vlnce_baselines.controllers.vlm_navigation_controller import VLMNavigationController
from vlnce_baselines.runtime.results_report import generate_results_report


MIN_EPISODE_ID = 1
MAX_EPISODE_ID = 1800


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
    parser.add_argument("--auto", action="store_true", help="全自动运行（无需确认）")
    return parser


def load_runtime_config(args: argparse.Namespace):
    config = get_config(args.exp_config, [])
    if args.max_steps is not None:
        config.defrost()
        config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_steps
        config.freeze()
        print(f"\n⚙️  覆盖最大步数: {args.max_steps} (命令行参数)")
    return config


def resolve_episode_ids(args: argparse.Namespace) -> List[int]:
    if args.episode_ids:
        episode_ids = [int(x.strip()) for x in args.episode_ids.split(",")]
        invalid_ids = [eid for eid in episode_ids if eid < MIN_EPISODE_ID or eid > MAX_EPISODE_ID]
        if invalid_ids:
            print(
                f"\n❌ 错误: 以下episode ID超出有效范围 "
                f"[{MIN_EPISODE_ID}, {MAX_EPISODE_ID}]: {invalid_ids}"
            )
            return []
        print(f"\n📝 指定运行 {len(episode_ids)} 个episodes")
        print(f"📊 Episodes: {episode_ids}")
        return episode_ids

    if args.random:
        random_seed = int(time.time() * 1000) % (2**32)
        random.seed(random_seed)
        print(f"\n🎲 随机选择模式（从有效范围 [{MIN_EPISODE_ID}, {MAX_EPISODE_ID}] 中选择）")
        print(f"   🎯 随机种子: {random_seed}")
        print("   ⚠️  不验证episode是否存在，不存在的会自动跳过")

        valid_range = range(MIN_EPISODE_ID, MAX_EPISODE_ID + 1)
        num_to_sample = min(args.num_episodes, len(valid_range))
        if num_to_sample == 0:
            print("\n❌ 错误: 请求的episode数量为0")
            return []

        episode_ids = random.sample(list(valid_range), num_to_sample)
        print(f"📊 随机选择了 {len(episode_ids)} 个episodes: {episode_ids}")
        return episode_ids

    start_id = args.episode_id
    end_id = args.episode_id + args.num_episodes - 1

    if start_id < MIN_EPISODE_ID:
        print(f"\n❌ 错误: 起始episode ID {start_id} 小于最小值 {MIN_EPISODE_ID}")
        print(f"   建议使用: --episode-id {MIN_EPISODE_ID}")
        return []

    if end_id > MAX_EPISODE_ID:
        max_num = MAX_EPISODE_ID - start_id + 1
        print(f"\n❌ 错误: 结束episode ID {end_id} 超过最大值 {MAX_EPISODE_ID}")
        print(f"   建议使用: --num-episodes {max_num} (最多可运行到episode {MAX_EPISODE_ID})")
        return []

    episode_ids = list(range(start_id, end_id + 1))
    print(f"\n📋 连续运行 episodes {start_id} 到 {end_id}")
    print(f"📊 Episodes: {episode_ids}")
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


def _episode_has_existing_sr1(results_dir: str, episode_id: int) -> bool:
    episode_dir = get_episode_detail_dir(results_dir, episode_id)
    candidate_paths = [
        os.path.join(results_dir, "log", f"episode_{int(episode_id)}.json"),
        os.path.join(episode_dir, "records", "result.json"),
        os.path.join(episode_dir, "records", "result_latest.json"),
    ]

    for path in candidate_paths:
        loaded = _load_json_if_exists(path)
        if _result_has_sr1(loaded):
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
        preview = skipped_episode_ids[:20]
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
    print(f"\n{'='*80}")
    print(f"🔄 [{index}/{total}] 开始Episode {episode_id}")
    print(f"{'='*80}")

    controller = None
    try:
        episode_config = build_episode_config(base_config, args, episode_id)
        controller, config_desc = create_controller(episode_config, args)
        controller.reset_episode(episode_id=episode_id)

        max_steps = episode_config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        print(f"\n📝 指令: {controller.current_instruction}")
        print(f"⚙️  配置: Episode {episode_id} | 最大步数 {max_steps} (从 Habitat 配置)")
        print(f"🔧 API config: {config_desc}")

        result = controller.run_vlm_navigation(max_subtask_steps=args.max_subtask_steps)
        total_steps = result.get("total_steps", result.get("steps", 0))

        print(
            f"[Episode] id={episode_id} success={result['success']} steps={total_steps}"
        )
        return {
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
        }
    except Exception as exc:
        import traceback

        error_msg = str(exc)
        timing_summary = {}
        if controller is not None and hasattr(controller, "_build_episode_timing_summary"):
            try:
                timing_summary = controller._build_episode_timing_summary()
            except Exception:
                timing_summary = {}
        print(f"\n❌ Episode {episode_id} 运行失败: {error_msg}")
        print("\n完整错误堆栈:")
        traceback.print_exc()
        return {
            "episode_id": episode_id,
            "success": False,
            "steps": 0,
            "episode_duration_including_failed_s": timing_summary.get("episode_duration_including_failed_s", 0.0),
            "episode_duration_excluding_failed_s": timing_summary.get("episode_duration_excluding_failed_s", 0.0),
            "api_total_duration_s": (timing_summary.get("api_summary") or {}).get("total_duration_s", 0.0),
            "failed_wasted_duration_s": timing_summary.get("failed_wasted_duration_s", 0.0),
            "error": error_msg,
        }
    finally:
        if controller is not None:
            try:
                controller.envs.close()
            except Exception as cleanup_error:
                print(f"⚠️  清理环境时出错: {cleanup_error}")


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

    print("\n详细结果:")
    for result in results_summary:
        status = "✅" if result["success"] else "❌"
        error_msg = f" (错误: {result['error']})" if result["error"] else ""
        print(f"  {status} Episode {result['episode_id']}: 步数={result['steps']}{error_msg}")

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
    episode_ids = resolve_episode_ids(args)
    episode_ids = filter_episode_ids(args, config, episode_ids)

    if not episode_ids:
        if args.skip_sr1:
            print("\n✅ 没有需要运行的 episodes：目标范围内都已有 SR=1 最佳结果")
            maybe_generate_report(args, config, verbose=False)
            return 0
        print("\n❌ 错误: 没有可运行的episodes")
        return 1

    results_summary = []
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
