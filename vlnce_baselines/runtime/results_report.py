import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List

from vlnce_baselines.config.core.params.thresholds import EVAL_SUCCESS_DISTANCE_M


def check_inf_nan(value: Any) -> Any:
    if isinstance(value, (int, float)) and (math.isinf(value) or math.isnan(value)):
        return 0
    return value

def load_results(results_dir: str) -> List[Dict[str, Any]]:
    log_dir = os.path.join(results_dir, "log")
    if not os.path.exists(log_dir):
        return []

    results: List[Dict[str, Any]] = []
    for filename in sorted(os.listdir(log_dir)):
        if not (filename.startswith("episode_") and filename.endswith(".json")):
            continue
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception as exc:
            print(f"⚠️  读取文件失败 {filename}: {exc}")
    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    n = len(results)
    ne_list = [check_inf_nan(item.get("ne", -1.0)) for item in results]
    osr_list = [check_inf_nan(item.get("osr", 0)) for item in results]
    sr_list = [check_inf_nan(item.get("sr", 0)) for item in results]
    spl_list = [check_inf_nan(item.get("spl", 0.0)) for item in results]
    ndtw_list = [check_inf_nan(item.get("ndtw", 0.0)) for item in results]

    valid_ne = [value for value in ne_list if value >= 0]
    sr_count = sum(sr_list)
    osr_count = sum(osr_list)
    return {
        "total_episodes": n,
        "avg_ne": sum(valid_ne) / len(valid_ne) if valid_ne else -1.0,
        "osr_count": osr_count,
        "avg_osr": osr_count / n if n > 0 else 0.0,
        "sr_count": sr_count,
        "avg_sr": sr_count / n if n > 0 else 0.0,
        "avg_spl": sum(spl_list) / n if n > 0 else 0.0,
        "avg_ndtw": sum(ndtw_list) / n if n > 0 else 0.0,
        "detailed_results": results,
    }


def print_summary(metrics: Dict[str, Any]) -> None:
    n = metrics["total_episodes"]
    print("\n" + "=" * 80)
    print("📊 SpaceVLN 评估结果汇总")
    print("=" * 80)
    print(f"\n🎯 统一指标:")
    print(f"  NE:    {metrics['avg_ne']:.3f}m")
    print(f"  OSR:   {metrics['osr_count']}/{n} ({metrics['avg_osr']:.3f})")
    print(f"  SR:    {metrics['sr_count']}/{n} ({metrics['avg_sr']:.3f})")
    print(f"  SPL:   {metrics['avg_spl']:.3f}")
    print(f"  nDTW:  {metrics['avg_ndtw']:.3f}")
    print(f"\n{'=' * 80}")


def print_debug_info(metrics: Dict[str, Any]) -> None:
    results = metrics["detailed_results"]
    success_distance_m = float(EVAL_SUCCESS_DISTANCE_M)
    print(f"\n🔍 指标计算调试信息:")
    print(f"{'=' * 80}")
    print(f"\nEpisode详情:")
    print(f"{'ID':<6} {'NE(m)':<10} {'OSR':<6} {'SR':<6} {'SPL':<8} {'nDTW':<8}")
    print(f"{'-' * 56}")

    for item in results:
        ep_id = item.get("episode_id", "?")
        ne = check_inf_nan(item.get("ne", -1.0))
        osr = check_inf_nan(item.get("osr", 0))
        sr = check_inf_nan(item.get("sr", 0))
        spl = check_inf_nan(item.get("spl", 0.0))
        ndtw = check_inf_nan(item.get("ndtw", 0.0))
        print(f"{ep_id:<6} {ne:<10.3f} {osr:<6} {sr:<6} {spl:<8.4f} {ndtw:<8.4f}")

    print(f"\n⚠️  异常检测:")
    for item in results:
        ep_id = item.get("episode_id", "?")
        ne = check_inf_nan(item.get("ne", -1.0))
        osr = check_inf_nan(item.get("osr", 0))
        sr = check_inf_nan(item.get("sr", 0))
        spl = check_inf_nan(item.get("spl", 0.0))
        ndtw = check_inf_nan(item.get("ndtw", 0.0))

        if sr == 1 and ne > success_distance_m:
            print(f"  ❌ Episode {ep_id}: SR=1 但 NE={ne:.3f}m > {success_distance_m:g}m")
        if sr == 0 and 0 <= ne < success_distance_m:
            print(f"  ⚠️  Episode {ep_id}: SR=0 但 NE={ne:.3f}m < {success_distance_m:g}m")
        if sr == 0 and spl > 0:
            print(f"  ❌ Episode {ep_id}: SR=0 但 SPL={spl:.4f} > 0")
        if osr < sr:
            print(f"  ❌ Episode {ep_id}: OSR={osr} < SR={sr}")
        if not 0.0 <= ndtw <= 1.0:
            print(f"  ⚠️  Episode {ep_id}: nDTW={ndtw:.4f} 超出[0,1]范围")
        if ne < 0:
            print(f"  ⚠️  Episode {ep_id}: NE={ne} < 0")

    print(f"{'=' * 80}")


def save_summary(metrics: Dict[str, Any], output_path: str) -> str:
    n = metrics["total_episodes"]
    content = f"""
================================================================================
📊 SpaceVLN 评估结果汇总
================================================================================

🎯 统一指标:
  NE:    {metrics['avg_ne']:.3f}m
  OSR:   {metrics['osr_count']}/{n} ({metrics['avg_osr']:.3f})
  SR:    {metrics['sr_count']}/{n} ({metrics['avg_sr']:.3f})
  SPL:   {metrics['avg_spl']:.3f}
  nDTW:  {metrics['avg_ndtw']:.3f}

================================================================================
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def _format_metric_value(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)


def save_episode_tables(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    results_dir: str,
) -> Dict[str, str]:
    csv_path = os.path.join(results_dir, "episode_results.csv")
    md_path = os.path.join(results_dir, "episode_results.md")
    sorted_results = sorted(results, key=lambda item: int(item.get("episode_id", -1)))
    headers = ["episode_id", "NE", "OSR", "SR", "SPL", "nDTW"]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for item in sorted_results:
            writer.writerow({
                "episode_id": item.get("episode_id", ""),
                "NE": _format_metric_value(check_inf_nan(item.get("ne", -1.0)), 3),
                "OSR": str(int(check_inf_nan(item.get("osr", 0)))),
                "SR": str(int(check_inf_nan(item.get("sr", 0)))),
                "SPL": _format_metric_value(check_inf_nan(item.get("spl", 0.0)), 4),
                "nDTW": _format_metric_value(check_inf_nan(item.get("ndtw", 0.0)), 4),
            })
        writer.writerow({
            "episode_id": "AVERAGE",
            "NE": _format_metric_value(metrics["avg_ne"], 3),
            "OSR": _format_metric_value(metrics["avg_osr"], 4),
            "SR": _format_metric_value(metrics["avg_sr"], 4),
            "SPL": _format_metric_value(metrics["avg_spl"], 4),
            "nDTW": _format_metric_value(metrics["avg_ndtw"], 4),
        })

    md_lines = [
        "# Episode Results",
        "",
        "| Episode | NE(m) | OSR | SR | SPL | nDTW |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in sorted_results:
        md_lines.append(
            "| {episode} | {ne} | {osr} | {sr} | {spl} | {ndtw} |".format(
                episode=item.get("episode_id", ""),
                ne=_format_metric_value(check_inf_nan(item.get("ne", -1.0)), 3),
                osr=str(int(check_inf_nan(item.get("osr", 0)))),
                sr=str(int(check_inf_nan(item.get("sr", 0)))),
                spl=_format_metric_value(check_inf_nan(item.get("spl", 0.0)), 4),
                ndtw=_format_metric_value(check_inf_nan(item.get("ndtw", 0.0)), 4),
            )
        )
    md_lines.extend([
        "",
        "## Average",
        "",
        "| Episodes | NE(m) | OSR | SR | SPL | nDTW |",
        "| --- | --- | --- | --- | --- | --- |",
        "| {total} | {avg_ne} | {avg_osr} | {avg_sr} | {avg_spl} | {avg_ndtw} |".format(
            total=metrics["total_episodes"],
            avg_ne=_format_metric_value(metrics["avg_ne"], 3),
            avg_osr=_format_metric_value(metrics["avg_osr"], 4),
            avg_sr=_format_metric_value(metrics["avg_sr"], 4),
            avg_spl=_format_metric_value(metrics["avg_spl"], 4),
            avg_ndtw=_format_metric_value(metrics["avg_ndtw"], 4),
        ),
        "",
        "> Note: repeated evaluation of the same episode keeps only the better result in `log/episode_XXX.json`.",
    ])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return {"csv": csv_path, "md": md_path}


def generate_results_report(
    results_dir: str,
    *,
    save: bool = True,
    debug: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"目录不存在: {results_dir}")

    if verbose:
        print(f"📂 加载结果: {results_dir}")

    results = load_results(results_dir)
    if not results:
        raise FileNotFoundError(f"未找到任何episode结果: {os.path.join(results_dir, 'log')}")

    if verbose:
        print(f"✅ 加载了 {len(results)} 个episode")

    metrics = compute_metrics(results)
    saved_paths: Dict[str, str] = {}

    if verbose:
        print_summary(metrics)
        if debug:
            print_debug_info(metrics)

    if save:
        summary_path = os.path.join(results_dir, "summary.txt")
        save_summary(metrics, summary_path)
        saved_paths["summary"] = summary_path
        saved_paths.update(save_episode_tables(results, metrics, results_dir))
        if verbose:
            print(f"📋 汇总报告已保存: {summary_path}")
            print(f"📋 Episode表格已保存: {saved_paths['csv']}")
            print(f"📋 Markdown表格已保存: {saved_paths['md']}")

    return {
        "results": results,
        "metrics": metrics,
        "saved_paths": saved_paths,
    }


def build_results_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分析 SpaceVLN 评估结果")
    parser.add_argument("--path", type=str, required=True, help="结果目录路径")
    parser.add_argument("--save", action="store_true", help="保存 summary 和结果表格")
    parser.add_argument("--debug", action="store_true", help="打印逐 episode 调试信息")
    return parser


def run_results_report_from_args(args: argparse.Namespace) -> int:
    try:
        generate_results_report(
            args.path,
            save=bool(args.save),
            debug=bool(args.debug),
            verbose=True,
        )
        return 0
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1


def main() -> int:
    parser = build_results_arg_parser()
    args = parser.parse_args()
    return run_results_report_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
