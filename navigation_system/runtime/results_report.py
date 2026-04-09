import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional

from navigation_system.config import get_config
from navigation_system.config.core.params.thresholds import EVAL_SUCCESS_DISTANCE_M
from navigation_system.runtime.storage.artifacts import iter_all_episode_log_paths


def check_inf_nan(value: Any) -> Any:
    if isinstance(value, (int, float)) and (math.isinf(value) or math.isnan(value)):
        return 0
    return value


def _as_float(value: Any, default: float = 0.0) -> float:
    value = check_inf_nan(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    value = check_inf_nan(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _summary_value(item: Dict[str, Any], prefix: str, key: str, default: float = 0.0) -> float:
    summary = item.get(f"{prefix}_api_summary")
    if not isinstance(summary, dict):
        return _as_float(default, default)
    return _as_float(summary.get(key, default), default)


def _summary_count(item: Dict[str, Any], prefix: str, key: str, default: int = 0) -> int:
    summary = item.get(f"{prefix}_api_summary")
    if not isinstance(summary, dict):
        return _as_int(default, default)
    return _as_int(summary.get(key, default), default)


def _episode_duration_s(item: Dict[str, Any]) -> float:
    return _as_float(item.get("episode_duration_s", 0.0))


def _normalize_report_item(item: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(item or {})
    if "episode_duration_s" not in payload:
        payload["episode_duration_s"] = _as_float(
            payload.get("episode_duration_including_failed_s", 0.0),
            0.0,
        )

    for prefix in ("thinking", "action"):
        summary_key = f"{prefix}_api_summary"
        summary = payload.get(summary_key)
        if isinstance(summary, dict):
            continue
        payload[summary_key] = {
            "count": _as_int(payload.get(f"{prefix}_api_count", 0), 0),
            "failure_count": _as_int(payload.get(f"{prefix}_api_failed_count", 0), 0),
            "avg_duration_s": _as_float(payload.get(f"{prefix}_api_avg_duration_s", 0.0), 0.0),
            "total_duration_s": _as_float(payload.get(f"{prefix}_api_total_duration_s", 0.0), 0.0),
            "failed_total_duration_s": _as_float(
                payload.get(f"{prefix}_api_failed_total_duration_s", 0.0),
                0.0,
            ),
        }
    return payload


def _compute_timing_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    thinking_count = sum(_summary_count(item, "thinking", "count", 0) for item in results)
    thinking_fail_count = sum(_summary_count(item, "thinking", "failure_count", 0) for item in results)
    thinking_total_s = sum(_summary_value(item, "thinking", "total_duration_s", 0.0) for item in results)
    thinking_failed_total_s = sum(
        _summary_value(item, "thinking", "failed_total_duration_s", 0.0) for item in results
    )

    action_count = sum(_summary_count(item, "action", "count", 0) for item in results)
    action_fail_count = sum(_summary_count(item, "action", "failure_count", 0) for item in results)
    action_total_s = sum(_summary_value(item, "action", "total_duration_s", 0.0) for item in results)
    action_failed_total_s = sum(
        _summary_value(item, "action", "failed_total_duration_s", 0.0) for item in results
    )

    api_total_s = thinking_total_s + action_total_s
    api_failed_total_s = thinking_failed_total_s + action_failed_total_s
    failed_retry_wait_total_s = sum(_as_float(item.get("failed_retry_wait_duration_s", 0.0)) for item in results)
    failed_wasted_total_s = sum(_as_float(item.get("failed_wasted_duration_s", 0.0)) for item in results)
    episode_total_s = sum(_episode_duration_s(item) for item in results)

    return {
        "thinking_api_count": thinking_count,
        "thinking_api_failed_count": thinking_fail_count,
        "thinking_api_total_duration_s": thinking_total_s,
        "thinking_api_failed_total_duration_s": thinking_failed_total_s,
        "thinking_api_avg_duration_s": thinking_total_s / thinking_count if thinking_count > 0 else 0.0,
        "action_api_count": action_count,
        "action_api_failed_count": action_fail_count,
        "action_api_total_duration_s": action_total_s,
        "action_api_failed_total_duration_s": action_failed_total_s,
        "action_api_avg_duration_s": action_total_s / action_count if action_count > 0 else 0.0,
        "api_total_duration_s": api_total_s,
        "api_failed_total_duration_s": api_failed_total_s,
        "failed_retry_wait_duration_s_total": failed_retry_wait_total_s,
        "failed_wasted_duration_s_total": failed_wasted_total_s,
        "episode_duration_s_total": episode_total_s,
        "episode_duration_s_avg": episode_total_s / n if n > 0 else 0.0,
    }


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    log_dir = os.path.join(results_dir, "log")
    if not os.path.exists(log_dir):
        return []

    results_by_episode: Dict[str, Dict[str, Any]] = {}
    result_mtime_by_episode: Dict[str, float] = {}
    for filepath in iter_all_episode_log_paths(results_dir):
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"⚠️  读取文件失败 {filename}: {exc}")
            continue

        episode_key = str(payload.get("episode_id", filename))
        payload = _normalize_report_item(payload)
        current_mtime = os.path.getmtime(filepath)
        previous_mtime = result_mtime_by_episode.get(episode_key, float("-inf"))
        if episode_key not in results_by_episode or current_mtime >= previous_mtime:
            results_by_episode[episode_key] = payload
            result_mtime_by_episode[episode_key] = current_mtime

    def _episode_sort_key(item: Dict[str, Any]) -> Any:
        episode_id = item.get("episode_id", "")
        try:
            return (0, int(episode_id))
        except (TypeError, ValueError):
            return (1, str(episode_id))

    return sorted(results_by_episode.values(), key=_episode_sort_key)


def _episode_id_as_int(item: Dict[str, Any]) -> Optional[int]:
    try:
        return int(item.get("episode_id"))
    except (TypeError, ValueError):
        return None


def filter_results_by_episode_range(
    results: List[Dict[str, Any]],
    *,
    start_episode_id: Optional[int] = None,
    end_episode_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if start_episode_id is None and end_episode_id is None:
        return list(results)

    filtered: List[Dict[str, Any]] = []
    for item in results:
        episode_id = _episode_id_as_int(item)
        if episode_id is None:
            continue
        if start_episode_id is not None and episode_id < int(start_episode_id):
            continue
        if end_episode_id is not None and episode_id > int(end_episode_id):
            continue
        filtered.append(item)
    return filtered


def _resolve_report_output_dir(
    results_dir: str,
    *,
    start_episode_id: Optional[int] = None,
    end_episode_id: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> str:
    if output_dir:
        return os.path.abspath(output_dir)
    if start_episode_id is None and end_episode_id is None:
        return results_dir

    start_label = "start" if start_episode_id is None else str(int(start_episode_id))
    end_label = "end" if end_episode_id is None else str(int(end_episode_id))
    return os.path.join(results_dir, "reports", f"{start_label}-{end_label}")


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    n = len(results)
    ne_list = [_as_float(item.get("ne", -1.0), -1.0) for item in results]
    osr_list = [_as_int(item.get("osr", 0), 0) for item in results]
    sr_list = [_as_int(item.get("sr", 0), 0) for item in results]
    spl_list = [_as_float(item.get("spl", 0.0), 0.0) for item in results]
    ndtw_list = [_as_float(item.get("ndtw", 0.0), 0.0) for item in results]

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
        "timing": _compute_timing_metrics(results),
        "detailed_results": results,
    }


def print_summary(metrics: Dict[str, Any]) -> None:
    n = metrics["total_episodes"]
    timing = metrics["timing"]
    print("\n" + "=" * 80)
    print("📊 SpaceVLN 评估结果汇总")
    print("=" * 80)
    print(f"\n🎯 统一指标:")
    print(f"  NE:    {metrics['avg_ne']:.3f}m")
    print(f"  OSR:   {metrics['osr_count']}/{n} ({metrics['avg_osr']:.3f})")
    print(f"  SR:    {metrics['sr_count']}/{n} ({metrics['avg_sr']:.3f})")
    print(f"  SPL:   {metrics['avg_spl']:.3f}")
    print(f"  nDTW:  {metrics['avg_ndtw']:.3f}")
    print(f"\n⏱️  计时统计:")
    print(
        "  Thinking API: "
        f"avg={timing['thinking_api_avg_duration_s']:.2f}s "
        f"| total={timing['thinking_api_total_duration_s']:.2f}s "
        f"| ok={timing['thinking_api_count']} fail={timing['thinking_api_failed_count']}"
    )
    print(
        "  Action API:   "
        f"avg={timing['action_api_avg_duration_s']:.2f}s "
        f"| total={timing['action_api_total_duration_s']:.2f}s "
        f"| ok={timing['action_api_count']} fail={timing['action_api_failed_count']}"
    )
    print(
        "  API总耗时:     "
        f"ok={timing['api_total_duration_s']:.2f}s "
        f"| fail={timing['api_failed_total_duration_s']:.2f}s"
    )
    print(
        "  失败浪费:       "
        f"retry_wait={timing['failed_retry_wait_duration_s_total']:.2f}s "
        f"| total={timing['failed_wasted_duration_s_total']:.2f}s"
    )
    print(f"  Episode平均:   {timing['episode_duration_s_avg']:.2f}s")
    print(f"\n{'=' * 80}")


def _resolve_success_distance_m(exp_config: Optional[str]) -> float:
    if not exp_config:
        return float(EVAL_SUCCESS_DISTANCE_M)
    try:
        config = get_config(exp_config, [])
        return float(getattr(config.EVAL, "SUCCESS_DISTANCE_M", EVAL_SUCCESS_DISTANCE_M))
    except Exception:
        return float(EVAL_SUCCESS_DISTANCE_M)


def print_debug_info(metrics: Dict[str, Any], success_distance_m: float) -> None:
    results = metrics["detailed_results"]
    print(f"\n🔍 指标计算调试信息:")
    print(f"{'=' * 80}")
    print(f"\nEpisode详情:")
    print(f"{'ID':<6} {'NE(m)':<10} {'OSR':<6} {'SR':<6} {'SPL':<8} {'nDTW':<8}")
    print(f"{'-' * 56}")

    for item in results:
        ep_id = item.get("episode_id", "?")
        ne = _as_float(item.get("ne", -1.0), -1.0)
        osr = _as_int(item.get("osr", 0), 0)
        sr = _as_int(item.get("sr", 0), 0)
        spl = _as_float(item.get("spl", 0.0), 0.0)
        ndtw = _as_float(item.get("ndtw", 0.0), 0.0)
        print(f"{ep_id:<6} {ne:<10.3f} {osr:<6} {sr:<6} {spl:<8.4f} {ndtw:<8.4f}")

    print(f"\n⚠️  异常检测:")
    for item in results:
        ep_id = item.get("episode_id", "?")
        ne = _as_float(item.get("ne", -1.0), -1.0)
        osr = _as_int(item.get("osr", 0), 0)
        sr = _as_int(item.get("sr", 0), 0)
        spl = _as_float(item.get("spl", 0.0), 0.0)
        ndtw = _as_float(item.get("ndtw", 0.0), 0.0)

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
    timing = metrics["timing"]
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

⏱️  计时统计:
  Thinking API: avg={timing['thinking_api_avg_duration_s']:.2f}s | total={timing['thinking_api_total_duration_s']:.2f}s | ok={timing['thinking_api_count']} fail={timing['thinking_api_failed_count']}
  Action API:   avg={timing['action_api_avg_duration_s']:.2f}s | total={timing['action_api_total_duration_s']:.2f}s | ok={timing['action_api_count']} fail={timing['action_api_failed_count']}
  API总耗时:     ok={timing['api_total_duration_s']:.2f}s | fail={timing['api_failed_total_duration_s']:.2f}s
  失败浪费:       retry_wait={timing['failed_retry_wait_duration_s_total']:.2f}s | total={timing['failed_wasted_duration_s_total']:.2f}s
  Episode平均:   {timing['episode_duration_s_avg']:.2f}s

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


def _build_episode_row(item: Dict[str, Any]) -> Dict[str, str]:
    think_avg = _summary_value(item, "thinking", "avg_duration_s", 0.0)
    think_total = _summary_value(item, "thinking", "total_duration_s", 0.0)
    action_avg = _summary_value(item, "action", "avg_duration_s", 0.0)
    action_total = _summary_value(item, "action", "total_duration_s", 0.0)
    api_total = think_total + action_total
    return {
        "episode_id": str(item.get("episode_id", "")),
        "NE": _format_metric_value(_as_float(item.get("ne", -1.0), -1.0), 3),
        "OSR": str(_as_int(item.get("osr", 0), 0)),
        "SR": str(_as_int(item.get("sr", 0), 0)),
        "SPL": _format_metric_value(_as_float(item.get("spl", 0.0), 0.0), 4),
        "nDTW": _format_metric_value(_as_float(item.get("ndtw", 0.0), 0.0), 4),
        "ThinkAvg(s)": _format_metric_value(think_avg, 3),
        "ThinkTot(s)": _format_metric_value(think_total, 3),
        "ActAvg(s)": _format_metric_value(action_avg, 3),
        "ActTot(s)": _format_metric_value(action_total, 3),
        "API(s)": _format_metric_value(api_total, 3),
        "FailWaste(s)": _format_metric_value(_as_float(item.get("failed_wasted_duration_s", 0.0), 0.0), 3),
        "Episode(s)": _format_metric_value(_episode_duration_s(item), 3),
    }


def save_episode_tables(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    results_dir: str,
) -> Dict[str, str]:
    csv_path = os.path.join(results_dir, "episode_results.csv")
    md_path = os.path.join(results_dir, "episode_results.md")
    sorted_results = sorted(results, key=lambda item: int(item.get("episode_id", -1)))
    timing = metrics["timing"]
    headers = [
        "episode_id",
        "NE",
        "OSR",
        "SR",
        "SPL",
        "nDTW",
        "ThinkAvg(s)",
        "ThinkTot(s)",
        "ActAvg(s)",
        "ActTot(s)",
        "API(s)",
        "FailWaste(s)",
        "Episode(s)",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for item in sorted_results:
            writer.writerow(_build_episode_row(item))
        writer.writerow(
            {
                "episode_id": "SUMMARY",
                "NE": _format_metric_value(metrics["avg_ne"], 3),
                "OSR": _format_metric_value(metrics["avg_osr"], 4),
                "SR": _format_metric_value(metrics["avg_sr"], 4),
                "SPL": _format_metric_value(metrics["avg_spl"], 4),
                "nDTW": _format_metric_value(metrics["avg_ndtw"], 4),
                "ThinkAvg(s)": _format_metric_value(timing["thinking_api_avg_duration_s"], 3),
                "ThinkTot(s)": _format_metric_value(timing["thinking_api_total_duration_s"], 3),
                "ActAvg(s)": _format_metric_value(timing["action_api_avg_duration_s"], 3),
                "ActTot(s)": _format_metric_value(timing["action_api_total_duration_s"], 3),
                "API(s)": _format_metric_value(timing["api_total_duration_s"], 3),
                "FailWaste(s)": _format_metric_value(timing["failed_wasted_duration_s_total"], 3),
                "Episode(s)": _format_metric_value(timing["episode_duration_s_avg"], 3),
            }
        )

    md_lines = [
        "# Episode Results",
        "",
        "| Episode | NE(m) | OSR | SR | SPL | nDTW | ThinkAvg(s) | ThinkTot(s) | ActAvg(s) | ActTot(s) | API(s) | FailWaste(s) | Episode(s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in sorted_results:
        row = _build_episode_row(item)
        md_lines.append(
            "| {episode} | {ne} | {osr} | {sr} | {spl} | {ndtw} | {think_avg} | {think_total} | {act_avg} | {act_total} | {api_total} | {fail_waste} | {episode_s} |".format(
                episode=row["episode_id"],
                ne=row["NE"],
                osr=row["OSR"],
                sr=row["SR"],
                spl=row["SPL"],
                ndtw=row["nDTW"],
                think_avg=row["ThinkAvg(s)"],
                think_total=row["ThinkTot(s)"],
                act_avg=row["ActAvg(s)"],
                act_total=row["ActTot(s)"],
                api_total=row["API(s)"],
                fail_waste=row["FailWaste(s)"],
                episode_s=row["Episode(s)"],
            )
        )
    md_lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Episodes | NE(m) | OSR | SR | SPL | nDTW | ThinkAvg(s) | ThinkTot(s) | ActAvg(s) | ActTot(s) | API(s) | FailWaste(s) | Episode(s) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "| {total} | {avg_ne} | {avg_osr} | {avg_sr} | {avg_spl} | {avg_ndtw} | {think_avg} | {think_total} | {action_avg} | {action_total} | {api_total} | {fail_waste} | {episode_avg} |".format(
                total=metrics["total_episodes"],
                avg_ne=_format_metric_value(metrics["avg_ne"], 3),
                avg_osr=_format_metric_value(metrics["avg_osr"], 4),
                avg_sr=_format_metric_value(metrics["avg_sr"], 4),
                avg_spl=_format_metric_value(metrics["avg_spl"], 4),
                avg_ndtw=_format_metric_value(metrics["avg_ndtw"], 4),
                think_avg=_format_metric_value(timing["thinking_api_avg_duration_s"], 3),
                think_total=_format_metric_value(timing["thinking_api_total_duration_s"], 3),
                action_avg=_format_metric_value(timing["action_api_avg_duration_s"], 3),
                action_total=_format_metric_value(timing["action_api_total_duration_s"], 3),
                api_total=_format_metric_value(timing["api_total_duration_s"], 3),
                fail_waste=_format_metric_value(timing["failed_wasted_duration_s_total"], 3),
                episode_avg=_format_metric_value(timing["episode_duration_s_avg"], 3),
            ),
            "",
            "> `ThinkTot/ActTot/API(s)/FailWaste(s)` in Summary are batch totals; `ThinkAvg/ActAvg/Episode(s)` are batch averages.",
            "",
            "> Repeated evaluation of the same episode keeps only the better result in `log/<range>/episode_XXX.json`.",
        ]
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return {"csv": csv_path, "md": md_path}


def generate_results_report(
    results_dir: str,
    *,
    save: bool = True,
    debug: bool = False,
    verbose: bool = True,
    start_episode_id: Optional[int] = None,
    end_episode_id: Optional[int] = None,
    output_dir: Optional[str] = None,
    exp_config: Optional[str] = None,
) -> Dict[str, Any]:
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"目录不存在: {results_dir}")

    if verbose:
        print(f"📂 加载结果: {results_dir}")

    all_results = load_results(results_dir)
    if not all_results:
        raise FileNotFoundError(f"未找到任何episode结果: {os.path.join(results_dir, 'log')}")

    results = filter_results_by_episode_range(
        all_results,
        start_episode_id=start_episode_id,
        end_episode_id=end_episode_id,
    )
    if not results:
        if start_episode_id is not None or end_episode_id is not None:
            raise FileNotFoundError(
                f"未找到指定范围内的episode结果: [{start_episode_id}, {end_episode_id}]"
            )
        raise FileNotFoundError(f"未找到任何episode结果: {os.path.join(results_dir, 'log')}")

    report_output_dir = _resolve_report_output_dir(
        results_dir,
        start_episode_id=start_episode_id,
        end_episode_id=end_episode_id,
        output_dir=output_dir,
    )

    if verbose:
        if start_episode_id is not None or end_episode_id is not None:
            print(
                f"✅ 总共加载 {len(all_results)} 个episode，"
                f"范围过滤后保留 {len(results)} 个episode "
                f"([{start_episode_id}, {end_episode_id}])"
            )
            print(f"📁 部分报告输出目录: {report_output_dir}")
        else:
            print(f"✅ 加载了 {len(results)} 个episode")

    metrics = compute_metrics(results)
    saved_paths: Dict[str, str] = {}
    success_distance_m = _resolve_success_distance_m(exp_config)

    if verbose:
        print_summary(metrics)
        if debug:
            print_debug_info(metrics, success_distance_m)

    if save:
        os.makedirs(report_output_dir, exist_ok=True)
        summary_path = os.path.join(report_output_dir, "summary.txt")
        save_summary(metrics, summary_path)
        saved_paths["summary"] = summary_path
        saved_paths.update(save_episode_tables(results, metrics, report_output_dir))
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
    parser.add_argument(
        "--exp-config",
        type=str,
        default="navigation_system/config/experiments/r2r_eval.yaml",
        help="实验配置文件路径，用于读取 success distance 等评测参数",
    )
    parser.add_argument("--save", action="store_true", help="保存 summary 和结果表格")
    parser.add_argument("--debug", action="store_true", help="打印逐 episode 调试信息")
    parser.add_argument("--start-id", type=int, default=None, help="只统计起始 episode ID")
    parser.add_argument("--end-id", type=int, default=None, help="只统计结束 episode ID")
    parser.add_argument("--output-dir", type=str, default=None, help="报告输出目录")
    return parser


def run_results_report_from_args(args: argparse.Namespace) -> int:
    try:
        if args.start_id is not None and args.end_id is not None and args.end_id < args.start_id:
            print(f"❌ end-id 不能小于 start-id: {args.start_id} -> {args.end_id}")
            return 1
        generate_results_report(
            args.path,
            save=bool(args.save),
            debug=bool(args.debug),
            verbose=True,
            start_episode_id=args.start_id,
            end_episode_id=args.end_id,
            output_dir=args.output_dir,
            exp_config=args.exp_config,
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
