import argparse
import csv
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

from navigation_system.runtime.storage.artifacts import (
    get_episode_log_path,
    iter_all_episode_log_paths,
)


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


def _load_result_payload(filepath: str) -> Optional[Dict[str, Any]]:
    filename = os.path.basename(filepath)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        print(f"⚠️  Failed to read {filename}: {exc}")
        return None
    return _normalize_report_item(payload)


def _bounded_load_workers(load_workers: int, item_count: int) -> int:
    try:
        parsed_workers = int(load_workers)
    except (TypeError, ValueError):
        parsed_workers = 1
    return max(1, min(parsed_workers, max(1, int(item_count))))


def _load_payloads(filepaths: Iterable[str], load_workers: int = 1) -> List[Optional[Dict[str, Any]]]:
    filepath_list = list(filepaths)
    worker_count = _bounded_load_workers(load_workers, len(filepath_list))
    if worker_count <= 1:
        return [_load_result_payload(filepath) for filepath in filepath_list]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_load_result_payload, filepath_list))


def _load_payload_with_mtime(filepath: str) -> Optional[Tuple[str, Dict[str, Any], float]]:
    payload = _load_result_payload(filepath)
    if payload is None:
        return None
    try:
        current_mtime = os.path.getmtime(filepath)
    except OSError:
        current_mtime = float("-inf")
    return filepath, payload, current_mtime


def _load_payloads_with_mtime(
    filepaths: Iterable[str],
    load_workers: int = 1,
) -> List[Tuple[str, Dict[str, Any], float]]:
    filepath_list = list(filepaths)
    worker_count = _bounded_load_workers(load_workers, len(filepath_list))
    if worker_count <= 1:
        loaded_items = [_load_payload_with_mtime(filepath) for filepath in filepath_list]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            loaded_items = list(executor.map(_load_payload_with_mtime, filepath_list))
    return [item for item in loaded_items if item is not None]


def _default_load_workers() -> int:
    raw_value = str(os.environ.get("SPACEVLN_REPORT_WORKERS", "") or "").strip()
    if raw_value:
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 1

    # I/O-heavy JSON aggregation: use an aggressive but bounded default.
    cpu_count = int(os.cpu_count() or 4)
    recommended = max(8, cpu_count * 4)
    return min(64, recommended)


def load_results(results_dir: str, *, load_workers: int = 1) -> List[Dict[str, Any]]:
    log_dir = os.path.join(results_dir, "log")
    if not os.path.exists(log_dir):
        return []

    results_by_episode: Dict[str, Dict[str, Any]] = {}
    result_mtime_by_episode: Dict[str, float] = {}
    for filepath, payload, current_mtime in _load_payloads_with_mtime(
        iter_all_episode_log_paths(results_dir),
        load_workers=load_workers,
    ):
        filename = os.path.basename(filepath)
        episode_key = str(payload.get("episode_id", filename))
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


def load_results_in_episode_range(
    results_dir: str,
    *,
    start_episode_id: int,
    end_episode_id: int,
    load_workers: int = 1,
) -> List[Dict[str, Any]]:
    log_dir = os.path.join(results_dir, "log")
    if not os.path.exists(log_dir):
        return []

    episode_filepaths = (
        get_episode_log_path(results_dir, episode_id)
        for episode_id in range(int(start_episode_id), int(end_episode_id) + 1)
    )
    return [
        payload
        for payload in _load_payloads(episode_filepaths, load_workers=load_workers)
        if payload is not None
    ]


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
    print("📊 SpaceVLN evaluation summary")
    print("=" * 80)
    print(f"\n🎯 Unified metrics:")
    print(f"  NE:    {metrics['avg_ne']:.3f}m")
    print(f"  OSR:   {metrics['osr_count']}/{n} ({metrics['avg_osr']:.3f})")
    print(f"  SR:    {metrics['sr_count']}/{n} ({metrics['avg_sr']:.3f})")
    print(f"  SPL:   {metrics['avg_spl']:.3f}")
    print(f"  nDTW:  {metrics['avg_ndtw']:.3f}")
    print(f"\n⏱️  Timing:")
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
        "  API total:      "
        f"ok={timing['api_total_duration_s']:.2f}s "
        f"| fail={timing['api_failed_total_duration_s']:.2f}s"
    )
    print(
        "  Failure waste:  "
        f"retry_wait={timing['failed_retry_wait_duration_s_total']:.2f}s "
        f"| total={timing['failed_wasted_duration_s_total']:.2f}s"
    )
    print(f"  Episode avg:    {timing['episode_duration_s_avg']:.2f}s")
    print(f"\n{'=' * 80}")


def _resolve_success_distance_m(exp_config: Optional[str]) -> float:
    default_success_distance_m = 3.0
    if not exp_config:
        return default_success_distance_m
    try:
        from navigation_system.config import get_config
        from navigation_system.config.core.params.thresholds import EVAL_SUCCESS_DISTANCE_M

        config = get_config(exp_config, [])
        return float(getattr(config.EVAL, "SUCCESS_DISTANCE_M", EVAL_SUCCESS_DISTANCE_M))
    except Exception:
        return default_success_distance_m


def print_debug_info(metrics: Dict[str, Any], success_distance_m: float) -> None:
    results = metrics["detailed_results"]
    print(f"\n🔍 Metric debug:")
    print(f"{'=' * 80}")
    print(f"\nEpisode details:")
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

    print(f"\n⚠️  Consistency checks:")
    for item in results:
        ep_id = item.get("episode_id", "?")
        ne = _as_float(item.get("ne", -1.0), -1.0)
        osr = _as_int(item.get("osr", 0), 0)
        sr = _as_int(item.get("sr", 0), 0)
        spl = _as_float(item.get("spl", 0.0), 0.0)
        ndtw = _as_float(item.get("ndtw", 0.0), 0.0)

        if sr == 1 and ne > success_distance_m:
            print(f"  ❌ Episode {ep_id}: SR=1 but NE={ne:.3f}m > {success_distance_m:g}m")
        if sr == 0 and 0 <= ne < success_distance_m:
            print(f"  ⚠️  Episode {ep_id}: SR=0 but NE={ne:.3f}m < {success_distance_m:g}m")
        if sr == 0 and spl > 0:
            print(f"  ❌ Episode {ep_id}: SR=0 but SPL={spl:.4f} > 0")
        if osr < sr:
            print(f"  ❌ Episode {ep_id}: OSR={osr} < SR={sr}")
        if not 0.0 <= ndtw <= 1.0:
            print(f"  ⚠️  Episode {ep_id}: nDTW={ndtw:.4f} is outside [0, 1]")
        if ne < 0:
            print(f"  ⚠️  Episode {ep_id}: NE={ne} < 0")

    print(f"{'=' * 80}")


def save_summary(metrics: Dict[str, Any], output_path: str) -> str:
    n = metrics["total_episodes"]
    timing = metrics["timing"]
    content = f"""
================================================================================
📊 SpaceVLN evaluation summary
================================================================================

🎯 Unified metrics:
  NE:    {metrics['avg_ne']:.3f}m
  OSR:   {metrics['osr_count']}/{n} ({metrics['avg_osr']:.3f})
  SR:    {metrics['sr_count']}/{n} ({metrics['avg_sr']:.3f})
  SPL:   {metrics['avg_spl']:.3f}
  nDTW:  {metrics['avg_ndtw']:.3f}

⏱️  Timing:
  Thinking API: avg={timing['thinking_api_avg_duration_s']:.2f}s | total={timing['thinking_api_total_duration_s']:.2f}s | ok={timing['thinking_api_count']} fail={timing['thinking_api_failed_count']}
  Action API:   avg={timing['action_api_avg_duration_s']:.2f}s | total={timing['action_api_total_duration_s']:.2f}s | ok={timing['action_api_count']} fail={timing['action_api_failed_count']}
  API total:      ok={timing['api_total_duration_s']:.2f}s | fail={timing['api_failed_total_duration_s']:.2f}s
  Failure waste:  retry_wait={timing['failed_retry_wait_duration_s_total']:.2f}s | total={timing['failed_wasted_duration_s_total']:.2f}s
  Episode avg:    {timing['episode_duration_s_avg']:.2f}s

================================================================================
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def save_metrics_json(metrics: Dict[str, Any], output_path: str) -> str:
    payload = {
        "total_episodes": int(metrics.get("total_episodes", 0) or 0),
        "avg_ne": _as_float(metrics.get("avg_ne", -1.0), -1.0),
        "avg_osr": _as_float(metrics.get("avg_osr", 0.0), 0.0),
        "avg_sr": _as_float(metrics.get("avg_sr", 0.0), 0.0),
        "avg_spl": _as_float(metrics.get("avg_spl", 0.0), 0.0),
        "avg_ndtw": _as_float(metrics.get("avg_ndtw", 0.0), 0.0),
        "timing": dict(metrics.get("timing") or {}),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


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
    summary_only: bool = False,
    debug: bool = False,
    verbose: bool = True,
    start_episode_id: Optional[int] = None,
    end_episode_id: Optional[int] = None,
    output_dir: Optional[str] = None,
    exp_config: Optional[str] = None,
    load_workers: int = 1,
) -> Dict[str, Any]:
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    if verbose:
        print(f"📂 Loading results: {results_dir}")
        if int(load_workers) > 1:
            print(f"⚙️  Parallel JSON workers: {int(load_workers)}")

    use_direct_range_load = start_episode_id is not None and end_episode_id is not None
    if use_direct_range_load:
        results = load_results_in_episode_range(
            results_dir,
            start_episode_id=int(start_episode_id),
            end_episode_id=int(end_episode_id),
            load_workers=load_workers,
        )
    else:
        all_results = load_results(results_dir, load_workers=load_workers)
        results = filter_results_by_episode_range(
            all_results,
            start_episode_id=start_episode_id,
            end_episode_id=end_episode_id,
        )

    if not results:
        if start_episode_id is not None or end_episode_id is not None:
            raise FileNotFoundError(
                f"No episode results found in range [{start_episode_id}, {end_episode_id}]"
            )
        raise FileNotFoundError(f"No episode results found under: {os.path.join(results_dir, 'log')}")
    if (start_episode_id is not None or end_episode_id is not None) and not use_direct_range_load:
        filtered_results = filter_results_by_episode_range(
            results,
            start_episode_id=start_episode_id,
            end_episode_id=end_episode_id,
        )
        if not filtered_results:
            raise FileNotFoundError(
                f"No episode results found in range [{start_episode_id}, {end_episode_id}]"
            )
        results = filtered_results

    report_output_dir = _resolve_report_output_dir(
        results_dir,
        start_episode_id=start_episode_id,
        end_episode_id=end_episode_id,
        output_dir=output_dir,
    )

    if verbose:
        if start_episode_id is not None or end_episode_id is not None:
            print(f"✅ Loaded {len(results)} episodes ([{start_episode_id}, {end_episode_id}])")
            print(f"📁 Partial report directory: {report_output_dir}")
        else:
            print(f"✅ Loaded {len(results)} episodes")

    metrics = compute_metrics(results)
    saved_paths: Dict[str, str] = {}

    if verbose:
        print_summary(metrics)
        if debug:
            success_distance_m = _resolve_success_distance_m(exp_config)
            print_debug_info(metrics, success_distance_m)

    if save:
        os.makedirs(report_output_dir, exist_ok=True)
        summary_path = os.path.join(report_output_dir, "summary.txt")
        save_summary(metrics, summary_path)
        saved_paths["summary"] = summary_path
        metrics_json_path = os.path.join(report_output_dir, "metrics.json")
        save_metrics_json(metrics, metrics_json_path)
        saved_paths["metrics_json"] = metrics_json_path
        if not summary_only:
            saved_paths.update(save_episode_tables(results, metrics, report_output_dir))
        if verbose:
            print(f"📋 Saved summary report: {summary_path}")
            print(f"📋 Saved metrics JSON: {metrics_json_path}")
            if not summary_only:
                print(f"📋 Saved episode CSV: {saved_paths['csv']}")
                print(f"📋 Saved episode Markdown: {saved_paths['md']}")

    return {
        "results": results,
        "metrics": metrics,
        "saved_paths": saved_paths,
    }


def build_results_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze SpaceVLN evaluation results")
    parser.add_argument("--path", type=str, required=True, help="Results directory")
    parser.add_argument(
        "--exp-config",
        type=str,
        default="navigation_system/config/experiments/vlnce/r2r_eval.yaml",
        help="Experiment config path used to resolve evaluation thresholds such as success distance",
    )
    parser.add_argument("--save", action="store_true", help="Save summary and result tables")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Fast mode: save only `summary` + `metrics.json`, skip episode CSV/Markdown",
    )
    parser.add_argument("--debug", action="store_true", help="Print per-episode debug information")
    parser.add_argument("--start-id", type=int, default=None, help="Only include episodes from this id")
    parser.add_argument("--end-id", type=int, default=None, help="Only include episodes up to this id")
    parser.add_argument("--output-dir", type=str, default=None, help="Report output directory")
    parser.add_argument(
        "--load-workers",
        type=int,
        default=_default_load_workers(),
        help="Number of workers for loading episode JSON files in parallel",
    )
    return parser


def run_results_report_from_args(args: argparse.Namespace) -> int:
    try:
        if args.start_id is not None and args.end_id is not None and args.end_id < args.start_id:
            print(f"❌ end-id cannot be smaller than start-id: {args.start_id} -> {args.end_id}")
            return 1
        generate_results_report(
            args.path,
            save=bool(args.save),
            summary_only=bool(args.summary_only),
            debug=bool(args.debug),
            verbose=True,
            start_episode_id=args.start_id,
            end_episode_id=args.end_id,
            output_dir=args.output_dir,
            exp_config=args.exp_config,
            load_workers=args.load_workers,
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
