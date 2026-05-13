#!/usr/bin/env python3
"""Backfill VLM token/cache/cost summaries into existing SpaceVLN results."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navigation_system.runtime.results_report import generate_results_report
from navigation_system.runtime.storage.artifacts import SaveManager
from navigation_system.vlm.reporting.usage import (
    load_price_table,
    merge_vlm_usage_summaries,
    summarize_vlm_usage,
)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f) or {}
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _usage_count(usage_summary: Dict[str, Any]) -> int:
    try:
        return int(((usage_summary or {}).get("overall") or {}).get("count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _merge_usage(payload: Dict[str, Any], usage_summary: Dict[str, Any]) -> bool:
    if _usage_count(usage_summary) <= 0:
        return False

    before = json.dumps(payload.get("vlm_usage_summary", {}), sort_keys=True, ensure_ascii=False)
    payload["vlm_usage_summary"] = usage_summary

    for prefix in ("thinking", "action"):
        api_key = f"{prefix}_api_summary"
        api_summary = dict(payload.get(api_key) or {})
        for key, value in dict(usage_summary.get(prefix) or {}).items():
            if key == "count":
                api_summary["token_count"] = int(value or 0)
            else:
                api_summary[key] = value
        payload[api_key] = api_summary

    after = json.dumps(payload.get("vlm_usage_summary", {}), sort_keys=True, ensure_ascii=False)
    return before != after


def _iter_result_roots(results_root: Path) -> List[Path]:
    roots: List[Path] = []
    for log_dir in sorted(results_root.rglob("log")):
        if not log_dir.is_dir():
            continue
        root = log_dir.parent
        if (root / "detail").is_dir():
            roots.append(root)
    return roots


def _detail_dir_for_log(log_path: Path, result_root: Path) -> Path:
    bucket = log_path.parent.name
    entry_name = log_path.stem
    return result_root / "detail" / bucket / entry_name


def _request_kind(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "thinking" in parts:
        return "thinking"
    if "action" in parts:
        return "action"
    return "other"


def _load_usage_record(item: tuple) -> Optional[tuple]:
    key, kind, info_path = item
    payload = _load_json(info_path)
    if not isinstance(payload, dict):
        return None
    return key, kind, payload


def _build_usage_index(
    result_root: Path,
    *,
    wanted_keys: Iterable[tuple],
    workers: int = 8,
) -> Dict[tuple, Dict[str, Any]]:
    detail_root = result_root / "detail"
    grouped: Dict[tuple, Dict[str, List[Dict[str, Any]]]] = {}
    if not detail_root.is_dir():
        return {}
    wanted = set(wanted_keys)
    price_table = load_price_table()

    info_items: List[tuple] = []
    for current_root, _dirnames, filenames in os.walk(detail_root):
        if "vlm_info.json" not in filenames:
            continue
        info_path = Path(current_root) / "vlm_info.json"
        try:
            relative = info_path.relative_to(detail_root)
        except ValueError:
            continue
        parts = relative.parts
        if len(parts) < 3:
            continue
        key = (parts[0], parts[1])
        if wanted and key not in wanted:
            continue
        info_items.append((key, _request_kind(info_path), info_path))

    worker_count = max(1, min(int(workers or 1), 64))
    if worker_count <= 1:
        loaded_records = [_load_usage_record(item) for item in info_items]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            loaded_records = list(executor.map(_load_usage_record, info_items))

    for loaded in loaded_records:
        if not loaded:
            continue
        key, kind, payload = loaded
        per_episode = grouped.setdefault(key, {"thinking": [], "action": [], "other": []})
        per_episode.setdefault(kind, []).append(payload)

    usage_index: Dict[tuple, Dict[str, Any]] = {}
    for key, payloads in grouped.items():
        thinking = summarize_vlm_usage(payloads.get("thinking", []), price_table=price_table)
        action = summarize_vlm_usage(payloads.get("action", []), price_table=price_table)
        other = summarize_vlm_usage(payloads.get("other", []), price_table=price_table)
        usage_index[key] = {
            "thinking": thinking,
            "action": action,
            "other": other,
            "overall": merge_vlm_usage_summaries([thinking, action, other]),
        }
    return usage_index


def backfill_result_root(result_root: Path, *, write: bool = True, workers: int = 8) -> Dict[str, Any]:
    log_paths = sorted((result_root / "log").rglob("*.json"))
    wanted_keys = {(path.parent.name, path.stem) for path in log_paths}
    usage_index = _build_usage_index(result_root, wanted_keys=wanted_keys, workers=workers)
    updated_logs = 0
    updated_records = 0
    usage_episode_count = 0

    for log_path in log_paths:
        bucket = log_path.parent.name
        entry_name = log_path.stem
        detail_dir = result_root / "detail" / bucket / entry_name
        usage_summary = usage_index.get((bucket, entry_name), {})
        if _usage_count(usage_summary) <= 0:
            continue
        usage_episode_count += 1

        log_payload = _load_json(log_path)
        if isinstance(log_payload, dict):
            original_log_payload = dict(log_payload)
            _merge_usage(log_payload, usage_summary)
            compact_log_payload = SaveManager._build_log_result(log_payload)
            if json.dumps(compact_log_payload, sort_keys=True, ensure_ascii=False) != json.dumps(
                original_log_payload,
                sort_keys=True,
                ensure_ascii=False,
            ):
                updated_logs += 1
                if write:
                    _write_json(log_path, compact_log_payload)

        records_path = detail_dir / "records" / "result.json"
        record_payload = _load_json(records_path)
        if isinstance(record_payload, dict) and _merge_usage(record_payload, usage_summary):
            updated_records += 1
            if write:
                _write_json(records_path, record_payload)

    return {
        "result_root": str(result_root),
        "log_files": len(log_paths),
        "usage_episode_count": usage_episode_count,
        "updated_logs": updated_logs,
        "updated_records": updated_records,
    }


def _report_title_for_root(result_root: Path) -> str:
    parts = result_root.parts
    if "navgbench" in parts:
        idx = parts.index("navgbench")
        mode = parts[idx + 1] if len(parts) > idx + 1 else "unknown"
        model = parts[idx + 2] if len(parts) > idx + 2 else result_root.name
        return f"NavGBench report | mode: {mode} | model: {model} | range: all"
    if "r2rce" in parts:
        idx = parts.index("r2rce")
        model = result_root.name
        if len(parts) > idx + 1 and parts[idx + 1] == "ablation":
            ablation = parts[idx + 2] if len(parts) > idx + 2 else "unknown"
            return f"R2R-CE report | ablation: {ablation} | model: {model} | range: all"
        return f"R2R-CE report | model: {model} | range: all"
    if "ovon" in parts:
        return f"OVON report | model: {result_root.name} | range: all"
    return f"SpaceVLN report | model: {result_root.name} | range: all"


def _exp_config_for_root(result_root: Path) -> str:
    parts = set(result_root.parts)
    if "navgbench" in parts:
        return "navigation_system/config/experiments/vlnce/navgbench_eval.yaml"
    if "ovon" in parts:
        return "navigation_system/config/experiments/object_navigation/ovon_val_unseen_eval.yaml"
    return "navigation_system/config/experiments/vlnce/r2r_eval.yaml"


def regenerate_report(result_root: Path, *, workers: int) -> bool:
    try:
        generate_results_report(
            str(result_root),
            save=True,
            md_only=False,
            verbose=False,
            exp_config=_exp_config_for_root(result_root),
            load_workers=max(1, int(workers)),
            report_title=_report_title_for_root(result_root),
        )
        return True
    except Exception as exc:
        print(f"[WARN] report failed: {result_root} | {type(exc).__name__}: {exc}")
        return False


def _parse_roots(args: argparse.Namespace) -> List[Path]:
    if args.result_dir:
        return [Path(item).expanduser().resolve() for item in args.result_dir]
    results_root = Path(args.results_root).expanduser().resolve()
    return _iter_result_roots(results_root)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=str((PROJECT_ROOT.parent / "result").resolve()),
        help="Root containing SpaceVLN result directories.",
    )
    parser.add_argument(
        "--result-dir",
        action="append",
        default=[],
        help="Backfill one specific result directory. Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report what would change, without writing JSON files.",
    )
    parser.add_argument(
        "--regenerate-reports",
        action="store_true",
        help="Regenerate root episode_results.md/csv/metrics.json after backfill.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="JSON load workers used when regenerating reports.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    roots = _parse_roots(args)
    if not roots:
        print("No result directories found.")
        return 1

    total_logs = 0
    total_records = 0
    total_usage_episodes = 0
    report_count = 0
    for root in roots:
        if not (root / "log").is_dir():
            print(f"[SKIP] missing log: {root}")
            continue
        print(f"[Scan] {root}", flush=True)
        summary = backfill_result_root(root, write=not args.dry_run, workers=args.workers)
        total_logs += int(summary["updated_logs"])
        total_records += int(summary["updated_records"])
        total_usage_episodes += int(summary["usage_episode_count"])
        print(
            "[Backfill] "
            f"{root} | usage_episodes={summary['usage_episode_count']} "
            f"| logs={summary['updated_logs']} | records={summary['updated_records']}"
        , flush=True)
        if args.regenerate_reports and not args.dry_run:
            if regenerate_report(root, workers=args.workers):
                report_count += 1
                print(f"[Report] updated {root / 'episode_results.md'}", flush=True)

    mode = "dry-run" if args.dry_run else "write"
    print(
        f"[Done] mode={mode} | roots={len(roots)} | usage_episodes={total_usage_episodes} "
        f"| updated_logs={total_logs} | updated_records={total_records} | reports={report_count}"
    , flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
