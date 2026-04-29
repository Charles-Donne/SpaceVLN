"""Generate OVON range reports from existing sample logs only."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from navigation_system.runtime.object_navigation.ovon.thresholds import OVON_SUCCESS_DISTANCE_M


def _format_ovon_metric(value: float, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.0"


def _build_aggregate(all_results: Sequence[dict]) -> dict:
    rows = list(all_results or [])
    count = len(rows)
    if count <= 0:
        return {
            "episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_distance_to_goal": 0.0,
            "avg_spl": 0.0,
            "avg_soft_spl": 0.0,
        }

    return {
        "episodes": count,
        "successes": sum(1 for item in rows if bool(item.get("success", False))),
        "success_rate": sum(1 for item in rows if bool(item.get("success", False))) / count,
        "avg_steps": sum(float(item.get("steps", 0) or 0) for item in rows) / count,
        "avg_distance_to_goal": (
            sum(float(item.get("distance_to_goal", -1.0) or -1.0) for item in rows) / count
        ),
        "avg_spl": sum(float(item.get("spl", 0.0) or 0.0) for item in rows) / count,
        "avg_soft_spl": sum(float(item.get("soft_spl", 0.0) or 0.0) for item in rows) / count,
    }


def _iter_sample_log_paths(results_dir: Path) -> Iterable[Path]:
    log_root = results_dir / "log"
    if not log_root.exists():
        return []
    paths: List[Path] = []
    for bucket_dir in sorted(log_root.iterdir()):
        if not bucket_dir.is_dir():
            continue
        for file_path in sorted(bucket_dir.glob("sample_*.json")):
            if file_path.is_file():
                paths.append(file_path)
    return paths


def _sample_index_from_path(log_path: Path) -> int:
    stem = str(log_path.stem or "")
    if not stem.startswith("sample_"):
        return 0
    return _safe_int(stem.split("_", 1)[1], 0)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _default_load_workers() -> int:
    raw_value = str(os.environ.get("SPACEVLN_REPORT_WORKERS", "") or "").strip()
    if raw_value:
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 1

    cpu_count = int(os.cpu_count() or 4)
    recommended = max(8, cpu_count * 4)
    return min(64, recommended)


def _bounded_load_workers(load_workers: int, item_count: int) -> int:
    try:
        parsed_workers = int(load_workers)
    except (TypeError, ValueError):
        parsed_workers = 1
    return max(1, min(parsed_workers, max(1, int(item_count))))


def _read_sample_row(log_path: Path) -> Optional[Dict]:
    sample_index_hint = _sample_index_from_path(log_path)
    try:
        with log_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    sample_index = _safe_int(payload.get("sample_index"), default=sample_index_hint)
    if sample_index <= 0:
        return None

    return {
        "sample_index": sample_index,
        "episode_id": _safe_int(payload.get("episode_id"), -1),
        "success": bool(_safe_int(payload.get("sr", payload.get("success", 0)), 0)),
        "steps": _safe_int(payload.get("total_steps", payload.get("steps", 0)), 0),
        "distance_to_goal": _safe_float(
            payload.get("ne", payload.get("distance_to_goal", -1.0)),
            -1.0,
        ),
        "spl": _safe_float(payload.get("spl", 0.0), 0.0),
        "soft_spl": _safe_float(
            payload.get("soft_spl", payload.get("oracle_spl", 0.0)),
            0.0,
        ),
        "reason": str(payload.get("reason", "") or ""),
        "error": str(payload.get("error", "") or ""),
    }


def _load_sample_rows(
    results_dir: Path,
    *,
    start_index: Optional[int],
    end_index: Optional[int],
    load_workers: int,
    verbose: bool,
) -> List[Dict]:
    candidate_paths = []
    for log_path in _iter_sample_log_paths(results_dir):
        sample_index = _sample_index_from_path(log_path)
        if sample_index > 0:
            if start_index is not None and sample_index < start_index:
                continue
            if end_index is not None and sample_index > end_index:
                continue
        candidate_paths.append(log_path)

    worker_count = _bounded_load_workers(load_workers, len(candidate_paths))
    if verbose:
        log_root = results_dir / "log"
        print(f"📂 Loading {len(candidate_paths)} OVON sample logs from {log_root}")
        if worker_count > 1:
            print(f"⚙️  Parallel JSON workers: {worker_count}")

    if worker_count <= 1:
        loaded_rows = [_read_sample_row(log_path) for log_path in candidate_paths]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            loaded_rows = list(executor.map(_read_sample_row, candidate_paths))

    rows: List[Dict] = []
    for row in loaded_rows:
        if row is None:
            continue
        sample_index = _safe_int(row.get("sample_index"), 0)
        if start_index is not None and sample_index < start_index:
            continue
        if end_index is not None and sample_index > end_index:
            continue
        rows.append(row)

    rows.sort(key=lambda item: int(item.get("sample_index", 0)))
    return rows


def _report_subdir_name(start_index: Optional[int], end_index: Optional[int]) -> str:
    if start_index is None and end_index is None:
        return "all"
    start_label = str(start_index) if start_index is not None else "start"
    end_label = str(end_index) if end_index is not None else "end"
    return f"{start_label}-{end_label}"


def _write_ovon_range_reports(
    *,
    output_dir: Path,
    rows: List[Dict],
    aggregate: Dict,
    success_distance_m: float,
    summary_meta: Dict,
    summary_only: bool,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_txt_path = output_dir / "summary.txt"
    metrics_json_path = output_dir / "metrics.json"
    csv_path = output_dir / "episode_results.csv"
    md_path = output_dir / "episode_results.md"

    summary_text = (
        "========================================\n"
        "OVON evaluation summary\n"
        "========================================\n"
        f"Episodes:      {int(aggregate.get('episodes', 0) or 0)}\n"
        f"Successes:     {int(aggregate.get('successes', 0) or 0)}\n"
        f"SR:            {_format_ovon_metric(float(aggregate.get('success_rate', 0.0) or 0.0), 3)}\n"
        f"SPL:           {_format_ovon_metric(float(aggregate.get('avg_spl', 0.0) or 0.0), 3)}\n"
        f"SoftSPL:       {_format_ovon_metric(float(aggregate.get('avg_soft_spl', 0.0) or 0.0), 3)}\n"
        f"Avg DTG:       {_format_ovon_metric(float(aggregate.get('avg_distance_to_goal', 0.0) or 0.0), 3)}m\n"
        f"Avg Steps:     {_format_ovon_metric(float(aggregate.get('avg_steps', 0.0) or 0.0), 2)}\n"
        f"Success dist:  {float(success_distance_m):.2f}m\n"
        f"Selection:     {summary_meta.get('selection_mode', 'sample_index_range_from_existing_logs')}\n"
        "========================================\n"
    )
    summary_txt_path.write_text(summary_text, encoding="utf-8")

    metrics_payload = {"meta": dict(summary_meta), "aggregate": dict(aggregate)}
    metrics_json_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    saved_paths = {
        "summary": str(summary_txt_path),
        "metrics_json": str(metrics_json_path),
    }
    if summary_only:
        return saved_paths

    headers = [
        "episode_id",
        "sample_index",
        "sr",
        "distance_to_goal",
        "spl",
        "soft_spl",
        "steps",
        "reason",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "episode_id": int(row.get("episode_id", -1) or -1),
                    "sample_index": int(row.get("sample_index", -1) or -1),
                    "sr": int(bool(row.get("success", False))),
                    "distance_to_goal": _format_ovon_metric(float(row.get("distance_to_goal", -1.0) or -1.0), 4),
                    "spl": _format_ovon_metric(float(row.get("spl", 0.0) or 0.0), 4),
                    "soft_spl": _format_ovon_metric(float(row.get("soft_spl", 0.0) or 0.0), 4),
                    "steps": int(row.get("steps", 0) or 0),
                    "reason": str(row.get("reason", "") or ""),
                    "error": str(row.get("error", "") or ""),
                }
            )

    md_lines = [
        "# OVON Episode Results",
        "",
        "| Episode | Sample | SR | DTG(m) | SPL | SoftSPL | Steps |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| {episode} | {sample} | {sr} | {dtg} | {spl} | {soft_spl} | {steps} |".format(
                episode=int(row.get("episode_id", -1) or -1),
                sample=int(row.get("sample_index", -1) or -1),
                sr=int(bool(row.get("success", False))),
                dtg=_format_ovon_metric(float(row.get("distance_to_goal", -1.0) or -1.0), 4),
                spl=_format_ovon_metric(float(row.get("spl", 0.0) or 0.0), 4),
                soft_spl=_format_ovon_metric(float(row.get("soft_spl", 0.0) or 0.0), 4),
                steps=int(row.get("steps", 0) or 0),
            )
        )
    md_lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Episodes | SR | SPL | SoftSPL | Avg DTG(m) | Avg Steps |",
            "| --- | --- | --- | --- | --- | --- |",
            "| {episodes} | {sr} | {spl} | {soft_spl} | {dtg} | {steps} |".format(
                episodes=int(aggregate.get("episodes", 0) or 0),
                sr=_format_ovon_metric(float(aggregate.get("success_rate", 0.0) or 0.0), 4),
                spl=_format_ovon_metric(float(aggregate.get("avg_spl", 0.0) or 0.0), 4),
                soft_spl=_format_ovon_metric(float(aggregate.get("avg_soft_spl", 0.0) or 0.0), 4),
                dtg=_format_ovon_metric(float(aggregate.get("avg_distance_to_goal", 0.0) or 0.0), 4),
                steps=_format_ovon_metric(float(aggregate.get("avg_steps", 0.0) or 0.0), 2),
            ),
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    saved_paths["csv"] = str(csv_path)
    saved_paths["markdown"] = str(md_path)
    return saved_paths


def generate_ovon_range_report(
    *,
    results_dir: str,
    exp_config: str,
    split: str,
    data_path: str,
    gpu_id: int,
    max_steps: int,
    start_index: Optional[int],
    end_index: Optional[int],
    summary_only: bool = False,
    load_workers: int = 1,
    verbose: bool = True,
) -> Dict[str, str]:
    base_results_dir = Path(results_dir).resolve()
    rows = _load_sample_rows(
        base_results_dir,
        start_index=start_index,
        end_index=end_index,
        load_workers=load_workers,
        verbose=verbose,
    )
    if not rows:
        range_label = _report_subdir_name(start_index, end_index)
        raise RuntimeError(f"No OVON sample logs found for range '{range_label}' in {base_results_dir}")

    aggregate = _build_aggregate(rows)
    success_distance_m = float(OVON_SUCCESS_DISTANCE_M)

    report_dir = base_results_dir / "reports" / _report_subdir_name(start_index, end_index)
    if verbose:
        print(f"✅ Loaded {len(rows)} OVON sample logs")
        print(f"📁 Report directory: {report_dir}")

    summary_meta = {
        "selection_mode": "sample_index_range_from_existing_logs",
        "start_index": start_index,
        "end_index": end_index,
        "results_dir": str(base_results_dir),
        "split": str(split),
        "data_path": str(data_path),
        "load_workers": int(_bounded_load_workers(load_workers, len(rows))),
        "summary_only": bool(summary_only),
        "success_distance_m": success_distance_m,
    }
    saved_paths = _write_ovon_range_reports(
        output_dir=report_dir,
        rows=rows,
        aggregate=aggregate,
        success_distance_m=success_distance_m,
        summary_meta=summary_meta,
        summary_only=summary_only,
    )
    if verbose:
        print(f"📋 Saved summary report: {saved_paths['summary']}")
        print(f"📋 Saved metrics JSON: {saved_paths['metrics_json']}")
        if not summary_only:
            print(f"📋 Saved episode CSV: {saved_paths['csv']}")
            print(f"📋 Saved episode Markdown: {saved_paths['markdown']}")
    return saved_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an OVON sample-index range report from existing logs only.",
    )
    parser.add_argument("--path", required=True, help="OVON result directory")
    parser.add_argument(
        "--exp-config",
        default="ovon/config/experiments/transformer_dagger.yaml",
        help="official OVON experiment config",
    )
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument(
        "--data-path",
        default="data/datasets/ovon/hm3d/v1/val_unseen/val_unseen_hard.json.gz",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Fast mode: save only summary + metrics.json, skip episode CSV/Markdown",
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=_default_load_workers(),
        help="Number of workers for loading sample JSON files in parallel",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    report_paths = generate_ovon_range_report(
        results_dir=str(args.path),
        exp_config=str(args.exp_config),
        split=str(args.split),
        data_path=str(args.data_path),
        gpu_id=int(args.gpu_id),
        max_steps=int(args.max_steps),
        start_index=args.start_index,
        end_index=args.end_index,
        summary_only=bool(args.summary_only),
        load_workers=int(args.load_workers),
        verbose=True,
    )
    print("OVON range report generated:")
    for key, value in report_paths.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
