"""Generate OVON range reports from existing sample logs only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from navigation_system.runtime.object_navigation.runner import (
    _build_aggregate,
    _format_ovon_metric,
    _resolve_success_distance_from_ovon_config,
    _prepare_ovon_config,
)


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


def _load_sample_rows(
    results_dir: Path,
    *,
    start_index: Optional[int],
    end_index: Optional[int],
) -> List[Dict]:
    rows: List[Dict] = []
    for log_path in _iter_sample_log_paths(results_dir):
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        sample_index = _safe_int(
            payload.get("sample_index"),
            default=_safe_int(log_path.stem.split("_", 1)[1], 0),
        )
        if sample_index <= 0:
            continue
        if start_index is not None and sample_index < start_index:
            continue
        if end_index is not None and sample_index > end_index:
            continue

        rows.append(
            {
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
        )

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

    return {
        "summary": str(summary_txt_path),
        "metrics_json": str(metrics_json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }


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
) -> Dict[str, str]:
    base_results_dir = Path(results_dir).resolve()
    rows = _load_sample_rows(
        base_results_dir,
        start_index=start_index,
        end_index=end_index,
    )
    if not rows:
        range_label = _report_subdir_name(start_index, end_index)
        raise RuntimeError(f"No OVON sample logs found for range '{range_label}' in {base_results_dir}")

    ovon_config = _prepare_ovon_config(
        exp_config=exp_config,
        split=split,
        data_path=data_path,
        gpu_id=gpu_id,
        max_steps=max_steps,
    )
    aggregate = _build_aggregate(rows)
    success_distance_m = _resolve_success_distance_from_ovon_config(ovon_config)

    report_dir = base_results_dir / "reports" / _report_subdir_name(start_index, end_index)

    summary_meta = {
        "selection_mode": "sample_index_range_from_existing_logs",
        "start_index": start_index,
        "end_index": end_index,
        "results_dir": str(base_results_dir),
        "split": str(split),
        "data_path": str(data_path),
    }
    return _write_ovon_range_reports(
        output_dir=report_dir,
        rows=rows,
        aggregate=aggregate,
        success_distance_m=success_distance_m,
        summary_meta=summary_meta,
    )


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
    )
    print("OVON range report generated:")
    for key, value in report_paths.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
