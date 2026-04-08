#!/usr/bin/env python3
"""Audit and optionally backfill SpaceVLN result layout artifacts."""

import argparse
import importlib.util
import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_results_dir() -> str:
    return os.path.abspath(os.path.join(_repo_root(), "..", "data", "result", "spacevln"))


def _load_save_manager_module():
    module_path = os.path.join(
        _repo_root(),
        "vlnce_baselines",
        "vlm",
        "support",
        "save_manager.py",
    )
    spec = importlib.util.spec_from_file_location("spacevln_save_manager_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load save_manager.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _first_existing_path(paths: List[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _missing_fields(payload: Dict[str, Any], required_fields: Tuple[str, ...]) -> List[str]:
    if not isinstance(payload, dict):
        return list(required_fields)
    return [field for field in required_fields if field not in payload]


def _summarize_episode_for_log(payload: Dict[str, Any], source_relpath: str) -> str:
    episode_id = int(payload.get("episode_id", 0) or 0)
    sr = int(payload.get("sr", 0) or 0)
    osr = int(payload.get("osr", 0) or 0)
    steps = int(payload.get("total_steps", 0) or 0)
    subtask_count = int(payload.get("subtask_count", 0) or 0)
    ne = payload.get("ne", -1)
    spl = payload.get("spl", 0.0)
    ndtw = payload.get("ndtw", 0.0)
    instruction = str(payload.get("instruction", "") or "").strip()
    timestamp = str(payload.get("timestamp", "") or "").strip()

    return "\n".join(
        [
            f"Episode {episode_id} legacy summary log",
            "=" * 60,
            "This text log was backfilled from the saved JSON result.",
            "The original per-step stdout/stderr log was not preserved for this older run.",
            f"Source JSON: {source_relpath}",
            f"Backfilled at: {datetime.now().isoformat()}",
            f"Original result timestamp: {timestamp or 'Unknown'}",
            "",
            f"Instruction: {instruction}",
            "",
            "Final Metrics",
            f"- SR: {sr}",
            f"- OSR: {osr}",
            f"- Steps: {steps}",
            f"- Subtasks: {subtask_count}",
            f"- NE: {ne}",
            f"- SPL: {spl}",
            f"- nDTW: {ndtw}",
            "",
        ]
    )


def _write_json_if_missing(path: str, payload: Dict[str, Any]) -> bool:
    if os.path.exists(path):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return True


def _write_text_if_missing(path: str, text: str) -> bool:
    if os.path.exists(path):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and backfill SpaceVLN result layout artifacts.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=_default_results_dir(),
        help="SpaceVLN results root directory.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Backfill missing records files for complete episodes.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="",
        help="Optional path to save the audit report.",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    save_manager = _load_save_manager_module()

    log_paths = save_manager.iter_all_episode_log_paths(results_dir)
    log_paths = sorted(log_paths)

    counts = Counter()
    missing_field_counts = Counter()
    incomplete_rows: List[Dict[str, Any]] = []
    complete_missing_rows: List[Dict[str, Any]] = []

    backfilled_result_json = 0
    backfilled_result_latest = 0
    backfilled_text_logs = 0

    for log_path in log_paths:
        payload = _load_json(log_path)
        rel_log_path = os.path.relpath(log_path, results_dir)
        episode_id = int(payload.get("episode_id", -1) or -1)

        detail_candidates = save_manager.get_episode_detail_path_candidates(results_dir, episode_id)
        detail_dir = _first_existing_path(detail_candidates) or detail_candidates[0]
        records_dir = os.path.join(detail_dir, "records")
        text_log_path = os.path.join(records_dir, f"episode_{episode_id}.log")
        result_json_path = os.path.join(records_dir, "result.json")
        result_latest_path = os.path.join(records_dir, "result_latest.json")

        is_complete = save_manager.SaveManager.is_complete_saved_result(payload)
        is_sr1 = save_manager.SaveManager.result_has_sr1(payload)

        if is_complete:
            counts["complete"] += 1
            if not os.path.exists(text_log_path) or not os.path.exists(result_json_path) or not os.path.exists(result_latest_path):
                row = {
                    "episode_id": episode_id,
                    "log_path": rel_log_path,
                    "detail_dir": os.path.relpath(detail_dir, results_dir),
                    "sr1": is_sr1,
                    "missing_text_log": not os.path.exists(text_log_path),
                    "missing_result_json": not os.path.exists(result_json_path),
                    "missing_result_latest": not os.path.exists(result_latest_path),
                }
                complete_missing_rows.append(row)

                if args.write:
                    if row["missing_result_json"]:
                        if _write_json_if_missing(result_json_path, payload):
                            backfilled_result_json += 1
                    if row["missing_result_latest"]:
                        if _write_json_if_missing(result_latest_path, payload):
                            backfilled_result_latest += 1
                    if row["missing_text_log"]:
                        text = _summarize_episode_for_log(payload, rel_log_path)
                        if _write_text_if_missing(text_log_path, text):
                            backfilled_text_logs += 1
        else:
            counts["incomplete"] += 1
            common_missing = _missing_fields(payload, save_manager.SaveManager.COMMON_RESULT_FIELDS)
            log_missing = _missing_fields(payload, save_manager.SaveManager.LOG_RESULT_FIELDS)
            full_missing = _missing_fields(payload, save_manager.SaveManager.FULL_RESULT_FIELDS)
            for field in common_missing + log_missing:
                missing_field_counts[field] += 1
            incomplete_rows.append(
                {
                    "episode_id": episode_id,
                    "log_path": rel_log_path,
                    "sr1": is_sr1,
                    "missing_common_fields": common_missing,
                    "missing_log_fields": log_missing,
                    "missing_full_fields": full_missing,
                }
            )

    incomplete_rows.sort(key=lambda item: int(item["episode_id"]))
    complete_missing_rows.sort(key=lambda item: int(item["episode_id"]))

    lines: List[str] = []
    lines.append(f"Results Root: {results_dir}")
    lines.append(f"Scanned Logs: {len(log_paths)}")
    lines.append(f"Complete Logs: {counts['complete']}")
    lines.append(f"Incomplete Logs: {counts['incomplete']}")
    lines.append(f"Complete Episodes Missing New Records Files: {len(complete_missing_rows)}")
    lines.append("")

    if missing_field_counts:
        lines.append("Most Common Missing Fields In Incomplete Logs:")
        for field, count in missing_field_counts.most_common(20):
            lines.append(f"- {field}: {count}")
        lines.append("")

    lines.append(f"Incomplete SR=1 Episodes: {sum(1 for row in incomplete_rows if row['sr1'])}")
    lines.append(
        "Incomplete SR=1 Episode IDs: "
        + ", ".join(str(row["episode_id"]) for row in incomplete_rows if row["sr1"])
    )
    lines.append("")
    lines.append(f"Incomplete Non-SR1 Episodes: {sum(1 for row in incomplete_rows if not row['sr1'])}")
    lines.append(
        "Incomplete Non-SR1 Episode IDs: "
        + ", ".join(str(row["episode_id"]) for row in incomplete_rows if not row["sr1"])
    )
    lines.append("")

    if complete_missing_rows:
        lines.append("Complete Episodes Missing New Records Artifacts:")
        for row in complete_missing_rows[:200]:
            missing_bits = []
            if row["missing_text_log"]:
                missing_bits.append("episode.log")
            if row["missing_result_json"]:
                missing_bits.append("result.json")
            if row["missing_result_latest"]:
                missing_bits.append("result_latest.json")
            lines.append(
                f"- episode_{row['episode_id']}: missing {', '.join(missing_bits)} | "
                f"detail={row['detail_dir']} | log={row['log_path']}"
            )
        if len(complete_missing_rows) > 200:
            lines.append(f"- ... truncated {len(complete_missing_rows) - 200} more complete episodes")
        lines.append("")

    if args.write:
        lines.append("Backfill Summary:")
        lines.append(f"- result.json created: {backfilled_result_json}")
        lines.append(f"- result_latest.json created: {backfilled_result_latest}")
        lines.append(f"- episode_xxx.log created: {backfilled_text_logs}")
        lines.append("")

    report_text = "\n".join(lines).rstrip() + "\n"
    print(report_text, end="")

    report_path = str(args.report_path or "").strip()
    if not report_path:
        report_path = os.path.join(results_dir, "results_layout_audit.txt")
    report_path = os.path.abspath(report_path)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Audit report saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
