"""Summarize per-request VLM info, with optional cache metrics, from a results directory."""

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


VLM_INFO_FILENAME = "vlm_info.json"
REQUEST_KIND_ORDER = ("thinking", "action", "other")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _median(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    return float(statistics.median(numbers)) if numbers else 0.0


def _mean(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    return float(sum(numbers) / len(numbers)) if numbers else 0.0


def _classify_request_kind(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "thinking" in parts:
        return "thinking"
    if "action" in parts:
        return "action"
    return "other"


@dataclass(frozen=True)
class CacheRequestRecord:
    kind: str
    artifact_dir: str
    source_file: str
    model: str
    provider: str
    success: bool
    duration_s: float
    failed_attempts: int
    failed_wasted_time_s: float
    input_tokens: int
    input_text_tokens: int
    input_image_tokens: int
    output_tokens: int
    total_tokens: int
    cache_enabled: bool
    provider_reported_cache_counters: bool
    cached_tokens: int
    cache_write_tokens: int
    uncached_tokens: int
    cache_hit_ratio: float
    cost_ratio: float
    savings_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "duration_s",
            "failed_wasted_time_s",
            "cache_hit_ratio",
            "cost_ratio",
            "savings_ratio",
        ):
            payload[key] = round(float(payload[key]), 6)
        return payload


@dataclass(frozen=True)
class CacheAggregateSummary:
    request_count: int
    success_count: int
    failed_count: int
    failed_attempts: int
    failed_wasted_time_s: float
    total_duration_s: float
    average_duration_s: float
    median_duration_s: float
    input_tokens: int
    input_text_tokens: int
    input_image_tokens: int
    output_tokens: int
    total_tokens: int
    average_output_tokens_per_second: float
    median_output_tokens_per_second: float
    cache_enabled_requests: int
    cache_reported_requests: int
    cached_tokens: int
    cache_write_tokens: int
    uncached_tokens: int
    weighted_cache_hit_ratio: float
    cost_ratio: float
    savings_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, float):
                payload[key] = round(value, 6)
        return payload


@dataclass(frozen=True)
class CacheRunReport:
    results_dir: str
    request_files_scanned: int
    overall: CacheAggregateSummary
    thinking: CacheAggregateSummary
    action: CacheAggregateSummary
    other: CacheAggregateSummary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results_dir": self.results_dir,
            "request_files_scanned": self.request_files_scanned,
            "overall": self.overall.to_dict(),
            "thinking": self.thinking.to_dict(),
            "action": self.action.to_dict(),
            "other": self.other.to_dict(),
        }


def _load_request_record(path: Path) -> CacheRequestRecord:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f) or {}

    cache = dict(payload.get("cache") or {})
    duration_s = _safe_float(payload.get("duration_s"), 0.0)
    output_tokens = _safe_int(payload.get("output_tokens"), 0)
    return CacheRequestRecord(
        kind=_classify_request_kind(path),
        artifact_dir=str(path.parent),
        source_file=path.name,
        model=str(payload.get("model") or ""),
        provider=str(payload.get("provider") or ""),
        success=bool(payload.get("success", False)),
        duration_s=duration_s,
        failed_attempts=_safe_int(payload.get("failed_attempts"), 0),
        failed_wasted_time_s=_safe_float(payload.get("failed_wasted_time_s"), 0.0),
        input_tokens=_safe_int(payload.get("input_tokens"), 0),
        input_text_tokens=_safe_int(payload.get("input_text_tokens"), 0),
        input_image_tokens=_safe_int(payload.get("input_image_tokens"), 0),
        output_tokens=output_tokens,
        total_tokens=_safe_int(payload.get("total_tokens"), 0),
        cache_enabled=bool(cache.get("enabled", False)),
        provider_reported_cache_counters=bool(cache.get("reported", False)),
        cached_tokens=_safe_int(cache.get("cached_tokens"), 0),
        cache_write_tokens=_safe_int(cache.get("write_tokens"), 0),
        uncached_tokens=_safe_int(cache.get("uncached_tokens"), 0),
        cache_hit_ratio=_safe_float(cache.get("hit_ratio"), 0.0),
        cost_ratio=_safe_float(cache.get("cost_ratio"), 0.0),
        savings_ratio=_safe_float(cache.get("savings_ratio"), 0.0),
    )


def _iter_request_files(results_dir: str) -> List[Path]:
    root = Path(results_dir).expanduser().resolve()
    return sorted(root.rglob(VLM_INFO_FILENAME))


def _build_aggregate(records: List[CacheRequestRecord]) -> CacheAggregateSummary:
    durations = [record.duration_s for record in records if record.duration_s > 0]
    speed_values = [
        float(record.output_tokens) / float(record.duration_s)
        for record in records
        if record.duration_s > 0 and record.output_tokens > 0
    ]
    cache_records = [record for record in records if record.provider_reported_cache_counters]
    covered_input_tokens = sum(record.input_tokens for record in cache_records)
    cached_tokens = sum(record.cached_tokens for record in cache_records)
    cache_write_tokens = sum(record.cache_write_tokens for record in cache_records)
    uncached_tokens = sum(record.uncached_tokens for record in cache_records)
    effective_input_units = (
        float(uncached_tokens)
        + float(cached_tokens) * 0.1
        + float(cache_write_tokens) * 1.25
    )
    baseline_units = max(1.0, float(covered_input_tokens))
    cost_ratio = effective_input_units / baseline_units if cache_records else 0.0
    savings_ratio = 1.0 - cost_ratio if cache_records else 0.0
    weighted_cache_hit_ratio = float(cached_tokens) / baseline_units if cache_records else 0.0

    return CacheAggregateSummary(
        request_count=len(records),
        success_count=sum(1 for record in records if record.success),
        failed_count=sum(1 for record in records if not record.success),
        failed_attempts=sum(record.failed_attempts for record in records),
        failed_wasted_time_s=sum(record.failed_wasted_time_s for record in records),
        total_duration_s=sum(durations),
        average_duration_s=_mean(durations),
        median_duration_s=_median(durations),
        input_tokens=sum(record.input_tokens for record in records),
        input_text_tokens=sum(record.input_text_tokens for record in records),
        input_image_tokens=sum(record.input_image_tokens for record in records),
        output_tokens=sum(record.output_tokens for record in records),
        total_tokens=sum(record.total_tokens for record in records),
        average_output_tokens_per_second=_mean(speed_values),
        median_output_tokens_per_second=_median(speed_values),
        cache_enabled_requests=sum(1 for record in records if record.cache_enabled),
        cache_reported_requests=len(cache_records),
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        uncached_tokens=uncached_tokens,
        weighted_cache_hit_ratio=weighted_cache_hit_ratio,
        cost_ratio=cost_ratio,
        savings_ratio=savings_ratio,
    )


def build_cache_report(results_dir: str) -> CacheRunReport:
    request_files = _iter_request_files(results_dir)
    records = [_load_request_record(path) for path in request_files]
    grouped: Dict[str, List[CacheRequestRecord]] = {kind: [] for kind in REQUEST_KIND_ORDER}
    for record in records:
        grouped.setdefault(record.kind, []).append(record)

    return CacheRunReport(
        results_dir=str(Path(results_dir).expanduser().resolve()),
        request_files_scanned=len(request_files),
        overall=_build_aggregate(records),
        thinking=_build_aggregate(grouped.get("thinking", [])),
        action=_build_aggregate(grouped.get("action", [])),
        other=_build_aggregate(grouped.get("other", [])),
    )


def render_cache_report(report: CacheRunReport) -> str:
    lines = [
        "VLM Cache Report",
        f"results_dir: {report.results_dir}",
        f"request_files: {report.request_files_scanned}",
    ]

    def add_section(name: str, summary: CacheAggregateSummary) -> None:
        lines.append("")
        lines.append(f"[{name}]")
        lines.append(
            f"requests={summary.request_count} | success={summary.success_count} | fail={summary.failed_count}"
        )
        lines.append(
            f"input={summary.input_tokens} | text={summary.input_text_tokens} | "
            f"image={summary.input_image_tokens} | output={summary.output_tokens}"
        )
        lines.append(
            f"time(total/avg/med)={summary.total_duration_s:.2f}s/{summary.average_duration_s:.2f}s/"
            f"{summary.median_duration_s:.2f}s | speed(avg/med)="
            f"{summary.average_output_tokens_per_second:.1f}/{summary.median_output_tokens_per_second:.1f} tok/s"
        )
        lines.append(
            f"fail_attempts={summary.failed_attempts} | failed_time={summary.failed_wasted_time_s:.2f}s"
        )
        if summary.cache_enabled_requests > 0:
            if summary.cache_reported_requests > 0:
                lines.append(
                    f"cache=req_enabled {summary.cache_enabled_requests} | reported {summary.cache_reported_requests} | "
                    f"cached={summary.cached_tokens} | write={summary.cache_write_tokens} | "
                    f"uncached={summary.uncached_tokens}"
                )
                lines.append(
                    f"cache_hit={summary.weighted_cache_hit_ratio * 100:.1f}% | "
                    f"cost={summary.cost_ratio:.3f}x | "
                    f"savings={summary.savings_ratio * 100:+.1f}%"
                )
            else:
                lines.append(
                    f"cache=req_enabled {summary.cache_enabled_requests} | reported 0 | counters unavailable"
                )

    add_section("overall", report.overall)
    add_section("thinking", report.thinking)
    add_section("action", report.action)
    if report.other.request_count > 0:
        add_section("other", report.other)
    return "\n".join(lines)


def write_cache_report_json(report: CacheRunReport, output_path: str) -> str:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    return str(output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize VLM request info and cache metrics")
    parser.add_argument("--results-dir", required=True, help="Run results directory to scan")
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to save the aggregated report as JSON",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_cache_report(args.results_dir)
    print(render_cache_report(report))
    if args.json_output:
        output_path = write_cache_report_json(report, args.json_output)
        print(f"\njson_report: {output_path}")
    return 0


__all__ = [
    "CacheAggregateSummary",
    "CacheRequestRecord",
    "CacheRunReport",
    "build_cache_report",
    "render_cache_report",
    "write_cache_report_json",
]


if __name__ == "__main__":
    raise SystemExit(main())
