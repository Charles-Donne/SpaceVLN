"""Summarize explicit-cache usage and request speed from a results directory."""

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CACHE_USAGE_FILENAME = "cache_usage.json"
PROVIDER_RESPONSE_FILENAME = "provider_response.json"
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


def _extract_prompt_token_details(usage: Dict[str, Any]) -> Dict[str, Any]:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        return details
    details = usage.get("input_tokens_details")
    if isinstance(details, dict):
        return details
    return {}


def _has_explicit_cache_counters(usage: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> bool:
    if isinstance(payload, dict) and "provider_reported_explicit_cache_counters" in payload:
        return bool(payload.get("provider_reported_explicit_cache_counters"))
    details = _extract_prompt_token_details(usage)
    cache_creation = details.get("cache_creation")
    return any(
        (
            "cached_tokens" in usage,
            "cached_tokens" in details,
            "cache_creation_input_tokens" in usage,
            "cache_creation_input_tokens" in details,
            isinstance(cache_creation, dict) and bool(cache_creation),
        )
    )


def _classify_request_kind(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "thinking" in parts:
        return "thinking"
    if "action" in parts:
        return "action"
    return "other"


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CacheRequestRecord:
    kind: str
    artifact_dir: str
    source_file: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_text_tokens: int
    prompt_image_tokens: int
    cached_tokens: int
    cache_creation_input_tokens: int
    uncached_prompt_tokens: int
    cache_hit_ratio: float
    effective_input_cost_multiplier: float
    input_cost_savings_ratio: float
    latency_s: Optional[float]
    tokens_per_second: Optional[float]
    provider_reported_explicit_cache_counters: bool

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "cache_hit_ratio",
            "effective_input_cost_multiplier",
            "input_cost_savings_ratio",
            "latency_s",
            "tokens_per_second",
        ):
            if payload[key] is not None:
                payload[key] = round(float(payload[key]), 6)
        return payload


@dataclass(frozen=True)
class CacheAggregateSummary:
    request_count: int
    requests_with_latency: int
    requests_with_provider_counters: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_text_tokens: int
    prompt_image_tokens: int
    prompt_tokens_with_provider_counters: int
    cached_tokens: int
    cache_creation_input_tokens: int
    uncached_prompt_tokens: int
    cache_metric_request_coverage_ratio: float
    cache_metric_prompt_coverage_ratio: float
    weighted_cache_hit_ratio: float
    average_cache_hit_ratio: float
    median_cache_hit_ratio: float
    effective_input_cost_multiplier: float
    input_cost_savings_ratio: float
    average_latency_s: float
    median_latency_s: float
    average_tokens_per_second: float
    median_tokens_per_second: float
    end_to_end_tokens_per_second: float

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


def _derive_summary_fields(
    usage: Dict[str, Any],
    context_cache_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    details = _extract_prompt_token_details(usage)
    cache_creation = dict(details.get("cache_creation") or {})

    prompt_tokens = _safe_int(
        (context_cache_payload or {}).get("prompt_tokens"),
        _safe_int(usage.get("prompt_tokens", usage.get("input_tokens", 0))),
    )
    completion_tokens = _safe_int(
        (context_cache_payload or {}).get("completion_tokens"),
        _safe_int(usage.get("completion_tokens", usage.get("output_tokens", 0))),
    )
    total_tokens = _safe_int(
        (context_cache_payload or {}).get("total_tokens"),
        _safe_int(usage.get("total_tokens", prompt_tokens + completion_tokens)),
    )
    cached_tokens = _safe_int(
        (context_cache_payload or {}).get("cached_tokens"),
        _safe_int(details.get("cached_tokens", usage.get("cached_tokens", 0))),
    )
    cache_creation_input_tokens = _safe_int(
        (context_cache_payload or {}).get("cache_creation_input_tokens"),
        _safe_int(
            details.get("cache_creation_input_tokens")
            or cache_creation.get("cache_creation_input_tokens")
            or cache_creation.get("ephemeral_5m_input_tokens")
            or 0
        ),
    )
    uncached_prompt_tokens = _safe_int(
        (context_cache_payload or {}).get("uncached_prompt_tokens"),
        max(0, prompt_tokens - cached_tokens - cache_creation_input_tokens),
    )

    baseline_units = max(1.0, float(prompt_tokens))
    effective_units = (
        float(uncached_prompt_tokens)
        + float(cached_tokens) * 0.1
        + float(cache_creation_input_tokens) * 1.25
    )
    effective_input_cost_multiplier = (
        _coerce_optional_float((context_cache_payload or {}).get("effective_input_cost_multiplier"))
        if context_cache_payload is not None
        else None
    )
    if effective_input_cost_multiplier is None:
        effective_input_cost_multiplier = effective_units / baseline_units

    input_cost_savings_ratio = (
        _coerce_optional_float((context_cache_payload or {}).get("input_cost_savings_ratio"))
        if context_cache_payload is not None
        else None
    )
    if input_cost_savings_ratio is None:
        input_cost_savings_ratio = 1.0 - float(effective_input_cost_multiplier)

    cache_hit_ratio = (
        _coerce_optional_float((context_cache_payload or {}).get("cache_hit_ratio"))
        if context_cache_payload is not None
        else None
    )
    if cache_hit_ratio is None:
        cache_hit_ratio = float(cached_tokens) / baseline_units

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_text_tokens": _safe_int(details.get("text_tokens", 0)),
        "prompt_image_tokens": _safe_int(details.get("image_tokens", 0)),
        "cached_tokens": cached_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "uncached_prompt_tokens": uncached_prompt_tokens,
        "cache_hit_ratio": float(cache_hit_ratio),
        "effective_input_cost_multiplier": float(effective_input_cost_multiplier),
        "input_cost_savings_ratio": float(input_cost_savings_ratio),
    }


def _load_request_record(path: Path) -> CacheRequestRecord:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f) or {}

    usage = dict(payload.get("usage") or {})
    context_cache_payload = dict(payload.get("context_cache") or {}) if path.name == CACHE_USAGE_FILENAME else None
    summary_fields = _derive_summary_fields(usage, context_cache_payload=context_cache_payload)
    latency_s = None
    if context_cache_payload is not None:
        latency_s = _coerce_optional_float(context_cache_payload.get("latency_s"))

    tokens_per_second = None
    if latency_s is not None and latency_s > 0 and summary_fields["completion_tokens"] > 0:
        tokens_per_second = float(summary_fields["completion_tokens"]) / float(latency_s)

    return CacheRequestRecord(
        kind=_classify_request_kind(path),
        artifact_dir=str(path.parent),
        source_file=path.name,
        model=str(payload.get("model") or ""),
        provider=str(payload.get("provider") or ""),
        prompt_tokens=summary_fields["prompt_tokens"],
        completion_tokens=summary_fields["completion_tokens"],
        total_tokens=summary_fields["total_tokens"],
        prompt_text_tokens=summary_fields["prompt_text_tokens"],
        prompt_image_tokens=summary_fields["prompt_image_tokens"],
        cached_tokens=summary_fields["cached_tokens"],
        cache_creation_input_tokens=summary_fields["cache_creation_input_tokens"],
        uncached_prompt_tokens=summary_fields["uncached_prompt_tokens"],
        cache_hit_ratio=summary_fields["cache_hit_ratio"],
        effective_input_cost_multiplier=summary_fields["effective_input_cost_multiplier"],
        input_cost_savings_ratio=summary_fields["input_cost_savings_ratio"],
        latency_s=latency_s,
        tokens_per_second=tokens_per_second,
        provider_reported_explicit_cache_counters=_has_explicit_cache_counters(usage, payload),
    )


def _iter_request_files(results_dir: str) -> List[Path]:
    root = Path(results_dir).expanduser().resolve()
    usage_paths = sorted(root.rglob(CACHE_USAGE_FILENAME))
    usage_parent_dirs = {path.parent for path in usage_paths}
    provider_paths = [
        path
        for path in sorted(root.rglob(PROVIDER_RESPONSE_FILENAME))
        if path.parent not in usage_parent_dirs
    ]
    return usage_paths + provider_paths


def _build_aggregate(records: List[CacheRequestRecord]) -> CacheAggregateSummary:
    request_count = len(records)
    latency_values = [record.latency_s for record in records if record.latency_s and record.latency_s > 0]
    speed_values = [
        record.tokens_per_second
        for record in records
        if record.tokens_per_second and record.tokens_per_second > 0
    ]
    counter_records = [
        record for record in records if record.provider_reported_explicit_cache_counters
    ]

    prompt_tokens = sum(record.prompt_tokens for record in records)
    completion_tokens = sum(record.completion_tokens for record in records)
    total_tokens = sum(record.total_tokens for record in records)
    prompt_text_tokens = sum(record.prompt_text_tokens for record in records)
    prompt_image_tokens = sum(record.prompt_image_tokens for record in records)

    covered_prompt_tokens = sum(record.prompt_tokens for record in counter_records)
    cached_tokens = sum(record.cached_tokens for record in counter_records)
    cache_creation_input_tokens = sum(
        record.cache_creation_input_tokens for record in counter_records
    )
    uncached_prompt_tokens = sum(record.uncached_prompt_tokens for record in counter_records)

    covered_prompt_baseline = max(1.0, float(covered_prompt_tokens))
    effective_input_units = (
        float(uncached_prompt_tokens)
        + float(cached_tokens) * 0.1
        + float(cache_creation_input_tokens) * 1.25
    )
    weighted_cache_hit_ratio = (
        float(cached_tokens) / covered_prompt_baseline if counter_records else 0.0
    )
    effective_input_cost_multiplier = (
        effective_input_units / covered_prompt_baseline if counter_records else 0.0
    )
    input_cost_savings_ratio = (
        1.0 - effective_input_cost_multiplier if counter_records else 0.0
    )

    return CacheAggregateSummary(
        request_count=request_count,
        requests_with_latency=len(latency_values),
        requests_with_provider_counters=len(counter_records),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_text_tokens=prompt_text_tokens,
        prompt_image_tokens=prompt_image_tokens,
        prompt_tokens_with_provider_counters=covered_prompt_tokens,
        cached_tokens=cached_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        uncached_prompt_tokens=uncached_prompt_tokens,
        cache_metric_request_coverage_ratio=(
            float(len(counter_records)) / float(request_count) if request_count else 0.0
        ),
        cache_metric_prompt_coverage_ratio=(
            float(covered_prompt_tokens) / float(prompt_tokens) if prompt_tokens else 0.0
        ),
        weighted_cache_hit_ratio=weighted_cache_hit_ratio,
        average_cache_hit_ratio=_mean(record.cache_hit_ratio for record in counter_records),
        median_cache_hit_ratio=_median(record.cache_hit_ratio for record in counter_records),
        effective_input_cost_multiplier=effective_input_cost_multiplier,
        input_cost_savings_ratio=input_cost_savings_ratio,
        average_latency_s=_mean(latency_values),
        median_latency_s=_median(latency_values),
        average_tokens_per_second=_mean(speed_values),
        median_tokens_per_second=_median(speed_values),
        end_to_end_tokens_per_second=(
            float(sum(record.completion_tokens for record in records if record.latency_s and record.latency_s > 0))
            / float(sum(latency_values))
            if latency_values
            else 0.0
        ),
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
        "Cache Report",
        f"results_dir: {report.results_dir}",
        f"request_files: {report.request_files_scanned}",
    ]

    def add_section(name: str, summary: CacheAggregateSummary) -> None:
        lines.append("")
        lines.append(f"[{name}]")
        lines.append(
            f"requests={summary.request_count} | latency={summary.requests_with_latency} | "
            f"provider_counters={summary.requests_with_provider_counters}"
        )
        lines.append(
            f"prompt={summary.prompt_tokens} | text={summary.prompt_text_tokens} | "
            f"image={summary.prompt_image_tokens} | completion={summary.completion_tokens}"
        )
        if summary.requests_with_provider_counters > 0:
            lines.append(
                f"cache_coverage=req {summary.cache_metric_request_coverage_ratio * 100:.1f}% | "
                f"prompt {summary.cache_metric_prompt_coverage_ratio * 100:.1f}%"
            )
            lines.append(
                f"cached={summary.cached_tokens} | create={summary.cache_creation_input_tokens} | "
                f"uncached={summary.uncached_prompt_tokens}"
            )
            lines.append(
                f"hit(weighted)={summary.weighted_cache_hit_ratio * 100:.1f}% | "
                f"hit(avg/med)={summary.average_cache_hit_ratio * 100:.1f}%/"
                f"{summary.median_cache_hit_ratio * 100:.1f}%"
            )
            lines.append(
                f"input_cost_x={summary.effective_input_cost_multiplier:.3f} | "
                f"savings={summary.input_cost_savings_ratio * 100:+.1f}%"
            )
        else:
            lines.append("cache metrics unavailable: provider explicit cache counters not found")

        if summary.requests_with_latency > 0:
            lines.append(
                f"latency(avg/med)={summary.average_latency_s:.2f}s/{summary.median_latency_s:.2f}s | "
                f"speed(avg/med/e2e)={summary.average_tokens_per_second:.1f}/"
                f"{summary.median_tokens_per_second:.1f}/"
                f"{summary.end_to_end_tokens_per_second:.1f} tok/s"
            )
        else:
            lines.append("speed metrics unavailable: latency artifacts not found")

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
    parser = argparse.ArgumentParser(description="Summarize Qwen explicit-cache usage artifacts")
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
