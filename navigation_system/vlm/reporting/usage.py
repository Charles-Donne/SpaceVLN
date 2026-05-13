"""Token and cost aggregation helpers for VLM request artifacts and episode logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml


DEFAULT_PRICE_TABLE_PATH = Path(__file__).resolve().parents[2] / "config" / "vlm" / "model_pricing.yaml"
DEFAULT_CURRENCY = "CNY"


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


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def load_price_table(path: Optional[str] = None) -> Dict[str, Any]:
    candidate = Path(path).expanduser().resolve() if path else DEFAULT_PRICE_TABLE_PATH
    if not candidate.exists():
        return {}
    with candidate.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _match_model_pricing(
    *,
    provider: str,
    model: str,
    price_table: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    provider_key = _normalize_text(provider)
    model_key = _normalize_text(model)
    if not provider_key or not model_key:
        return None

    provider_models = dict((price_table.get("models") or {}).get(provider_key) or {})
    if not provider_models:
        return None

    if model_key in provider_models:
        return dict(provider_models[model_key])

    for name, payload in provider_models.items():
        if model_key == _normalize_text(name):
            return dict(payload or {})
        aliases = list((payload or {}).get("aliases") or [])
        if any(model_key == _normalize_text(alias) for alias in aliases):
            return dict(payload or {})
        if model_key.startswith(_normalize_text(name)):
            return dict(payload or {})
    return None


def _select_tier(pricing: Dict[str, Any], input_tokens: int) -> Dict[str, Any]:
    tiers = list(pricing.get("tiers") or [])
    if not tiers:
        return {}
    target = max(0, int(input_tokens or 0))
    sorted_tiers = sorted(
        [dict(item or {}) for item in tiers],
        key=lambda item: (
            int(item.get("max_input_tokens", 0) or 0) if item.get("max_input_tokens") is not None else 10**18
        ),
    )
    for tier in sorted_tiers:
        max_input = tier.get("max_input_tokens")
        if max_input is None:
            return tier
        try:
            if target <= int(max_input):
                return tier
        except (TypeError, ValueError):
            continue
    return sorted_tiers[-1]


def _pricing_currency(pricing: Optional[Dict[str, Any]], price_table: Dict[str, Any]) -> str:
    return str((pricing or {}).get("currency") or price_table.get("currency") or DEFAULT_CURRENCY)


def _single_cost_currency(costs: Sequence[Dict[str, Any]], fallback: str) -> str:
    currencies = {
        str(item.get("currency") or "").strip()
        for item in costs
        if bool(item.get("cost_available", False)) and str(item.get("currency") or "").strip()
    }
    if len(currencies) == 1:
        return next(iter(currencies))
    if len(currencies) > 1:
        return "mixed"
    return str(fallback or DEFAULT_CURRENCY)


def compact_vlm_info_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    cache = dict(payload.get("cache") or {})
    compact = {
        "model": str(payload.get("model") or ""),
        "provider": str(payload.get("provider") or ""),
        "success": bool(payload.get("success", False)),
        "duration_s": _safe_float(payload.get("duration_s"), 0.0),
        "request_latency_s": _safe_float(payload.get("request_latency_s"), 0.0),
        "total_call_duration_s": _safe_float(payload.get("total_call_duration_s"), 0.0),
        "attempts": _safe_int(payload.get("attempts"), 1),
        "failed_attempts": _safe_int(payload.get("failed_attempts"), 0),
        "failed_retry_wait_time_s": _safe_float(payload.get("failed_retry_wait_time_s"), 0.0),
        "failed_wasted_time_s": _safe_float(payload.get("failed_wasted_time_s"), 0.0),
        "input_tokens": _safe_int(payload.get("input_tokens"), 0),
        "input_text_tokens": _safe_int(payload.get("input_text_tokens"), 0),
        "input_image_tokens": _safe_int(payload.get("input_image_tokens"), 0),
        "output_tokens": _safe_int(payload.get("output_tokens"), 0),
        "total_tokens": _safe_int(payload.get("total_tokens"), 0),
    }
    if cache:
        compact["cache"] = {
            "enabled": bool(cache.get("enabled", False)),
            "type": str(cache.get("type") or ""),
            "reported": bool(cache.get("reported", False)),
            "cached_tokens": _safe_int(cache.get("cached_tokens"), 0),
            "write_tokens": _safe_int(cache.get("write_tokens"), 0),
            "uncached_tokens": _safe_int(cache.get("uncached_tokens"), 0),
            "hit_ratio": _safe_float(cache.get("hit_ratio"), 0.0),
            "cost_ratio": _safe_float(cache.get("cost_ratio"), 0.0),
            "savings_ratio": _safe_float(cache.get("savings_ratio"), 0.0),
        }
    status = str(payload.get("status") or "").strip()
    error = str(payload.get("error") or "").strip()
    http_status = payload.get("http_status")
    if status:
        compact["status"] = status
    if error:
        compact["error"] = error
    if http_status:
        compact["http_status"] = _safe_int(http_status, 0)
    return compact


def _resolve_cache_multipliers(cache_type: str, table: Dict[str, Any]) -> Dict[str, float]:
    cache_cfg = dict(table.get("cache") or {})
    explicit = dict(cache_cfg.get("explicit") or {})
    implicit = dict(cache_cfg.get("implicit") or {})
    normalized = _normalize_text(cache_type)
    if normalized in {"ephemeral", "explicit", "context_cache", "provider"}:
        return {
            "read": _safe_float(explicit.get("read_input_multiplier"), 0.1),
            "write": _safe_float(explicit.get("write_input_multiplier"), 1.25),
        }
    if normalized in {"implicit"}:
        return {
            "read": _safe_float(implicit.get("read_input_multiplier"), 0.2),
            "write": _safe_float(implicit.get("write_input_multiplier"), 1.0),
        }
    return {
        "read": 1.0,
        "write": 1.0,
    }


def estimate_vlm_usage_cost(
    payload: Optional[Dict[str, Any]],
    *,
    price_table: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    info = compact_vlm_info_payload(payload)
    if not info:
        return {
            "cost_available": False,
            "currency": str((price_table or {}).get("currency") or DEFAULT_CURRENCY),
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "billed_input_tokens": 0.0,
            "input_multiplier": 1.0,
            "output_multiplier": 1.0,
        }

    price_table = dict(price_table or load_price_table())
    pricing = _match_model_pricing(
        provider=info.get("provider", ""),
        model=info.get("model", ""),
        price_table=price_table,
    )
    if not pricing:
        return {
            "cost_available": False,
            "currency": str(price_table.get("currency") or DEFAULT_CURRENCY),
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "billed_input_tokens": 0.0,
            "input_multiplier": 1.0,
            "output_multiplier": 1.0,
        }

    currency = _pricing_currency(pricing, price_table)
    input_tokens = _safe_int(info.get("input_tokens"), 0)
    output_tokens = _safe_int(info.get("output_tokens"), 0)
    tier = _select_tier(pricing, input_tokens)
    input_price = _safe_float(tier.get("input"), 0.0)
    output_price = _safe_float(tier.get("output"), 0.0)
    cache = dict((payload or {}).get("cache") or {})
    cache_reported = bool(cache.get("reported", False))
    cached_tokens = _safe_int(cache.get("cached_tokens"), 0)
    write_tokens = _safe_int(cache.get("write_tokens"), 0)
    uncached_tokens = _safe_int(cache.get("uncached_tokens"), 0)
    cache_type = str(cache.get("type") or "")
    if cache_reported:
        if uncached_tokens <= 0 and (cached_tokens > 0 or write_tokens > 0):
            uncached_tokens = max(0, input_tokens - cached_tokens - write_tokens)
        multipliers = _resolve_cache_multipliers(cache_type, price_table)
        billed_input_tokens = (
            float(max(0, uncached_tokens))
            + float(max(0, cached_tokens)) * float(multipliers["read"])
            + float(max(0, write_tokens)) * float(multipliers["write"])
        )
    else:
        billed_input_tokens = float(input_tokens)

    input_cost = billed_input_tokens * input_price / 1_000_000.0
    output_cost = float(output_tokens) * output_price / 1_000_000.0
    total_cost = input_cost + output_cost

    if input_tokens > 0 and cache_reported:
        billed_input_multiplier = billed_input_tokens / float(input_tokens)
    else:
        billed_input_multiplier = 1.0

    return {
        "cost_available": True,
        "currency": currency,
        "input_price_per_1m": input_price,
        "output_price_per_1m": output_price,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "billed_input_tokens": billed_input_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "input_multiplier": billed_input_multiplier,
        "output_multiplier": 1.0,
        "cache_reported": cache_reported,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": write_tokens,
        "uncached_tokens": uncached_tokens,
        "cache_hit_ratio": _safe_float(cache.get("hit_ratio"), 0.0),
        "cache_cost_ratio": _safe_float(cache.get("cost_ratio"), 0.0),
        "cache_savings_ratio": _safe_float(cache.get("savings_ratio"), 0.0),
    }


def _mean(values: Sequence[float]) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


def _normalize_cost_counts(summary: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(summary or {})
    count = _safe_int(payload.get("count"), 0)
    if count <= 0:
        payload.setdefault("cost_available_count", 0)
        payload.setdefault("cost_unavailable_count", 0)
        return payload

    available = _safe_int(payload.get("cost_available_count"), 0)
    unavailable = _safe_int(payload.get("cost_unavailable_count"), 0)
    if available + unavailable <= 0:
        has_cost_fields = any(key in payload for key in ("total_cost", "input_cost", "output_cost"))
        if has_cost_fields:
            available = count
        else:
            unavailable = count
    elif available + unavailable < count:
        unavailable += count - available - unavailable

    payload["cost_available_count"] = min(available, count)
    payload["cost_unavailable_count"] = min(unavailable, count - payload["cost_available_count"])
    return payload


def _normalize_token_counts(summary: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(summary or {})
    count = _safe_int(payload.get("count"), 0)
    if count <= 0:
        payload.setdefault("token_available_count", 0)
        payload.setdefault("token_unavailable_count", 0)
        return payload

    available = _safe_int(payload.get("token_available_count"), 0)
    unavailable = _safe_int(payload.get("token_unavailable_count"), 0)
    if available + unavailable <= 0:
        has_token_fields = any(key in payload for key in ("total_tokens", "input_tokens", "output_tokens"))
        if has_token_fields:
            available = count
        else:
            unavailable = count
    elif available + unavailable < count:
        unavailable += count - available - unavailable

    payload["token_available_count"] = min(available, count)
    payload["token_unavailable_count"] = min(unavailable, count - payload["token_available_count"])
    return payload


def summarize_vlm_usage(
    payloads: Iterable[Optional[Dict[str, Any]]],
    *,
    price_table: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    price_table = dict(price_table or load_price_table())
    records = [compact_vlm_info_payload(payload) for payload in payloads if isinstance(payload, dict)]
    records = [record for record in records if record]
    count = len(records)
    if count <= 0:
        return {
            "count": 0,
            "currency": str(price_table.get("currency") or DEFAULT_CURRENCY),
            "cost_available_count": 0,
            "cost_unavailable_count": 0,
            "token_available_count": 0,
            "token_unavailable_count": 0,
            "input_tokens": 0,
            "input_text_tokens": 0,
            "input_image_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "avg_total_tokens": 0.0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "uncached_tokens": 0,
            "cached_nonzero_tokens": 0,
            "cache_write_nonzero_tokens": 0,
            "uncached_nonzero_tokens": 0,
            "cache_reported_count": 0,
            "cache_nonzero_count": 0,
            "avg_cached_tokens_per_cache_call": 0.0,
            "avg_cache_write_tokens_per_cache_call": 0.0,
            "avg_uncached_tokens_per_cache_call": 0.0,
            "weighted_cache_hit_ratio": 0.0,
            "cache_cost_ratio": 0.0,
            "cache_savings_ratio": 0.0,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "avg_cost_per_call": 0.0,
            "avg_input_cost_per_call": 0.0,
            "avg_output_cost_per_call": 0.0,
            "cost_per_1k_output_tokens": 0.0,
            "latency_total_s": 0.0,
            "latency_avg_s": 0.0,
        }

    costs = [estimate_vlm_usage_cost(record, price_table=price_table) for record in records]
    cost_available = [item for item in costs if bool(item.get("cost_available", False))]
    cache_reported_records = [item for item in costs if bool(item.get("cache_reported", False))]
    cache_nonzero_records = [
        item
        for item in cache_reported_records
        if int(item.get("cached_tokens", 0) or 0) > 0
        or int(item.get("cache_write_tokens", 0) or 0) > 0
    ]
    latency_values = [_safe_float(record.get("duration_s"), 0.0) for record in records]

    input_tokens = sum(_safe_int(item.get("input_tokens"), 0) for item in records)
    input_text_tokens = sum(_safe_int(item.get("input_text_tokens"), 0) for item in records)
    input_image_tokens = sum(_safe_int(item.get("input_image_tokens"), 0) for item in records)
    output_tokens = sum(_safe_int(item.get("output_tokens"), 0) for item in records)
    total_tokens = sum(_safe_int(item.get("total_tokens"), 0) for item in records)
    cached_tokens = sum(_safe_int(item.get("cached_tokens"), 0) for item in cache_reported_records)
    cache_write_tokens = sum(_safe_int(item.get("cache_write_tokens"), 0) for item in cache_reported_records)
    uncached_tokens = sum(_safe_int(item.get("uncached_tokens"), 0) for item in cache_reported_records)
    cached_nonzero_tokens = sum(_safe_int(item.get("cached_tokens"), 0) for item in cache_nonzero_records)
    cache_write_nonzero_tokens = sum(_safe_int(item.get("cache_write_tokens"), 0) for item in cache_nonzero_records)
    uncached_nonzero_tokens = sum(_safe_int(item.get("uncached_tokens"), 0) for item in cache_nonzero_records)

    cache_denominator = float(sum(_safe_int(item.get("input_tokens"), 0) for item in cache_nonzero_records))
    weighted_cache_hit_ratio = (
        float(cached_nonzero_tokens) / cache_denominator if cache_denominator > 0 else 0.0
    )
    cache_cost_ratio = (
        float(sum(_safe_float(item.get("cache_cost_ratio"), 0.0) for item in cache_nonzero_records)) / len(cache_nonzero_records)
        if cache_nonzero_records
        else 0.0
    )
    cache_savings_ratio = (
        float(sum(_safe_float(item.get("cache_savings_ratio"), 0.0) for item in cache_nonzero_records)) / len(cache_nonzero_records)
        if cache_nonzero_records
        else 0.0
    )
    summary_currency = _single_cost_currency(
        cost_available,
        str(price_table.get("currency") or DEFAULT_CURRENCY),
    )
    cost_available_for_sum = cost_available if summary_currency != "mixed" else []
    total_cost = sum(_safe_float(item.get("total_cost"), 0.0) for item in cost_available_for_sum)
    input_cost = sum(_safe_float(item.get("input_cost"), 0.0) for item in cost_available_for_sum)
    output_cost = sum(_safe_float(item.get("output_cost"), 0.0) for item in cost_available_for_sum)

    return {
        "count": count,
        "currency": summary_currency,
        "cost_available_count": len(cost_available_for_sum),
        "cost_unavailable_count": count - len(cost_available_for_sum),
        "token_available_count": count,
        "token_unavailable_count": 0,
        "input_tokens": input_tokens,
        "input_text_tokens": input_text_tokens,
        "input_image_tokens": input_image_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "avg_input_tokens": float(input_tokens) / count if count > 0 else 0.0,
        "avg_output_tokens": float(output_tokens) / count if count > 0 else 0.0,
        "avg_total_tokens": float(total_tokens) / count if count > 0 else 0.0,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "uncached_tokens": uncached_tokens,
        "cached_nonzero_tokens": cached_nonzero_tokens,
        "cache_write_nonzero_tokens": cache_write_nonzero_tokens,
        "uncached_nonzero_tokens": uncached_nonzero_tokens,
        "cache_reported_count": len(cache_reported_records),
        "cache_nonzero_count": len(cache_nonzero_records),
        "avg_cached_tokens_per_cache_call": float(cached_nonzero_tokens) / len(cache_nonzero_records) if cache_nonzero_records else 0.0,
        "avg_cache_write_tokens_per_cache_call": float(cache_write_nonzero_tokens) / len(cache_nonzero_records) if cache_nonzero_records else 0.0,
        "avg_uncached_tokens_per_cache_call": float(uncached_nonzero_tokens) / len(cache_nonzero_records) if cache_nonzero_records else 0.0,
        "weighted_cache_hit_ratio": weighted_cache_hit_ratio,
        "cache_cost_ratio": cache_cost_ratio,
        "cache_savings_ratio": cache_savings_ratio,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "avg_cost_per_call": total_cost / count if count > 0 else 0.0,
        "avg_input_cost_per_call": input_cost / count if count > 0 else 0.0,
        "avg_output_cost_per_call": output_cost / count if count > 0 else 0.0,
        "cost_per_1k_output_tokens": (total_cost * 1000.0 / output_tokens) if output_tokens > 0 else 0.0,
        "latency_total_s": sum(latency_values),
        "latency_avg_s": _mean(latency_values),
    }


def _classify_request_kind(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "thinking" in parts:
        return "thinking"
    if "action" in parts:
        return "action"
    return "other"


def summarize_vlm_usage_from_artifact_dir(root_dir: str) -> Dict[str, Any]:
    root = Path(root_dir).expanduser().resolve()
    grouped: Dict[str, List[Dict[str, Any]]] = {
        "thinking": [],
        "action": [],
        "other": [],
    }
    if not root.exists():
        empty = summarize_vlm_usage([])
        return {
            "thinking": empty,
            "action": empty,
            "other": empty,
            "overall": empty,
        }
    for path in sorted(root.rglob("vlm_info.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f) or {}
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        grouped.setdefault(_classify_request_kind(path), []).append(payload)

    thinking = summarize_vlm_usage(grouped.get("thinking", []))
    action = summarize_vlm_usage(grouped.get("action", []))
    other = summarize_vlm_usage(grouped.get("other", []))
    overall = merge_vlm_usage_summaries([thinking, action, other])
    return {
        "thinking": thinking,
        "action": action,
        "other": other,
        "overall": overall,
    }


def merge_vlm_usage_summaries(summaries: Iterable[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    rows = [
        _normalize_token_counts(_normalize_cost_counts(dict(summary or {})))
        for summary in summaries
        if isinstance(summary, dict) and summary
    ]
    if not rows:
        return summarize_vlm_usage([])

    cost_currencies = {
        str(row.get("currency") or "").strip()
        for row in rows
        if _safe_int(row.get("cost_available_count"), 0) > 0
        and str(row.get("currency") or "").strip()
    }
    currency = next(iter(cost_currencies)) if len(cost_currencies) == 1 else (
        "mixed" if len(cost_currencies) > 1 else str(rows[0].get("currency") or DEFAULT_CURRENCY)
    )
    count = sum(_safe_int(row.get("count"), 0) for row in rows)
    if count <= 0:
        return summarize_vlm_usage([])

    input_tokens = sum(_safe_int(row.get("input_tokens"), 0) for row in rows)
    input_text_tokens = sum(_safe_int(row.get("input_text_tokens"), 0) for row in rows)
    input_image_tokens = sum(_safe_int(row.get("input_image_tokens"), 0) for row in rows)
    output_tokens = sum(_safe_int(row.get("output_tokens"), 0) for row in rows)
    total_tokens = sum(_safe_int(row.get("total_tokens"), 0) for row in rows)
    cached_tokens = sum(_safe_int(row.get("cached_tokens"), 0) for row in rows)
    cache_write_tokens = sum(_safe_int(row.get("cache_write_tokens"), 0) for row in rows)
    uncached_tokens = sum(_safe_int(row.get("uncached_tokens"), 0) for row in rows)
    cached_nonzero_tokens = sum(
        _safe_int(row.get("cached_nonzero_tokens", row.get("cached_tokens", 0)), 0)
        for row in rows
    )
    cache_write_nonzero_tokens = sum(
        _safe_int(row.get("cache_write_nonzero_tokens", row.get("cache_write_tokens", 0)), 0)
        for row in rows
    )
    uncached_nonzero_tokens = sum(
        _safe_int(row.get("uncached_nonzero_tokens", row.get("uncached_tokens", 0)), 0)
        for row in rows
    )
    raw_cost_available_count = sum(_safe_int(row.get("cost_available_count"), 0) for row in rows)
    raw_cost_unavailable_count = sum(_safe_int(row.get("cost_unavailable_count"), 0) for row in rows)
    cost_available_count = raw_cost_available_count if currency != "mixed" else 0
    cost_unavailable_count = (
        raw_cost_unavailable_count if currency != "mixed" else count
    )
    token_available_count = sum(_safe_int(row.get("token_available_count"), 0) for row in rows)
    token_unavailable_count = sum(_safe_int(row.get("token_unavailable_count"), 0) for row in rows)
    cache_reported_count = sum(_safe_int(row.get("cache_reported_count"), 0) for row in rows)
    cache_nonzero_count = sum(_safe_int(row.get("cache_nonzero_count"), 0) for row in rows)
    total_cost = sum(_safe_float(row.get("total_cost"), 0.0) for row in rows) if currency != "mixed" else 0.0
    input_cost = sum(_safe_float(row.get("input_cost"), 0.0) for row in rows) if currency != "mixed" else 0.0
    output_cost = sum(_safe_float(row.get("output_cost"), 0.0) for row in rows) if currency != "mixed" else 0.0
    latency_total_s = sum(_safe_float(row.get("latency_total_s"), 0.0) for row in rows)
    latency_avg_s = latency_total_s / count if count > 0 else 0.0
    return {
        "count": count,
        "currency": currency,
        "cost_available_count": cost_available_count,
        "cost_unavailable_count": cost_unavailable_count,
        "token_available_count": token_available_count,
        "token_unavailable_count": token_unavailable_count,
        "input_tokens": input_tokens,
        "input_text_tokens": input_text_tokens,
        "input_image_tokens": input_image_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "avg_input_tokens": float(input_tokens) / count if count > 0 else 0.0,
        "avg_output_tokens": float(output_tokens) / count if count > 0 else 0.0,
        "avg_total_tokens": float(total_tokens) / count if count > 0 else 0.0,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "uncached_tokens": uncached_tokens,
        "cache_reported_count": cache_reported_count,
        "cache_nonzero_count": cache_nonzero_count,
        "cached_nonzero_tokens": cached_nonzero_tokens,
        "cache_write_nonzero_tokens": cache_write_nonzero_tokens,
        "uncached_nonzero_tokens": uncached_nonzero_tokens,
        "avg_cached_tokens_per_cache_call": float(cached_nonzero_tokens) / cache_nonzero_count if cache_nonzero_count > 0 else 0.0,
        "avg_cache_write_tokens_per_cache_call": float(cache_write_nonzero_tokens) / cache_nonzero_count if cache_nonzero_count > 0 else 0.0,
        "avg_uncached_tokens_per_cache_call": float(uncached_nonzero_tokens) / cache_nonzero_count if cache_nonzero_count > 0 else 0.0,
        "weighted_cache_hit_ratio": (
            float(cached_nonzero_tokens) / float(cached_nonzero_tokens + cache_write_nonzero_tokens + uncached_nonzero_tokens)
            if (cached_nonzero_tokens + cache_write_nonzero_tokens + uncached_nonzero_tokens) > 0
            else 0.0
        ),
        "cache_cost_ratio": sum(_safe_float(row.get("cache_cost_ratio"), 0.0) for row in rows) / len(rows),
        "cache_savings_ratio": sum(_safe_float(row.get("cache_savings_ratio"), 0.0) for row in rows) / len(rows),
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "avg_cost_per_call": total_cost / count if count > 0 else 0.0,
        "avg_input_cost_per_call": input_cost / count if count > 0 else 0.0,
        "avg_output_cost_per_call": output_cost / count if count > 0 else 0.0,
        "cost_per_1k_output_tokens": (total_cost * 1000.0 / output_tokens) if output_tokens > 0 else 0.0,
        "latency_total_s": latency_total_s,
        "latency_avg_s": latency_avg_s,
    }


__all__ = [
    "DEFAULT_PRICE_TABLE_PATH",
    "compact_vlm_info_payload",
    "DEFAULT_CURRENCY",
    "estimate_vlm_usage_cost",
    "load_price_table",
    "merge_vlm_usage_summaries",
    "summarize_vlm_usage_from_artifact_dir",
    "summarize_vlm_usage",
]
