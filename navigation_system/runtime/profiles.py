"""Runtime profiles that wire together result dirs, model stacks, and post-run hooks."""

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from navigation_system.vlm.api.api_client import (
    build_default_results_dir_from_api_config,
)
from navigation_system.vlm.api.qwen_context_cache_client import (
    build_default_qwen_context_cache_results_dir,
)
from navigation_system.vlm.reporting.cache_report import (
    build_cache_report,
    render_cache_report,
    write_cache_report_json,
)
from navigation_system.vlm.runtime_factory import (
    build_default_navigation_model_stack,
    build_qwen_context_cache_navigation_model_stack,
)


DEFAULT_API_CONFIG = "navigation_system/config/vlm/vlm_api_config.yaml"
DEFAULT_QWEN_CACHE_API_CONFIG = "navigation_system/config/vlm/vlm_api_config_qwen_cache.yaml"


@dataclass(frozen=True)
class NavigationRuntimeProfile:
    name: str
    default_api_config_path: str
    default_results_dir_builder: Callable[[str], str]
    model_stack_builder: Callable[..., Any]
    post_run_hook: Optional[Callable[[Any, Any], None]] = None


def _write_text_report(report_text: str, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        if not report_text.endswith("\n"):
            f.write("\n")
    return output_path


def _maybe_generate_qwen_cache_report(args, config) -> None:
    results_dir = str(
        getattr(args, "results_dir", "") or getattr(config, "RESULTS_DIR", "") or ""
    ).strip()
    if not results_dir:
        return

    try:
        report = build_cache_report(results_dir)
    except Exception as exc:
        print(f"⚠️  无法生成缓存报告: {exc}")
        return

    if report.overall.request_count <= 0:
        return

    json_path = os.path.join(results_dir, "cache_report_latest.json")
    txt_path = os.path.join(results_dir, "cache_report_latest.txt")
    report_text = render_cache_report(report)
    write_cache_report_json(report, json_path)
    _write_text_report(report_text, txt_path)

    overall = report.overall
    output_speed = (
        float(overall.output_tokens) / float(overall.total_duration_s)
        if overall.total_duration_s > 0
        else 0.0
    )
    reported_ratio = (
        float(overall.cache_reported_requests) / float(overall.request_count)
        if overall.request_count > 0
        else 0.0
    )
    if overall.cache_reported_requests > 0:
        print(
            "[VLM][cache][overall] "
            f"req={overall.request_count} "
            f"| reported={reported_ratio * 100:.1f}% "
            f"| hit={overall.weighted_cache_hit_ratio * 100:.1f}% "
            f"| cost={overall.cost_ratio:.3f}x "
            f"| speed={output_speed:.1f} tok/s"
        )
    else:
        print(
            "[VLM][cache][overall] "
            f"req={overall.request_count} | reported=0 "
            "| cache metrics unavailable in current artifacts"
        )
    print(f"[VLM][cache][report] {txt_path}")


STANDARD_RUNTIME_PROFILE = NavigationRuntimeProfile(
    name="standard",
    default_api_config_path=DEFAULT_API_CONFIG,
    default_results_dir_builder=build_default_results_dir_from_api_config,
    model_stack_builder=build_default_navigation_model_stack,
)

QWEN_CONTEXT_CACHE_RUNTIME_PROFILE = NavigationRuntimeProfile(
    name="qwen_context_cache",
    default_api_config_path=DEFAULT_QWEN_CACHE_API_CONFIG,
    default_results_dir_builder=build_default_qwen_context_cache_results_dir,
    model_stack_builder=build_qwen_context_cache_navigation_model_stack,
    post_run_hook=_maybe_generate_qwen_cache_report,
)

RUNTIME_PROFILES_BY_NAME: Dict[str, NavigationRuntimeProfile] = {
    STANDARD_RUNTIME_PROFILE.name: STANDARD_RUNTIME_PROFILE,
    QWEN_CONTEXT_CACHE_RUNTIME_PROFILE.name: QWEN_CONTEXT_CACHE_RUNTIME_PROFILE,
}


def resolve_runtime_profile(profile_name: str) -> NavigationRuntimeProfile:
    normalized = str(profile_name or "").strip()
    if normalized not in RUNTIME_PROFILES_BY_NAME:
        raise KeyError(f"Unknown runtime profile: {profile_name}")
    return RUNTIME_PROFILES_BY_NAME[normalized]


__all__ = [
    "DEFAULT_API_CONFIG",
    "DEFAULT_QWEN_CACHE_API_CONFIG",
    "NavigationRuntimeProfile",
    "QWEN_CONTEXT_CACHE_RUNTIME_PROFILE",
    "RUNTIME_PROFILES_BY_NAME",
    "STANDARD_RUNTIME_PROFILE",
    "resolve_runtime_profile",
]
