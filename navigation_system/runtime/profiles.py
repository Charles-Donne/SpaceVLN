"""Runtime profiles that wire together result dirs, model stacks, and post-run hooks."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from navigation_system.vlm.api.api_client import (
    build_default_results_dir_from_api_config,
)
from navigation_system.vlm.api.qwen_context_cache_client import (
    build_default_qwen_context_cache_results_dir,
)
from navigation_system.vlm.reporting.cache_report import (
    print_cache_report_summary,
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
    default_results_dir_builder: Callable[..., str]
    model_stack_builder: Callable[..., Any]
    post_run_hook: Optional[Callable[[Any, Any], None]] = None


def _maybe_generate_qwen_cache_report(args, config) -> None:
    results_dir = str(
        getattr(args, "results_dir", "") or getattr(config, "RESULTS_DIR", "") or ""
    ).strip()
    print_cache_report_summary(results_dir)


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
