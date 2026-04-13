"""Runtime profiles for the isolated ablation entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from navigation_system.ablation.config import load_ablation_spec
from navigation_system.ablation.runtime.layout import (
    build_ablation_results_dir_from_model_dir,
    build_default_qwen_context_cache_results_dir,
    build_default_results_dir_from_api_config,
)
from navigation_system.vlm.reporting.cache_report import (
    print_cache_report_summary,
)


DEFAULT_API_CONFIG = "navigation_system/config/vlm/vlm_api_config.yaml"
DEFAULT_QWEN_CACHE_API_CONFIG = "navigation_system/config/vlm/vlm_api_config_qwen_cache.yaml"


@dataclass(frozen=True)
class AblationRuntimeProfile:
    name: str
    default_api_config_path: str
    default_results_dir_builder: Callable[..., str]
    model_stack_builder: Callable[..., Any]
    post_run_hook: Optional[Callable[[Any, Any], None]] = None


def _build_standard_model_stack(**kwargs):
    from navigation_system.ablation.runtime.factory import (
        build_ablation_navigation_model_stack,
    )

    return build_ablation_navigation_model_stack(**kwargs)


def _build_qwen_cache_model_stack(**kwargs):
    from navigation_system.ablation.runtime.factory import (
        build_ablation_qwen_context_cache_navigation_model_stack,
    )

    return build_ablation_qwen_context_cache_navigation_model_stack(**kwargs)


def _maybe_generate_qwen_cache_report(args, config) -> None:
    results_dir = str(
        getattr(args, "results_dir", "") or getattr(config, "RESULTS_DIR", "") or ""
    ).strip()
    print_cache_report_summary(results_dir)


def _build_ablation_results_dir(
    config_path: str,
    repo_root: str = None,
    results_root: str = None,
) -> str:
    spec = load_ablation_spec()
    return build_ablation_results_dir_from_model_dir(
        build_default_results_dir_from_api_config(
            config_path,
            repo_root=repo_root,
            results_root=results_root,
        ),
        ablation_slug=spec.slug,
        repo_root=repo_root,
        results_root=results_root,
    )


def _build_ablation_qwen_cache_results_dir(
    config_path: str,
    repo_root: str = None,
    results_root: str = None,
) -> str:
    spec = load_ablation_spec()
    return build_ablation_results_dir_from_model_dir(
        build_default_qwen_context_cache_results_dir(
            config_path,
            repo_root=repo_root,
            results_root=results_root,
        ),
        ablation_slug=spec.slug,
        repo_root=repo_root,
        results_root=results_root,
    )


ABLATION_STANDARD_RUNTIME_PROFILE = AblationRuntimeProfile(
    name="ablation_standard",
    default_api_config_path=DEFAULT_API_CONFIG,
    default_results_dir_builder=_build_ablation_results_dir,
    model_stack_builder=_build_standard_model_stack,
)

ABLATION_QWEN_CONTEXT_CACHE_RUNTIME_PROFILE = AblationRuntimeProfile(
    name="ablation_qwen_context_cache",
    default_api_config_path=DEFAULT_QWEN_CACHE_API_CONFIG,
    default_results_dir_builder=_build_ablation_qwen_cache_results_dir,
    model_stack_builder=_build_qwen_cache_model_stack,
    post_run_hook=_maybe_generate_qwen_cache_report,
)

ABLATION_RUNTIME_PROFILES_BY_NAME: Dict[str, AblationRuntimeProfile] = {
    ABLATION_STANDARD_RUNTIME_PROFILE.name: ABLATION_STANDARD_RUNTIME_PROFILE,
    ABLATION_QWEN_CONTEXT_CACHE_RUNTIME_PROFILE.name: ABLATION_QWEN_CONTEXT_CACHE_RUNTIME_PROFILE,
}


def resolve_ablation_runtime_profile(profile_name: str) -> AblationRuntimeProfile:
    normalized = str(profile_name or "").strip()
    if normalized not in ABLATION_RUNTIME_PROFILES_BY_NAME:
        raise KeyError(f"Unknown ablation runtime profile: {profile_name}")
    return ABLATION_RUNTIME_PROFILES_BY_NAME[normalized]


__all__ = [
    "ABLATION_QWEN_CONTEXT_CACHE_RUNTIME_PROFILE",
    "ABLATION_RUNTIME_PROFILES_BY_NAME",
    "ABLATION_STANDARD_RUNTIME_PROFILE",
    "AblationRuntimeProfile",
    "DEFAULT_API_CONFIG",
    "DEFAULT_QWEN_CACHE_API_CONFIG",
    "resolve_ablation_runtime_profile",
]
