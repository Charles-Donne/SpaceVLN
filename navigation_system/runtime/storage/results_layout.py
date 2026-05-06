"""Shared results-path and model-dir naming helpers for navigation runtimes."""

from __future__ import annotations

import os
import re
from typing import Tuple

import yaml


DEFAULT_RESULTS_ROOT_RELATIVE = "result"
DEFAULT_RESULTS_FAMILY = "r2rce"
DEFAULT_CONTEXT_CACHE_API_CONFIG = os.path.join(
    "navigation_system",
    "config",
    "vlm",
    "vlm_api_config.yaml",
)
DEFAULT_SYSTEM_RUNTIME_CONFIG_RELATIVE = os.path.join(
    "navigation_system",
    "config",
    "system",
    "00_runtime.yaml",
)


def resolve_api_config_path(config_path: str) -> str:
    """Normalize the configured unified API config path."""
    normalized = str(config_path or "").strip()
    if not normalized:
        return ""

    repo_root = _resolve_repo_root()
    expanded = os.path.expanduser(normalized)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded) if os.path.exists(expanded) else expanded

    repo_candidate = os.path.abspath(os.path.join(repo_root, expanded))
    if os.path.exists(repo_candidate):
        return repo_candidate
    if os.path.exists(expanded):
        return os.path.abspath(expanded)

    return normalized


def _resolve_repo_root(repo_root: str = None) -> str:
    if repo_root:
        return os.path.abspath(repo_root)
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )


def _resolve_workspace_root(repo_root: str = None) -> str:
    return os.path.abspath(os.path.join(_resolve_repo_root(repo_root), ".."))


def _resolve_optional_results_path(raw_path: str, repo_root: str = None) -> str:
    text = str(raw_path or "").strip()
    if not text:
        return ""
    expanded = os.path.expanduser(text)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(_resolve_workspace_root(repo_root), expanded))


def resolve_results_root_path(results_root: str, repo_root: str = None) -> str:
    resolved = _resolve_optional_results_path(results_root, repo_root=repo_root)
    if resolved:
        return resolved
    return os.path.abspath(
        os.path.join(_resolve_workspace_root(repo_root), DEFAULT_RESULTS_ROOT_RELATIVE)
    )


def resolve_results_dir_path(results_dir: str, repo_root: str = None) -> str:
    return _resolve_optional_results_path(results_dir, repo_root=repo_root)


def _load_yaml_mapping(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def _load_system_results_root_setting(repo_root: str = None) -> str:
    runtime_config_path = os.path.join(
        _resolve_repo_root(repo_root),
        DEFAULT_SYSTEM_RUNTIME_CONFIG_RELATIVE,
    )
    if not os.path.exists(runtime_config_path):
        return ""
    raw = _load_yaml_mapping(runtime_config_path)
    return str(((raw.get("PATHS") or {}).get("RESULTS_ROOT")) or "").strip()


def build_default_results_root(repo_root: str = None) -> str:
    configured_root = (
        str(os.getenv("SPACEVLN_RESULTS_ROOT", "") or "").strip()
        or _load_system_results_root_setting(repo_root=repo_root)
    )
    return resolve_results_root_path(configured_root, repo_root=repo_root)


def build_default_results_family_root(
    family: str = DEFAULT_RESULTS_FAMILY,
    *,
    repo_root: str = None,
    results_root: str = None,
) -> str:
    base_root = (
        resolve_results_root_path(results_root, repo_root=repo_root)
        if str(results_root or "").strip()
        else build_default_results_root(repo_root=repo_root)
    )
    family_name = str(family or "").strip() or DEFAULT_RESULTS_FAMILY
    return os.path.abspath(os.path.join(base_root, family_name))


def _slugify_results_component(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/").replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    return text or "unknown"


def _build_results_component_fallback(
    config_path: str,
    *,
    provider: str,
    role: str,
) -> str:
    resolved_path = resolve_api_config_path(config_path)
    config_stem = os.path.splitext(os.path.basename(resolved_path))[0] if resolved_path else ""
    fallback = "-".join(
        part for part in (str(provider or "").strip(), config_stem.strip(), str(role or "").strip()) if part
    )
    slug = _slugify_results_component(fallback)
    return slug if slug != "unknown" else str(role or "model")


def get_active_provider_and_models(config_path: str) -> Tuple[str, str, str]:
    resolved_path = resolve_api_config_path(config_path)
    if not os.path.exists(resolved_path):
        return "", "", ""

    raw = _load_yaml_mapping(resolved_path)
    provider = str(raw.get("provider", "") or "")
    provider_config = raw.get(provider) or {}
    llm_model = str(provider_config.get("llm_model") or provider_config.get("model") or "")
    vlm_model = str(provider_config.get("vlm_model") or provider_config.get("model") or "")
    return provider, llm_model, vlm_model


def build_model_results_dir_name(config_path: str) -> str:
    provider, llm_model, vlm_model = get_active_provider_and_models(config_path)
    llm_slug = (
        _slugify_results_component(llm_model)
        if str(llm_model or "").strip()
        else _build_results_component_fallback(
            config_path,
            provider=provider,
            role="llm",
        )
    )
    vlm_slug = (
        _slugify_results_component(vlm_model)
        if str(vlm_model or "").strip()
        else _build_results_component_fallback(
            config_path,
            provider=provider,
            role="vlm",
        )
    )
    if llm_slug == vlm_slug:
        return llm_slug
    return f"{llm_slug}__{vlm_slug}"


def build_default_results_dir_from_api_config(
    config_path: str,
    repo_root: str = None,
    results_root: str = None,
    family: str = DEFAULT_RESULTS_FAMILY,
) -> str:
    return os.path.abspath(
        os.path.join(
            build_default_results_family_root(
                family,
                repo_root=repo_root,
                results_root=results_root,
            ),
            build_model_results_dir_name(config_path),
        )
    )


def build_default_context_cache_results_dir(
    config_path: str,
    repo_root: str = None,
    results_root: str = None,
    family: str = DEFAULT_RESULTS_FAMILY,
) -> str:
    base_dir = build_default_results_dir_from_api_config(
        config_path,
        repo_root=repo_root,
        results_root=results_root,
        family=family,
    )
    suffix = "_cache"
    resolved_path = resolve_api_config_path(config_path)
    if resolved_path and os.path.exists(resolved_path):
        raw = _load_yaml_mapping(resolved_path)
        cache_block = dict(raw.get("qwen_context_cache") or {})
        suffix = str(cache_block.get("results_dir_suffix") or "_cache").strip() or "_cache"
    return f"{base_dir}{suffix}" if suffix else base_dir


__all__ = [
    "DEFAULT_CONTEXT_CACHE_API_CONFIG",
    "DEFAULT_RESULTS_FAMILY",
    "DEFAULT_RESULTS_ROOT_RELATIVE",
    "DEFAULT_SYSTEM_RUNTIME_CONFIG_RELATIVE",
    "build_default_context_cache_results_dir",
    "build_default_results_dir_from_api_config",
    "build_default_results_family_root",
    "build_default_results_root",
    "build_model_results_dir_name",
    "get_active_provider_and_models",
    "resolve_api_config_path",
    "resolve_results_dir_path",
    "resolve_results_root_path",
]
