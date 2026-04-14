"""Filesystem layout helpers for isolated ablation runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from navigation_system.runtime.storage.results_layout import (
    DEFAULT_RESULTS_FAMILY,
    build_default_context_cache_results_dir,
    build_default_results_dir_from_api_config,
    build_default_results_family_root,
    build_default_results_root,
    build_model_results_dir_name,
    get_active_provider_and_models,
    resolve_results_root_path,
)


def build_ablation_results_dir_from_model_dir(
    model_results_dir: str,
    *,
    ablation_slug: str,
    repo_root: Optional[str] = None,
    results_root: Optional[str] = None,
    family: str = DEFAULT_RESULTS_FAMILY,
) -> str:
    family_root = build_default_results_family_root(
        family,
        repo_root=repo_root,
        results_root=results_root,
    )
    model_dir_name = os.path.basename(os.path.abspath(str(model_results_dir or "").strip()))
    return os.path.abspath(
        os.path.join(
            family_root,
            "ablation",
            str(ablation_slug or "default").strip() or "default",
            model_dir_name,
        )
    )


def build_ablation_manifest_dir(results_dir: str) -> Path:
    return Path(str(results_dir or "").strip()) / "ablation"


__all__ = [
    "DEFAULT_RESULTS_FAMILY",
    "build_default_context_cache_results_dir",
    "build_default_results_dir_from_api_config",
    "build_default_results_family_root",
    "build_default_results_root",
    "build_ablation_manifest_dir",
    "build_ablation_results_dir_from_model_dir",
    "build_model_results_dir_name",
    "get_active_provider_and_models",
    "resolve_results_root_path",
]
