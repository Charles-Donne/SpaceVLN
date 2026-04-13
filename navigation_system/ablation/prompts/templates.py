"""Static template loader for ablation-specific prompt copies."""

from __future__ import annotations

from navigation_system.ablation.config import AblationSpec
from navigation_system.ablation.presets import (
    get_ablation_template_root,
    resolve_ablation_template_slug,
)
from navigation_system.vlm.prompts.common import load_prompt_template


_ABLATION_TEMPLATE_ROOT = get_ablation_template_root()


def load_ablation_template(spec: AblationSpec, template_name: str) -> str:
    slug = str(getattr(spec, "slug", "") or "").strip()
    rel_path = str(template_name or "").strip()
    if slug and slug != "default":
        candidate_slug = resolve_ablation_template_slug(slug)
        candidate = _ABLATION_TEMPLATE_ROOT / candidate_slug / rel_path
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return load_prompt_template(rel_path)


__all__ = [
    "load_ablation_template",
]
