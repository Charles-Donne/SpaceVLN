"""Canonical ablation preset registry and path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class AblationPresetDefinition:
    key: str
    slug: str
    config_filename: str
    description: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def config_path(self) -> Path:
        return get_ablation_config_root() / self.config_filename


def get_ablation_root() -> Path:
    return Path(__file__).resolve().parent


def get_ablation_config_root() -> Path:
    return get_ablation_root() / "configs"


def get_ablation_template_root() -> Path:
    return get_ablation_root() / "templates"


_PRESET_DEFINITIONS: tuple[AblationPresetDefinition, ...] = (
    AblationPresetDefinition(
        key="default",
        slug="default",
        config_filename="default.yaml",
        description="Run the isolated ablation runtime without removing any cues.",
        aliases=("full", "all"),
    ),
    AblationPresetDefinition(
        key="landmark",
        slug="no-landmark",
        config_filename="no_landmark.yaml",
        description="Remove landmark perception inputs only.",
        aliases=("no_landmark", "without_landmark"),
    ),
    AblationPresetDefinition(
        key="space_structure",
        slug="no-space-structure",
        config_filename="no_space_structure.yaml",
        description="Remove space-structure prompt and render inputs only.",
        aliases=("space", "no_space_structure", "without_space_structure"),
    ),
    AblationPresetDefinition(
        key="planning_reasoning",
        slug="no-planning-reasoning",
        config_filename="no_planning_reasoning.yaml",
        description="Remove explicit planning reasoning prompt sections only.",
        aliases=("planning", "no_planning_reasoning", "without_planning_reasoning"),
    ),
    AblationPresetDefinition(
        key="action_reasoning",
        slug="no-action-reasoning",
        config_filename="no_action_reasoning.yaml",
        description="Remove explicit action reasoning prompt sections only.",
        aliases=("action", "no_action_reasoning", "without_action_reasoning"),
    ),
    AblationPresetDefinition(
        key="planning_action_reasoning",
        slug="no-planning-action-reasoning",
        config_filename="no_planning_action_reasoning.yaml",
        description="Remove explicit planning and action reasoning prompt sections.",
        aliases=(
            "planning_action",
            "no_planning_action_reasoning",
            "without_planning_action_reasoning",
            "reasoning",
            "no_reasoning",
            "without_reasoning",
        ),
    ),
    AblationPresetDefinition(
        key="both",
        slug="no-landmark-no-space-structure",
        config_filename="no_landmark_no_space_structure.yaml",
        description="Remove both landmark and space-structure inputs.",
        aliases=("none", "no_landmark_no_space_structure", "without_both"),
    ),
)

_PRESETS_BY_KEY: Dict[str, AblationPresetDefinition] = {
    preset.key: preset for preset in _PRESET_DEFINITIONS
}
_PRESET_ALIASES: Dict[str, str] = {}
for _preset in _PRESET_DEFINITIONS:
    _PRESET_ALIASES[_preset.key] = _preset.key
    _PRESET_ALIASES[_preset.slug] = _preset.key
    for _alias in _preset.aliases:
        _PRESET_ALIASES[_alias] = _preset.key

_TEMPLATE_SLUG_ALIASES: Dict[str, str] = {
    "no-reasoning": "no-planning-action-reasoning",
}


def iter_ablation_presets() -> Iterable[AblationPresetDefinition]:
    return iter(_PRESET_DEFINITIONS)


def resolve_ablation_preset_key(name: Optional[str]) -> Optional[str]:
    normalized = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    return _PRESET_ALIASES.get(normalized)


def get_ablation_preset(name: Optional[str]) -> Optional[AblationPresetDefinition]:
    key = resolve_ablation_preset_key(name)
    if not key:
        return None
    return _PRESETS_BY_KEY.get(key)


def resolve_ablation_template_slug(slug: str) -> str:
    normalized = str(slug or "").strip()
    return _TEMPLATE_SLUG_ALIASES.get(normalized, normalized)


def detect_ablation_preset_key_from_slug(slug: Optional[str]) -> Optional[str]:
    normalized = str(slug or "").strip().lower()
    if not normalized:
        return None
    if normalized in _PRESETS_BY_KEY:
        return normalized
    for preset in _PRESET_DEFINITIONS:
        if preset.slug == normalized:
            return preset.key
    normalized = normalized.replace("-", "_")
    return _PRESET_ALIASES.get(normalized)


__all__ = [
    "AblationPresetDefinition",
    "detect_ablation_preset_key_from_slug",
    "get_ablation_config_root",
    "get_ablation_preset",
    "get_ablation_root",
    "get_ablation_template_root",
    "iter_ablation_presets",
    "resolve_ablation_preset_key",
    "resolve_ablation_template_slug",
]
