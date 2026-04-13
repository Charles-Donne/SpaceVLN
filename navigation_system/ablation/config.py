"""Config loading for isolated ablation experiments."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


ABLATION_CONFIG_ENV = "SPACEVLN_ABLATION_CONFIG"
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else str(default or "")


def _slugify(text: str, default: str = "default") -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", str(text or "").strip().lower()).strip("-")
    return normalized or default


@dataclass(frozen=True)
class ThinkingPromptAblation:
    include_status_block: bool = True
    include_detected_landmarks: bool = True
    include_waypoint_summary: bool = True
    include_previous_subtask_landmark_summary: bool = True
    include_verify_notice: bool = True

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "ThinkingPromptAblation":
        payload = dict(payload or {})
        return cls(
            include_status_block=_safe_bool(payload.get("include_status_block"), True),
            include_detected_landmarks=_safe_bool(payload.get("include_detected_landmarks"), True),
            include_waypoint_summary=_safe_bool(payload.get("include_waypoint_summary"), True),
            include_previous_subtask_landmark_summary=_safe_bool(
                payload.get("include_previous_subtask_landmark_summary"),
                True,
            ),
            include_verify_notice=_safe_bool(payload.get("include_verify_notice"), True),
        )


@dataclass(frozen=True)
class ActionPromptAblation:
    include_status_block: bool = True
    include_progress_summary: bool = True
    include_previous_action_reason: bool = True
    include_detected_landmarks: bool = True
    include_obstacle_summary: bool = True
    include_landmark_map_info: bool = True

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "ActionPromptAblation":
        payload = dict(payload or {})
        return cls(
            include_status_block=_safe_bool(payload.get("include_status_block"), True),
            include_progress_summary=_safe_bool(payload.get("include_progress_summary"), True),
            include_previous_action_reason=_safe_bool(payload.get("include_previous_action_reason"), True),
            include_detected_landmarks=_safe_bool(payload.get("include_detected_landmarks"), True),
            include_obstacle_summary=_safe_bool(payload.get("include_obstacle_summary"), True),
            include_landmark_map_info=_safe_bool(payload.get("include_landmark_map_info"), True),
        )


@dataclass(frozen=True)
class ThinkingImageAblation:
    include_detection_boxes: bool = True
    include_obstacle_text: bool = True
    include_landmark_strip: bool = True
    include_space_waypoint_strip: bool = True
    include_last_visited_marker: bool = True
    include_global_map_space_structure: bool = True

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "ThinkingImageAblation":
        payload = dict(payload or {})
        return cls(
            include_detection_boxes=_safe_bool(payload.get("include_detection_boxes"), True),
            include_obstacle_text=_safe_bool(payload.get("include_obstacle_text"), True),
            include_landmark_strip=_safe_bool(payload.get("include_landmark_strip"), True),
            include_space_waypoint_strip=_safe_bool(payload.get("include_space_waypoint_strip"), True),
            include_last_visited_marker=_safe_bool(payload.get("include_last_visited_marker"), True),
            include_global_map_space_structure=_safe_bool(
                payload.get("include_global_map_space_structure"),
                True,
            ),
        )


@dataclass(frozen=True)
class ActionImageAblation:
    use_detection_overlay: bool = True

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "ActionImageAblation":
        payload = dict(payload or {})
        return cls(
            use_detection_overlay=_safe_bool(payload.get("use_detection_overlay"), True),
        )


@dataclass(frozen=True)
class AblationSpec:
    name: str = "default"
    description: str = ""
    thinking_prompt: ThinkingPromptAblation = field(default_factory=ThinkingPromptAblation)
    action_prompt: ActionPromptAblation = field(default_factory=ActionPromptAblation)
    thinking_image: ThinkingImageAblation = field(default_factory=ThinkingImageAblation)
    action_image: ActionImageAblation = field(default_factory=ActionImageAblation)

    @property
    def slug(self) -> str:
        return _slugify(self.name or "default")

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "AblationSpec":
        payload = dict(payload or {})
        return cls(
            name=_safe_str(payload.get("name"), "default"),
            description=_safe_str(payload.get("description"), ""),
            thinking_prompt=ThinkingPromptAblation.from_dict(payload.get("thinking_prompt")),
            action_prompt=ActionPromptAblation.from_dict(payload.get("action_prompt")),
            thinking_image=ThinkingImageAblation.from_dict(payload.get("thinking_image")),
            action_image=ActionImageAblation.from_dict(payload.get("action_image")),
        )


def get_default_ablation_config_path() -> str:
    return str(_DEFAULT_CONFIG_PATH)


def resolve_ablation_config_path(config_path: Optional[str] = None) -> str:
    candidate = _safe_str(config_path or os.environ.get(ABLATION_CONFIG_ENV), get_default_ablation_config_path())
    if not os.path.isabs(candidate):
        candidate = os.path.abspath(candidate)
    return candidate


@lru_cache(maxsize=None)
def _load_ablation_spec_from_path(resolved_path: str) -> AblationSpec:
    if not resolved_path or not os.path.exists(resolved_path):
        return AblationSpec()
    with open(resolved_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Ablation config must be a mapping: {resolved_path}")
    return AblationSpec.from_dict(raw)


def load_ablation_spec(config_path: Optional[str] = None) -> AblationSpec:
    return _load_ablation_spec_from_path(resolve_ablation_config_path(config_path))


def save_ablation_manifest(
    results_dir: str,
    *,
    config_path: Optional[str] = None,
    spec: Optional[AblationSpec] = None,
) -> Optional[str]:
    target_dir = str(results_dir or "").strip()
    if not target_dir:
        return None

    resolved_config_path = resolve_ablation_config_path(config_path)
    resolved_spec = spec or load_ablation_spec(resolved_config_path)
    manifest_dir = Path(target_dir) / "ablation"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / "manifest.json"
    manifest_payload = {
        "config_path": resolved_config_path,
        "spec": asdict(resolved_spec),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if resolved_config_path and os.path.exists(resolved_config_path):
        copied_config_path = manifest_dir / "config.yaml"
        if os.path.abspath(str(copied_config_path)) != os.path.abspath(resolved_config_path):
            shutil.copy2(resolved_config_path, copied_config_path)
    return str(manifest_path)


__all__ = [
    "ABLATION_CONFIG_ENV",
    "AblationSpec",
    "ActionImageAblation",
    "ActionPromptAblation",
    "ThinkingImageAblation",
    "ThinkingPromptAblation",
    "get_default_ablation_config_path",
    "load_ablation_spec",
    "resolve_ablation_config_path",
    "save_ablation_manifest",
]
