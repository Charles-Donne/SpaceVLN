"""Shared output policy for metric runs and visual/debug runs."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


OUTPUT_PROFILE_CHOICES = ("metric", "debug", "config")
DEFAULT_OUTPUT_PROFILE = "metric"
OUTPUT_PROFILE_ENV = "SPACEVLN_OUTPUT_PROFILE"


@dataclass(frozen=True)
class OutputPolicy:
    """Resolved runtime output switches."""

    profile: str
    save_step_images: bool
    save_gif: bool
    save_vlm_artifacts: bool
    save_map_artifacts: bool


_PROFILE_DEFAULTS: Dict[str, Dict[str, bool]] = {
    # Metric batches keep only final evaluation artifacts by default.
    "metric": {
        "save_step_images": False,
        "save_gif": False,
        "save_vlm_artifacts": False,
        "save_map_artifacts": False,
    },
    # Debug keeps model-facing prompts/images, map artifacts, and a replay GIF,
    # but still avoids replay-frame PNGs unless explicitly requested.
    "debug": {
        "save_step_images": False,
        "save_gif": True,
        "save_vlm_artifacts": True,
        "save_map_artifacts": True,
    },
}


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return None


def _default_profile() -> str:
    raw = str(os.getenv(OUTPUT_PROFILE_ENV, "") or "").strip().lower()
    if raw in OUTPUT_PROFILE_CHOICES:
        return raw
    return DEFAULT_OUTPUT_PROFILE


def add_output_profile_arg(parser: argparse.ArgumentParser) -> None:
    """Add the shared output profile selector to a runner parser."""
    parser.add_argument(
        "--output-profile",
        choices=OUTPUT_PROFILE_CHOICES,
        default=_default_profile(),
        help=(
            "Output policy: metric saves only final evaluation artifacts; "
            "debug saves VLM artifacts, map artifacts, and GIF; "
            "config preserves YAML defaults."
        ),
    )


def add_output_artifact_args(
    parser: argparse.ArgumentParser,
    *,
    include_vlm_artifacts: bool = True,
    include_gif_alias: bool = True,
) -> None:
    """Add common save/skip flags. CLI flags override the selected profile."""
    parser.set_defaults(
        save_step_images=None,
        save_gif=None,
        save_vlm_artifacts=None,
        save_map_artifacts=None,
    )
    parser.add_argument(
        "--save-step-images",
        dest="save_step_images",
        action="store_true",
        help="Save per-step replay PNGs under episode visualization directories.",
    )
    parser.add_argument(
        "--no-save-step-images",
        dest="save_step_images",
        action="store_false",
        help="Do not save per-step replay PNGs.",
    )
    save_gif_flags = ["--save-gif"]
    no_gif_flags = ["--no-save-gif"]
    if include_gif_alias:
        no_gif_flags.insert(0, "--no-gif")
    parser.add_argument(
        *save_gif_flags,
        dest="save_gif",
        action="store_true",
        help="Save final episode navigation.gif replay.",
    )
    parser.add_argument(
        *no_gif_flags,
        dest="save_gif",
        action="store_false",
        help="Skip final episode navigation.gif replay generation.",
    )
    if include_vlm_artifacts:
        parser.add_argument(
            "--save-vlm-artifacts",
            dest="save_vlm_artifacts",
            action="store_true",
            help="Save VLM prompts, request previews, model images, and responses.",
        )
        parser.add_argument(
            "--no-vlm-artifacts",
            dest="save_vlm_artifacts",
            action="store_false",
            help="Skip VLM prompt/image/debug artifacts for metric batch runs.",
        )
    parser.add_argument(
        "--save-map-artifacts",
        "--save-local-map",
        dest="save_map_artifacts",
        action="store_true",
        help="Save key global/local map PNG snapshots for thinking/detection debugging.",
    )
    parser.add_argument(
        "--no-save-map-artifacts",
        "--no-local-map",
        dest="save_map_artifacts",
        action="store_false",
        help="Do not save global/local map PNG snapshots.",
    )


def resolve_output_policy(
    args: argparse.Namespace,
    *,
    config_save_step_images: bool = False,
    config_save_gif: bool = False,
    config_save_vlm_artifacts: bool = True,
    config_save_map_artifacts: bool = False,
) -> OutputPolicy:
    profile = str(getattr(args, "output_profile", "") or _default_profile()).strip().lower()
    if profile not in OUTPUT_PROFILE_CHOICES:
        profile = DEFAULT_OUTPUT_PROFILE
    profile_defaults = _PROFILE_DEFAULTS.get(profile, {})

    def resolve(name: str, config_value: bool) -> bool:
        explicit = _bool_or_none(getattr(args, name, None))
        if explicit is not None:
            return explicit
        if name in profile_defaults:
            return bool(profile_defaults[name])
        return bool(config_value)

    return OutputPolicy(
        profile=profile,
        save_step_images=resolve("save_step_images", config_save_step_images),
        save_gif=resolve("save_gif", config_save_gif),
        save_vlm_artifacts=resolve("save_vlm_artifacts", config_save_vlm_artifacts),
        save_map_artifacts=resolve("save_map_artifacts", config_save_map_artifacts),
    )


def apply_output_policy_to_config(config: Any, args: argparse.Namespace) -> OutputPolicy:
    """Apply the shared output policy to a mutable SpaceVLN config object."""
    output_cfg = getattr(config, "OUTPUT", None)
    request_cfg = getattr(output_cfg, "REQUESTS", None)
    maps_cfg = getattr(output_cfg, "MAPS", None)
    replay_cfg = getattr(output_cfg, "REPLAY", None)
    policy = resolve_output_policy(
        args,
        config_save_step_images=bool(getattr(replay_cfg, "SAVE_STEP_IMAGES", False)),
        config_save_gif=bool(getattr(replay_cfg, "SAVE_GIF", False)),
        config_save_vlm_artifacts=bool(getattr(request_cfg, "SAVE_VLM_ARTIFACTS", True)),
        config_save_map_artifacts=bool(getattr(maps_cfg, "SAVE_STEP_ARTIFACTS", False)),
    )
    if request_cfg is not None:
        request_cfg.SAVE_VLM_ARTIFACTS = bool(policy.save_vlm_artifacts)
    if maps_cfg is not None:
        maps_cfg.SAVE_STEP_ARTIFACTS = bool(policy.save_map_artifacts)
    if replay_cfg is not None:
        replay_cfg.SAVE_STEP_IMAGES = bool(policy.save_step_images)
        replay_cfg.SAVE_GIF = bool(policy.save_gif)
    return policy


def build_output_job_fields(args: argparse.Namespace) -> Dict[str, Any]:
    """Serialize shared output args into a parallel worker job spec."""
    return {
        "output_profile": str(getattr(args, "output_profile", "") or _default_profile()),
        "save_step_images": getattr(args, "save_step_images", None),
        "save_gif": getattr(args, "save_gif", None),
        "save_vlm_artifacts": getattr(args, "save_vlm_artifacts", None),
        "save_map_artifacts": getattr(args, "save_map_artifacts", None),
        "no_report": bool(getattr(args, "no_report", False)),
    }


def build_output_namespace_kwargs(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Restore shared output args from a parallel worker job spec."""
    return {
        "output_profile": str(job_spec.get("output_profile") or _default_profile()),
        "save_step_images": job_spec.get("save_step_images"),
        "save_gif": job_spec.get("save_gif"),
        "save_vlm_artifacts": job_spec.get("save_vlm_artifacts"),
        "save_map_artifacts": job_spec.get("save_map_artifacts"),
        "no_report": bool(job_spec.get("no_report", False)),
    }
