"""Unified Python entrypoint for the Navigation Agent benchmarks."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import types
from typing import Callable, Sequence


RUNTIME_CHOICES = ("standard", "context_cache")


def _suppress_gym_notice() -> None:
    gym_notices = types.ModuleType("gym_notices.notices")
    gym_notices.notices = {}
    sys.modules.setdefault("gym_notices.notices", gym_notices)


def _run_r2r(argv: Sequence[str]) -> int:
    from navigation_system.ablation.presets import get_ablation_preset
    from navigation_system.ablation.runtime.profiles import (
        ABLATION_CONTEXT_CACHE_RUNTIME_PROFILE,
        ABLATION_STANDARD_RUNTIME_PROFILE,
    )
    from navigation_system.ablation.runtime.runner import (
        run_navigation_from_args as run_ablation_navigation_from_args,
    )
    from navigation_system.runtime.vlnce.profiles import (
        CONTEXT_CACHE_RUNTIME_PROFILE,
        STANDARD_RUNTIME_PROFILE,
    )
    from navigation_system.runtime.vlnce.r2r.runner import (
        build_arg_parser,
        run_navigation_from_args,
    )

    def _augment_parser(parser) -> None:
        parser.add_argument(
            "--runtime",
            type=str,
            choices=RUNTIME_CHOICES,
            default="standard",
            help="Runtime profile: standard or context_cache",
        )
        parser.add_argument(
            "--ablation",
            "--preset",
            "--ablation-preset",
            type=str,
            dest="ablation_preset",
            default=None,
            help="Ablation preset name or ablation YAML path",
        )
        parser.add_argument(
            "--ablation-config",
            type=str,
            default=None,
            help="Explicit ablation experiment config path",
        )

    def _resolve_ablation_config(raw_value: str | None) -> str | None:
        text = str(raw_value or "").strip()
        if not text:
            return None

        preset = get_ablation_preset(text)
        candidate = str(preset.config_path) if preset is not None else os.path.abspath(text)
        if not os.path.isabs(candidate):
            candidate = os.path.abspath(candidate)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Ablation config does not exist: {text}")
        return candidate

    def _resolve_runtime_profile(runtime_name: str):
        normalized = str(runtime_name or "standard").strip().lower()
        if normalized == "context_cache":
            return CONTEXT_CACHE_RUNTIME_PROFILE
        return STANDARD_RUNTIME_PROFILE

    def _resolve_ablation_runtime_profile(runtime_name: str):
        normalized = str(runtime_name or "standard").strip().lower()
        if normalized == "context_cache":
            return ABLATION_CONTEXT_CACHE_RUNTIME_PROFILE
        return ABLATION_STANDARD_RUNTIME_PROFILE

    argv = list(argv)
    parser = build_arg_parser(profile=STANDARD_RUNTIME_PROFILE)
    _augment_parser(parser)
    args = parser.parse_args(argv)

    api_config_explicit = any(
        arg == "--vlm-api-config"
        or arg.startswith("--vlm-api-config=")
        or arg == "--config"
        or arg.startswith("--config=")
        for arg in argv
    )
    if not api_config_explicit and getattr(args, "runtime", "standard") == "context_cache":
        args.vlm_api_config = CONTEXT_CACHE_RUNTIME_PROFILE.default_api_config_path

    ablation_config = _resolve_ablation_config(
        getattr(args, "ablation_config", None) or getattr(args, "ablation_preset", None)
    )
    if ablation_config:
        args.ablation_config = ablation_config
        profile = _resolve_ablation_runtime_profile(getattr(args, "runtime", "standard"))
        return run_ablation_navigation_from_args(args, profile=profile)

    profile = _resolve_runtime_profile(getattr(args, "runtime", "standard"))
    return run_navigation_from_args(args, profile=profile)


def _run_navgbench(argv: Sequence[str]) -> int:
    with contextlib.redirect_stderr(io.StringIO()):
        _suppress_gym_notice()
        from navigation_system.runtime.vlnce.navgbench.runner import main

    previous_argv = sys.argv
    sys.argv = [previous_argv[0], *argv]
    try:
        return int(main() or 0)
    finally:
        sys.argv = previous_argv


def _run_ovon(argv: Sequence[str]) -> int:
    with contextlib.redirect_stderr(io.StringIO()):
        _suppress_gym_notice()
        from navigation_system.runtime.object_navigation.ovon.runner import main

    previous_argv = sys.argv
    sys.argv = [previous_argv[0], *argv]
    try:
        return int(main() or 0)
    finally:
        sys.argv = previous_argv


COMMANDS: dict[str, Callable[[Sequence[str]], int]] = {
    "r2r": _run_r2r,
    "r2rce": _run_r2r,
    "vlnce": _run_r2r,
    "navgbench": _run_navgbench,
    "ovon": _run_ovon,
    "object_navigation": _run_ovon,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = "r2r"
    if args and args[0].strip().lower() in COMMANDS:
        command = args.pop(0).strip().lower()
    try:
        return COMMANDS[command](args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    from navigation_system.runtime.process_lifecycle import exit_process

    exit_process(main())
