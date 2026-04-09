"""Render sample explicit-cache prompts for manual inspection.

This utility intentionally avoids importing the top-level ``navigation_system``
package, so prompt preview stays usable even in lightweight environments that
do not have Habitat installed.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path


OBS_BLOCKED_M = 0.5
OBS_RISKY_M = 1.0
OBS_OPEN_M = 2.0
ARRIVAL_NEAR_M = 1.5
ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M = 0.5
ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M = 0.75

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates" / "cache"


@dataclass(frozen=True)
class CachePromptPreviewBundle:
    system_prompt: str
    user_prompt: str
    full_prompt: str


def _fmt_threshold_m(value: float) -> str:
    text = f"{float(value):.2f}"
    if text.endswith("00"):
        return f"{float(value):.1f}"
    return text.rstrip("0").rstrip(".")


def _load_template(template_name: str) -> str:
    return (_TEMPLATE_DIR / template_name).read_text(encoding="utf-8")


def _compose_full_prompt(system_prompt: str, user_prompt: str) -> str:
    parts = [str(part).strip() for part in (system_prompt, user_prompt) if str(part).strip()]
    return "\n\n".join(parts)


def _render_initial_bundle():
    system_prompt = _load_template("planning_initial.system.prompt.md").format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(ARRIVAL_NEAR_M),
    )
    user_prompt = _load_template("planning_initial.user.prompt.md").format(
        instruction="Walk out of the bedroom, turn left into the hallway, then stop by the rug in the living room.",
    )
    return CachePromptPreviewBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=_compose_full_prompt(system_prompt, user_prompt),
    )


def _render_verify_bundle():
    system_prompt = _load_template("planning_verify.system.prompt.md").format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(ARRIVAL_NEAR_M),
    )
    user_prompt = _load_template("planning_verify.user.prompt.md").format(
        verify_replan_prompt_notice_block=(
            "**Stuck Notice**: Front route was blocked in the previous attempt; "
            "verify whether a left-side bypass is now correct."
        ),
        instruction="Walk out of the bedroom, turn left into the hallway, then stop by the rug in the living room.",
        subtask_destination="Hallway's left turn opening",
        subtask_instruction="From IMAGE 5 (Left 120deg) view, start, move toward the hallway's left turn opening.",
        previous_subtask_landmark_block="- Landmark: [hallway opening], 1.8m, Left 120deg",
        waypoint_summary=(
            "Your Current Area: Bedroom doorway side\n"
            "Space WP1: Bedroom - doorway / bed side (INITIAL POSITION)\n"
            "Space WP2: Hallway - left turn opening\n"
            "Space Waypoint Chain: Bedroom's doorway(Current) -> Hallway's left turn opening"
        ),
    )
    return CachePromptPreviewBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=_compose_full_prompt(system_prompt, user_prompt),
    )


def _render_action_bundle():
    system_prompt = _load_template("action.system.prompt.md").format(
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        solid_autocomplete_m=_fmt_threshold_m(
            ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M
        ),
        open_autocomplete_m=_fmt_threshold_m(
            ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M
        ),
    )
    user_prompt = _load_template("action.user.prompt.md").format(
        subtask_destination="Hallway's left turn opening",
        subtask_instruction="From IMAGE 5 (Left 120deg) view, start, move toward the hallway's left turn opening.",
        progress_summary="Exit the bedroom and approach the hallway opening(Current)",
        previous_action_reason=(
            "Previous forward move was blocked by a near front obstacle, "
            "so a left-side reorientation may be needed."
        ),
        obstacle_perception_summary="FRONT 0.42m warning | Left 30deg 1.85m | Right 30deg 0.63m",
        landmark_perception_summary=(
            "- hallway opening | 1.8m | Left 30deg | task-aligned next connector\n"
            "- door frame | 0.7m | FRONT | near doorway edge, not the destination"
        ),
        waypoint_summary=(
            "Your Current Area: Bedroom doorway side\n"
            "Current waypoint: Bedroom - doorway / bed side\n"
            "Next waypoint: Hallway - left turn opening\n"
            "Behind waypoint: Bedroom - initial position"
        ),
        detected_landmarks="hallway opening, door frame",
        allowed_action_bullets=(
            "- `MOVE_FORWARD {0.25m, 0.5m, 0.75m, 1.0m, 1.25m}`\n"
            "- `TURN_LEFT 30deg` | `TURN_RIGHT 30deg`\n"
            "- `STOP`"
        ),
    )
    return CachePromptPreviewBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=_compose_full_prompt(system_prompt, user_prompt),
    )


def _write_bundle(prefix: str, bundle, output_dir: Path = None) -> None:
    payload = {
        f"{prefix}.system.md": bundle.system_prompt,
        f"{prefix}.user.md": bundle.user_prompt,
        f"{prefix}.full.md": bundle.full_prompt,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, text in payload.items():
            (output_dir / filename).write_text(text, encoding="utf-8")
        return

    for filename, text in payload.items():
        print(f"===== {filename} =====")
        print(text)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview sample explicit-cache prompts.")
    parser.add_argument(
        "--mode",
        choices=("initial", "verify", "action", "all"),
        default="all",
        help="Which cache prompt example to render.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save rendered markdown files instead of printing.",
    )
    args = parser.parse_args()

    bundles = []
    if args.mode in {"initial", "all"}:
        bundles.append(("planning_initial", _render_initial_bundle()))
    if args.mode in {"verify", "all"}:
        bundles.append(("planning_verify", _render_verify_bundle()))
    if args.mode in {"action", "all"}:
        bundles.append(("action", _render_action_bundle()))

    for prefix, bundle in bundles:
        _write_bundle(prefix, bundle, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
