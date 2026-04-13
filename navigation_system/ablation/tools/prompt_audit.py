"""Static audit for ablation template completeness and forbidden placeholders."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_TEMPLATE_ROOT = ROOT.parent / "vlm" / "prompts" / "templates"
ABLATION_TEMPLATE_ROOT = ROOT / "templates"

REQUIRED_ORIGINAL_FILES = sorted(
    path.relative_to(ORIGINAL_TEMPLATE_ROOT)
    for path in ORIGINAL_TEMPLATE_ROOT.rglob("*")
    if path.is_file()
)

FORBIDDEN_PLACEHOLDERS = {
    "no-space-structure": {"{waypoint_summary}"},
    "no-landmark": {
        "{detected_landmarks}",
        "{landmark_perception_summary}",
        "{previous_subtask_landmark_block}",
        "{previous_subtask_landmark_summary}",
    },
    "no-landmark-no-space-structure": {
        "{detected_landmarks}",
        "{landmark_perception_summary}",
        "{previous_subtask_landmark_block}",
        "{previous_subtask_landmark_summary}",
        "{waypoint_summary}",
    },
}


def _iter_variant_dirs():
    for path in sorted(ABLATION_TEMPLATE_ROOT.iterdir()):
        if path.is_dir():
            yield path


def _check_variant(variant_dir: Path) -> list[str]:
    errors: list[str] = []
    existing_files = {
        path.relative_to(variant_dir)
        for path in variant_dir.rglob("*")
        if path.is_file()
    }
    for rel_path in REQUIRED_ORIGINAL_FILES:
        if rel_path not in existing_files:
            errors.append(f"missing template: {variant_dir.name}/{rel_path}")

    forbidden = FORBIDDEN_PLACEHOLDERS.get(variant_dir.name, set())
    if not forbidden:
        return errors

    for path in sorted(variant_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for token in sorted(forbidden):
            if token in text:
                errors.append(f"forbidden placeholder {token} in {path.relative_to(ROOT)}")
    return errors


def main() -> int:
    errors: list[str] = []
    for variant_dir in _iter_variant_dirs():
        errors.extend(_check_variant(variant_dir))

    if errors:
        print("Ablation prompt audit failed:")
        for item in errors:
            print(f"- {item}")
        return 1

    print("Ablation prompt audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
