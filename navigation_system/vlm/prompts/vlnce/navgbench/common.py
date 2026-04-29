"""Prompt-template helpers for NavGBench-specific VLNCE prompt overlays."""

from functools import lru_cache
from pathlib import Path

from navigation_system.vlm.prompts.common import join_prompt_blocks


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@lru_cache(maxsize=None)
def load_navgbench_prompt_template(template_name: str) -> str:
    template_path = _TEMPLATE_DIR / str(template_name or "").strip()
    if not template_path.is_file():
        raise FileNotFoundError(f"NavGBench prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def normalize_navgbench_prompt_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower().replace("-", "_")
    if normalized in {"complex", "full", "grounded", "grounded_route"}:
        return "complex"
    if normalized in {"simple", "short", "raw", "goal"}:
        return "simple"
    if normalized in {"moving", "metric", "turn_by_turn"}:
        return "moving"
    return "complex"


def is_complex_navgbench_prompt_mode(mode: str) -> bool:
    return normalize_navgbench_prompt_mode(mode) == "complex"


def complex_instruction_policy_block() -> str:
    return load_navgbench_prompt_template("complex_instruction_policy.prompt.md").strip()


def inject_complex_instruction_policy(prompt: str, *, instruction_mode: str) -> str:
    text = str(prompt or "")
    if not is_complex_navgbench_prompt_mode(instruction_mode):
        return text

    policy = complex_instruction_policy_block()
    marker = "\n**Global Task**:"
    if marker in text:
        return text.replace(marker, f"\n\n{policy}{marker}", 1)
    return join_prompt_blocks([text, policy])


__all__ = [
    "complex_instruction_policy_block",
    "inject_complex_instruction_policy",
    "is_complex_navgbench_prompt_mode",
    "load_navgbench_prompt_template",
    "normalize_navgbench_prompt_mode",
]
