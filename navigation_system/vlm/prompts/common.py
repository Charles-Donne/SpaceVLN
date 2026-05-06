"""Shared prompt-template loading and system/user prompt bundle helpers."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Union


_TEMPLATE_DIR = Path(__file__).resolve().parent / "vlnce" / "templates"


@lru_cache(maxsize=None)
def load_prompt_template(template_name: str) -> str:
    template_path = _TEMPLATE_DIR / str(template_name or "").strip()
    if not template_path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class PromptBundle:
    """Exact prompt split sent as separate system and user messages."""

    system_prompt: str
    user_prompt: str
    full_prompt: str


PromptLike = Union[str, PromptBundle]


def extract_prompt_debug_text(prompt: PromptLike) -> str:
    """Return the full prompt text that should be saved for debugging."""
    if isinstance(prompt, PromptBundle):
        return str(prompt.full_prompt or "")
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8")
    return str(prompt or "")


def join_prompt_blocks(blocks) -> str:
    """Join non-empty prompt blocks while preserving original wording."""
    normalized = [str(block).strip("\n") for block in list(blocks or []) if str(block or "").strip()]
    return "\n\n".join(normalized).strip()


def compose_full_prompt(system_prompt: str, user_prompt: str) -> str:
    """Combine the system/user text for legacy debug summaries."""
    return join_prompt_blocks([system_prompt, user_prompt])


__all__ = [
    "PromptBundle",
    "PromptLike",
    "compose_full_prompt",
    "extract_prompt_debug_text",
    "join_prompt_blocks",
    "load_prompt_template",
]
