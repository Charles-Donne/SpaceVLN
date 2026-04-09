"""Shared prompt-template loading and explicit-cache bundle helpers."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Union


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@lru_cache(maxsize=None)
def load_prompt_template(template_name: str) -> str:
    template_path = _TEMPLATE_DIR / str(template_name or "").strip()
    if not template_path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class ExplicitCachePromptBundle:
    """Exact prompt split for runtimes that cache a stable system prompt."""

    system_prompt: str
    user_prompt: str
    full_prompt: str


PromptLike = Union[str, ExplicitCachePromptBundle]


def extract_prompt_debug_text(prompt: PromptLike) -> str:
    """Return the full prompt text that should be saved for debugging."""
    if isinstance(prompt, ExplicitCachePromptBundle):
        return str(prompt.full_prompt or "")
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8")
    return str(prompt or "")


def join_prompt_blocks(blocks) -> str:
    """Join non-empty prompt blocks while preserving original wording."""
    normalized = [str(block).strip("\n") for block in list(blocks or []) if str(block or "").strip()]
    return "\n\n".join(normalized).strip()


def compose_full_prompt(system_prompt: str, user_prompt: str) -> str:
    """Combine the cache system/user text for prompt artifact debugging."""
    return join_prompt_blocks([system_prompt, user_prompt])


__all__ = [
    "ExplicitCachePromptBundle",
    "PromptLike",
    "compose_full_prompt",
    "extract_prompt_debug_text",
    "join_prompt_blocks",
    "load_prompt_template",
]
