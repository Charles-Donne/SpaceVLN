"""Prompt-template loading helpers for OVON-specific prompt stacks."""

from functools import lru_cache
from pathlib import Path


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@lru_cache(maxsize=None)
def load_objectnav_prompt_template(template_name: str) -> str:
    template_path = _TEMPLATE_DIR / str(template_name or "").strip()
    if not template_path.is_file():
        raise FileNotFoundError(f"Object-nav prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")
