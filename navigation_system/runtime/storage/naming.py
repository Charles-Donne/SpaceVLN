"""Shared artifact naming helpers for user-facing navigation outputs."""

import re
from typing import Any, Optional


_ACTION_PHASE_RE = re.compile(r"^action[_-]?(\d+)[a-z]*$", re.IGNORECASE)
_VERIFY_PHASE_RE = re.compile(r"^verify[_-]?(\d+)[a-z]*$", re.IGNORECASE)
_SUBTASK_RE = re.compile(r"^subtask[_-]?(\d+)$", re.IGNORECASE)
_DIGIT_RE = re.compile(r"(\d+)")


def _normalize_fallback_label(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip()).strip("_").lower()
    return text or "subtask"


def build_subtask_name(subtask_index: int) -> str:
    return f"subtask{max(1, int(subtask_index))}"


def extract_subtask_index_from_phase(phase: Any) -> Optional[int]:
    text = str(phase or "").strip().lower()
    if not text:
        return None
    if text == "initial":
        return 1

    match = _SUBTASK_RE.match(text)
    if match:
        return max(1, int(match.group(1)))

    match = _ACTION_PHASE_RE.match(text)
    if match:
        return max(1, int(match.group(1)))

    match = _VERIFY_PHASE_RE.match(text)
    if match:
        return max(1, int(match.group(1)) + 1)

    return None


def extract_subtask_index_from_token(token: Any) -> Optional[int]:
    phase_index = extract_subtask_index_from_phase(token)
    if phase_index is not None:
        return phase_index

    text = str(token or "").strip().lower()
    if not text:
        return None

    match = _DIGIT_RE.search(text)
    if match:
        return max(1, int(match.group(1)))
    return None


def build_subtask_name_from_token(token: Any, fallback: Any = None) -> str:
    subtask_index = extract_subtask_index_from_token(token)
    if subtask_index is not None:
        return build_subtask_name(subtask_index)
    return _normalize_fallback_label(fallback if fallback is not None else token)


def build_step_artifact_filename(step: int, token: Any, suffix: str = ".png") -> str:
    return f"step_{int(step):04d}_{build_subtask_name_from_token(token)}{suffix}"

