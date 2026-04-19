"""Lazy import facade for Habitat extension modules.

This keeps package import side effects minimal so non-VLN runtimes can reuse
utility modules such as `pose_utils` without immediately importing the legacy
VLN dataset/task stack.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "Simulator",
    "VLNCEDatasetV1",
    "measures",
    "sensors",
]


def __getattr__(name: str) -> Any:
    if name == "VLNCEDatasetV1":
        return getattr(import_module("habitat_extensions.task"), name)
    if name == "Simulator":
        return getattr(import_module("habitat_extensions.habitat_simulator"), name)
    if name in {"measures", "sensors"}:
        return import_module(f"habitat_extensions.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
