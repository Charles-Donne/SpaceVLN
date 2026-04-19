"""Compatibility helpers for Habitat action identifiers across versions."""

from __future__ import annotations


def resolve_habitat_action(name: str):
    from habitat.sims.habitat_simulator.actions import HabitatSimActions

    legacy_name = str(name or "").strip().upper()
    modern_name = legacy_name.lower()

    try:
        return getattr(HabitatSimActions, legacy_name)
    except Exception:
        pass

    try:
        return getattr(HabitatSimActions, modern_name)
    except Exception:
        return modern_name

