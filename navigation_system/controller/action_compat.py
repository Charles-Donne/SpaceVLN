"""Compatibility helpers for Habitat action identifiers across versions."""

from __future__ import annotations


def resolve_habitat_action(name: str):
    legacy_name = str(name or "").strip().upper()
    modern_name = legacy_name.lower()

    try:
        from habitat.sims.habitat_simulator.actions import HabitatSimActions

        try:
            return getattr(HabitatSimActions, legacy_name)
        except Exception:
            return getattr(HabitatSimActions, modern_name)
    except Exception:
        fallback_ids = {
            "STOP": 0,
            "MOVE_FORWARD": 1,
            "TURN_LEFT": 2,
            "TURN_RIGHT": 3,
        }
        return fallback_ids.get(legacy_name, modern_name)
