"""OVON benchmark plugin for the object-navigation task."""

__all__ = ["OVONObjectNavigationController"]


def __getattr__(name: str):
    if name == "OVONObjectNavigationController":
        from navigation_system.runtime.object_navigation.ovon.controller import (
            OVONObjectNavigationController,
        )

        return OVONObjectNavigationController
    raise AttributeError(name)
