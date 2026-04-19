"""OVON object-navigation system built alongside, not inside, the R2R stack."""

from navigation_system.object_navigation.controller import OVONObjectNavigationController
from navigation_system.object_navigation.goal_task import build_raw_object_goal_instruction
from navigation_system.object_navigation.runner import main

__all__ = [
    "build_raw_object_goal_instruction",
    "OVONObjectNavigationController",
    "main",
]
