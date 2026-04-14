"""Config loading for the real-robot runtime."""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml

from spacevln_real.models import RealRobotConfig


def load_real_robot_config(config_path: str) -> RealRobotConfig:
    resolved = os.path.abspath(str(config_path or "").strip())
    if not resolved:
        raise ValueError("real robot config path is empty")
    if not os.path.exists(resolved):
        raise FileNotFoundError("real robot config not found: %s" % resolved)
    with open(resolved, "r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = yaml.safe_load(handle) or {}
    return RealRobotConfig.from_dict(payload)

