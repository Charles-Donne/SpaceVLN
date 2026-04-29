"""Shared Navigation Agent environment adapter contract and action helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Protocol, Sequence, Tuple

import numpy as np


NAVIGATION_ACTION_ID_TO_NAME = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}

NAVIGATION_ACTION_NAME_TO_ID = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "FORWARD": 1,
    "TURN_LEFT": 2,
    "LEFT": 2,
    "TURN_RIGHT": 3,
    "RIGHT": 3,
}


class NavigationVectorEnv(Protocol):
    """Minimal VectorEnv surface consumed by the shared navigation controller."""

    number_of_episodes: int

    def reset(self) -> List[Any]:
        ...

    def step(self, actions: Iterable[Any]) -> List[Tuple[Any, float, bool, Dict[str, Any]]]:
        ...

    def current_episodes(self) -> List[Any]:
        ...

    def call_at(self, index: int, method_name: str, *args: Any, **kwargs: Any) -> Any:
        ...

    def close(self) -> None:
        ...


def check_single_env_index(index: int) -> None:
    if int(index) != 0:
        raise IndexError(f"Single-env adapter only supports env index 0, got {index}")


def extract_raw_action(action: Any) -> Any:
    raw_action = action.get("action", action) if isinstance(action, dict) else action
    if hasattr(raw_action, "value"):
        raw_action = raw_action.value
    return raw_action


def normalize_action_id(action: Any) -> int:
    raw_action = extract_raw_action(action)
    if isinstance(raw_action, dict):
        raw_action = raw_action.get("action", 0)
    if hasattr(raw_action, "value"):
        raw_action = raw_action.value
    if isinstance(raw_action, str):
        normalized = raw_action.strip().upper().replace("-", "_")
        action_id = NAVIGATION_ACTION_NAME_TO_ID.get(normalized)
        if action_id is not None:
            return int(action_id)
        return int(normalized)
    return int(raw_action)


def normalize_action_name(action: Any, *, lower: bool = False) -> str:
    name = NAVIGATION_ACTION_ID_TO_NAME.get(normalize_action_id(action), "STOP")
    return name.lower() if lower else name


def vector_step_result(
    obs: Any,
    *,
    reward: float = 0.0,
    done: bool = False,
    info: Dict[str, Any] | None = None,
) -> List[Tuple[Any, float, bool, Dict[str, Any]]]:
    payload = dict(info or {})
    payload["done"] = bool(done)
    return [(obs, float(reward), bool(done), payload)]


def yaw_from_quaternion_like(rotation: Any) -> float:
    quat = np.asarray(rotation, dtype=np.float64).reshape(-1)
    if quat.size >= 4:
        try:
            from scipy.spatial.transform import Rotation as R

            return float(R.from_quat(quat[:4]).as_euler("zyx", degrees=False)[0])
        except Exception:
            x, y, z, w = [float(v) for v in quat[:4]]
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return float(math.atan2(siny_cosp, cosy_cosp))
    if quat.size == 1:
        return math.radians(float(quat[0]))
    return 0.0


def habitat_style_position(sim_position: Sequence[float], *, default_height: float = 1.3) -> np.ndarray:
    xyz = np.asarray(sim_position, dtype=np.float32).reshape(-1)
    x = float(xyz[0]) if xyz.size >= 1 else 0.0
    y = float(xyz[1]) if xyz.size >= 2 else 0.0
    z = float(xyz[2]) if xyz.size >= 3 else default_height
    # The shared floor logic reads height at Habitat's Y axis (index 1).
    return np.asarray([x, z, y], dtype=np.float32)
