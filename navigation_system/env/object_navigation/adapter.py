"""Single-environment OVON adapter that mimics the small VectorEnv subset SpaceVLN uses."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

from habitat_extensions.pose_utils import get_rel_pose_change, get_sim_location
from navigation_system.env.object_navigation.goal_task import (
    build_raw_object_goal_instruction,
)


ACTION_ID_TO_NAME = {
    0: "stop",
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
}


@dataclass
class SpaceVLNEpisodeFacade:
    """Expose OVON episodes with the `instruction.instruction_text` field SpaceVLN expects."""

    base_episode: Any

    def __post_init__(self) -> None:
        child_categories = list(
            getattr(self.base_episode, "children_object_categories", None) or []
        )
        self.object_goal = getattr(self.base_episode, "object_category", "")
        self.goal_aliases = tuple(child_categories)
        self.instruction = SimpleNamespace(
            instruction_text=build_raw_object_goal_instruction(
                self.object_goal,
                child_categories=child_categories,
            )
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_episode, name)


class SingleOVONVectorEnvAdapter:
    """Wrap `habitat.Env` so existing SpaceVLN controllers can run unchanged."""

    def __init__(self, env: Any, *, episode_count: Optional[int] = None) -> None:
        self.env = env
        self.number_of_episodes = int(episode_count or 1)
        self._last_pose: Optional[Sequence[float]] = None
        self._last_sim_position: Optional[np.ndarray] = None
        self._path_length_m: float = 0.0

    def _current_pose(self) -> Sequence[float]:
        return get_sim_location(self.env.sim)

    def _current_sim_position(self) -> np.ndarray:
        agent_state_getter = getattr(self.env.sim, "get_agent_state", None)
        if callable(agent_state_getter):
            try:
                state = agent_state_getter()
            except TypeError:
                state = agent_state_getter(0)
            return np.asarray(state.position[:3], dtype=np.float32)
        return np.asarray((0.0, 0.0, 0.0), dtype=np.float32)

    def _augment_observation(self, obs: Any, *, reset: bool) -> Any:
        if obs is None:
            return obs

        obs = dict(obs)
        current_pose = self._current_pose()
        current_sim_position = self._current_sim_position()
        if reset or self._last_pose is None:
            sensor_pose = np.zeros((3,), dtype=np.float32)
            self._last_sim_position = current_sim_position
            self._path_length_m = 0.0
        else:
            dx, dy, do = get_rel_pose_change(current_pose, self._last_pose)
            sensor_pose = np.asarray([dx, dy, do], dtype=np.float32)
            if self._last_sim_position is not None:
                self._path_length_m += float(
                    np.linalg.norm(current_sim_position - self._last_sim_position, ord=2)
                )
            self._last_sim_position = current_sim_position
        self._last_pose = current_pose
        obs["sensor_pose"] = sensor_pose
        return obs

    def _augment_metrics(self, metrics: Any) -> dict:
        payload = dict(metrics or {})
        payload["path_length"] = float(self._path_length_m)
        return payload

    @staticmethod
    def _normalize_action(action: Any) -> dict:
        if isinstance(action, dict):
            if "action" in action:
                raw_action = action["action"]
            else:
                raw_action = action
        else:
            raw_action = action

        if hasattr(raw_action, "value"):
            raw_action = raw_action.value

        if isinstance(raw_action, str):
            action_name = raw_action.strip().lower()
        else:
            action_name = ACTION_ID_TO_NAME.get(int(raw_action), "stop")

        return {"action": action_name}

    def reset(self) -> List[Any]:
        obs = self.env.reset()
        return [self._augment_observation(obs, reset=True)]

    def step(self, actions: Iterable[Any]) -> List[Any]:
        action = list(actions)[0]
        obs = self.env.step(self._normalize_action(action))
        done = bool(getattr(self.env, "episode_over", False))
        obs = self._augment_observation(obs, reset=False)
        info = self._augment_metrics(self.env.get_metrics())
        info["done"] = done
        reward = float(info.get("distance_to_goal_reward", 0.0) or 0.0)
        return [
            (
                obs,
                reward,
                done,
                info,
            )
        ]

    def current_episodes(self) -> List[SpaceVLNEpisodeFacade]:
        return [SpaceVLNEpisodeFacade(self.env.current_episode)]

    def call_at(self, index: int, method_name: str, *args: Any, **kwargs: Any) -> Any:
        if int(index) != 0:
            raise IndexError(f"SingleOVONVectorEnvAdapter only supports env index 0, got {index}")
        if method_name == "get_metrics":
            return self._augment_metrics(self.env.get_metrics())
        if method_name == "get_agent_pose":
            return self._current_pose()
        target = getattr(self.env, method_name)
        return target(*args, **kwargs)

    def close(self) -> None:
        self.env.close()
