"""NavGBench environment adapter for the shared Navigation Agent controller."""

from __future__ import annotations

import math
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from habitat_extensions.pose_utils import get_rel_pose_change
from navigation_system.env.common import (
    check_single_env_index,
    habitat_style_position,
    normalize_action_id,
    vector_step_result,
    yaw_from_quaternion_like,
)
from navigation_system.env.vlnce.navgbench.visualization import (
    build_navgbench_topdown_trajectory,
)
from navigation_system.env.vlnce.navgbench.protocol import receive_message, send_message


_WORKER_EOF_MESSAGE = "NavGBench subprocess closed its protocol pipe."


def normalize_navgbench_instruction_mode(mode: str) -> str:
    """Normalize public NavGBench instruction aliases to stored episode fields."""
    normalized = str(mode or "").strip().lower().replace("-", "_")
    if normalized in {"complex", "full", "grounded", "grounded_route"}:
        return "grounded"
    if normalized in {"simple", "short", "raw", "goal"}:
        return "raw"
    if normalized in {"moving", "metric", "turn_by_turn"}:
        return "moving"
    return "grounded"


def get_navgbench_episode_id(episode: Any) -> str:
    """Return NavGBench's stable scene/trajectory id when available."""
    ref_json = str(getattr(episode, "ref_json", "") or "").strip()
    if ref_json:
        from pathlib import Path

        p_ref = Path(ref_json)
        return f"{p_ref.parent.name}_{p_ref.stem}"
    return str(getattr(episode, "episode_id", "") or "").strip()


def format_navgbench_complex_instruction(
    *,
    simple_instruction: str,
    grounded_instruction: str,
) -> str:
    """Combine NavGBench's short goal and grounded route for complex prompts."""
    simple_text = str(simple_instruction or "").strip()
    grounded_text = str(grounded_instruction or "").strip()
    if not simple_text:
        return grounded_text
    if not grounded_text:
        return simple_text
    return (
        "Task Goal: "
        f"{simple_text}\n"
        "Task Instruction: "
        f"{grounded_text}"
    )


def _load_episode_json_payload(episode: Any) -> Dict[str, Any]:
    ref_json = str(getattr(episode, "ref_json", "") or "").strip()
    if not ref_json:
        return {}
    try:
        with Path(ref_json).expanduser().open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


@dataclass
class NavGBenchEpisodeFacade:
    """Expose a GN_Bench episode with the fields consumed by the agent."""

    base_episode: Any
    use_grounded_instruction: bool = False
    instruction_mode: str = "raw"

    def __post_init__(self) -> None:
        raw_payload = _load_episode_json_payload(self.base_episode)
        grounded_instruction = str(
            getattr(self.base_episode, "grounded_instruction", "")
            or raw_payload.get("grounded_instruction", "")
            or ""
        ).strip()
        raw_instruction = str(
            getattr(self.base_episode, "instruction", "")
            or raw_payload.get("instruction", "")
            or ""
        ).strip()
        moving_instruction = str(
            getattr(self.base_episode, "moving_instruction", "")
            or raw_payload.get("moving_instruction", "")
            or ""
        ).strip()
        mode = normalize_navgbench_instruction_mode(self.instruction_mode)
        if mode == "raw":
            instruction_text = raw_instruction or grounded_instruction or moving_instruction
        elif mode == "moving":
            instruction_text = moving_instruction or grounded_instruction or raw_instruction
        elif mode == "grounded":
            instruction_text = (
                format_navgbench_complex_instruction(
                    simple_instruction=raw_instruction,
                    grounded_instruction=grounded_instruction,
                )
                or moving_instruction
            )
        else:
            instruction_text = (
                format_navgbench_complex_instruction(
                    simple_instruction=raw_instruction,
                    grounded_instruction=grounded_instruction,
                )
                if self.use_grounded_instruction and grounded_instruction
                else raw_instruction or grounded_instruction or moving_instruction
            )
        self.instruction = SimpleNamespace(instruction_text=instruction_text)
        self.reference_path = self._build_reference_path()
        self.navgbench_id = get_navgbench_episode_id(self.base_episode)

    def _build_reference_path(self) -> List[List[float]]:
        path_info = getattr(self.base_episode, "path_info", None) or {}
        points = path_info.get("raster_world") or path_info.get("keypoints_world") or []
        reference_path: List[List[float]] = []
        for point in points:
            if isinstance(point, dict):
                raw_x = point.get("x")
                raw_y = point.get("y")
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                raw_x, raw_y = point[0], point[1]
            else:
                continue
            try:
                reference_path.append([float(raw_x), float(raw_y), 1.3])
            except (TypeError, ValueError):
                continue
        return reference_path

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_episode, name)


class _SubprocessSimFacade:
    def __init__(self, client: "NavGBenchSubprocessEnvClient") -> None:
        self.client = client

    def get_agent_state(self, *_args: Any) -> Any:
        return self.client.agent_state

    def get_observations_at(self) -> Any:
        return self.client.get_observations_at()


class NavGBenchSubprocessEnvClient:
    """Small proxy for a GN_Bench.Env running in NavGBench's own conda env."""

    def __init__(
        self,
        *,
        python_bin: str,
        navgbench_root: str | Path,
        gnbench_exp_config: str,
        episode_key: str,
        sensor_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.python_bin = str(Path(python_bin).expanduser())
        self.navgbench_root = Path(navgbench_root).expanduser().resolve()
        self.gnbench_exp_config = str(gnbench_exp_config)
        self.episode_key = str(episode_key)
        self.sensor_config = dict(sensor_config or {})
        self.number_of_episodes = 1
        self.episode_over = False
        self.current_episode = SimpleNamespace(episode_id=str(episode_key))
        self._last_state: Dict[str, Any] = {
            "position": [0.0, 0.0, 1.3],
            "yaw": 0.0,
        }
        self._last_observation: Any = None
        self._last_metrics: Dict[str, Any] = {}
        self._closed = False
        self.sim = _SubprocessSimFacade(self)

        self._process: subprocess.Popen[Any]
        self._reader: Any
        self._writer: Any
        self._start_worker()
        response = self._request(
            {
                "cmd": "init",
                "gnbench_root": str(self.navgbench_root),
                "gnbench_exp_config": self.gnbench_exp_config,
                "episode_key": self.episode_key,
                "sensor_config": self.sensor_config,
            }
        )
        self.current_episode = self._episode_from_payload(response.get("episode") or {})
        self.episode_over = bool(response.get("done", False))

    @staticmethod
    def _episode_from_payload(payload: Dict[str, Any]) -> Any:
        goals = [
            SimpleNamespace(position=list(goal.get("position") or []))
            for goal in payload.get("goals", []) or []
            if isinstance(goal, dict)
        ]
        return SimpleNamespace(
            episode_id=str(payload.get("episode_id", "")),
            scene_id=str(payload.get("scene_id", "")),
            ref_json=str(payload.get("ref_json", "")),
            instruction=payload.get("instruction", ""),
            moving_instruction=payload.get("moving_instruction", ""),
            grounded_instruction=payload.get("grounded_instruction", ""),
            path_info=payload.get("path_info") or {},
            label_info=payload.get("label_info") or {},
            goals=goals,
        )

    def _child_env(self) -> Dict[str, str]:
        child_env = os.environ.copy()
        env_root = Path(self.python_bin).resolve().parents[1]
        old_pythonpath = child_env.get("PYTHONPATH", "")
        pythonpath_parts = [
            str(Path(__file__).resolve().parents[4]),
            str(self.navgbench_root / "GN-Bench-Tools"),
            str(self.navgbench_root),
        ]
        if old_pythonpath:
            pythonpath_parts.append(old_pythonpath)
        child_env["PYTHONPATH"] = ":".join(pythonpath_parts)
        child_env["CONDA_PREFIX"] = str(env_root)
        child_env.pop("PYTHONHOME", None)

        # The parent launcher may preload Habitat/torch libraries from another
        # conda env. Drop those for the GN_Bench worker so its CUDA extensions
        # resolve against the NavGBench env instead.
        child_env.pop("LD_PRELOAD", None)
        old_ld_path = child_env.get("LD_LIBRARY_PATH", "")
        filtered_ld_path = [
            item
            for item in old_ld_path.split(":")
            if item
            and "spacevln" not in item
            and "habitat_sim" not in item
        ]
        torch_lib = next(env_root.glob("lib/python*/site-packages/torch/lib"), None)
        prepend = [str(env_root / "lib")]
        if torch_lib is not None and torch_lib.is_dir():
            prepend.append(str(torch_lib))
        child_env["LD_LIBRARY_PATH"] = ":".join(prepend + filtered_ld_path)
        return child_env

    def _start_worker(self) -> None:
        worker_path = Path(__file__).with_name("worker.py")
        if not worker_path.is_file():
            raise FileNotFoundError(f"NavGBench worker script not found: {worker_path}")
        if not Path(self.python_bin).is_file():
            raise FileNotFoundError(f"NavGBench Python not found: {self.python_bin}")

        parent_to_child_r, parent_to_child_w = os.pipe()
        child_to_parent_r, child_to_parent_w = os.pipe()
        child_env = self._child_env()
        child_env["SPACEVLN_NAVGBENCH_READ_FD"] = str(parent_to_child_r)
        child_env["SPACEVLN_NAVGBENCH_WRITE_FD"] = str(child_to_parent_w)

        self._process = subprocess.Popen(
            [self.python_bin, str(worker_path)],
            cwd=str(self.navgbench_root),
            env=child_env,
            pass_fds=(parent_to_child_r, child_to_parent_w),
            stdin=subprocess.DEVNULL,
        )
        os.close(parent_to_child_r)
        os.close(child_to_parent_w)
        self._writer = os.fdopen(parent_to_child_w, "wb", buffering=0)
        self._reader = os.fdopen(child_to_parent_r, "rb", buffering=0)

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("NavGBench subprocess client is already closed.")
        try:
            send_message(self._writer, payload)
            response = receive_message(self._reader, eof_message=_WORKER_EOF_MESSAGE)
        except Exception:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"NavGBench subprocess exited with code {self._process.returncode}."
                )
            raise
        if not response.get("ok", False):
            error = str(response.get("error") or "NavGBench subprocess request failed.")
            tb = str(response.get("traceback") or "").strip()
            if tb:
                error = f"{error}\n{tb}"
            raise RuntimeError(error)
        return response

    @property
    def agent_state(self) -> Any:
        position = np.asarray(
            self._last_state.get("position", [0.0, 0.0, 1.3]),
            dtype=np.float32,
        ).reshape(-1)
        if position.size < 3:
            position = np.pad(position, (0, 3 - position.size), constant_values=0.0)
            position[2] = 1.3
        yaw = float(self._last_state.get("yaw", 0.0) or 0.0)
        rotation = np.asarray(
            [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
            dtype=np.float64,
        )
        return SimpleNamespace(position=position, rotation=rotation)

    def _apply_observation_response(self, response: Dict[str, Any]) -> Any:
        self.episode_over = bool(response.get("done", False))
        self._last_state = dict(response.get("state") or self._last_state)
        self._last_metrics = dict(response.get("metrics") or {})
        self._last_observation = response.get("observation")
        return self._last_observation

    def reset(self) -> Any:
        return self._apply_observation_response(self._request({"cmd": "reset"}))

    def step(self, action: Any) -> Any:
        return self._apply_observation_response(
            self._request({"cmd": "step", "action": action})
        )

    def get_observations_at(self) -> Any:
        return self._apply_observation_response(
            self._request({"cmd": "get_observations_at"})
        )

    def get_metrics(self) -> Dict[str, Any]:
        response = self._request({"cmd": "get_metrics"})
        self.episode_over = bool(response.get("done", False))
        self._last_metrics = dict(response.get("metrics") or {})
        return dict(self._last_metrics)

    def get_visualizations(self) -> Dict[str, Any]:
        response = self._request({"cmd": "get_visualizations"})
        return dict(response.get("visualizations") or {})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.poll() is None:
                try:
                    send_message(self._writer, {"cmd": "close"})
                except Exception:
                    pass
        finally:
            for stream in (getattr(self, "_writer", None), getattr(self, "_reader", None)):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            if self._process.poll() is None:
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()


class SingleNavGBenchVectorEnvAdapter:
    """Wrap one `GN_Bench.Env` with the shared small VectorEnv API."""

    def __init__(
        self,
        env: Any,
        *,
        episode_count: Optional[int] = None,
        use_grounded_instruction: bool = False,
        instruction_mode: str = "raw",
        require_depth: bool = True,
        turn_repeat: int = 2,
    ) -> None:
        self.env = env
        self.number_of_episodes = int(episode_count or 1)
        self.use_grounded_instruction = bool(use_grounded_instruction)
        self.instruction_mode = normalize_navgbench_instruction_mode(instruction_mode)
        self.require_depth = bool(require_depth)
        self.turn_repeat = max(1, int(turn_repeat or 1))
        self._last_pose: Optional[Sequence[float]] = None

    def _agent_state(self) -> Any:
        getter = getattr(self.env.sim, "get_agent_state", None)
        if not callable(getter):
            raise RuntimeError("NavGBench simulator does not expose get_agent_state().")
        try:
            return getter()
        except TypeError:
            return getter(0)

    @staticmethod
    def _yaw_from_quat(rotation: Any) -> float:
        return yaw_from_quaternion_like(rotation)

    def _current_pose(self) -> Sequence[float]:
        state = self._agent_state()
        position = np.asarray(getattr(state, "position", [0.0, 0.0, 1.3]), dtype=np.float32)
        yaw = self._yaw_from_quat(getattr(state, "rotation", [0.0, 0.0, 0.0, 1.0]))
        return (float(position[0]), float(position[1]), float(yaw))

    def _current_sim_position(self) -> np.ndarray:
        state = self._agent_state()
        return np.asarray(getattr(state, "position", [0.0, 0.0, 1.3]), dtype=np.float32)

    def _augment_observation(self, obs: Any, *, reset: bool) -> Any:
        if obs is None:
            return obs

        obs = dict(obs)
        if self.require_depth and "depth" not in obs:
            raise RuntimeError(
                "NavGBench observation does not contain `depth`. "
                "Enable DEPTH_SENSOR in the GN_Bench simulator config before running the agent."
            )

        current_pose = self._current_pose()
        if reset or self._last_pose is None:
            sensor_pose = np.zeros((3,), dtype=np.float32)
        else:
            dx, dy, do = get_rel_pose_change(current_pose, self._last_pose)
            sensor_pose = np.asarray([dx, dy, do], dtype=np.float32)
        self._last_pose = current_pose

        sim_position = self._current_sim_position()
        obs["sensor_pose"] = sensor_pose
        obs["position"] = habitat_style_position(sim_position)
        return obs

    @staticmethod
    def _normalize_action(action: Any) -> dict:
        return {"action": normalize_action_id(action)}

    def _augment_metrics(self, metrics: Any) -> dict:
        payload = dict(metrics or {})
        payload["done"] = bool(getattr(self.env, "episode_over", False))
        return payload

    def _current_topdown_map_payload(self) -> Optional[Dict[str, Any]]:
        """Return GN-Bench's standard occupancy trajectory map for shared visualization."""
        image = None
        color_space = "rgb"

        getter = getattr(self.env, "get_visualizations", None)
        if callable(getter):
            visualizations = dict(getter() or {})
            image = visualizations.get("occ_trajectory_green_gt")
            if image is None:
                image = visualizations.get("occ_trajectory")
            if image is None:
                image = visualizations.get("bev_trajectory")
                color_space = "bgr"
            if image is None:
                image = visualizations.get("occ_current")
                color_space = "rgb"
        else:
            sim = getattr(self.env, "sim", None)
            if sim is not None:
                image = build_navgbench_topdown_trajectory(sim)
                method = getattr(sim, "get_occ_map_with_trajectory", None)
                if image is None and callable(method):
                    image = method()
                if image is None:
                    method = getattr(sim, "get_bev_map_with_trajectory", None)
                    if callable(method):
                        image = method()
                        color_space = "bgr"
                if image is None:
                    method = getattr(sim, "get_occ_map", None)
                    if callable(method):
                        image = method()
                        color_space = "rgb"

        if image is None:
            return None
        array = np.asarray(image)
        if array.size == 0:
            return None
        return {
            "image_array": array,
            "color_space": color_space,
            "artifact_name": "top_down_map.png",
            "name": "top_down_map",
        }

    def get_global_map_input(self) -> Optional[Dict[str, Any]]:
        """Expose NavGBench top-down trajectory through the shared visualizer API."""
        return self._current_topdown_map_payload()

    def reset(self) -> List[Any]:
        obs = self.env.reset()
        return [self._augment_observation(obs, reset=True)]

    def step(self, actions: Iterable[Any]) -> List[Any]:
        action_list = list(actions)
        if not action_list:
            raise ValueError("SingleNavGBenchVectorEnvAdapter.step requires one action.")
        normalized_action = self._normalize_action(action_list[0])
        raw_action = normalized_action.get("action", 0)
        repeat = self.turn_repeat if int(raw_action) in (2, 3) else 1

        obs = None
        for _ in range(repeat):
            if bool(getattr(self.env, "episode_over", False)):
                break
            obs = self.env.step(normalized_action)
        if obs is None:
            obs = self.env.sim.get_observations_at()
        done = bool(getattr(self.env, "episode_over", False))
        obs = self._augment_observation(obs, reset=False)
        info = self._augment_metrics(self.env.get_metrics())
        return vector_step_result(obs, reward=0.0, done=done, info=info)

    def current_episodes(self) -> List[NavGBenchEpisodeFacade]:
        return [
            NavGBenchEpisodeFacade(
                self.env.current_episode,
                use_grounded_instruction=self.use_grounded_instruction,
                instruction_mode=self.instruction_mode,
            )
        ]

    def call_at(self, index: int, method_name: str, *args: Any, **kwargs: Any) -> Any:
        check_single_env_index(index)
        if method_name == "get_metrics":
            return self._augment_metrics(self.env.get_metrics())
        if method_name == "get_agent_pose":
            return self._current_pose()
        if method_name == "get_global_map_input":
            return self.get_global_map_input()
        if method_name == "get_visualizations":
            getter = getattr(self.env, "get_visualizations", None)
            if callable(getter):
                return dict(getter() or {})
        target = getattr(self.env, method_name)
        return target(*args, **kwargs)

    def close(self) -> None:
        self.env.close()
