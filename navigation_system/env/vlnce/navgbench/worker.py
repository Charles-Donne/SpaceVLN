"""GN_Bench subprocess worker used by the SpaceVLN NavGBench adapter."""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np

from navigation_system.env.vlnce.navgbench.protocol import receive_message, send_message
from navigation_system.env.vlnce.navgbench.visualization import (
    build_navgbench_topdown_trajectory,
)


_PARENT_EOF_MESSAGE = "Parent closed NavGBench protocol pipe."


@contextlib.contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _add_navgbench_paths(root: Path) -> None:
    for candidate in (root / "GN-Bench-Tools", root):
        text = str(candidate.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)


def _abspath_if_relative(value: str, *, root: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((root / path).resolve())


def _resolve_path(path_like: str | Path, *, base: Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _stable_episode_id(episode: Any) -> str:
    ref_json = str(getattr(episode, "ref_json", "") or "").strip()
    if ref_json:
        p_ref = Path(ref_json)
        return f"{p_ref.parent.name}_{p_ref.stem}"
    return str(getattr(episode, "episode_id", "") or "").strip()


def _yaw_from_rotation(rotation: Any) -> float:
    if hasattr(rotation, "w") and hasattr(rotation, "x"):
        w = float(rotation.w)
        x = float(rotation.x)
        y = float(rotation.y)
        z = float(rotation.z)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))

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


def _json_or_empty(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _episode_payload(episode: Any) -> Dict[str, Any]:
    goals = []
    for goal in getattr(episode, "goals", []) or []:
        position = getattr(goal, "position", None)
        if position is not None:
            goals.append({"position": np.asarray(position, dtype=np.float32).tolist()})

    ref_json = str(getattr(episode, "ref_json", "") or "")
    raw = _json_or_empty(ref_json)
    return {
        "episode_id": str(getattr(episode, "episode_id", "")),
        "scene_id": str(getattr(episode, "scene_id", "")),
        "ref_json": ref_json,
        "instruction": getattr(episode, "instruction", raw.get("instruction", "")),
        "moving_instruction": getattr(
            episode,
            "moving_instruction",
            raw.get("moving_instruction", ""),
        ),
        "grounded_instruction": getattr(
            episode,
            "grounded_instruction",
            raw.get("grounded_instruction", ""),
        ),
        "path_info": getattr(episode, "path_info", raw.get("path", {})),
        "label_info": getattr(episode, "label_info", raw.get("label", {})),
        "goals": goals,
        "navgbench_id": _stable_episode_id(episode),
    }


def _sanitize_metrics(metrics: Any) -> Dict[str, Any]:
    result = {}
    for key, value in dict(metrics or {}).items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def _prepare_gnbench_config(config: Any, *, root: Path, sensor_config: Dict[str, Any]) -> Any:
    config.defrost()
    task_config = config.TASK_CONFIG
    dataset_cfg = task_config.DATASET
    dataset_cfg.DATA_PATH = _abspath_if_relative(dataset_cfg.DATA_PATH, root=root)
    dataset_cfg.SCENES_DIR = _abspath_if_relative(dataset_cfg.SCENES_DIR, root=root)
    if hasattr(dataset_cfg, "DATASET_CONFIG"):
        dataset_cfg.DATASET_CONFIG = _abspath_if_relative(
            dataset_cfg.DATASET_CONFIG,
            root=root,
        )

    simulator = task_config.SIMULATOR
    simulator.FORWARD_STEP_SIZE = float(getattr(simulator, "FORWARD_STEP_SIZE", 0.25))
    simulator.TURN_ANGLE = float(getattr(simulator, "TURN_ANGLE", 15.0))
    if hasattr(simulator, "AGENT_0"):
        simulator.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
        simulator.AGENT_0.HEIGHT = float(sensor_config.get("agent_height_m", 1.3))

    rgb_sensor = getattr(simulator, "RGB_SENSOR", None)
    depth_sensor = getattr(simulator, "DEPTH_SENSOR", None)
    for sensor in (rgb_sensor, depth_sensor):
        if sensor is None:
            continue
        if hasattr(sensor, "WIDTH"):
            sensor.WIDTH = int(sensor_config.get("frame_width", 480))
        if hasattr(sensor, "HEIGHT"):
            sensor.HEIGHT = int(sensor_config.get("frame_height", 360))
        if hasattr(sensor, "HFOV"):
            sensor.HFOV = float(
                sensor_config.get(
                    "gnbench_sensor_fov_deg",
                    sensor_config.get("hfov_deg", 70.0),
                )
            )
    if depth_sensor is not None:
        if hasattr(depth_sensor, "MIN_DEPTH"):
            depth_sensor.MIN_DEPTH = float(sensor_config.get("min_depth_m", 0.5))
        if hasattr(depth_sensor, "MAX_DEPTH"):
            depth_sensor.MAX_DEPTH = float(sensor_config.get("max_depth_m", 5.0))
        if hasattr(depth_sensor, "NORMALIZE_DEPTH"):
            depth_sensor.NORMALIZE_DEPTH = True

    config.freeze()
    return config


def _select_episode(dataset: Any, episode_key: str) -> Any:
    wanted = str(episode_key).strip()
    dataset.episodes.sort(
        key=lambda ep: (
            0,
            int(ep.episode_id),
        )
        if str(ep.episode_id).isdigit()
        else (1, str(ep.episode_id))
    )
    for episode in dataset.episodes:
        if str(getattr(episode, "episode_id", "")) == wanted:
            return episode
        if _stable_episode_id(episode) == wanted:
            return episode
    raise RuntimeError(f"NavGBench episode id not found in worker: {wanted}")


class WorkerState:
    def __init__(self) -> None:
        self.root: Optional[Path] = None
        self.config: Any = None
        self.dataset: Any = None
        self.episode: Any = None
        self.env: Any = None

    def init(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.root = Path(payload["gnbench_root"]).expanduser().resolve()
        exp_config = str(payload["gnbench_exp_config"])
        episode_key = str(payload["episode_key"])
        sensor_config = dict(payload.get("sensor_config") or {})
        _add_navgbench_paths(self.root)

        from GN_Bench import Env
        from GN_Bench.datasets import make_dataset
        from VLN_CE.vlnce_baselines.config.default import get_config as get_gn_config
        import VLN_CE.GN_Bench_extensions  # noqa: F401

        with _pushd(self.root):
            config = get_gn_config(str(_resolve_path(exp_config, base=self.root)))
        self.config = _prepare_gnbench_config(
            config,
            root=self.root,
            sensor_config=sensor_config,
        )
        self.dataset = make_dataset(
            id_dataset=self.config.TASK_CONFIG.DATASET.TYPE,
            config=self.config.TASK_CONFIG.DATASET,
        )
        self.episode = _select_episode(self.dataset, episode_key)
        single_episode_dataset = copy.copy(self.dataset)
        single_episode_dataset.episodes = [self.episode]
        self.env = Env(self.config.TASK_CONFIG, single_episode_dataset)
        return {
            "ok": True,
            "episode": _episode_payload(self.episode),
            "done": bool(getattr(self.env, "episode_over", False)),
        }

    def _state_payload(self) -> Dict[str, Any]:
        getter = getattr(self.env.sim, "get_agent_state")
        try:
            state = getter()
        except TypeError:
            state = getter(0)
        position = np.asarray(
            getattr(state, "position", [0.0, 0.0, 1.3]),
            dtype=np.float32,
        ).reshape(-1)
        return {
            "position": position.tolist(),
            "yaw": _yaw_from_rotation(getattr(state, "rotation", [0, 0, 0, 1])),
        }

    def _observation_response(self, observation: Any) -> Dict[str, Any]:
        return {
            "ok": True,
            "observation": dict(observation or {}),
            "state": self._state_payload(),
            "metrics": _sanitize_metrics(self.env.get_metrics()),
            "done": bool(getattr(self.env, "episode_over", False)),
        }

    def reset(self) -> Dict[str, Any]:
        return self._observation_response(self.env.reset())

    def step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._observation_response(self.env.step(payload.get("action")))

    def get_observations_at(self) -> Dict[str, Any]:
        return self._observation_response(self.env.sim.get_observations_at())

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "metrics": _sanitize_metrics(self.env.get_metrics()),
            "done": bool(getattr(self.env, "episode_over", False)),
        }

    def get_visualizations(self) -> Dict[str, Any]:
        sim = getattr(self.env, "sim", None)
        visualizations: Dict[str, Any] = {}
        if sim is not None:
            green_gt = build_navgbench_topdown_trajectory(sim)
            if green_gt is not None:
                visualizations["occ_trajectory_green_gt"] = green_gt
            for name, method_name in (
                ("occ_trajectory", "get_occ_map_with_trajectory"),
                ("bev_trajectory", "get_bev_map_with_trajectory"),
                ("occ_current", "get_occ_map"),
            ):
                method = getattr(sim, method_name, None)
                if callable(method):
                    image = method()
                    if image is not None:
                        visualizations[name] = image
            current_pixel = getattr(sim, "get_current_pixel_position", None)
            if callable(current_pixel):
                pixel = current_pixel()
                if pixel is not None:
                    visualizations["current_pixel"] = list(pixel)
        return {
            "ok": True,
            "visualizations": visualizations,
            "done": bool(getattr(self.env, "episode_over", False)),
        }

    def close(self) -> Dict[str, Any]:
        if self.env is not None:
            self.env.close()
            self.env = None
        return {"ok": True}


def _handle_command(state: WorkerState, payload: Dict[str, Any]) -> Dict[str, Any]:
    cmd = str(payload.get("cmd", ""))
    if cmd == "init":
        return state.init(payload)
    if state.env is None:
        raise RuntimeError(f"NavGBench worker received {cmd!r} before init.")
    if cmd == "reset":
        return state.reset()
    if cmd == "step":
        return state.step(payload)
    if cmd == "get_observations_at":
        return state.get_observations_at()
    if cmd == "get_metrics":
        return state.get_metrics()
    if cmd == "get_visualizations":
        return state.get_visualizations()
    if cmd == "close":
        return state.close()
    raise RuntimeError(f"Unsupported NavGBench worker command: {cmd}")


def main() -> int:
    read_fd = int(os.environ["SPACEVLN_NAVGBENCH_READ_FD"])
    write_fd = int(os.environ["SPACEVLN_NAVGBENCH_WRITE_FD"])
    reader = os.fdopen(read_fd, "rb", buffering=0)
    writer = os.fdopen(write_fd, "wb", buffering=0)
    state = WorkerState()

    while True:
        try:
            payload = receive_message(reader, eof_message=_PARENT_EOF_MESSAGE)
        except EOFError:
            break
        try:
            response = _handle_command(state, payload)
        except BaseException as exc:
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {str(exc)}",
                "traceback": traceback.format_exc(),
            }
        send_message(writer, response)
        if payload.get("cmd") == "close":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
