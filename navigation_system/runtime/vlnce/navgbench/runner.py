"""Run the Navigation Agent on NavGBench/GN-Bench episodes."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import json
import math
import multiprocessing
import os
import random
import signal
import sys
import traceback
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from navigation_system.config import get_config as get_spacevln_config
from navigation_system.config.core.setup import apply_runtime_derived_fields
from navigation_system.env.vlnce.navgbench import (
    NavGBenchSubprocessEnvClient,
    SingleNavGBenchVectorEnvAdapter,
    get_navgbench_episode_id,
    normalize_navgbench_instruction_mode,
)
from navigation_system.runtime.episode_io import (
    get_episode_records_log_path,
    is_abnormal_episode_failure,
    load_json_if_exists,
    redirect_process_output_to_file,
    redirect_process_output_to_null,
    save_episode_stdout_log_enabled,
    should_suppress_normal_failure_reason,
)
from navigation_system.runtime.output_policy import (
    add_output_artifact_args,
    add_output_profile_arg,
    apply_output_policy_to_config,
)
from navigation_system.runtime.results_report import generate_results_report
from navigation_system.runtime.storage.artifacts import (
    SaveManager,
    get_episode_log_path,
)
from navigation_system.runtime.storage.results_layout import (
    build_default_results_family_root,
    build_model_results_dir_name,
    resolve_api_config_path,
    resolve_results_dir_path,
    resolve_results_root_path,
)
from navigation_system.runtime.vlnce.profiles import (
    CONTEXT_CACHE_RUNTIME_PROFILE,
    STANDARD_RUNTIME_PROFILE,
    NavigationRuntimeProfile,
)
from navigation_system.vlm.vlnce.navgbench_runtime_factory import (
    build_navgbench_context_cache_navigation_model_stack,
    build_navgbench_navigation_model_stack,
)
from navigation_system.vlm.api.qwen_context_cache_client import (
    validate_qwen_context_cache_api_config,
)


RUNTIME_CHOICES = ("standard", "context_cache")
DEFAULT_SPACEVLN_CONFIG = "navigation_system/config/experiments/vlnce/navgbench_eval.yaml"
DEFAULT_GNBENCH_CONFIG = "VLN_CE/vlnce_baselines/config/baselines/bae_InteriorGS.yaml"
DEFAULT_API_CONFIG = "navigation_system/config/vlm/vlm_api_config.yaml"
DEFAULT_BACKEND = "auto"
DEFAULT_RUNTIME = "context_cache"
DEFAULT_INSTRUCTION_MODE = "complex"


def _format_exception_message(exc: BaseException) -> str:
    exc_type = type(exc).__name__
    exc_text = str(exc).strip()
    return f"{exc_type}: {exc_text}" if exc_text else exc_type


def _parallel_worker_initializer() -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(max(1, int(os.getenv("SPACEVLN_TORCH_NUM_THREADS", "1") or 1)))
        torch.set_num_interop_threads(
            max(1, int(os.getenv("SPACEVLN_TORCH_INTEROP_THREADS", "1") or 1))
        )
    except Exception:
        pass


def _shutdown_parallel_executor(
    executor: concurrent.futures.ProcessPoolExecutor,
    *,
    interrupted: bool,
) -> None:
    if interrupted:
        processes = list((getattr(executor, "_processes", None) or {}).values())
        for process in processes:
            try:
                if process is not None and process.is_alive():
                    process.terminate()
            except Exception:
                pass
        for process in processes:
            try:
                if process is not None:
                    process.join(timeout=0.5)
            except Exception:
                pass
    try:
        executor.shutdown(wait=not interrupted)
    except Exception:
        if not interrupted:
            raise


@contextlib.contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_navgbench_root() -> Path:
    env_root = str(os.getenv("NAVGBENCH_ROOT", "") or "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (_workspace_root() / "Nav-GBench").resolve()


def _default_navgbench_python() -> str:
    env_python = str(
        os.getenv("SPACEVLN_NAVGBENCH_PYTHON", "")
        or os.getenv("NAVGBENCH_PYTHON", "")
        or ""
    ).strip()
    if env_python:
        return str(Path(env_python).expanduser())

    env_name = "gn_bench"
    candidates: List[Path] = []
    conda_prefix = str(os.getenv("CONDA_PREFIX", "") or "").strip()
    if conda_prefix:
        candidates.append(Path(conda_prefix).expanduser().resolve().parent / env_name / "bin/python")
    mamba_root = str(os.getenv("MAMBA_ROOT_PREFIX", "") or "").strip()
    if mamba_root:
        candidates.append(Path(mamba_root).expanduser() / "envs" / env_name / "bin/python")
    candidates.extend(
        [
            Path.home() / ".conda/envs" / env_name / "bin/python",
            Path.home() / "miniconda3/envs" / env_name / "bin/python",
            Path.home() / "anaconda3/envs" / env_name / "bin/python",
            Path.home() / "miniforge3/envs" / env_name / "bin/python",
            Path.home() / "mambaforge/envs" / env_name / "bin/python",
            Path("/opt/conda/envs") / env_name / "bin/python",
        ]
    )
    for env_root in (
        Path("/home").glob("*/.conda/envs"),
        Path("/home").glob("*/miniconda3/envs"),
        Path("/home").glob("*/anaconda3/envs"),
        Path("/home").glob("*/miniforge3/envs"),
        Path("/home").glob("*/mambaforge/envs"),
    ):
        for root in env_root:
            candidates.append(root / env_name / "bin/python")
    if Path(sys.executable).resolve().parent.parent.name == env_name:
        candidates.insert(0, Path(sys.executable))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _navgbench_install_hint(root: Path) -> str:
    return "\n".join(
        [
            "Install NavGBench in the selected Python environment, for example:",
            f"  cd {root / 'GN-Bench-Tools'} && pip install -e .",
            f"  cd {root} && pip install plyfile",
            f"  cd {root} && pip install ./submodules/diff-gaussian-rasterization --no-build-isolation",
            f"  cd {root} && pip install ./submodules/simple-knn --no-build-isolation",
        ]
    )


def _navgbench_import_hint(exc: BaseException, *, root: Path) -> RuntimeError:
    missing = str(getattr(exc, "name", "") or "").strip()
    install_lines = [
        "NavGBench Python/CUDA dependencies are not available in the selected Python environment.",
        f"Missing module: {missing or type(exc).__name__}",
        _navgbench_install_hint(root),
    ]
    return RuntimeError("\n".join(install_lines))


def _resolve_path(path_like: str | Path, *, base: Optional[Path] = None) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((base or _project_root()) / path).resolve()


def _add_navgbench_paths(root: Path) -> None:
    for candidate in (root / "GN-Bench-Tools", root):
        text = str(candidate.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)


def _ensure_gymnasium_compat() -> None:
    try:
        import gymnasium  # noqa: F401
        return
    except ImportError:
        pass

    try:
        import gym
    except ImportError:
        return

    if not hasattr(gym.spaces, "Text"):
        class Text(gym.Space):
            def __init__(self, max_length: int = 1000, min_length: int = 0, **kwargs):
                super().__init__(shape=(), dtype=str)
                self.max_length = int(max_length)
                self.min_length = int(min_length)

            def sample(self):
                return ""

            def contains(self, x):
                return isinstance(x, str) and self.min_length <= len(x) <= self.max_length

        gym.spaces.Text = Text

    sys.modules.setdefault("gymnasium", gym)


def _can_use_in_process_backend(navgbench_root: Path) -> bool:
    _add_navgbench_paths(navgbench_root)
    _ensure_gymnasium_compat()
    try:
        import GN_Bench  # noqa: F401
        from GN_Bench.datasets import make_dataset  # noqa: F401
        from VLN_CE.vlnce_baselines.config.default import get_config  # noqa: F401
        import VLN_CE.GN_Bench_extensions  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_navgbench_runtime_profile(
    runtime_name: str,
    *,
    instruction_mode: str,
) -> NavigationRuntimeProfile:
    normalized = str(runtime_name or "standard").strip().lower()
    if normalized == "context_cache":
        return replace(
            CONTEXT_CACHE_RUNTIME_PROFILE,
            model_stack_builder=partial(
                build_navgbench_context_cache_navigation_model_stack,
                instruction_mode=instruction_mode,
            ),
        )
    return replace(
        STANDARD_RUNTIME_PROFILE,
        model_stack_builder=partial(
            build_navgbench_navigation_model_stack,
            instruction_mode=instruction_mode,
        ),
    )


def _results_instruction_dir_name(instruction_mode: str) -> str:
    normalized = normalize_navgbench_instruction_mode(instruction_mode)
    if normalized == "grounded":
        return "complex"
    if normalized == "raw":
        return "simple"
    return normalized or "complex"


def _default_results_dir(
    api_config: str,
    results_root: str = "",
    *,
    runtime: str = "standard",
    instruction_mode: str = DEFAULT_INSTRUCTION_MODE,
) -> str:
    family_root = build_default_results_family_root(
        "navgbench",
        results_root=results_root or None,
    )
    model_dir = build_model_results_dir_name(api_config)
    if str(runtime or "").strip().lower() == "context_cache":
        suffix = "_cache"
        resolved_api_config = resolve_api_config_path(api_config)
        if resolved_api_config and os.path.exists(resolved_api_config):
            try:
                with open(resolved_api_config, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                cache_block = dict(raw.get("qwen_context_cache") or {})
                suffix = str(cache_block.get("results_dir_suffix") or "_cache").strip()
            except Exception:
                suffix = "_cache"
        model_dir = f"{model_dir}{suffix}" if suffix else model_dir
    instruction_dir = _results_instruction_dir_name(instruction_mode)
    return str((Path(family_root) / instruction_dir / model_dir).resolve())


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _abspath_if_relative(value: str, *, root: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((root / path).resolve())


def _gnbench_sensor_fov_from_horizontal(
    horizontal_fov_deg: float,
    *,
    width: int,
    height: int,
) -> float:
    """GN-Bench's HFOV field is consumed as vertical FOV by its renderer."""
    width = max(1, int(width))
    height = max(1, int(height))
    half_horizontal = math.radians(float(horizontal_fov_deg)) / 2.0
    vertical = 2.0 * math.atan(math.tan(half_horizontal) * (height / width))
    return math.degrees(vertical)


def _prepare_gnbench_config(config: Any, *, root: Path, space_config: Any = None) -> Any:
    """Make GN_Bench config usable from the agent process cwd."""
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
        simulator.AGENT_0.HEIGHT = float(getattr(simulator.AGENT_0, "HEIGHT", 1.3) or 1.3)

    rgb_sensor = getattr(simulator, "RGB_SENSOR", None)
    depth_sensor = getattr(simulator, "DEPTH_SENSOR", None)
    if rgb_sensor is not None and depth_sensor is not None:
        for key in ("WIDTH", "HEIGHT", "HFOV"):
            if hasattr(rgb_sensor, key):
                setattr(depth_sensor, key, getattr(rgb_sensor, key))
        if hasattr(depth_sensor, "NORMALIZE_DEPTH"):
            depth_sensor.NORMALIZE_DEPTH = True

    if space_config is not None:
        try:
            sensor_cfg = space_config.SPACE.SENSOR
            sensor_width = int(sensor_cfg.FRAME_WIDTH)
            sensor_height = int(sensor_cfg.FRAME_HEIGHT)
            gnbench_sensor_fov_deg = _gnbench_sensor_fov_from_horizontal(
                float(sensor_cfg.HFOV_DEG),
                width=sensor_width,
                height=sensor_height,
            )
            if rgb_sensor is not None:
                rgb_sensor.WIDTH = sensor_width
                rgb_sensor.HEIGHT = sensor_height
                rgb_sensor.HFOV = float(gnbench_sensor_fov_deg)
            if depth_sensor is not None:
                depth_sensor.WIDTH = sensor_width
                depth_sensor.HEIGHT = sensor_height
                depth_sensor.HFOV = float(gnbench_sensor_fov_deg)
            if hasattr(simulator, "AGENT_0"):
                simulator.AGENT_0.HEIGHT = float(sensor_cfg.AGENT_HEIGHT_M)
            task_depth_cfg = getattr(space_config.TASK_CONFIG.SIMULATOR, "DEPTH_SENSOR", None)
            if depth_sensor is not None and task_depth_cfg is not None:
                if hasattr(depth_sensor, "MIN_DEPTH"):
                    depth_sensor.MIN_DEPTH = float(task_depth_cfg.MIN_DEPTH)
                if hasattr(depth_sensor, "MAX_DEPTH"):
                    depth_sensor.MAX_DEPTH = float(task_depth_cfg.MAX_DEPTH)
                if hasattr(depth_sensor, "NORMALIZE_DEPTH"):
                    depth_sensor.NORMALIZE_DEPTH = True
        except Exception:
            pass

    config.freeze()
    return config


def _load_gnbench_config_and_dataset(
    *,
    navgbench_root: Path,
    gnbench_exp_config: str,
    space_config: Any = None,
) -> Tuple[Any, Any]:
    _add_navgbench_paths(navgbench_root)
    _ensure_gymnasium_compat()
    with _pushd(navgbench_root):
        try:
            from GN_Bench.datasets import make_dataset
            from VLN_CE.vlnce_baselines.config.default import get_config as get_gn_config
            import VLN_CE.GN_Bench_extensions  # noqa: F401
        except ModuleNotFoundError as exc:
            raise _navgbench_import_hint(exc, root=navgbench_root) from exc

        config = get_gn_config(gnbench_exp_config)
    config = _prepare_gnbench_config(
        config,
        root=navgbench_root,
        space_config=space_config,
    )
    dataset = make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    dataset.episodes.sort(
        key=lambda ep: (
            0,
            int(ep.episode_id),
        )
        if str(ep.episode_id).isdigit()
        else (1, str(ep.episode_id))
    )
    return config, dataset


def _load_navgbench_metadata_dataset(
    *,
    navgbench_root: Path,
    gnbench_exp_config: str,
) -> Any:
    """Load episode metadata directly, without importing GN_Bench rendering deps."""
    exp_path = _resolve_path(gnbench_exp_config, base=navgbench_root)
    with exp_path.open("r", encoding="utf-8") as f:
        exp_payload = yaml.safe_load(f) or {}

    task_config_path = str(exp_payload.get("BASE_TASK_CONFIG_PATH") or "").strip()
    if not task_config_path:
        raise RuntimeError(f"GNBench config has no BASE_TASK_CONFIG_PATH: {exp_path}")
    task_path = _resolve_path(task_config_path, base=navgbench_root)
    with task_path.open("r", encoding="utf-8") as f:
        task_payload = yaml.safe_load(f) or {}

    dataset_payload = dict(task_payload.get("DATASET") or {})
    data_path = _resolve_path(dataset_payload.get("DATA_PATH", ""), base=navgbench_root)
    scenes_dir = _resolve_path(dataset_payload.get("SCENES_DIR", ""), base=navgbench_root)
    dataset_config = _resolve_path(
        dataset_payload.get("DATASET_CONFIG", ""),
        base=navgbench_root,
    )
    with dataset_config.open("r", encoding="utf-8") as f:
        selection_payload = json.load(f)

    scene_to_traj_ids: Dict[str, set[str]] = {}
    for split in ("easy", "hard", "medium"):
        scenes = selection_payload.get(split) or {}
        if not isinstance(scenes, dict):
            continue
        for scene_name, traj_info in scenes.items():
            if not isinstance(traj_info, dict):
                continue
            traj_ids = {
                str(traj_id).strip()
                for traj_id in traj_info.keys()
                if str(traj_id).strip().isdigit()
            }
            if traj_ids:
                scene_to_traj_ids.setdefault(str(scene_name), set()).update(traj_ids)

    episodes = []
    global_episode_id = 1
    for scene_name in sorted(scene_to_traj_ids):
        scene_path = data_path / scene_name
        for traj_id in sorted(scene_to_traj_ids[scene_name], key=int):
            episode_json = scene_path / f"{traj_id}.json"
            if not episode_json.is_file():
                continue
            with episode_json.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            goal = payload.get("goal") or {}
            goal_world = goal.get("world") or {}
            goals = []
            if "x" in goal_world and "y" in goal_world:
                goals = [
                    SimpleNamespace(
                        position=[
                            float(goal_world["x"]),
                            float(goal_world["y"]),
                            1.3,
                        ]
                    )
                ]
            episodes.append(
                SimpleNamespace(
                    episode_id=str(global_episode_id),
                    scene_id=str(scenes_dir / scene_name),
                    ref_json=str(episode_json),
                    instruction=payload.get("instruction"),
                    moving_instruction=payload.get("moving_instruction"),
                    grounded_instruction=payload.get("grounded_instruction"),
                    path_info=payload.get("path"),
                    label_info=payload.get("label"),
                    goals=goals,
                )
            )
            global_episode_id += 1

    return SimpleNamespace(episodes=episodes)


def _load_spacevln_config(args: argparse.Namespace, *, results_dir: str) -> Any:
    config = get_spacevln_config(args.spacevln_config, [])
    config.defrost()
    config.PATHS.RESULTS_ROOT = resolve_results_root_path(args.results_root or "")
    config.PATHS.RESULTS_DIR = results_dir
    if args.max_steps is not None:
        config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = int(args.max_steps)

    simulator = config.TASK_CONFIG.SIMULATOR
    simulator.FORWARD_STEP_SIZE = 0.25
    # Keep the agent's model-facing action space at 30 degrees. The NavGBench
    # adapter expands one SpaceVLN turn into two 15-degree GN-Bench primitives.
    simulator.TURN_ANGLE = 30
    apply_output_policy_to_config(config, args)
    apply_runtime_derived_fields(config)
    config.freeze()
    return config


def _parse_id_list(raw: Optional[str]) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _select_episodes(episodes: Sequence[Any], args: argparse.Namespace) -> List[Any]:
    requested_ids = _parse_id_list(args.episode_ids)
    if args.episode_id:
        requested_ids.append(str(args.episode_id).strip())

    if requested_ids:
        wanted = set(requested_ids)
        selected = [
            ep
            for ep in episodes
            if str(ep.episode_id) in wanted or get_navgbench_episode_id(ep) in wanted
        ]
        missing = sorted(wanted - {str(ep.episode_id) for ep in selected} - {get_navgbench_episode_id(ep) for ep in selected})
        if missing:
            raise RuntimeError(f"NavGBench episode id(s) not found: {', '.join(missing)}")
        return selected

    if args.random:
        rng = random.Random(int(args.seed))
        pool = list(episodes)
        rng.shuffle(pool)
        return pool[: max(1, int(args.num_episodes))]

    start_sample = getattr(args, "start_sample", None)
    if start_sample is not None:
        start_idx = max(0, int(start_sample) - 1)
    else:
        start_idx = max(0, int(args.start_idx))
    if args.end_idx is not None and int(args.end_idx) >= 0:
        end_idx = int(args.end_idx)
    else:
        end_idx = start_idx + max(1, int(args.num_episodes))
    return list(episodes[start_idx:end_idx])


def _single_episode_dataset(base_dataset: Any, episode: Any) -> Any:
    dataset = copy.copy(base_dataset)
    dataset.episodes = [episode]
    return dataset


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else float(default)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _navgbench_log_rank_key(payload: Optional[Dict[str, Any]]) -> tuple:
    if not payload:
        return (-1, -1, float("-inf"), float("-inf"), float("-inf"))
    dtg = _as_float(payload.get("distance_to_goal", float("inf")), float("inf"))
    path_length = _as_float(payload.get("path_length", float("inf")), float("inf"))
    return (
        _as_int(payload.get("success", 0), 0),
        _as_int(payload.get("oracle_success", 0), 0),
        _as_float(payload.get("spl", 0.0), 0.0),
        -dtg,
        -path_length,
    )


def _is_better_navgbench_log(new_payload: Dict[str, Any], old_payload: Dict[str, Any]) -> bool:
    return _navgbench_log_rank_key(new_payload) > _navgbench_log_rank_key(old_payload)


def _save_navgbench_metrics(results_dir: str, episode: Any, metrics: Dict[str, Any]) -> str:
    stable_id = get_navgbench_episode_id(episode)
    log_dir = Path(results_dir) / "navgbench_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": stable_id,
        "episode_id": getattr(episode, "episode_id", ""),
        "distance_to_goal": metrics.get("distance_to_goal", -1.0),
        "success": metrics.get("success", 0),
        "spl": metrics.get("spl", 0.0),
        "path_length": metrics.get("path_length", 0.0),
        "oracle_success": metrics.get("oracle_success", 0),
    }
    save_path = log_dir / f"{stable_id}.json"
    existing_payload = load_json_if_exists(str(save_path))
    if not existing_payload or _is_better_navgbench_log(payload, existing_payload):
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
    return str(save_path)


def _storage_episode_id(episode: Any, index: int) -> int:
    raw_episode_id = str(getattr(episode, "episode_id", "") or "").strip()
    return int(raw_episode_id) if raw_episode_id.isdigit() else int(index)


def _sample_index_for_episode(episode: Any, fallback_index: int) -> int:
    return _storage_episode_id(episode, fallback_index)


def _navgbench_instruction_label(mode: str) -> str:
    return _results_instruction_dir_name(mode)


def _navgbench_console_prefix(
    *,
    index: int,
    total: int,
    sample_index: int,
    stable_id: str,
    worker_index: int = 0,
    worker_count: int = 0,
) -> str:
    order = (
        f"[W{worker_index}/{worker_count} {index}/{total}]"
        if worker_index > 0 and worker_count > 0
        else f"[{index}/{total}]"
    )
    return f"{order} Sample {int(sample_index)} | NavGBench {stable_id}"


def _format_navgbench_finish_line(
    *,
    prefix: str,
    success: bool,
    steps: int,
    metrics: Dict[str, Any],
    reason: str = "",
    error: str = "",
) -> str:
    parts = [
        f"{prefix} | {'OK' if success else 'FAIL'}",
        f"steps={int(steps or 0)}",
        f"SR={int(_as_int(metrics.get('success', 0), 0))}",
        f"DTG={_as_float(metrics.get('distance_to_goal', -1.0), -1.0):.3f}m",
        f"SPL={_as_float(metrics.get('spl', 0.0), 0.0):.4f}",
    ]
    reason_text = str(reason or "").strip()
    error_text = str(error or "").strip()
    if reason_text and not success and not should_suppress_normal_failure_reason(
        status="FAIL",
        reason=reason_text,
        error=error_text,
    ):
        parts.append(f"reason={reason_text}")
    if error_text:
        parts.append(f"error={error_text}")
    return " | ".join(parts)


def _episode_has_existing_sr1(results_dir: str, storage_episode_id: int) -> bool:
    existing = load_json_if_exists(get_episode_log_path(results_dir, int(storage_episode_id)))
    return SaveManager.result_has_complete_sr1(existing)


def _filter_existing_sr1(
    episodes: Sequence[Any],
    *,
    results_dir: str,
) -> List[Any]:
    kept: List[Any] = []
    skipped = 0
    for index, episode in enumerate(episodes, 1):
        storage_episode_id = _sample_index_for_episode(episode, index)
        if _episode_has_existing_sr1(results_dir, storage_episode_id):
            skipped += 1
            continue
        kept.append(episode)
    if skipped:
        print(
            f"NavGBench skip-sr1 skipped {skipped} existing successful episode(s); "
            f"remaining={len(kept)}",
            flush=True,
        )
    return kept


def _maybe_generate_report(results_dir: str) -> None:
    try:
        report_payload = generate_results_report(
            results_dir,
            save=True,
            debug=False,
            verbose=False,
        )
    except FileNotFoundError:
        return
    except Exception as exc:
        print(f"Failed to generate Navigation Agent report: {exc}")
        return

    metrics = dict(report_payload.get("metrics") or {})
    saved_paths = dict(report_payload.get("saved_paths") or {})
    total_episodes = int(metrics.get("total_episodes", 0) or 0)
    if total_episodes <= 0:
        return
    print(
        "\n📊 Evaluation Summary "
        f"| episodes={total_episodes} "
        f"| NE={_as_float(metrics.get('avg_ne', -1.0), -1.0):.3f}m "
        f"| OSR={_as_float(metrics.get('avg_osr', 0.0), 0.0):.3f} "
        f"| SR={_as_float(metrics.get('avg_sr', 0.0), 0.0):.3f} "
        f"| SPL={_as_float(metrics.get('avg_spl', 0.0), 0.0):.3f} "
        f"| nDTW={_as_float(metrics.get('avg_ndtw', 0.0), 0.0):.3f}"
    )
    timing = dict(metrics.get("timing") or {})
    if timing:
        print(
            "⏱️  Timing Summary "
            f"| episode_avg={_as_float(timing.get('episode_duration_s_avg', 0.0), 0.0):.2f}s "
            f"| api_total={_as_float(timing.get('api_total_duration_s', 0.0), 0.0):.2f}s"
        )
    summary_path = str(saved_paths.get("summary") or "").strip()
    csv_path = str(saved_paths.get("csv") or "").strip()
    if summary_path:
        print(f"📄 Summary file: {summary_path}")
    if csv_path:
        print(f"📄 Episode table: {csv_path}")


def _run_one_episode(
    *,
    index: int,
    total: int,
    episode: Any,
    gn_config: Any,
    base_dataset: Any,
    space_config: Any,
    args: argparse.Namespace,
    profile: NavigationRuntimeProfile,
    worker_index: int = 0,
    worker_count: int = 0,
) -> Dict[str, Any]:
    stable_id = get_navgbench_episode_id(episode)
    storage_episode_id = _sample_index_for_episode(episode, index)
    prefix = _navgbench_console_prefix(
        index=index,
        total=total,
        sample_index=storage_episode_id,
        stable_id=stable_id,
        worker_index=worker_index,
        worker_count=worker_count,
    )
    print(
        f"{prefix} | START",
        flush=True,
    )
    save_stdout_log = save_episode_stdout_log_enabled(space_config)
    episode_log_path = (
        get_episode_records_log_path(space_config.PATHS.RESULTS_DIR, storage_episode_id)
        if save_stdout_log
        else ""
    )
    redirect_context = (
        redirect_process_output_to_file(episode_log_path, mode="w")
        if save_stdout_log and episode_log_path
        else redirect_process_output_to_null()
    )

    env = None
    controller = None
    try:
        with redirect_context:
            from navigation_system.controller.agent.controller import NavigationAgentController

            if args.backend == "subprocess":
                sensor_cfg = space_config.SPACE.SENSOR
                depth_cfg = space_config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR
                env = NavGBenchSubprocessEnvClient(
                    python_bin=args.navgbench_python,
                    navgbench_root=args.gnbench_root,
                    gnbench_exp_config=args.gnbench_exp_config,
                    episode_key=stable_id,
                    sensor_config={
                        "frame_width": int(sensor_cfg.FRAME_WIDTH),
                        "frame_height": int(sensor_cfg.FRAME_HEIGHT),
                        "hfov_deg": float(sensor_cfg.HFOV_DEG),
                        "gnbench_sensor_fov_deg": _gnbench_sensor_fov_from_horizontal(
                            float(sensor_cfg.HFOV_DEG),
                            width=int(sensor_cfg.FRAME_WIDTH),
                            height=int(sensor_cfg.FRAME_HEIGHT),
                        ),
                        "agent_height_m": float(sensor_cfg.AGENT_HEIGHT_M),
                        "min_depth_m": float(depth_cfg.MIN_DEPTH),
                        "max_depth_m": float(depth_cfg.MAX_DEPTH),
                    },
                )
            else:
                from GN_Bench import Env

                episode_dataset = _single_episode_dataset(base_dataset, episode)
                env = Env(gn_config.TASK_CONFIG, episode_dataset)
            adapter = SingleNavGBenchVectorEnvAdapter(
                env,
                use_grounded_instruction=(
                    normalize_navgbench_instruction_mode(args.instruction_mode)
                    == "grounded"
                ),
                instruction_mode=args.instruction_mode,
                turn_repeat=2,
            )
            controller = NavigationAgentController(
                space_config,
                config_path=resolve_api_config_path(args.vlm_api_config),
                model_stack_builder=profile.model_stack_builder,
                envs=adapter,
            )
            controller.reset_episode(episode_id=storage_episode_id)
            controller.result_benchmark = "navgbench"
            controller.result_metadata = {
                "sample_index": storage_episode_id,
                "navgbench_id": stable_id,
            }
            result = controller.run_navigation(max_subtask_steps=args.max_subtask_steps)
            metrics = dict(env.get_metrics() or {})
            navgbench_log = _save_navgbench_metrics(space_config.PATHS.RESULTS_DIR, episode, metrics)
        success = int(metrics.get("success", 0) or 0) == 1
        steps = int(result.get("total_steps", result.get("steps", 0)) or 0)
        reason = str(result.get("reason", "") or "").strip()
        error = str(result.get("error", "") or "").strip()
        print(
            _format_navgbench_finish_line(
                prefix=prefix,
                success=success,
                steps=steps,
                metrics=metrics,
                reason=reason,
                error=error,
            ),
            flush=True,
        )
        return {
            "episode_id": storage_episode_id,
            "sample_index": storage_episode_id,
            "navgbench_id": stable_id,
            "success": success,
            "error": "",
            "reason": reason,
            "steps": steps,
            "distance_to_goal": _as_float(metrics.get("distance_to_goal", -1.0), -1.0),
            "spl": _as_float(metrics.get("spl", 0.0), 0.0),
            "oracle_success": _as_int(metrics.get("oracle_success", 0), 0),
            "result_file": result.get("result_file", ""),
            "navgbench_log": navgbench_log,
        }
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        error_msg = f"{type(exc).__name__}: {str(exc).strip()}"
        print(f"{prefix} | ERROR {error_msg}", flush=True)
        if save_stdout_log and episode_log_path:
            with redirect_process_output_to_file(episode_log_path, mode="a"):
                traceback.print_exc()
        return {
            "episode_id": storage_episode_id,
            "sample_index": storage_episode_id,
            "navgbench_id": stable_id,
            "success": False,
            "reason": "runtime_exception",
            "error": error_msg,
            "episode_log_path": episode_log_path if save_stdout_log else "",
        }
    finally:
        target = controller.envs if controller is not None else env
        if target is not None:
            try:
                target.close()
            except Exception:
                pass


def _find_episode_by_stable_id(episodes: Sequence[Any], stable_id: str) -> Any:
    wanted = str(stable_id).strip()
    for episode in episodes:
        if str(getattr(episode, "episode_id", "")) == wanted:
            return episode
        if get_navgbench_episode_id(episode) == wanted:
            return episode
    raise RuntimeError(f"NavGBench episode id not found in worker: {wanted}")


def _run_parallel_episode_job(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    os.chdir(_project_root())
    args = argparse.Namespace(**dict(job_spec.get("args") or {}))
    results_dir = str(job_spec.get("results_dir") or "")
    episode_key = str(job_spec.get("episode_key") or "")

    space_config = _load_spacevln_config(args, results_dir=results_dir)
    profile = _resolve_navgbench_runtime_profile(
        args.runtime,
        instruction_mode=args.instruction_mode,
    )

    navgbench_root = _resolve_path(args.gnbench_root, base=_workspace_root())
    with redirect_process_output_to_null():
        if args.backend == "in-process":
            gn_config, dataset = _load_gnbench_config_and_dataset(
                navgbench_root=navgbench_root,
                gnbench_exp_config=args.gnbench_exp_config,
                space_config=space_config,
            )
        else:
            gn_config = None
            dataset = _load_navgbench_metadata_dataset(
                navgbench_root=navgbench_root,
                gnbench_exp_config=args.gnbench_exp_config,
            )

    episode = _find_episode_by_stable_id(dataset.episodes, episode_key)
    return _run_one_episode(
        index=int(job_spec.get("index", 1)),
        total=int(job_spec.get("total", 1)),
        episode=episode,
        gn_config=gn_config,
        base_dataset=dataset,
        space_config=space_config,
        args=args,
        profile=profile,
        worker_index=int(job_spec.get("worker_index", 0) or 0),
        worker_count=int(job_spec.get("worker_count", 0) or 0),
    )


def _run_parallel_episodes(
    *,
    episodes: Sequence[Any],
    args: argparse.Namespace,
    results_dir: str,
) -> List[Dict[str, Any]]:
    worker_count = max(1, min(int(args.parallel_workers or 1), len(episodes)))
    print(
        f"NavGBench parallel execution enabled: workers={worker_count}, episodes={len(episodes)}",
        flush=True,
    )

    serializable_args = dict(vars(args))
    serializable_args["results_dir"] = results_dir
    mp_context = multiprocessing.get_context("spawn")
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
        initializer=_parallel_worker_initializer,
    )

    results_by_order: List[Optional[Dict[str, Any]]] = [None] * len(episodes)
    future_to_job: Dict[concurrent.futures.Future, Dict[str, Any]] = {}
    next_job_cursor = 0
    interrupted = False
    pool_broken_error = ""
    try:
        def _submit_next(worker_index: int) -> bool:
            nonlocal next_job_cursor
            if next_job_cursor >= len(episodes):
                return False
            episode = episodes[next_job_cursor]
            index = next_job_cursor + 1
            job_spec = {
                "args": serializable_args,
                "results_dir": results_dir,
                "episode_key": get_navgbench_episode_id(episode),
                "episode_id": getattr(episode, "episode_id", ""),
                "index": index,
                "total": len(episodes),
                "worker_index": int(worker_index),
                "worker_count": worker_count,
            }
            future = executor.submit(_run_parallel_episode_job, job_spec)
            future_to_job[future] = job_spec
            next_job_cursor += 1
            return True

        for worker_index in range(1, worker_count + 1):
            if not _submit_next(worker_index):
                break

        while future_to_job:
            done, _ = concurrent.futures.wait(
                future_to_job,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                job_spec = future_to_job.pop(future)
                episode_id = job_spec.get("episode_id", "?")
                stable_id = job_spec.get("episode_key", "?")
                order_index = int(job_spec.get("index", 1)) - 1
                try:
                    results_by_order[order_index] = future.result()
                except BrokenProcessPool as exc:
                    pool_broken_error = _format_exception_message(exc)
                    print(f"NavGBench parallel worker pool broke: {pool_broken_error}")
                    for pending in future_to_job.values():
                        pending_order_index = int(pending.get("index", 1)) - 1
                        results_by_order[pending_order_index] = (
                            {
                                "episode_id": pending.get("episode_id", "?"),
                                "sample_index": pending.get("episode_id", "?"),
                                "navgbench_id": pending.get("episode_key", "?"),
                                "success": False,
                                "reason": "parallel_worker_pool_broken",
                                "error": f"parallel worker pool broken: {pool_broken_error}",
                            }
                        )
                    future_to_job.clear()
                    break
                except BaseException as exc:
                    if isinstance(exc, KeyboardInterrupt):
                        interrupted = True
                        raise
                    error_msg = _format_exception_message(exc)
                    print(f"NavGBench {stable_id} ERROR {error_msg}", flush=True)
                    results_by_order[order_index] = {
                        "episode_id": episode_id,
                        "sample_index": episode_id,
                        "navgbench_id": stable_id,
                        "success": False,
                        "reason": "parallel_worker_failed",
                        "error": f"parallel worker failed: {error_msg}",
                    }
                if not pool_broken_error:
                    _submit_next(int(job_spec.get("worker_index", 1) or 1))
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        _shutdown_parallel_executor(executor, interrupted=interrupted)

    results = [result for result in results_by_order if result is not None]
    if pool_broken_error and not results:
        return [
            {
                "episode_id": getattr(episode, "episode_id", "?"),
                "sample_index": getattr(episode, "episode_id", "?"),
                "navgbench_id": get_navgbench_episode_id(episode),
                "success": False,
                "reason": "parallel_worker_pool_broken",
                "error": f"parallel worker pool broken: {pool_broken_error}",
            }
            for episode in episodes
        ]
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Navigation Agent on NavGBench.")
    parser.add_argument("--spacevln-config", default=DEFAULT_SPACEVLN_CONFIG)
    parser.add_argument("--gnbench-root", default=str(_default_navgbench_root()))
    parser.add_argument("--gnbench-exp-config", default=DEFAULT_GNBENCH_CONFIG)
    parser.add_argument("--navgbench-python", default=_default_navgbench_python())
    parser.add_argument(
        "--backend",
        choices=("auto", "subprocess", "in-process"),
        default=DEFAULT_BACKEND,
        help=(
            "auto uses in-process when the selected Python can import GN_Bench; "
            "otherwise it falls back to subprocess."
        ),
    )
    parser.add_argument("--vlm-api-config", "--config", dest="vlm_api_config", default=DEFAULT_API_CONFIG)
    parser.add_argument("--runtime", choices=RUNTIME_CHOICES, default=DEFAULT_RUNTIME)

    parser.add_argument(
        "--start-sample",
        type=int,
        default=None,
        help="1-based NavGBench sample index (recommended). 0 is accepted as sample 1.",
    )
    parser.add_argument("--start-idx", type=int, default=0, help="legacy 0-based NavGBench dataset index")
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-id", type=str, default="")
    parser.add_argument("--episode-ids", type=str, default="")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-subtask-steps", type=int, default=5)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of parallel episode workers (1 means serial execution).",
    )
    parser.add_argument(
        "--skip-sr1",
        "--skip-existing-sr1",
        action="store_true",
        dest="skip_sr1",
        help="Skip selected episodes that already have complete SR=1 best logs.",
    )
    parser.add_argument("--results-root", default="")
    parser.add_argument("--results-dir", default="")
    add_output_profile_arg(parser)
    add_output_artifact_args(parser)
    parser.add_argument(
        "--instruction-mode",
        choices=("complex", "simple", "grounded", "raw", "moving"),
        default=DEFAULT_INSTRUCTION_MODE,
        help=(
            "NavGBench task text: complex/grounded=landmark-rich route (default), "
            "simple/raw=short object/room goal, moving=metric turn-by-turn route."
        ),
    )
    parser.add_argument(
        "--complex-instruction",
        action="store_const",
        const="complex",
        dest="instruction_mode",
        help="Use the landmark-rich NavGBench instruction.",
    )
    parser.add_argument(
        "--simple-instruction",
        action="store_const",
        const="simple",
        dest="instruction_mode",
        help="Use the short object/room goal instruction.",
    )
    parser.add_argument("--use-raw-instruction", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    return parser


def run_navigation_from_args(args: argparse.Namespace) -> int:
    os.chdir(_project_root())
    navgbench_root = _resolve_path(args.gnbench_root, base=_workspace_root())
    if not navgbench_root.is_dir():
        raise FileNotFoundError(f"NavGBench root does not exist: {navgbench_root}")

    args.vlm_api_config = resolve_api_config_path(args.vlm_api_config)
    if str(args.runtime or "").strip().lower() == "context_cache":
        validate_qwen_context_cache_api_config(args.vlm_api_config)
    if bool(args.use_raw_instruction):
        args.instruction_mode = "simple"
    args.instruction_mode = normalize_navgbench_instruction_mode(args.instruction_mode)

    resolved_results_dir = resolve_results_dir_path(args.results_dir)
    if not resolved_results_dir:
        resolved_results_dir = _default_results_dir(
            args.vlm_api_config,
            args.results_root,
            runtime=args.runtime,
            instruction_mode=args.instruction_mode,
        )
    os.makedirs(resolved_results_dir, exist_ok=True)
    args.results_dir = resolved_results_dir

    space_config = _load_spacevln_config(args, results_dir=resolved_results_dir)
    args.gnbench_root = str(navgbench_root)
    args.navgbench_python = str(Path(args.navgbench_python).expanduser())
    args.parallel_workers = max(1, int(args.parallel_workers or 1))

    if args.backend == "auto":
        with redirect_process_output_to_null():
            args.backend = (
                "in-process"
                if _can_use_in_process_backend(navgbench_root)
                else "subprocess"
            )

    if args.backend == "in-process":
        try:
            with redirect_process_output_to_null():
                gn_config, dataset = _load_gnbench_config_and_dataset(
                    navgbench_root=navgbench_root,
                    gnbench_exp_config=args.gnbench_exp_config,
                    space_config=space_config,
                )
        except RuntimeError as exc:
            if not args.dry_run:
                raise
            print(str(exc))
            print("Dry-run fallback: reading NavGBench episode metadata directly.")
            gn_config = None
            with redirect_process_output_to_null():
                dataset = _load_navgbench_metadata_dataset(
                    navgbench_root=navgbench_root,
                    gnbench_exp_config=args.gnbench_exp_config,
                )
    else:
        if not Path(args.navgbench_python).is_file():
            raise FileNotFoundError(
                f"NavGBench Python does not exist: {args.navgbench_python}"
            )
        with redirect_process_output_to_null():
            dataset = _load_navgbench_metadata_dataset(
                navgbench_root=navgbench_root,
                gnbench_exp_config=args.gnbench_exp_config,
            )
        gn_config = None
    episodes = _select_episodes(dataset.episodes, args)
    if args.skip_sr1:
        episodes = _filter_existing_sr1(
            episodes,
            results_dir=space_config.PATHS.RESULTS_DIR,
        )
    if not episodes:
        if args.skip_sr1:
            print("No NavGBench episodes need to run: selected episodes already have SR=1 best logs.")
            if not args.no_report:
                _maybe_generate_report(space_config.PATHS.RESULTS_DIR)
            return 0
        print("No NavGBench episodes selected.")
        return 1

    print(f"NavGBench root: {navgbench_root}")
    print(f"NavGBench backend: {args.backend}")
    if args.backend == "subprocess":
        print(f"NavGBench Python: {args.navgbench_python}")
    print(f"Instruction: {_navgbench_instruction_label(args.instruction_mode)}")
    print(f"Selected samples: {len(episodes)}")
    print(f"Results dir: {space_config.PATHS.RESULTS_DIR}")
    for idx, episode in enumerate(episodes[:10], 1):
        sample_index = _sample_index_for_episode(episode, idx)
        print(
            f"  {idx}. Sample {sample_index} | NavGBench {get_navgbench_episode_id(episode)}"
        )
    if len(episodes) > 10:
        print(f"  ... {len(episodes) - 10} more")
    if args.dry_run:
        return 0

    if not os.path.exists(args.vlm_api_config):
        raise FileNotFoundError(f"API config does not exist: {args.vlm_api_config}")

    profile = _resolve_navgbench_runtime_profile(
        args.runtime,
        instruction_mode=args.instruction_mode,
    )
    if args.parallel_workers > 1 and len(episodes) > 1:
        results = _run_parallel_episodes(
            episodes=episodes,
            args=args,
            results_dir=resolved_results_dir,
        )
    else:
        results = []
        for index, episode in enumerate(episodes, 1):
            results.append(
                _run_one_episode(
                    index=index,
                    total=len(episodes),
                    episode=episode,
                    gn_config=gn_config,
                    base_dataset=dataset,
                    space_config=space_config,
                    args=args,
                    profile=profile,
                )
            )

    failed = [item for item in results if is_abnormal_episode_failure(item)]
    if failed:
        reason_counts: Dict[str, int] = {}
        for item in failed:
            reason = str(item.get("reason") or "").strip()
            error = str(item.get("error") or "").strip()
            key = reason or ("runtime_error" if error else "unknown")
            reason_counts[key] = reason_counts.get(key, 0) + 1
        reason_text = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(reason_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        )
        print(f"\n⚠️ NavGBench failures by reason: {reason_text}", flush=True)
    if not args.no_report:
        _maybe_generate_report(space_config.PATHS.RESULTS_DIR)

    return 1 if failed else 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run_navigation_from_args(args)
    except RuntimeError as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
