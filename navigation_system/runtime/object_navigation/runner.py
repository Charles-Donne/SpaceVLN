"""Run SpaceVLN's prompt-based controller on OVON episodes."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import json
import multiprocessing
import os
import random
import shutil
import signal
import sys
from pathlib import Path
from typing import List, Sequence

import yaml
from navigation_system.runtime.object_navigation.thresholds import (
    OVON_SUCCESS_DISTANCE_M,
)
from navigation_system.runtime.episode_io import load_json_if_exists
from navigation_system.runtime.episode_io import (
    build_episode_console_summary,
    build_episode_start_summary,
    get_episode_records_log_path,
    redirect_process_output_to_file,
    redirect_process_output_to_null,
    save_episode_stdout_log_enabled,
)
from navigation_system.runtime.storage.results_layout import (
    build_default_results_family_root,
    build_model_results_dir_name,
    resolve_api_config_path,
    resolve_results_root_path,
)
from navigation_system.runtime.storage.artifacts import (
    get_episode_detail_path_candidates,
    get_episode_log_path_candidates,
)


RUNTIME_CHOICES = ("standard", "context_cache")


def _format_exception_message(exc: BaseException) -> str:
    exc_type = type(exc).__name__
    exc_text = str(exc).strip()
    return f"{exc_type}: {exc_text}" if exc_text else exc_type


def _parallel_worker_initializer() -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
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


def _nav_ws_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_run_config() -> str:
    return str(
        (
            _nav_ws_root()
            / "SpaceVLN"
            / "navigation_system"
            / "config"
            / "experiments"
            / "ovon_val_unseen_eval.yaml"
        ).resolve()
    )


def _load_run_defaults(config_path: str | None) -> dict:
    resolved_path = str(config_path or _default_run_config()).strip()
    if not resolved_path:
        return {}
    path = Path(resolved_path).resolve()
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        return {}
    block = payload.get("OVON") or payload
    return dict(block) if isinstance(block, dict) else {}


def _resolve_nav_ws_path(path_like: str | Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate.resolve()
    return (_nav_ws_root() / candidate).resolve()


def _build_parser_defaults(run_defaults: dict | None = None) -> dict:
    defaults = dict(run_defaults or {})
    split = str(defaults.get("split", "val_unseen")).strip() or "val_unseen"
    runtime = str(defaults.get("runtime", "context_cache")).strip().lower() or "context_cache"
    return {
        "runtime": runtime if runtime in RUNTIME_CHOICES else "context_cache",
        "exp_config": str(
            _resolve_nav_ws_path(
                defaults.get(
                    "official_exp_config",
                    _nav_ws_root() / "ovon" / "config" / "experiments" / "transformer_dagger.yaml",
                )
            )
        ),
        "data_path": str(
            _resolve_nav_ws_path(
                defaults.get(
                    "data_path",
                    _nav_ws_root()
                    / "data"
                    / "datasets"
                    / "ovon"
                    / "hm3d"
                    / "v1"
                    / split
                    / (
                        "val_unseen_hard.json.gz"
                        if split == "val_unseen"
                        else f"{split}.json.gz"
                    ),
                )
            )
        ),
        "split": split,
        "gpu_id": int(defaults.get("gpu_id", 0) or 0),
        "max_steps": int(defaults.get("max_steps", 500) or 500),
        "max_subtask_steps": int(defaults.get("max_subtask_steps", 4) or 4),
        "save_step_images": bool(defaults.get("save_step_images", True)),
        "save_gif": bool(defaults.get("save_gif", False)),
    }


def _unique_path_candidates(paths: Sequence[Path]) -> List[Path]:
    unique: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(Path(path).resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(Path(normalized))
    return unique


def _build_ovon_dataset_path_candidates(
    *,
    split: str,
    configured_path: str | None = None,
) -> List[Path]:
    normalized_split = str(split or "val_unseen").strip() or "val_unseen"
    default_root = _nav_ws_root() / "data" / "datasets" / "ovon" / "hm3d" / "v1" / normalized_split
    ovon_repo_root = _nav_ws_root() / "ovon" / "data" / "datasets" / "ovon" / "hm3d" / "v1" / normalized_split

    default_names: List[str] = []
    if normalized_split == "val_unseen":
        default_names.extend([
            "val_unseen_hard.json.gz",
            "val_unseen.json.gz",
            "val_unseen_easy.json.gz",
        ])
    else:
        default_names.append(f"{normalized_split}.json.gz")

    candidates: List[Path] = []
    if str(configured_path or "").strip():
        resolved_configured = _resolve_nav_ws_path(str(configured_path))
        candidates.append(resolved_configured)
        parent_dir = resolved_configured.parent
        for file_name in default_names:
            candidates.append(parent_dir / file_name)

    for base_dir in (default_root, ovon_repo_root):
        for file_name in default_names:
            candidates.append(base_dir / file_name)
        if base_dir.is_dir():
            candidates.extend(sorted(base_dir.glob("*.json.gz")))

    return _unique_path_candidates(candidates)


def _resolve_ovon_dataset_path(
    *,
    split: str,
    configured_path: str | None = None,
) -> Path:
    candidates = _build_ovon_dataset_path_candidates(
        split=split,
        configured_path=configured_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried_lines = "\n".join(f"  - {candidate}" for candidate in candidates[:12])
    raise RuntimeError(
        "OVON dataset file not found.\n"
        f"Requested split: '{split}'.\n"
        f"Tried:\n{tried_lines}\n"
        "Install or symlink the OVON dataset under "
        "`data/datasets/ovon/hm3d/v1/<split>/`, or pass the exact file with `--data-path`."
    )


def _default_results_dir() -> str:
    return str((Path(build_default_results_family_root("ovon")) / "ovon_smoke").resolve())


def _default_api_config(runtime: str) -> str:
    runtime_name = str(runtime or "context_cache").strip().lower()
    del runtime_name
    file_name = "vlm_api_config.yaml"
    return str(
        (
            _nav_ws_root()
            / "SpaceVLN"
            / "navigation_system"
            / "config"
            / "vlm"
            / file_name
        ).resolve()
    )


def _default_results_dir_for_runtime(
    runtime: str,
    api_config_path: str,
    *,
    results_root: str | None = None,
) -> str:
    runtime_name = str(runtime or "context_cache").strip().lower()
    suffix = "_cache" if runtime_name == "context_cache" else ""
    model_dir = build_model_results_dir_name(api_config_path)
    return str(
        (
            Path(build_default_results_family_root("ovon", results_root=results_root))
            / f"{model_dir}{suffix}"
        ).resolve()
    )


def _select_model_stack_builder(runtime: str):
    from navigation_system.vlm.object_navigation.runtime_factory import (
        build_ovon_context_cache_navigation_model_stack,
        build_ovon_navigation_model_stack,
    )

    return (
        build_ovon_context_cache_navigation_model_stack
        if str(runtime or "").strip().lower() == "context_cache"
        else build_ovon_navigation_model_stack
    )


def _parse_episode_ids(raw: str | None) -> List[int]:
    if not raw:
        return []
    result: List[int] = []
    for piece in str(raw).split(","):
        text = piece.strip()
        if not text:
            continue
        result.append(int(text))
    return result


def _prepare_ovon_config(
    *,
    exp_config: str,
    split: str,
    data_path: str,
    gpu_id: int,
    max_steps: int,
):
    import contextlib
    import io

    ovon_repo = str((_nav_ws_root() / "ovon").resolve())
    ovon_habitat_lab = str(
        (_nav_ws_root() / "ovon" / "habitat-lab-v0.2.3" / "habitat-lab").resolve()
    )
    for source_path in (ovon_habitat_lab, ovon_repo):
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import habitat
        from habitat.config import read_write
        from habitat.config.default_structured_configs import (
            HabitatSimDepthSensorConfig,
            TopDownMapMeasurementConfig,
            register_hydra_plugin,
        )
        from navigation_system.runtime.object_navigation.visualization_patch import (
            install_ovon_topdown_visualization_patch,
        )

    ovon_repo_root = (_nav_ws_root() / "ovon").resolve()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from ovon.config import HabitatConfigPlugin
        install_ovon_topdown_visualization_patch()

    register_hydra_plugin(HabitatConfigPlugin)
    config = habitat.get_config(str(Path(exp_config).resolve()))

    resolved_data_path = _resolve_ovon_dataset_path(
        split=split,
        configured_path=data_path,
    )

    with read_write(config):
        config.habitat.dataset.split = str(split)
        config.habitat.dataset.data_path = str(resolved_data_path)
        config.habitat.environment.max_episode_steps = int(max_steps)
        config.habitat.simulator.turn_angle = 30
        config.habitat.simulator.scene_dataset = str(
            (_nav_ws_root() / "data" / "scene_datasets" / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json").resolve()
        )
        if hasattr(config.habitat.task.measurements, "success"):
            config.habitat.task.measurements.success.success_distance = float(
                OVON_SUCCESS_DISTANCE_M
            )
        if hasattr(config.habitat.task.lab_sensors, "objnav_explorer"):
            config.habitat.task.lab_sensors.objnav_explorer.success_distance = float(
                OVON_SUCCESS_DISTANCE_M
            )

        for sensor_name, sensor_cfg in config.habitat.task.lab_sensors.items():
            cache_path = getattr(sensor_cfg, "cache", None)
            if isinstance(cache_path, str) and cache_path and not os.path.isabs(cache_path):
                sensor_cfg.cache = str((ovon_repo_root / cache_path).resolve())

        agent = config.habitat.simulator.agents.main_agent
        agent.height = 0.88
        agent.radius = 0.18
        agent.sim_sensors["rgb_sensor"].width = 640
        agent.sim_sensors["rgb_sensor"].height = 480
        agent.sim_sensors["rgb_sensor"].hfov = 79
        agent.sim_sensors["rgb_sensor"].position = [0, 0.88, 0]
        if "depth_sensor" not in agent.sim_sensors:
            agent.sim_sensors["depth_sensor"] = HabitatSimDepthSensorConfig(
                width=640,
                height=480,
                hfov=79,
                min_depth=0.5,
                max_depth=5.0,
                position=[0, 0.88, 0],
            )
        else:
            agent.sim_sensors["depth_sensor"].width = 640
            agent.sim_sensors["depth_sensor"].height = 480
            agent.sim_sensors["depth_sensor"].hfov = 79
            agent.sim_sensors["depth_sensor"].min_depth = 0.5
            agent.sim_sensors["depth_sensor"].max_depth = 5.0
            agent.sim_sensors["depth_sensor"].position = [0, 0.88, 0]

        config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(gpu_id)

        if not hasattr(config.habitat.task.measurements, "top_down_map"):
            config.habitat.task.measurements["top_down_map"] = TopDownMapMeasurementConfig()
        config.habitat.task.measurements.top_down_map.draw_goal_aabbs = True
        config.habitat.task.measurements.top_down_map.draw_shortest_path = True
        config.habitat.task.measurements.top_down_map.draw_goal_positions = True
        config.habitat.task.measurements.top_down_map.draw_view_points = True

        if hasattr(config.habitat.task.measurements, "frontier_exploration_map"):
            config.habitat.task.measurements.pop("frontier_exploration_map")
        if hasattr(config.habitat.task.lab_sensors, "objnav_explorer"):
            config.habitat.task.lab_sensors.pop("objnav_explorer")
        if hasattr(config.habitat_baselines.rl.policy, "obs_transforms") and hasattr(
            config.habitat_baselines.rl.policy.obs_transforms,
            "relabel_teacher_actions",
        ):
            config.habitat_baselines.rl.policy.obs_transforms.pop(
                "relabel_teacher_actions"
            )

    return config


def _load_dataset_for_episodes(config, episode_ids: Sequence[int] | None):
    import habitat

    dataset = habitat.make_dataset(
        config.habitat.dataset.type,
        config=config.habitat.dataset,
    )

    episodes = list(getattr(dataset, "episodes", []) or [])
    ovon_repo_root = (_nav_ws_root() / "ovon").resolve()
    for episode in episodes:
        scene_id = str(getattr(episode, "scene_id", "") or "").strip()
        if scene_id and not os.path.isabs(scene_id):
            episode.scene_id = str((ovon_repo_root / scene_id).resolve())

    if episode_ids:
        wanted = {int(ep_id) for ep_id in episode_ids}
        episodes = [episode for episode in episodes if int(episode.episode_id) in wanted]

    dataset.episodes = episodes
    return dataset


def _load_dataset_for_sample_indices(config, sample_indices: Sequence[int] | None):
    dataset = _load_dataset_for_episodes(config, None)
    if not sample_indices:
        return dataset
    episodes = list(getattr(dataset, "episodes", []) or [])
    selected = []
    for raw_index in sample_indices:
        sample_index = int(raw_index)
        if sample_index < 1 or sample_index > len(episodes):
            continue
        selected.append(episodes[sample_index - 1])
    dataset.episodes = selected
    return dataset


def _index_dataset_episodes(dataset) -> dict[int, list]:
    episode_lookup: dict[int, list] = {}
    for episode in list(getattr(dataset, "episodes", []) or []):
        episode_lookup.setdefault(int(episode.episode_id), []).append(episode)
    return episode_lookup


def _build_split_index_lookup(dataset) -> dict[int, int]:
    split_index_lookup: dict[int, int] = {}
    for sample_index, episode in enumerate(list(getattr(dataset, "episodes", []) or []), 1):
        episode_id = int(getattr(episode, "episode_id"))
        split_index_lookup.setdefault(episode_id, int(sample_index))
    return split_index_lookup


def _clone_dataset_with_episode(base_dataset, episode):
    dataset = copy.copy(base_dataset)
    dataset.episodes = [episode]
    return dataset


def _sample_indices_from_range(
    dataset,
    *,
    start_index: int | None,
    num_episodes: int,
) -> List[int]:
    episodes = list(getattr(dataset, "episodes", []) or [])
    if not episodes:
        return []
    requested_index = max(1, int(start_index or 1))
    start_offset = requested_index - 1
    if start_offset >= len(episodes):
        return []
    end_offset = min(len(episodes), start_offset + max(1, int(num_episodes)))
    return list(range(start_offset + 1, end_offset + 1))


def _random_sample_indices(
    dataset,
    *,
    num_episodes: int,
    seed: int,
) -> List[int]:
    total = len(list(getattr(dataset, "episodes", []) or []))
    if total <= 0:
        return []
    sample_size = min(max(1, int(num_episodes)), total)
    return random.Random(int(seed)).sample(list(range(1, total + 1)), sample_size)


def _episode_by_sample_index(dataset, sample_index: int):
    episodes = list(getattr(dataset, "episodes", []) or [])
    index = int(sample_index)
    if index < 1 or index > len(episodes):
        return None
    return episodes[index - 1]


def _build_sample_episode_specs(dataset, sample_indices: Sequence[int]) -> List[dict]:
    specs: List[dict] = []
    for sample_index in sample_indices:
        episode = _episode_by_sample_index(dataset, int(sample_index))
        if episode is None:
            continue
        specs.append(
            {
                "episode_id": int(getattr(episode, "episode_id")),
                "sample_index": int(sample_index),
                "storage_entry_id": int(sample_index),
                "entry_kind": "sample",
            }
        )
    return specs


def _build_episode_id_specs(dataset, episode_ids: Sequence[int]) -> List[dict]:
    episode_lookup = _index_dataset_episodes(dataset)
    split_index_lookup = _build_split_index_lookup(dataset)
    specs: List[dict] = []
    for episode_id in episode_ids:
        selected_episode_id = int(episode_id)
        if selected_episode_id not in episode_lookup:
            continue
        sample_index = split_index_lookup.get(selected_episode_id)
        specs.append(
            {
                "episode_id": selected_episode_id,
                "sample_index": sample_index,
                "storage_entry_id": (
                    int(sample_index) if sample_index is not None else selected_episode_id
                ),
                "entry_kind": "sample" if sample_index is not None else "episode",
            }
        )
    return specs


def _discover_episode_ids(dataset, *, num_episodes: int) -> List[int]:
    return _discover_episode_ids_from_start(
        dataset,
        episode_id=None,
        num_episodes=num_episodes,
    )


def _discover_episode_ids_from_index(
    dataset,
    *,
    start_index: int | None,
    num_episodes: int,
) -> List[int]:
    episodes = list(getattr(dataset, "episodes", []) or [])
    if not episodes:
        return []

    requested_index = int(start_index or 1)
    start_offset = max(0, requested_index - 1)
    if start_offset >= len(episodes):
        return []

    selected = episodes[start_offset : start_offset + max(1, int(num_episodes))]
    return [int(ep.episode_id) for ep in selected]


def _discover_episode_ids_from_start(
    dataset,
    *,
    episode_id: int | None,
    num_episodes: int,
) -> List[int]:
    episodes = list(getattr(dataset, "episodes", []) or [])
    ids = sorted({int(ep.episode_id) for ep in episodes})
    if episode_id is not None:
        start_id = int(episode_id)
        ids = [candidate_id for candidate_id in ids if candidate_id >= start_id]
    return ids[: max(1, int(num_episodes))]


def _discover_random_episode_ids(
    dataset,
    *,
    num_episodes: int,
    seed: int,
) -> List[int]:
    ids = sorted({int(ep.episode_id) for ep in list(getattr(dataset, "episodes", []) or [])})
    if not ids:
        return []
    sample_size = min(max(1, int(num_episodes)), len(ids))
    return random.Random(int(seed)).sample(ids, sample_size)


def _result_payload_is_sr1(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in ("sr", "success"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return bool(int(value))
        except Exception:
            return bool(value)
    return False


def _episode_has_existing_sr1(
    results_dir: str,
    episode_id: int,
    *,
    entry_kind: str = "episode",
) -> bool:
    candidate_paths = []
    candidate_paths.extend(
        get_episode_log_path_candidates(
            results_dir,
            episode_id,
            entry_kind=entry_kind,
        )
    )
    for detail_dir in get_episode_detail_path_candidates(
        results_dir,
        episode_id,
        entry_kind=entry_kind,
    ):
        candidate_paths.append(os.path.join(detail_dir, "records", "result.json"))

    seen = set()
    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)
        if _result_payload_is_sr1(load_json_if_exists(path)):
            return True
    return False


def _filter_existing_sr1(
    *,
    episode_ids: Sequence[int],
    results_dir: str,
) -> List[int]:
    kept = []
    skipped = []
    for episode_id in episode_ids:
        if _episode_has_existing_sr1(results_dir, int(episode_id)):
            skipped.append(int(episode_id))
        else:
            kept.append(int(episode_id))
    if skipped:
        preview = ",".join(str(item) for item in skipped[:20])
        suffix = "..." if len(skipped) > 20 else ""
        print(
            f"[OVON-ObjectNav] skip-sr1 skipped {len(skipped)} existing "
            f"successful episodes: {preview}{suffix}"
        )
    return kept


def _filter_existing_sr1_specs(
    *,
    episode_specs: Sequence[dict],
    results_dir: str,
) -> List[dict]:
    kept: List[dict] = []
    skipped: List[str] = []
    for spec in episode_specs:
        storage_entry_id = int(spec.get("storage_entry_id", spec.get("episode_id", 0)) or 0)
        entry_kind = str(spec.get("entry_kind", "episode") or "episode")
        if _episode_has_existing_sr1(
            results_dir,
            storage_entry_id,
            entry_kind=entry_kind,
        ):
            skipped.append(f"{entry_kind}_{storage_entry_id}")
        else:
            kept.append(dict(spec))
    if skipped:
        preview = ",".join(skipped[:20])
        suffix = "..." if len(skipped) > 20 else ""
        print(
            f"[OVON-ObjectNav] skip-sr1 skipped {len(skipped)} existing "
            f"successful samples: {preview}{suffix}"
        )
    return kept


def _prune_empty_parents(path: str | Path, *, stop_at: str | Path) -> None:
    current = Path(path)
    stop = Path(stop_at)
    try:
        current = current.resolve()
    except FileNotFoundError:
        current = current.absolute()
    try:
        stop = stop.resolve()
    except FileNotFoundError:
        stop = stop.absolute()

    while current != stop and stop in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _cleanup_entry_artifacts(
    *,
    results_dir: str,
    storage_entry_id: int,
    entry_kind: str,
) -> None:
    detail_root = Path(results_dir) / "detail"
    log_root = Path(results_dir) / "log"

    for detail_dir in get_episode_detail_path_candidates(
        results_dir,
        int(storage_entry_id),
        entry_kind=entry_kind,
    ):
        detail_path = Path(detail_dir)
        if detail_path.exists():
            shutil.rmtree(detail_path, ignore_errors=True)
            _prune_empty_parents(detail_path.parent, stop_at=detail_root)

    for log_path in get_episode_log_path_candidates(
        results_dir,
        int(storage_entry_id),
        entry_kind=entry_kind,
    ):
        candidate = Path(log_path)
        if candidate.exists():
            try:
                candidate.unlink()
            except IsADirectoryError:
                shutil.rmtree(candidate, ignore_errors=True)
            _prune_empty_parents(candidate.parent, stop_at=log_root)


def _cleanup_entry_log_artifacts(
    *,
    results_dir: str,
    storage_entry_id: int,
    entry_kind: str,
) -> None:
    log_root = Path(results_dir) / "log"
    for log_path in get_episode_log_path_candidates(
        results_dir,
        int(storage_entry_id),
        entry_kind=entry_kind,
    ):
        candidate = Path(log_path)
        if candidate.exists():
            try:
                candidate.unlink()
            except IsADirectoryError:
                shutil.rmtree(candidate, ignore_errors=True)
            _prune_empty_parents(candidate.parent, stop_at=log_root)


def _cleanup_interrupted_specs(*, results_dir: str, episode_specs: Sequence[dict]) -> None:
    seen: set[tuple[str, int]] = set()
    for spec in episode_specs:
        storage_entry_id = int(spec.get("storage_entry_id", spec.get("episode_id", 0)) or 0)
        entry_kind = str(spec.get("entry_kind", "episode") or "episode")
        key = (entry_kind, storage_entry_id)
        if key in seen:
            continue
        seen.add(key)
        _cleanup_entry_log_artifacts(
            results_dir=results_dir,
            storage_entry_id=storage_entry_id,
            entry_kind=entry_kind,
        )


def _is_abnormal_failure(result: dict | None) -> bool:
    payload = dict(result or {})
    if bool(payload.get("success", False)):
        return False

    error_text = str(payload.get("error", "") or "").strip()
    if error_text:
        return True

    reason = str(payload.get("reason", "") or "").strip().lower()
    abnormal_reasons = {
        "runtime_exception",
        "parallel_worker_failed",
        "interrupted",
        "incomplete",
    }
    return reason in abnormal_reasons


def _cleanup_failed_artifacts(
    *,
    results_dir: str,
    storage_entry_id: int,
    entry_kind: str,
    result: dict,
) -> dict:
    cleaned = dict(result or {})
    if bool(cleaned.get("success", False)):
        return cleaned
    if not _is_abnormal_failure(cleaned):
        return cleaned

    _cleanup_entry_log_artifacts(
        results_dir=results_dir,
        storage_entry_id=int(storage_entry_id),
        entry_kind=str(entry_kind or "episode"),
    )
    return cleaned


def _build_episode_summary_row(
    episode_id: int,
    result: dict,
    *,
    sample_index: int | None = None,
) -> dict:
    saved_metrics = _resolve_saved_metrics(result)
    return {
        "sample_index": int(sample_index) if sample_index is not None else None,
        "episode_id": int(episode_id),
        "success": bool(
            _coalesce_metric(
                saved_metrics,
                "sr",
                result,
                "success",
                "sr",
                default=False,
            )
        ),
        "steps": int(result.get("total_steps", result.get("steps", 0)) or 0),
        "distance_to_goal": float(
            _coalesce_metric(
                saved_metrics,
                "ne",
                result,
                "distance_to_goal",
                "ne",
                default=-1.0,
            )
            or -1.0
        ),
        "spl": float(
            _coalesce_metric(
                saved_metrics,
                "spl",
                result,
                "spl",
                default=0.0,
            )
            or 0.0
        ),
        "soft_spl": float(
            _coalesce_metric(
                saved_metrics,
                "soft_spl",
                result,
                "soft_spl",
                "oracle_spl",
                default=saved_metrics.get("oracle_spl", 0.0),
            )
            or 0.0
        ),
        "gif_path": str(result.get("gif_path", "") or ""),
        "topdown_path": str(result.get("topdown_path", "") or ""),
        "result_file": str(result.get("result_file", "") or ""),
        "reason": str(result.get("reason", "") or ""),
        "error": str(result.get("error", "") or ""),
    }


def _build_console_metrics(summary_row: dict) -> dict:
    return {
        "sr": int(bool(summary_row.get("success", False))),
        "dtg": float(summary_row.get("distance_to_goal", -1.0) or -1.0),
        "spl": float(summary_row.get("spl", 0.0) or 0.0),
        "soft_spl": float(summary_row.get("soft_spl", 0.0) or 0.0),
    }


def _build_parallel_episode_spec(
    *,
    args: argparse.Namespace,
    episode_spec: dict,
    index: int,
    total: int,
    worker_index: int,
    worker_count: int,
) -> dict:
    return {
        "runtime": str(args.runtime),
        "exp_config": str(args.exp_config),
        "split": str(args.split),
        "data_path": str(args.data_path),
        "gpu_id": int(args.gpu_id),
        "max_steps": int(args.max_steps),
        "max_subtask_steps": int(args.max_subtask_steps),
        "results_dir": str(args.results_dir),
        "vlm_api_config": str(args.vlm_api_config),
        "save_step_images": bool(args.save_step_images),
        "save_gif": bool(args.save_gif),
        "episode_id": int(episode_spec["episode_id"]),
        "sample_index": (
            int(episode_spec["sample_index"])
            if episode_spec.get("sample_index") is not None
            else None
        ),
        "storage_entry_id": int(
            episode_spec.get("storage_entry_id", episode_spec["episode_id"])
        ),
        "entry_kind": str(episode_spec.get("entry_kind", "episode") or "episode"),
        "index": int(index),
        "total": int(total),
        "worker_index": int(worker_index),
        "worker_count": int(worker_count),
    }


def _run_parallel_episode_job(job_spec: dict) -> dict:
    episode_id = int(job_spec["episode_id"])
    raw_sample_index = job_spec.get("sample_index")
    sample_index = int(raw_sample_index) if raw_sample_index is not None else None
    storage_entry_id = int(job_spec.get("storage_entry_id", episode_id) or episode_id)
    entry_kind = str(job_spec.get("entry_kind", "episode") or "episode")
    index = int(job_spec["index"])
    total = int(job_spec["total"])
    worker_index = int(job_spec["worker_index"])
    worker_count = int(job_spec["worker_count"])

    print(
        build_episode_start_summary(
            episode_id=episode_id,
            sample_index=sample_index,
            index=index,
            total=total,
            worker_index=worker_index,
            worker_count=worker_count,
        ),
        flush=True,
    )

    with redirect_process_output_to_null():
        ovon_config = _prepare_ovon_config(
            exp_config=str(job_spec["exp_config"]),
            split=str(job_spec["split"]),
            data_path=str(job_spec["data_path"]),
            gpu_id=int(job_spec["gpu_id"]),
            max_steps=int(job_spec["max_steps"]),
        )
        if sample_index is not None and entry_kind == "sample":
            discovery_dataset = _load_dataset_for_sample_indices(ovon_config, [sample_index])
        else:
            discovery_dataset = _load_dataset_for_episodes(ovon_config, [episode_id])
    episode_lookup = _index_dataset_episodes(discovery_dataset)

    try:
        result = _run_one_episode(
            ovon_config=ovon_config,
            api_config_path=str(job_spec["vlm_api_config"]),
            episode_id=episode_id,
            results_dir=str(job_spec["results_dir"]),
            max_subtask_steps=int(job_spec["max_subtask_steps"]),
            save_step_images=bool(job_spec["save_step_images"]),
            save_gif=bool(job_spec["save_gif"]),
            model_stack_builder=_select_model_stack_builder(str(job_spec["runtime"])),
            base_dataset=discovery_dataset,
            episode_lookup=episode_lookup,
            sample_index=sample_index,
            storage_entry_id=storage_entry_id,
            entry_kind=entry_kind,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        result = {
            "success": False,
            "steps": 0,
            "total_steps": 0,
            "distance_to_goal": -1.0,
            "reason": "runtime_exception",
            "error": _format_exception_message(exc),
            "gif_path": "",
            "topdown_path": "",
            "result_file": "",
        }

    result = _cleanup_failed_artifacts(
        results_dir=str(job_spec["results_dir"]),
        storage_entry_id=storage_entry_id,
        entry_kind=entry_kind,
        result=result,
    )

    summary_row = _build_episode_summary_row(
        episode_id,
        result,
        sample_index=sample_index,
    )
    print(
        build_episode_console_summary(
            episode_id=episode_id,
            sample_index=sample_index,
            index=index,
            total=total,
            result=summary_row,
            metrics=_build_console_metrics(summary_row),
            worker_index=worker_index,
            worker_count=worker_count,
        ),
        flush=True,
    )
    return summary_row


def _run_parallel_episodes(
    args: argparse.Namespace,
    episode_specs: Sequence[dict],
) -> List[dict]:
    worker_count = max(1, min(int(args.parallel_workers or 1), len(episode_specs)))
    if worker_count <= 1 or len(episode_specs) <= 1:
        return []

    ordered_specs = [dict(item) for item in episode_specs]
    total = len(ordered_specs)
    next_job_cursor = 0
    results_by_order: List[dict | None] = [None] * total
    mp_context = multiprocessing.get_context("spawn")
    interrupted = False
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
        initializer=_parallel_worker_initializer,
    )
    try:
        future_to_job: dict[concurrent.futures.Future, dict] = {}

        def _submit_next_job(worker_index: int) -> bool:
            nonlocal next_job_cursor
            if next_job_cursor >= total:
                return False
            job_spec = _build_parallel_episode_spec(
                args=args,
                episode_spec=ordered_specs[next_job_cursor],
                index=next_job_cursor + 1,
                total=total,
                worker_index=worker_index,
                worker_count=worker_count,
            )
            future = executor.submit(_run_parallel_episode_job, job_spec)
            future_to_job[future] = job_spec
            next_job_cursor += 1
            return True

        for worker_index in range(1, worker_count + 1):
            if not _submit_next_job(worker_index):
                break

        while future_to_job:
            try:
                done, _ = concurrent.futures.wait(
                    list(future_to_job.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
            except KeyboardInterrupt:
                interrupted = True
                for pending_future in list(future_to_job.keys()):
                    pending_future.cancel()
                _cleanup_interrupted_specs(
                    results_dir=args.results_dir,
                    episode_specs=list(future_to_job.values()),
                )
                raise
            for future in done:
                job_spec = future_to_job.pop(future)
                order_index = int(job_spec["index"]) - 1
                try:
                    results_by_order[order_index] = future.result()
                except (KeyboardInterrupt, SystemExit):
                    interrupted = True
                    future.cancel()
                    _cleanup_interrupted_specs(
                        results_dir=args.results_dir,
                        episode_specs=[job_spec, *list(future_to_job.values())],
                    )
                    for pending_future in list(future_to_job.keys()):
                        pending_future.cancel()
                    raise
                except Exception as exc:
                    results_by_order[order_index] = {
                        "episode_id": int(job_spec["episode_id"]),
                        "sample_index": job_spec.get("sample_index"),
                        "success": False,
                        "steps": 0,
                        "distance_to_goal": -1.0,
                        "spl": 0.0,
                        "soft_spl": 0.0,
                        "gif_path": "",
                        "topdown_path": "",
                        "result_file": "",
                        "reason": "parallel_worker_failed",
                        "error": _format_exception_message(exc),
                    }
                _submit_next_job(int(job_spec["worker_index"]))
    finally:
        _shutdown_parallel_executor(executor, interrupted=interrupted)

    return [item for item in results_by_order if item is not None]


def _run_one_episode(
    *,
    ovon_config,
    api_config_path: str,
    episode_id: int,
    results_dir: str,
    max_subtask_steps: int,
    save_step_images: bool,
    save_gif: bool,
    model_stack_builder,
    base_dataset=None,
    episode_lookup: dict[int, list] | None = None,
    sample_index: int | None = None,
    storage_entry_id: int | None = None,
    entry_kind: str = "episode",
):
    from navigation_system.runtime.object_navigation.runtime_config import (
        build_objectnav_runtime_config,
    )

    runtime_config = build_objectnav_runtime_config(
        results_dir=results_dir,
        max_episode_steps=max(
            1,
            int(
                getattr(
                    ovon_config.habitat.environment,
                    "max_episode_steps",
                    80,
                )
                or 80
            ),
        ),
        save_step_images=save_step_images,
        save_gif=save_gif,
    )
    save_stdout_log = save_episode_stdout_log_enabled(runtime_config)
    episode_log_path = (
        get_episode_records_log_path(
            results_dir,
            int(storage_entry_id if storage_entry_id is not None else episode_id),
            entry_kind=entry_kind,
        )
        if save_stdout_log
        else ""
    )
    redirect_context = (
        redirect_process_output_to_file(episode_log_path, mode="w")
        if save_stdout_log and episode_log_path
        else redirect_process_output_to_null()
    )

    with redirect_context:
        if sample_index is not None:
            selected_episode = None
            if base_dataset is not None:
                base_episodes = list(getattr(base_dataset, "episodes", []) or [])
                if len(base_episodes) == 1:
                    selected_episode = base_episodes[0]
                else:
                    selected_episode = _episode_by_sample_index(base_dataset, int(sample_index))
            if selected_episode is None:
                dataset = _load_dataset_for_sample_indices(ovon_config, [int(sample_index)])
            else:
                dataset = _clone_dataset_with_episode(base_dataset, selected_episode)
        else:
            selected_candidates = list((episode_lookup or {}).get(int(episode_id), []))
            if selected_candidates:
                selected_episode = selected_candidates[0]
                if len(selected_candidates) > 1:
                    print(
                        f"[WARN] Episode id {episode_id} matched {len(selected_candidates)} OVON goals; "
                        f"using first goal={getattr(selected_episode, 'object_category', '')!r}"
                    )
                dataset = _clone_dataset_with_episode(base_dataset, selected_episode)
            else:
                dataset = _load_dataset_for_episodes(ovon_config, [episode_id])

        if len(dataset.episodes) < 1:
            raise RuntimeError(f"Episode {episode_id} not found in OVON dataset")
        if len(dataset.episodes) > 1:
            selected_episode = dataset.episodes[0]
            print(
                f"[WARN] Episode id {episode_id} matched {len(dataset.episodes)} OVON goals; "
                f"using first goal={getattr(selected_episode, 'object_category', '')!r}"
            )
            dataset.episodes = [selected_episode]

        import habitat
        from navigation_system.controller.object_navigation.controller import (
            OVONObjectNavigationController,
        )
        from navigation_system.env.object_navigation.adapter import (
            SingleOVONVectorEnvAdapter,
        )

        env = habitat.Env(config=ovon_config.habitat, dataset=dataset)
        adapter = SingleOVONVectorEnvAdapter(env, episode_count=1)

        controller = OVONObjectNavigationController(
            runtime_config,
            config_path=api_config_path,
            model_stack_builder=model_stack_builder,
            envs=adapter,
        )

        try:
            controller.reset_episode(episode_id=episode_id, sample_index=sample_index)
            result = controller.run_vlm_navigation(max_subtask_steps=max_subtask_steps)
        finally:
            controller.close()

    return result


def _resolve_saved_metrics(result: dict) -> dict:
    result_file = str(result.get("result_file", "") or "").strip()
    if not result_file:
        return {}
    payload = load_json_if_exists(result_file)
    if not isinstance(payload, dict):
        return {}
    return payload


def _coalesce_metric(saved_metrics: dict, saved_key: str, runtime_result: dict, *runtime_keys, default=None):
    saved_value = saved_metrics.get(saved_key)
    if saved_value is not None:
        return saved_value
    for key in runtime_keys:
        value = runtime_result.get(key)
        if value is not None:
            return value
    return default


def _resolve_success_distance_from_ovon_config(ovon_config) -> float:
    try:
        return float(
            getattr(
                getattr(getattr(ovon_config.habitat.task.measurements, "success"), "success_distance"),
                "__float__",
                lambda: OVON_SUCCESS_DISTANCE_M,
            )()
        )
    except Exception:
        try:
            return float(
                getattr(ovon_config.habitat.task.measurements.success, "success_distance")
            )
        except Exception:
            return float(OVON_SUCCESS_DISTANCE_M)


def _build_aggregate(all_results: Sequence[dict]) -> dict:
    count = len(list(all_results or []))
    if count <= 0:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_distance_to_goal": 0.0,
            "avg_spl": 0.0,
            "avg_soft_spl": 0.0,
        }

    rows = list(all_results)
    return {
        "episodes": count,
        "successes": sum(1 for item in rows if bool(item.get("success", False))),
        "success_rate": sum(1 for item in rows if bool(item.get("success", False))) / count,
        "avg_steps": sum(float(item.get("steps", 0) or 0) for item in rows) / count,
        "avg_distance_to_goal": (
            sum(float(item.get("distance_to_goal", -1.0) or -1.0) for item in rows) / count
        ),
        "avg_spl": sum(float(item.get("spl", 0.0) or 0.0) for item in rows) / count,
        "avg_soft_spl": sum(float(item.get("soft_spl", 0.0) or 0.0) for item in rows) / count,
    }


def _format_ovon_metric(value: float, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.0"


def _write_ovon_reports(
    *,
    results_dir: str,
    rows: Sequence[dict],
    aggregate: dict,
    success_distance_m: float,
    summary_meta: dict,
) -> dict:
    report_dir = Path(results_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_txt_path = report_dir / "summary.txt"
    metrics_json_path = report_dir / "metrics.json"
    csv_path = report_dir / "episode_results.csv"
    md_path = report_dir / "episode_results.md"

    summary_text = (
        "========================================\n"
        "OVON evaluation summary\n"
        "========================================\n"
        f"Episodes:      {int(aggregate.get('episodes', 0) or 0)}\n"
        f"Successes:     {int(aggregate.get('successes', 0) or 0)}\n"
        f"SR:            {_format_ovon_metric(float(aggregate.get('success_rate', 0.0) or 0.0), 3)}\n"
        f"SPL:           {_format_ovon_metric(float(aggregate.get('avg_spl', 0.0) or 0.0), 3)}\n"
        f"SoftSPL:       {_format_ovon_metric(float(aggregate.get('avg_soft_spl', 0.0) or 0.0), 3)}\n"
        f"Avg DTG:       {_format_ovon_metric(float(aggregate.get('avg_distance_to_goal', 0.0) or 0.0), 3)}m\n"
        f"Avg Steps:     {_format_ovon_metric(float(aggregate.get('avg_steps', 0.0) or 0.0), 2)}\n"
        f"Success dist:  {float(success_distance_m):.2f}m\n"
        f"Selection:     {summary_meta.get('selection_mode', 'episode_id_order')}\n"
        "========================================\n"
    )
    summary_txt_path.write_text(summary_text, encoding="utf-8")

    metrics_payload = {
        "meta": dict(summary_meta),
        "aggregate": dict(aggregate),
    }
    metrics_json_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    headers = [
        "episode_id",
        "sample_index",
        "sr",
        "distance_to_goal",
        "spl",
        "soft_spl",
        "steps",
        "reason",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "episode_id": int(row.get("episode_id", -1) or -1),
                    "sample_index": (
                        int(row["sample_index"]) if row.get("sample_index") is not None else ""
                    ),
                    "sr": int(bool(row.get("success", False))),
                    "distance_to_goal": _format_ovon_metric(
                        float(row.get("distance_to_goal", -1.0) or -1.0),
                        4,
                    ),
                    "spl": _format_ovon_metric(float(row.get("spl", 0.0) or 0.0), 4),
                    "soft_spl": _format_ovon_metric(float(row.get("soft_spl", 0.0) or 0.0), 4),
                    "steps": int(row.get("steps", 0) or 0),
                    "reason": str(row.get("reason", "") or ""),
                    "error": str(row.get("error", "") or ""),
                }
            )

    md_lines = [
        "# OVON Episode Results",
        "",
        "| Episode | Sample | SR | DTG(m) | SPL | SoftSPL | Steps |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        sample_value = (
            str(int(row["sample_index"]))
            if row.get("sample_index") is not None
            else "-"
        )
        md_lines.append(
            "| {episode} | {sample} | {sr} | {dtg} | {spl} | {soft_spl} | {steps} |".format(
                episode=int(row.get("episode_id", -1) or -1),
                sample=sample_value,
                sr=int(bool(row.get("success", False))),
                dtg=_format_ovon_metric(float(row.get("distance_to_goal", -1.0) or -1.0), 4),
                spl=_format_ovon_metric(float(row.get("spl", 0.0) or 0.0), 4),
                soft_spl=_format_ovon_metric(float(row.get("soft_spl", 0.0) or 0.0), 4),
                steps=int(row.get("steps", 0) or 0),
            )
        )
    md_lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Episodes | SR | SPL | SoftSPL | Avg DTG(m) | Avg Steps |",
            "| --- | --- | --- | --- | --- | --- |",
            "| {episodes} | {sr} | {spl} | {soft_spl} | {dtg} | {steps} |".format(
                episodes=int(aggregate.get("episodes", 0) or 0),
                sr=_format_ovon_metric(float(aggregate.get("success_rate", 0.0) or 0.0), 4),
                spl=_format_ovon_metric(float(aggregate.get("avg_spl", 0.0) or 0.0), 4),
                soft_spl=_format_ovon_metric(float(aggregate.get("avg_soft_spl", 0.0) or 0.0), 4),
                dtg=_format_ovon_metric(float(aggregate.get("avg_distance_to_goal", 0.0) or 0.0), 4),
                steps=_format_ovon_metric(float(aggregate.get("avg_steps", 0.0) or 0.0), 2),
            ),
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "summary": str(summary_txt_path),
        "metrics_json": str(metrics_json_path),
        "csv": str(csv_path),
        "md": str(md_path),
    }


def build_arg_parser(run_defaults: dict | None = None) -> argparse.ArgumentParser:
    defaults = _build_parser_defaults(run_defaults)
    parser = argparse.ArgumentParser(
        description="Run a small-batch OVON evaluation with the SpaceVLN controller",
    )
    parser.add_argument(
        "--run-config",
        type=str,
        default=_default_run_config(),
        help="SpaceVLN OVON runtime defaults YAML",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        choices=RUNTIME_CHOICES,
        default=defaults["runtime"],
        help="runtime profile for OVON: standard or context_cache",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default=defaults["exp_config"],
    )
    parser.add_argument(
        "--vlm-api-config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=defaults["data_path"],
    )
    parser.add_argument("--split", type=str, default=defaults["split"])
    parser.add_argument(
        "--episode-id",
        type=int,
        default=None,
        help="exact starting OVON episode id",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="1-based start index within the current split order",
    )
    parser.add_argument("--episode-ids", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument(
        "--random",
        action="store_true",
        help="sample random episode ids from the selected OVON split",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=defaults["gpu_id"])
    parser.add_argument("--max-steps", type=int, default=defaults["max_steps"])
    parser.add_argument(
        "--max-subtask-steps",
        type=int,
        default=defaults["max_subtask_steps"],
    )
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument(
        "--skip-sr1",
        "--skip-existing-sr1",
        dest="skip_sr1",
        action="store_true",
        help="skip selected episodes that already have successful result logs",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="number of parallel episode workers",
    )
    parser.set_defaults(
        save_step_images=bool(defaults["save_step_images"]),
        save_gif=bool(defaults["save_gif"]),
    )
    parser.add_argument(
        "--save-step-images",
        dest="save_step_images",
        action="store_true",
        help="save per-step visualization PNGs under detail/<bucket>/sample_xxx/visualization",
    )
    parser.add_argument(
        "--no-save-step-images",
        dest="save_step_images",
        action="store_false",
        help="disable per-step visualization PNG saving",
    )
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--no-save-gif", dest="save_gif", action="store_false")
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="skip OVON aggregate report generation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--run-config", type=str, default=_default_run_config())
    bootstrap_args, _ = bootstrap_parser.parse_known_args(raw_argv)
    parser = build_arg_parser(_load_run_defaults(bootstrap_args.run_config))
    args = parser.parse_args(raw_argv)
    args.vlm_api_config = resolve_api_config_path(
        str(args.vlm_api_config or _default_api_config(args.runtime))
    )
    results_root = (
        resolve_results_root_path(args.results_root)
        if str(args.results_root or "").strip()
        else None
    )
    args.results_dir = str(
        args.results_dir
        or _default_results_dir_for_runtime(
            args.runtime,
            args.vlm_api_config,
            results_root=results_root,
        )
        or _default_results_dir()
    )
    model_stack_builder = _select_model_stack_builder(args.runtime)

    with redirect_process_output_to_null():
        ovon_config = _prepare_ovon_config(
            exp_config=args.exp_config,
            split=args.split,
            data_path=args.data_path,
            gpu_id=args.gpu_id,
            max_steps=args.max_steps,
        )

        requested_episode_ids = _parse_episode_ids(args.episode_ids)
        discovery_dataset = _load_dataset_for_episodes(ovon_config, requested_episode_ids or None)
    episode_lookup = _index_dataset_episodes(discovery_dataset)
    split_index_lookup = _build_split_index_lookup(discovery_dataset)
    selected_specs: List[dict]
    if requested_episode_ids:
        selected_specs = _build_episode_id_specs(discovery_dataset, requested_episode_ids)
    elif bool(args.random):
        selected_specs = _build_sample_episode_specs(
            discovery_dataset,
            _random_sample_indices(
                discovery_dataset,
                num_episodes=args.num_episodes,
                seed=args.seed,
            ),
        )
    elif args.start_index is not None:
        sample_indices = _sample_indices_from_range(
            discovery_dataset,
            start_index=int(args.start_index),
            num_episodes=args.num_episodes,
        )
        if not sample_indices:
            raise RuntimeError(
                f"Requested start index {int(args.start_index)} is outside split '{args.split}'"
            )
        selected_specs = _build_sample_episode_specs(discovery_dataset, sample_indices)
        print(
            f"[OVON-ObjectNav] start-index {int(args.start_index)} "
            f"mapped to sample {int(sample_indices[0])} / episode id "
            f"{int(selected_specs[0]['episode_id'])} in split '{args.split}'."
        )
    elif args.episode_id is not None:
        episode_ids = _discover_episode_ids_from_start(
            discovery_dataset,
            episode_id=int(args.episode_id),
            num_episodes=args.num_episodes,
        )
        if episode_ids and int(episode_ids[0]) != int(args.episode_id):
            print(
                f"[OVON-ObjectNav] requested start episode id {int(args.episode_id)} "
                f"is not present in split '{args.split}'; "
                f"starting from next available id {int(episode_ids[0])}."
            )
        selected_specs = _build_episode_id_specs(discovery_dataset, episode_ids)
    else:
        selected_specs = _build_sample_episode_specs(
            discovery_dataset,
            _sample_indices_from_range(
                discovery_dataset,
                start_index=1,
                num_episodes=args.num_episodes,
            ),
        )

    if bool(args.skip_sr1):
        selected_specs = _filter_existing_sr1_specs(
            episode_specs=selected_specs,
            results_dir=args.results_dir,
        )

    if not selected_specs:
        if bool(args.skip_sr1):
            print(
                "[OVON-ObjectNav] no episodes need to run: requested selection "
                "already has SR=1 results"
            )
            return 0
        raise RuntimeError("No OVON episodes selected for evaluation")

    os.makedirs(args.results_dir, exist_ok=True)
    total = len(selected_specs)

    try:
        if int(args.parallel_workers or 1) > 1 and total > 1:
            selection_label = (
                "split-order sample index"
                if args.start_index is not None
                else "ascending episode id"
            )
            print(
                f"[OVON-ObjectNav] parallel execution enabled | workers={int(args.parallel_workers)} | "
                f"episodes={total} | selection={selection_label} | completion logs may interleave",
            )
            all_results = _run_parallel_episodes(args, selected_specs)
        else:
            all_results = []
            for index, episode_spec in enumerate(selected_specs, 1):
                episode_id = int(episode_spec["episode_id"])
                sample_index = episode_spec.get("sample_index")
                print(
                    build_episode_start_summary(
                        episode_id=int(episode_id),
                        sample_index=sample_index,
                        index=index,
                        total=total,
                    ),
                    flush=True,
                )
                try:
                    result = _run_one_episode(
                        ovon_config=ovon_config,
                        api_config_path=args.vlm_api_config,
                        episode_id=episode_id,
                        results_dir=args.results_dir,
                        max_subtask_steps=args.max_subtask_steps,
                        save_step_images=bool(args.save_step_images),
                        save_gif=bool(args.save_gif),
                        model_stack_builder=model_stack_builder,
                        base_dataset=discovery_dataset,
                        episode_lookup=episode_lookup,
                        sample_index=(
                            int(sample_index) if sample_index is not None else None
                        ),
                        storage_entry_id=int(
                            episode_spec.get("storage_entry_id", episode_id)
                        ),
                        entry_kind=str(episode_spec.get("entry_kind", "episode") or "episode"),
                    )
                except KeyboardInterrupt:
                    _cleanup_interrupted_specs(
                        results_dir=args.results_dir,
                        episode_specs=[episode_spec],
                    )
                    raise
                result = _cleanup_failed_artifacts(
                    results_dir=args.results_dir,
                    storage_entry_id=int(
                        episode_spec.get("storage_entry_id", episode_id)
                    ),
                    entry_kind=str(episode_spec.get("entry_kind", "episode") or "episode"),
                    result=result,
                )
                summary_row = _build_episode_summary_row(
                    int(episode_id),
                    result,
                    sample_index=(
                        int(sample_index) if sample_index is not None else None
                    ),
                )
                print(
                    build_episode_console_summary(
                        episode_id=int(episode_id),
                        sample_index=sample_index,
                        index=index,
                        total=total,
                        result=summary_row,
                        metrics=_build_console_metrics(summary_row),
                    ),
                    flush=True,
                )
                all_results.append(summary_row)
    except KeyboardInterrupt:
        print(
            "\n[OVON-ObjectNav] interrupted by user; incomplete log entries were discarded, detail artifacts were kept.",
            flush=True,
        )
        return 130

    aggregate = _build_aggregate(all_results)
    success_distance_m = _resolve_success_distance_from_ovon_config(ovon_config)
    summary_payload = {
        "meta": {
            "runtime": str(args.runtime),
            "split": str(args.split),
            "data_path": str(Path(args.data_path).resolve()),
            "exp_config": str(Path(args.exp_config).resolve()),
            "vlm_api_config": str(args.vlm_api_config),
            "results_dir": str(Path(args.results_dir).resolve()),
            "success_distance_m": success_distance_m,
            "max_episode_steps": int(
                getattr(ovon_config.habitat.environment, "max_episode_steps", 0) or 0
            ),
            "benchmark_note": (
                "This run follows the local naokiyokoyama/ovon repo eval stack: "
                f"STOP + distance_to_goal < success_distance, with success_distance={success_distance_m:.2f} in the loaded config."
            ),
            "selection_mode": (
                "split_order"
                if args.start_index is not None
                else "episode_id_range"
                if args.episode_id is not None
                else "explicit_episode_ids"
                if requested_episode_ids
                else "random"
                if bool(args.random)
                else "split_order"
            ),
        },
        "aggregate": aggregate,
        "episodes": all_results,
    }

    summary_path = Path(args.results_dir) / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary_payload, file, ensure_ascii=False, indent=2)

    episodes_path = Path(args.results_dir) / "episodes.json"
    with episodes_path.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, ensure_ascii=False, indent=2)

    if not bool(args.no_report):
        try:
            report_paths = _write_ovon_reports(
                results_dir=args.results_dir,
                rows=all_results,
                aggregate=aggregate,
                success_distance_m=success_distance_m,
                summary_meta=summary_payload["meta"],
            )
            print(f"[OVON-ObjectNav] saved OVON report: {report_paths['summary']}")
        except Exception as exc:
            print(f"[WARN] Failed to generate OVON aggregate report: {exc}")

    successes = sum(1 for item in all_results if item["success"])
    print(
        f"\n[OVON-ObjectNav] finished {len(all_results)} episodes | "
        f"success={successes}/{len(all_results)} | summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
