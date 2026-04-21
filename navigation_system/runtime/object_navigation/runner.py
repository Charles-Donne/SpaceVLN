"""Run SpaceVLN's prompt-based controller on OVON episodes."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import os
import random
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


def _default_results_dir() -> str:
    return str((Path(build_default_results_family_root("ovon")) / "ovon_smoke").resolve())


def _default_api_config(runtime: str) -> str:
    runtime_name = str(runtime or "context_cache").strip().lower()
    file_name = (
        "vlm_api_config_context_cache.yaml"
        if runtime_name == "context_cache"
        else "vlm_api_config.yaml"
    )
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

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import habitat
        from habitat.config import read_write
        from habitat.config.default_structured_configs import (
            HabitatSimDepthSensorConfig,
            TopDownMapMeasurementConfig,
            register_hydra_plugin,
        )

    ovon_repo = str((_nav_ws_root() / "ovon").resolve())
    if ovon_repo not in sys.path:
        sys.path.insert(0, ovon_repo)
    ovon_repo_root = (_nav_ws_root() / "ovon").resolve()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from ovon.config import HabitatConfigPlugin

    register_hydra_plugin(HabitatConfigPlugin)
    config = habitat.get_config(str(Path(exp_config).resolve()))

    with read_write(config):
        config.habitat.dataset.split = str(split)
        config.habitat.dataset.data_path = str(Path(data_path).resolve())
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
        config.habitat.task.measurements.top_down_map.draw_goal_aabbs = False

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


def _index_dataset_episodes(dataset) -> dict[int, list]:
    episode_lookup: dict[int, list] = {}
    for episode in list(getattr(dataset, "episodes", []) or []):
        episode_lookup.setdefault(int(episode.episode_id), []).append(episode)
    return episode_lookup


def _clone_dataset_with_episode(base_dataset, episode):
    dataset = copy.copy(base_dataset)
    dataset.episodes = [episode]
    return dataset


def _discover_episode_ids(dataset, *, num_episodes: int) -> List[int]:
    episodes = list(getattr(dataset, "episodes", []) or [])
    return [int(ep.episode_id) for ep in episodes[: max(1, int(num_episodes))]]


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


def _episode_has_existing_sr1(results_dir: str, episode_id: int) -> bool:
    candidate_paths = []
    candidate_paths.extend(get_episode_log_path_candidates(results_dir, episode_id))
    for detail_dir in get_episode_detail_path_candidates(results_dir, episode_id):
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


def _build_episode_summary_row(episode_id: int, result: dict) -> dict:
    saved_metrics = _resolve_saved_metrics(result)
    return {
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
        "oracle_success": int(
            _coalesce_metric(
                saved_metrics,
                "osr",
                result,
                "oracle_success",
                "osr",
                default=0,
            )
            or 0
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
        "osr": int(summary_row.get("oracle_success", 0) or 0),
        "ne": float(summary_row.get("distance_to_goal", -1.0) or -1.0),
        "spl": float(summary_row.get("spl", 0.0) or 0.0),
    }


def _build_parallel_episode_spec(
    *,
    args: argparse.Namespace,
    episode_id: int,
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
        "episode_id": int(episode_id),
        "index": int(index),
        "total": int(total),
        "worker_index": int(worker_index),
        "worker_count": int(worker_count),
    }


def _run_parallel_episode_job(job_spec: dict) -> dict:
    episode_id = int(job_spec["episode_id"])
    index = int(job_spec["index"])
    total = int(job_spec["total"])
    worker_index = int(job_spec["worker_index"])
    worker_count = int(job_spec["worker_count"])

    print(
        build_episode_start_summary(
            episode_id=episode_id,
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
        )
    except BaseException as exc:
        result = {
            "success": False,
            "steps": 0,
            "total_steps": 0,
            "distance_to_goal": -1.0,
            "reason": "",
            "error": str(exc),
            "gif_path": "",
            "topdown_path": "",
            "result_file": "",
        }

    summary_row = _build_episode_summary_row(episode_id, result)
    print(
        build_episode_console_summary(
            episode_id=episode_id,
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


def _run_parallel_episodes(args: argparse.Namespace, episode_ids: Sequence[int]) -> List[dict]:
    worker_count = max(1, min(int(args.parallel_workers or 1), len(episode_ids)))
    if worker_count <= 1 or len(episode_ids) <= 1:
        return []

    ordered_ids = [int(item) for item in episode_ids]
    total = len(ordered_ids)
    next_job_cursor = 0
    results_by_episode_id: dict[int, dict] = {}
    mp_context = multiprocessing.get_context("spawn")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
    ) as executor:
        future_to_job: dict[concurrent.futures.Future, dict] = {}

        def _submit_next_job(worker_index: int) -> bool:
            nonlocal next_job_cursor
            if next_job_cursor >= total:
                return False
            episode_id = ordered_ids[next_job_cursor]
            job_spec = _build_parallel_episode_spec(
                args=args,
                episode_id=episode_id,
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
            done, _ = concurrent.futures.wait(
                list(future_to_job.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                job_spec = future_to_job.pop(future)
                episode_id = int(job_spec["episode_id"])
                try:
                    results_by_episode_id[episode_id] = future.result()
                except BaseException as exc:
                    results_by_episode_id[episode_id] = {
                        "episode_id": episode_id,
                        "success": False,
                        "steps": 0,
                        "distance_to_goal": -1.0,
                        "spl": 0.0,
                        "soft_spl": 0.0,
                        "oracle_success": 0,
                        "gif_path": "",
                        "topdown_path": "",
                        "result_file": "",
                        "reason": "",
                        "error": str(exc),
                    }
                _submit_next_job(int(job_spec["worker_index"]))

    return [results_by_episode_id[int(episode_id)] for episode_id in ordered_ids]


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
        get_episode_records_log_path(results_dir, episode_id)
        if save_stdout_log
        else ""
    )
    redirect_context = (
        redirect_process_output_to_file(episode_log_path, mode="w")
        if save_stdout_log and episode_log_path
        else redirect_process_output_to_null()
    )

    with redirect_context:
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
            controller.reset_episode(episode_id=episode_id)
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
            "oracle_success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_distance_to_goal": 0.0,
            "avg_spl": 0.0,
            "avg_soft_spl": 0.0,
        }

    rows = list(all_results)
    return {
        "episodes": count,
        "successes": sum(1 for item in rows if bool(item.get("success", False))),
        "oracle_successes": sum(
            1 for item in rows if int(item.get("oracle_success", 0) or 0) > 0
        ),
        "success_rate": sum(1 for item in rows if bool(item.get("success", False))) / count,
        "oracle_success_rate": sum(
            1 for item in rows if int(item.get("oracle_success", 0) or 0) > 0
        )
        / count,
        "avg_steps": sum(float(item.get("steps", 0) or 0) for item in rows) / count,
        "avg_distance_to_goal": (
            sum(float(item.get("distance_to_goal", -1.0) or -1.0) for item in rows) / count
        ),
        "avg_spl": sum(float(item.get("spl", 0.0) or 0.0) for item in rows) / count,
        "avg_soft_spl": sum(float(item.get("soft_spl", 0.0) or 0.0) for item in rows) / count,
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
        help="start episode id for VLNCE-style positional launch commands",
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
        help="save per-step visualization PNGs under detail/<bucket>/episode_xxx/visualization",
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
        help="skip CE-style aggregate report generation",
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
    if requested_episode_ids:
        episode_ids = requested_episode_ids
    elif bool(args.random):
        episode_ids = _discover_random_episode_ids(
            discovery_dataset,
            num_episodes=args.num_episodes,
            seed=args.seed,
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
    else:
        episode_ids = _discover_episode_ids(
            discovery_dataset,
            num_episodes=args.num_episodes,
        )

    if bool(args.skip_sr1):
        episode_ids = _filter_existing_sr1(
            episode_ids=episode_ids,
            results_dir=args.results_dir,
        )

    if not episode_ids:
        if bool(args.skip_sr1):
            print(
                "[OVON-ObjectNav] no episodes need to run: requested selection "
                "already has SR=1 results"
            )
            return 0
        raise RuntimeError("No OVON episodes selected for evaluation")

    os.makedirs(args.results_dir, exist_ok=True)
    total = len(episode_ids)

    if int(args.parallel_workers or 1) > 1 and total > 1:
        print(
            f"[OVON-ObjectNav] parallel execution enabled | workers={int(args.parallel_workers)} | "
            f"episodes={total}",
        )
        all_results = _run_parallel_episodes(args, episode_ids)
    else:
        all_results = []
        for index, episode_id in enumerate(episode_ids, 1):
            print(
                build_episode_start_summary(
                    episode_id=int(episode_id),
                    index=index,
                    total=total,
                ),
                flush=True,
            )
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
            )
            summary_row = _build_episode_summary_row(int(episode_id), result)
            print(
                build_episode_console_summary(
                    episode_id=int(episode_id),
                    index=index,
                    total=total,
                    result=summary_row,
                    metrics=_build_console_metrics(summary_row),
                ),
                flush=True,
            )
            all_results.append(summary_row)

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
            from navigation_system.runtime.results_report import generate_results_report

            generate_results_report(
                args.results_dir,
                save=True,
                verbose=True,
                exp_config=args.exp_config,
            )
        except Exception as exc:
            print(f"[WARN] Failed to generate CE-style aggregate report: {exc}")

    successes = sum(1 for item in all_results if item["success"])
    print(
        f"\n[OVON-ObjectNav] finished {len(all_results)} episodes | "
        f"success={successes}/{len(all_results)} | summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
