"""Run SpaceVLN's prompt-based controller on OVON episodes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import habitat
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    HabitatSimDepthSensorConfig,
    TopDownMapMeasurementConfig,
    register_hydra_plugin,
)

from navigation_system.object_navigation.controller import (
    OVONObjectNavigationController,
)
from navigation_system.object_navigation.env_adapter import SingleOVONVectorEnvAdapter
from navigation_system.object_navigation.runtime_config import build_objectnav_runtime_config
from navigation_system.object_navigation.runtime_factory import (
    build_ovon_context_cache_navigation_model_stack,
    build_ovon_navigation_model_stack,
)
from navigation_system.object_navigation.thresholds import OVON_SUCCESS_DISTANCE_M
from navigation_system.runtime.episode_io import load_json_if_exists
from navigation_system.runtime.results_report import generate_results_report
from navigation_system.runtime.storage.results_layout import (
    build_model_results_dir_name,
    resolve_api_config_path,
)


RUNTIME_CHOICES = ("standard", "context_cache")


def _nav_ws_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_results_dir() -> str:
    return str((_nav_ws_root() / "result" / "ovon" / "ovon_smoke").resolve())


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


def _default_results_dir_for_runtime(runtime: str, api_config_path: str) -> str:
    runtime_name = str(runtime or "context_cache").strip().lower()
    suffix = "_cache" if runtime_name == "context_cache" else ""
    model_dir = build_model_results_dir_name(api_config_path)
    return str(
        (
            _nav_ws_root()
            / "result"
            / "ovon"
            / f"{model_dir}{suffix}"
        ).resolve()
    )


def _select_model_stack_builder(runtime: str):
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
    sys.path.insert(0, str((_nav_ws_root() / "ovon").resolve()))
    ovon_repo_root = (_nav_ws_root() / "ovon").resolve()
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


def _discover_episode_ids(dataset, *, num_episodes: int) -> List[int]:
    episodes = list(getattr(dataset, "episodes", []) or [])
    return [int(ep.episode_id) for ep in episodes[: max(1, int(num_episodes))]]


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
):
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

    env = habitat.Env(config=ovon_config.habitat, dataset=dataset)
    adapter = SingleOVONVectorEnvAdapter(env, episode_count=1)
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small-batch OVON evaluation with the SpaceVLN controller",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        choices=RUNTIME_CHOICES,
        default="context_cache",
        help="runtime profile for OVON: standard or context_cache",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default=str((_nav_ws_root() / "ovon" / "config" / "experiments" / "transformer_dagger.yaml").resolve()),
    )
    parser.add_argument(
        "--vlm-api-config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str((_nav_ws_root() / "data" / "datasets" / "ovon" / "hm3d" / "v1" / "val_unseen" / "val_unseen_hard.json.gz").resolve()),
    )
    parser.add_argument("--split", type=str, default="val_unseen")
    parser.add_argument("--episode-ids", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-subtask-steps", type=int, default=4)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.set_defaults(save_step_images=True)
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
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="skip CE-style aggregate report generation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.vlm_api_config = resolve_api_config_path(
        str(args.vlm_api_config or _default_api_config(args.runtime))
    )
    args.results_dir = str(
        args.results_dir
        or _default_results_dir_for_runtime(args.runtime, args.vlm_api_config)
        or _default_results_dir()
    )
    model_stack_builder = _select_model_stack_builder(args.runtime)

    ovon_config = _prepare_ovon_config(
        exp_config=args.exp_config,
        split=args.split,
        data_path=args.data_path,
        gpu_id=args.gpu_id,
        max_steps=args.max_steps,
    )

    requested_episode_ids = _parse_episode_ids(args.episode_ids)
    discovery_dataset = _load_dataset_for_episodes(ovon_config, requested_episode_ids or None)
    if requested_episode_ids:
        episode_ids = requested_episode_ids
    else:
        episode_ids = _discover_episode_ids(
            discovery_dataset,
            num_episodes=args.num_episodes,
        )

    if not episode_ids:
        raise RuntimeError("No OVON episodes selected for evaluation")

    os.makedirs(args.results_dir, exist_ok=True)
    all_results = []

    for episode_id in episode_ids:
        print(f"\n[OVON-ObjectNav] Running episode {episode_id}")
        result = _run_one_episode(
            ovon_config=ovon_config,
            api_config_path=args.vlm_api_config,
            episode_id=episode_id,
            results_dir=args.results_dir,
            max_subtask_steps=args.max_subtask_steps,
            save_step_images=bool(args.save_step_images),
            save_gif=bool(args.save_gif),
            model_stack_builder=model_stack_builder,
        )
        saved_metrics = _resolve_saved_metrics(result)
        all_results.append(
            {
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
                "result_file": result.get("result_file", ""),
            }
        )

    aggregate = _build_aggregate(all_results)
    summary_payload = {
        "meta": {
            "runtime": str(args.runtime),
            "split": str(args.split),
            "data_path": str(Path(args.data_path).resolve()),
            "exp_config": str(Path(args.exp_config).resolve()),
            "vlm_api_config": str(args.vlm_api_config),
            "results_dir": str(Path(args.results_dir).resolve()),
            "success_distance_m": _resolve_success_distance_from_ovon_config(ovon_config),
            "max_episode_steps": int(
                getattr(ovon_config.habitat.environment, "max_episode_steps", 0) or 0
            ),
            "benchmark_note": (
                "This run follows the local naokiyokoyama/ovon repo eval stack: "
                f"STOP + distance_to_goal < success_distance, with success_distance={_resolve_success_distance_from_ovon_config(ovon_config):.2f} in the loaded config."
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
