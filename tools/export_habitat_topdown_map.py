#!/usr/bin/env python
"""Export a pure Habitat top-down scene map for one VLN-CE episode/sample."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
from typing import Optional

import habitat
import numpy as np
import quaternion
from habitat import make_dataset
from habitat.utils.visualizations import maps as habitat_maps

import habitat_extensions  # noqa: F401 - base package
import habitat_extensions.task  # noqa: F401 - registers SpaceVLN datasets/tasks.
import habitat_extensions.habitat_simulator  # noqa: F401 - registers Sim-v1.
import habitat_extensions.measures  # noqa: F401 - registers VLN measures.
import habitat_extensions.sensors  # noqa: F401 - registers custom sensors.
from habitat_extensions import maps as spacevln_maps
from navigation_system.config import get_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workspace_root() -> Path:
    return _repo_root().parent


def _rxr_dataset_dir() -> Path:
    return _workspace_root() / "data" / "datasets" / "RxR_VLNCE_v0" / "val_unseen"


def _load_rxr_en_us_episode_ids() -> list[int]:
    episode_ids = set()
    for role in ("guide", "follower"):
        path = _rxr_dataset_dir() / f"val_unseen_{role}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        for episode in payload.get("episodes", []):
            instruction = episode.get("instruction") or {}
            if instruction.get("language") == "en-US":
                episode_ids.add(int(episode["episode_id"]))
    return sorted(episode_ids)


def _resolve_episode_id(sample_index: Optional[int], episode_id: Optional[int]) -> int:
    if episode_id is not None:
        return int(episode_id)
    if sample_index is None:
        raise ValueError("Either --sample-index or --episode-id is required")
    episode_ids = _load_rxr_en_us_episode_ids()
    if int(sample_index) < 1 or int(sample_index) > len(episode_ids):
        raise ValueError(
            f"--sample-index must be in [1, {len(episode_ids)}], got {sample_index}"
        )
    return int(episode_ids[int(sample_index) - 1])


def _default_exp_config(family: str) -> str:
    if str(family).lower() == "rxrce":
        return "navigation_system/config/experiments/vlnce/rxr_eval.yaml"
    return "navigation_system/config/experiments/vlnce/r2r_eval.yaml"


def _default_output_path(sample_index: Optional[int], episode_id: int, mode: str, family: str) -> Path:
    name = (
        f"sample_{int(sample_index)}_episode_{episode_id}"
        if sample_index is not None
        else f"episode_{episode_id}"
    )
    folder = "habitat_topdown_rgb" if mode == "rgb" else "habitat_topdown_maps"
    return (
        _workspace_root()
        / "result"
        / str(family)
        / folder
        / f"{name}.png"
    )


def _save_rgb(path: Path, image, *, transparent_background: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2 = spacevln_maps.cv2
    if transparent_background:
        rgb = np.asarray(image)
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgba = rgb.copy()
        else:
            alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
            empty_mask = np.all(rgb[:, :, :3] <= int(8), axis=2)
            alpha[empty_mask] = 0
            rgba = np.dstack([rgb[:, :, :3], alpha])
        cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        return
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _quat_from_angle_axis(angle_rad: float, axis) -> np.quaternion:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = float(angle_rad) / 2.0
    return np.quaternion(
        math.cos(half),
        *(math.sin(half) * axis),
    )


def _birdseye_position(sim, episode, *, center: str, height_above_m: float) -> np.ndarray:
    if center == "scene":
        lower, upper = sim.pathfinder.get_bounds()
        x = (float(lower[0]) + float(upper[0])) / 2.0
        z = (float(lower[2]) + float(upper[2])) / 2.0
        base_y = float(getattr(episode, "start_position", [0.0, 0.0, 0.0])[1])
        return np.array([x, base_y + float(height_above_m), z], dtype=np.float32)
    start_position = np.asarray(episode.start_position, dtype=np.float32)
    return start_position + np.array([0.0, float(height_above_m), 0.0], dtype=np.float32)


def _project_world_point_to_image(
    world_point,
    camera_position: np.ndarray,
    camera_rotation,
    *,
    width: int,
    height: int,
    hfov_deg: float,
):
    point = np.asarray(world_point, dtype=np.float32)
    if point.ndim != 1 or point.shape[0] < 3:
        return None
    rel = point[:3] - np.asarray(camera_position, dtype=np.float32)
    rotation_matrix = quaternion.as_rotation_matrix(camera_rotation)
    camera_rel = rotation_matrix.T @ rel
    # Habitat RGB cameras look along -Z in the camera frame.
    depth = -float(camera_rel[2])
    if depth <= 1e-6:
        return None
    fx = (width / 2.0) / math.tan(math.radians(float(hfov_deg)) / 2.0)
    fy = fx
    u = fx * float(camera_rel[0]) / depth + (width - 1) / 2.0
    v = -fy * float(camera_rel[1]) / depth + (height - 1) / 2.0
    if u < 0 or u >= width or v < 0 or v >= height:
        return None
    return int(round(u)), int(round(v))


def _normalize_episode_point(point):
    if point is None:
        return None
    if isinstance(point, dict):
        if "position" in point:
            return point.get("position")
        if "view_points" in point and point.get("view_points"):
            vp = point["view_points"][0]
            if isinstance(vp, dict) and "position" in vp:
                return vp.get("position")
    if isinstance(point, (list, tuple)):
        return point
    if hasattr(point, "position"):
        return getattr(point, "position")
    if hasattr(point, "view_points"):
        view_points = getattr(point, "view_points")
        if view_points:
            return _normalize_episode_point(view_points[0])
    return None


def _draw_overlay_points(image, points, *, color, radius=6, thickness=-1):
    cv2 = spacevln_maps.cv2
    for point in points:
        if point is None:
            continue
        cv2.circle(image, (int(point[0]), int(point[1])), int(radius), color, thickness, lineType=cv2.LINE_AA)


def _draw_overlay_label(image, point, text: str, *, color):
    if point is None:
        return
    cv2 = spacevln_maps.cv2
    x, y = int(point[0]), int(point[1])
    cv2.putText(
        image,
        str(text),
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        str(text),
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def _draw_overlay_lines(image, points, *, color, thickness=2):
    cv2 = spacevln_maps.cv2
    if len(points) < 2:
        return
    for p1, p2 in zip(points[:-1], points[1:]):
        if p1 is None or p2 is None:
            continue
        cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, int(thickness), lineType=cv2.LINE_AA)


def _apply_rgb_overlays(image, sim, episode, args: argparse.Namespace) -> None:
    if not any([args.overlay_path, args.overlay_goals, args.overlay_start]):
        return
    height, width = image.shape[:2]
    camera_position = _birdseye_position(
        sim,
        episode,
        center=str(args.rgb_center),
        height_above_m=float(args.height_above),
    )
    camera_rotation = _quat_from_angle_axis(-math.pi / 2.0, [1.0, 0.0, 0.0])
    projected_path = []
    if args.overlay_path:
        for point in list(getattr(episode, "reference_path", None) or []):
            normalized = _normalize_episode_point(point)
            projected_path.append(
                _project_world_point_to_image(
                    normalized,
                    camera_position,
                    camera_rotation,
                    width=width,
                    height=height,
                    hfov_deg=float(args.hfov),
                )
            )
        _draw_overlay_lines(image, projected_path, color=(0, 255, 255), thickness=3)
        _draw_overlay_points(image, projected_path, color=(0, 255, 255), radius=5)
        if projected_path:
            for index, point in enumerate(projected_path, 1):
                if point is not None:
                    _draw_overlay_label(image, point, str(index), color=(0, 255, 255))

    if args.overlay_goals:
        goal_points = []
        for goal in list(getattr(episode, "goals", None) or []):
            normalized = _normalize_episode_point(goal)
            projected = _project_world_point_to_image(
                normalized,
                camera_position,
                camera_rotation,
                width=width,
                height=height,
                hfov_deg=float(args.hfov),
            )
            if projected is not None:
                goal_points.append(projected)
        _draw_overlay_points(image, goal_points, color=(0, 0, 255), radius=9)
        for index, point in enumerate(goal_points, 1):
            label = "G" if len(goal_points) == 1 else f"G{index}"
            _draw_overlay_label(image, point, label, color=(0, 0, 255))

    if args.overlay_start:
        start_point = _project_world_point_to_image(
            getattr(episode, "start_position", None),
            camera_position,
            camera_rotation,
            width=width,
            height=height,
            hfov_deg=float(args.hfov),
        )
        if start_point is not None:
            _draw_overlay_points(image, [start_point], color=(0, 255, 0), radius=10)
            _draw_overlay_label(image, start_point, "S", color=(0, 255, 0))


def _render_birdseye_rgb(sim, episode, args: argparse.Namespace):
    position = _birdseye_position(
        sim,
        episode,
        center=str(args.rgb_center),
        height_above_m=float(args.height_above),
    )
    # Habitat RGB sensors look along -Z. Rotate -90 degrees around X so the
    # camera looks down along -Y.
    rotation = _quat_from_angle_axis(-math.pi / 2.0, [1.0, 0.0, 0.0])
    observations = sim.get_observations_at(
        position=position,
        rotation=rotation,
        keep_agent_at_new_pose=False,
    )
    if observations is None or "rgb" not in observations:
        raise RuntimeError("Habitat did not return an RGB observation for birdseye render")
    rgb = np.asarray(observations["rgb"])
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    rgb = np.ascontiguousarray(rgb.astype(np.uint8, copy=False))
    _apply_rgb_overlays(rgb, sim, episode, args)
    return rgb


def export_map(args: argparse.Namespace) -> Path:
    if not args.exp_config:
        args.exp_config = _default_exp_config(args.family)
    episode_id = _resolve_episode_id(args.sample_index, args.episode_id)
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else _default_output_path(args.sample_index, episode_id, args.mode, args.family)
    )
    if not output_path.is_absolute():
        output_path = output_path.resolve()

    config = get_config(args.exp_config)
    config.defrost()
    config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = [int(episode_id)]
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = config.TASK.SENSORS
    if args.mode == "rgb":
        config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = int(args.width)
        config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = int(args.height)
        config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV = float(args.hfov)
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = int(args.gpu_id)
    config.freeze()

    dataset = make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    if len(dataset.episodes) != 1:
        raise RuntimeError(
            f"Expected exactly 1 episode for id {episode_id}, got {len(dataset.episodes)}"
        )

    env = habitat.Env(config=config.TASK_CONFIG, dataset=dataset)
    try:
        env.reset()
        sim = env.sim
        episode = dataset.episodes[0]
        if args.mode == "rgb":
            image = _render_birdseye_rgb(sim, episode, args)
        else:
            meters_per_pixel = (
                float(args.meters_per_pixel)
                if args.meters_per_pixel is not None
                else habitat_maps.calculate_meters_per_pixel(int(args.resolution), sim)
            )
            topdown = spacevln_maps.get_top_down_map(
                sim,
                map_resolution=int(args.resolution),
                meters_per_pixel=meters_per_pixel,
            )
            image = spacevln_maps.colorize_topdown_map(topdown)
        _save_rgb(
            output_path,
            image,
            transparent_background=bool(args.transparent_background),
        )
    finally:
        env.close()

    print(f"episode_id={episode_id}")
    if args.sample_index is not None:
        print(f"sample_index={int(args.sample_index)}")
    print(f"output={output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a pure Habitat top-down occupancy scene map for one RxR/R2R episode.",
    )
    parser.add_argument(
        "--exp-config",
        default="",
        help="SpaceVLN experiment config; defaults from --family",
    )
    parser.add_argument("--family", choices=("r2rce", "rxrce"), default="rxrce", help="Result family/output root")
    parser.add_argument("--sample-index", type=int, default=None, help="RxR en-US val_unseen sample index")
    parser.add_argument("--episode-id", type=int, default=None, help="Raw episode id")
    parser.add_argument(
        "--mode",
        choices=("map", "rgb"),
        default="map",
        help="map exports Habitat navmesh occupancy; rgb renders a real top-down RGB camera view",
    )
    parser.add_argument("--output", default="", help="Output PNG path")
    parser.add_argument("--resolution", type=int, default=1024, help="Map resolution in pixels")
    parser.add_argument("--meters-per-pixel", type=float, default=None, help="Override Habitat map scale")
    parser.add_argument("--width", type=int, default=1024, help="RGB birdseye width")
    parser.add_argument("--height", type=int, default=1024, help="RGB birdseye height")
    parser.add_argument("--hfov", type=float, default=90.0, help="RGB birdseye horizontal FOV")
    parser.add_argument("--height-above", type=float, default=3.5, help="RGB camera height above the floor/start point")
    parser.add_argument(
        "--rgb-center",
        choices=("start", "scene"),
        default="start",
        help="RGB birdseye center: episode start point or scene bounds center",
    )
    parser.add_argument("--overlay-path", action="store_true", help="Draw the episode reference path")
    parser.add_argument("--overlay-goals", action="store_true", help="Draw episode goal points")
    parser.add_argument("--overlay-start", action="store_true", help="Draw the start position")
    parser.add_argument(
        "--transparent-background",
        action="store_true",
        help="Save near-black empty Habitat background pixels as transparent PNG alpha",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="Habitat-Sim GPU id")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    export_map(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
