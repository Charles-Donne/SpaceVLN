"""Smoke test for the local geometric waypoint planner."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navigation_system.space.geometric.waypoint_planner import (
    GeometricPlannerConfig,
    GeometricWaypointPlanner,
    render_candidate_map,
)


def main() -> None:
    config = GeometricPlannerConfig(
        enabled=True,
        max_candidates=5,
        min_candidate_distance_m=0.8,
        max_candidate_distance_m=4.0,
        candidate_stride_m=0.75,
        obstacle_inflation_radius_m=0.30,
        min_clearance_m=0.20,
    )
    planner = GeometricWaypointPlanner(config, resolution_cm=5.0)

    full_map = np.zeros((2, 480, 480), dtype=np.float32)
    full_map[1, 200:280, 210:370] = 1.0
    full_map[0, 200:280, 305:325] = 1.0

    candidates, candidate_paths = planner.build_candidates(
        full_map=full_map,
        pose_xytheta=(12.0, 12.0, 0.0),
        crop_offset=(0, 0),
        trajectory_points=[(238, 238), (239, 239), (240, 240)],
    )
    assert candidates, planner.last_debug
    assert len(candidates) <= config.max_candidates
    assert all(candidate.candidate_id in candidate_paths for candidate in candidates)

    plan = planner.build_plan(
        candidate=candidates[0],
        candidate_paths=candidate_paths,
        full_map=full_map,
        pose_xytheta=(12.0, 12.0, 0.0),
        crop_offset=(0, 0),
    )
    assert plan is not None
    assert len(plan.path_cells) >= 2
    assert len(plan.action_points) >= 2

    rendered = render_candidate_map(
        full_map=full_map,
        candidates=candidates,
        plan=plan,
        output_size_px=512,
    )
    assert rendered is not None
    assert rendered.shape == (512, 512, 3)
    assert int(rendered.sum()) > 0

    print(
        "ok geometric planner smoke",
        {
            "candidates": len(candidates),
            "path_cells": len(plan.path_cells),
            "action_points": len(plan.action_points),
            "debug": planner.last_debug,
        },
    )


if __name__ == "__main__":
    main()
