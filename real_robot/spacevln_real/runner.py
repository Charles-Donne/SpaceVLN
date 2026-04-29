"""CLI runner for SpaceVLN real-robot sessions."""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
import uuid

from navigation_system.controller.agent.controller import NavigationAgentController
from navigation_system.runtime.vlnce.r2r.execution import load_runtime_config
from navigation_system.runtime.vlnce.profiles import (
    CONTEXT_CACHE_RUNTIME_PROFILE,
    STANDARD_RUNTIME_PROFILE,
)

from spacevln_real.command_bridge import ActionCommandBridge
from spacevln_real.config import load_real_robot_config
from spacevln_real.env_adapter import RealRobotVectorEnv
from spacevln_real.observation_hub import ObservationHub
from spacevln_real.ros_runtime import build_ros_runtime


def _resolve_runtime_profile(runtime_name: str):
    normalized = str(runtime_name or "standard").strip().lower()
    if normalized == "context_cache":
        return CONTEXT_CACHE_RUNTIME_PROFILE
    return STANDARD_RUNTIME_PROFILE


def _resolve_instruction(args: argparse.Namespace) -> str:
    text = str(getattr(args, "instruction", "") or "").strip()
    if text:
        return text
    instruction_file = str(getattr(args, "instruction_file", "") or "").strip()
    if instruction_file:
        with open(instruction_file, "r", encoding="utf-8") as handle:
            text = handle.read().strip()
    if not text:
        raise ValueError("provide --instruction or --instruction-file")
    return text


def _apply_real_overrides(config, real_config) -> None:
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = int(real_config.rgb_width)
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = int(real_config.rgb_height)
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV = float(real_config.hfov_deg)
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = int(real_config.rgb_width)
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = int(real_config.rgb_height)
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV = float(real_config.hfov_deg)
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = float(real_config.min_depth_m)
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = float(real_config.max_depth_m)
    config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = float(real_config.forward_step_m)
    config.TASK_CONFIG.SIMULATOR.TURN_ANGLE = float(real_config.turn_angle_deg)
    config.SPACE.SENSOR.HFOV_DEG = float(real_config.hfov_deg)
    config.SPACE.SENSOR.FRAME_WIDTH = int(real_config.rgb_width)
    config.SPACE.SENSOR.FRAME_HEIGHT = int(real_config.rgb_height)
    config.freeze()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpaceVLN real-robot navigation entrypoint")
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Task instruction for the real robot",
    )
    parser.add_argument(
        "--instruction-file",
        type=str,
        default="",
        help="Read the task instruction from a text file",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default="navigation_system/config/experiments/vlnce/r2r_eval.yaml",
        help="SpaceVLN experiment config",
    )
    parser.add_argument(
        "--real-config",
        type=str,
        default="real_robot/config/real_robot.yaml",
        help="Real-robot ROS/OAK-D Lite config",
    )
    parser.add_argument(
        "--vlm-api-config",
        "--config",
        type=str,
        dest="vlm_api_config",
        default=STANDARD_RUNTIME_PROFILE.default_api_config_path,
        help="Unified VLM/LLM API config path",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="",
        help="Output results directory",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        choices=("standard", "context_cache"),
        default="standard",
        help="Runtime mode",
    )
    parser.add_argument(
        "--max-subtask-steps",
        type=int,
        default=5,
        help="Maximum low-level steps per subtask",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum total steps for the session",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=0,
        help="Episode id used for result files",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="",
        help="Real-robot session id; auto-generated if omitted",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    instruction_text = _resolve_instruction(args)
    runtime_profile = _resolve_runtime_profile(getattr(args, "runtime", "standard"))

    if (
        getattr(args, "runtime", "standard") == "context_cache"
        and not any(
            arg == "--vlm-api-config"
            or arg.startswith("--vlm-api-config=")
            or arg == "--config"
            or arg.startswith("--config=")
            for arg in sys.argv[1:]
        )
    ):
        args.vlm_api_config = CONTEXT_CACHE_RUNTIME_PROFILE.default_api_config_path

    load_args = SimpleNamespace(
        exp_config=args.exp_config,
        results_dir=args.results_dir,
        vlm_api_config=args.vlm_api_config,
        max_steps=args.max_steps,
        save_episode_stdout_log=False,
    )
    config = load_runtime_config(load_args, profile=runtime_profile)
    real_config = load_real_robot_config(args.real_config)
    _apply_real_overrides(config, real_config)

    observation_hub = ObservationHub(real_config)
    command_bridge = ActionCommandBridge(real_config)
    ros_runtime = build_ros_runtime(real_config, observation_hub, command_bridge)
    ros_runtime.start()

    session_id = str(args.session_id or uuid.uuid4())
    env = RealRobotVectorEnv(
        real_config,
        observation_hub,
        command_bridge,
        ros_runtime,
        instruction_text=instruction_text,
        session_id=session_id,
        episode_id=args.episode_id,
        success_distance_m=float(getattr(config.EVAL, "SUCCESS_DISTANCE_M", 3.0)),
    )

    controller = None
    try:
        controller = NavigationAgentController(
            config,
            config_path=args.vlm_api_config,
            model_stack_builder=runtime_profile.model_stack_builder,
            envs=env,
        )
        controller.reset_episode(episode_id=args.episode_id)
        result = controller.run_navigation(
            max_subtask_steps=int(args.max_subtask_steps or 5),
        )
        print(
            "\n[REAL] session=%s success=%s total_steps=%s result_file=%s"
            % (
                session_id,
                bool(result.get("success", False)),
                int(result.get("total_steps", 0) or 0),
                str(result.get("result_file", "") or ""),
            ),
            flush=True,
        )
        return 0
    finally:
        try:
            if controller is not None and hasattr(controller, "envs") and controller.envs is not None:
                controller.envs.close()
            else:
                env.close()
        except Exception:
            pass
