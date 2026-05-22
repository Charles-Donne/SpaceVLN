"""CLI runner for SpaceVLN real-robot sessions."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime

from navigation_system.controller.agent.controller import NavigationAgentController
from navigation_system.runtime.output_policy import (
    add_output_artifact_args,
    add_output_profile_arg,
)
from navigation_system.runtime.process_lifecycle import close_with_timeout
from navigation_system.runtime.vlnce.profiles import (
    STANDARD_RUNTIME_PROFILE,
)

from spacevln_real.command_bridge import ActionCommandBridge
from spacevln_real.config import load_real_robot_config
from spacevln_real.env_adapter import RealRobotVectorEnv
from spacevln_real.observation_hub import ObservationHub
from spacevln_real.runtime_config import build_real_runtime_config
from spacevln_real.ros_runtime import build_ros_runtime


def _resolve_runtime_profile(runtime_name: str):
    normalized = str(runtime_name or "standard").strip().lower()
    if normalized == "context_cache":
        print(
            "[REAL] context_cache is disabled for real-robot runs; using standard runtime",
            flush=True,
        )
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


def _write_real_session_status(
    *,
    results_dir: str,
    session_id: str,
    state: str,
    instruction: str,
    result: dict = None,
    error: str = "",
) -> str:
    os.makedirs(results_dir, exist_ok=True)
    payload = {
        "session_id": str(session_id),
        "state": str(state),
        "instruction": str(instruction),
        "timestamp": datetime.now().isoformat(),
        "results_dir": str(results_dir),
    }
    if result is not None:
        payload["result"] = dict(result)
    if error:
        payload["error"] = str(error)

    status_path = os.path.join(results_dir, "real_session_latest.json")
    tmp_path = f"{status_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, status_path)

    session_dir = os.path.join(results_dir, "real_sessions")
    os.makedirs(session_dir, exist_ok=True)
    session_path = os.path.join(session_dir, f"{session_id}.json")
    tmp_session_path = f"{session_path}.tmp"
    with open(tmp_session_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_session_path, session_path)
    return status_path


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
        default="",
        help="Deprecated compatibility option; real-robot runtime now uses real-only config.",
    )
    parser.add_argument(
        "--real-config",
        type=str,
        default="real_robot/config/real_robot.yaml",
        help="Real-robot ROS/camera config",
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
        help="Runtime mode; real-robot runs always use standard mode and ignore context_cache",
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
    add_output_profile_arg(parser)
    add_output_artifact_args(parser)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    instruction_text = _resolve_instruction(args)
    runtime_profile = _resolve_runtime_profile(getattr(args, "runtime", "standard"))

    real_config = load_real_robot_config(args.real_config)
    config = build_real_runtime_config(
        real_config=real_config,
        args=args,
        runtime_profile=runtime_profile,
    )
    os.makedirs(config.PATHS.RESULTS_DIR, exist_ok=True)
    session_id = str(args.session_id or uuid.uuid4())
    status_path = _write_real_session_status(
        results_dir=config.PATHS.RESULTS_DIR,
        session_id=session_id,
        state="starting",
        instruction=instruction_text,
    )
    print(f"[REAL] results_dir={config.PATHS.RESULTS_DIR}", flush=True)
    print(f"[REAL] live_status={status_path}", flush=True)

    observation_hub = ObservationHub(real_config)
    command_bridge = ActionCommandBridge(real_config)
    ros_runtime = build_ros_runtime(real_config, observation_hub, command_bridge)
    ros_runtime.start()

    _write_real_session_status(
        results_dir=config.PATHS.RESULTS_DIR,
        session_id=session_id,
        state="running",
        instruction=instruction_text,
    )
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
        _write_real_session_status(
            results_dir=config.PATHS.RESULTS_DIR,
            session_id=session_id,
            state="finished",
            instruction=instruction_text,
            result=result,
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
    except Exception as exc:
        _write_real_session_status(
            results_dir=config.PATHS.RESULTS_DIR,
            session_id=session_id,
            state="failed",
            instruction=instruction_text,
            error=repr(exc),
        )
        raise
    finally:
        if controller is not None and hasattr(controller, "envs") and controller.envs is not None:
            close_with_timeout(controller.envs.close, label="real-robot environment")
        else:
            close_with_timeout(env.close, label="real-robot environment")
