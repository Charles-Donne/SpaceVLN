"""Real-robot runtime config without Habitat/Habitat-Sim dependencies."""

from __future__ import annotations

import os
import copy
from typing import Any, Mapping, Optional

import yaml

from navigation_system.config.core.setup import apply_runtime_derived_fields
from navigation_system.config.core.params.detection import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
)
from navigation_system.config.core.params.rendering import (
    DEFAULT_DEBUG_SAVE_RENDERINGS,
    DEFAULT_ENABLE_ADAPTIVE_ZOOM,
    DEFAULT_ENABLE_GLOBAL_MAP_CROP,
)
from navigation_system.config.core.params.spatial import (
    DEFAULT_ENABLE_MULTI_FLOOR_TOPOLOGY,
    DEFAULT_MAP_CENTER_RESET_STEPS,
    DEFAULT_MAP_DU_SCALE,
    DEFAULT_MAP_EXPLORED_RAY_FILL,
    DEFAULT_MAP_GLOBAL_DOWNSCALING,
    DEFAULT_MAP_MIN_Z_CM,
    DEFAULT_MAP_RESOLUTION_CM,
    DEFAULT_MAP_SIZE_CM,
    DEFAULT_MAP_VISION_RANGE,
    DEFAULT_MAX_SEMANTIC_CATEGORIES,
    DEFAULT_OBSTACLE_MAX_HEIGHT_CM,
    DEFAULT_OBSTACLE_MIN_HEIGHT_CM,
    DEFAULT_STAIR_CLEAR_RADIUS_M,
)
from navigation_system.config.core.params.thresholds import (
    EVAL_SUCCESS_DISTANCE_M,
    FLOOR_SAME_Z_M,
    FLOOR_SWITCH_STABLE_STEPS,
    FLOOR_SWITCH_Z_M,
    LOW_LEVEL_STAGNATION_CAP_M,
    LOW_LEVEL_STAGNATION_RATIO,
    SEM_MAP_CAT_THRESH,
    SEM_MAP_EXP_THRESH,
    SEM_MAP_OBS_THRESH,
)
from navigation_system.runtime.output_policy import apply_output_policy_to_config
from navigation_system.runtime.storage.results_layout import (
    resolve_results_dir_path,
    resolve_results_root_path,
)

from spacevln_real.models import RealRobotConfig


class CN(dict):
    """Small yacs-compatible config node for the real-robot runtime."""

    def __init__(
        self,
        init_dict: Optional[Mapping[str, Any]] = None,
        key_list=None,
        new_allowed: bool = True,
    ) -> None:
        del key_list, new_allowed
        super().__init__()
        if init_dict:
            for key, value in init_dict.items():
                self[key] = self._wrap(value)
        self._frozen = False

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, CN):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = self._wrap(value)

    def clone(self) -> "CN":
        return copy.deepcopy(self)

    def defrost(self) -> None:
        self._frozen = False
        for value in self.values():
            if isinstance(value, CN):
                value.defrost()

    def freeze(self) -> None:
        self._frozen = True
        for value in self.values():
            if isinstance(value, CN):
                value.freeze()

    def is_frozen(self) -> bool:
        return bool(self._frozen)

    def merge_from_file(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"Config file must contain a YAML mapping: {path}")
        self.merge_from_mapping(payload)

    def merge_from_mapping(self, payload: Mapping[str, Any]) -> None:
        for key, value in payload.items():
            if (
                key in self
                and isinstance(self[key], CN)
                and isinstance(value, Mapping)
            ):
                self[key].merge_from_mapping(value)
            else:
                self[key] = self._wrap(value)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _workspace_root() -> str:
    return os.path.abspath(os.path.join(_repo_root(), ".."))


def _default_real_results_root() -> str:
    return os.path.join(_workspace_root(), "result")


def _merge_system_yaml(config: CN, relative_path: str) -> None:
    path = os.path.join(_repo_root(), relative_path)
    if os.path.exists(path):
        config.merge_from_file(path)


def _build_task_config(real_config: RealRobotConfig, max_steps: Optional[int]) -> CN:
    task_cfg = CN(new_allowed=True)

    task_cfg.SIMULATOR = CN(new_allowed=True)
    task_cfg.SIMULATOR.RGB_SENSOR = CN(new_allowed=True)
    task_cfg.SIMULATOR.RGB_SENSOR.WIDTH = int(real_config.rgb_width)
    task_cfg.SIMULATOR.RGB_SENSOR.HEIGHT = int(real_config.rgb_height)
    task_cfg.SIMULATOR.RGB_SENSOR.HFOV = float(real_config.hfov_deg)
    task_cfg.SIMULATOR.RGB_SENSOR.POSITION = [0.0, float(real_config.agent_height_m), 0.0]

    task_cfg.SIMULATOR.DEPTH_SENSOR = CN(new_allowed=True)
    task_cfg.SIMULATOR.DEPTH_SENSOR.WIDTH = int(real_config.rgb_width)
    task_cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT = int(real_config.rgb_height)
    task_cfg.SIMULATOR.DEPTH_SENSOR.HFOV = float(real_config.depth_hfov_deg)
    task_cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = float(real_config.min_depth_m)
    task_cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = float(real_config.max_depth_m)
    task_cfg.SIMULATOR.DEPTH_SENSOR.POSITION = [0.0, float(real_config.agent_height_m), 0.0]

    task_cfg.SIMULATOR.FORWARD_STEP_SIZE = float(real_config.forward_step_m)
    task_cfg.SIMULATOR.TURN_ANGLE = float(real_config.turn_angle_deg)
    task_cfg.SIMULATOR.AGENT_0 = CN(new_allowed=True)
    task_cfg.SIMULATOR.AGENT_0.HEIGHT = float(real_config.agent_height_m)
    task_cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]

    task_cfg.ENVIRONMENT = CN(new_allowed=True)
    task_cfg.ENVIRONMENT.MAX_EPISODE_STEPS = int(max_steps or 500)

    task_cfg.TASK = CN(new_allowed=True)
    task_cfg.TASK.SUCCESS_DISTANCE = float(EVAL_SUCCESS_DISTANCE_M)
    task_cfg.TASK.MEASUREMENTS = []

    task_cfg.DATASET = CN(new_allowed=True)
    task_cfg.DATASET.SPLIT = "real"
    task_cfg.DATASET.EPISODES_ALLOWED = None
    return task_cfg


def _build_base_config(real_config: RealRobotConfig, max_steps: Optional[int]) -> CN:
    config = CN(new_allowed=True)

    config.TRAINER_NAME = "SpaceVLN-Real"
    config.ENV_NAME = "RealRobotVectorEnv"
    config.VIDEO_OPTION = []

    config.TASK = CN(new_allowed=True)
    config.TASK.BASE_TASK_CONFIG_PATH = ""
    config.TASK.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.TASK_CONFIG = _build_task_config(real_config, max_steps)

    config.RUNTIME = CN(new_allowed=True)
    config.RUNTIME.SIMULATOR_GPU_IDS = [0]
    config.RUNTIME.TORCH_GPU_ID = 0
    config.RUNTIME.NUM_ENVIRONMENTS = 1

    config.PATHS = CN(new_allowed=True)
    config.PATHS.RESULTS_ROOT = ""
    config.PATHS.RESULTS_DIR = ""

    config.DETECTION = CN(new_allowed=True)
    config.DETECTION.MODEL = CN(new_allowed=True)
    config.DETECTION.MODEL.GROUNDING_DINO_CONFIG_PATH = "../data/model/grounded_sam/GroundingDINO_SwinT_OGC.py"
    config.DETECTION.MODEL.GROUNDING_DINO_CHECKPOINT_PATH = "../data/model/grounded_sam/groundingdino_swint_ogc.pth"
    config.DETECTION.MODEL.SAM_CHECKPOINT_PATH = "../data/model/grounded_sam/sam_vit_h_4b8939.pth"
    config.DETECTION.MODEL.REPVIT_SAM_CHECKPOINT_PATH = "../data/model/grounded_sam/repvit_sam.pt"
    config.DETECTION.MODEL.SAM_ENCODER_VERSION = "vit_h"
    config.DETECTION.MODEL.USE_REPVIT_SAM = True
    config.DETECTION.THRESHOLDS = CN(new_allowed=True)
    config.DETECTION.THRESHOLDS.BOX = DEFAULT_BOX_THRESHOLD
    config.DETECTION.THRESHOLDS.TEXT = DEFAULT_TEXT_THRESHOLD

    config.SPACE = CN(new_allowed=True)
    config.SPACE.SENSOR = CN(new_allowed=True)
    config.SPACE.SENSOR.HFOV_DEG = float(real_config.hfov_deg)
    config.SPACE.SENSOR.DEPTH_HFOV_DEG = float(real_config.depth_hfov_deg)
    config.SPACE.SENSOR.FRAME_WIDTH = int(real_config.rgb_width)
    config.SPACE.SENSOR.FRAME_HEIGHT = int(real_config.rgb_height)
    config.SPACE.SENSOR.AGENT_HEIGHT_M = float(real_config.agent_height_m)
    config.SPACE.SENSOR.CAMERA_ELEVATION_DEG = float(real_config.camera_pitch_deg)
    config.SPACE.SENSOR.DEVICE_ID = 0
    config.SPACE.SENSOR.NUM_ENVIRONMENTS = 1

    config.SPACE.MAP = CN(new_allowed=True)
    config.SPACE.MAP.RESOLUTION_CM = DEFAULT_MAP_RESOLUTION_CM
    config.SPACE.MAP.SIZE_CM = DEFAULT_MAP_SIZE_CM
    config.SPACE.MAP.GLOBAL_DOWNSCALING = DEFAULT_MAP_GLOBAL_DOWNSCALING
    config.SPACE.MAP.VISION_RANGE = DEFAULT_MAP_VISION_RANGE
    config.SPACE.MAP.DU_SCALE = DEFAULT_MAP_DU_SCALE
    config.SPACE.MAP.CATEGORY_THRESHOLD = SEM_MAP_CAT_THRESH
    config.SPACE.MAP.EXPLORED_THRESHOLD = SEM_MAP_EXP_THRESH
    config.SPACE.MAP.OBSTACLE_THRESHOLD = SEM_MAP_OBS_THRESH
    config.SPACE.MAP.MAX_SEMANTIC_CATEGORIES = DEFAULT_MAX_SEMANTIC_CATEGORIES
    config.SPACE.MAP.CENTER_RESET_STEPS = DEFAULT_MAP_CENTER_RESET_STEPS
    config.SPACE.MAP.MIN_Z_CM = DEFAULT_MAP_MIN_Z_CM
    config.SPACE.MAP.OBSTACLE_MIN_HEIGHT_CM = DEFAULT_OBSTACLE_MIN_HEIGHT_CM
    config.SPACE.MAP.OBSTACLE_MAX_HEIGHT_CM = DEFAULT_OBSTACLE_MAX_HEIGHT_CM
    config.SPACE.MAP.EXPLORED_RAY_FILL = DEFAULT_MAP_EXPLORED_RAY_FILL
    config.SPACE.MAP.SELECTIVE_DYNAMIC_OBSTACLE_UPDATE = bool(
        real_config.selective_dynamic_obstacle_update
    )
    config.SPACE.MAP.OBSTACLE_EVIDENCE_THRESHOLD = float(
        real_config.obstacle_evidence_threshold
    )
    config.SPACE.MAP.OBSTACLE_EVIDENCE_MAX_OBSERVATIONS = int(
        real_config.obstacle_evidence_max_observations
    )
    config.SPACE.MAP.VISUALIZE = False
    config.SPACE.MAP.PRINT_IMAGES = False

    config.SPACE.TOPOLOGY = CN(new_allowed=True)
    config.SPACE.TOPOLOGY.ENABLE_MULTI_FLOOR = DEFAULT_ENABLE_MULTI_FLOOR_TOPOLOGY
    config.SPACE.TOPOLOGY.FLOOR_Z_TOLERANCE_M = FLOOR_SAME_Z_M
    config.SPACE.TOPOLOGY.FLOOR_Z_SWITCH_THRESHOLD_M = FLOOR_SWITCH_Z_M
    config.SPACE.TOPOLOGY.FLOOR_SWITCH_STABLE_STEPS = FLOOR_SWITCH_STABLE_STEPS
    config.SPACE.TOPOLOGY.STAIR_CLEAR_RADIUS_M = DEFAULT_STAIR_CLEAR_RADIUS_M

    config.RENDER = CN(new_allowed=True)
    config.RENDER.MAP = CN(new_allowed=True)
    config.RENDER.MAP.ENABLE_GLOBAL_CROP = DEFAULT_ENABLE_GLOBAL_MAP_CROP
    config.RENDER.MAP.ENABLE_ADAPTIVE_ZOOM = DEFAULT_ENABLE_ADAPTIVE_ZOOM
    config.RENDER.MAP.DEBUG_SAVE_RENDERINGS = DEFAULT_DEBUG_SAVE_RENDERINGS

    config.OUTPUT = CN(new_allowed=True)
    config.OUTPUT.REQUESTS = CN(new_allowed=True)
    config.OUTPUT.REQUESTS.SAVE_VLM_ARTIFACTS = True
    config.OUTPUT.MAPS = CN(new_allowed=True)
    config.OUTPUT.MAPS.SAVE_STEP_ARTIFACTS = False
    config.OUTPUT.REPLAY = CN(new_allowed=True)
    config.OUTPUT.REPLAY.SAVE_STEP_IMAGES = False
    config.OUTPUT.REPLAY.SAVE_GIF = True
    config.OUTPUT.REPLAY.CLEANUP_STEP_IMAGES_AFTER_GIF = False
    config.OUTPUT.REPLAY.GIF_FPS = 2
    config.OUTPUT.REPLAY.GIF_MAX_WIDTH = 720
    config.OUTPUT.LOGS = CN(new_allowed=True)
    config.OUTPUT.LOGS.SAVE_EPISODE_STDOUT = False
    config.OUTPUT.STATE = CN(new_allowed=True)
    config.OUTPUT.STATE.SAVE_WAYPOINT_MEMORY = False

    config.CONTROL = CN(new_allowed=True)
    config.CONTROL.RECOVERY = CN(new_allowed=True)
    config.CONTROL.RECOVERY.ENABLE_AUTO_RETREAT = False
    config.CONTROL.RECOVERY.STOP_EARLY_IF_REVERSE_BLOCKED = False
    config.CONTROL.RECOVERY.ACTION_STAGNATION_REPLAN_STREAK = 3
    config.CONTROL.STAGNATION = CN(new_allowed=True)
    config.CONTROL.STAGNATION.LOW_LEVEL_RATIO = LOW_LEVEL_STAGNATION_RATIO
    config.CONTROL.STAGNATION.LOW_LEVEL_CAP_M = LOW_LEVEL_STAGNATION_CAP_M
    config.CONTROL.STOPPING = CN(new_allowed=True)
    config.CONTROL.STOPPING.ENABLE_FINAL_DESTINATION_MATCH_AUTOSTOP = True
    config.CONTROL.STOPPING.FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK = 3
    config.CONTROL.STOPPING.FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M = 1.0

    config.REAL_ROBOT = CN(new_allowed=True)
    config.REAL_ROBOT.ENABLED = True
    config.REAL_ROBOT.LOOKAROUND_SAMPLE_COUNT = int(real_config.lookaround_sample_count)
    config.REAL_ROBOT.LOOKAROUND_ANGLE_STEP_DEG = float(real_config.lookaround_angle_step_deg)
    config.REAL_ROBOT.DISABLE_DEPTH_MAP_UPDATE = bool(real_config.disable_depth_map_update)
    config.REAL_ROBOT.DEPTH_FUSION_FRAMES = int(real_config.depth_fusion_frames)

    config.EVAL = CN(new_allowed=True)
    config.EVAL.SPLIT = "real"
    config.EVAL.SAVE_RESULTS = True
    config.EVAL.SUCCESS_DISTANCE_M = EVAL_SUCCESS_DISTANCE_M

    config.MAP = CN(new_allowed=True)
    return config


def _resolve_results_root_setting() -> str:
    real_env_results_root = str(os.getenv("SPACEVLN_REAL_RESULTS_ROOT", "") or "").strip()
    if real_env_results_root:
        return real_env_results_root
    return _default_real_results_root()


def build_real_runtime_config(
    *,
    real_config: RealRobotConfig,
    args,
    runtime_profile,
) -> CN:
    """Build the controller config for real-robot navigation without Habitat imports."""
    if str(getattr(runtime_profile, "name", "") or "") == "context_cache":
        from navigation_system.vlm.api.qwen_context_cache_client import (
            validate_qwen_context_cache_api_config,
        )

        validate_qwen_context_cache_api_config(args.vlm_api_config)

    config = _build_base_config(real_config, getattr(args, "max_steps", None))
    _merge_system_yaml(config, "navigation_system/config/system/10_detection_models.yaml")
    _merge_system_yaml(config, "navigation_system/config/system/20_space_sensor.yaml")

    config.SPACE.SENSOR.HFOV_DEG = float(real_config.hfov_deg)
    config.SPACE.SENSOR.DEPTH_HFOV_DEG = float(real_config.depth_hfov_deg)
    config.SPACE.SENSOR.FRAME_WIDTH = int(real_config.rgb_width)
    config.SPACE.SENSOR.FRAME_HEIGHT = int(real_config.rgb_height)
    config.SPACE.SENSOR.AGENT_HEIGHT_M = float(real_config.agent_height_m)
    config.SPACE.SENSOR.CAMERA_ELEVATION_DEG = float(real_config.camera_pitch_deg)
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = int(real_config.rgb_width)
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = int(real_config.rgb_height)
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV = float(real_config.hfov_deg)
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.POSITION = [0.0, float(real_config.agent_height_m), 0.0]
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = int(real_config.rgb_width)
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = int(real_config.rgb_height)
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV = float(real_config.depth_hfov_deg)
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = float(real_config.min_depth_m)
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = float(real_config.max_depth_m)
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION = [0.0, float(real_config.agent_height_m), 0.0]
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = float(real_config.agent_height_m)
    config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = float(real_config.forward_step_m)
    config.TASK_CONFIG.SIMULATOR.TURN_ANGLE = float(real_config.turn_angle_deg)
    if getattr(args, "max_steps", None) is not None:
        config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = int(args.max_steps)

    configured_results_root = str(getattr(config.PATHS, "RESULTS_ROOT", "") or "").strip()
    yaml_results_root = _resolve_results_root_setting()
    selected_results_root = configured_results_root or yaml_results_root
    raw_results_dir = str(getattr(args, "results_dir", "") or "").strip()
    allow_results_dir_override = str(
        os.getenv("SPACEVLN_ALLOW_REAL_RESULTS_DIR_OVERRIDE", "") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if raw_results_dir and not allow_results_dir_override:
        print(
            "[REAL] ignoring --results-dir; real-robot runs write to the "
            "SpaceVLN sibling result dir by default. Set "
            "SPACEVLN_ALLOW_REAL_RESULTS_DIR_OVERRIDE=1 to override.",
            flush=True,
        )
        raw_results_dir = ""
    resolved_results_dir = resolve_results_dir_path(raw_results_dir)
    if not resolved_results_dir:
        previous_family = os.environ.get("SPACEVLN_RESULTS_FAMILY")
        os.environ["SPACEVLN_RESULTS_FAMILY"] = "real_robot"
        try:
            resolved_results_dir = runtime_profile.default_results_dir_builder(
                args.vlm_api_config,
                results_root=selected_results_root or None,
            )
        finally:
            if previous_family is None:
                os.environ.pop("SPACEVLN_RESULTS_FAMILY", None)
            else:
                os.environ["SPACEVLN_RESULTS_FAMILY"] = previous_family
    config.PATHS.RESULTS_ROOT = resolve_results_root_path(selected_results_root)
    config.PATHS.RESULTS_DIR = resolved_results_dir

    apply_output_policy_to_config(config, args)
    apply_runtime_derived_fields(config)
    config.freeze()
    return config
