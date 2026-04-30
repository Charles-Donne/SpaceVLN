"""Single-source runtime config assembly for SpaceVLN."""

import os
from typing import List, Optional, Union

try:
    import habitat_baselines.config.default as habitat_baselines_default
except ImportError:
    habitat_baselines_default = None

try:
    from habitat.config.default import CONFIG_FILE_SEPARATOR
    from habitat.config.default import Config as CN
    from habitat_extensions.config.default import get_extended_config as get_task_config
except ImportError:
    from yacs.config import CfgNode as _YacsCfgNode

    CONFIG_FILE_SEPARATOR = ","

    def CN(init_dict=None, key_list=None, new_allowed=True):
        return _YacsCfgNode(
            init_dict=init_dict,
            key_list=key_list,
            new_allowed=new_allowed,
        )

    def get_task_config(_config_path):
        cfg = CN()
        cfg.TASK = CN()
        cfg.SIMULATOR = CN()
        cfg.DATASET = CN()
        cfg.ENVIRONMENT = CN()
        return cfg

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
    DEFAULT_MAP_GLOBAL_DOWNSCALING,
    DEFAULT_MAP_MIN_Z_CM,
    DEFAULT_MAP_EXPLORED_RAY_FILL,
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


_DEFAULT_DEPLOYMENT_CONFIG_FILENAMES = (
    "00_runtime.yaml",
    "10_detection_models.yaml",
    "20_space_sensor.yaml",
)


def _build_space_defaults() -> CN:
    cfg = CN()

    cfg.SENSOR = CN()

    cfg.MAP = CN()
    cfg.MAP.RESOLUTION_CM = DEFAULT_MAP_RESOLUTION_CM
    cfg.MAP.SIZE_CM = DEFAULT_MAP_SIZE_CM
    cfg.MAP.GLOBAL_DOWNSCALING = DEFAULT_MAP_GLOBAL_DOWNSCALING
    cfg.MAP.VISION_RANGE = DEFAULT_MAP_VISION_RANGE
    cfg.MAP.DU_SCALE = DEFAULT_MAP_DU_SCALE
    cfg.MAP.CATEGORY_THRESHOLD = SEM_MAP_CAT_THRESH
    cfg.MAP.EXPLORED_THRESHOLD = SEM_MAP_EXP_THRESH
    cfg.MAP.OBSTACLE_THRESHOLD = SEM_MAP_OBS_THRESH
    cfg.MAP.MAX_SEMANTIC_CATEGORIES = DEFAULT_MAX_SEMANTIC_CATEGORIES
    cfg.MAP.CENTER_RESET_STEPS = DEFAULT_MAP_CENTER_RESET_STEPS
    cfg.MAP.MIN_Z_CM = DEFAULT_MAP_MIN_Z_CM
    cfg.MAP.OBSTACLE_MIN_HEIGHT_CM = DEFAULT_OBSTACLE_MIN_HEIGHT_CM
    cfg.MAP.OBSTACLE_MAX_HEIGHT_CM = DEFAULT_OBSTACLE_MAX_HEIGHT_CM
    cfg.MAP.EXPLORED_RAY_FILL = DEFAULT_MAP_EXPLORED_RAY_FILL
    cfg.MAP.VISUALIZE = False
    cfg.MAP.PRINT_IMAGES = False

    cfg.TOPOLOGY = CN()
    cfg.TOPOLOGY.ENABLE_MULTI_FLOOR = DEFAULT_ENABLE_MULTI_FLOOR_TOPOLOGY
    cfg.TOPOLOGY.FLOOR_Z_TOLERANCE_M = FLOOR_SAME_Z_M
    cfg.TOPOLOGY.FLOOR_Z_SWITCH_THRESHOLD_M = FLOOR_SWITCH_Z_M
    cfg.TOPOLOGY.FLOOR_SWITCH_STABLE_STEPS = FLOOR_SWITCH_STABLE_STEPS
    cfg.TOPOLOGY.STAIR_CLEAR_RADIUS_M = DEFAULT_STAIR_CLEAR_RADIUS_M
    return cfg


def _build_render_defaults() -> CN:
    cfg = CN()
    cfg.MAP = CN()
    cfg.MAP.ENABLE_GLOBAL_CROP = DEFAULT_ENABLE_GLOBAL_MAP_CROP
    cfg.MAP.ENABLE_ADAPTIVE_ZOOM = DEFAULT_ENABLE_ADAPTIVE_ZOOM
    cfg.MAP.DEBUG_SAVE_RENDERINGS = DEFAULT_DEBUG_SAVE_RENDERINGS
    return cfg


def _build_output_defaults() -> CN:
    cfg = CN()

    cfg.REQUESTS = CN()
    cfg.REQUESTS.SAVE_VLM_ARTIFACTS = True

    cfg.MAPS = CN()
    cfg.MAPS.SAVE_STEP_ARTIFACTS = False

    cfg.REPLAY = CN()
    cfg.REPLAY.SAVE_STEP_IMAGES = False
    cfg.REPLAY.SAVE_GIF = True
    cfg.REPLAY.CLEANUP_STEP_IMAGES_AFTER_GIF = False
    cfg.REPLAY.GIF_FPS = 2
    cfg.REPLAY.GIF_MAX_WIDTH = 720

    cfg.LOGS = CN()
    cfg.LOGS.SAVE_EPISODE_STDOUT = False

    cfg.STATE = CN()
    cfg.STATE.SAVE_WAYPOINT_MEMORY = False
    return cfg


def _build_control_defaults() -> CN:
    cfg = CN()

    cfg.RECOVERY = CN()
    cfg.RECOVERY.ENABLE_AUTO_RETREAT = False
    cfg.RECOVERY.STOP_EARLY_IF_REVERSE_BLOCKED = False
    cfg.RECOVERY.ACTION_STAGNATION_REPLAN_STREAK = 3

    cfg.STAGNATION = CN()
    cfg.STAGNATION.LOW_LEVEL_RATIO = LOW_LEVEL_STAGNATION_RATIO
    cfg.STAGNATION.LOW_LEVEL_CAP_M = LOW_LEVEL_STAGNATION_CAP_M

    cfg.STOPPING = CN()
    cfg.STOPPING.ENABLE_FINAL_DESTINATION_MATCH_AUTOSTOP = True
    cfg.STOPPING.FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK = 3
    cfg.STOPPING.FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M = 1.0
    return cfg


def _build_eval_defaults() -> CN:
    cfg = CN()
    cfg.SPLIT = "val_unseen"
    cfg.USE_CKPT_CONFIG = False
    cfg.EPISODE_COUNT = 5000
    cfg.SAVE_RESULTS = True
    cfg.SUCCESS_DISTANCE_M = EVAL_SUCCESS_DISTANCE_M
    return cfg


def _build_base_config() -> CN:
    cfg = CN()
    cfg.TASK_CONFIG = CN()
    cfg.TRAINER_NAME = "ZS-Evaluator"
    cfg.ENV_NAME = "VLNCEZeroShotEnv"
    cfg.VIDEO_OPTION = []

    cfg.TASK = CN()
    cfg.RUNTIME = CN()
    cfg.PATHS = CN()

    cfg.DETECTION = CN()
    cfg.DETECTION.MODEL = CN()
    cfg.DETECTION.THRESHOLDS = CN()
    cfg.DETECTION.THRESHOLDS.BOX = DEFAULT_BOX_THRESHOLD
    cfg.DETECTION.THRESHOLDS.TEXT = DEFAULT_TEXT_THRESHOLD

    cfg.SPACE = _build_space_defaults()
    cfg.RENDER = _build_render_defaults()
    cfg.OUTPUT = _build_output_defaults()
    cfg.CONTROL = _build_control_defaults()
    cfg.EVAL = _build_eval_defaults()

    cfg.MAP = CN()
    return cfg


_C = _build_base_config()


def _resolve_default_deployment_config_paths() -> List[str]:
    system_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "system"))
    return [
        os.path.join(system_dir, filename)
        for filename in _DEFAULT_DEPLOYMENT_CONFIG_FILENAMES
        if os.path.exists(os.path.join(system_dir, filename))
    ]


def _sync_task_success_distance(task_config: CN, success_distance_m: float) -> None:
    if task_config is None or not hasattr(task_config, "TASK"):
        return

    was_frozen = False
    is_frozen_fn = getattr(task_config, "is_frozen", None)
    if callable(is_frozen_fn):
        was_frozen = bool(is_frozen_fn())
    if was_frozen and hasattr(task_config, "defrost"):
        task_config.defrost()

    try:
        success_distance = float(success_distance_m)
        task = task_config.TASK
        task.SUCCESS_DISTANCE = success_distance
        for key in (
            "SUCCESS",
            "SPL",
            "NDTW",
            "SDTW",
            "ORACLE_SUCCESS",
            "ORACLE_NAVIGATION_ERROR",
            "ORACLE_SPL",
        ):
            if hasattr(task, key):
                getattr(task, key).SUCCESS_DISTANCE = success_distance
    finally:
        if was_frozen and hasattr(task_config, "freeze"):
            task_config.freeze()


def _refresh_task_config_if_needed(config: CN, prev_task_config: str) -> str:
    task_panel = getattr(config, "TASK", CN())
    current_task_config = str(getattr(task_panel, "BASE_TASK_CONFIG_PATH", "") or "").strip()
    if not current_task_config:
        return prev_task_config

    task_config = getattr(config, "TASK_CONFIG", None)
    task_config_missing = task_config is None or not hasattr(task_config, "TASK")
    if current_task_config != prev_task_config or task_config_missing:
        config.TASK_CONFIG = get_task_config(current_task_config)
        prev_task_config = current_task_config

    _sync_task_success_distance(
        config.TASK_CONFIG,
        float(getattr(config.EVAL, "SUCCESS_DISTANCE_M", EVAL_SUCCESS_DISTANCE_M)),
    )
    return prev_task_config


def purge_keys(config: CN, keys: List[str]) -> None:
    for k in keys:
        if k in config:
            del config[k]
        config.register_deprecated_key(k)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    """Create a SpaceVLN config from system defaults, experiment YAML, and opts."""
    config = CN()
    if habitat_baselines_default is not None:
        config.merge_from_other_cfg(habitat_baselines_default._C)
        purge_keys(config, ["SIMULATOR_GPU_ID", "TEST_EPISODE_COUNT"])
    config.merge_from_other_cfg(_C.clone())
    if habitat_baselines_default is None and hasattr(config, "set_new_allowed"):
        config.set_new_allowed(True)

    prev_task_config = ""
    merged_config_paths: List[str] = _resolve_default_deployment_config_paths()

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]
        merged_config_paths.extend(config_paths)

    if merged_config_paths:
        for config_path in merged_config_paths:
            config.merge_from_file(config_path)
            prev_task_config = _refresh_task_config_if_needed(config, prev_task_config)
    else:
        prev_task_config = _refresh_task_config_if_needed(config, prev_task_config)

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)
        prev_task_config = _refresh_task_config_if_needed(config, prev_task_config)

    apply_runtime_derived_fields(config)
    purge_keys(
        config,
        [
            "BASE_TASK_CONFIG_PATH",
            "SENSORS",
            "SIMULATOR_GPU_IDS",
            "TORCH_GPU_ID",
            "NUM_ENVIRONMENTS",
            "RESULTS_ROOT",
            "RESULTS_DIR",
        ],
    )
    config.freeze()
    return config


__all__ = ["apply_runtime_derived_fields", "get_config"]
