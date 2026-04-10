import os
from typing import List, Optional, Union

import habitat_baselines.config.default
from habitat.config.default import Config as CN
from habitat.config.default import CONFIG_FILE_SEPARATOR
from habitat_extensions.config.default import get_extended_config as get_task_config

from navigation_system.config.core.params.thresholds import (
    EVAL_SUCCESS_DISTANCE_M,
)
from navigation_system.config.runtime.panels import (
    build_path_panel,
    build_control_panel,
    build_detection_panel,
    build_eval_panel,
    build_output_panel,
    build_render_panel,
    build_runtime_panel,
    build_space_panel,
    build_task_panel,
)
from navigation_system.config.runtime.sync import sync_runtime_panels


# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.TRAINER_NAME = "ZS-Evaluator"
_C.ENV_NAME = "VLNCEZeroShotEnv"
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"

# -----------------------------------------------------------------------------
# SPACEVLN PANELS
# -----------------------------------------------------------------------------
_C.TASK = build_task_panel()
_C.RUNTIME = build_runtime_panel()
_C.PATHS = build_path_panel()
_C.DETECTION = build_detection_panel()
_C.SPACE = build_space_panel()
_C.RENDER = build_render_panel()
_C.OUTPUT = build_output_panel()
_C.CONTROL = build_control_panel()
_C.EVAL = build_eval_panel()

# Internal derived node kept for Habitat/map-runtime wiring.
_C.MAP = CN()

# Internal flattened fields retained for Habitat/Baselines compatibility.
_C.BASE_TASK_CONFIG_PATH = _C.TASK.BASE_TASK_CONFIG_PATH
_C.SENSORS = list(_C.TASK.SENSORS)
_C.SIMULATOR_GPU_IDS = list(_C.RUNTIME.SIMULATOR_GPU_IDS)
_C.TORCH_GPU_ID = int(_C.RUNTIME.TORCH_GPU_ID)
_C.TORCH_GPU_IDS = list(_C.RUNTIME.TORCH_GPU_IDS)
_C.GPU_NUMBERS = int(_C.RUNTIME.GPU_NUMBERS)
_C.NUM_ENVIRONMENTS = int(_C.RUNTIME.NUM_ENVIRONMENTS)
_C.TENSORBOARD_DIR = _C.PATHS.TENSORBOARD_DIR
_C.CHECKPOINT_FOLDER = _C.PATHS.CHECKPOINT_FOLDER
_C.EVAL_CKPT_PATH_DIR = _C.PATHS.EVAL_CKPT_PATH_DIR
_C.RESULTS_ROOT = _C.PATHS.RESULTS_ROOT
_C.RESULTS_DIR = _C.PATHS.RESULTS_DIR
_C.VIDEO_DIR = _C.PATHS.VIDEO_DIR
_C.KEYBOARD_CONTROL = int(_C.RUNTIME.KEYBOARD_CONTROL)

_DEFAULT_DEPLOYMENT_CONFIG_FILENAMES = (
    "00_runtime.yaml",
    "10_detection_models.yaml",
    "20_space_sensor.yaml",
)


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
    current_task_config = str(getattr(config, "BASE_TASK_CONFIG_PATH", "") or "").strip()
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
        del config[k]
        config.register_deprecated_key(k)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    """Create a unified config with default values. Initialized from the
    habitat_baselines default config. Overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = CN()
    config.merge_from_other_cfg(habitat_baselines.config.default._C)
    purge_keys(config, ["SIMULATOR_GPU_ID", "TEST_EPISODE_COUNT"])
    config.merge_from_other_cfg(_C.clone())
    sync_runtime_panels(config)

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
            sync_runtime_panels(config)
            prev_task_config = _refresh_task_config_if_needed(config, prev_task_config)
            sync_runtime_panels(config)
    else:
        prev_task_config = _refresh_task_config_if_needed(config, prev_task_config)
        sync_runtime_panels(config)

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)
        sync_runtime_panels(config)
        prev_task_config = _refresh_task_config_if_needed(config, prev_task_config)

    sync_runtime_panels(config)
    config.freeze()
    return config
