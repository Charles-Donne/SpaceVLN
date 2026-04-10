"""Synchronization helpers between structured runtime panels and derived runtime config."""

from habitat.config.default import Config as CN


def _bool_list(values) -> list:
    return [item for item in list(values or [])]


def sync_runtime_panels(config: CN) -> CN:
    """Sync structured config panels into derived fields used by runtime internals."""
    was_frozen = False
    is_frozen_fn = getattr(config, "is_frozen", None)
    if callable(is_frozen_fn):
        was_frozen = bool(is_frozen_fn())
    if was_frozen and hasattr(config, "defrost"):
        config.defrost()

    task_panel = getattr(config, "TASK", None)
    runtime_panel = getattr(config, "RUNTIME", None)
    paths_panel = getattr(config, "PATHS", None)

    if task_panel is not None:
        config.BASE_TASK_CONFIG_PATH = str(
            getattr(task_panel, "BASE_TASK_CONFIG_PATH", getattr(config, "BASE_TASK_CONFIG_PATH", ""))
            or ""
        )
        config.SENSORS = _bool_list(getattr(task_panel, "SENSORS", getattr(config, "SENSORS", [])))

    if runtime_panel is not None:
        config.SIMULATOR_GPU_IDS = _bool_list(
            getattr(runtime_panel, "SIMULATOR_GPU_IDS", getattr(config, "SIMULATOR_GPU_IDS", [0]))
        )
        config.TORCH_GPU_ID = int(
            getattr(runtime_panel, "TORCH_GPU_ID", getattr(config, "TORCH_GPU_ID", 0)) or 0
        )
        config.TORCH_GPU_IDS = _bool_list(
            getattr(runtime_panel, "TORCH_GPU_IDS", getattr(config, "TORCH_GPU_IDS", [config.TORCH_GPU_ID]))
        )
        config.GPU_NUMBERS = int(
            getattr(runtime_panel, "GPU_NUMBERS", getattr(config, "GPU_NUMBERS", 1)) or 1
        )
        config.NUM_ENVIRONMENTS = int(
            getattr(runtime_panel, "NUM_ENVIRONMENTS", getattr(config, "NUM_ENVIRONMENTS", 1)) or 1
        )
        config.KEYBOARD_CONTROL = int(
            getattr(runtime_panel, "KEYBOARD_CONTROL", getattr(config, "KEYBOARD_CONTROL", 0)) or 0
        )

    if paths_panel is not None:
        config.TENSORBOARD_DIR = str(
            getattr(paths_panel, "TENSORBOARD_DIR", getattr(config, "TENSORBOARD_DIR", "")) or ""
        )
        config.CHECKPOINT_FOLDER = str(
            getattr(paths_panel, "CHECKPOINT_FOLDER", getattr(config, "CHECKPOINT_FOLDER", "")) or ""
        )
        config.EVAL_CKPT_PATH_DIR = str(
            getattr(paths_panel, "EVAL_CKPT_PATH_DIR", getattr(config, "EVAL_CKPT_PATH_DIR", "")) or ""
        )
        config.RESULTS_ROOT = str(
            getattr(paths_panel, "RESULTS_ROOT", getattr(config, "RESULTS_ROOT", "")) or ""
        )
        config.RESULTS_DIR = str(
            getattr(paths_panel, "RESULTS_DIR", getattr(config, "RESULTS_DIR", "")) or ""
        )
        config.VIDEO_DIR = str(
            getattr(paths_panel, "VIDEO_DIR", getattr(config, "VIDEO_DIR", "")) or ""
        )

    task_cfg = getattr(config, "TASK_CONFIG", None)
    rgb_sensor = None
    depth_sensor = None
    agent_cfg = None
    if task_cfg is not None and hasattr(task_cfg, "SIMULATOR"):
        simulator = task_cfg.SIMULATOR
        rgb_sensor = getattr(simulator, "RGB_SENSOR", None)
        depth_sensor = getattr(simulator, "DEPTH_SENSOR", None)
        agent_cfg = getattr(simulator, "AGENT_0", None)

    config.SPACE.SENSOR.DEVICE_ID = int(getattr(config, "TORCH_GPU_ID", 0) or 0)
    config.SPACE.SENSOR.NUM_ENVIRONMENTS = int(getattr(config, "NUM_ENVIRONMENTS", 1) or 1)

    if task_cfg is not None:
        task_was_frozen = False
        is_task_frozen_fn = getattr(task_cfg, "is_frozen", None)
        if callable(is_task_frozen_fn):
            task_was_frozen = bool(is_task_frozen_fn())
        if task_was_frozen and hasattr(task_cfg, "defrost"):
            task_cfg.defrost()

        try:
            if rgb_sensor is not None:
                rgb_sensor.HFOV = float(config.SPACE.SENSOR.HFOV_DEG)
                rgb_sensor.WIDTH = int(config.SPACE.SENSOR.FRAME_WIDTH)
                rgb_sensor.HEIGHT = int(config.SPACE.SENSOR.FRAME_HEIGHT)
            if depth_sensor is not None:
                if hasattr(depth_sensor, "HFOV"):
                    depth_sensor.HFOV = float(config.SPACE.SENSOR.HFOV_DEG)
                if hasattr(depth_sensor, "WIDTH"):
                    depth_sensor.WIDTH = int(config.SPACE.SENSOR.FRAME_WIDTH)
                if hasattr(depth_sensor, "HEIGHT"):
                    depth_sensor.HEIGHT = int(config.SPACE.SENSOR.FRAME_HEIGHT)
            if agent_cfg is not None:
                agent_cfg.HEIGHT = float(config.SPACE.SENSOR.AGENT_HEIGHT_M)
                if hasattr(agent_cfg, "SENSORS"):
                    agent_cfg.SENSORS = _bool_list(getattr(config, "SENSORS", []))

            if hasattr(task_cfg, "DATASET") and hasattr(config, "EVAL"):
                task_cfg.DATASET.SPLIT = str(getattr(config.EVAL, "SPLIT", task_cfg.DATASET.SPLIT) or "")
        finally:
            if task_was_frozen and hasattr(task_cfg, "freeze"):
                task_cfg.freeze()

    if not hasattr(config, "MAP"):
        config.MAP = CN()

    detection = config.DETECTION
    space = config.SPACE
    render = config.RENDER
    output = config.OUTPUT
    control = config.CONTROL
    map_cfg = config.MAP

    map_cfg.GROUNDING_DINO_CONFIG_PATH = detection.MODEL.GROUNDING_DINO_CONFIG_PATH
    map_cfg.GROUNDING_DINO_CHECKPOINT_PATH = detection.MODEL.GROUNDING_DINO_CHECKPOINT_PATH
    map_cfg.SAM_CHECKPOINT_PATH = detection.MODEL.SAM_CHECKPOINT_PATH
    map_cfg.RepViTSAM_CHECKPOINT_PATH = detection.MODEL.REPVIT_SAM_CHECKPOINT_PATH
    map_cfg.SAM_ENCODER_VERSION = detection.MODEL.SAM_ENCODER_VERSION
    map_cfg.REPVITSAM = int(bool(detection.MODEL.USE_REPVIT_SAM))
    map_cfg.BOX_THRESHOLD = float(detection.THRESHOLDS.BOX)
    map_cfg.TEXT_THRESHOLD = float(detection.THRESHOLDS.TEXT)

    map_cfg.DEVICE = int(space.SENSOR.DEVICE_ID)
    map_cfg.HFOV = float(space.SENSOR.HFOV_DEG)
    map_cfg.FRAME_WIDTH = int(space.SENSOR.FRAME_WIDTH)
    map_cfg.FRAME_HEIGHT = int(space.SENSOR.FRAME_HEIGHT)
    map_cfg.AGENT_HEIGHT = float(space.SENSOR.AGENT_HEIGHT_M)
    map_cfg.NUM_ENVIRONMENTS = int(space.SENSOR.NUM_ENVIRONMENTS)

    map_cfg.MAP_RESOLUTION = int(space.MAP.RESOLUTION_CM)
    map_cfg.MAP_SIZE_CM = int(space.MAP.SIZE_CM)
    map_cfg.GLOBAL_DOWNSCALING = int(space.MAP.GLOBAL_DOWNSCALING)
    map_cfg.VISION_RANGE = int(space.MAP.VISION_RANGE)
    map_cfg.DU_SCALE = int(space.MAP.DU_SCALE)
    map_cfg.CAT_PRED_THRESHOLD = float(space.MAP.CATEGORY_THRESHOLD)
    map_cfg.EXP_PRED_THRESHOLD = float(space.MAP.EXPLORED_THRESHOLD)
    map_cfg.MAP_PRED_THRESHOLD = float(space.MAP.OBSTACLE_THRESHOLD)
    map_cfg.MAX_SEM_CATEGORIES = int(space.MAP.MAX_SEMANTIC_CATEGORIES)
    map_cfg.CENTER_RESET_STEPS = int(space.MAP.CENTER_RESET_STEPS)
    map_cfg.MIN_Z = int(space.MAP.MIN_Z_CM)
    map_cfg.VISUALIZE = bool(space.MAP.VISUALIZE)
    map_cfg.PRINT_IMAGES = bool(space.MAP.PRINT_IMAGES)

    map_cfg.ENABLE_MULTI_FLOOR_TOPOLOGY = bool(space.TOPOLOGY.ENABLE_MULTI_FLOOR)
    map_cfg.FLOOR_Z_TOLERANCE_M = float(space.TOPOLOGY.FLOOR_Z_TOLERANCE_M)
    map_cfg.FLOOR_Z_SWITCH_THRESHOLD_M = float(space.TOPOLOGY.FLOOR_Z_SWITCH_THRESHOLD_M)
    map_cfg.FLOOR_SWITCH_STABLE_STEPS = int(space.TOPOLOGY.FLOOR_SWITCH_STABLE_STEPS)
    map_cfg.STAIR_CLEAR_RADIUS_M = float(space.TOPOLOGY.STAIR_CLEAR_RADIUS_M)

    map_cfg.ENABLE_GLOBAL_MAP_CROP = bool(render.MAP.ENABLE_GLOBAL_CROP)
    map_cfg.ENABLE_ADAPTIVE_ZOOM = bool(render.MAP.ENABLE_ADAPTIVE_ZOOM)
    map_cfg.DEBUG_SAVE_RENDERINGS = bool(render.MAP.DEBUG_SAVE_RENDERINGS)

    map_cfg.SAVE_API_REQUEST_ARTIFACTS = bool(output.REQUESTS.SAVE_VLM_ARTIFACTS)
    map_cfg.SAVE_STEP_MAP_ARTIFACTS = bool(output.MAPS.SAVE_STEP_ARTIFACTS)
    map_cfg.SAVE_NAVIGATION_STEP_IMAGES = bool(output.REPLAY.SAVE_STEP_IMAGES)
    map_cfg.SAVE_NAVIGATION_GIF = bool(output.REPLAY.SAVE_GIF)
    map_cfg.CLEANUP_NAVIGATION_STEP_IMAGES_AFTER_GIF = bool(
        output.REPLAY.CLEANUP_STEP_IMAGES_AFTER_GIF
    )
    map_cfg.SAVE_EPISODE_STDOUT_LOG = bool(output.LOGS.SAVE_EPISODE_STDOUT)
    map_cfg.SAVE_WAYPOINT_MEMORY = bool(output.STATE.SAVE_WAYPOINT_MEMORY)

    map_cfg.ENABLE_AUTO_RETREAT = bool(control.RECOVERY.ENABLE_AUTO_RETREAT)
    map_cfg.AUTO_RETREAT_STOP_EARLY_IF_REVERSE_BLOCKED = bool(
        control.RECOVERY.STOP_EARLY_IF_REVERSE_BLOCKED
    )
    map_cfg.ACTION_STAGNATION_REPLAN_STREAK = int(control.RECOVERY.ACTION_STAGNATION_REPLAN_STREAK)
    map_cfg.LOW_LEVEL_STAGNATION_RATIO = float(control.STAGNATION.LOW_LEVEL_RATIO)
    map_cfg.LOW_LEVEL_STAGNATION_CAP_M = float(control.STAGNATION.LOW_LEVEL_CAP_M)
    map_cfg.FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK = int(
        control.STOPPING.FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK
    )
    map_cfg.FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M = float(
        control.STOPPING.FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M
    )
    map_cfg.RESULTS_DIR = str(getattr(config, "RESULTS_DIR", "") or "")

    if was_frozen and hasattr(config, "freeze"):
        config.freeze()
    return config
