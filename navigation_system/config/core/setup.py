"""Focused runtime config mutation helpers for SpaceVLN navigation."""

from typing import List, Optional
try:
    from habitat import Config
except ImportError:  # Habitat 0.2.x no longer exports Config
    from typing import Any as Config


def _bool_list(values) -> list:
    return [item for item in list(values or [])]


def apply_runtime_derived_fields(config: Config) -> Config:
    """Update internal Habitat task and map fields from canonical config sections."""
    was_frozen = False
    is_frozen_fn = getattr(config, "is_frozen", None)
    if callable(is_frozen_fn):
        was_frozen = bool(is_frozen_fn())
    if was_frozen and hasattr(config, "defrost"):
        config.defrost()

    runtime = config.RUNTIME
    space = config.SPACE
    task_panel = config.TASK

    space.SENSOR.DEVICE_ID = int(getattr(runtime, "TORCH_GPU_ID", 0) or 0)
    space.SENSOR.NUM_ENVIRONMENTS = int(getattr(runtime, "NUM_ENVIRONMENTS", 1) or 1)

    task_cfg = getattr(config, "TASK_CONFIG", None)
    if task_cfg is not None and hasattr(task_cfg, "SIMULATOR"):
        task_was_frozen = False
        is_task_frozen_fn = getattr(task_cfg, "is_frozen", None)
        if callable(is_task_frozen_fn):
            task_was_frozen = bool(is_task_frozen_fn())
        if task_was_frozen and hasattr(task_cfg, "defrost"):
            task_cfg.defrost()

        try:
            simulator = task_cfg.SIMULATOR
            rgb_sensor = getattr(simulator, "RGB_SENSOR", None)
            depth_sensor = getattr(simulator, "DEPTH_SENSOR", None)
            agent_cfg = getattr(simulator, "AGENT_0", None)

            if rgb_sensor is not None:
                rgb_sensor.HFOV = float(space.SENSOR.HFOV_DEG)
                rgb_sensor.WIDTH = int(space.SENSOR.FRAME_WIDTH)
                rgb_sensor.HEIGHT = int(space.SENSOR.FRAME_HEIGHT)
            if depth_sensor is not None:
                if hasattr(depth_sensor, "HFOV"):
                    depth_sensor.HFOV = float(space.SENSOR.HFOV_DEG)
                if hasattr(depth_sensor, "WIDTH"):
                    depth_sensor.WIDTH = int(space.SENSOR.FRAME_WIDTH)
                if hasattr(depth_sensor, "HEIGHT"):
                    depth_sensor.HEIGHT = int(space.SENSOR.FRAME_HEIGHT)
            if agent_cfg is not None:
                agent_cfg.HEIGHT = float(space.SENSOR.AGENT_HEIGHT_M)
                if hasattr(agent_cfg, "SENSORS"):
                    agent_cfg.SENSORS = _bool_list(getattr(task_panel, "SENSORS", []))

            if hasattr(task_cfg, "DATASET") and hasattr(config, "EVAL"):
                task_cfg.DATASET.SPLIT = str(getattr(config.EVAL, "SPLIT", task_cfg.DATASET.SPLIT) or "")
        finally:
            if task_was_frozen and hasattr(task_cfg, "freeze"):
                task_cfg.freeze()

    detection = config.DETECTION
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
    map_cfg.RESULTS_DIR = str(getattr(config.PATHS, "RESULTS_DIR", "") or "")

    if was_frozen and hasattr(config, "freeze"):
        config.freeze()
    return config


class ConfigHelper:
    """Helpers that keep derived runtime config in sync after mutations."""
    
    @staticmethod
    def setup_navigation_config(
        config: Config,
        torch_gpu_id: Optional[int] = None,
        num_environments: Optional[int] = None
    ) -> Config:
        """
        Configure navigation-related parameters.

        Args:
            config: Habitat config object.
            torch_gpu_id: Optional GPU device id.
            num_environments: Optional number of environments.

        Returns:
            Updated config object.
        """
        config.defrost()

        # Read defaults from the config when values are omitted.
        if torch_gpu_id is None:
            torch_gpu_id = config.RUNTIME.TORCH_GPU_ID
        if num_environments is None:
            num_environments = config.RUNTIME.NUM_ENVIRONMENTS

        config.RUNTIME.TORCH_GPU_ID = int(torch_gpu_id)
        config.RUNTIME.NUM_ENVIRONMENTS = int(num_environments)

        # ===== Enable required Habitat metrics =====
        required_measurements = [
            "TOP_DOWN_MAP_VLNCE",
            "DISTANCE_TO_GOAL",
            "SUCCESS",
            "SPL",
            "ORACLE_NAVIGATION_ERROR",
            "ORACLE_SUCCESS",
            "ORACLE_SPL",
        ]
        
        for measurement in required_measurements:
            if measurement not in config.TASK_CONFIG.TASK.MEASUREMENTS:
                config.TASK_CONFIG.TASK.MEASUREMENTS.append(measurement)

        apply_runtime_derived_fields(config)
        config.freeze()
        return config
    
    @staticmethod
    def setup_episode_config(
        config: Config,
        episode_ids: List[int],
        num_environments: int = 1
    ) -> Config:
        """
        Configure episode-related parameters.

        Args:
            config: Habitat config object.
            episode_ids: Episode ids to run.
            num_environments: Number of environments.

        Returns:
            Updated config object.
        """
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episode_ids
        config.RUNTIME.NUM_ENVIRONMENTS = int(num_environments)
        apply_runtime_derived_fields(config)
        config.freeze()
        return config
    
    @staticmethod
    def setup_results_dir(
        config: Config,
        results_dir: str
    ) -> Config:
        """
        Set the results directory.

        Args:
            config: Habitat config object.
            results_dir: Results directory.

        Returns:
            Updated config object.
        """
        config.defrost()
        config.PATHS.RESULTS_DIR = str(results_dir)
        apply_runtime_derived_fields(config)
        config.freeze()
        return config
