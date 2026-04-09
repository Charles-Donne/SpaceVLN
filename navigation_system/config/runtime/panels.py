"""Structured experiment control panels for SpaceVLN runtime config."""

from habitat.config.default import Config as CN

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
    DEFAULT_MAP_RESOLUTION_CM,
    DEFAULT_MAP_SIZE_CM,
    DEFAULT_MAP_VISION_RANGE,
    DEFAULT_MAX_SEMANTIC_CATEGORIES,
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


def build_task_panel() -> CN:
    cfg = CN()
    cfg.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/spacevln_task.yaml"
    cfg.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    return cfg


def build_runtime_panel() -> CN:
    cfg = CN()
    cfg.SIMULATOR_GPU_IDS = [0]
    cfg.TORCH_GPU_ID = 0
    cfg.TORCH_GPU_IDS = [0]
    cfg.GPU_NUMBERS = 1
    cfg.NUM_ENVIRONMENTS = 1
    cfg.KEYBOARD_CONTROL = 0
    return cfg


def build_path_panel() -> CN:
    cfg = CN()
    cfg.TENSORBOARD_DIR = "data/tensorboard_dirs/"
    cfg.CHECKPOINT_FOLDER = "data/checkpoints/"
    cfg.EVAL_CKPT_PATH_DIR = "data/checkpoints/"
    cfg.RESULTS_DIR = "data/logs/eval_results/"
    cfg.VIDEO_DIR = "data/logs/video/"
    return cfg


def build_detection_panel() -> CN:
    cfg = CN()

    cfg.MODEL = CN()
    cfg.MODEL.GROUNDING_DINO_CONFIG_PATH = "../data/model/grounded_sam/GroundingDINO_SwinT_OGC.py"
    cfg.MODEL.GROUNDING_DINO_CHECKPOINT_PATH = "../data/model/grounded_sam/groundingdino_swint_ogc.pth"
    cfg.MODEL.SAM_CHECKPOINT_PATH = "../data/model/grounded_sam/sam_vit_h_4b8939.pth"
    cfg.MODEL.REPVIT_SAM_CHECKPOINT_PATH = "../data/model/grounded_sam/repvit_sam.pt"
    cfg.MODEL.SAM_ENCODER_VERSION = "vit_h"
    cfg.MODEL.USE_REPVIT_SAM = False

    cfg.THRESHOLDS = CN()
    cfg.THRESHOLDS.BOX = DEFAULT_BOX_THRESHOLD
    cfg.THRESHOLDS.TEXT = DEFAULT_TEXT_THRESHOLD
    return cfg


def build_space_panel() -> CN:
    cfg = CN()

    cfg.SENSOR = CN()
    cfg.SENSOR.HFOV_DEG = 79.0
    cfg.SENSOR.FRAME_WIDTH = 160
    cfg.SENSOR.FRAME_HEIGHT = 120
    cfg.SENSOR.AGENT_HEIGHT_M = 0.88
    cfg.SENSOR.DEVICE_ID = 0
    cfg.SENSOR.NUM_ENVIRONMENTS = 1

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
    cfg.MAP.VISUALIZE = False
    cfg.MAP.PRINT_IMAGES = False

    cfg.TOPOLOGY = CN()
    cfg.TOPOLOGY.ENABLE_MULTI_FLOOR = DEFAULT_ENABLE_MULTI_FLOOR_TOPOLOGY
    cfg.TOPOLOGY.FLOOR_Z_TOLERANCE_M = FLOOR_SAME_Z_M
    cfg.TOPOLOGY.FLOOR_Z_SWITCH_THRESHOLD_M = FLOOR_SWITCH_Z_M
    cfg.TOPOLOGY.FLOOR_SWITCH_STABLE_STEPS = FLOOR_SWITCH_STABLE_STEPS
    cfg.TOPOLOGY.STAIR_CLEAR_RADIUS_M = DEFAULT_STAIR_CLEAR_RADIUS_M
    return cfg


def build_render_panel() -> CN:
    cfg = CN()

    cfg.MAP = CN()
    cfg.MAP.ENABLE_GLOBAL_CROP = DEFAULT_ENABLE_GLOBAL_MAP_CROP
    cfg.MAP.ENABLE_ADAPTIVE_ZOOM = DEFAULT_ENABLE_ADAPTIVE_ZOOM
    cfg.MAP.DEBUG_SAVE_RENDERINGS = DEFAULT_DEBUG_SAVE_RENDERINGS
    return cfg


def build_output_panel() -> CN:
    cfg = CN()

    cfg.REQUESTS = CN()
    cfg.REQUESTS.SAVE_VLM_ARTIFACTS = True

    cfg.MAPS = CN()
    cfg.MAPS.SAVE_STEP_ARTIFACTS = False

    cfg.REPLAY = CN()
    cfg.REPLAY.SAVE_STEP_IMAGES = True
    cfg.REPLAY.SAVE_GIF = True
    cfg.REPLAY.CLEANUP_STEP_IMAGES_AFTER_GIF = True

    cfg.LOGS = CN()
    cfg.LOGS.SAVE_EPISODE_STDOUT = False

    cfg.STATE = CN()
    cfg.STATE.SAVE_WAYPOINT_MEMORY = False
    return cfg


def build_control_panel() -> CN:
    cfg = CN()

    cfg.RECOVERY = CN()
    cfg.RECOVERY.ENABLE_AUTO_RETREAT = False
    cfg.RECOVERY.STOP_EARLY_IF_REVERSE_BLOCKED = False
    cfg.RECOVERY.ACTION_STAGNATION_REPLAN_STREAK = 3

    cfg.STAGNATION = CN()
    cfg.STAGNATION.LOW_LEVEL_RATIO = LOW_LEVEL_STAGNATION_RATIO
    cfg.STAGNATION.LOW_LEVEL_CAP_M = LOW_LEVEL_STAGNATION_CAP_M

    cfg.STOPPING = CN()
    cfg.STOPPING.FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK = 3
    cfg.STOPPING.FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M = 1.0
    return cfg


def build_eval_panel() -> CN:
    cfg = CN()
    cfg.SPLIT = "val_unseen"
    cfg.USE_CKPT_CONFIG = False
    cfg.EPISODE_COUNT = 5000
    cfg.SAVE_RESULTS = True
    cfg.SUCCESS_DISTANCE_M = EVAL_SUCCESS_DISTANCE_M
    return cfg
