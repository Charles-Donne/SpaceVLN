from typing import List, Optional, Union

import habitat_baselines.config.default
from habitat.config.default import Config as CN
from habitat.config.default import CONFIG_FILE_SEPARATOR
from habitat_extensions.config.default import get_extended_config as get_task_config

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


# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/spacevln_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.TRAINER_NAME = "ZS-Evaluator"
_C.ENV_NAME = "VLNCEZeroShotEnv"
_C.SIMULATOR_GPU_IDS = [0]
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR = "videos/debug"
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.RESULTS_DIR = "data/checkpoints/pretrained/evals"
# BLIP2 模型路径（根据实际路径修改）
_C.BLIP2_MODEL_DIR = "data/model_zoo/blip2/blip2_model.pt"
_C.BLIP2_VIS_PROCESSORS_DIR = "data/model_zoo/blip2/blip2_vis_processors.pt"
_C.BLIP2_TEXT_PROCESSORS_DIR = "data/model_zoo/blip2/blip2_text_processors.pt"
# VQA 模型路径（根据实际路径修改）
_C.VQA_MODEL_DIR = "data/model_zoo/vqa/vqa_model.pt"
_C.VQA_VIS_PROCESSORS_DIR = "data/model_zoo/vqa/vqa_vis_processors.pt"
_C.VQA_TEXT_PROCESSORS_DIR = "data/model_zoo/vqa/vqa_text_processors.pt"
_C.KEYBOARD_CONTROL = 0

# -----------------------------------------------------------------------------
# MAP CONFIG
# -----------------------------------------------------------------------------
_C.MAP = CN()
_C.MAP.GROUNDING_DINO_CONFIG_PATH = "../data/model/grounded_sam/GroundingDINO_SwinT_OGC.py"
_C.MAP.GROUNDING_DINO_CHECKPOINT_PATH = "../data/model/grounded_sam/groundingdino_swint_ogc.pth"
_C.MAP.SAM_CHECKPOINT_PATH = "../data/model/grounded_sam/sam_vit_h_4b8939.pth"
_C.MAP.RepViTSAM_CHECKPOINT_PATH = "../data/model/grounded_sam/repvit_sam.pt"
_C.MAP.SAM_ENCODER_VERSION = "vit_h"
_C.MAP.BOX_THRESHOLD = 0.25
_C.MAP.TEXT_THRESHOLD = 0.25
_C.MAP.FRAME_WIDTH = 160
_C.MAP.FRAME_HEIGHT = 120
_C.MAP.MAP_RESOLUTION = 5
_C.MAP.MAP_SIZE_CM = 2400  # 固定地图大小（24m×24m），可通过增大此值扩展边界
_C.MAP.GLOBAL_DOWNSCALING = 2
_C.MAP.VISION_RANGE = 100
_C.MAP.DU_SCALE = 1
_C.MAP.CAT_PRED_THRESHOLD = SEM_MAP_CAT_THRESH
_C.MAP.EXP_PRED_THRESHOLD = SEM_MAP_EXP_THRESH
_C.MAP.MAP_PRED_THRESHOLD = SEM_MAP_OBS_THRESH
_C.MAP.MAX_SEM_CATEGORIES = 16
_C.MAP.CENTER_RESET_STEPS = 25
_C.MAP.MIN_Z = 2 # a lager min_z could lost some information on the floor, 2cm is ok
_C.MAP.VISUALIZE = False
_C.MAP.PRINT_IMAGES = False
_C.MAP.REPVITSAM = 0

# 可视化配置
_C.MAP.ENABLE_GLOBAL_MAP_CROP = False  # 是否裁剪global map到440×440（默认False，保持480×480）
_C.MAP.ENABLE_ADAPTIVE_ZOOM = True     # 是否启用自适应缩放（内容较小时自动放大到接近边缘）
_C.MAP.DEBUG_SAVE_RENDERINGS = False   # 默认关闭debug渲染；关闭后只保存推理真正需要的最小可视化
_C.MAP.SAVE_API_REQUEST_ARTIFACTS = True   # 是否保存每次LLM/VLM调用的最小请求工件（prompt、输入图片、response、cache记录）
_C.MAP.SAVE_STEP_MAP_ARTIFACTS = False  # 是否额外保存逐步 global_map/local_map 中间地图工件
_C.MAP.SAVE_NAVIGATION_STEP_IMAGES = True   # 是否保存逐步navigation PNG
_C.MAP.SAVE_NAVIGATION_GIF = True     # 是否生成最终navigation.gif
_C.MAP.SAVE_EPISODE_STDOUT_LOG = False  # 是否额外保存整段stdout/stderr文本日志
_C.MAP.SAVE_WAYPOINT_MEMORY = False  # 是否保存waypoint_memory.json
_C.MAP.SAVE_BEST_RESULT_COPY = False  # 是否在records/里额外保留best result.json副本
_C.MAP.ENABLE_AUTO_RETREAT = False      # 是否启用卡住时自动转身后退重规划
_C.MAP.AUTO_RETREAT_STOP_EARLY_IF_REVERSE_BLOCKED = False  # 自动回退时，反向也阻挡是否提前停止
_C.MAP.ACTION_STAGNATION_REPLAN_STREAK = 3  # 连续多少个底层MOVE_FORWARD没推进时触发rethinking
_C.MAP.LOW_LEVEL_STAGNATION_RATIO = LOW_LEVEL_STAGNATION_RATIO
_C.MAP.LOW_LEVEL_STAGNATION_CAP_M = LOW_LEVEL_STAGNATION_CAP_M
_C.MAP.FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK = 3  # 连续多少次thinking检测到waypoint_chain末尾与destination一致时，才允许进入终点稳定区自动停止判定
_C.MAP.FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M = 1.0  # 以第一次终点匹配位置为圆心，连续命中都需留在半径1m内（约直径2m）才自动停止任务
_C.MAP.ENABLE_MULTI_FLOOR_TOPOLOGY = True  # 是否启用基于z轴的多楼层拓扑拆分
_C.MAP.FLOOR_Z_TOLERANCE_M = FLOOR_SAME_Z_M  # 高度差在该阈值内视为同楼层
_C.MAP.FLOOR_Z_SWITCH_THRESHOLD_M = FLOOR_SWITCH_Z_M  # 超过该阈值才允许进入新楼层候选
_C.MAP.FLOOR_SWITCH_STABLE_STEPS = FLOOR_SWITCH_STABLE_STEPS  # 连续多少步稳定命中新楼层高度后正式切层
_C.MAP.STAIR_CLEAR_RADIUS_M = 0.45  # 楼梯连接区域周围多少米内不再保留为障碍物



# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.SPLIT = "val_unseen"  # The split to evaluate on
_C.EVAL.USE_CKPT_CONFIG = True
_C.EVAL.EPISODE_COUNT = 5000
_C.EVAL.SAVE_RESULTS = True
_C.EVAL.SUCCESS_DISTANCE_M = EVAL_SUCCESS_DISTANCE_M


def _sync_task_success_distance(task_config: CN) -> None:
    if task_config is None or not hasattr(task_config, "TASK"):
        return

    was_frozen = False
    is_frozen_fn = getattr(task_config, "is_frozen", None)
    if callable(is_frozen_fn):
        was_frozen = bool(is_frozen_fn())
    if was_frozen and hasattr(task_config, "defrost"):
        task_config.defrost()

    try:
        success_distance = float(EVAL_SUCCESS_DISTANCE_M)
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

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        prev_task_config = ""
        for config_path in config_paths:
            config.merge_from_file(config_path)
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                config.TASK_CONFIG = get_task_config(
                    config.BASE_TASK_CONFIG_PATH
                )
                _sync_task_success_distance(config.TASK_CONFIG)
                prev_task_config = config.BASE_TASK_CONFIG_PATH

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config
