"""Build a lightweight SpaceVLN-style runtime config for OVON object navigation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

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
    FLOOR_SAME_Z_M,
    FLOOR_SWITCH_STABLE_STEPS,
    FLOOR_SWITCH_Z_M,
    LOW_LEVEL_STAGNATION_CAP_M,
    LOW_LEVEL_STAGNATION_RATIO,
)
from navigation_system.runtime.object_navigation.thresholds import (
    OVON_AUTOCOMPLETE_SOLID_M,
    OVON_BOX_THRESHOLD,
    OVON_TEXT_THRESHOLD,
    OVON_SUCCESS_DISTANCE_M,
)


class ConfigNode(dict):
    """Very small attribute-access config node with no-op freeze/defrost API."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def clone(self) -> "ConfigNode":
        return copy.deepcopy(self)

    def defrost(self) -> None:
        return None

    def freeze(self) -> None:
        return None

    def is_frozen(self) -> bool:
        return False


def _node(data: Dict[str, Any]) -> ConfigNode:
    result = ConfigNode()
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = _node(value)
        else:
            result[key] = value
    return result


def build_objectnav_runtime_config(
    *,
    results_dir: str,
    max_episode_steps: int = 500,
    frame_width: int = 640,
    frame_height: int = 480,
    hfov_deg: float = 79.0,
    agent_height_m: float = 0.88,
    agent_radius_m: float = 0.18,
    turn_angle_deg: float = 30.0,
    forward_step_m: float = 0.25,
    min_depth_m: float = 0.5,
    max_depth_m: float = 5.0,
    torch_gpu_id: int = 0,
    simulator_gpu_id: int = 0,
    save_request_artifacts: bool = True,
    save_step_images: bool = True,
    save_gif: bool = False,
) -> ConfigNode:
    nav_ws_root = Path(__file__).resolve().parents[4]
    grounded_sam_dir = nav_ws_root / "data" / "model" / "grounded_sam"

    config = _node(
        {
            "ENV_NAME": "OVONObjectNavAdapterEnv",
            "TRAINER_NAME": "OVON-ObjectNav-SpaceVLN",
            "VIDEO_OPTION": [],
            "TASK": {
                "SENSORS": [],
            },
            "RUNTIME": {
                "TORCH_GPU_ID": int(torch_gpu_id),
                "NUM_ENVIRONMENTS": 1,
                "SIMULATOR_GPU_IDS": [int(simulator_gpu_id)],
            },
            "PATHS": {
                "RESULTS_ROOT": str(Path(results_dir).resolve().parent),
                "RESULTS_DIR": str(Path(results_dir).resolve()),
            },
            "DETECTION": {
                "MODEL": {
                    "GROUNDING_DINO_CONFIG_PATH": str(grounded_sam_dir / "GroundingDINO_SwinT_OGC.py"),
                    "GROUNDING_DINO_CHECKPOINT_PATH": str(grounded_sam_dir / "groundingdino_swint_ogc.pth"),
                    "SAM_CHECKPOINT_PATH": str(grounded_sam_dir / "sam_vit_h_4b8939.pth"),
                    "REPVIT_SAM_CHECKPOINT_PATH": str(grounded_sam_dir / "repvit_sam.pt"),
                    "SAM_ENCODER_VERSION": "vit_h",
                    "USE_REPVIT_SAM": True,
                },
                "THRESHOLDS": {
                    "BOX": float(OVON_BOX_THRESHOLD),
                    "TEXT": float(OVON_TEXT_THRESHOLD),
                },
            },
            "SPACE": {
                "SENSOR": {
                    "HFOV_DEG": float(hfov_deg),
                    "FRAME_WIDTH": int(frame_width),
                    "FRAME_HEIGHT": int(frame_height),
                    "AGENT_HEIGHT_M": float(agent_height_m),
                    "DEVICE_ID": int(torch_gpu_id),
                    "NUM_ENVIRONMENTS": 1,
                },
                "MAP": {
                    "RESOLUTION_CM": int(DEFAULT_MAP_RESOLUTION_CM),
                    "SIZE_CM": int(DEFAULT_MAP_SIZE_CM),
                    "GLOBAL_DOWNSCALING": int(DEFAULT_MAP_GLOBAL_DOWNSCALING),
                    "VISION_RANGE": int(DEFAULT_MAP_VISION_RANGE),
                    "DU_SCALE": int(DEFAULT_MAP_DU_SCALE),
                    "CATEGORY_THRESHOLD": 5.0,
                    "EXPLORED_THRESHOLD": 1.0,
                    "OBSTACLE_THRESHOLD": 1.0,
                    "MAX_SEMANTIC_CATEGORIES": int(DEFAULT_MAX_SEMANTIC_CATEGORIES),
                    "CENTER_RESET_STEPS": int(DEFAULT_MAP_CENTER_RESET_STEPS),
                    "MIN_Z_CM": int(DEFAULT_MAP_MIN_Z_CM),
                    "VISUALIZE": False,
                    "PRINT_IMAGES": False,
                },
                "TOPOLOGY": {
                    "ENABLE_MULTI_FLOOR": bool(DEFAULT_ENABLE_MULTI_FLOOR_TOPOLOGY),
                    "FLOOR_Z_TOLERANCE_M": float(FLOOR_SAME_Z_M),
                    "FLOOR_Z_SWITCH_THRESHOLD_M": float(FLOOR_SWITCH_Z_M),
                    "FLOOR_SWITCH_STABLE_STEPS": int(FLOOR_SWITCH_STABLE_STEPS),
                    "STAIR_CLEAR_RADIUS_M": float(DEFAULT_STAIR_CLEAR_RADIUS_M),
                },
            },
            "RENDER": {
                "MAP": {
                    "ENABLE_GLOBAL_CROP": bool(DEFAULT_ENABLE_GLOBAL_MAP_CROP),
                    "ENABLE_ADAPTIVE_ZOOM": bool(DEFAULT_ENABLE_ADAPTIVE_ZOOM),
                    "DEBUG_SAVE_RENDERINGS": bool(DEFAULT_DEBUG_SAVE_RENDERINGS),
                }
            },
            "OUTPUT": {
                "REQUESTS": {
                    "SAVE_VLM_ARTIFACTS": bool(save_request_artifacts),
                },
                "MAPS": {
                    "SAVE_STEP_ARTIFACTS": False,
                },
                "REPLAY": {
                    "SAVE_STEP_IMAGES": bool(save_step_images),
                    "SAVE_GIF": bool(save_gif),
                    "CLEANUP_STEP_IMAGES_AFTER_GIF": bool(save_gif and not save_step_images),
                },
                "LOGS": {
                    "SAVE_EPISODE_STDOUT": False,
                },
                "STATE": {
                    "SAVE_WAYPOINT_MEMORY": True,
                },
            },
            "CONTROL": {
                "RECOVERY": {
                    "ENABLE_AUTO_RETREAT": False,
                    "STOP_EARLY_IF_REVERSE_BLOCKED": False,
                    "ACTION_STAGNATION_REPLAN_STREAK": 3,
                },
                "STAGNATION": {
                    "LOW_LEVEL_RATIO": float(LOW_LEVEL_STAGNATION_RATIO),
                    "LOW_LEVEL_CAP_M": float(LOW_LEVEL_STAGNATION_CAP_M),
                },
                "STOPPING": {
                    "FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK": 2,
                    "FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M": float(OVON_AUTOCOMPLETE_SOLID_M),
                },
            },
            "EVAL": {
                "SPLIT": "val_unseen",
                "USE_CKPT_CONFIG": False,
                "EPISODE_COUNT": 1,
                "SAVE_RESULTS": True,
                "SUCCESS_DISTANCE_M": float(OVON_SUCCESS_DISTANCE_M),
            },
            "MAP": {},
            "TASK_CONFIG": {
                "ENVIRONMENT": {
                    "MAX_EPISODE_STEPS": int(max_episode_steps),
                },
                "SIMULATOR": {
                    "TURN_ANGLE": float(turn_angle_deg),
                    "FORWARD_STEP_SIZE": float(forward_step_m),
                    "RGB_SENSOR": {
                        "WIDTH": int(frame_width),
                        "HEIGHT": int(frame_height),
                        "HFOV": float(hfov_deg),
                        "POSITION": [0.0, float(agent_height_m), 0.0],
                    },
                    "DEPTH_SENSOR": {
                        "WIDTH": int(frame_width),
                        "HEIGHT": int(frame_height),
                        "HFOV": float(hfov_deg),
                        "MIN_DEPTH": float(min_depth_m),
                        "MAX_DEPTH": float(max_depth_m),
                        "POSITION": [0.0, float(agent_height_m), 0.0],
                    },
                    "AGENT_0": {
                        "SENSORS": ["RGB_SENSOR", "DEPTH_SENSOR"],
                        "HEIGHT": float(agent_height_m),
                        "RADIUS": float(agent_radius_m),
                    },
                    "HABITAT_SIM_V0": {
                        "GPU_DEVICE_ID": int(simulator_gpu_id),
                        "ALLOW_SLIDING": False,
                    },
                },
                "TASK": {
                    "MEASUREMENTS": [
                        "DISTANCE_TO_GOAL",
                        "SUCCESS",
                        "SPL",
                        "SOFT_SPL",
                    ],
                    "SUCCESS_DISTANCE": float(OVON_SUCCESS_DISTANCE_M),
                },
                "DATASET": {
                    "TYPE": "OVON-v1",
                    "SPLIT": "val_unseen",
                    "CONTENT_SCENES": ["*"],
                    "EPISODES_ALLOWED": None,
                },
            },
        }
    )
    return config
