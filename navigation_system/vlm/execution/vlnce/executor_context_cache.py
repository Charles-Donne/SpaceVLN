"""Action runtime with DashScope explicit context cache."""

import os
from typing import Any, Dict, Optional, Sequence, Tuple

from navigation_system.vlm.execution.vlnce.executor import ActionExecutor
from navigation_system.vlm.prompts.common import PromptBundle
from navigation_system.vlm.prompts.vlnce.builders import build_action_prompt_bundle
from navigation_system.vlm.api.qwen_context_cache_client import QwenContextCacheMixin


class ContextCacheActionExecutor(QwenContextCacheMixin, ActionExecutor):
    """Action executor variant that reuses a long stable system prompt via explicit cache."""

    def __init__(
        self,
        config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml",
        turn_angle: float = 30.0,
        move_distance: float = 0.25,
    ):
        super().__init__(
            config_path=config_path,
            turn_angle=turn_angle,
            move_distance=move_distance,
        )
        self._init_qwen_context_cache(config_path)
        print(f"  ActionVLM(ContextCache): {self.config.model} | explicit-context-cache")

    def call_api(
        self,
        prompt_bundle: PromptBundle,
        image_paths,
        save_dir: str = None,
        no_compress_indices: set = None,
    ):
        return self.call_api_with_explicit_context_cache(
            prompt_bundle=prompt_bundle,
            image_paths=list(image_paths or []),
            save_dir=save_dir,
            no_compress_indices=no_compress_indices,
        )

    def decide_action(
        self,
        next_waypoint: str,
        subtask_instruction: str,
        first_person_image: Any,
        action_mapping: Dict[str, int],
        progress_summary: str = "",
        waypoint_summary: str = "",
        subtask_landmark: str = "",
        detection_image: Any = None,
        detected_landmarks: str = None,
        obstacle_distances: Dict[str, str] = None,
        landmark_map_info: str = None,
        allowed_action_names: Optional[Sequence[str]] = None,
        save_dir: str = None,
    ) -> Tuple[Optional[int], Optional[str], Optional[Dict], int, float, str]:
        if not obstacle_distances:
            obstacle_distances = {
                "front": "Unknown",
                "left_30": "Unknown",
                "right_30": "Unknown",
            }

        prompt_bundle = build_action_prompt_bundle(
            next_waypoint=next_waypoint,
            subtask_instruction=subtask_instruction,
            subtask_landmark=subtask_landmark,
            progress_summary=progress_summary,
            waypoint_summary=waypoint_summary,
            detected_landmarks=detected_landmarks,
            obstacle_distances=obstacle_distances,
            landmark_map_info=landmark_map_info,
            allowed_action_names=allowed_action_names,
            model_name=self.config.model,
        )

        images = []
        action_image_input = detection_image if detection_image is not None else first_person_image
        if isinstance(action_image_input, str):
            if action_image_input and os.path.exists(action_image_input):
                images.append(action_image_input)
            else:
                print("  [WARN] No detection image found")
        elif action_image_input is not None:
            images.append(action_image_input)
        else:
            print("  [WARN] No detection image found")

        response = self.call_api(prompt_bundle, images, save_dir=save_dir)
        if not response:
            print("✗ No response from VLM")
            return None, None, None, 0, 0.0, ""

        if not self.validate_response(response):
            return None, None, None, 0, 0.0, ""

        parsed_action = self._parse_action_command(response)
        if parsed_action is None:
            print(f"✗ Invalid action command: {response.get('action')}")
            return None, None, None, 0, 0.0, ""

        action_name, value = parsed_action
        action_variant = self._extract_action_variant(response.get("action"))
        normalized_allowed_actions = None
        if allowed_action_names:
            normalized_allowed_actions = {
                str(name or "").strip().upper()
                for name in allowed_action_names
                if str(name or "").strip()
            }
        if normalized_allowed_actions and action_name not in normalized_allowed_actions:
            print(
                f"✗ Forbidden action under current constraint: {action_name} | "
                f"Allowed: {sorted(normalized_allowed_actions)}"
            )
            response["_forbidden_action_name"] = action_name
            response["_allowed_action_names"] = sorted(normalized_allowed_actions)
            return None, action_name, response, 0, 0.0, prompt_bundle.full_prompt

        if action_name not in action_mapping:
            print(f"✗ Invalid action: {action_name}")
            print(f"✗ Valid actions: {list(action_mapping.keys())}")
            return None, None, None, 0, 0.0, ""

        action_id = action_mapping[action_name]
        degrees = 0
        meters = 0.0
        if action_name in ["TURN_LEFT", "TURN_RIGHT"]:
            degrees = int(value)
            response["action"] = f"{self._format_turn_action_label(action_name, action_variant)} {degrees}deg"
        elif action_name == "MOVE_FORWARD":
            meters = float(value)
            response["action"] = f"{action_name} {meters:g}m"
        elif action_name == "STOP":
            response["action"] = "STOP"

        if action_name in ("TURN_LEFT", "TURN_RIGHT"):
            info = f"{action_name} {degrees}°"
        elif action_name == "MOVE_FORWARD":
            info = f"{action_name} {meters}m"
        else:
            info = action_name
        print(f"  Action: {info} | {response.get('reasoning', '')[:60]}")
        return action_id, action_name, response, degrees, meters, prompt_bundle.full_prompt
