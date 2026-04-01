"""
Base Navigation Controller
底层导航控制基类：环境交互、建图、检测、可视化
"""
import os
import numpy as np
import cv2
import torch
from typing import Dict, Any, List, Optional, Tuple
from types import SimpleNamespace
from torchvision import transforms
from habitat import Config
from habitat.core.simulator import Observations
from habitat_baselines.common.environments import get_env_class

from vlnce_baselines.detection import GroundedSAM
from vlnce_baselines.mapping import Semantic_Mapping, SemanticMapper, SemanticProcessor
from vlnce_baselines.visualization import MapVisualizer
from vlnce_baselines.visualization.obstacle_analysis import (
    calculate_obstacle_distances_from_depth,
    classify_obstacle_distance_text,
    format_distance,
    parse_distance_text_m,
    sample_depth_distance_from_region,
)
from vlnce_baselines.config.core.params.thresholds import OBS_OPEN_M, OBS_RISKY_M
from vlnce_baselines.vlm.support.navigation_config import DIRECTION_CONFIG
from vlnce_baselines.config.core import ConfigHelper, create_category_config
from vlnce_baselines.env.env_utils import construct_envs
from vlnce_baselines.utils.system import get_device


class BaseNavigationController:
    """封装底层环境交互与感知建图能力的基础导航控制器。"""
    
    def __init__(self, config: Config):
        # print("[Init] 配置MAP参数...")
        self.config = ConfigHelper.setup_navigation_config(config)
        self.device = get_device(self.config.TORCH_GPU_ID)
        torch.cuda.set_device(self.device)
        
        self.map_args = self.config.MAP
        self.resolution = self.config.MAP.MAP_RESOLUTION
        self.width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        self.height = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
        self.map_shape = (self.config.MAP.MAP_SIZE_CM // self.resolution,
                         self.config.MAP.MAP_SIZE_CM // self.resolution)
        
        # print("[Init] 初始化Habitat环境...")
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,
        )
        # print(f"[Init] 环境初始化完成，episodes: {self.envs.number_of_episodes}")
        
        # print("[Init] 初始化GroundedSAM...")
        self.segment_module = GroundedSAM(self.config, self.device)
        
        # print("[Init] 初始化Semantic Mapping...")
        mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        mapping_module.eval()
        
        # print("[Init] 初始化Semantic Mapper...")
        self.mapper = SemanticMapper(mapping_module, self.map_shape, self.resolution)
        
        # print("[Init] 初始化Map Visualizer...")
        self.visualizer = MapVisualizer(
            self.config.RESULTS_DIR, 
            self.resolution, 
            self.map_shape,
            enable_global_map_crop=self.config.MAP.ENABLE_GLOBAL_MAP_CROP,
            enable_adaptive_zoom=self.config.MAP.ENABLE_ADAPTIVE_ZOOM,
            debug_save_renderings=bool(getattr(self.config.MAP, "DEBUG_SAVE_RENDERINGS", True)),
        )
        
        self.category_config = create_category_config()
        self.mapping_classes = self.category_config.mapping_classes
        self.landmark_classes = self.category_config.landmark_classes
        self.detection_classes = self.category_config.detection_classes
        self.classes = []
        
        from vlnce_baselines.config.core.constants import landmark_min_area_threshold, landmark_min_total_pixels
        self.landmark_min_area_threshold = landmark_min_area_threshold
        self.landmark_min_total_pixels = landmark_min_total_pixels
        
        self.current_episode_id = None
        self.current_step = 0
        self.latest_landmark_instances_world = []
        self._reset_navigation_runtime_state()

    def _reset_navigation_runtime_state(self) -> None:
        """Reset low-level observation/render caches shared by all controllers."""
        self.latest_obs = None
        self.latest_info = None
        self.latest_done = False
        self.current_episode = None
        self.initial_distance_to_goal = None
        self.reference_path_length = None
        self.final_stop_action_requested = False
        self.final_stop_skipped_due_to_done = False
        self.final_stop_was_executed = False
        self.latest_lookaround_end_reason = ""
        self.latest_depth_meters = None
        self.latest_global_map = None
        self.latest_local_map = None
        self.latest_action_detection_vis = None
        self.latest_lookaround_images: List[np.ndarray] = []
        self.latest_lookaround_depths: List[np.ndarray] = []
        self.latest_lookaround_detection_payloads: List[Any] = []
        self.latest_lookaround_phase = ""
        self.latest_obstacle_distances_12 = {
            f'angle_{i}': 'Unknown' for i in range(0, 360, 30)
        }
        self.latest_obstacle_distances = {
            'front': 'Unknown',
            'left_30': 'Unknown',
            'right_30': 'Unknown',
        }

    def _clear_landmark_detection_cache(self) -> None:
        """清空landmark检测缓存；仅在episode/subtask重置时调用。"""
        self.latest_landmark_dist_map = {}
        self.latest_landmark_dist_map_multi = {}
        self.latest_visible_landmark_entries = []
        self.latest_action_landmark_topk_entries = []

    def _ensure_landmark_detection_state(self) -> None:
        """Lazily initialize per-step landmark caches."""
        if not hasattr(self, 'current_step_landmarks'):
            self.current_step_landmarks = {}
        if not hasattr(self, 'current_step_landmark_entries'):
            self.current_step_landmark_entries = {}
        if not hasattr(self, 'current_step_action_landmark_topk_entries'):
            self.current_step_action_landmark_topk_entries = {}

    @staticmethod
    def _clone_landmark_entries(entries: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return [dict(item) for item in (entries or [])]

    def _record_landmark_detection_step(self, step_idx: int, detected_landmarks_step) -> None:
        """记录当前step的landmark检测结果和地图矫正后的距离/角度。"""
        self._ensure_landmark_detection_state()

        self.current_step_landmarks[step_idx] = detected_landmarks_step or []
        visible_entries = getattr(self, 'latest_visible_landmark_entries', []) or []
        topk_entries = getattr(self, 'latest_action_landmark_topk_entries', []) or []
        self.current_step_landmark_entries[step_idx] = self._clone_landmark_entries(visible_entries)
        self.current_step_action_landmark_topk_entries[step_idx] = self._clone_landmark_entries(topk_entries)

    def _get_detected_custom_landmarks(self) -> set:
        """读取当前帧检测结果中的自定义 landmark 类别。"""
        if not getattr(self, 'landmark_classes', None):
            return set()

        canonical = {name.strip().lower(): name for name in self.landmark_classes}
        detected = set()
        for label in getattr(self, 'latest_labels_full', []) or []:
            parts = label.split()
            label_name = ' '.join(parts[:-1]) if len(parts) > 1 else parts[0]
            matched_name = canonical.get(label_name.strip().lower())
            if matched_name:
                detected.add(matched_name)
        return detected

    def _print_custom_landmark_status(self, detection_enabled: bool) -> None:
        """命令行仅打印当前子任务自定义类别及其识别状态。"""
        tracked_landmarks = list(getattr(self, 'landmark_classes', []) or [])
        if not tracked_landmarks or not detection_enabled:
            return

        detected_landmarks = self._get_detected_custom_landmarks()
        status_items = [
            f"{name}=识别" if name in detected_landmarks else f"{name}=未识别"
            for name in tracked_landmarks
        ]

        print(f"| 自定义类别: {'; '.join(status_items)}")

    def _draw_detection_overlay(self, image: np.ndarray, detections, labels, color) -> np.ndarray:
        """在已有图像上补画一组检测框，便于同时保留多 query 结果。"""
        if image is None:
            return None
        if detections is None or getattr(detections, 'xyxy', None) is None or len(detections.xyxy) == 0:
            return image

        overlay = image.copy()
        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            label = labels[i] if i < len(labels) else f"object_{i}"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
        return overlay

    def _merge_detection_batches(self, rgb: np.ndarray, detection_batches) -> tuple:
        """合并多次检测结果，保留重叠框的多个 query 输出。"""
        merged_labels = []
        merged_masks = []
        xyxy_parts = []
        confidence_parts = []
        class_id_parts = []
        tracker_id_parts = []
        tracker_available = False
        annotated_image = rgb.copy()

        for batch_idx, batch in enumerate(detection_batches):
            masks, labels, batch_annotated, detections, class_offset = batch
            if batch_idx == 0 and batch_annotated is not None:
                annotated_image = batch_annotated.copy()
            else:
                annotated_image = self._draw_detection_overlay(
                    annotated_image, detections, labels, color=(0, 255, 255)
                )

            if labels:
                merged_labels.extend(labels)
            if masks is not None and getattr(masks, 'size', 0) > 0:
                merged_masks.append(masks.astype(np.float32))

            if detections is None or getattr(detections, 'xyxy', None) is None or len(detections.xyxy) == 0:
                continue

            xyxy_parts.append(detections.xyxy.astype(np.float32))

            confidence = getattr(detections, 'confidence', None)
            if confidence is None:
                confidence_parts.append(np.zeros((len(detections.xyxy),), dtype=np.float32))
            else:
                confidence_parts.append(np.asarray(confidence, dtype=np.float32))

            class_id = getattr(detections, 'class_id', None)
            if class_id is None:
                class_id_arr = np.full((len(detections.xyxy),), -1, dtype=np.int32)
            else:
                class_id_arr = np.asarray(class_id, dtype=np.int32).copy()
                valid_mask = class_id_arr >= 0
                class_id_arr[valid_mask] += class_offset
            class_id_parts.append(class_id_arr)

            tracker_id = getattr(detections, 'tracker_id', None)
            if tracker_id is not None:
                tracker_available = True
                tracker_id_parts.append(np.asarray(tracker_id))
            else:
                tracker_id_parts.append(np.full((len(detections.xyxy),), -1, dtype=np.int32))

        merged_mask_array = (
            np.concatenate(merged_masks, axis=0)
            if merged_masks else
            np.zeros((0, self.height, self.width), dtype=np.float32)
        )

        merged_detections = SimpleNamespace(
            xyxy=np.concatenate(xyxy_parts, axis=0) if xyxy_parts else np.zeros((0, 4), dtype=np.float32),
            confidence=np.concatenate(confidence_parts, axis=0) if confidence_parts else np.zeros((0,), dtype=np.float32),
            class_id=np.concatenate(class_id_parts, axis=0) if class_id_parts else np.zeros((0,), dtype=np.int32),
            tracker_id=(
                np.concatenate(tracker_id_parts, axis=0)
                if tracker_available and tracker_id_parts else None
            ),
            mask=merged_mask_array if merged_mask_array.size > 0 else None,
        )

        return merged_mask_array, merged_labels, annotated_image, merged_detections

    def _depth_to_meters(self, depth_obs: np.ndarray) -> np.ndarray:
        """将 Habitat 归一化 depth 转成米制深度图。"""
        if depth_obs is None:
            return None

        depth_raw = depth_obs[:, :, 0] if depth_obs.ndim == 3 else depth_obs
        min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        # Habitat depth is clipped to [MIN_DEPTH, MAX_DEPTH] before normalization.
        # So normalized 0.0 means "at the min-depth clip", not "invalid / missing".
        valid_mask = np.isfinite(depth_raw) & (depth_raw >= 0.0) & (depth_raw <= 1.0)
        return np.where(
            valid_mask,
            min_depth + depth_raw * (max_depth - min_depth),
            0.0,
        ).astype(np.float32)

    def _detect_landmarks_for_visualization(self,
                                            rgb: np.ndarray,
                                            landmark_queries: Optional[List[str]] = None):
        """仅为可视化做自定义 landmark 检测，不写入地图。"""
        queries = landmark_queries if landmark_queries is not None else list(getattr(self, 'landmark_classes', []) or [])
        if not queries:
            return None, [], None

        detection_batches = []
        for lm_idx, landmark_query in enumerate(queries):
            landmark_result = self.segment_module.segment(rgb, classes=[landmark_query])
            detection_batches.append((*landmark_result, lm_idx))

        merged_masks, merged_labels, annotated_image, merged_detections = \
            self._merge_detection_batches(rgb, detection_batches)
        return merged_detections, merged_labels, merged_masks
    
    @property
    def detected_classes(self):
        """便捷访问detected_classes（代理到category_config）"""
        return self.category_config._detected_classes
    
    def reset_episode(self, episode_id: int = None):
        print(f"\n{'='*60}\nEpisode {episode_id if episode_id else 0}\n{'='*60}")
        
        self.envs.reset()
        self.current_step = 0
        self.current_episode_id = episode_id if episode_id is not None else 0
        self._reset_navigation_runtime_state()
        
        self.category_config.reset_detected()
        self.classes = []
        self.latest_landmark_instances_world = []
        self.mapper.reset()
        self.mapper.init_map_and_pose(num_detected_classes=0)
        self.current_step_landmarks = {}
        self.current_step_landmark_entries = {}
        self.current_step_action_landmark_topk_entries = {}
        self._clear_landmark_detection_cache()
        
        current_episodes = self.envs.current_episodes()
        self.current_episode = current_episodes[0]
        self.current_instruction = self.current_episode.instruction.instruction_text
        self.reference_path_length = self._estimate_episode_reference_path_length(self.current_episode)
        self.initial_distance_to_goal = self._load_initial_distance_to_goal()
        
        print(f"Instruction: {self.current_instruction[:100]}{'...' if len(self.current_instruction) > 100 else ''}")

    def _load_initial_distance_to_goal(self) -> Optional[float]:
        try:
            if hasattr(self.envs, 'call_at'):
                metrics = self.envs.call_at(0, 'get_metrics')
                if isinstance(metrics, dict):
                    distance_to_goal = metrics.get('distance_to_goal')
                    if (
                        isinstance(distance_to_goal, (int, float)) and
                        np.isfinite(distance_to_goal) and
                        float(distance_to_goal) >= 0.0
                    ):
                        return float(distance_to_goal)
        except Exception:
            pass

        if (
            isinstance(self.reference_path_length, (int, float)) and
            np.isfinite(self.reference_path_length) and
            float(self.reference_path_length) > 0.0
        ):
            return float(self.reference_path_length)
        return None

    @staticmethod
    def _estimate_episode_reference_path_length(episode: Any) -> Optional[float]:
        reference_path = list(getattr(episode, 'reference_path', None) or [])
        if len(reference_path) < 2:
            return None

        total_distance = 0.0
        previous = None
        for point in reference_path:
            if point is None or len(point) < 3:
                continue
            current = np.asarray(point[:3], dtype=np.float32)
            if previous is not None:
                total_distance += float(np.linalg.norm(current - previous, ord=2))
            previous = current

        return total_distance if total_distance > 0.0 else None

    def _episode_done_cached(self) -> bool:
        """Return whether the current episode has already terminated locally."""
        if bool(getattr(self, 'latest_done', False)):
            return True
        info = getattr(self, 'latest_info', None)
        return bool(isinstance(info, dict) and info.get('done', False))

    def _terminal_step_result(self) -> Dict[str, Any]:
        """Build a consistent terminal result when a step is skipped after done."""
        info = dict(self.latest_info) if isinstance(self.latest_info, dict) else {}
        info['done'] = True
        return {
            'obs': self.latest_obs,
            'reward': 0.0,
            'done': True,
            'info': info,
            'detected_classes': list(self.detected_classes),
        }

    def _cache_env_step_outcome(
        self,
        obs: List[Any],
        dones: List[Any],
        infos: List[Any],
    ) -> Dict[str, Any]:
        """Cache the latest env transition and normalize the info payload."""
        if obs:
            self.latest_obs = obs[0]

        done_flag = bool(dones[0]) if dones else False
        self.latest_done = done_flag

        cached_info: Dict[str, Any] = {}
        if infos and infos[0] is not None:
            if isinstance(infos[0], dict):
                cached_info = dict(infos[0])
            else:
                try:
                    cached_info = dict(infos[0])
                except Exception:
                    cached_info = {"raw_info": infos[0]}
        cached_info['done'] = done_flag
        self.latest_info = cached_info

        if infos:
            infos[0] = cached_info
        return cached_info

    def _safe_env_step(
        self,
        actions: List[Any],
        *,
        context: str,
    ) -> Optional[Tuple[List[Any], List[Any], List[Any], List[Any]]]:
        """Step the vector env only when the episode is still active."""
        if self._episode_done_cached():
            print(f"[WARN] Episode already done, skip {context}")
            return None

        try:
            outputs = self.envs.step(actions)
        except AssertionError as exc:
            self.latest_done = True
            self.latest_lookaround_end_reason = "episode_done"
            if not isinstance(self.latest_info, dict):
                self.latest_info = {}
            self.latest_info['done'] = True
            print(f"[WARN] Episode already done during {context}: {exc}")
            return None

        obs, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        self._cache_env_step_outcome(obs, dones, infos)
        return obs, rewards, dones, infos

    def _store_latest_map_paths(self, paths: Optional[Dict[str, Any]]) -> None:
        if not paths:
            return
        if paths.get('global_map'):
            self.latest_global_map = paths.get('global_map')
        if paths.get('local_map'):
            self.latest_local_map = paths.get('local_map')

    def _save_visualization_snapshot(
        self,
        *,
        map_state: Dict[str, Any],
        rgb_bgr: np.ndarray,
        phase: str,
        step: Optional[int] = None,
        detections=None,
        labels=None,
        masks=None,
        landmark_classes: Optional[List[str]] = None,
        render_policy: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[Any], Optional[float]]:
        """Shared wrapper around `visualizer.save_step_visualization`."""
        if self.visualizer is None:
            return {}, [], None

        paths, detected_landmarks_step, last_waypoint_angle = self.visualizer.save_step_visualization(
            step=self.current_step if step is None else int(step),
            episode_id=self.current_episode_id,
            rgb=rgb_bgr,
            full_map=map_state['full_map'],
            trajectory_points=map_state.get('subtask_trajectory_points', []),
            detected_classes=list(self.detected_classes),
            current_pose=map_state['full_pose'],
            floor=map_state['floor'],
            hfov=self.config.MAP.HFOV,
            detections=detections,
            labels=labels,
            masks=masks,
            landmark_classes=list(landmark_classes if landmark_classes is not None else getattr(self, 'landmark_classes', [])),
            mapping_classes=self.mapping_classes,
            landmark_config={
                'min_total_pixels': self.landmark_min_total_pixels,
                'min_area_threshold': self.landmark_min_area_threshold
            },
            waypoint_positions=map_state.get('waypoint_positions', []),
            waypoint_ids=map_state.get('waypoint_ids', []),
            space_area_layer=map_state.get('space_area_layer'),
            space_area_records=map_state.get('space_area_records', []),
            phase=phase,
            global_trajectory_points=map_state.get('global_trajectory_points', []),
            crop_offset=map_state.get('crop_offset'),
            controller=self,
            render_policy=render_policy,
        )
        self._store_latest_map_paths(paths)
        return paths, detected_landmarks_step, last_waypoint_angle

    def _get_cached_rgb_bgr(self, phase: Optional[str] = None) -> Optional[np.ndarray]:
        """Return the best cached RGB frame for refresh-only rendering."""
        if self.latest_obs is not None and 'rgb' in self.latest_obs:
            return cv2.cvtColor(self.latest_obs['rgb'], cv2.COLOR_RGB2BGR)
        if (
            phase is not None and
            self.latest_lookaround_phase == phase and
            self.latest_lookaround_images
        ):
            return self.latest_lookaround_images[-1].copy()
        return None

    def _refresh_current_map_snapshots(
        self,
        phase: str,
        landmark_classes: Optional[List[str]] = None,
    ) -> bool:
        """Refresh global/local map renders from cached observation and current map state."""
        if self.mapper is None:
            return False

        rgb_bgr = self._get_cached_rgb_bgr(phase=phase)
        if rgb_bgr is None:
            return False

        map_state = self.mapper.get_map_state()
        self._save_visualization_snapshot(
            map_state=map_state,
            rgb_bgr=rgb_bgr,
            phase=phase,
            landmark_classes=landmark_classes,
        )
        return True

    def _get_agent_pose(self) -> tuple:
        """Read the current Habitat agent pose from environment 0."""
        return self.envs.call_at(0, "get_agent_pose")

    def _draw_waypoints_on_view(self, image: np.ndarray, waypoint_entry: Dict[str, Any]) -> np.ndarray:
        """Draw the visible space-waypoint label and distance on a thinking view."""
        if not waypoint_entry:
            return image

        label_text = str(
            waypoint_entry.get("label")
            or waypoint_entry.get("area_label")
            or waypoint_entry.get("description")
            or "Unknown"
        ).strip()
        if not label_text:
            return image

        try:
            distance_text = f"{float(waypoint_entry.get('distance_m', 0.0)):.1f}m"
        except (TypeError, ValueError):
            distance_text = "unknown"

        display_text = f"{label_text} {distance_text}".strip()
        h, w = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.62
        thickness = 2
        padding_x = 8
        padding_y = 6
        (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)

        box_w = text_w + padding_x * 2
        box_h = text_h + baseline + padding_y * 2
        box_x1 = max(8, (w - box_w) // 2)
        box_y1 = max(12, h // 2 - box_h - 18)
        box_x2 = min(w - 8, box_x1 + box_w)
        box_y2 = min(h - 8, box_y1 + box_h)

        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 2)
        text_x = box_x1 + padding_x
        text_y = box_y2 - baseline - padding_y
        cv2.putText(
            image,
            display_text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        return image

    def _draw_distance_rays_on_first_person_view(self, image: np.ndarray, distances: Dict[str, str]) -> np.ndarray:
        """Overlay the action-view obstacle rays on the first-person image."""
        h, w = image.shape[:2]
        center_x, bottom_y = w // 2, h - 20
        hfov = float(self.config.MAP.HFOV)
        fov_half = hfov / 2.0
        ray_map = {
            'left_30': -30,
            'front': 0,
            'right_30': 30,
        }

        for key, angle in ray_map.items():
            if key not in distances or abs(angle) > fov_half:
                continue

            dist_str = distances[key]
            status = classify_obstacle_distance_text(dist_str)
            if status == "blocked":
                color, y_ratio = (0, 0, 255), 0.7
            elif status == "open":
                color, y_ratio = (0, 255, 0), 0.1
            else:
                try:
                    dist_val = float(parse_distance_text_m(dist_str))
                    color = (0, 255, 255)
                    y_ratio = (
                        0.7 if dist_val < float(OBS_RISKY_M)
                        else (0.5 if dist_val < float(OBS_OPEN_M) else 0.3)
                    )
                except Exception:
                    color, y_ratio = (0, 255, 255), 0.5

            x_ratio = (angle + fov_half) / (2 * fov_half)
            end_x, end_y = int(x_ratio * w), int(bottom_y - bottom_y * y_ratio)
            cv2.line(image, (center_x, bottom_y), (end_x, end_y), color, 2)
            text_x = end_x - len(dist_str) * 3
            text_y = end_y - 5
            cv2.rectangle(image, (text_x - 2, text_y - 12), (text_x + len(dist_str) * 7, text_y + 2), (0, 0, 0), -1)
            cv2.putText(image, dist_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return image
    
    def look_around(self) -> None:
        """360度环视建图(12步×30°)，步数0-11"""
# print("🔄 360°环视...", end="", flush=True)
        
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        
        for step in range(12):
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            step_data = self._safe_env_step(actions, context=f"lookaround scan {step + 1}/12")
            if step_data is None:
                print(" [WARN] Episode ended early")
                self.current_step = step
                return
            obs, _, dones, _ = step_data
            
            if dones[0]:
                print(" [WARN] Episode ended early")
                self.current_step = step + 1
                return
            
            prev_class_count = len(self.detected_classes)
            batch_obs = self._batch_obs(obs, save_object_detection=False, step=step)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            
            map_state = self.mapper.update_map(
                batch_obs, poses, step,
                list(self.detected_classes), self.current_episode_id,
                observations=obs,
            )
            
            new_classes = len(self.detected_classes) - prev_class_count
            # 不再打印每步的进度，只在最后汇总
        
        self.current_step = 12
        # print(f" ✅ {len(self.detected_classes)}类")

    def _on_lookaround_step(
        self,
        *,
        phase: str,
        look_index: int,
        look_step: int,
        obs: Dict[str, Any],
        info: Dict[str, Any],
    ) -> None:
        """Subclass hook for per-step side effects during lookaround."""
        return None

    def _capture_lookaround_scan(
        self,
        phase: str,
        enable_landmark_detection: bool = False,
        prepare_thinking_detection: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Run the shared 12-step lookaround scan and cache the resulting map/render state."""
        from habitat.sims.habitat_simulator.actions import HabitatSimActions

        debug_save_renderings = bool(getattr(self.config.MAP, 'DEBUG_SAVE_RENDERINGS', True))
        self.latest_lookaround_end_reason = ""
        lookaround_images: List[np.ndarray] = []
        lookaround_depths: List[Optional[np.ndarray]] = []
        lookaround_detection_payloads: List[Any] = []
        final_map_state = None
        final_snapshot_paths: Dict[str, Any] = {}
        final_last_waypoint_angle = None
        look_step = self.current_step

        for look_index in range(1, 13):
            self.current_step += 1
            look_step = self.current_step

            step_data = self._safe_env_step(
                [{"action": HabitatSimActions.TURN_LEFT}],
                context=f"lookaround step {look_index}/12",
            )
            if step_data is None:
                self.current_step = max(0, self.current_step - 1)
                self.latest_lookaround_end_reason = "episode_done"
                print(f"[WARN] Episode ended at lookaround step {look_index}/12")
                return None
            obs, _, dones, infos = step_data

            if dones[0]:
                self.latest_lookaround_end_reason = "episode_done"
                print(f"[WARN] Episode ended at lookaround step {look_index}/12")
                return None

            batch_obs = self._batch_obs(obs, save_object_detection=enable_landmark_detection)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)

            map_state = self.mapper.update_map(
                batch_obs, poses, look_step,
                list(self.detected_classes), self.current_episode_id,
                observations=obs,
            )
            final_map_state = map_state

            rgb_bgr = cv2.cvtColor(obs[0]['rgb'], cv2.COLOR_RGB2BGR)
            lookaround_images.append(rgb_bgr.copy())
            lookaround_depths.append(self._depth_to_meters(obs[0]['depth']))
            if prepare_thinking_detection and getattr(self, 'landmark_classes', None):
                detections, labels, masks = self._detect_landmarks_for_visualization(
                    rgb_bgr,
                    list(self.landmark_classes),
                )
                lookaround_detection_payloads.append((detections, list(labels or []), masks))
            else:
                lookaround_detection_payloads.append((None, [], None))

            if debug_save_renderings:
                final_snapshot_paths, _detected_landmarks_step, final_last_waypoint_angle = self._save_visualization_snapshot(
                    map_state=map_state,
                    rgb_bgr=rgb_bgr,
                    phase=phase,
                    step=look_step,
                    detections=None,
                    labels=None,
                    masks=None,
                    landmark_classes=list(self.landmark_classes) if enable_landmark_detection else [],
                )

            # Always expose every real lookaround env step to the navigation visualizer,
            # even when debug map/rgb render dumps are disabled.
            self._on_lookaround_step(
                phase=phase,
                look_index=look_index,
                look_step=look_step,
                obs=obs[0],
                info=infos[0] if infos and len(infos) > 0 else {},
            )

        self.latest_obs = obs[0]
        self._update_obstacle_distances_12_directions(lookaround_depths)

        if len(lookaround_images) < 12:
            self.latest_lookaround_end_reason = "incomplete"
            print(f"[WARN] Lookaround incomplete: {len(lookaround_images)}/12 images")
            return None

        self.latest_lookaround_images = [img.copy() for img in lookaround_images]
        self.latest_lookaround_depths = [
            depth.copy() if depth is not None else None for depth in lookaround_depths
        ]
        self.latest_lookaround_detection_payloads = list(lookaround_detection_payloads)
        self.latest_lookaround_phase = str(phase)

        if final_map_state is not None and not debug_save_renderings:
            final_snapshot_paths, _detected_landmarks_step, final_last_waypoint_angle = self._save_visualization_snapshot(
                map_state=final_map_state,
                rgb_bgr=lookaround_images[-1].copy(),
                phase=phase,
                step=look_step,
                landmark_classes=[],
            )

        return {
            "look_step": look_step,
            "lookaround_images": lookaround_images,
            "lookaround_depths": lookaround_depths,
            "lookaround_detection_payloads": lookaround_detection_payloads,
            "final_map_state": final_map_state,
            "final_snapshot_paths": final_snapshot_paths,
            "final_last_waypoint_angle": final_last_waypoint_angle,
        }

    def step(self, action: int, save_vis: bool = True, phase: str = "action",
             enable_landmark_detection: bool = True) -> Dict[str, Any]:
        """执行一步动作，更新地图并保存可视化

        Args:
            action: Habitat动作ID
            save_vis: 是否保存可视化
            phase: 文件命名阶段
            enable_landmark_detection: 是否启用landmark检测（False时仅检测mapping_classes）
        """
        if self._episode_done_cached():
            print(f"[WARN] Episode already done, skip {self._action_name(action)}")
            return self._terminal_step_result()

        # ⚠️ 关键修复：在使用current_step之前先累加，避免覆盖环视最后一步
        self.current_step += 1
        
        print(f"[{self.current_step}]{self._action_name(action)}", end=" ")
        
        step_data = self._safe_env_step([action], context=f"{self._action_name(action)} step")
        if step_data is None:
            self.current_step = max(0, self.current_step - 1)
            print(" → Episode已结束，跳过")
            return self._terminal_step_result()
        obs, rewards, dones, infos = step_data
        
        if dones[0]:
            print(" → Episode结束")
            return {
                'obs': obs[0],
                'reward': rewards[0],
                'done': dones[0],
                'info': infos[0],
                'detected_classes': list(self.detected_classes)
            }
        
        prev_class_count = len(self.detected_classes)
        batch_obs = self._batch_obs(obs, save_object_detection=enable_landmark_detection)
        poses = torch.from_numpy(
            np.array([item['sensor_pose'] for item in obs])
        ).float().to(self.device)
        
        map_state = self.mapper.update_map(
            batch_obs, poses, self.current_step,
            list(self.detected_classes), self.current_episode_id,
            observations=obs,
        )
        
        # print(f"[Controller.step] 从mapper接收轨迹: 全局={len(map_state.get('global_trajectory_points', []))}, 子任务={len(map_state.get('subtask_trajectory_points', []))}")
        
        new_classes = len(self.detected_classes) - prev_class_count
# print(f" +{new_classes}类" if new_classes > 0 else "")
        
        if save_vis:
            if enable_landmark_detection:
                vis_landmark_classes = self.landmark_classes
                vis_detections = self.latest_detections_full if hasattr(self, 'latest_detections_full') else None
                vis_labels = self.latest_labels_full if hasattr(self, 'latest_labels_full') else None
                vis_masks = self.latest_masks_full if hasattr(self, 'latest_masks_full') else None
            else:
                # 即使当前帧不做新的 landmark 检测，也继续渲染地图中已累计的 custom landmark
                vis_landmark_classes = self.landmark_classes
                vis_detections = None
                vis_labels = None
                vis_masks = None
            rgb_bgr = cv2.cvtColor(obs[0]['rgb'], cv2.COLOR_RGB2BGR)
            _, detected_landmarks_step, _ = self._save_visualization_snapshot(
                map_state=map_state,
                rgb_bgr=rgb_bgr,
                phase=phase,
                detections=vis_detections,
                labels=vis_labels,
                masks=vis_masks,
                landmark_classes=vis_landmark_classes,
            )
            
            # 记录当前step的landmark检测结果（用于action决策与去重）
            if enable_landmark_detection:
                self._record_landmark_detection_step(self.current_step, detected_landmarks_step)

        self._print_custom_landmark_status(enable_landmark_detection)
        
        return {
            'obs': obs[0],
            'reward': rewards[0],
            'done': dones[0],
            'info': infos[0],
            'detected_classes': list(self.detected_classes)
        }
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        map_state = self.mapper.get_map_state()
        
        return {
            'step': self.current_step,
            'episode_id': self.current_episode_id,
            'full_map': map_state['full_map'],
            # 'trajectory_points': map_state['trajectory_points'],  # 已废弃，轨迹在 Channel 2
            'floor': map_state['floor'],
            'detected_classes': list(self.detected_classes),
            'current_pose': map_state['full_pose']
        }

    def _refresh_step_visualization_snapshot(
        self,
        phase: str,
        enable_landmark_detection: bool = False,
        force: bool = False,
    ) -> bool:
        """Re-render the current step visualization from cached obs without re-fusing the map."""
        if self.latest_obs is None:
            return False

        render_policy = self.visualizer.get_render_policy(phase)
        required_paths = []
        if render_policy.get('save_rgb', False):
            required_paths.append(
                os.path.join(self.current_episode_dir, 'rgb', f'step_{self.current_step:04d}_{phase}.png')
            )
        if render_policy.get('save_global_map', False):
            required_paths.append(
                os.path.join(self.current_episode_dir, 'global_map', f'step_{self.current_step:04d}_{phase}.png')
            )
        if render_policy.get('save_local_map', False):
            required_paths.append(
                os.path.join(self.current_episode_dir, 'local_map', f'step_{self.current_step:04d}_{phase}.png')
            )
        if enable_landmark_detection and render_policy.get('save_detection', False):
            required_paths.append(
                os.path.join(self.current_episode_dir, 'detection', f'step_{self.current_step:04d}_{phase}.png')
            )

        if not force:
            if enable_landmark_detection and hasattr(self, 'current_step_landmarks'):
                if self.current_step in self.current_step_landmarks:
                    return True
            if required_paths and all(os.path.exists(path) for path in required_paths):
                if not enable_landmark_detection:
                    return True
                if hasattr(self, 'current_step_landmarks') and self.current_step in self.current_step_landmarks:
                    return True

        self._batch_obs([self.latest_obs], save_object_detection=enable_landmark_detection)
        map_state = self.mapper.get_map_state()

        rgb_bgr = cv2.cvtColor(self.latest_obs['rgb'], cv2.COLOR_RGB2BGR)
        landmark_classes = list(self.landmark_classes) if enable_landmark_detection else []
        detections = self.latest_detections_full if enable_landmark_detection and hasattr(self, 'latest_detections_full') else None
        labels = self.latest_labels_full if enable_landmark_detection and hasattr(self, 'latest_labels_full') else None
        masks = self.latest_masks_full if enable_landmark_detection and hasattr(self, 'latest_masks_full') else None

        _, detected_landmarks_step, _ = self._save_visualization_snapshot(
            map_state=map_state,
            rgb_bgr=rgb_bgr,
            phase=phase,
            detections=detections,
            labels=labels,
            masks=masks,
            landmark_classes=landmark_classes,
            render_policy=render_policy,
        )

        if enable_landmark_detection:
            self._record_landmark_detection_step(self.current_step, detected_landmarks_step)
        return True

    def _run_pre_action_detection_snapshot(self, action_phase: str) -> bool:
        """Refresh the current action-LLM frame with landmark detection before the next action call."""
        return self._refresh_step_visualization_snapshot(
            phase=action_phase,
            enable_landmark_detection=True,
            force=False,
        )

    def _refresh_post_action_landmark_detection_state(self, action_phase: str) -> bool:
        """Refresh the latest moved-to frame for action landmark memory and auto-stop checks."""
        if self.latest_obs is None or self.mapper is None:
            return False
        if not getattr(self, "landmark_classes", None):
            return False

        self._batch_obs([self.latest_obs], save_object_detection=True)
        map_state = self.mapper.get_map_state()
        rgb_bgr = cv2.cvtColor(self.latest_obs["rgb"], cv2.COLOR_RGB2BGR)
        _, detected_landmarks_step, _ = self._save_visualization_snapshot(
            map_state=map_state,
            rgb_bgr=rgb_bgr,
            phase=action_phase,
            detections=self.latest_detections_full if hasattr(self, "latest_detections_full") else None,
            labels=self.latest_labels_full if hasattr(self, "latest_labels_full") else None,
            masks=self.latest_masks_full if hasattr(self, "latest_masks_full") else None,
            landmark_classes=list(self.landmark_classes),
            render_policy={
                "save_rgb": False,
                "render_global_map": False,
                "save_global_map": False,
                "render_local_map": False,
                "save_local_map": False,
                "render_detection": True,
                "save_detection": False,
            },
        )
        self._record_landmark_detection_step(self.current_step, detected_landmarks_step)
        return True

    def _update_obstacle_distances_12_directions(self, lookaround_depths: Optional[List[np.ndarray]] = None):
        """Update 12-view obstacle distances from depth, with per-view map fallback only when depth is unknown."""
        depth_views = list(lookaround_depths or [])
        map_fallback = {}
        try:
            if self.mapper is not None and self.visualizer is not None:
                map_state = self.mapper.get_map_state()
                map_fallback = self.visualizer.calculate_obstacle_distances_12_directions_from_full_map(
                    map_state.get('full_map'),
                )
        except Exception:
            map_fallback = {}
        distances = {}
        for config in DIRECTION_CONFIG:
            step_idx = int(config["step"])
            angle = int(config["angle"])
            # Keep 12-view obstacle text aligned with the exact rendered IMAGE:
            # IMAGE 1 (Front 0deg) uses the step-12 depth frame, IMAGE 2 uses step-1, etc.
            depth_meters = depth_views[step_idx - 1] if step_idx - 1 < len(depth_views) else None
            try:
                distance_m = sample_depth_distance_from_region(
                    depth_meters,
                    center_x_ratio=0.5,
                    width_ratio=0.26,
                    row_start_ratio=0.38,
                    row_end_ratio=0.92,
                    max_distance_m=5.0,
                    sensor_min_depth_m=float(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH),
                    sample_count=96,
                    sample_percentile=20.0,
                )
            except Exception:
                distance_m = None
            dist_key = f'angle_{angle}'
            distances[dist_key] = (
                format_distance(distance_m)
                if distance_m is not None else
                map_fallback.get(dist_key, "Unknown")
            )
        self.latest_obstacle_distances_12 = distances

    def _update_obstacle_distances(self):
        """Update action-view obstacle distances from current depth only."""
        try:
            self.latest_obstacle_distances = calculate_obstacle_distances_from_depth(
                getattr(self, 'latest_depth_meters', None),
                hfov_deg=float(self.config.MAP.HFOV),
                angle_band_deg=5.0,
                sensor_min_depth_m=float(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH),
                fallback_distances=None,
            )
        except Exception:
            self.latest_obstacle_distances = {
                'front': 'Unknown',
                'left_30': 'Unknown',
                'right_30': 'Unknown',
            }

    @property
    def current_episode_dir(self) -> str:
        """Return the current episode output directory if the subclass uses RESULTS_DIR layout."""
        return os.path.join(self.config.RESULTS_DIR, f'episode_{self.current_episode_id}')
    
    def finish_episode(self, success: bool = False, stop_action: bool = False) -> dict:
        """
        Episode结束总结
        
        重要：调用STOP动作以正确触发Habitat的Success判定
        Success需要同时满足:
        1. distance_to_goal < SUCCESS_DISTANCE (3米)
        2. is_stop_called = True (必须调用STOP动作)
        
        Returns:
            final_metrics: 调用STOP后的最终评估指标
        """
        final_metrics = {}
        self.final_stop_action_requested = bool(stop_action)
        self.final_stop_skipped_due_to_done = False
        self.final_stop_was_executed = False
        
        # 检查episode是否已经结束（避免在已done的episode上调用step）
        episode_already_done = self._episode_done_cached()
        
        if stop_action and not episode_already_done:
            try:
                self.current_step += 1
                step_data = self._safe_env_step([0], context="final STOP")
                if step_data is None:
                    self.current_step = max(0, self.current_step - 1)
                    self.final_stop_skipped_due_to_done = True
                    if hasattr(self, 'latest_info') and self.latest_info:
                        final_metrics = self.latest_info.copy()
                    step_data = None
                if step_data is not None:
                    self.final_stop_was_executed = True
                    _observations, _rewards, _dones, infos = step_data
                    if _observations and len(_observations) > 0:
                        self.latest_obs = _observations[0]
                    if infos and len(infos) > 0:
                        final_metrics = infos[0]
                        self.latest_info = infos[0]
            except AssertionError:
                self.final_stop_skipped_due_to_done = True
                if hasattr(self, 'latest_info') and self.latest_info:
                    final_metrics = self.latest_info.copy()
            except Exception as e:
                print(f"[ERR] STOP failed: {e}")
                final_metrics = {}
        elif stop_action and episode_already_done:
            self.final_stop_skipped_due_to_done = True
            if hasattr(self, 'latest_info') and self.latest_info:
                final_metrics = self.latest_info.copy()
        else:
            if self.latest_info:
                final_metrics = self.latest_info.copy()

        return final_metrics
    
    def _concat_obs(self, obs: Observations) -> np.ndarray:
        """合并RGB和Depth"""
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        return state
    
    def _get_sem_pred(self, rgb: np.ndarray, save_object_detection: bool = False, step: int = None) -> np.ndarray:
        """
        语义分割：GroundedSAM检测 + Winner-Takes-All
        
        检测逻辑：
        - 默认不检测固定 mapping_classes，减少扫描建图开销
        - obstacle / explored 来自深度建图，不依赖语义检测
        - 仅在 save_object_detection=True 时检测当前自定义 landmark
        - 自定义 landmark 检测结果 → 投影到额外 landmark 通道，用于可视化与地图距离/方向
        
        Returns:
            semantic_masks: [H, W, 15] 固定15个通道的语义地图
        """
        # 为了压缩扫描/建图开销，默认不再跑固定 mapping 类别检测：
        # - obstacle / explored 由深度建图提供
        # - floor 由 explored 与 obstacle 推导
        # - 仅在需要时独立检测当前自定义 landmark，并把结果投影进额外 landmark 通道
        landmark_classes_list = (
            self.landmark_classes
            if (save_object_detection and hasattr(self, 'landmark_classes'))
            else []
        )

        if not landmark_classes_list:
            self.mapper.mapping_module.rgb_vis = rgb.copy()
            self.latest_detections_full = None
            self.latest_labels_full = []
            self.latest_masks_full = np.zeros((0, self.height, self.width), dtype=np.float32)
            self.latest_rgb_original = rgb.copy()
            return np.zeros((self.height, self.width, len(self.mapping_classes)), dtype=np.float32)

        detection_batches = []
        for lm_idx, landmark_query in enumerate(landmark_classes_list):
            landmark_result = self.segment_module.segment(rgb, classes=[landmark_query])
            detection_batches.append((*landmark_result, lm_idx))

        masks_all, labels_all, annotated_images, current_detections = \
            self._merge_detection_batches(rgb, detection_batches)
        self.mapper.mapping_module.rgb_vis = annotated_images
        
        self.latest_detections_full = current_detections
        self.latest_labels_full = labels_all.copy()
        self.latest_masks_full = masks_all.copy()  # 保存原始masks用于地面分割
        self.latest_rgb_original = rgb.copy()
        
        # 预定义的基础类别（固定15个）
        predefined_classes = self.mapping_classes
        
        # 分类处理检测结果
        valid_masks = []        # 用于建图的mapping类别
        valid_labels = []
        valid_confidences = []
        # Landmark masks (投影到地图的额外通道，channel 3+N_mapping 开始)
        # 仅在 save_object_detection=True 时启用landmark检测（action LLM 当前朝向快照）
        landmark_masks = np.zeros((len(landmark_classes_list), self.height, self.width), dtype=np.float32)
        lm_name_to_idx = {lm.strip().lower(): idx for idx, lm in enumerate(landmark_classes_list)}

        for i, label in enumerate(labels_all):
            parts = label.split()
            label_name = ' '.join(parts[:-1]) if len(parts) > 1 else parts[0]
            label_name_norm = label_name.strip().lower()
            confidence = float(parts[-1]) if len(parts) > 1 else 0.5
            
            # 只有mapping_classes的检测进入语义地图
            if label_name in predefined_classes:
                valid_masks.append(masks_all[i])
                valid_labels.append(label_name)
                valid_confidences.append(confidence)
            
            # Landmark classes：收集mask投影到地图额外通道（仅精确短语匹配，避免与通用类混淆）
            if label_name_norm in lm_name_to_idx:
                lm_idx = lm_name_to_idx[label_name_norm]
                landmark_masks[lm_idx] = np.maximum(
                    landmark_masks[lm_idx], masks_all[i].astype(np.float32))
            
            # 所有检测到的类别都记录（包括landmark）
            self.detected_classes.add(label_name)
            # 规范化精确匹配到landmark时，记录canonical短语名
            if label_name_norm in lm_name_to_idx:
                self.detected_classes.add(landmark_classes_list[lm_name_to_idx[label_name_norm]])

        global_masks = np.zeros((len(predefined_classes), self.height, self.width), dtype=np.float32)

        if len(valid_masks) > 0:
            # Winner-Takes-All处理（只处理mapping类别）
            valid_masks = np.array(valid_masks)
            masks_processed = self._process_masks_with_labels(valid_masks, valid_labels, valid_confidences)

            # 按照预定义类别顺序组织mask（固定15通道）
            for i, cls_name in enumerate(valid_labels):
                if cls_name in predefined_classes:
                    global_idx = predefined_classes.index(cls_name)
                    if i < masks_processed.shape[0]:
                        global_masks[global_idx] = masks_processed[i]
        
        # 合并mapping通道 + landmark通道：[15+N, H, W] → [H, W, 15+N]
        combined = np.concatenate([global_masks, landmark_masks], axis=0)

        return combined.transpose(1, 2, 0)  # [H, W, 15+N_lm]
    
    def _process_masks_with_labels(self, masks: np.ndarray, labels: list, confidences: list = None) -> np.ndarray:
        """Winner-Takes-All掩码处理"""
        return SemanticProcessor.apply_winner_takes_all(
            masks, labels, confidences, self.height, self.width
        )
    
    def _preprocess_depth(self, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """预处理深度图"""
        depth = depth[:, :, 0] * 1
        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()
        mask2 = depth > 0.99
        depth[mask2] = 0.
        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_depth * 100.0 + depth * max_depth * 100.0
        return depth
    
    def _preprocess_state(self, state: np.ndarray, save_object_detection: bool = False, step: int = None) -> np.ndarray:
        """预处理状态：RGB+Depth+Semantic"""
        state = state.transpose(1, 2, 0)
        rgb = state[:, :, :3].astype(np.uint8)
        rgb = rgb[:,:,::-1]
        depth = state[:, :, 3:4]
        
        min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        env_frame_width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH

        # 保存原始深度（米）用于landmark距离可视化（在预处理之前）
        self.latest_depth_meters = self._depth_to_meters(state[:, :, 3:4])

        sem_seg_pred = self._get_sem_pred(rgb, save_object_detection, step)
        depth = self._preprocess_depth(depth, min_depth, max_depth)
        
        ds = env_frame_width // self.map_args.FRAME_WIDTH
        if ds != 1:
            trans = transforms.Resize((self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH))
            rgb_tensor = torch.from_numpy(rgb.astype(np.uint8)).permute(2,0,1)
            rgb = np.asarray(trans(rgb_tensor).permute(1,2,0))
            depth = depth[ds//2::ds, ds//2::ds]
            sem_seg_pred = sem_seg_pred[ds//2::ds, ds//2::ds]
        
        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)
        return state
    
    def _preprocess_obs(self, obs: np.ndarray, save_object_detection: bool = False, step: int = None) -> np.ndarray:
        """预处理观察"""
        concated_obs = self._concat_obs(obs)
        return self._preprocess_state(concated_obs, save_object_detection, step)
    
    def _batch_obs(self, n_obs: list, save_object_detection: bool = False, step: int = None) -> torch.Tensor:
        """批处理观察"""
        n_states = [self._preprocess_obs(obs, save_object_detection, step) for obs in n_obs]
        max_channels = max([len(state) for state in n_states])
        batch = np.stack([np.pad(state,
                [(0, max_channels - state.shape[0]),
                 (0, 0),
                 (0, 0)],
                mode='constant')
         for state in n_states], axis=0)
        return torch.from_numpy(batch).float().to(self.device)
    
    def toggle_trajectory(self):
        status = self.mapper.toggle_trajectory()
        # print(f"[轨迹] {status}")
    
    def clear_trajectory(self):
        self.mapper.clear_trajectory()
        # print("[轨迹] 已清空")
    
    def get_keyboard_action(self) -> int:
        """获取键盘输入：w=前进 a=左转 d=右转 t=切换轨迹 c=清空轨迹"""
        a = input("action: ")
        if a == 'w':
            return 1
        elif a == 'a':
            return 2
        elif a == 'd':
            return 3
        elif a == 't':
            self.toggle_trajectory()
            return self.get_keyboard_action()
        elif a == 'c':
            self.clear_trajectory()
            return self.get_keyboard_action()
        else:
            return 0
    
    @staticmethod
    def _action_name(action: int) -> str:
        names = {0: 'STOP', 1: 'FORWARD', 2: 'LEFT', 3: 'RIGHT'}
        return names.get(action, f'UNKNOWN({action})')
    
    def close(self):
        # print("\n[Close] 关闭环境...")
        self.envs.close()
        # print("[Close] 完成！")
