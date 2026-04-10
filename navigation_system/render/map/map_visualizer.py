"""Visualization orchestrator for semantic maps, detections, and render caching."""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from navigation_system.runtime.storage.artifacts import get_episode_detail_dir
from navigation_system.runtime.storage.naming import (
    build_step_artifact_filename,
    build_subtask_name_from_token,
)
from navigation_system.space.description.direction_format import format_relative_direction
from navigation_system.space.geometry.map_projection import RotatedMapProjector
from navigation_system.space.map.obstacle_analysis import (
    ACTION_VIEW_DIRECTIONS,
    build_rotated_obstacle_mask,
    calculate_obstacle_distances_from_depth as scan_obstacle_distances_from_depth,
    calculate_obstacle_distances_12_directions as scan_obstacle_distances_12_directions,
    calculate_obstacle_distances_from_rotated_map as scan_obstacle_distances_from_rotated_map,
)
from navigation_system.config.core.params.api import (
    ACTION_VIEW_MODEL_CONTENT_WIDTH,
)
from navigation_system.config.core.constants import (
    color_palette,
    landmark_duplicate_angle_diff_deg,
    landmark_duplicate_iou_loose,
    landmark_duplicate_iou_strict,
    landmark_duplicate_rel_dist_m,
    local_map_landmark_topk,
)
from navigation_system.render.image_resize import (
    resize_image_to_width,
    resize_strip_to_width,
)


class MapVisualizer:
    """Thin coordinator that owns render policy, caching, and save orchestration."""

    GLOBAL_TRAJECTORY_COLOR = (0, 0, 170)
    LOCAL_TRAJECTORY_COLOR = (0, 0, 170)
    SPACE_AREA_COLOR_PALETTE: Tuple[Tuple[int, int, int], ...] = (
        (255, 219, 182),  # powder blue
        (186, 214, 255),  # peach
        (199, 190, 255),  # rose pink
        (255, 203, 224),  # lavender
        (179, 236, 255),  # butter yellow
        (200, 240, 193),  # mint
        (171, 195, 244),  # soft coral
        (235, 231, 189),  # aqua
        (255, 206, 199),  # periwinkle
        (224, 196, 232),  # mauve
    )
    SPACE_TYPE_PREFERRED_COLOR_INDEX: Dict[str, int] = {
        "hallway": 0,
        "stairs": 6,
        "living room": 1,
        "bedroom": 3,
        "kitchen": 4,
        "bathroom": 7,
        "dining room": 2,
        "office": 8,
        "laundry room": 5,
        "closet": 9,
        "balcony": 7,
        "garage": 8,
    }
    SPACE_TYPE_ALIAS_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("hallway", ("hallway", "hall", "corridor", "passage", "entry", "entryway", "foyer", "doorway")),
        ("stairs", ("stairs", "stair", "stairway", "staircase", "landing")),
        ("living room", ("living room", "living area", "lounge", "family room", "sitting room")),
        ("bedroom", ("bedroom", "bed room", "master bedroom", "guest bedroom", "nursery")),
        ("bathroom", ("bathroom", "restroom", "washroom", "toilet room")),
        ("kitchen", ("kitchen", "kitchenette")),
        ("dining room", ("dining room", "dining area", "breakfast room")),
        ("office", ("office", "study", "workspace", "work room")),
        ("laundry room", ("laundry room", "utility room")),
        ("closet", ("closet", "wardrobe")),
        ("garage", ("garage",)),
        ("balcony", ("balcony", "patio", "terrace", "deck")),
    )

    @staticmethod
    def _build_local_map_landmark_debug_lines(
        selected_instances: Optional[List[Dict[str, Any]]],
        topk: int = 2,
    ) -> List[str]:
        lines: List[str] = []
        ordered = sorted(
            [dict(item) for item in (selected_instances or []) if isinstance(item, dict)],
            key=lambda item: (
                int(item.get("selection_rank", 1e9) or 1e9),
                -float(item.get("confidence", 0.0) or 0.0),
                float(item.get("distance_m", 1e9) or 1e9),
                str(item.get("name", "")),
            ),
        )
        for item in ordered[:max(1, int(topk))]:
            name = str(item.get("name") or "Unknown").strip() or "Unknown"
            try:
                display_id = int(item.get("display_id"))
            except (TypeError, ValueError):
                display_id = None
            try:
                distance_m = float(item.get("distance_m"))
                distance_text = f"{distance_m:.2f}m"
            except (TypeError, ValueError):
                distance_text = "Unknown"
            try:
                angle_deg = float(item.get("angle_deg"))
                direction_text = format_relative_direction(angle_deg)
            except (TypeError, ValueError):
                direction_text = "Unknown"
            try:
                confidence = float(item.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            prefix = f"#{display_id}" if display_id is not None and display_id > 0 else "#?"
            lines.append(
                f"{prefix} {name} | {distance_text} | {direction_text} | conf: {confidence:.3f}"
            )
        return lines

    def __init__(self, 
                 results_dir: str,
                 resolution: int = 5,
                 map_shape: Tuple[int, int] = (480, 480),
                 enable_global_map_crop: bool = False,
                 enable_adaptive_zoom: bool = False,
                 debug_save_renderings: bool = True,
                 save_step_map_artifacts: bool = False):
        """
        Args:
            results_dir: 保存根目录（如：data/manual_navigation）
            resolution: 地图分辨率（cm/pixel）
            map_shape: 地图尺寸
            enable_global_map_crop: 是否裁剪global map到440×440（默认False，保持480×480）
            enable_adaptive_zoom: 是否启用自适应缩放（根据地图内容动态放大显示区域）
        """
        self.results_dir = results_dir
        self.resolution = resolution
        self.map_shape = map_shape
        self.enable_global_map_crop = enable_global_map_crop
        self.enable_adaptive_zoom = enable_adaptive_zoom
        self.debug_save_renderings = bool(debug_save_renderings)
        self.save_step_map_artifacts = bool(save_step_map_artifacts)
        self.color_palette = [int(x * 255.) for x in color_palette]
        self._render_cache: Dict[str, Dict[Any, Any]] = {
            "obstacle_mask_raw": {},
            "obstacle_mask_logic": {},
            "obstacle_mask_display": {},
            "space_area_layer": {},
            "space_area_mask": {},
            "space_area_contours": {},
            "space_area_label_anchor": {},
            "space_area_color_assignments": {},
        }
        self._active_render_cache_key = None

    def _build_map_projector(
        self,
        full_map: Optional[np.ndarray],
        current_pose: Optional[Tuple[float, float, float]],
        crop_offset: Optional[Tuple[int, int]],
    ) -> Optional[RotatedMapProjector]:
        if full_map is None or current_pose is None or crop_offset is None:
            return None
        return RotatedMapProjector(
            map_h=full_map.shape[1],
            map_w=full_map.shape[2],
            crop_offset=crop_offset,
            agent_orientation_deg=float(current_pose[2]),
        )

    def _prune_render_cache(self, keep_token: Optional[Any]) -> None:
        if keep_token is None:
            for bucket in self._render_cache.values():
                bucket.clear()
            return

        for name in ("obstacle_mask_raw", "obstacle_mask_logic", "obstacle_mask_display"):
            bucket = self._render_cache.get(name, {})
            stale_keys = [key for key in bucket.keys() if key != keep_token]
            for key in stale_keys:
                bucket.pop(key, None)

        for name in (
            "space_area_layer",
            "space_area_mask",
            "space_area_contours",
            "space_area_label_anchor",
            "space_area_color_assignments",
        ):
            bucket = self._render_cache.get(name, {})
            stale_keys = [key for key in bucket.keys() if not isinstance(key, tuple) or not key or key[0] != keep_token]
            for key in stale_keys:
                bucket.pop(key, None)

    def _build_raw_obstacle_mask(
        self,
        full_map: np.ndarray,
        cache_key: Optional[Any] = None,
    ) -> np.ndarray:
        cache_token = cache_key if cache_key is not None else self._active_render_cache_key
        if cache_token is not None:
            cached = self._render_cache["obstacle_mask_raw"].get(cache_token)
            if cached is not None:
                return cached.copy()

        obstacle_mask = build_rotated_obstacle_mask(
            full_map,
            threshold=0.5,
            open_kernel_size=0,
            close_kernel_size=0,
            axis_close_kernel_size=0,
            min_component_area=0,
        )
        if cache_token is not None:
            self._render_cache["obstacle_mask_raw"][cache_token] = obstacle_mask.copy()
        return obstacle_mask

    def _build_logic_obstacle_mask(
        self,
        full_map: np.ndarray,
        cache_key: Optional[Any] = None,
    ) -> np.ndarray:
        cache_token = cache_key if cache_key is not None else self._active_render_cache_key
        if cache_token is not None:
            cached = self._render_cache["obstacle_mask_logic"].get(cache_token)
            if cached is not None:
                return cached.copy()

        raw_obstacle_mask = self._build_raw_obstacle_mask(full_map, cache_key=cache_token)
        obstacle_mask = build_rotated_obstacle_mask(
            full_map,
            threshold=0.55,
            open_kernel_size=2,
            close_kernel_size=3,
            axis_close_kernel_size=5,
            min_component_area=6,
            hole_fill_area=12,
        )
        raw_pixels = int(np.count_nonzero(raw_obstacle_mask))
        cleaned_pixels = int(np.count_nonzero(obstacle_mask))
        if raw_pixels > 0 and (cleaned_pixels == 0 or cleaned_pixels < max(6, int(raw_pixels * 0.08))):
            obstacle_mask = raw_obstacle_mask
        if cache_token is not None:
            self._render_cache["obstacle_mask_logic"][cache_token] = obstacle_mask.copy()
        return obstacle_mask

    def _build_display_obstacle_mask(
        self,
        full_map: np.ndarray,
        cache_key: Optional[Any] = None,
    ) -> np.ndarray:
        cache_token = cache_key if cache_key is not None else self._active_render_cache_key
        if cache_token is not None:
            cached = self._render_cache["obstacle_mask_display"].get(cache_token)
            if cached is not None:
                return cached.copy()

        # Keep display-side obstacles identical to the navigation-side mask so
        # we do not maintain a separate "beautified" obstacle map anymore.
        obstacle_mask = self._build_logic_obstacle_mask(full_map, cache_key=cache_token)
        if cache_token is not None:
            self._render_cache["obstacle_mask_display"][cache_token] = obstacle_mask.copy()
        return obstacle_mask

    @staticmethod
    def _compute_adaptive_zoom_box(
        images: List[np.ndarray],
    ) -> Optional[Tuple[int, int, int, int]]:
        if not images:
            return None

        reference = images[0]
        height, width = reference.shape[:2]
        if height <= 0 or width <= 0:
            return None

        combined_mask = np.zeros((height, width), dtype=bool)
        for image in images:
            if image is None or image.shape[:2] != (height, width):
                continue
            combined_mask |= np.any(image != 255, axis=2)

        if not np.any(combined_mask):
            return None

        ys, xs = np.nonzero(combined_mask)
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())

        margin = max(12, int(round(min(height, width) * 0.05)))
        content_w = x_max - x_min + 1
        content_h = y_max - y_min + 1
        crop_side = max(content_w, content_h) + margin * 2
        max_side = min(height, width)
        if crop_side >= max_side - 4:
            return None

        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        half_side = crop_side / 2.0

        crop_x1 = int(np.floor(center_x - half_side))
        crop_y1 = int(np.floor(center_y - half_side))
        crop_x2 = crop_x1 + crop_side
        crop_y2 = crop_y1 + crop_side

        if crop_x1 < 0:
            crop_x2 -= crop_x1
            crop_x1 = 0
        if crop_y1 < 0:
            crop_y2 -= crop_y1
            crop_y1 = 0
        if crop_x2 > width:
            shift = crop_x2 - width
            crop_x1 = max(0, crop_x1 - shift)
            crop_x2 = width
        if crop_y2 > height:
            shift = crop_y2 - height
            crop_y1 = max(0, crop_y1 - shift)
            crop_y2 = height

        if crop_x2 - crop_x1 >= width - 2 or crop_y2 - crop_y1 >= height - 2:
            return None

        return crop_x1, crop_y1, crop_x2, crop_y2

    def _apply_adaptive_zoom(
        self,
        images: List[np.ndarray],
    ) -> List[np.ndarray]:
        if not self.enable_adaptive_zoom or not images:
            return images

        crop_box = self._compute_adaptive_zoom_box(images)
        if crop_box is None:
            return images

        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        output_images: List[np.ndarray] = []
        for image in images:
            if image is None:
                output_images.append(image)
                continue
            height, width = image.shape[:2]
            cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
            resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_NEAREST)
            output_images.append(resized)
        return output_images

    @staticmethod
    def _normalize_artifact_phase(phase: str) -> str:
        return build_subtask_name_from_token(phase, fallback=phase)

    def get_map_artifact_filename(self, step: int, phase: str) -> str:
        return build_step_artifact_filename(step, phase, suffix=".png")

    def get_map_artifact_dir(self, episode_id: int, map_kind: str) -> str:
        episode_dir = self._create_episode_directories(int(episode_id))
        if str(map_kind).strip().lower() == "global":
            dir_name = "global_map"
        elif str(map_kind).strip().lower() == "local":
            dir_name = "local_map"
        else:
            raise ValueError(f"Unsupported map artifact kind: {map_kind}")
        save_dir = os.path.join(episode_dir, dir_name)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def get_map_artifact_path(self, step: int, episode_id: int, phase: str, map_kind: str) -> str:
        return os.path.join(
            self.get_map_artifact_dir(episode_id, map_kind),
            self.get_map_artifact_filename(step, phase),
        )

    def get_model_input_map_artifact_dir(self, episode_id: int, map_kind: str) -> str:
        return self.get_map_artifact_dir(episode_id, map_kind)

    def get_model_input_map_artifact_filename(
        self,
        step: int,
        phase: str,
        map_kind: str,
    ) -> str:
        kind = str(map_kind).strip().lower()
        if kind not in {"global", "local"}:
            raise ValueError(f"Unsupported model-input map kind: {map_kind}")
        return f"{build_subtask_name_from_token(phase, fallback=phase)}.png"

    def get_model_input_map_artifact_path(
        self,
        step: int,
        episode_id: int,
        phase: str,
        map_kind: str,
    ) -> str:
        return os.path.join(
            self.get_model_input_map_artifact_dir(episode_id, map_kind),
            self.get_model_input_map_artifact_filename(step, phase, map_kind),
        )

    def get_render_policy(
        self,
        phase: str,
        render_policy: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, bool]:
        phase_text = str(phase or "")
        is_action = phase_text.startswith("action")
        is_thinking = phase_text == "initial" or phase_text.startswith("verify")
        debug_mode = bool(self.debug_save_renderings)

        policy = {
            "save_rgb": False,
            "render_global_map": debug_mode or is_thinking,
            "save_global_map": self.save_step_map_artifacts,
            "render_local_map": debug_mode or self.save_step_map_artifacts,
            "save_local_map": self.save_step_map_artifacts,
            "render_detection": debug_mode or is_action,
            "save_detection": False,
        }
        if render_policy:
            policy.update({key: bool(value) for key, value in render_policy.items()})
        return policy

    @staticmethod
    def _make_render_cache_key(
        step: int,
        full_map: Optional[np.ndarray],
        space_area_layer: Optional[np.ndarray],
    ) -> Tuple[int, int, int]:
        return (
            int(step),
            int(id(full_map)) if full_map is not None else 0,
            int(id(space_area_layer)) if space_area_layer is not None else 0,
        )

    @staticmethod
    def _bbox_iou(
        bbox_a: Tuple[int, int, int, int],
        bbox_b: Tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = float(inter_w * inter_h)
        if inter_area <= 0.0:
            return 0.0
        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        union_area = area_a + area_b - inter_area
        if union_area <= 1e-6:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _angle_diff_deg(angle_a: float, angle_b: float) -> float:
        diff = float(angle_a) - float(angle_b)
        while diff > 180.0:
            diff -= 360.0
        while diff < -180.0:
            diff += 360.0
        return abs(diff)

    def _is_duplicate_detection_candidate(
        self,
        candidate: Dict[str, Any],
        kept_candidate: Dict[str, Any],
    ) -> bool:
        if candidate.get("name") != kept_candidate.get("name"):
            return False

        bbox = candidate["bbox"]
        kept_bbox = kept_candidate["bbox"]
        iou = self._bbox_iou(bbox, kept_bbox)
        if iou >= float(landmark_duplicate_iou_strict):
            return True

        rel_xy = candidate.get("det_rel_xy")
        kept_rel_xy = kept_candidate.get("det_rel_xy")
        if rel_xy is None or kept_rel_xy is None:
            return False

        rel_dist = float(np.hypot(rel_xy[0] - kept_rel_xy[0], rel_xy[1] - kept_rel_xy[1]))
        angle_a = float(np.degrees(np.arctan2(rel_xy[1], rel_xy[0]))) if np.hypot(rel_xy[0], rel_xy[1]) > 1e-6 else 0.0
        angle_b = float(np.degrees(np.arctan2(kept_rel_xy[1], kept_rel_xy[0]))) if np.hypot(kept_rel_xy[0], kept_rel_xy[1]) > 1e-6 else 0.0
        angle_diff = self._angle_diff_deg(angle_a, angle_b)

        return (
            rel_dist <= float(landmark_duplicate_rel_dist_m) and
            angle_diff <= float(landmark_duplicate_angle_diff_deg) and
            iou >= float(landmark_duplicate_iou_loose)
        )

    @staticmethod
    def _rel_xy_to_world_xy(
        rel_xy: Optional[Tuple[float, float]],
        current_pose: Optional[Tuple[float, float, float]],
    ) -> Optional[Tuple[float, float]]:
        if rel_xy is None or current_pose is None:
            return None
        forward_m, right_m = float(rel_xy[0]), float(rel_xy[1])
        curr_x_m, curr_y_m, curr_ori_deg = (
            float(current_pose[0]),
            float(current_pose[1]),
            float(current_pose[2]),
        )
        theta = np.deg2rad(curr_ori_deg)
        dx = forward_m * np.cos(theta) + right_m * np.sin(theta)
        dy = forward_m * np.sin(theta) - right_m * np.cos(theta)
        return curr_x_m + dx, curr_y_m + dy

    def _create_episode_directories(self, episode_id: int):
        """为特定episode创建保存目录"""
        episode_dir = get_episode_detail_dir(self.results_dir, episode_id)
        os.makedirs(episode_dir, exist_ok=True)
        return episode_dir

    def calculate_obstacle_distances_from_rotated_map(
        self,
        obstacle_mask_rotated: np.ndarray,
        center_x: int = 240,
        center_y: int = 240
    ) -> Dict[str, str]:
        """
        在旋转后的obstacle map上计算障碍物距离
        
        ⚠️ 关键优势：
        - 地图已经旋转，箭头朝上（-90°），agent在(240, 240)
        - 直接在像素坐标系中测距，无需复杂的Habitat角度转换
        - 上方 = FRONT, 左上30° = LEFT_30, 右上30° = RIGHT_30
        
        Args:
            obstacle_mask_rotated: [480, 480] 旋转后的障碍物掩码（bool或0/1）
            center_x: Agent中心X坐标（默认240）
            center_y: Agent中心Y坐标（默认240）
            
        Returns:
            距离字典 {
                'front': "X.XXm" | ">2.0m open" | "X.XXm WARNING",
                'left_30': ...,
                'right_30': ...
            }
        """
        return scan_obstacle_distances_from_rotated_map(
            obstacle_mask_rotated,
            center_x=center_x,
            center_y=center_y,
        )

    def calculate_obstacle_distances_from_depth(
        self,
        depth_meters: np.ndarray,
        hfov: float = 79.0,
        fallback_distances: Optional[Dict[str, str]] = None,
        angle_band_deg: float = 5.0,
        sensor_min_depth_m: float = 0.5,
    ) -> Dict[str, str]:
        """Estimate front-view obstacle distances from the current depth frame with fallback."""
        return scan_obstacle_distances_from_depth(
            depth_meters,
            hfov_deg=hfov,
            directions=ACTION_VIEW_DIRECTIONS,
            angle_band_deg=angle_band_deg,
            sensor_min_depth_m=sensor_min_depth_m,
            fallback_distances=fallback_distances,
        )

    def calculate_obstacle_distances_from_full_map(
        self,
        full_map: Optional[np.ndarray],
        center_x: int = 240,
        center_y: int = 240,
    ) -> Dict[str, str]:
        """Fallback obstacle distances from the logic-side obstacle mask."""
        if full_map is None:
            return {}
        obstacle_mask_rotated = self._build_logic_obstacle_mask(full_map)
        return self.calculate_obstacle_distances_from_rotated_map(
            obstacle_mask_rotated,
            center_x=center_x,
            center_y=center_y,
        )

    def calculate_obstacle_distances_12_directions(
        self,
        obstacle_mask_rotated: np.ndarray,
        center_x: int = 240,
        center_y: int = 240
    ) -> Dict[str, str]:
        """
        在旋转后的obstacle map上计算12个方向的障碍物距离（用于Thinking模式环视）
        
        覆盖完整360°：每30°一个方向，对应12张IMAGE
        
        Args:
            obstacle_mask_rotated: [480, 480] 旋转后的障碍物掩码
            center_x: Agent中心X坐标（默认240）
            center_y: Agent中心Y坐标（默认240）
            
        Returns:
            距离字典 {
                'angle_0': "X.XXm",    # IMAGE 1: Front (0°)
                'angle_30': "X.XXm",   # IMAGE 2: Left (30°)
                'angle_60': "X.XXm",   # IMAGE 3: Left (60°)
                ...
                'angle_330': "X.XXm"   # IMAGE 12: Right (30°)
            }
        """
        return scan_obstacle_distances_12_directions(
            obstacle_mask_rotated,
            center_x=center_x,
            center_y=center_y,
        )

    def calculate_obstacle_distances_12_directions_from_full_map(
        self,
        full_map: Optional[np.ndarray],
        center_x: int = 240,
        center_y: int = 240,
    ) -> Dict[str, str]:
        """Fallback 12-view obstacle distances from the logic-side obstacle mask."""
        if full_map is None:
            return {}
        obstacle_mask_rotated = self._build_logic_obstacle_mask(full_map)
        return self.calculate_obstacle_distances_12_directions(
            obstacle_mask_rotated,
            center_x=center_x,
            center_y=center_y,
        )

    @staticmethod
    def get_distance_summary(distances: Dict[str, str]) -> str:
        """生成距离摘要字符串（供日志打印）"""
        return (f"FRONT={distances.get('front', 'Unknown')}, "
                f"L30={distances.get('left_30', 'Unknown')}, "
                f"R30={distances.get('right_30', 'Unknown')}")

    def save_step_visualization(self,
                               step: int,
                               episode_id: int,
                               rgb: np.ndarray,
                               full_map: np.ndarray,
                               trajectory_points: List[Tuple[int, int]],
                               detected_classes: List[str],
                               current_pose: Tuple[float, float, float],
                               floor: Optional[np.ndarray] = None,
                               hfov: float = 90.0,
                               detections=None,  # sv.Detections对象（新）
                               labels: Optional[List[str]] = None,
                               landmark_classes: Optional[List[str]] = None,
                               mapping_classes: Optional[List[str]] = None,
                               landmark_config: Optional[Dict] = None,
                               waypoint_positions: Optional[List[Tuple[int, int]]] = None,
                               waypoint_ids: Optional[List[int]] = None,
                               space_area_layer: Optional[np.ndarray] = None,
                               space_area_records: Optional[List[Dict[str, Any]]] = None,
                               masks: Optional[np.ndarray] = None,
                               phase: str = "action",
                               global_trajectory_points: Optional[List[Tuple[int, int]]] = None,
                               controller=None,
                               crop_offset: Optional[Tuple[int, int]] = None,
                               render_policy: Optional[Dict[str, bool]] = None) -> Tuple[Dict[str, str], List, Optional[float]]:
        """
        一键保存当前步骤的所有可视化（支持新detection渲染 + 平滑轨迹线 + waypoint标记）
            trajectory_points: [(x, y), ...] 当前子任务轨迹（用于local map）
            global_trajectory_points: [(x, y), ...] 完整导航历史轨迹（用于global map，可选）
                - 如果提供，global map显示此轨迹
                - 如果未提供，global map回退使用trajectory_points（向后兼容）
            floor: [H, W] floor地图（通过形态学方法计算，像ZS_Evaluator）
            detections: supervision Detections对象（优先使用）
            masks: 检测掩码（向后兼容，已废弃）
            mapping_classes: Mapping类别列表
            landmark_classes: Landmark类别列表
            waypoint_positions: [(map_x, map_y), ...] waypoint位置列表（可选，从mapper.get_waypoints()获取）
            waypoint_ids: [1, 2, 3, ...] waypoint ID列表（可选，从mapper.get_waypoints()获取）
            phase: 阶段标识 ("initial", "action1a", "verify1a" 等)
            controller: VLMNavigationController实例（用于绘制距离线）
        
        Returns:
            (paths, landmarks, obstacle_distances, last_waypoint_angle)
            - paths: 保存路径字典 {'rgb', 'global_map', 'local_map', 'detection'}
            - landmarks: Landmark列表
            - obstacle_distances: {'front': "X.XXm", 'left_30': ..., 'right_30': ...}
            - last_waypoint_angle: 最后一个waypoint相对于正前方的角度（弧度），None表示无waypoint
            
        注意:
        1. floor通过形态学方法计算（像ZS_Evaluator._process_map）
        2. waypoint数据建议直接从mapper.get_waypoints()传入，无需手动管理
        """
        paths = {}
        policy = self.get_render_policy(phase, render_policy)
        render_cache_key = self._make_render_cache_key(step, full_map, space_area_layer)
        previous_cache_key = self._active_render_cache_key
        if render_cache_key != previous_cache_key:
            self._prune_render_cache(render_cache_key)
        self._active_render_cache_key = render_cache_key
        depth_for_instances = getattr(controller, 'latest_depth_meters', None) if controller is not None else None
        landmark_memory = controller.landmark_memory if controller is not None else None
        existing_landmark_instances = (
            landmark_memory.get_world_instances()
            if landmark_memory is not None else []
        )
        projected_landmark_instances = self._project_landmark_instances_from_detections(
            detections=detections,
            labels=labels,
            landmark_classes=landmark_classes,
            depth_meters=depth_for_instances,
            current_pose=current_pose,
            hfov=hfov,
            topk=None,
        )

        landmark_instances_world = self._merge_landmark_instances_world(
            existing_instances=existing_landmark_instances,
            new_instances=projected_landmark_instances,
            current_pose=current_pose,
        )

        if landmark_memory is not None:
            landmark_memory.set_world_instances(landmark_instances_world)

        locked_action_landmark_entries = (
            landmark_memory.get_latest_prompt_entries()
            if landmark_memory is not None else []
        )
        has_fresh_landmark_detection = (
            detections is not None and
            labels is not None and
            bool(landmark_classes)
        )
        if locked_action_landmark_entries and not has_fresh_landmark_detection:
            action_landmark_context = self._build_action_landmark_context_from_locked_entries(
                landmark_instances_world,
                locked_action_landmark_entries,
                topk=local_map_landmark_topk,
            )
        else:
            action_landmark_context = self._build_action_landmark_context(
                landmark_instances_world,
                topk=local_map_landmark_topk,
            )
        selected_action_landmark_instances = list(action_landmark_context.get("selected_instances", []) or [])
        try:
            paths['rgb'] = self.save_rgb(step, episode_id, rgb, phase, controller) if policy.get('save_rgb', False) else None

            global_traj_to_use = global_trajectory_points if global_trajectory_points is not None else trajectory_points
            global_map_with_trajectory = None
            landmarks = []
            last_waypoint_angle = None
            if policy.get('render_global_map', False):
                _, global_map_with_trajectory, landmarks, _global_map_clean, last_waypoint_angle = self.render_global_map(
                    full_map, global_traj_to_use, detected_classes, floor,
                    current_pose, landmark_classes, landmark_instances_world, landmark_config,
                    waypoint_positions, waypoint_ids, space_area_layer, space_area_records, crop_offset,
                    mapping_classes=mapping_classes
                )
                if global_map_with_trajectory is not None:
                    paths['global_map_input'] = {
                        "image_array": global_map_with_trajectory,
                        "color_space": "bgr",
                        "artifact_name": "global_map.jpg",
                        "name": "global_map",
                        "original_artifact_path": self.get_model_input_map_artifact_path(
                            step=step,
                            episode_id=episode_id,
                            phase=phase,
                            map_kind="global",
                        ),
                    }
            elif landmark_instances_world:
                landmarks = self._build_landmarks_from_instances(
                    landmark_instances_world,
                    full_map,
                    current_pose,
                    crop_offset,
                )
            elif landmark_classes and landmark_config:
                landmarks = self._extract_landmarks(
                    full_map,
                    detected_classes,
                    landmark_classes,
                    landmark_config['min_total_pixels'],
                    landmark_config['min_area_threshold'],
                    mapping_classes=mapping_classes,
                )

            paths['global_map'] = (
                self.save_global_map(step, episode_id, global_map_with_trajectory, phase)
                if policy.get('save_global_map', False) and global_map_with_trajectory is not None else None
            )

            local_map = None
            if policy.get('render_local_map', False):
                local_map = self.render_local_map(
                    full_map, trajectory_points, detected_classes, current_pose,
                    floor, landmark_classes, selected_action_landmark_instances, landmark_config, hfov,
                    waypoint_positions, waypoint_ids, space_area_layer, space_area_records, crop_offset,
                    mapping_classes=mapping_classes
                )
            local_map_debug_lines = self._build_local_map_landmark_debug_lines(
                selected_action_landmark_instances,
                topk=local_map_landmark_topk,
            )
            if not local_map_debug_lines and controller is not None:
                local_map_debug_lines = list(
                    getattr(controller, "latest_action_local_map_debug_lines", []) or []
                )
            paths['local_map'] = (
                self.save_local_map(
                    step,
                    episode_id,
                    local_map,
                    phase,
                    debug_lines=local_map_debug_lines,
                )
                if policy.get('save_local_map', False) and local_map is not None else None
            )

            detected_landmarks_step = []
            landmark_dist_map = {}
            landmark_dist_map_multi = {}
            for _, _, cls_name, dist_m, angle_deg in landmarks:
                landmark_dist_map_multi.setdefault(cls_name, []).append((dist_m, angle_deg))
                if cls_name not in landmark_dist_map or dist_m < landmark_dist_map[cls_name][0]:
                    landmark_dist_map[cls_name] = (dist_m, angle_deg)

            if landmark_memory is not None:
                landmark_memory.set_latest_distance_maps(
                    dist_map=landmark_dist_map if landmark_dist_map else {},
                    dist_map_multi=landmark_dist_map_multi if landmark_dist_map_multi else {},
                )

            should_render_detection = (
                policy.get('render_detection', False) and (
                    (detections is not None and labels is not None) or
                    bool(landmark_dist_map) or
                    bool(landmark_dist_map_multi) or
                    bool(landmark_instances_world)
                )
            )

            if should_render_detection:
                try:
                    obstacle_distances = self.calculate_obstacle_distances_from_depth(
                        getattr(controller, 'latest_depth_meters', None) if controller is not None else None,
                        hfov=hfov,
                        fallback_distances=None,
                    )
                except Exception:
                    obstacle_distances = {
                        'front': 'Unknown',
                        'left_30': 'Unknown',
                        'right_30': 'Unknown',
                    }

                rgb_for_det = rgb.copy()
                detection_vis, detected_landmarks_step, _visible, landmark_strip = self.render_detection_bbox(
                    rgb_for_det, detections, labels or [],
                    landmark_classes, mapping_classes,
                    depth_meters=getattr(controller, 'latest_depth_meters', None) if controller is not None else None,
                    hfov=hfov,
                    landmark_dist_map=landmark_dist_map if landmark_dist_map else None,
                    landmark_dist_map_multi=landmark_dist_map_multi if landmark_dist_map_multi else None,
                    append_bottom_strip=False,
                    controller=controller,
                    selected_landmark_instances=selected_action_landmark_instances,
                    action_landmark_context=action_landmark_context,
                    action_distance_overlay=obstacle_distances,
                )

                detection_vis = self.draw_distance_on_action_view(detection_vis, obstacle_distances)
                model_detection_vis = resize_image_to_width(
                    detection_vis,
                    ACTION_VIEW_MODEL_CONTENT_WIDTH,
                )
                model_landmark_strip = (
                    resize_strip_to_width(landmark_strip, model_detection_vis.shape[1])
                    if landmark_strip is not None else None
                )
                if controller is not None:
                    controller.latest_obstacle_distances = obstacle_distances
                    if model_landmark_strip is not None:
                        controller.latest_action_detection_vis = np.vstack(
                            [model_detection_vis, model_landmark_strip]
                        )
                    else:
                        controller.latest_action_detection_vis = model_detection_vis.copy()

                if landmark_strip is not None:
                    detection_vis = np.vstack([detection_vis, landmark_strip])
                paths['detection'] = (
                    self.save_detection(step, episode_id, detection_vis, phase)
                    if policy.get('save_detection', False) else None
                )
            else:
                paths['detection'] = None
                if controller is not None:
                    controller.latest_action_detection_vis = None
                    if landmark_memory is not None:
                        landmark_memory.clear_latest_prompt_view()

            n_cfg = len(landmark_classes) if landmark_classes else 0
            n_det_inst = len(detected_landmarks_step)
            n_det_cls = len(set([n for n, _ in detected_landmarks_step])) if detected_landmarks_step else 0
            n_map_inst = len(landmarks)
            n_map_cls = len(set([x[2] for x in landmarks])) if landmarks else 0
            if controller is not None:
                controller.latest_landmark_stats = {
                    'step': step,
                    'configured_classes': n_cfg,
                    'detected_instances': n_det_inst,
                    'detected_classes': n_det_cls,
                    'mapped_instances': n_map_inst,
                    'mapped_classes': n_map_cls,
                }

            if masks is not None and self.debug_save_renderings:
                paths['masks'] = self.save_semantic_masks(step, episode_id, masks, phase)

            return paths, detected_landmarks_step, last_waypoint_angle
        finally:
            self._active_render_cache_key = previous_cache_key

    def _get_channel_mask(self, full_map: np.ndarray, channel_idx: int,
                          threshold: float = 0.5) -> Optional[np.ndarray]:
        """从 full_map 提取指定通道的二值 mask（统一基础函数）。

        所有层均来自同一投影流程（RGB-D → splat_feat_nd → forward()）：
          full_map[0]              : obstacle
          full_map[1]              : explored
          full_map[2]              : agent / waypoint
          full_map[3 .. 3+M-1]     : mapping_classes[0..M-1]  （含 floor）
          full_map[3+M .. 3+M+N-1] : landmark_classes[0..N-1]
        """
        if channel_idx < 0 or channel_idx >= full_map.shape[0]:
            return None
        return full_map[channel_idx, ...] > threshold

    def _get_channel_centroids(self,
                               full_map: np.ndarray,
                               channel_idx: int,
                               min_area: int,
                               merge_dist: float,
                               threshold: float = 0.5,
                               apply_closing: bool = True) -> List[Tuple[int, int, int]]:
        """从 full_map 指定通道提取连通域质心（landmark 的额外聚类步骤）。

        obstacle / floor 只需 _get_channel_mask；
        landmark 在此基础上进一步做形态学闭运算 + 连通域聚类。

        Returns: [(cx, cy, area), ...]
        """
        mask = self._get_channel_mask(full_map, channel_idx, threshold)

        if mask is None or not mask.any():
            return []

        if apply_closing:
            # 形态学闭运算（填补间隙，合并相近检测）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            proc_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        else:
            # 对 landmark 直接使用原始投影 mask，避免把不同实例粘成一个紫色点。
            proc_mask = mask.astype(np.uint8)

        # 连通域分析
        _, _, stats, centroids = cv2.connectedComponentsWithStats(proc_mask, connectivity=8)

        regions: dict = {}  # {(cx, cy): area}
        for i in range(1, len(stats)):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            cx, cy = int(centroids[i][0]), int(centroids[i][1])

            # 空间合并：距离 < merge_dist 的连通域归并到面积较大者
            if merge_dist <= 0:
                regions[(cx, cy)] = area
                continue
            merged = False
            for pos in list(regions.keys()):
                if np.hypot(cx - pos[0], cy - pos[1]) < merge_dist:
                    if area > regions[pos]:
                        del regions[pos]
                        regions[(cx, cy)] = area
                    merged = True
                    break
            if not merged:
                regions[(cx, cy)] = area

        return [(cx, cy, a) for (cx, cy), a in regions.items()]

    def _extract_landmarks(self,
                          full_map: np.ndarray,
                          detected_classes: List[str],
                          landmark_classes: List[str],
                          min_total_pixels: int,
                          min_area_threshold: int,
                          mapping_classes: Optional[List[str]] = None) -> List[Tuple]:
        """提取 landmark 质心并计算到 agent 的距离和偏角。

        所有语义通道均来自同一个 full_map（RGB-D → splat_feat_nd → forward()）：
          [0] obstacle  [1] explored  [2] agent
          [3 .. 3+M-1]  mapping_classes（含 floor）
          [3+M .. ]     landmark_classes

        obstacle / floor 只需 _get_channel_mask；
        landmark 在此基础上额外做连通域聚类（_get_channel_centroids）。

        Args:
            min_total_pixels: 已弃用，保留兼容
        Returns:
            [(cx, cy, class_name, dist_m, rel_angle_deg), ...]
        """
        if not landmark_classes:
            return []

        from navigation_system.config.core.constants import landmark_merge_distance
        import math as _math

        MAP_CH = 3
        n_mapping = len(mapping_classes) if mapping_classes is not None else 0
        h_map, w_map = full_map.shape[1], full_map.shape[2]
        agent_cy, agent_cx = h_map // 2, w_map // 2
        resolution_m = self.resolution / 100.0  # cm/pixel → m/pixel

        landmarks = []
        for lm_idx, cls_name in enumerate(landmark_classes):
            if mapping_classes is not None:
                ch_idx = MAP_CH + n_mapping + lm_idx
            else:
                # 向后兼容：没有 mapping_classes 时的旧公式
                cls_idx = detected_classes.index(cls_name)
                ch_idx = 4 + cls_idx

            # _get_channel_centroids 内部调用 _get_channel_mask，与 obstacle 同一基础
            for cx, cy, _area in self._get_channel_centroids(
                    full_map,
                    ch_idx,
                    min_area_threshold,
                    landmark_merge_distance,
                    threshold=0.0,
                    apply_closing=False):
                d_fwd   = cy - agent_cy   # 正 = 前方
                d_right = cx - agent_cx   # 正 = 右侧
                dist_m = _math.hypot(d_fwd, d_right) * resolution_m
                rel_angle_deg = _math.degrees(_math.atan2(d_right, d_fwd)) if dist_m > 0 else 0.0
                landmarks.append((cx, cy, cls_name, dist_m, rel_angle_deg))

        return landmarks

    def save_semantic_masks(self, step: int, episode_id: int, masks: np.ndarray, phase: str = "action") -> str:
        """
        保存semantic masks到numpy文件
        
        Args:
            step: 当前步数
            episode_id: episode ID
            masks: semantic masks [num_classes, H, W]
            phase: 阶段标识
            
        Returns:
            保存路径
        """
        episode_dir = self._create_episode_directories(episode_id)
        masks_dir = os.path.join(episode_dir, 'semantic_masks')
        os.makedirs(masks_dir, exist_ok=True)
        
        save_path = os.path.join(masks_dir, f'step_{step:04d}_{phase}.npy')
        np.save(save_path, masks)
        
        return save_path

from navigation_system.render.views import detection_renderer as _detection_renderer
from navigation_system.space.landmarks import landmark_selection as _landmark_selection
from . import map_renderer as _map_renderer
from . import space_area_overlay as _space_area_overlay

# Space-area overlay / label-layout helpers
MapVisualizer._normalize_space_area_type = _space_area_overlay._normalize_space_area_type
MapVisualizer._space_area_color_preference_order = _space_area_overlay._space_area_color_preference_order
MapVisualizer._build_space_area_adjacency = _space_area_overlay._build_space_area_adjacency
MapVisualizer._resolve_space_area_colors = _space_area_overlay._resolve_space_area_colors
MapVisualizer._prepare_space_area_display_layer = _space_area_overlay._prepare_space_area_display_layer
MapVisualizer._refine_space_area_mask = _space_area_overlay._refine_space_area_mask
MapVisualizer._draw_space_areas_in_place = _space_area_overlay._draw_space_areas_in_place
MapVisualizer._overlay_space_areas = _space_area_overlay._overlay_space_areas
MapVisualizer._get_space_area_render_assets = _space_area_overlay._get_space_area_render_assets
MapVisualizer._compute_space_area_label_anchor = _space_area_overlay._compute_space_area_label_anchor
MapVisualizer._compact_space_area_overlay_label = _space_area_overlay._compact_space_area_overlay_label
MapVisualizer._label_box_intersection_area = _space_area_overlay._label_box_intersection_area
MapVisualizer._build_label_box_from_center = _space_area_overlay._build_label_box_from_center
MapVisualizer._label_center_from_box = _space_area_overlay._label_center_from_box
MapVisualizer._get_space_area_label_style = _space_area_overlay._get_space_area_label_style
MapVisualizer._resolve_space_area_label_layout = _space_area_overlay._resolve_space_area_label_layout
MapVisualizer._draw_space_area_label = _space_area_overlay._draw_space_area_label

# Landmark selection / matching helpers
MapVisualizer._candidate_distance_m = staticmethod(_landmark_selection._candidate_distance_m)
MapVisualizer._candidate_angle_deg = _landmark_selection._candidate_angle_deg
MapVisualizer._should_merge_detection_candidates = _landmark_selection._should_merge_detection_candidates
MapVisualizer._merge_detection_candidate_entries = _landmark_selection._merge_detection_candidate_entries
MapVisualizer._dedupe_detection_candidates = _landmark_selection._dedupe_detection_candidates
MapVisualizer._landmark_instance_uid = staticmethod(_landmark_selection._landmark_instance_uid)
MapVisualizer._landmark_instance_rel_xy = staticmethod(_landmark_selection._landmark_instance_rel_xy)
MapVisualizer._sort_landmark_instances_for_action = staticmethod(_landmark_selection._sort_landmark_instances_for_action)
MapVisualizer._select_action_landmark_instances = _landmark_selection._select_action_landmark_instances
MapVisualizer._build_landmark_display_index_lookup = _landmark_selection._build_landmark_display_index_lookup
MapVisualizer._build_landmark_class_totals = staticmethod(_landmark_selection._build_landmark_class_totals)
MapVisualizer._build_action_landmark_context = _landmark_selection._build_action_landmark_context
MapVisualizer._build_action_landmark_context_from_locked_entries = _landmark_selection._build_action_landmark_context_from_locked_entries
MapVisualizer._match_candidate_to_world_instance = _landmark_selection._match_candidate_to_world_instance
MapVisualizer._estimate_mask_rel_xy = _landmark_selection._estimate_mask_rel_xy
MapVisualizer._analyze_mask_depth_profile = _landmark_selection._analyze_mask_depth_profile
MapVisualizer._project_landmark_instances_from_detections = _landmark_selection._project_landmark_instances_from_detections
MapVisualizer._merge_landmark_instances_world = _landmark_selection._merge_landmark_instances_world
MapVisualizer._world_instance_to_rotated_landmark = _landmark_selection._world_instance_to_rotated_landmark
MapVisualizer._build_landmarks_from_instances = _landmark_selection._build_landmarks_from_instances
MapVisualizer._build_local_landmarks_from_instances = _landmark_selection._build_local_landmarks_from_instances

# Map rendering helpers
MapVisualizer.render_global_map = _map_renderer.render_global_map
MapVisualizer.render_local_map = _map_renderer.render_local_map
MapVisualizer.add_orientation_labels = _map_renderer.add_orientation_labels
MapVisualizer.save_global_map = _map_renderer.save_global_map
MapVisualizer.save_local_map = _map_renderer.save_local_map

# Detection / action-view rendering helpers
MapVisualizer.render_detection_bbox = _detection_renderer.render_detection_bbox
MapVisualizer.save_rgb = _detection_renderer.save_rgb
MapVisualizer.draw_floor_from_saved_mask = _detection_renderer.draw_floor_from_saved_mask
MapVisualizer.draw_distance_on_view = _detection_renderer.draw_distance_on_view
MapVisualizer.draw_distance_on_action_view = _detection_renderer.draw_distance_on_action_view
MapVisualizer.prepare_action_image_with_enhancements = _detection_renderer.prepare_action_image_with_enhancements
MapVisualizer.save_detection = _detection_renderer.save_detection
