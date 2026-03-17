"""
地图可视化工具 - MapVisualizer
================================
职责：
1. 地图渲染（全局地图、局部地图）
2. 检测结果可视化
3. 轨迹绘制
4. 文件保存

设计原则：
- 单一职责：只负责可视化和保存，不涉及建图逻辑
- 解耦：独立于Controller和Mapper
- 可复用：支持多种可视化场景
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any

from vlnce_baselines.common.spatial_formatter import format_relative_direction
from vlnce_baselines.visualization import rendering as vu
from vlnce_baselines.visualization.landmark_overlay import (
    LandmarkDrawItem,
    build_landmark_strip_lines,
    draw_action_partition_lines,
    draw_landmark_boxes,
    draw_landmark_labels,
    render_landmark_strip,
)
from vlnce_baselines.visualization.map_projection import RotatedMapProjector
from vlnce_baselines.visualization.obstacle_analysis import (
    ACTION_VIEW_DIRECTIONS,
    build_rotated_obstacle_mask,
    calculate_obstacle_distances_from_depth as scan_obstacle_distances_from_depth,
    calculate_obstacle_distances_12_directions as scan_obstacle_distances_12_directions,
    calculate_obstacle_distances_from_rotated_map as scan_obstacle_distances_from_rotated_map,
)
from vlnce_baselines.config_system.constants import (
    color_palette, 
    detection_visible_topk,
    detection_colors,
    detection_thickness,
    landmark_marker_color,
    landmark_marker_border,
    landmark_marker_radius,
    landmark_instance_topk,
    landmark_instance_merge_radius_m,
)


class MapVisualizer:
    """地图可视化器 - 统一管理所有可视化和保存逻辑"""
    
    def __init__(self, 
                 results_dir: str,
                 resolution: int = 5,
                 map_shape: Tuple[int, int] = (480, 480),
                 enable_global_map_crop: bool = False,
                 enable_adaptive_zoom: bool = False):
        """
        Args:
            results_dir: 保存根目录（如：data/manual_navigation）
            resolution: 地图分辨率（cm/pixel）
            map_shape: 地图尺寸
            enable_global_map_crop: 是否裁剪global map到440×440（默认False，保持480×480）
            enable_adaptive_zoom: 是否启用自适应缩放（根据轨迹范围自动调整显示区域）
        """
        self.results_dir = results_dir
        self.resolution = resolution
        self.map_shape = map_shape
        self.enable_global_map_crop = enable_global_map_crop
        self.enable_adaptive_zoom = enable_adaptive_zoom
        self.color_palette = [int(x * 255.) for x in color_palette]
        
        # 注意：不在初始化时创建目录，而是在保存时根据episode_id动态创建

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

    @staticmethod
    def _room_area_color(area_id: int, room_type: str) -> Tuple[int, int, int]:
        room_text = str(room_type)
        seed = int(area_id) * 131
        for index, ch in enumerate(room_text):
            seed += (index + 17) * ord(ch)
        hue = int(seed % 180)
        hsv = np.uint8([[[hue, 110, 235]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def _prepare_room_area_display_layer(
        self,
        room_area_layer: Optional[np.ndarray],
        output_size: int = 480,
    ) -> Optional[np.ndarray]:
        if room_area_layer is None or room_area_layer.size == 0:
            return None
        layer = np.flipud(np.asarray(room_area_layer, dtype=np.int32))
        return cv2.resize(layer, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

    def _overlay_room_areas(
        self,
        image: np.ndarray,
        room_area_layer: Optional[np.ndarray],
        room_area_records: Optional[List[Dict[str, Any]]],
        alpha: float = 0.32,
    ) -> np.ndarray:
        display_layer = self._prepare_room_area_display_layer(room_area_layer, output_size=image.shape[1])
        if display_layer is None or not room_area_records:
            return image

        overlay = image.copy()
        output = image.copy()
        for record in room_area_records:
            area_id = int(record.get("id", 0) or 0)
            if area_id <= 0:
                continue
            mask = display_layer == area_id
            if not np.any(mask):
                continue
            color = self._room_area_color(area_id, str(record.get("room_type", "")))
            overlay[mask] = color

            mask_uint8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(output, contours, -1, color, 2)

        blended = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)
        return blended

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
        if iou >= 0.65:
            return True

        rel_xy = candidate.get("det_rel_xy")
        kept_rel_xy = kept_candidate.get("det_rel_xy")
        if rel_xy is None or kept_rel_xy is None:
            return False

        rel_dist = float(np.hypot(rel_xy[0] - kept_rel_xy[0], rel_xy[1] - kept_rel_xy[1]))
        angle_a = float(np.degrees(np.arctan2(rel_xy[1], rel_xy[0]))) if np.hypot(rel_xy[0], rel_xy[1]) > 1e-6 else 0.0
        angle_b = float(np.degrees(np.arctan2(kept_rel_xy[1], kept_rel_xy[0]))) if np.hypot(kept_rel_xy[0], kept_rel_xy[1]) > 1e-6 else 0.0
        angle_diff = self._angle_diff_deg(angle_a, angle_b)

        return rel_dist <= 0.35 and angle_diff <= 12.0 and iou >= 0.25
    
    def _create_episode_directories(self, episode_id: int):
        """为特定episode创建保存目录"""
        episode_dir = os.path.join(self.results_dir, f'episode_{episode_id}')
        dirs = ['rgb', 'global_map', 'local_map', 'detection']
        for dir_name in dirs:
            os.makedirs(os.path.join(episode_dir, dir_name), exist_ok=True)
        return episode_dir
    
    # ========== 距离计算方法 ==========
    
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
                'front': "X.XXm" | ">2.0m open" | "<0.5m WARNING",
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
    ) -> Dict[str, str]:
        """Estimate front-view obstacle distances from the current depth frame with fallback."""
        return scan_obstacle_distances_from_depth(
            depth_meters,
            hfov_deg=hfov,
            directions=ACTION_VIEW_DIRECTIONS,
            angle_band_deg=angle_band_deg,
            fallback_distances=fallback_distances,
        )

    def calculate_obstacle_distances_from_full_map(
        self,
        full_map: Optional[np.ndarray],
        center_x: int = 240,
        center_y: int = 240,
    ) -> Dict[str, str]:
        """Fallback obstacle distances from the rotated obstacle map."""
        if full_map is None:
            return {}
        obstacle_mask_rotated = build_rotated_obstacle_mask(full_map)
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
                'angle_30': "X.XXm",   # IMAGE 2: Right (30°)
                'angle_60': "X.XXm",   # IMAGE 3: Right (60°)
                ...
                'angle_330': "X.XXm"   # IMAGE 12: Left (330°)
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
        """Fallback 12-view obstacle distances from the rotated obstacle map."""
        if full_map is None:
            return {}
        obstacle_mask_rotated = build_rotated_obstacle_mask(full_map)
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
    
    # ========== 渲染方法 ==========
    
    def render_global_map(self,
                         full_map: np.ndarray,
                         trajectory_points: List[Tuple[int, int]],
                         detected_classes: List[str],
                         floor: Optional[np.ndarray] = None,
                         current_pose: Optional[Tuple[float, float, float]] = None,
                         landmark_classes: Optional[List[str]] = None,
                         landmark_instances: Optional[List[Dict[str, Any]]] = None,
                         landmark_config: Optional[Dict] = None,
                         waypoint_positions: Optional[List[Tuple[int, int]]] = None,
                         waypoint_ids: Optional[List[int]] = None,
                         room_area_layer: Optional[np.ndarray] = None,
                         room_area_records: Optional[List[Dict[str, Any]]] = None,
                         crop_offset: Optional[Tuple[int, int]] = None,
                         mapping_classes: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List, np.ndarray, Optional[float]]:
        """
        渲染全局地图（严格按照ZS_Evaluator的渲染逻辑 + 平滑轨迹线）
        
        Args:
            full_map: [C, H, W] 全局地图
                [0] = obstacle map (障碍物)
                [1] = explored map (已探索)
                [2] = Agent通道 (合并：0.5=轨迹, 1.0=当前位置)
                [3+] = semantic classes (用于landmark标注，不用于floor渲染)
            trajectory_points: [(x, y), ...] 轨迹坐标列表（像素坐标）
            detected_classes: 已检测类别列表
            floor: [H, W] floor地图（通过形态学方法计算，像ZS_Evaluator）
            current_pose: (x, y, orientation) 当前位姿
            landmark_classes: landmark类别列表
            landmark_config: landmark配置 {min_total_pixels, min_area_threshold}
        
        Returns:
            (sem_map_vis, global_map_with_trajectory, landmarks, global_map_rotated, last_waypoint_angle)
            - sem_map_vis: 基础渲染地图 (480×480)
            - global_map_with_trajectory: 带轨迹的旋转地图（默认480×480，裁剪后440×440）
            - landmarks: [(x, y, class_name), ...] 标注列表
            - global_map_rotated: 旋转地图（无轨迹，默认480×480，裁剪后440×440）
            - last_waypoint_angle: 最后一个waypoint相对于正前方的角度（弧度），None表示无waypoint
        
        渲染层次:
            - 白色(0): 未探索区域
            - 浅灰色(2): 已探索自由空间
            - 黑色(1): 障碍物
            - 浅绿色(5): Floor
            - 橙色: 轨迹（OpenCV后绘制）
            - 蓝色: waypoint（由 waypoint_positions 列表绘制）

        注意：
        - 不再从 Channel 2 读取 waypoint；waypoint 只由 mapper 返回的世界坐标列表绘制
        - 不渲染 bed/chair 等语义类别颜色，只用于 landmark 标注
        """
        # ===== 阶段1: 从 full_map 提取各层 mask（统一流程，obstacle/floor/landmark 均来自同一投影）=====
        # 通道布局：[0] obstacle  [1] explored  [3..3+M-1] mapping_classes  [3+M..] landmark_classes
        h, w = full_map.shape[1], full_map.shape[2]
        obstacle_mask = self._get_channel_mask(full_map, 0)   # channel 0: obstacle
        explored_mask = self._get_channel_mask(full_map, 1)   # channel 1: explored

        # ===== 阶段1.1: 创建语义地图 =====
        semantic_map = np.zeros((h, w), dtype=np.uint8)

        # Layer 1: 已探索自由空间（浅灰色）
        explored_free_mask = np.logical_and(explored_mask, ~obstacle_mask)
        semantic_map[explored_free_mask] = 2

        # Layer 2: Floor（浅绿色）
        # 使用 mapper.extract_floor() 预计算的形态学 floor（比 full_map[floor_ch] 更完整可靠）
        if floor is not None:
            floor_display_mask = np.logical_and(floor.astype(bool), explored_mask)
            semantic_map[floor_display_mask] = 5  # 浅绿色

        # 轨迹与 waypoint 都在后续用 OpenCV 叠加；这里不再读取 Channel 2 的旧残留逻辑
        
        # ===== 阶段2: PIL调色板渲染 =====
        # 现在semantic_map包含：0=未知, 1=障碍物, 2=已探索, 4=waypoint, 5=floor（轨迹稍后用OpenCV绘制）
        sem_map_vis = Image.new("P", (w, h))
        sem_map_vis.putpalette(self.color_palette)
        sem_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        
        # 坐标系变换：翻转Y轴 + RGB→BGR
        sem_map_vis = np.flipud(sem_map_vis)
        sem_map_vis = np.array(sem_map_vis)
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]  # RGB → BGR
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST)
        sem_map_vis = self._overlay_room_areas(sem_map_vis, room_area_layer, room_area_records)
        
        # ===== 阶段3: 提取Landmark位置（但不绘制）=====
        landmarks = []
        if landmark_instances:
            landmarks = self._build_landmarks_from_instances(
                landmark_instances, full_map, current_pose, crop_offset
            )
        elif landmark_classes and landmark_config:
            landmarks = self._extract_landmarks(
                full_map, detected_classes, landmark_classes,
                landmark_config['min_total_pixels'],
                landmark_config['min_area_threshold'],
                mapping_classes=mapping_classes
            )
        
        # ===== 阶段4: 准备显示（地图已在提取时旋转，agent朝向向上）=====
        # 注意：从 semantic_mapping.get_full_map_for_rendering() 返回的 full_map
        # 已经根据 agent 朝向旋转过了，所以：
        # - Agent 在地图中心 (240, 240)
        # - Agent 朝向已经是正上方（地图坐标的北）
        # - trajectory_points 也已经在旋转后的坐标系中
        # 所以这里不需要再旋转地图，直接使用即可
        
        projector = self._build_map_projector(full_map, current_pose, crop_offset)
        global_map_rotated = sem_map_vis.copy()
        global_map_with_trajectory = global_map_rotated.copy()
        last_waypoint_angle = None

        if current_pose is not None:
            # ===== 阶段5: 创建global_map的显示副本（用于绘制trajectory和landmark）=====
            # trajectory_points 是世界像素坐标，统一通过 projector 转到当前旋转显示坐标。
            if projector is not None and trajectory_points is not None and len(trajectory_points) > 1:
                trajectory_color = (0, 165, 255)  # 橙色BGR
                display_points = projector.world_points_to_global_display(trajectory_points)
                if len(display_points) > 1:
                    cv2.polylines(
                        global_map_with_trajectory,
                        [np.array(display_points, dtype=np.int32)],
                        isClosed=False,
                        color=trajectory_color,
                        thickness=3,
                    )
            
            # ===== 阶段5.3: 绘制深红色虚线指示正前方（在waypoint和箭头之前）=====
            center_x, center_y = 240, 240
            forward_line_length = 120  # 延伸120像素（约3米）
            forward_color = (0, 0, 180)  # 深红色 BGR
            forward_thickness = 2
            
            # 绘制从agent中心向正上方延伸的虚线
            # 虚线：每段10像素，间隙5像素
            dash_length = 10
            gap_length = 5
            num_dashes = int(forward_line_length / (dash_length + gap_length))
            
            for i in range(num_dashes):
                dash_start_y = 240 - i * (dash_length + gap_length)
                dash_end_y = dash_start_y - dash_length
                if dash_end_y < 240 - forward_line_length:
                    dash_end_y = 240 - forward_line_length
                cv2.line(global_map_with_trajectory, (240, int(dash_start_y)), 
                        (240, int(dash_end_y)), forward_color, forward_thickness)
            
            # ===== 阶段5.4: 绘制历史waypoint（蓝色圆圈+ID数字，在箭头之前的倒数第二层）=====
            # waypoint_positions 是世界像素坐标，需要转换到旋转后的full_map坐标
            if projector is not None and waypoint_positions is not None and waypoint_ids is not None and len(waypoint_positions) > 0:
                for idx, wp_pos in enumerate(waypoint_positions):
                    projected = projector.world_to_global_display(wp_pos[0], wp_pos[1])
                    if projected is None:
                        continue

                    display_x, display_y = int(projected[0]), int(projected[1])
                    display_id = idx + 1
                    cv2.circle(
                        global_map_with_trajectory,
                        (display_x, display_y),
                        radius=10,
                        color=(255, 0, 0),
                        thickness=-1,
                    )
                    text = str(display_id)
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.putText(
                        global_map_with_trajectory,
                        text,
                        (display_x - text_w // 2, display_y + text_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            
            # ===== 阶段5.5: 在中心绘制箭头（最后一层，覆盖waypoint）=====
            arrow_angle = np.deg2rad(-90)  # 朝上
            agent_pos = (center_x, center_y, arrow_angle)
            agent_arrow = vu.get_contour_points(agent_pos, origin=(0, 0), size=12)
            cv2.drawContours(global_map_with_trajectory, [agent_arrow], 0, (0, 0, 255), -1)
            
            # ===== 阶段5.6: 叠加黑色障碍物层（覆盖在箭头之上，使障碍物更醒目）=====
            # obstacle_mask 来自 _get_channel_mask(full_map, 0)，与 semantic_map 同源
            obstacle_mask_display = build_rotated_obstacle_mask(full_map)
            
            # 注意：obstacle_mask 现在已经在 full_map 中旋转过了
            # 所以这里直接使用，不需要再旋转
            
            # 用黑色覆盖障碍物区域（会覆盖箭头，使障碍物更醒目）
            global_map_with_trajectory[obstacle_mask_display] = [0, 0, 0]  # 黑色BGR
            global_map_rotated[obstacle_mask_display] = [0, 0, 0]  # 无轨迹版本也叠加
            
            # ===== 阶段6: global map 不绘制自定义 landmark，仅保留内部 landmarks 列表供后续距离/角度计算 =====

            # ===== 可选：裁剪到440×440（中心区域）=====
            # 默认关闭裁剪，保持完整的480×480地图
            if self.enable_global_map_crop:
                # 从480x480裁剪中心440x440区域
                crop_offset = (480 - 440) // 2  # = 20
                global_map_with_trajectory = global_map_with_trajectory[crop_offset:crop_offset+440, crop_offset:crop_offset+440].copy()
                global_map_rotated = global_map_rotated[crop_offset:crop_offset+440, crop_offset:crop_offset+440].copy()
                # print(f"✂️  Global Map 裁剪: 480×480 → 440×440")
            # else:
                # print(f"📐 Global Map 尺寸: 480×480 (未裁剪，显示完整地图)")
        
        # 添加方位标签到global map
        global_map_with_trajectory = self.add_orientation_labels(global_map_with_trajectory)
        global_map_rotated = self.add_orientation_labels(global_map_rotated)
        
        # 返回：基础地图 + 显示副本（带轨迹和landmark+waypoint） + 无轨迹的旋转地图（供local_map裁剪） + 距离信息 + 最后waypoint角度
        return sem_map_vis, global_map_with_trajectory, landmarks, global_map_rotated, last_waypoint_angle
    
    def render_local_map(self, 
                        full_map: np.ndarray,
                        trajectory_points: List[Tuple[int, int]],
                        detected_classes: List[str],
                        current_pose: Tuple[float, float, float],
                        floor: Optional[np.ndarray] = None,
                        landmark_classes: Optional[List[str]] = None,
                        landmark_instances: Optional[List[Dict[str, Any]]] = None,
                        landmark_config: Optional[Dict] = None,
                        hfov: float = 90.0,
                        waypoint_positions: Optional[List[Tuple[int, int]]] = None,
                        waypoint_ids: Optional[List[int]] = None,
                        room_area_layer: Optional[np.ndarray] = None,
                        room_area_records: Optional[List[Dict[str, Any]]] = None,
                        crop_offset: Optional[Tuple[int, int]] = None,
                        mapping_classes: Optional[List[str]] = None) -> np.ndarray:
        """
        独立渲染局部地图（不继承全局地图，完全独立构建）
        
        注意：Local Map不渲染waypoint标记，因为action模块不需要waypoint信息
        
        Args:
            full_map: [C, H, W] 全局地图数据
            trajectory_points: [(x, y), ...] 原始轨迹坐标列表（地图像素坐标）
            detected_classes: 已检测类别列表
            current_pose: (x, y, orientation) 当前位姿（米）
            floor: [H, W] floor地图
            landmark_classes: landmark类别列表
            landmark_config: landmark配置
            hfov: 水平视野角度（默认90度）
            waypoint_positions: 未使用（保留接口兼容性）
            waypoint_ids: 未使用（保留接口兼容性）
        
        Returns:
            local_map: 局部地图（最终 440×440）
        """
        if full_map is None:
            return None
        
        # ===== 阶段1: 从 full_map 提取各层 mask（与 render_global_map 完全相同的通道布局）=====
        h, w = full_map.shape[1], full_map.shape[2]
        obstacle_mask = self._get_channel_mask(full_map, 0)   # channel 0: obstacle
        explored_mask = self._get_channel_mask(full_map, 1)   # channel 1: explored

        # 创建语义地图
        semantic_map = np.zeros((h, w), dtype=np.uint8)

        # Layer 1: 已探索自由空间（浅灰色）
        explored_free_mask = np.logical_and(explored_mask, ~obstacle_mask)
        semantic_map[explored_free_mask] = 2

        # Layer 2: Floor（浅绿色）
        # 使用 mapper.extract_floor() 预计算的形态学 floor（与 render_global_map 逻辑一致）
        if floor is not None:
            floor_display_mask = np.logical_and(floor.astype(bool), explored_mask)
            semantic_map[floor_display_mask] = 5

        # Layer 3: 不渲染轨迹和waypoint（后续用OpenCV绘制轨迹）
        # Local map不显示历史waypoint，只显示轨迹
        
        # ===== 阶段2: PIL调色板渲染 =====
        sem_map_vis = Image.new("P", (w, h))
        sem_map_vis.putpalette(self.color_palette)
        sem_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        
        # 坐标系变换
        sem_map_vis = np.flipud(sem_map_vis)
        sem_map_vis = np.array(sem_map_vis)
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]  # RGB → BGR
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST)
        sem_map_vis = self._overlay_room_areas(sem_map_vis, room_area_layer, room_area_records, alpha=0.28)
        
        # ===== 阶段3: 准备显示（地图已在提取时旋转）=====
        projector = self._build_map_projector(full_map, current_pose, crop_offset)
        local_map = sem_map_vis.copy()
        
        # Agent在中心 (240, 240)
        center_x, center_y = 240, 240
        
        # ===== 阶段4: 裁剪中心240×240区域并放大到480×480 =====
        center_x, center_y = 240, 240
        crop_size = 240
        crop_half = crop_size // 2
        
        x1 = center_x - crop_half
        x2 = center_x + crop_half
        y1 = center_y - crop_half
        y2 = center_y + crop_half
        
        local_map = local_map[y1:y2, x1:x2].copy()
        local_map = cv2.resize(local_map, (480, 480), interpolation=cv2.INTER_NEAREST)
        
        # ===== 阶段5: 先准备轨迹点数据，稍后在FOV之后绘制 =====
        trajectory_display_points = []
        if projector is not None and trajectory_points is not None and len(trajectory_points) > 1:
            trajectory_display_points = projector.world_points_to_local_display(trajectory_points)
        
        # ===== 阶段6: 绘制FOV可见区域（考虑障碍物遮挡）=====
        # 480像素 = 12m，所以1像素 = 2.5cm
        # 5米 = 500cm ÷ 2.5cm/pixel = 200像素
        fov_center_x, fov_center_y = 240, 240
        fov_radius = 200  # 5米视野半径
        
        # Agent朝上（-90度），FOV扇形中心线也朝上
        fov_center_angle = -90
        fov_start_angle = fov_center_angle - hfov / 2
        fov_end_angle = fov_center_angle + hfov / 2
        
        import math
        
        # 先获取旋转后的障碍物掩码（用于raycasting）
        # obstacle_mask 来自 _get_channel_mask(full_map, 0)，已在 full_map 中旋转
        obstacle_mask_resized = build_rotated_obstacle_mask(full_map)
        
        # 裁剪中心240×240区域
        obstacle_crop = obstacle_mask_resized[120:360, 120:360]
        obstacle_local = cv2.resize(obstacle_crop.astype(np.uint8) * 255, 
                                   (480, 480), 
                                   interpolation=cv2.INTER_NEAREST) > 127
        
        # 对障碍物掩码进行形态学膨胀，填补小缺口，减少突出的射线
        kernel = np.ones((3, 3), np.uint8)
        obstacle_local_dilated = cv2.dilate(obstacle_local.astype(np.uint8), kernel, iterations=1).astype(bool)
        
        # 使用raycasting计算可见多边形
        num_rays = 180  # 每度2条射线，确保精细度
        angle_step = (fov_end_angle - fov_start_angle) / num_rays
        
        visible_points = [(fov_center_x, fov_center_y)]  # 起始点是agent位置
        
        for i in range(num_rays + 1):
            angle = fov_start_angle + i * angle_step
            angle_rad = math.radians(angle)
            
            # 沿射线方向逐步检测
            max_distance = fov_radius
            ray_end_x, ray_end_y = fov_center_x, fov_center_y
            
            # 使用0.5像素步长提高检测精度
            step_size = 0.5
            num_steps = int(max_distance / step_size)
            
            for step in range(num_steps):
                distance = step * step_size
                test_x = fov_center_x + distance * math.cos(angle_rad)
                test_y = fov_center_y + distance * math.sin(angle_rad)
                
                # 检查是否越界
                if test_x < 0 or test_x >= 480 or test_y < 0 or test_y >= 480:
                    ray_end_x, ray_end_y = test_x, test_y
                    break
                
                # 检查是否碰到障碍物（使用膨胀后的障碍物掩码）
                if obstacle_local_dilated[int(test_y), int(test_x)]:
                    ray_end_x, ray_end_y = test_x, test_y
                    break
                
                # 未碰到障碍物，继续延伸
                ray_end_x, ray_end_y = test_x, test_y
            
            visible_points.append((int(ray_end_x), int(ray_end_y)))
        
        # 绘制可见区域多边形（蓝色填充，不透明）
        if len(visible_points) > 2:
            visible_polygon = np.array(visible_points, dtype=np.int32)
            
            # 直接填充蓝色（不需要透明度，因为后续会叠加障碍物、轨迹等）
            fill_color = (255, 200, 100)  # 蓝色 BGR格式，明显但不刺眼
            cv2.fillPoly(local_map, [visible_polygon], color=fill_color)
            
            # 绘制可见区域边框（深蓝色实线）
            border_color = (180, 100, 0)  # 深蓝色 BGR
            border_thickness = 2
            cv2.polylines(local_map, [visible_polygon], isClosed=True, 
                         color=border_color, thickness=border_thickness)
        
        # ===== 阶段6.5: 绘制轨迹线（在FOV之后，确保轨迹可见）=====
        if len(trajectory_display_points) > 1:
            trajectory_color = (0, 165, 255)  # 橙色BGR
            for i in range(len(trajectory_display_points) - 1):
                pt1 = trajectory_display_points[i]
                pt2 = trajectory_display_points[i + 1]
                if (0 <= pt1[0] < 480 and 0 <= pt1[1] < 480 and
                    0 <= pt2[0] < 480 and 0 <= pt2[1] < 480):
                    cv2.line(local_map, pt1, pt2, trajectory_color, thickness=3)
        
        # ===== 绘制深红色虚线指示正前方（在箭头下层）=====
        forward_line_length = 120  # 延伸120像素（约3米）
        forward_color = (0, 0, 180)  # 深红色 BGR
        forward_thickness = 2
        
        # 绘制从agent中心向正上方延伸的虚线
        start_point = (fov_center_x, fov_center_y)
        end_point = (fov_center_x, fov_center_y - forward_line_length)  # 朝上是Y减小
        
        # 虚线：每段10像素，间隙5像素
        dash_length = 10
        gap_length = 5
        total_length = forward_line_length
        num_dashes = int(total_length / (dash_length + gap_length))
        
        for i in range(num_dashes):
            dash_start_y = fov_center_y - i * (dash_length + gap_length)
            dash_end_y = dash_start_y - dash_length
            if dash_end_y < fov_center_y - forward_line_length:
                dash_end_y = fov_center_y - forward_line_length
            cv2.line(local_map, (fov_center_x, int(dash_start_y)), 
                    (fov_center_x, int(dash_end_y)), forward_color, forward_thickness)
        
        # ===== 绘制0.5m半径圆圈（深绿色，标识当前位置附近区域）=====
        # 480像素 = 12m，所以1m = 40像素，0.5m = 20像素
        nearby_radius = 20  # 0.5m半径
        nearby_color = (0, 100, 0)  # 深绿色BGR
        nearby_thickness = 2  # 2像素线宽
        cv2.circle(local_map, (fov_center_x, fov_center_y), nearby_radius, nearby_color, nearby_thickness)
        
        # ===== 阶段7: 绘制朝上的大箭头（在轨迹和虚线之上）=====
        arrow_color = (0, 0, 255)  # 亮红色BGR
        arrow_angle = np.deg2rad(-90)  # 朝上
        agent_pos = (fov_center_x, fov_center_y, arrow_angle)
        agent_arrow = vu.get_contour_points(agent_pos, origin=(0, 0), size=24)
        cv2.drawContours(local_map, [agent_arrow], 0, arrow_color, -1)
        
        # ===== 阶段8: 叠加黑色障碍物层（覆盖在箭头之上，使障碍物更醒目）=====
        # 用黑色覆盖障碍物区域（obstacle_local已在阶段6计算）
        local_map[obstacle_local] = [0, 0, 0]  # 黑色BGR
        
        # ===== 阶段9: 绘制Landmark标记（紫色圆球，最上层，不被遮挡）=====
        landmarks = []
        if landmark_instances:
            landmarks = self._build_landmarks_from_instances(
                landmark_instances, full_map, current_pose, crop_offset
            )
        elif landmark_classes and landmark_config:
            # full_map 已由 get_full_map_for_rendering(rotate_to_agent_heading=True) 旋转过
            # _extract_landmarks 返回的 (marker_x, marker_y) 已经是旋转后地图的像素坐标
            # 与 render_global_map 的处理完全一致：scale + flipud + 裁剪中心区域
            # 不需要再做额外旋转（否则会双重旋转导致位置偏移）
            landmarks = self._extract_landmarks(
                full_map, detected_classes, landmark_classes,
                landmark_config['min_total_pixels'],
                landmark_config['min_area_threshold'],
                mapping_classes=mapping_classes
            )

        for marker_x, marker_y, cls_name, _dist_m, _angle_deg in landmarks:
            local_display = None
            if projector is not None:
                local_display = projector.rotated_to_local_display(marker_y, marker_x)
            if local_display is not None:
                local_x, local_y = local_display
                local_landmark_radius = 10
                cv2.circle(local_map,
                           (int(local_x), int(local_y)),
                           local_landmark_radius,
                           landmark_marker_color, -1)
                cv2.circle(local_map,
                           (int(local_x), int(local_y)),
                           local_landmark_radius,
                           landmark_marker_border, 1)
        
        # ===== 阶段10: 最终裁剪到440×440（中心区域）=====
        # 从480x480裁剪中心440x440区域
        crop_offset = (480 - 440) // 2  # = 20
        local_map_cropped = local_map[crop_offset:crop_offset+440, crop_offset:crop_offset+440].copy()
        
        # 添加方位标签
        local_map_cropped = self.add_orientation_labels(local_map_cropped)
        
        return local_map_cropped
    
    def add_orientation_labels(self, map_image: np.ndarray) -> np.ndarray:
        """
        在地图四周添加方位标签（俯视图）- 深红字+白底
        地图尺寸：440x440
        
        Args:
            map_image: 地图图像 (440, 440, 3) BGR格式
        
        Returns:
            带方位标签的地图
        """
        h, w = map_image.shape[:2]
        labeled_map = map_image.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # 加粗字体
        text_thickness = 2  # 加粗
        text_color = (0, 0, 139)  # 深红色BGR
        bg_color = (255, 255, 255)  # 白色背景
        
        # 定义方位标签
        labels = {
            'FRONT': (w // 2, 20),  # 上方
            'BACK': (w // 2, h - 8),  # 下方
            'LEFT': (40, h // 2),  # 左侧
            'RIGHT': (w - 40, h // 2)  # 右侧
        }
        
        for text, (x, y) in labels.items():
            # 计算文字大小
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
            
            # 调整位置使文字居中
            if text in ['FRONT', 'BACK']:
                text_x = x - text_width // 2
                text_y = y
            else:  # LEFT, RIGHT
                text_x = x - text_width // 2
                text_y = y + text_height // 2
            
            # 绘制白色背景矩形（底部间距更小）
            padding_top = 3
            padding_side = 3
            padding_bottom = 1
            cv2.rectangle(labeled_map,
                         (text_x - padding_side, text_y - text_height - padding_top),
                         (text_x + text_width + padding_side, text_y + baseline + padding_bottom),
                         bg_color, -1)
            
            # 绘制深红色文字
            cv2.putText(labeled_map, text, (text_x, text_y),
                       font, font_scale, text_color, text_thickness, cv2.LINE_AA)
        
        return labeled_map

    def _estimate_mask_rel_xy(self,
                              mask_2d: np.ndarray,
                              depth_img: np.ndarray,
                              hfov: float,
                              sample_stride: int = 4) -> Optional[Tuple[float, float]]:
        """用 mask+depth 估计目标在 agent 坐标系中的前向/右向位置。"""
        if mask_2d is None or depth_img is None:
            return None

        if mask_2d.shape != depth_img.shape:
            mask_2d = cv2.resize(
                mask_2d.astype(np.float32),
                (depth_img.shape[1], depth_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        mask_bool = mask_2d > 0.5
        valid_mask = mask_bool & np.isfinite(depth_img) & (depth_img > 0.02)
        if not np.any(valid_mask):
            return None

        sample_mask = valid_mask
        if mask_bool.sum() >= 36:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            interior_mask = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=1) > 0
            interior_valid = interior_mask & valid_mask
            edge_valid = valid_mask & (~interior_mask)

            if np.count_nonzero(interior_valid) >= 24 and np.count_nonzero(edge_valid) >= 24:
                edge_depth = depth_img[edge_valid].astype(np.float32)
                interior_depth = depth_img[interior_valid].astype(np.float32)
                edge_median = float(np.median(edge_depth))
                interior_median = float(np.median(interior_depth))
                opening_gap_m = interior_median - edge_median
                opening_gap_threshold = max(0.6, 0.35 * max(edge_median, 0.1))
                if opening_gap_m >= opening_gap_threshold:
                    # Opening-like structures (doorways / hallways) are often much deeper
                    # in the center than at the frame edges, so use edge geometry instead.
                    sample_mask = edge_valid

        ys, xs = np.nonzero(sample_mask)
        if sample_stride > 1 and ys.size > sample_stride:
            ys = ys[::sample_stride]
            xs = xs[::sample_stride]

        depth_vals = depth_img[ys, xs].astype(np.float32)
        if depth_vals.size == 0:
            return None

        _, w_img = depth_img.shape[:2]
        xc = (w_img - 1) / 2.0
        focal = (w_img / 2.0) / np.tan(np.deg2rad(float(hfov)) / 2.0)
        if focal <= 1e-6:
            return None

        right_vals = ((xs.astype(np.float32) - xc) * depth_vals) / float(focal)
        forward_vals = depth_vals
        return float(np.median(forward_vals)), float(np.median(right_vals))

    def _project_landmark_instances_from_detections(self,
                                                    detections,
                                                    labels: Optional[List[str]],
                                                    landmark_classes: Optional[List[str]],
                                                    depth_meters: Optional[np.ndarray],
                                                    current_pose: Optional[Tuple[float, float, float]],
                                                    hfov: float,
                                                    topk: Optional[int] = None) -> List[Dict[str, Any]]:
        """将每个 landmark 检测实例直接投影为世界坐标实例列表。"""
        if (detections is None or getattr(detections, 'xyxy', None) is None or
                len(detections.xyxy) == 0 or not labels or not landmark_classes or
                depth_meters is None or current_pose is None):
            return []

        canonical = {name.strip().lower(): name for name in landmark_classes}
        curr_x_m, curr_y_m, curr_ori_deg = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])
        theta = np.deg2rad(curr_ori_deg)

        per_class: Dict[str, List[Dict[str, Any]]] = {}
        for i in range(len(detections.xyxy)):
            label = labels[i] if i < len(labels) else f"object_{i}"
            parts = label.split()
            label_name = ' '.join(parts[:-1]) if len(parts) > 1 else (parts[0] if parts else "unknown")
            confidence = float(parts[-1]) if len(parts) > 1 else 0.0
            matched_landmark = canonical.get(label_name.strip().lower())
            if matched_landmark is None:
                continue

            det_mask = None
            if getattr(detections, 'mask', None) is not None and i < len(detections.mask):
                det_mask = detections.mask[i]
            rel_xy = self._estimate_mask_rel_xy(det_mask, depth_meters, hfov)
            if rel_xy is None:
                continue

            forward_m, right_m = rel_xy
            dx = forward_m * np.cos(theta) + right_m * np.sin(theta)
            dy = forward_m * np.sin(theta) - right_m * np.cos(theta)
            world_x_m = curr_x_m + dx
            world_y_m = curr_y_m + dy

            world_row_px = int(round(world_y_m * 100.0 / self.resolution))
            world_col_px = int(round(world_x_m * 100.0 / self.resolution))
            abs_angle = np.degrees(np.arctan2(dy, dx))
            rel_bearing = curr_ori_deg - abs_angle
            rel_bearing = ((rel_bearing + 180.0) % 360.0) - 180.0
            dist_m = float(np.hypot(dx, dy))

            per_class.setdefault(matched_landmark, []).append({
                "name": matched_landmark,
                "confidence": float(confidence),
                "distance_m": dist_m,
                "angle_deg": float(rel_bearing),
                "world_row_px": world_row_px,
                "world_col_px": world_col_px,
                "world_x_m": float(world_x_m),
                "world_y_m": float(world_y_m),
            })

        projected_instances: List[Dict[str, Any]] = []
        for cls_name, candidates in per_class.items():
            ranked = sorted(candidates, key=lambda item: (-item["confidence"], item["distance_m"]))
            if topk is not None and int(topk) > 0:
                ranked = ranked[:max(1, int(topk))]
            for inst_idx, item in enumerate(ranked):
                item["instance_idx"] = inst_idx
                projected_instances.append(item)

        return projected_instances

    def _merge_landmark_instances_world(self,
                                        existing_instances: Optional[List[Dict[str, Any]]],
                                        new_instances: Optional[List[Dict[str, Any]]],
                                        current_pose: Optional[Tuple[float, float, float]],
                                        topk: int = landmark_instance_topk,
                                        merge_radius_m: float = landmark_instance_merge_radius_m
                                        ) -> List[Dict[str, Any]]:
        """在同一子任务内累计 landmark 实例，并按世界坐标去重后保留每类 top-k。"""
        merged_by_class: Dict[str, List[Dict[str, Any]]] = {}

        def _curr_metrics(inst: Dict[str, Any]) -> Tuple[float, float]:
            if current_pose is None or "world_x_m" not in inst or "world_y_m" not in inst:
                return (
                    float(inst.get("distance_m", 1e9)),
                    float(inst.get("angle_deg", 0.0)),
                )
            curr_x, curr_y, curr_ori = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])
            dx_m = float(inst["world_x_m"]) - curr_x
            dy_m = float(inst["world_y_m"]) - curr_y
            dist_m = float(np.hypot(dx_m, dy_m))
            abs_angle = np.degrees(np.arctan2(dy_m, dx_m)) if dist_m > 1e-6 else curr_ori
            rel_bearing = curr_ori - abs_angle
            rel_bearing = ((rel_bearing + 180.0) % 360.0) - 180.0
            return dist_m, float(rel_bearing)

        def _merge_one(inst: Dict[str, Any]) -> None:
            cls_name = inst.get("name")
            if not cls_name:
                return
            cls_bucket = merged_by_class.setdefault(cls_name, [])
            best_idx = None
            best_dist = None
            if "world_x_m" in inst and "world_y_m" in inst:
                for idx, old in enumerate(cls_bucket):
                    if "world_x_m" not in old or "world_y_m" not in old:
                        continue
                    dist = float(np.hypot(
                        float(old["world_x_m"]) - float(inst["world_x_m"]),
                        float(old["world_y_m"]) - float(inst["world_y_m"]),
                    ))
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_idx = idx

            if best_idx is not None and best_dist is not None and best_dist <= merge_radius_m:
                old = cls_bucket[best_idx]
                refreshed = dict(old)
                refreshed.update(inst)
                refreshed["confidence"] = max(
                    float(old.get("confidence", 0.0)),
                    float(inst.get("confidence", 0.0)),
                )
                cls_bucket[best_idx] = refreshed
            else:
                cls_bucket.append(dict(inst))

        for inst in existing_instances or []:
            _merge_one(dict(inst))
        for inst in new_instances or []:
            _merge_one(dict(inst))

        merged_instances: List[Dict[str, Any]] = []
        keep_n = max(1, int(topk))
        for cls_name, bucket in merged_by_class.items():
            ranked = sorted(
                bucket,
                key=lambda item: (-float(item.get("confidence", 0.0)), _curr_metrics(item)[0]),
            )
            kept = ranked[:keep_n]
            kept = sorted(kept, key=lambda item: _curr_metrics(item)[0])
            for inst_idx, item in enumerate(kept):
                dist_m, angle_deg = _curr_metrics(item)
                normalized = dict(item)
                normalized["distance_m"] = float(dist_m)
                normalized["angle_deg"] = float(angle_deg)
                normalized["instance_idx"] = inst_idx
                merged_instances.append(normalized)

        return merged_instances

    def _world_instance_to_rotated_landmark(self,
                                            inst: Dict[str, Any],
                                            full_map: np.ndarray,
                                            current_pose: Optional[Tuple[float, float, float]],
                                            crop_offset: Optional[Tuple[int, int]]) -> Optional[Tuple[float, float, str, float, float]]:
        """将世界坐标实例转换到当前旋转后 full_map 像素坐标。"""
        if current_pose is None or crop_offset is None or full_map is None:
            return None

        projector = self._build_map_projector(full_map, current_pose, crop_offset)
        if projector is None:
            return None

        world_row_px = int(inst["world_row_px"])
        world_col_px = int(inst["world_col_px"])

        rotated = projector.world_to_rotated_pixel(world_row_px, world_col_px)
        if rotated is None:
            return None
        rotated_row, rotated_col = rotated

        if "world_x_m" in inst and "world_y_m" in inst:
            dx_m = float(inst["world_x_m"]) - float(current_pose[0])
            dy_m = float(inst["world_y_m"]) - float(current_pose[1])
            dist_m = float(np.hypot(dx_m, dy_m))
            abs_angle = np.degrees(np.arctan2(dy_m, dx_m)) if dist_m > 1e-6 else float(current_pose[2])
            rel_bearing = float(current_pose[2]) - abs_angle
            rel_bearing = ((rel_bearing + 180.0) % 360.0) - 180.0
        else:
            dist_m = float(inst.get("distance_m", 0.0))
            rel_bearing = float(inst.get("angle_deg", 0.0))

        return (
            float(rotated_col),
            float(rotated_row),
            inst["name"],
            dist_m,
            rel_bearing,
        )

    def _build_landmarks_from_instances(self,
                                        landmark_instances: Optional[List[Dict[str, Any]]],
                                        full_map: np.ndarray,
                                        current_pose: Optional[Tuple[float, float, float]],
                                        crop_offset: Optional[Tuple[int, int]]) -> List[Tuple[float, float, str, float, float]]:
        """把显式实例列表转换成当前渲染使用的 landmark 点。"""
        if not landmark_instances:
            return []

        landmarks: List[Tuple[float, float, str, float, float]] = []
        for inst in landmark_instances:
            converted = self._world_instance_to_rotated_landmark(inst, full_map, current_pose, crop_offset)
            if converted is not None:
                landmarks.append(converted)
        return landmarks
    
    def render_detection_bbox(self, 
                              rgb: np.ndarray,
                              detections,  # sv.Detections object
                              labels: List[str],
                              landmark_classes: Optional[List[str]] = None,
                              mapping_classes: Optional[List[str]] = None,
                              depth_meters: Optional[np.ndarray] = None,
                              hfov: float = 79.0,
                              landmark_dist_map: Optional[Dict[str, Tuple[float, float]]] = None,
                              landmark_dist_map_multi: Optional[Dict[str, List[Tuple[float, float]]]] = None,
                              landmark_masks: Optional[np.ndarray] = None,
                              show_action_partitions: bool = True,
                              append_bottom_strip: bool = True,
                              controller=None) -> np.ndarray:
        """
        直接在RGB上渲染边界框（只标注Landmark类别，显示距离+水平偏角）
        
        Args:
            rgb: RGB图像 (H, W, 3) BGR格式
            detections: supervision Detections对象
            labels: 标签列表 (例如: ["chair 0.85", "table 0.92"])
            landmark_classes: Landmark类别列表（只标注这些类别）
            mapping_classes: Mapping类别列表（不标注，仅用于建图）
            depth_meters: 深度图；仅在同类多实例时用于把当前检测实例匹配到地图实例
            hfov: 相机水平视场角；仅用于实例匹配，不用于最终距离/角度显示
            landmark_dist_map: {class_name: (dist_m, rel_angle_deg)} 由地图世界坐标预计算
            landmark_dist_map_multi: {class_name: [(dist_m, rel_angle_deg), ...]} 同类多实例地图信息
        
        Returns:
            detection_vis: 检测可视化图像（只显示Landmark边界框）
        """
        detection_vis = rgb.copy()
        if show_action_partitions:
            draw_action_partition_lines(detection_vis, hfov_deg=float(hfov))
        
        if detections is None or len(detections.xyxy) == 0:
            if controller is not None:
                controller.latest_visible_landmark_entries = []
            return detection_vis, [], set(), None
        
        # 统计检测到的landmark
        detected_landmarks = []
        visible_entries_meta = []
        matched_in_view: set = set()  # 当前帧中实际可见的landmark类名
        candidate_entries: List[Dict[str, Any]] = []
        draw_items: List[LandmarkDrawItem] = []

        def _bbox_center_angle_deg(x1: int, x2: int, width: int) -> float:
            xc = (width - 1) / 2.0
            focal = (width / 2.0) / np.tan(np.deg2rad(float(hfov)) / 2.0)
            if focal <= 1e-6:
                return 0.0
            center_x = (float(x1) + float(x2)) / 2.0
            return float(np.degrees(np.arctan2(center_x - xc, focal)))

        depth_for_match = depth_meters
        if depth_for_match is None and controller is not None:
            depth_for_match = getattr(controller, "latest_depth_meters", None)

        for i in range(len(detections.xyxy)):
            bbox = detections.xyxy[i]
            label = labels[i] if i < len(labels) else f"object_{i}"
            
            # 提取类别名和置信度
            parts = label.split()
            label_name = ' '.join(parts[:-1]) if len(parts) > 1 else (parts[0] if len(parts) > 0 else "unknown")
            confidence = float(parts[-1]) if len(parts) > 1 else 0.0
            
            # 只标注在landmark_classes中的类别（规范化后精确短语匹配）
            matched_landmark = None
            if landmark_classes:
                lm_name_map = {lm.strip().lower(): lm for lm in landmark_classes}
                label_name_norm = label_name.strip().lower()
                if label_name_norm in lm_name_map:
                    matched_landmark = lm_name_map[label_name_norm]
            if matched_landmark is None:
                continue  # 跳过非Landmark类别
            
            x1, y1, x2, y2 = map(int, bbox)
            label_name = matched_landmark  # 用完整landmark名称显示

            # 仅用 bbox 中心做文字框定位；同类多实例时才做 mask+depth 到地图实例的匹配。
            _, w_img = rgb.shape[:2]
            det_mask = None
            if getattr(detections, "mask", None) is not None and i < len(detections.mask):
                det_mask = detections.mask[i]
            det_rel_xy = None
            if det_mask is not None and depth_for_match is not None:
                det_rel_xy = self._estimate_mask_rel_xy(det_mask, depth_for_match, hfov)

            candidate_entries.append({
                "name": label_name,
                "confidence": float(confidence),
                "bbox": (x1, y1, x2, y2),
                "det_rel_xy": det_rel_xy,
                "w_img": w_img,
                "raw_index": i,
            })

        candidate_entries.sort(key=lambda item: (-float(item["confidence"]), int(item["raw_index"])))
        selected_entries: List[Dict[str, Any]] = []
        for candidate in candidate_entries:
            if any(
                self._is_duplicate_detection_candidate(candidate, kept_candidate)
                for kept_candidate in selected_entries
            ):
                continue
            selected_entries.append(candidate)
            if len(selected_entries) >= max(1, int(detection_visible_topk)):
                break
        used_map_candidates = {}

        for candidate in selected_entries:
            label_name = candidate["name"]
            confidence = float(candidate["confidence"])
            x1, y1, x2, y2 = candidate["bbox"]
            det_rel_xy = candidate["det_rel_xy"]
            w_img = candidate["w_img"]

            detected_landmarks.append((label_name, confidence))
            matched_in_view.add(label_name)

            same_cls_total = len(landmark_dist_map_multi.get(label_name, [])) if landmark_dist_map_multi else 1

            # 只使用地图里已经投影/存储的 landmark 相对当前 pose 的距离与角度。
            map_dist_m = None
            map_angle_deg = None
            map_instance_idx = None
            if landmark_dist_map_multi and label_name in landmark_dist_map_multi:
                used_set = used_map_candidates.setdefault(label_name, set())
                candidates = sorted(landmark_dist_map_multi[label_name], key=lambda x: x[0])
                ranked_candidates = []
                for idx_c, (dist_m_c, angle_deg_c) in enumerate(candidates):
                    if idx_c in used_set:
                        continue
                    angle_rad_c = np.deg2rad(angle_deg_c)
                    cand_rel_xy = (
                        float(dist_m_c * np.cos(angle_rad_c)),
                        float(dist_m_c * np.sin(angle_rad_c)),
                    )
                    if det_rel_xy is not None:
                        match_cost = float(np.hypot(
                            cand_rel_xy[0] - det_rel_xy[0],
                            cand_rel_xy[1] - det_rel_xy[1],
                        ))
                    else:
                        match_cost = float(dist_m_c)
                    ranked_candidates.append(
                        (idx_c, dist_m_c, angle_deg_c, match_cost)
                    )
                if ranked_candidates:
                    ranked_candidates.sort(key=lambda item: (item[3], item[1]))
                    map_instance_idx, map_dist_m, map_angle_deg, _ = ranked_candidates[0]
                    used_set.add(map_instance_idx)
            elif landmark_dist_map and label_name in landmark_dist_map:
                map_dist_m, map_angle_deg = landmark_dist_map[label_name]
                map_instance_idx = 0

            display_dist_m = map_dist_m
            display_angle_deg = map_angle_deg

            # bbox上方只显示最终用于决策的距离和动作角度
            row1 = ""
            inst_prefix = ""
            if map_instance_idx is not None and same_cls_total > 1:
                inst_prefix = f"#{map_instance_idx + 1} "
            shown_dist_m = display_dist_m
            shown_angle_deg = display_angle_deg
            if display_dist_m is not None and display_angle_deg is not None:
                row1 = (
                    f"{inst_prefix}{display_dist_m:.1f}m {format_relative_direction(display_angle_deg)}"
                )
            elif display_dist_m is not None:
                row1 = f"{inst_prefix}{display_dist_m:.1f}m"
            else:
                fallback_angle_deg = None
                fallback_dist_str = None
                fallback_dist_m = None
                if det_rel_xy is not None:
                    forward_m, right_m = det_rel_xy
                    fallback_dist_m = float(np.hypot(forward_m, right_m))
                    fallback_angle_deg = float(np.degrees(np.arctan2(right_m, forward_m)))
                    if fallback_dist_m > 0.05:
                        fallback_dist_str = f"{min(fallback_dist_m, 5.0):.1f}m"
                if fallback_angle_deg is None:
                    fallback_angle_deg = _bbox_center_angle_deg(x1, x2, w_img)
                if fallback_dist_str is None:
                    fallback_dist_str = ">5.0m"
                shown_dist_m = fallback_dist_m if fallback_dist_m is not None else 5.1
                shown_angle_deg = fallback_angle_deg
                row1 = f"{inst_prefix}{fallback_dist_str} {format_relative_direction(fallback_angle_deg)}"

            if shown_dist_m is not None and shown_angle_deg is not None:
                visible_entries_meta.append({
                    "name": label_name,
                    "confidence": float(confidence),
                    "distance_m": float(shown_dist_m),
                    "angle_deg": float(shown_angle_deg),
                    "instance_idx": map_instance_idx,
                })
            draw_items.append(
                LandmarkDrawItem(
                    bbox=(x1, y1, x2, y2),
                    label_text=row1,
                    distance_m=float(shown_dist_m) if shown_dist_m is not None else 999.0,
                )
            )

        # 先渲染bbox框
        color = detection_colors["landmark"]
        thickness = detection_thickness["landmark"]
        draw_landmark_boxes(detection_vis, draw_items, color, thickness)
        draw_landmark_labels(detection_vis, draw_items, color)
        
        # ── 底部条带：所有landmark信息（可见+离屏）拼接在图像下方，白底黑字 ──
        # [Visible] = 当前帧检测到，[Off-screen] = 地图中存在但视野外
        # 构建离屏landmark列表：优先保留多实例（同类多个实例分别显示距离/角度）
        # 仅用于渲染展示，不改变任何真实地图坐标。
        offscreen_items = []  # [{"name", "instance_idx", "distance_m", "angle_deg", "confidence"}]
        if controller is not None and getattr(controller, "latest_landmark_instances_world", None):
            visible_instance_indices = {}
            for entry in visible_entries_meta:
                if entry.get("instance_idx") is None:
                    continue
                visible_instance_indices.setdefault(entry["name"], set()).add(entry["instance_idx"])

            ranked_instances = sorted(
                (dict(item) for item in controller.latest_landmark_instances_world),
                key=lambda item: (
                    float(item.get("distance_m", 1e9)),
                    str(item.get("name", "")),
                    int(item.get("instance_idx", 0) or 0),
                ),
            )
            for inst in ranked_instances:
                cls_name = inst.get("name")
                inst_idx = inst.get("instance_idx")
                distance_m = inst.get("distance_m")
                angle_deg = inst.get("angle_deg")
                if cls_name is None or inst_idx is None or distance_m is None or angle_deg is None:
                    continue
                if int(inst_idx) in visible_instance_indices.get(cls_name, set()):
                    continue
                offscreen_items.append({
                    "name": str(cls_name),
                    "instance_idx": int(inst_idx),
                    "distance_m": float(distance_m),
                    "angle_deg": float(angle_deg),
                    "confidence": float(inst.get("confidence", 0.0)),
                })
        elif landmark_dist_map_multi:
            visible_instance_indices = {}
            for entry in visible_entries_meta:
                if entry.get("instance_idx") is None:
                    continue
                visible_instance_indices.setdefault(entry["name"], set()).add(entry["instance_idx"])

            for cls_name, candidates in landmark_dist_map_multi.items():
                sorted_candidates = sorted(candidates, key=lambda x: x[0])
                used_set = visible_instance_indices.get(cls_name, set())
                for inst_idx, (d_m, a_deg) in enumerate(sorted_candidates):
                    if inst_idx in used_set:
                        continue
                    offscreen_items.append({
                        "name": str(cls_name),
                        "instance_idx": int(inst_idx),
                        "distance_m": float(d_m),
                        "angle_deg": float(a_deg),
                        "confidence": 0.0,
                    })
        elif landmark_dist_map:
            for cls_name, (d_m, a_deg) in sorted(landmark_dist_map.items(), key=lambda x: x[1][0]):
                if cls_name in matched_in_view:
                    continue
                offscreen_items.append({
                    "name": str(cls_name),
                    "instance_idx": 0,
                    "distance_m": float(d_m),
                    "angle_deg": float(a_deg),
                    "confidence": 0.0,
                })

        strip = None
        if visible_entries_meta or offscreen_items:
            item_lines = build_landmark_strip_lines(
                visible_entries_meta,
                offscreen_items,
                landmark_dist_map_multi=landmark_dist_map_multi,
            )
            strip = render_landmark_strip(detection_vis.shape[1], item_lines)

            if append_bottom_strip:
                detection_vis = np.vstack([detection_vis, strip])

        if controller is not None:
            controller.latest_visible_landmark_entries = visible_entries_meta

        # 返回检测可视化、检测到的landmark列表、已匹配的类名集合和底部条带
        return detection_vis, detected_landmarks, matched_in_view, strip
    
    # ========== 保存方法 ==========
    
    def save_rgb(self, step: int, episode_id: int, rgb: np.ndarray, phase: str = "action", controller = None) -> str:
        """
        保存原始RGB帧（添加距离线）
        
        Args:
            step: 步数
            episode_id: episode ID
            rgb: RGB图像 (H, W, 3) BGR格式
            phase: 阶段标识 ("initial", "action1a", "verify1a" 等)
            controller: VLMNavigationController实例（用于访问_draw_distance_rays_on_first_person_view）
        
        Returns:
            save_path: 保存路径
        """
        # 如果是action阶段且提供了controller，绘制距离线
        if phase.startswith('action') and controller is not None:
            if hasattr(controller, '_draw_distance_rays_on_first_person_view') and hasattr(controller, 'latest_obstacle_distances'):
                rgb = controller._draw_distance_rays_on_first_person_view(rgb.copy(), controller.latest_obstacle_distances)
        
        episode_dir = self._create_episode_directories(episode_id)
        save_path = os.path.join(episode_dir, 'rgb', f'step_{step:04d}_{phase}.png')
        cv2.imwrite(save_path, rgb)
        return save_path
    
    def draw_floor_from_saved_mask(self, image: np.ndarray, mask_path: str, classes: List[str]) -> np.ndarray:
        """
        使用保存的semantic mask绘制地面分割（直接使用原始检测的floor mask）
        
        Args:
            image: 图像 (H, W, 3) BGR格式
            mask_path: semantic mask的numpy文件路径
            classes: 类别列表（用于查找floor索引）
            
        Returns:
            绘制了地面分割的图像
        """
        try:
            if not os.path.exists(mask_path):
                print(f"  ⚠️  Mask file not found: {mask_path}")
                return image
                
            masks = np.load(mask_path)
            floor_idx = None
            for i, cls in enumerate(classes):
                if cls.lower() == 'floor':
                    floor_idx = i
                    break
            
            if floor_idx is None:
                print(f"  ⚠️  'floor' not found in classes: {classes}")
                return image
            
            if floor_idx >= masks.shape[0]:
                print(f"  ⚠️  floor_idx {floor_idx} >= masks.shape[0] {masks.shape[0]}")
                return image
            
            floor_mask = masks[floor_idx]
            
            # 增强可见性：更明显的绿色覆盖
            overlay = image.copy()
            green_color = np.array([0, 255, 0], dtype=np.uint8)  # 纯绿色
            floor_bool = floor_mask > 0.1
            
            # 如果mask有效像素太少，打印警告
            if np.sum(floor_bool) < 100:
                print(f"  ⚠️  Floor mask has too few pixels: {np.sum(floor_bool)}")
                return image
            
            # 绘制半透明绿色覆盖
            overlay[floor_bool] = overlay[floor_bool] * 0.6 + green_color * 0.4
            alpha = 0.7  # 增加透明度，让绿色更明显
            result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            # Floor mask应用成功（静默）
            return result
        except Exception as e:
            # 静默失败处理
            return image
    
    def draw_distance_on_view(self, image: np.ndarray, distance_str: str) -> np.ndarray:
        """
        在视图上绘制距离信息（梯形线条 - 用于thinking模式12个方向view）
        
        Args:
            image: 图像 (H, W, 3) BGR格式
            distance_str: 距离字符串
        """
        h, w = image.shape[:2]
        center_x = w // 2
        bottom_y = h - 5
        side_offset = int(w * 0.25)  # 增大两侧宽度：0.15 → 0.25
        
        if "WARNING" in distance_str or "<0.5" in distance_str:
            color, line_ratio, top_shrink = (180, 105, 255), 0.15, 0.8  # 淡粉红(HotPink)：只延伸一点点，顶部收缩到0.8
        elif ">2.0" in distance_str or "open" in distance_str:
            color, line_ratio, top_shrink = (0, 255, 0), 0.65, 0.3  # 绿色：降到之前黄色位置，顶部收缩到0.3（最窄）
        else:
            color, line_ratio, top_shrink = (0, 255, 255), 0.4, 0.5  # 黄色：再低一点，顶部收缩到0.5（中等）
        
        max_length = bottom_y - h // 2
        end_y = bottom_y - int(max_length * line_ratio)
        
        cv2.line(image, (center_x, bottom_y), (center_x, end_y), color, 3)
        cv2.line(image, (center_x - side_offset, bottom_y), (center_x - int(side_offset * top_shrink), end_y), color, 2)
        cv2.line(image, (center_x + side_offset, bottom_y), (center_x + int(side_offset * top_shrink), end_y), color, 2)
        
        text_x = center_x + 10
        text_y = (bottom_y + h // 2) // 2
        font_scale, thickness = 0.6, 2
        text_size = cv2.getTextSize(distance_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cv2.rectangle(image, (text_x - 2, text_y - text_size[1] - 1),
                     (text_x + text_size[0] + 2, text_y + 2), (0, 0, 0), -1)
        cv2.putText(image, distance_str, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return image
    
    def draw_distance_on_action_view(self, image: np.ndarray, distance_dict: Dict[str, str]) -> np.ndarray:
        """
        在Action模式视图上绘制3个方向的距离信息（Left 30 / Front / Right 30）

        Args:
            image: 图像 (H, W, 3) BGR格式
            distance_dict: 距离字典，key为方向（'front', 'left_30', 'right_30'）
        """
        h, w = image.shape[:2]
        center_x = w // 2
        # 自动检测白底条带：如果图像底部存在白色条带（np.vstack拼接的landmark信息栏），
        # 则只在原始RGB区域内画距离线，避免射线起点落在白色区域内。
        # 检测方法：若最后一行全白（mean>253），向上扫描找到第一个非白行。
        h_rgb = h
        if image[-1].mean() > 253:
            for r in range(h - 1, -1, -1):
                if image[r].mean() < 250:
                    h_rgb = r + 1
                    break
        bottom_y = h_rgb - 10

        # 3个方向：左30, 前, 右30
        direction_configs = [
            {'key': 'left_30', 'angle': -120, 'label': 'Left 30'},
            {'key': 'front', 'angle': -90, 'label': 'FRONT'},
            {'key': 'right_30', 'angle': -60, 'label': 'Right 30'},
        ]
        
        for config in direction_configs:
            dist_str = distance_dict.get(config['key'], 'Unknown')
            if dist_str == 'Unknown':
                continue
            
            # 根据距离确定颜色和长度（FRONT线条更长）
            if "WARNING" in dist_str or "<0.5" in dist_str:
                color = (180, 105, 255)  # 淡粉红(HotPink)
                line_length = 65 if config['key'] == 'front' else 60
            elif ">2.0" in dist_str or "open" in dist_str:
                color = (0, 255, 0)  # 绿色
                line_length = 140 if config['key'] == 'front' else 120
            else:
                color = (0, 255, 255)  # 黄色
                line_length = 105 if config['key'] == 'front' else 90
            
            # 计算终点
            angle_rad = np.deg2rad(config['angle'])
            end_x = int(center_x + line_length * np.cos(angle_rad))
            end_y = int(bottom_y + line_length * np.sin(angle_rad))
            
            # 绘制线条（中心线粗一点）
            thickness = 3 if config['key'] == 'front' else 2
            cv2.line(image, (center_x, bottom_y), (end_x, end_y), color, thickness)
            
            # FRONT用大字号，其他用稍大字号
            font_scale = 0.72 if config['key'] == 'front' else 0.62
            font_thickness = 2 if config['key'] == 'front' else 2
            
            # 合并标签为单行："Left 90 1.3m" 或 "FRONT 0.70m"
            combined_label = f"{config['label']} {dist_str}"
            label_size = cv2.getTextSize(combined_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # 标签位置：从线条终点沿方向延伸
            label_offset = 25
            base_x = int(end_x + label_offset * np.cos(angle_rad))
            base_y = int(end_y + label_offset * np.sin(angle_rad))
            
            # 根据方向调整标签位置，使其向两侧延伸，远离中心
            if config['key'] == 'front':
                # FRONT标签居中
                text_x = base_x - label_size[0] // 2
                text_y = base_y + label_size[1] // 2
            elif config['key'] == 'left_30':
                # 左侧标签：向左延伸，右对齐（文字在线条左侧）
                side_offset = 15
                text_x = base_x - label_size[0] - side_offset
                text_y = base_y + label_size[1] // 2
            else:  # right_30
                # 右侧标签：向右延伸，左对齐（文字在线条右侧）
                side_offset = 15
                text_x = base_x + side_offset
                text_y = base_y + label_size[1] // 2
            
            # 绘制黑色背景和文字
            cv2.rectangle(image, (text_x - 2, text_y - label_size[1] - 2),
                         (text_x + label_size[0] + 2, text_y + 2), (0, 0, 0), -1)
            cv2.putText(image, combined_label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        
        return image
    
    def prepare_action_image_with_enhancements(self, image_path: str, mask_path: str = None, 
                                               distance_dict: Dict[str, str] = None, classes: List[str] = None,
                                               use_floor: bool = True, use_distance: bool = True) -> str:
        """
        为action模式准备增强图像：添加地面分割（绿色）和3方向距离辅助线
        
        Args:
            image_path: 原始图像路径
            mask_path: semantic mask路径
            distance_dict: 距离字典 {'front': 'X.XXm', 'left_30': 'X.XXm', ...}
            classes: 类别列表
            use_floor: 是否绘制地面分割
            use_distance: 是否绘制距离辅助线
            
        Returns:
            增强后的图像路径
        """
        if not os.path.exists(image_path):
            return image_path
        
        image = cv2.imread(image_path)
        if image is None:
            return image_path
        
        if use_floor and mask_path and os.path.exists(mask_path) and classes:
            image = self.draw_floor_from_saved_mask(image, mask_path, classes)
        
        if use_distance and distance_dict:
            image = self.draw_distance_on_action_view(image, distance_dict)
        
        base_path = os.path.splitext(image_path)[0]
        enhanced_path = f"{base_path}_enhanced.png"
        cv2.imwrite(enhanced_path, image)
        return enhanced_path
    
    def save_global_map(self, 
                       step: int,
                       episode_id: int,
                       global_map: np.ndarray,
                       phase: str = "action") -> str:
        """
        保存全局地图（添加标签）
        
        Args:
            step: 步数
            episode_id: episode ID
            global_map: 旋转后的全局地图 (480×480)
            phase: 阶段标识 ("initial", "action1a", "verify1a" 等)
        
        Returns:
            save_path: 保存路径
        """
        if global_map is None:
            return None
        
        # 添加Global Map标签（不显示IMAGE编号）
        label_text = "Global Map"
        
        # 创建白色标签背景（高度40像素）
        label_height = 40
        label_bg = np.ones((label_height, global_map.shape[1], 3), dtype=np.uint8) * 255
        
        # 绘制红色文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0  # 增大字体
        font_thickness = 2  # 保持加粗
        text_color = (0, 0, 255)  # BGR: 红色
        
        # 计算文字位置（居中）
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        text_x = (label_bg.shape[1] - text_size[0]) // 2
        text_y = (label_height + text_size[1]) // 2
        
        # 在标签背景上绘制文字
        cv2.putText(label_bg, label_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
        
        # 垂直拼接：地图在上，标签在下
        labeled_map = np.vstack([global_map, label_bg])
        
        # 保存带标签的地图（无压缩，保持高清）
        episode_dir = self._create_episode_directories(episode_id)
        save_path = os.path.join(episode_dir, 'global_map', f'step_{step:04d}_{phase}.png')
        cv2.imwrite(save_path, labeled_map)
        return save_path
    
    def save_local_map(self,
                      step: int,
                      episode_id: int,
                      local_map: np.ndarray,
                      phase: str = "action") -> str:
        """
        保存局部地图（添加标签）
        
        Args:
            step: 步数
            episode_id: episode ID
            local_map: 局部地图 (400×400)
            phase: 阶段标识 ("initial", "action1a", "verify1a" 等)
        
        Returns:
            save_path: 保存路径
        """
        if local_map is None:
            return None
        
        # 添加Local Map标签（不显示IMAGE编号）
        label_text = "Local Map"
        
        # 创建白色标签背景（高度40像素）
        label_height = 40
        label_bg = np.ones((label_height, local_map.shape[1], 3), dtype=np.uint8) * 255
        
        # 绘制红色文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # 增大字体
        font_thickness = 2  # 加粗
        text_color = (0, 0, 255)  # BGR: 红色
        
        # 计算文字位置（居中）
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        text_x = (label_bg.shape[1] - text_size[0]) // 2
        text_y = (label_height + text_size[1]) // 2
        
        # 在标签背景上绘制文字
        cv2.putText(label_bg, label_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
        
        # 垂直拼接：地图在上，标签在下
        labeled_map = np.vstack([local_map, label_bg])
        
        # 保存带标签的地图
        episode_dir = self._create_episode_directories(episode_id)
        save_path = os.path.join(episode_dir, 'local_map', f'step_{step:04d}_{phase}.png')
        cv2.imwrite(save_path, labeled_map)
        return save_path
    
    def save_detection(self,
                      step: int,
                      episode_id: int,
                      detection_vis: np.ndarray,
                      phase: str = "action") -> str:
        """
        保存检测可视化
        
        Args:
            step: 步数
            episode_id: episode ID
            detection_vis: 检测可视化图像
            phase: 阶段标识 ("initial", "action1a", "verify1a" 等)
        
        Returns:
            save_path: 保存路径
        """
        if detection_vis is None:
            return None
        
        # 简化路径：data/manual_navigation/episode_X/detection/step_XXXX.png
        episode_dir = self._create_episode_directories(episode_id)
        save_path = os.path.join(episode_dir, 'detection', f'step_{step:04d}_{phase}.png')
        cv2.imwrite(save_path, detection_vis)
        return save_path
    
    # ========== 一键保存方法 ==========
    
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
                               room_area_layer: Optional[np.ndarray] = None,
                               room_area_records: Optional[List[Dict[str, Any]]] = None,
                               masks: Optional[np.ndarray] = None,
                               phase: str = "action",
                               global_trajectory_points: Optional[List[Tuple[int, int]]] = None,
                               controller=None,
                               crop_offset: Optional[Tuple[int, int]] = None) -> Tuple[Dict[str, str], List, Optional[float]]:
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
        depth_for_instances = getattr(controller, 'latest_depth_meters', None) if controller is not None else None
        existing_landmark_instances = list(getattr(controller, 'latest_landmark_instances_world', []) or []) \
            if controller is not None else []
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

        if controller is not None:
            controller.latest_landmark_instances_world = [dict(inst) for inst in landmark_instances_world]
        
        # 1. 保存RGB（传入controller用于绘制距离线）
        paths['rgb'] = self.save_rgb(step, episode_id, rgb, phase, controller)
        
        # 2. 渲染并保存全局地图（使用global_trajectory_points或回退到trajectory_points）
        global_traj_to_use = global_trajectory_points if global_trajectory_points is not None else trajectory_points
        # print(f"[DEBUG] Global map trajectory: {len(global_traj_to_use) if global_traj_to_use else 0} points")
        # print(f"[DEBUG] Local map trajectory: {len(trajectory_points) if trajectory_points else 0} points")
        _, global_map_with_trajectory, landmarks, global_map_clean, last_waypoint_angle = self.render_global_map(
            full_map, global_traj_to_use, detected_classes, floor,
            current_pose, landmark_classes, landmark_instances_world, landmark_config,
            waypoint_positions, waypoint_ids, room_area_layer, room_area_records, crop_offset,
            mapping_classes=mapping_classes
        )
        paths['global_map'] = self.save_global_map(step, episode_id, global_map_with_trajectory, phase)
        
        # 3. 渲染并保存局部地图（使用trajectory_points）
        local_map = self.render_local_map(
            full_map, trajectory_points, detected_classes, current_pose,
            floor, landmark_classes, landmark_instances_world, landmark_config, hfov,
            waypoint_positions, waypoint_ids, room_area_layer, room_area_records, crop_offset,
            mapping_classes=mapping_classes
        )
        paths['local_map'] = self.save_local_map(step, episode_id, local_map, phase)        # 4. 渲染并保存检测结果
        detected_landmarks_step = []
        landmark_dist_map = {}
        landmark_dist_map_multi = {}
        for _, _, cls_name, dist_m, angle_deg in landmarks:
            landmark_dist_map_multi.setdefault(cls_name, []).append((dist_m, angle_deg))
            if cls_name not in landmark_dist_map or dist_m < landmark_dist_map[cls_name][0]:
                landmark_dist_map[cls_name] = (dist_m, angle_deg)

        if controller is not None:
            controller.latest_landmark_dist_map = landmark_dist_map if landmark_dist_map else {}
            controller.latest_landmark_dist_map_multi = landmark_dist_map_multi if landmark_dist_map_multi else {}

        if detections is not None and labels is not None:
            # 先渲染bbox，再叠加当前深度采样的距离线，避免距离线参与实例匹配。
            rgb_for_det = rgb.copy()
            detection_vis, detected_landmarks_step, _visible, landmark_strip = self.render_detection_bbox(
                rgb_for_det, detections, labels,
                landmark_classes, mapping_classes,
                depth_meters=getattr(controller, 'latest_depth_meters', None) if controller is not None else None,
                hfov=hfov,
                landmark_dist_map=landmark_dist_map if landmark_dist_map else None,
                landmark_dist_map_multi=landmark_dist_map_multi if landmark_dist_map_multi else None,
                append_bottom_strip=False,
                controller=controller
            )

            map_obstacle_distances = self.calculate_obstacle_distances_from_full_map(full_map)
            try:
                obstacle_distances = self.calculate_obstacle_distances_from_depth(
                    getattr(controller, 'latest_depth_meters', None) if controller is not None else None,
                    hfov=hfov,
                    fallback_distances=map_obstacle_distances,
                )
            except Exception:
                obstacle_distances = map_obstacle_distances or {
                    'front': '>2.0m open',
                    'left_30': '>2.0m open',
                    'right_30': '>2.0m open',
                }

            detection_vis = self.draw_distance_on_action_view(detection_vis, obstacle_distances)
            if controller is not None:
                controller.latest_obstacle_distances = obstacle_distances

            if landmark_strip is not None:
                detection_vis = np.vstack([detection_vis, landmark_strip])
            paths['detection'] = self.save_detection(step, episode_id, detection_vis, phase)

        else:
            if controller is not None:
                controller.latest_visible_landmark_entries = []

        # 统计：本步 landmark 配置/检测/地图实例数量
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

        # 5. 保存semantic masks（用于action模式的地面分割）
        if masks is not None:
            paths['masks'] = self.save_semantic_masks(step, episode_id, masks, phase)
        
        return paths, detected_landmarks_step, last_waypoint_angle
    
    # ========== 辅助方法 ==========

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

        from vlnce_baselines.config_system.constants import landmark_merge_distance
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


# ========== 便捷函数 ==========

def create_visualizer(results_dir: str, 
                     resolution: int = 5,
                     map_shape: Tuple[int, int] = (480, 480),
                     enable_global_map_crop: bool = False,
                     enable_adaptive_zoom: bool = False) -> MapVisualizer:
    """创建MapVisualizer实例"""
    return MapVisualizer(results_dir, resolution, map_shape, 
                        enable_global_map_crop, enable_adaptive_zoom)
