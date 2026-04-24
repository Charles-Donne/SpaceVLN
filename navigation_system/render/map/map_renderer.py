"""Auto-extracted helper module from MapVisualizer for clearer separation of concerns."""
import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple

from . import rendering as vu
from navigation_system.config.core.constants import (
    landmark_marker_border,
    landmark_marker_color,
    local_map_landmark_topk,
)

GLOBAL_MAP_BASE_SIZE = 480
GLOBAL_MAP_OUTPUT_SIZE = 440


def _compute_global_map_sizes(output_scale: int = 1) -> Tuple[int, int]:
    scale = max(1, int(output_scale))
    return GLOBAL_MAP_BASE_SIZE * scale, GLOBAL_MAP_OUTPUT_SIZE * scale


def _contour_bounding_box(points: np.ndarray, image_shape: Tuple[int, int], padding: int = 0) -> Tuple[int, int, int, int]:
    contour = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(contour)
    x1 = max(0, int(x) - int(padding))
    y1 = max(0, int(y) - int(padding))
    x2 = min(int(image_shape[1]), int(x + w) + int(padding))
    y2 = min(int(image_shape[0]), int(y + h) + int(padding))
    return x1, y1, x2, y2


def _crop_square_image(
    image: Optional[np.ndarray],
    crop_box: Optional[Tuple[int, int, int, int]],
) -> Optional[np.ndarray]:
    if image is None:
        return None
    if crop_box is None:
        return image.copy()

    x1, y1, x2, y2 = [int(value) for value in crop_box]
    height, width = image.shape[:2]
    x1 = max(0, min(width, x1))
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(y1 + 1, min(height, y2))
    return image[y1:y2, x1:x2].copy()


def _apply_display_ops_to_image(
    image: Optional[np.ndarray],
    display_ops: List[Dict[str, Any]],
    interpolation: int = cv2.INTER_NEAREST,
) -> Optional[np.ndarray]:
    if image is None:
        return None

    output = image.copy()
    for op in display_ops:
        kind = str(op.get("kind") or "").strip().lower()
        if kind == "crop":
            output = _crop_square_image(output, op.get("box"))
        elif kind == "resize":
            output_size = int(op.get("output_size") or 0)
            if output_size > 0 and output.shape[:2] != (output_size, output_size):
                output = cv2.resize(output, (output_size, output_size), interpolation=interpolation)
    return output


def _apply_display_ops_to_point(
    point: Tuple[float, float],
    display_ops: List[Dict[str, Any]],
) -> Optional[Tuple[int, int]]:
    x = float(point[0])
    y = float(point[1])
    current_size = None

    for op in display_ops:
        kind = str(op.get("kind") or "").strip().lower()
        if kind == "crop":
            x1, y1, x2, y2 = [float(value) for value in op.get("box", (0, 0, 0, 0))]
            if not (x1 <= x < x2 and y1 <= y < y2):
                return None
            x -= x1
            y -= y1
            current_size = int(round(min(x2 - x1, y2 - y1)))
        elif kind == "resize":
            input_size = float(op.get("input_size") or current_size or 0.0)
            output_size = float(op.get("output_size") or 0.0)
            if input_size <= 0.0 or output_size <= 0.0:
                return None
            x = x * output_size / input_size
            y = y * output_size / input_size
            current_size = int(round(output_size))

    if current_size is not None and current_size > 0:
        x = min(max(x, 0.0), float(current_size - 1))
        y = min(max(y, 0.0), float(current_size - 1))
    return int(round(x)), int(round(y))


def _apply_display_ops_to_points(
    points: List[Tuple[int, int]],
    display_ops: List[Dict[str, Any]],
) -> List[Tuple[int, int]]:
    transformed_points: List[Tuple[int, int]] = []
    for point in list(points or []):
        transformed = _apply_display_ops_to_point(point, display_ops)
        if transformed is not None:
            transformed_points.append(transformed)
    return transformed_points


def render_global_map(owner,
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
                     space_area_layer: Optional[np.ndarray] = None,
                     space_area_records: Optional[List[Dict[str, Any]]] = None,
                     crop_offset: Optional[Tuple[int, int]] = None,
                     mapping_classes: Optional[List[str]] = None,
                     output_scale: int = 1) -> Tuple[np.ndarray, np.ndarray, List, np.ndarray, Optional[float]]:
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
    obstacle_mask = owner._get_channel_mask(full_map, 0)   # channel 0: obstacle
    explored_mask = owner._get_channel_mask(full_map, 1)   # channel 1: explored

    # ===== 阶段1.1: 创建语义地图 =====
    semantic_map = np.zeros((h, w), dtype=np.uint8)

    # Layer 1: 已探索自由空间（浅灰色）
    explored_free_mask = np.logical_and(explored_mask, ~obstacle_mask)
    explored_free_mask = owner._refine_render_region_mask(
        explored_free_mask,
        close_kernel_size=5,
        min_component_area=18,
        hole_fill_area=28,
    )
    semantic_map[explored_free_mask] = 2

    # Layer 2: Floor（渲染时做额外平滑，避免毛刺和小孔）
    # 使用 mapper 预计算的 floor（由 explored/obstacle 直接得到，避免额外语义扫描）
    if floor is not None:
        floor_display_mask = np.logical_and(floor.astype(bool), explored_mask)
        floor_display_mask = owner._refine_render_region_mask(
            floor_display_mask,
            close_kernel_size=7,
            min_component_area=42,
            hole_fill_area=84,
        )
        semantic_map[floor_display_mask] = 5  # 浅绿色

    # 轨迹与 waypoint 都在后续用 OpenCV 叠加；这里不再读取 Channel 2 的旧残留逻辑

    # ===== 阶段2: PIL调色板渲染 =====
    # 现在semantic_map包含：0=未知, 1=障碍物, 2=已探索, 4=waypoint, 5=floor（轨迹稍后用OpenCV绘制）
    sem_map_vis = Image.new("P", (w, h))
    sem_map_vis.putpalette(owner.color_palette)
    sem_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")

    # 坐标系变换：翻转Y轴 + RGB→BGR
    sem_map_vis = np.flipud(sem_map_vis)
    sem_map_vis = np.array(sem_map_vis)
    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]  # RGB → BGR
    global_map_base_size, global_map_output_size = _compute_global_map_sizes(output_scale)
    sem_map_vis = cv2.resize(
        sem_map_vis,
        (global_map_base_size, global_map_base_size),
        interpolation=cv2.INTER_NEAREST,
    )
    global_label_display_layer = owner._prepare_space_area_display_layer(
        space_area_layer,
        output_size=global_map_base_size,
    )
    owner._draw_space_areas_in_place(
        sem_map_vis,
        space_area_layer,
        space_area_records,
        alpha=1.0,
        fill_regions=True,
        show_labels=False,
    )
    obstacle_mask_display = owner._build_display_obstacle_mask(full_map)
    if obstacle_mask_display.shape[:2] != sem_map_vis.shape[:2]:
        obstacle_mask_display = cv2.resize(
            obstacle_mask_display.astype(np.uint8),
            (global_map_base_size, global_map_base_size),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    sem_map_vis[obstacle_mask_display] = owner.OBSTACLE_COLOR

    # ===== 阶段3: 提取Landmark位置（但不绘制）=====
    landmarks = []
    if landmark_instances:
        landmarks = owner._build_landmarks_from_instances(
            landmark_instances, full_map, current_pose, crop_offset
        )
    elif landmark_classes and landmark_config:
        landmarks = owner._extract_landmarks(
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

    projector = owner._build_map_projector(full_map, current_pose, crop_offset)
    display_ops: List[Dict[str, Any]] = []
    preview_image = sem_map_vis.copy()

    if owner.enable_global_map_crop and preview_image.shape[0] > global_map_output_size:
        crop_margin = max(0, (preview_image.shape[0] - global_map_output_size) // 2)
        center_crop_box = (
            crop_margin,
            crop_margin,
            preview_image.shape[1] - crop_margin,
            preview_image.shape[0] - crop_margin,
        )
        display_ops.append({"kind": "crop", "box": center_crop_box})
        preview_image = _crop_square_image(preview_image, center_crop_box)

    if owner.enable_adaptive_zoom:
        zoom_crop_box = owner._compute_adaptive_zoom_box([preview_image])
        if zoom_crop_box is not None:
            display_ops.append({"kind": "crop", "box": zoom_crop_box})
            preview_image = _crop_square_image(preview_image, zoom_crop_box)

    if preview_image.shape[0] != global_map_output_size:
        display_ops.append(
            {
                "kind": "resize",
                "input_size": int(preview_image.shape[0]),
                "output_size": global_map_output_size,
            }
        )

    transformed_label_display_layer = _apply_display_ops_to_image(
        global_label_display_layer,
        display_ops,
        interpolation=cv2.INTER_NEAREST,
    )
    global_map_rotated = _apply_display_ops_to_image(
        sem_map_vis,
        display_ops,
        interpolation=cv2.INTER_NEAREST,
    )
    global_map_with_trajectory = global_map_rotated.copy()
    last_waypoint_angle = None

    if current_pose is not None:
        # ===== 阶段5: 地图底层(area/floor/obstacle)已先完成裁剪/缩放；
        # 再后渲染轨迹、标签、箭头，确保它们都在 obstacle 之上 =====
        if projector is not None and trajectory_points is not None and len(trajectory_points) > 1:
            trajectory_color = owner.GLOBAL_TRAJECTORY_COLOR
            display_points = projector.world_points_to_global_display(trajectory_points)
            display_points = _apply_display_ops_to_points(display_points, display_ops)
            if len(display_points) > 1:
                trajectory_thickness = max(
                    3,
                    int(round(min(global_map_with_trajectory.shape[:2]) * 0.009)),
                )
                cv2.polylines(
                    global_map_with_trajectory,
                    [np.array(display_points, dtype=np.int32)],
                    isClosed=False,
                    color=trajectory_color,
                    thickness=trajectory_thickness,
                    lineType=cv2.LINE_AA,
                )

        owner._draw_space_areas_in_place(
            global_map_rotated,
            space_area_layer,
            space_area_records,
            fill_regions=False,
            show_labels=True,
            use_display_label=True,
            display_layer_override=transformed_label_display_layer,
            reserved_boxes=[],
        )

        # Anchor the agent arrow to the pre-crop agent center, then project it
        # through crop/resize ops so adaptive zoom does not force it to the
        # visual center of the final 440x440 map.
        agent_display_center = _apply_display_ops_to_point(
            (global_map_base_size / 2.0, global_map_base_size / 2.0),
            display_ops,
        )
        if agent_display_center is None:
            center_x = global_map_rotated.shape[1] // 2
            center_y = global_map_rotated.shape[0] // 2
        else:
            center_x, center_y = agent_display_center
        arrow_angle = np.deg2rad(-90)
        agent_pos = (center_x, center_y, arrow_angle)
        arrow_size = max(20, int(round(min(global_map_rotated.shape[:2]) * 0.05)))
        agent_arrow = vu.get_contour_points(agent_pos, origin=(0, 0), size=arrow_size)
        agent_arrow_box = _contour_bounding_box(
            agent_arrow,
            global_map_rotated.shape[:2],
            padding=max(6, int(round(min(global_map_rotated.shape[:2]) * 0.01))),
        )
        owner._draw_space_areas_in_place(
            global_map_with_trajectory,
            space_area_layer,
            space_area_records,
            fill_regions=False,
            show_labels=True,
            use_display_label=True,
            display_layer_override=transformed_label_display_layer,
            reserved_boxes=[agent_arrow_box],
        )
        cv2.drawContours(global_map_rotated, [agent_arrow], 0, (255, 255, 255), 1)
        cv2.drawContours(global_map_with_trajectory, [agent_arrow], 0, (255, 255, 255), 1)
        cv2.drawContours(global_map_rotated, [agent_arrow], 0, owner.AGENT_ARROW_COLOR, -1)
        cv2.drawContours(global_map_with_trajectory, [agent_arrow], 0, owner.AGENT_ARROW_COLOR, -1)

    # 添加方位标签到global map
    global_map_with_trajectory = owner.add_orientation_labels(
        global_map_with_trajectory,
        text_thickness=1,
    )
    global_map_rotated = owner.add_orientation_labels(
        global_map_rotated,
        text_thickness=1,
    )

    # 返回：基础地图 + 显示副本（带轨迹和landmark+waypoint） + 无轨迹的旋转地图（供local_map裁剪） + 距离信息 + 最后waypoint角度
    return sem_map_vis, global_map_with_trajectory, landmarks, global_map_rotated, last_waypoint_angle

def render_local_map(owner, 
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
                    space_area_layer: Optional[np.ndarray] = None,
                    space_area_records: Optional[List[Dict[str, Any]]] = None,
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
    obstacle_mask = owner._get_channel_mask(full_map, 0)   # channel 0: obstacle
    explored_mask = owner._get_channel_mask(full_map, 1)   # channel 1: explored

    # 创建语义地图
    semantic_map = np.zeros((h, w), dtype=np.uint8)

    # Layer 1: 已探索自由空间（浅灰色）
    explored_free_mask = np.logical_and(explored_mask, ~obstacle_mask)
    explored_free_mask = owner._refine_render_region_mask(
        explored_free_mask,
        close_kernel_size=5,
        min_component_area=18,
        hole_fill_area=28,
    )
    semantic_map[explored_free_mask] = 2

    # Layer 2: Floor（渲染时做额外平滑，避免毛刺和小孔）
    # 使用 mapper 预计算的 floor（与 render_global_map 逻辑一致）
    if floor is not None:
        floor_display_mask = np.logical_and(floor.astype(bool), explored_mask)
        floor_display_mask = owner._refine_render_region_mask(
            floor_display_mask,
            close_kernel_size=7,
            min_component_area=42,
            hole_fill_area=84,
        )
        semantic_map[floor_display_mask] = 5

    # Layer 3: 不渲染轨迹和waypoint（后续用OpenCV绘制轨迹）
    # Local map不显示历史waypoint，只显示轨迹

    # ===== 阶段2: PIL调色板渲染 =====
    sem_map_vis = Image.new("P", (w, h))
    sem_map_vis.putpalette(owner.color_palette)
    sem_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")

    # 坐标系变换
    sem_map_vis = np.flipud(sem_map_vis)
    sem_map_vis = np.array(sem_map_vis)
    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]  # RGB → BGR
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST)
    owner._draw_space_areas_in_place(
        sem_map_vis,
        space_area_layer,
        space_area_records,
        alpha=1.0,
        show_labels=False,
    )
    obstacle_mask_resized = owner._build_display_obstacle_mask(full_map)
    sem_map_vis[obstacle_mask_resized] = owner.OBSTACLE_COLOR

    # ===== 阶段3: 准备显示（地图已在提取时旋转）=====
    projector = owner._build_map_projector(full_map, current_pose, crop_offset)
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

    # ===== 阶段6.4: 先把 obstacle 重新压回 FOV 之上，再画轨迹/landmark/箭头 =====
    local_map[obstacle_local] = owner.OBSTACLE_COLOR

    # ===== 阶段6.5: 绘制轨迹线（在 obstacle 之后，确保轨迹位于其上）=====
    if len(trajectory_display_points) > 1:
        trajectory_color = owner.LOCAL_TRAJECTORY_COLOR
        valid_points = [
            (int(point[0]), int(point[1]))
            for point in trajectory_display_points
            if 0 <= int(point[0]) < 480 and 0 <= int(point[1]) < 480
        ]
        if len(valid_points) > 1:
            cv2.polylines(
                local_map,
                [np.array(valid_points, dtype=np.int32)],
                isClosed=False,
                color=trajectory_color,
                thickness=4,
                lineType=cv2.LINE_AA,
            )

    # ===== 绘制0.5m半径圆圈（深绿色，标识当前位置附近区域）=====
    # 480像素 = 12m，所以1m = 40像素，0.5m = 20像素
    nearby_radius = 20  # 0.5m半径
    nearby_color = (0, 100, 0)  # 深绿色BGR
    nearby_thickness = 2  # 2像素线宽
    cv2.circle(local_map, (fov_center_x, fov_center_y), nearby_radius, nearby_color, nearby_thickness)

    # ===== 阶段8: 绘制Landmark标记 =====
    landmarks = []
    if landmark_instances:
        landmarks = owner._build_local_landmarks_from_instances(
            landmark_instances, full_map, current_pose, crop_offset,
            topk=local_map_landmark_topk,
        )
    elif landmark_classes and landmark_config:
        # full_map 已由 get_full_map_for_rendering(rotate_to_agent_heading=True) 旋转过
        # _extract_landmarks 返回的 (marker_x, marker_y) 已经是旋转后地图的像素坐标
        # 与 render_global_map 的处理完全一致：scale + flipud + 裁剪中心区域
        # 不需要再做额外旋转（否则会双重旋转导致位置偏移）
        landmarks = owner._extract_landmarks(
            full_map, detected_classes, landmark_classes,
            landmark_config['min_total_pixels'],
            landmark_config['min_area_threshold'],
            mapping_classes=mapping_classes
        )

    for landmark in landmarks:
        raw_display_id = None
        if isinstance(landmark, dict):
            marker_x = float(landmark.get("marker_x", 0.0))
            marker_y = float(landmark.get("marker_y", 0.0))
            raw_display_id = landmark.get("display_id")
        elif isinstance(landmark, (tuple, list)) and len(landmark) >= 2:
            try:
                marker_x = float(landmark[0])
                marker_y = float(landmark[1])
            except (TypeError, ValueError):
                continue
        else:
            continue

        local_display = None
        if projector is not None:
            local_display = projector.rotated_to_local_display(marker_y, marker_x)
        if local_display is not None:
            local_x, local_y = local_display
            local_landmark_radius = 10
            center = (int(local_x), int(local_y))
            cv2.circle(local_map, center, local_landmark_radius, landmark_marker_color, -1)
            cv2.circle(local_map, center, local_landmark_radius, landmark_marker_border, 1)

            try:
                display_id = int(raw_display_id) if raw_display_id is not None else None
            except (TypeError, ValueError):
                display_id = None
            if display_id is not None and display_id > 0:
                label_text = f"#{display_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.42
                font_thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                pad = 3
                tag_x1 = max(0, min(local_map.shape[1] - (text_w + pad * 2), center[0] + 10))
                tag_y1 = max(0, min(local_map.shape[0] - (text_h + baseline + pad * 2), center[1] - text_h - 12))
                tag_x2 = tag_x1 + text_w + pad * 2
                tag_y2 = tag_y1 + text_h + baseline + pad * 2
                cv2.rectangle(local_map, (tag_x1, tag_y1), (tag_x2, tag_y2), (255, 255, 255), -1)
                cv2.rectangle(local_map, (tag_x1, tag_y1), (tag_x2, tag_y2), landmark_marker_border, 1)
                cv2.putText(
                    local_map,
                    label_text,
                    (tag_x1 + pad, tag_y2 - baseline - pad),
                    font,
                    font_scale,
                    landmark_marker_border,
                    font_thickness,
                    cv2.LINE_AA,
                )

    # ===== 阶段9: 绘制朝上的箭头（最上层）=====
    arrow_color = owner.AGENT_ARROW_COLOR
    arrow_angle = np.deg2rad(-90)
    agent_pos = (fov_center_x, fov_center_y, arrow_angle)
    agent_arrow = vu.get_contour_points(agent_pos, origin=(0, 0), size=26)
    cv2.drawContours(local_map, [agent_arrow], 0, arrow_color, -1)

    # ===== 阶段10: 最终裁剪到440×440（中心区域）=====
    # 从480x480裁剪中心440x440区域
    crop_offset = (480 - 440) // 2  # = 20
    local_map_cropped = local_map[crop_offset:crop_offset+440, crop_offset:crop_offset+440].copy()

    # 添加方位标签
    local_map_cropped = owner.add_orientation_labels(local_map_cropped)

    return local_map_cropped

def add_orientation_labels(
    owner,
    map_image: np.ndarray,
    *,
    text_thickness: int = 2,
) -> np.ndarray:
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
    font_scale = 0.58
    text_color = (0, 0, 139)  # 深红色BGR
    bg_color = (255, 255, 255)  # 白色背景

    # 定义方位标签
    labels = {
        'FRONT': (w // 2, 20),  # 上方
        'BACK': (w // 2, h - 8),  # 下方
        'LEFT': (32, h // 2),  # 左侧，进一步向外挪动
        'RIGHT': (w - 32, h // 2)  # 右侧，进一步向外挪动
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
        rect_x1 = text_x - padding_side
        rect_y1 = text_y - text_height - padding_top
        rect_x2 = text_x + text_width + padding_side
        rect_y2 = text_y + baseline + padding_bottom
        cv2.rectangle(
            labeled_map,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            bg_color,
            -1,
        )
        cv2.rectangle(
            labeled_map,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            text_color,
            1,
        )

        # 绘制深红色文字
        cv2.putText(labeled_map, text, (text_x, text_y),
                   font, font_scale, text_color, text_thickness, cv2.LINE_AA)

    return labeled_map

def save_global_map(owner,
                   step: int,
                   episode_id: int,
                   global_map: np.ndarray,
                   phase: str = "action") -> str:
    """
    保存全局地图原图（无标题叠加，PNG无压缩）

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

    save_path = owner.get_map_artifact_path(
        step=step,
        episode_id=episode_id,
        phase=phase,
        map_kind="global",
    )
    cv2.imwrite(save_path, global_map, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return save_path

def save_local_map(owner,
                  step: int,
                  episode_id: int,
                  local_map: np.ndarray,
                  phase: str = "action",
                  debug_lines: Optional[List[str]] = None) -> str:
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

    debug_lines = [str(line).strip() for line in list(debug_lines or []) if str(line or "").strip()]

    # 创建白色标签背景（标题 + 可选debug行）
    label_height = 40 + (22 * len(debug_lines))
    label_bg = np.ones((label_height, local_map.shape[1], 3), dtype=np.uint8) * 255

    # 绘制红色文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7  # 增大字体
    font_thickness = 2  # 加粗
    text_color = (0, 0, 255)  # BGR: 红色

    # 计算文字位置（居中）
    text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
    text_x = (label_bg.shape[1] - text_size[0]) // 2
    text_y = 26

    # 在标签背景上绘制标题
    cv2.putText(label_bg, label_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    if debug_lines:
        debug_font = cv2.FONT_HERSHEY_SIMPLEX
        debug_scale = 0.46
        debug_thickness = 1
        debug_color = (40, 40, 40)
        base_y = 40
        for idx, line in enumerate(debug_lines):
            line_y = base_y + (idx + 1) * 20
            cv2.putText(
                label_bg,
                line,
                (10, line_y),
                debug_font,
                debug_scale,
                debug_color,
                debug_thickness,
                cv2.LINE_AA,
            )

    # 垂直拼接：地图在上，标签在下
    labeled_map = np.vstack([local_map, label_bg])

    # 保存带标签的地图
    save_path = owner.get_map_artifact_path(
        step=step,
        episode_id=episode_id,
        phase=phase,
        map_kind="local",
    )
    cv2.imwrite(save_path, labeled_map)
    return save_path
