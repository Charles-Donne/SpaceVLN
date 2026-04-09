"""Canonical project-level constant exports.

Runtime-switchable experiment config stays in `navigation_system/config/runtime/default.py`.
Project-level static parameters are grouped under `navigation_system/config/core/params/`.
This module re-exports the legacy lowercase names used throughout the codebase.
"""

from navigation_system.config.core.params.landmarks import (
    DETECTION_VISIBLE_TOPK,
    LANDMARK_DUPLICATE_ANGLE_DIFF_DEG,
    LANDMARK_DUPLICATE_IOU_LOOSE,
    LANDMARK_DUPLICATE_IOU_STRICT,
    LANDMARK_DUPLICATE_REL_DIST_M,
    LANDMARK_EDGE_DEPTH_KEYWORDS,
    LANDMARK_EDGE_DEPTH_MIN_GAP_M,
    LANDMARK_INSTANCE_MERGE_RADIUS_M,
    LANDMARK_INSTANCE_TOPK,
    LANDMARK_MERGE_DISTANCE,
    LANDMARK_MIN_AREA_THRESHOLD,
    LANDMARK_MIN_TOTAL_PIXELS,
    LANDMARK_STRIP_TOPK,
    LOCAL_MAP_LANDMARK_TOPK,
)
from navigation_system.config.core.params.rendering import (
    COLOR_PALETTE,
    DETECTION_COLORS,
    DETECTION_THICKNESS,
    LANDMARK_MARKER_BORDER,
    LANDMARK_MARKER_COLOR,
    LANDMARK_MARKER_RADIUS,
    LEGEND_COLOR_PALETTE,
)
from navigation_system.config.core.params.semantic import (
    LANDMARK_CLASSES as CFG_LANDMARK_CLASSES,
    MAP_CHANNELS as CFG_MAP_CHANNELS,
    MAPPING_CLASSES as CFG_MAPPING_CLASSES,
    NAVIGABLE_CLASSES as CFG_NAVIGABLE_CLASSES,
    NUM_SEMANTIC_CATEGORIES as CFG_NUM_SEMANTIC_CATEGORIES,
)

# Legacy lowercase exports.
color_palette = COLOR_PALETTE
legend_color_palette = LEGEND_COLOR_PALETTE
map_channels = CFG_MAP_CHANNELS
mapping_classes = CFG_MAPPING_CLASSES
NUM_SEMANTIC_CATEGORIES = CFG_NUM_SEMANTIC_CATEGORIES
navigable_classes = CFG_NAVIGABLE_CLASSES
landmark_classes = CFG_LANDMARK_CLASSES
landmark_marker_color = LANDMARK_MARKER_COLOR
landmark_marker_border = LANDMARK_MARKER_BORDER
landmark_marker_radius = LANDMARK_MARKER_RADIUS
local_map_landmark_topk = LOCAL_MAP_LANDMARK_TOPK
landmark_min_total_pixels = LANDMARK_MIN_TOTAL_PIXELS
landmark_min_area_threshold = LANDMARK_MIN_AREA_THRESHOLD
landmark_merge_distance = LANDMARK_MERGE_DISTANCE
landmark_instance_topk = LANDMARK_INSTANCE_TOPK
landmark_instance_merge_radius_m = LANDMARK_INSTANCE_MERGE_RADIUS_M
detection_colors = DETECTION_COLORS
detection_thickness = DETECTION_THICKNESS
detection_visible_topk = DETECTION_VISIBLE_TOPK
landmark_strip_topk = LANDMARK_STRIP_TOPK
landmark_duplicate_iou_strict = LANDMARK_DUPLICATE_IOU_STRICT
landmark_duplicate_iou_loose = LANDMARK_DUPLICATE_IOU_LOOSE
landmark_duplicate_rel_dist_m = LANDMARK_DUPLICATE_REL_DIST_M
landmark_duplicate_angle_diff_deg = LANDMARK_DUPLICATE_ANGLE_DIFF_DEG
landmark_edge_depth_keywords = LANDMARK_EDGE_DEPTH_KEYWORDS
landmark_edge_depth_min_gap_m = LANDMARK_EDGE_DEPTH_MIN_GAP_M
