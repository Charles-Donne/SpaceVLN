"""Core project configuration helpers and static parameter exports."""

from vlnce_baselines.config.core.constants import *
from vlnce_baselines.config.core.setup import ConfigHelper
from vlnce_baselines.config.core.categories import CategoryConfig, create_category_config

__all__ = [
    'color_palette',
    'legend_color_palette',
    'detection_colors',
    'navigable_classes',
    'map_channels',
    'landmark_min_area_threshold',
    'landmark_min_total_pixels',
    'detection_thickness',
    'landmark_marker_color',
    'landmark_marker_border',
    'landmark_marker_radius',
    'ConfigHelper',
    'CategoryConfig',
    'create_category_config',
]
