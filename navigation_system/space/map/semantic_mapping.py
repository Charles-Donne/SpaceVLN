r""" 
We followed the process of constructing semantic map provided by chaplot.
However, their work doesn't support to build open-vocabulary semantic map.
We improved this by using dynamic feature map.

REFERENCE:
https://github.com/devendrachaplot/Object-Goal-Navigation/tree/master
"""

import os
import cv2
import copy
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from navigation_system.config.core.constants import *
from navigation_system.config.core.params.thresholds import SEM_MAP_LANDMARK_THRESH
from navigation_system.space.map.map_utils import *
import navigation_system.space.geometry.depth_utils as du
import navigation_system.render.map.rendering as vu

# OrderedSet implementation (previously in data_utils.py)
class OrderedSet:
    """有序集合 - 保持插入顺序的set"""
    def __init__(self, iterable=None):
        self._dict = {}
        if iterable:
            for item in iterable:
                self.add(item)
    
    def add(self, item):
        self._dict[item] = None
    
    def __contains__(self, item):
        return item in self._dict
    
    def __iter__(self):
        return iter(self._dict.keys())
    
    def __len__(self):
        return len(self._dict)
    
    def __repr__(self):
        return f"OrderedSet({list(self._dict.keys())})"


class Semantic_Mapping(nn.Module):
    r"""
    分块可扩展语义地图 (Tiled Semantic Mapping)
    
    核心改进：
    1. 支持负世界坐标 - agent可以向任意方向移动
    2. 分块存储 - 每个块240×240像素(12m×12m)
    3. 按需扩展 - 动态创建新块，理论上无限大小
    4. 内存高效 - 只存储探索过的区域
    
    坐标系统详解：
    ============
    
    1. 世界坐标（米）：
       - agent_x_m, agent_y_m: agent的世界坐标，单位：米
       - 初始位置：(6.0m, 6.0m) 位于Tile(0,0)中心
       - 可以是负值，如(-3.5, 10.2)
    
    2. 世界像素坐标：
       - px, gx: Y轴像素（行，高度）= agent_y_m * 20
       - py, gy: X轴像素（列，宽度）= agent_x_m * 20
       - 注意：遗留命名中gx对应Y轴，gy对应X轴！
    
    3. Tile索引（块索引）：
       - tile_x: X轴方向的块索引 = floor(agent_x_m / 12.0)
       - tile_y: Y轴方向的块索引 = floor(agent_y_m / 12.0)
       - 可以是负值，如(-1, 2)表示X轴左边1块，Y轴上边2块
       - Tile(0,0)覆盖：X∈[0,12)m, Y∈[0,12)m
       - Tile(-1,1)覆盖：X∈[-12,0)m, Y∈[12,24)m
    
    4. Tensor维度：
       - [batch, channels, H, W] = [bs, C, height, width]
       - H（高度）= rows = Y轴方向的像素数
       - W（宽度）= cols = X轴方向的像素数
       - 索引：tensor[b, c, py, px] 其中 py=X像素, px=Y像素
    
    5. 命名规范（新代码请遵循）：
       - agent_x_m, agent_y_m: 世界坐标（米）
       - agent_px: Y轴像素 = agent_y_m * 20
       - agent_py: X轴像素 = agent_x_m * 20
       - tile_x, tile_y: Tile索引
       - local_h, local_w: tensor内部索引（h=height=Y, w=width=X）
    
    6. 遗留变量（兼容旧代码）：
       - gx, gx1, gx2: Y轴像素
       - gy, gy1, gy2: X轴像素
       - lmb: [gx1, gx2, gy1, gy2] Local Map边界
    
    Map结构（优化版 - 节省通道）：
    1. Obstacle Map (通道0)
    2. Explored Area (通道1)
    3. Agent通道 (通道2) - 合并轨迹/当前/waypoint
       - 0.0 = 无标记
       - 0.5 = 历史轨迹线
       - 0.7 = 当前位置
       - 1.0, 2.0, 3.0... = Waypoint ID（整数）
    4. Semantic Categories (通道3+，动态扩展)
    """
    MAP_CHANNELS = 3  # 基础通道数：0-2 (优化后)
    TILE_SIZE = 240  # 每个块的尺寸（像素）
    TILE_SIZE_M = 12.0  # 每个块的物理尺寸（米）

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()
        self.args = args
        self.dropout = 0.5
        self.n_channels = 3
        self.goal = None
        self.curr_loc = None
        self.last_loc = None
        self.vis_classes = []
        
        # 地图参数
        self.fov = args.HFOV
        self.min_z = args.MIN_Z
        self.device = args.DEVICE
        self.du_scale = args.DU_SCALE
        self.visualize = args.VISUALIZE
        self.screen_w = args.FRAME_WIDTH
        self.screen_h = args.FRAME_HEIGHT
        self.vision_range = args.VISION_RANGE  # 100cm
        self.resolution = args.MAP_RESOLUTION  # 5cm/pixel
        self.print_images = args.PRINT_IMAGES
        self.z_resolution = args.MAP_RESOLUTION
        self.num_environments = args.NUM_ENVIRONMENTS
        self.global_downscaling = args.GLOBAL_DOWNSCALING
        self.cat_pred_threshold = args.CAT_PRED_THRESHOLD
        self.exp_pred_threshold = args.EXP_PRED_THRESHOLD
        self.map_pred_threshold = args.MAP_PRED_THRESHOLD
        self.explored_ray_fill = bool(getattr(args, "EXPLORED_RAY_FILL", False))
        self.selective_dynamic_obstacle_update = bool(
            getattr(args, "SELECTIVE_DYNAMIC_OBSTACLE_UPDATE", False)
        )
        self.obstacle_evidence_threshold = min(
            1.0,
            max(0.0, float(getattr(args, "OBSTACLE_EVIDENCE_THRESHOLD", 0.5))),
        )
        self.obstacle_evidence_max_observations = max(
            0,
            int(getattr(args, "OBSTACLE_EVIDENCE_MAX_OBSERVATIONS", 0)),
        )
        self.obstacle_clear_explored_threshold = 0.6
        
        # 分块地图：{(tile_x, tile_y): tensor[batch, C, 240, 240]}
        self.tiles = defaultdict(lambda: None)
        self.one_step_tiles = defaultdict(lambda: None)
        self.obstacle_evidence_tiles = defaultdict(lambda: None)
        
        # 地图尺寸（用于兼容性）
        self.map_shape = (args.MAP_SIZE_CM // args.MAP_RESOLUTION, 
                          args.MAP_SIZE_CM // args.MAP_RESOLUTION)  # (480, 480)
        
        # Local Map尺寸（固定240×240）
        self.local_w = self.TILE_SIZE
        self.local_h = self.TILE_SIZE
        self.map_size_cm = self.TILE_SIZE * self.resolution  # 1200cm = 12m
        
        # 兼容旧代码的属性（渲染用）
        self.full_w = self.TILE_SIZE  # 初始大小，会动态变化
        self.full_h = self.TILE_SIZE
        
        # Agent全局坐标（世界坐标，米，可以是负值）
        self.agent_global_x = 0.0  # 初始在原点
        self.agent_global_y = 0.0
        self.agent_orientation = 0.0  # 弧度
        
        # Local Pose（相对Local Map的坐标）
        self.local_pose = torch.zeros(self.num_environments, 3).float().to(self.device)
        self.full_pose = torch.zeros(self.num_environments, 3).float().to(self.device)
        self.curr_loc = torch.zeros(self.num_environments, 3).float().to(self.device)
        
        # Origins（Local Map左上角的世界坐标，可以是负值）
        self.origins = np.zeros((self.num_environments, 3))
        
        # Local Map Boundaries（在世界坐标系中的像素范围，可以是负值）
        self.lmb = np.zeros((self.num_environments, 4)).astype(int)
        
        # State (7 dimensions): [global_x, global_y, orientation, gx1, gx2, gy1, gy2]
        self.state = np.zeros((self.num_environments, 7))
        
        # 轨迹点列表（只存储坐标，渲染时连线）
        self.global_trajectory_points = []  # 全局轨迹，永不清空，用于global map
        self.subtask_trajectory_points = []  # 子任务轨迹，子任务开始时清空，用于local map
        self.last_trajectory_pos = None  # (tile_x, tile_y, local_px, local_py)
        
        # 当前Local Map和Full Map（用于兼容旧接口）
        self.local_map = None
        self.one_step_local_map = None
        self.full_map = None
        self.one_step_full_map = None
        self.full_map_crop_offset = (0, 0)  # 裁剪区域偏移量（用于坐标转换）
        self.one_step_full_map_crop_offset = (0, 0)
        
        if self.visualize or self.print_images:
            self.vis_image = vu.init_vis_image()
            self.rgb_vis = None

        # 高度参数
        self.max_height = int(360 / self.z_resolution)  # 72
        self.min_height = int(-40 / self.z_resolution)  # -8
        self.agent_height = args.AGENT_HEIGHT * 100.  # 88cm
        self.obstacle_min_height_cm = float(getattr(args, "OBSTACLE_MIN_HEIGHT_CM", 15.0))
        self.obstacle_max_height_cm = float(getattr(args, "OBSTACLE_MAX_HEIGHT_CM", 130.0))
        if self.obstacle_max_height_cm <= self.obstacle_min_height_cm:
            self.obstacle_max_height_cm = self.obstacle_min_height_cm + self.z_resolution
        self.shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, self.fov)

        # Feat通道
        self.feat = torch.ones(
            args.NUM_ENVIRONMENTS, 1, 
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)
        
        # Init grid（用于3D体素化，初始为1通道，后续动态扩展）
        self.init_grid = torch.zeros(
            args.NUM_ENVIRONMENTS, 1,
            self.vision_range, self.vision_range,
            self.max_height - self.min_height
        ).float().to(self.device)

    def clear_trajectory(self) -> None:
        """清空子任务轨迹点列表（用于local map，global map的轨迹保留）"""
        self.subtask_trajectory_points.clear()
        # 注意：不清空 global_trajectory_points，它用于global map显示完整历史

    def clear_landmark_channels(self, n_mapping: int = 15) -> None:
        """清空自定义 landmark 语义通道，避免跨子任务残留。"""
        lm_start_ch = self.MAP_CHANNELS + n_mapping

        def _zero_map_channels(map_tensor) -> None:
            if map_tensor is None or map_tensor.shape[1] <= lm_start_ch:
                return
            map_tensor[:, lm_start_ch:, :, :] = 0.0

        _zero_map_channels(self.local_map)
        _zero_map_channels(self.one_step_local_map)
        _zero_map_channels(self.full_map)
        _zero_map_channels(self.one_step_full_map)

        for tile in self.tiles.values():
            _zero_map_channels(tile)
        for tile in self.one_step_tiles.values():
            _zero_map_channels(tile)

    def clear_one_step_buffers(self) -> None:
        """清空 one-step 地图缓存，避免 recentering 时读回历史残留。"""
        if self.one_step_local_map is not None:
            self.one_step_local_map.zero_()
        if self.one_step_full_map is not None:
            self.one_step_full_map.zero_()
        self.one_step_tiles.clear()
        self.one_step_full_map_crop_offset = (0, 0)

    @staticmethod
    def _clone_optional_tensor(
        tensor: Optional[Any],
    ) -> Optional[Any]:
        if tensor is None:
            return None
        if torch.is_tensor(tensor):
            return tensor.clone()
        if isinstance(tensor, np.ndarray):
            return np.array(tensor, copy=True)
        if hasattr(tensor, "clone"):
            return tensor.clone()
        return copy.deepcopy(tensor)

    @staticmethod
    def _clone_optional_array(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if array is None:
            return None
        return np.array(array, copy=True)

    def _clone_tile_state(self, tiles_dict) -> Dict[Tuple[int, int], torch.Tensor]:
        cloned: Dict[Tuple[int, int], torch.Tensor] = {}
        for key, value in dict(tiles_dict).items():
            if value is None:
                continue
            cloned[(int(key[0]), int(key[1]))] = value.clone()
        return cloned

    def export_state(self) -> Dict[str, Any]:
        return {
            "tiles": self._clone_tile_state(self.tiles),
            "one_step_tiles": self._clone_tile_state(self.one_step_tiles),
            "obstacle_evidence_tiles": self._clone_tile_state(
                self.obstacle_evidence_tiles
            ),
            "local_map": self._clone_optional_tensor(self.local_map),
            "one_step_local_map": self._clone_optional_tensor(self.one_step_local_map),
            "full_map": self._clone_optional_tensor(self.full_map),
            "one_step_full_map": self._clone_optional_tensor(self.one_step_full_map),
            "full_pose": self._clone_optional_tensor(self.full_pose),
            "local_pose": self._clone_optional_tensor(self.local_pose),
            "curr_loc": self._clone_optional_tensor(self.curr_loc),
            "last_loc": self._clone_optional_tensor(self.last_loc),
            "origins": self._clone_optional_array(self.origins),
            "lmb": self._clone_optional_array(self.lmb),
            "state": self._clone_optional_array(self.state),
            "global_trajectory_points": [
                (int(point[0]), int(point[1]))
                for point in list(self.global_trajectory_points or [])
            ],
            "subtask_trajectory_points": [
                (int(point[0]), int(point[1]))
                for point in list(self.subtask_trajectory_points or [])
            ],
            "last_trajectory_pos": (
                tuple(int(value) for value in self.last_trajectory_pos)
                if self.last_trajectory_pos is not None
                else None
            ),
            "full_map_crop_offset": tuple(self.full_map_crop_offset or (0, 0)),
            "one_step_full_map_crop_offset": tuple(self.one_step_full_map_crop_offset or (0, 0)),
            "full_w": int(self.full_w),
            "full_h": int(self.full_h),
            "local_w": int(self.local_w),
            "local_h": int(self.local_h),
            "visited_vis": self._clone_optional_array(getattr(self, "visited_vis", None)),
        }

    def import_state(self, state: Optional[Dict[str, Any]]) -> None:
        def _restore_tiles(raw_tiles) -> defaultdict:
            restored = defaultdict(lambda: None)
            for key, value in dict(raw_tiles or {}).items():
                restored[(int(key[0]), int(key[1]))] = value.clone() if value is not None else None
            return restored

        if not state:
            self.reset()
            return

        self.tiles = _restore_tiles(state.get("tiles"))
        self.one_step_tiles = _restore_tiles(state.get("one_step_tiles"))
        self.obstacle_evidence_tiles = _restore_tiles(
            state.get("obstacle_evidence_tiles", state.get("obstacle_evidence_count_tiles"))
        )
        self.local_map = self._clone_optional_tensor(state.get("local_map"))
        self.one_step_local_map = self._clone_optional_tensor(state.get("one_step_local_map"))
        self.full_map = self._clone_optional_tensor(state.get("full_map"))
        self.one_step_full_map = self._clone_optional_tensor(state.get("one_step_full_map"))
        self.full_pose = self._clone_optional_tensor(state.get("full_pose"))
        self.local_pose = self._clone_optional_tensor(state.get("local_pose"))
        self.curr_loc = self._clone_optional_tensor(state.get("curr_loc"))
        self.last_loc = self._clone_optional_tensor(state.get("last_loc"))
        self.origins = self._clone_optional_array(state.get("origins"))
        self.lmb = self._clone_optional_array(state.get("lmb"))
        self.state = self._clone_optional_array(state.get("state"))
        self.global_trajectory_points = [
            (int(point[0]), int(point[1]))
            for point in list(state.get("global_trajectory_points", []) or [])
        ]
        self.subtask_trajectory_points = [
            (int(point[0]), int(point[1]))
            for point in list(state.get("subtask_trajectory_points", []) or [])
        ]
        last_trajectory_pos = state.get("last_trajectory_pos")
        self.last_trajectory_pos = (
            tuple(int(value) for value in last_trajectory_pos)
            if last_trajectory_pos is not None
            else None
        )
        self.full_map_crop_offset = tuple(state.get("full_map_crop_offset", (0, 0)) or (0, 0))
        self.one_step_full_map_crop_offset = tuple(
            state.get("one_step_full_map_crop_offset", (0, 0)) or (0, 0)
        )
        self.full_w = int(state.get("full_w", self.TILE_SIZE))
        self.full_h = int(state.get("full_h", self.TILE_SIZE))
        self.local_w = int(state.get("local_w", self.TILE_SIZE))
        self.local_h = int(state.get("local_h", self.TILE_SIZE))
        self.visited_vis = self._clone_optional_array(state.get("visited_vis"))

    def recenter_to_world_pose(
        self,
        world_pose: Optional[Tuple[float, float, float]],
        clear_one_step: bool = True,
    ) -> None:
        """Reload the local window around an externally supplied world pose."""
        if world_pose is None:
            return

        world_x_m, world_y_m, world_o_deg = [float(value) for value in world_pose[:3]]
        nc = int(self.local_map.shape[1]) if self.local_map is not None else (self.MAP_CHANNELS + 1)
        if self.local_map is None:
            self._prepare(nc)

        self.full_pose.fill_(0.0)
        self.full_pose[:, 0] = world_x_m
        self.full_pose[:, 1] = world_y_m
        self.full_pose[:, 2] = world_o_deg

        self.local_pose.fill_(0.0)
        self.local_pose[:, 0] = 6.0
        self.local_pose[:, 1] = 6.0
        self.local_pose[:, 2] = world_o_deg

        self.local_map = self.get_local_map_from_tiles(nc)
        if clear_one_step:
            self.one_step_tiles.clear()
            self.one_step_local_map = torch.zeros_like(self.local_map)
            self.one_step_full_map = None
            self.one_step_full_map_crop_offset = (0, 0)
        else:
            self.one_step_local_map = self.get_local_map_from_tiles(nc, is_one_step=True)

        self.curr_loc = self.full_pose.clone()
        self.last_loc = self.full_pose.clone()
        locs = self.full_pose.cpu().numpy()
        self.state[:, :3] = locs
        for e in range(self.num_environments):
            self.state[e, 3:] = self.lmb[e]

        self.full_map = None
        self.full_map_crop_offset = (0, 0)
    
    def _draw_line_on_tile(self, tile_key, x0, y0, x1, y1):
        """
        在tile的通道2上画线（Bresenham算法）
        
        Args:
            tile_key: (tile_x, tile_y)
            x0, y0: 起点（tile内局部坐标，px, py）
            x1, y1: 终点（tile内局部坐标，px, py）
        """
        tile = self.tiles[tile_key]
        if tile.shape[1] <= 2:
            return
        
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            # 标记当前点及周围（3x3）- 使用0.5表示轨迹
            for dx_mark in [-1, 0, 1]:
                for dy_mark in [-1, 0, 1]:
                    mark_x = x + dx_mark
                    mark_y = y + dy_mark
                    if 0 <= mark_x < self.TILE_SIZE and 0 <= mark_y < self.TILE_SIZE:
                        tile[0, 2, mark_x, mark_y] = 0.5
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def reset(self) -> None:
        """重置地图系统（分块架构）"""
        # 清空分类和tiles
        self.vis_classes = []
        self.tiles.clear()
        self.one_step_tiles.clear()
        self.obstacle_evidence_tiles.clear()

        # 清空轨迹
        self.global_trajectory_points = []
        self.subtask_trajectory_points = []
        self.last_trajectory_pos = None
        self.trajectory_points = []  # 清空轨迹点列表
        
        # 重置pose tensors
        self.local_pose.fill_(0.)
        self.full_pose.fill_(0.)
        self.curr_loc = torch.zeros(self.num_environments, 3).float().to(self.device)
        self.last_loc = None
        
        # 重置origins和lmb（numpy arrays）
        self.origins.fill(0.)
        self.lmb.fill(0.)
        self.state.fill(0.)
        
        # 重置feat
        self.feat = torch.ones(
            self.args.NUM_ENVIRONMENTS, 1, 
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)
        
        # 清空local_map和full_map（将在init_map_and_pose中重新创建）
        if hasattr(self, 'local_map'):
            self.local_map = None
            self.one_step_local_map = None
        if hasattr(self, 'full_map'):
            self.full_map = None
            self.one_step_full_map = None
        self.full_map_crop_offset = (0, 0)
        self.one_step_full_map_crop_offset = (0, 0)
        if hasattr(self, 'visited_vis'):
            self.visited_vis = None

        if self.visualize or self.print_images:
            self.vis_image = vu.init_vis_image()
            self.rgb_vis = None
    
    def _world_to_tile_coords(self, world_x_m, world_y_m):
        """
        世界坐标（米）→ 块索引 + 块内像素坐标
        
        Args:
            world_x_m, world_y_m: 世界坐标（米，可以是负值）
        
        Returns:
            tile_x, tile_y: 块索引（可以是负值）
            local_px, local_py: 块内像素坐标[0, 239]
        """
        # 转换为世界像素坐标（可以是负值）
        world_px = int(world_x_m * 100 / self.resolution)
        world_py = int(world_y_m * 100 / self.resolution)
        
        # 计算块索引（使用floor division处理负数）
        tile_x = world_px // self.TILE_SIZE
        tile_y = world_py // self.TILE_SIZE
        
        # 块内坐标（使用模运算，确保在[0, 239]范围内）
        local_px = world_px % self.TILE_SIZE
        local_py = world_py % self.TILE_SIZE
        
        return tile_x, tile_y, local_px, local_py
    
    def _tile_to_world_coords(self, tile_x, tile_y, local_px, local_py):
        """
        块坐标 → 世界坐标（米）
        
        Args:
            tile_x, tile_y: 块索引（可以是负值）
            local_px, local_py: 块内像素坐标
        
        Returns:
            world_x_m, world_y_m: 世界坐标（米）
        """
        # 计算世界像素坐标
        world_px = tile_x * self.TILE_SIZE + local_px
        world_py = tile_y * self.TILE_SIZE + local_py
        
        # 转换为米
        world_x_m = world_px * self.resolution / 100.0
        world_y_m = world_py * self.resolution / 100.0
        
        return world_x_m, world_y_m
    
    def _create_empty_tile(self, nc):
        """创建空白块"""
        return torch.zeros(
            self.num_environments, nc,
            self.TILE_SIZE, self.TILE_SIZE
        ).float().to(self.device)
    
    def _ensure_tiles_exist(self, tile_indices, nc, is_one_step=False):
        """
        确保指定的块存在
        
        Args:
            tile_indices: [(tile_x, tile_y), ...]
            nc: 通道数
            is_one_step: 是否为one_step_tiles
        """
        # 选择使用的tiles字典
        tiles_dict = self.one_step_tiles if is_one_step else self.tiles
        
        for (tile_x, tile_y) in tile_indices:
            if tiles_dict.get((tile_x, tile_y)) is None:
                tiles_dict[(tile_x, tile_y)] = self._create_empty_tile(nc)
                if not is_one_step:  # 只在主tiles创建时打印
                    pass  # tile 创建不再打印
    
    def _get_tiles_for_region(self, center_x_m, center_y_m, size_m):
        """
        获取指定区域需要的所有块索引
        
        Args:
            center_x_m, center_y_m: 区域中心世界坐标（米）
            size_m: 区域尺寸（米）
        
        Returns:
            tiles: 块索引列表 [(tile_x, tile_y), ...]
        """
        half_size_m = size_m / 2.0
        
        # 区域四角的世界坐标
        min_x = center_x_m - half_size_m
        max_x = center_x_m + half_size_m
        min_y = center_y_m - half_size_m
        max_y = center_y_m + half_size_m
        
        # 转换为块索引
        # 注意：区域是[min, max)左闭右开，所以max减去一个小量避免包含边界上的下一个块
        # 例如：[0, 12)m 应该只包含 tile_x=0，不包含 tile_x=1
        tile_x_min = int(np.floor(min_x / self.TILE_SIZE_M))
        tile_x_max = int(np.floor((max_x - 0.001) / self.TILE_SIZE_M))
        tile_y_min = int(np.floor(min_y / self.TILE_SIZE_M))
        tile_y_max = int(np.floor((max_y - 0.001) / self.TILE_SIZE_M))
        
        # 生成所有需要的块
        tiles = [
            (tx, ty)
            for tx in range(tile_x_min, tile_x_max + 1)
            for ty in range(tile_y_min, tile_y_max + 1)
        ]
        
        return tiles
    
    def get_local_map_from_tiles(self, nc, env_id=None, is_one_step=False):
        """
        从块中拼接Local Map（240×240，以agent为中心）
        
        Args:
            nc: 通道数
            env_id: 环境ID，如果为None则处理所有环境并返回[batch, C, H, W]
            is_one_step: 是否使用one_step_tiles
        
        Returns:
            local_map: [C, 240, 240] (if env_id specified) 或 [batch, C, 240, 240]
        """
        # 确定使用哪个环境的pose
        if env_id is not None:
            agent_x_m = self.full_pose[env_id, 0].item()
            agent_y_m = self.full_pose[env_id, 1].item()
            num_envs = 1
        else:
            # 使用第一个环境（兼容性）
            agent_x_m = self.full_pose[0, 0].item()
            agent_y_m = self.full_pose[0, 1].item()
            num_envs = self.num_environments
        
        # 选择使用的tiles字典
        tiles_dict = self.one_step_tiles if is_one_step else self.tiles
        
        # 获取需要的块
        tiles_needed = self._get_tiles_for_region(agent_x_m, agent_y_m, self.TILE_SIZE_M)
        self._ensure_tiles_exist(tiles_needed, nc, is_one_step=is_one_step)
        
        # 创建Local Map
        local_map = torch.zeros(
            num_envs, nc,
            self.TILE_SIZE, self.TILE_SIZE
        ).float().to(self.device)
        
        # Local Map的世界像素范围
        # 关键：tensor维度[bs, C, H, W]，H是行（Y轴），W是列（X轴）
        # px = Y轴像素（行，对应gx），py = X轴像素（列，对应gy）
        half_size_px = self.TILE_SIZE // 2  # 120像素
        
        # agent世界坐标(m) -> 世界像素
        agent_px = int(agent_y_m * 100 / self.resolution)  # Y轴像素 = agent_y * 20
        agent_py = int(agent_x_m * 100 / self.resolution)  # X轴像素 = agent_x * 20
        
        # Local Map范围：[agent - 6m, agent + 6m) = [agent - 120px, agent + 120px)
        start_px = agent_px - half_size_px  # Y轴起始 (gx1)
        end_px = agent_px + half_size_px    # Y轴结束 (gx2)
        start_py = agent_py - half_size_px  # X轴起始 (gy1)
        end_py = agent_py + half_size_px    # X轴结束 (gy2)
        
        # 从各个块中提取数据
        for (tile_x, tile_y) in tiles_needed:
            tile = tiles_dict.get((tile_x, tile_y))
            if tile is None:
                continue
            
            # 块的世界像素范围
            # tile_y对应Y轴(gx/px), tile_x对应X轴(gy/py)
            tile_start_px = tile_y * self.TILE_SIZE  # Y轴像素起始
            tile_end_px = tile_start_px + self.TILE_SIZE
            tile_start_py = tile_x * self.TILE_SIZE  # X轴像素起始
            tile_end_py = tile_start_py + self.TILE_SIZE
            
            # 计算交集
            copy_start_px = max(start_px, tile_start_px)
            copy_end_px = min(end_px, tile_end_px)
            copy_start_py = max(start_py, tile_start_py)
            copy_end_py = min(end_py, tile_end_py)
            
            if copy_start_px >= copy_end_px or copy_start_py >= copy_end_py:
                continue
            
            # 块内坐标（tensor: [H,W] = [Y,X]）
            tile_h_start = copy_start_px - tile_start_px
            tile_h_end = copy_end_px - tile_start_px
            tile_w_start = copy_start_py - tile_start_py
            tile_w_end = copy_end_py - tile_start_py
            
            # Local Map坐标
            local_h_start = copy_start_px - start_px
            local_h_end = copy_end_px - start_px
            local_w_start = copy_start_py - start_py
            local_w_end = copy_end_py - start_py
            
            # 复制数据（tensor索引: [batch, channel, H, W]）
            local_map[:, :,
                     local_h_start:local_h_end,
                     local_w_start:local_w_end] = \
                tile[:, :,
                     tile_h_start:tile_h_end,
                     tile_w_start:tile_w_end]
        
        # 更新lmb（Local Map在世界坐标系中的像素边界）
        # lmb顺序：[gx1, gx2, gy1, gy2] 其中gx是Y方向像素，gy是X方向像素
        if env_id is not None:
            # 只更新指定环境
            self.lmb[env_id] = [start_px, end_px, start_py, end_py]
            # origins顺序：[X, Y, 0] 米
            self.origins[env_id] = [
                start_py * self.resolution / 100.0,  # X方向（米）= gy * 0.05
                start_px * self.resolution / 100.0,  # Y方向（米）= gx * 0.05
                0.0
            ]
            # 返回单个环境的map，去除batch维度
            return local_map[0]
        else:
            # 更新所有环境
            for e in range(self.num_environments):
                self.lmb[e] = [start_px, end_px, start_py, end_py]
                self.origins[e] = [
                    start_py * self.resolution / 100.0,  # X方向（米）= gy * 0.05
                    start_px * self.resolution / 100.0,  # Y方向（米）= gx * 0.05
                    0.0
                ]
            return local_map

    def _get_local_obstacle_evidence_channel(
        self,
        channel: int,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Load one persistent obstacle evidence channel for the current local window."""
        count_map = torch.zeros(
            1,
            1,
            self.TILE_SIZE,
            self.TILE_SIZE,
        ).float().to(self.device)
        gx1, gx2, gy1, gy2 = self.lmb[env_id]
        tile_y_min = gx1 // self.TILE_SIZE
        tile_y_max = (gx2 - 1) // self.TILE_SIZE
        tile_x_min = gy1 // self.TILE_SIZE
        tile_x_max = (gy2 - 1) // self.TILE_SIZE

        for tile_y in range(tile_y_min, tile_y_max + 1):
            for tile_x in range(tile_x_min, tile_x_max + 1):
                tile = self.obstacle_evidence_tiles.get((tile_x, tile_y))
                if tile is None or tile.shape[1] <= int(channel):
                    continue

                tile_start_gx = tile_y * self.TILE_SIZE
                tile_end_gx = tile_start_gx + self.TILE_SIZE
                tile_start_gy = tile_x * self.TILE_SIZE
                tile_end_gy = tile_start_gy + self.TILE_SIZE
                copy_start_gx = max(gx1, tile_start_gx)
                copy_end_gx = min(gx2, tile_end_gx)
                copy_start_gy = max(gy1, tile_start_gy)
                copy_end_gy = min(gy2, tile_end_gy)
                if copy_start_gx >= copy_end_gx or copy_start_gy >= copy_end_gy:
                    continue

                tile_h_start = copy_start_gx - tile_start_gx
                tile_h_end = copy_end_gx - tile_start_gx
                tile_w_start = copy_start_gy - tile_start_gy
                tile_w_end = copy_end_gy - tile_start_gy
                local_h_start = copy_start_gx - gx1
                local_h_end = copy_end_gx - gx1
                local_w_start = copy_start_gy - gy1
                local_w_end = copy_end_gy - gy1
                count_map[0, 0, local_h_start:local_h_end, local_w_start:local_w_end] = (
                    tile[env_id, channel, tile_h_start:tile_h_end, tile_w_start:tile_w_end]
                )
        return count_map

    def _get_local_obstacle_evidence_counts(self, env_id: int = 0) -> torch.Tensor:
        return self._get_local_obstacle_evidence_channel(0, env_id=env_id)

    def _get_local_obstacle_evidence_scores(self, env_id: int = 0) -> torch.Tensor:
        return self._get_local_obstacle_evidence_channel(1, env_id=env_id)

    def _update_obstacle_evidence_tiles(
        self,
        local_counts: torch.Tensor,
        local_scores: Optional[torch.Tensor] = None,
        env_id: int = 0,
    ) -> None:
        """Persist local obstacle observation counts and scores beside the tiled map."""
        gx1, gx2, gy1, gy2 = self.lmb[env_id]
        tile_y_min = gx1 // self.TILE_SIZE
        tile_y_max = (gx2 - 1) // self.TILE_SIZE
        tile_x_min = gy1 // self.TILE_SIZE
        tile_x_max = (gy2 - 1) // self.TILE_SIZE

        for tile_y in range(tile_y_min, tile_y_max + 1):
            for tile_x in range(tile_x_min, tile_x_max + 1):
                key = (tile_x, tile_y)
                tile = self.obstacle_evidence_tiles.get(key)
                if tile is None:
                    tile = self._create_empty_tile(2)
                    self.obstacle_evidence_tiles[key] = tile
                elif tile.shape[1] < 2:
                    tile_pad = torch.zeros(
                        tile.shape[0],
                        2 - tile.shape[1],
                        tile.shape[2],
                        tile.shape[3],
                    ).float().to(tile.device)
                    tile = torch.cat([tile, tile_pad], axis=1)
                    self.obstacle_evidence_tiles[key] = tile

                tile_start_gx = tile_y * self.TILE_SIZE
                tile_end_gx = tile_start_gx + self.TILE_SIZE
                tile_start_gy = tile_x * self.TILE_SIZE
                tile_end_gy = tile_start_gy + self.TILE_SIZE
                copy_start_gx = max(gx1, tile_start_gx)
                copy_end_gx = min(gx2, tile_end_gx)
                copy_start_gy = max(gy1, tile_start_gy)
                copy_end_gy = min(gy2, tile_end_gy)
                if copy_start_gx >= copy_end_gx or copy_start_gy >= copy_end_gy:
                    continue

                tile_h_start = copy_start_gx - tile_start_gx
                tile_h_end = copy_end_gx - tile_start_gx
                tile_w_start = copy_start_gy - tile_start_gy
                tile_w_end = copy_end_gy - tile_start_gy
                local_h_start = copy_start_gx - gx1
                local_h_end = copy_end_gx - gx1
                local_w_start = copy_start_gy - gy1
                local_w_end = copy_end_gy - gy1
                tile[env_id, 0, tile_h_start:tile_h_end, tile_w_start:tile_w_end] = (
                    local_counts[0, 0, local_h_start:local_h_end, local_w_start:local_w_end]
                )
                if local_scores is not None:
                    tile[env_id, 1, tile_h_start:tile_h_end, tile_w_start:tile_w_end] = (
                        local_scores[0, 0, local_h_start:local_h_end, local_w_start:local_w_end]
                    )
    
    def update_tiles_from_local_map(self, local_map, env_id=0, is_one_step=False):
        """
        将更新后的Local Map写回到块中
        
        Args:
            local_map: [C, 240, 240] 单个环境的Local Map
            env_id: 环境ID
            is_one_step: 是否写回到one_step_tiles
        """
        # 选择使用的tiles字典
        tiles_dict = self.one_step_tiles if is_one_step else self.tiles
        
        # Local Map的世界像素范围（从lmb获取）
        # lmb格式：[gx1, gx2, gy1, gy2] = [Y轴起, Y轴终, X轴起, X轴终]
        gx1, gx2, gy1, gy2 = self.lmb[env_id]
        
        # 计算涉及的块
        # tile_y 对应 Y轴 (gx)，tile_x 对应 X轴 (gy)
        tile_y_min = gx1 // self.TILE_SIZE
        tile_y_max = (gx2 - 1) // self.TILE_SIZE
        tile_x_min = gy1 // self.TILE_SIZE
        tile_x_max = (gy2 - 1) // self.TILE_SIZE

        tiles_needed = [
            (tile_x, tile_y)
            for tile_y in range(tile_y_min, tile_y_max + 1)
            for tile_x in range(tile_x_min, tile_x_max + 1)
        ]
        self._ensure_tiles_exist(tiles_needed, local_map.shape[0], is_one_step=is_one_step)
        
        # 写回数据到各个块
        for tile_y in range(tile_y_min, tile_y_max + 1):
            for tile_x in range(tile_x_min, tile_x_max + 1):
                tile = tiles_dict.get((tile_x, tile_y))
                if tile is None:
                    continue
                
                # 块的世界像素范围
                # tile_y对应Y轴(gx), tile_x对应X轴(gy)
                tile_start_gx = tile_y * self.TILE_SIZE
                tile_end_gx = tile_start_gx + self.TILE_SIZE
                tile_start_gy = tile_x * self.TILE_SIZE
                tile_end_gy = tile_start_gy + self.TILE_SIZE
                
                # 计算交集
                copy_start_gx = max(gx1, tile_start_gx)
                copy_end_gx = min(gx2, tile_end_gx)
                copy_start_gy = max(gy1, tile_start_gy)
                copy_end_gy = min(gy2, tile_end_gy)
                
                if copy_start_gx >= copy_end_gx or copy_start_gy >= copy_end_gy:
                    continue
                
                # 块内坐标（tensor索引: [H, W] = [Y, X]）
                tile_h_start = copy_start_gx - tile_start_gx
                tile_h_end = copy_end_gx - tile_start_gx
                tile_w_start = copy_start_gy - tile_start_gy
                tile_w_end = copy_end_gy - tile_start_gy
                
                # Local Map坐标
                local_h_start = copy_start_gx - gx1
                local_h_end = copy_end_gx - gx1
                local_w_start = copy_start_gy - gy1
                local_w_end = copy_end_gy - gy1
                
                # 写回数据（local_map是单个环境的，形状[C, H, W]）
                # 直接覆盖，不需要保留 Channel 2（已废弃）
                tile[env_id, :,
                     tile_h_start:tile_h_end,
                         tile_w_start:tile_w_end] = \
                        local_map[:,
                                 local_h_start:local_h_end,
                                 local_w_start:local_w_end]
    
    def get_full_map_for_rendering(self, crop_size_m=24.0, is_one_step=False, rotate_to_agent_heading=True):
        """
        获取用于渲染的全局地图（以agent为中心裁剪，可选旋转）
        
        仅从已存在的tiles提取数据，不创建空白块（内存优化）
        未探索区域保持为0（黑色/未知）
        
        Args:
            crop_size_m: 裁剪尺寸（米），默认24m×24m
            is_one_step: 是否使用one_step_tiles
            rotate_to_agent_heading: 是否根据agent朝向旋转地图，使agent朝向向上
        
        Returns:
            full_map: [batch, C, H, W] - 如果rotate_to_agent_heading=True，则已旋转
            map_size: (H, W) 实际尺寸
            crop_offset: (start_px, start_py) 裁剪偏移量（世界坐标）
        """
        # 选择使用的tiles字典
        tiles_dict = self.one_step_tiles if is_one_step else self.tiles
        
        # 获取需要的块
        agent_x_m = self.full_pose[0, 0].item()
        agent_y_m = self.full_pose[0, 1].item()
        
        tiles_needed = self._get_tiles_for_region(agent_x_m, agent_y_m, crop_size_m)
        
        if not tiles_needed:
            # 没有任何块，返回空地图
            nc = self.MAP_CHANNELS + 1
            map_size_px = int(crop_size_m * 100 / self.resolution)
            return torch.zeros(self.num_environments, nc, map_size_px, map_size_px).float().to(self.device), (map_size_px, map_size_px)
        
        # 获取通道数：优先从local_map获取（最新），如果没有则从已有tile获取
        if self.local_map is not None:
            nc = self.local_map.shape[1]
        else:
            # 从已存在的tile获取通道数
            nc = None
            for tile_idx in tiles_needed:
                tile = tiles_dict.get(tile_idx)
                if tile is not None:
                    nc = tile.shape[1]
                    break
            if nc is None:
                nc = self.MAP_CHANNELS + 1
        
        # 注意：不调用_ensure_tiles_exist()
        # 只从已存在的tiles中提取，未探索区域保持为0
        
        # 计算裁剪区域的世界像素范围
        crop_size_px = int(crop_size_m * 100 / self.resolution)
        half_crop = crop_size_px // 2
        
        # agent世界坐标(m) -> 世界像素
        # px = Y轴像素，py = X轴像素
        agent_px = int(agent_y_m * 100 / self.resolution)  # Y轴像素
        agent_py = int(agent_x_m * 100 / self.resolution)  # X轴像素
        
        start_px = agent_px - half_crop  # Y轴起始
        end_px = agent_px + half_crop    # Y轴结束
        start_py = agent_py - half_crop  # X轴起始
        end_py = agent_py + half_crop    # X轴结束
        
        # 创建Full Map
        full_map = torch.zeros(
            self.num_environments, nc,
            crop_size_px, crop_size_px
        ).float().to(self.device)
        
        # 从各个块中拼接
        for (tile_x, tile_y) in tiles_needed:
            tile = tiles_dict.get((tile_x, tile_y))
            if tile is None:
                continue
            
            # 块的世界像素范围
            # tile_y对应Y轴(px), tile_x对应X轴(py)
            tile_start_px = tile_y * self.TILE_SIZE  # Y轴像素起始
            tile_end_px = tile_start_px + self.TILE_SIZE
            tile_start_py = tile_x * self.TILE_SIZE  # X轴像素起始
            tile_end_py = tile_start_py + self.TILE_SIZE
            
            # 计算交集
            copy_start_px = max(start_px, tile_start_px)
            copy_end_px = min(end_px, tile_end_px)
            copy_start_py = max(start_py, tile_start_py)
            copy_end_py = min(end_py, tile_end_py)
            
            if copy_start_px >= copy_end_px or copy_start_py >= copy_end_py:
                continue
            
            # 块内坐标（tensor: [H,W] = [Y,X]）
            tile_h_start = copy_start_px - tile_start_px
            tile_h_end = copy_end_px - tile_start_px
            tile_w_start = copy_start_py - tile_start_py
            tile_w_end = copy_end_py - tile_start_py
            
            # Full Map坐标
            full_h_start = copy_start_px - start_px
            full_h_end = copy_end_px - start_px
            full_w_start = copy_start_py - start_py
            full_w_end = copy_end_py - start_py
            
            # 复制数据（tensor索引: [batch, channel, H, W]）
            full_map[:, :,
                    full_h_start:full_h_end,
                    full_w_start:full_w_end] = \
                tile[:, :,
                     tile_h_start:tile_h_end,
                     tile_w_start:tile_w_end]
        
        # 更新full_w和full_h（用于兼容旧代码）
        self.full_w = crop_size_px
        self.full_h = crop_size_px
        
        # 返回裁剪区域的世界像素偏移量（用于trajectory_points的坐标转换）
        crop_offset = (start_px, start_py)
        
        # 如果需要，根据agent朝向旋转地图（同时旋转trajectory_points坐标）
        if rotate_to_agent_heading:
            agent_orientation = self.full_pose[0, 2].item()  # 度数
            full_map = self._rotate_map_to_agent_heading(full_map, agent_orientation)
            # 注意：crop_offset 保持不变（旋转前的偏移量），在mapper中使用时会一起转换trajectory_points
        
        return full_map, (crop_size_px, crop_size_px), crop_offset
    
    def _rotate_map_to_agent_heading(self, map_tensor, agent_orientation_deg):
        """
        根据agent朝向旋转地图，使agent的前方（朝向）对应地图的正上方
        
        使用PyTorch的grid_sample进行高质量旋转，避免OpenCV插值失真
        
        Args:
            map_tensor: [batch, C, H, W] 输入地图
            agent_orientation_deg: agent朝向（度数，0=东，90=北，180=西，270=南）
        
        Returns:
            rotated_map: [batch, C, H, W] 旋转后的地图，agent朝向向上
        """
        import torch.nn.functional as F
        import math
        
        batch, nc, h, w = map_tensor.shape
        
        # 计算旋转角度：让agent朝向变成正上方（90度）
        # agent_orientation: 0=东，90=北
        # 目标：旋转后agent朝向=90度（正上方）
        # 所以旋转角度 = 90 - agent_orientation
        rotation_angle_deg = 90 - agent_orientation_deg
        rotation_angle_rad = math.radians(rotation_angle_deg)
        
        # 如果不需要旋转，直接返回
        if abs(rotation_angle_deg) < 0.1:
            return map_tensor
        
        # 创建旋转矩阵（围绕中心旋转）
        cos_theta = math.cos(rotation_angle_rad)
        sin_theta = math.sin(rotation_angle_rad)
        
        # 仿射变换矩阵 (2x3) for PyTorch grid_sample
        # 注意：PyTorch的grid_sample使用归一化坐标[-1, 1]
        theta = torch.tensor([
            [cos_theta, sin_theta, 0],
            [-sin_theta, cos_theta, 0]
        ], dtype=torch.float32, device=map_tensor.device)
        
        theta = theta.unsqueeze(0).repeat(batch, 1, 1)  # [batch, 2, 3]
        
        # 生成采样网格
        grid = F.affine_grid(theta, map_tensor.size(), align_corners=False)
        
        # 采样（使用最近邻插值保持语义清晰度）
        rotated_map = F.grid_sample(
            map_tensor, grid, 
            mode='nearest',  # 最近邻插值，保持语义信息不失真
            padding_mode='zeros',  # 边界外用0填充（未探索区域）
            align_corners=False
        )
        
        return rotated_map
    
    def _dynamic_process(self, num_detected_classes: int) -> None:
        """
        动态调整通道数以适应新检测到的类别
        """
        vr = self.vision_range
        target_channels = 1 + num_detected_classes
        
        # 重新创建init_grid
        self.init_grid = torch.zeros(
            self.args.NUM_ENVIRONMENTS, target_channels, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        
        # 同步调整feat的通道数
        current_feat_channels = self.feat.shape[1]
        if target_channels > current_feat_channels:
            pad_num = target_channels - current_feat_channels
            feat_pad = torch.ones(
                self.num_environments, 
                pad_num, 
                self.screen_h // self.du_scale * self.screen_w // self.du_scale
            ).float().to(self.device)
            self.feat = torch.cat([self.feat, feat_pad], axis=1)
        elif target_channels < current_feat_channels:
            self.feat = self.feat[:, :target_channels, :]
        
        # 调整Local Map的通道数
        new_nc = num_detected_classes + self.MAP_CHANNELS
        if new_nc > self.local_map.shape[1]:
            pad_num = new_nc - self.local_map.shape[1]
            local_map_pad = torch.zeros(self.num_environments, 
                                        pad_num, 
                                        self.local_w, 
                                        self.local_h).float().to(self.device)
            self.local_map = torch.cat([self.local_map, local_map_pad], axis=1)
            self.one_step_local_map = torch.cat([self.one_step_local_map, local_map_pad], axis=1)
            
            # 对于分块系统，需要扩展所有tiles的通道数
            for tile_key, tile in self.tiles.items():
                if tile is not None:
                    tile_pad = torch.zeros(self.num_environments,
                                          pad_num,
                                          self.TILE_SIZE,
                                          self.TILE_SIZE).float().to(self.device)
                    self.tiles[tile_key] = torch.cat([tile, tile_pad], axis=1)
            for tile_key, tile in self.one_step_tiles.items():
                if tile is not None:
                    tile_pad = torch.zeros(self.num_environments,
                                          pad_num,
                                          self.TILE_SIZE,
                                          self.TILE_SIZE).float().to(self.device)
                    self.one_step_tiles[tile_key] = torch.cat([tile, tile_pad], axis=1)
            
            # 更新渲染用的Full Map通道数
            if hasattr(self, 'full_map') and self.full_map is not None:
                full_map_pad = torch.zeros(self.num_environments,
                                          pad_num,
                                          self.full_map.shape[2],
                                          self.full_map.shape[3]).float().to(self.device)
                self.full_map = torch.cat([self.full_map, full_map_pad], axis=1)
                self.one_step_full_map = torch.cat([self.one_step_full_map, full_map_pad], axis=1)
            
    def _prepare(self, nc: int) -> None:
        """
        初始化Local Map、姿态、边界等
        
        注意：分块系统不需要预先创建full_map，tiles按需创建
        """
        # Local Map大小：240×240（TILE_SIZE）
        self.full_w, self.full_h = self.map_shape  # 保持兼容性，但实际不用于tiles
        self.local_w = self.TILE_SIZE
        self.local_h = self.TILE_SIZE
        self.visited_vis = np.zeros(self.map_shape)  # 用于可视化的agent轨迹
        
        # 只创建Local Map（不再创建固定的full_map）
        self.local_map = torch.zeros(self.num_environments, 
                                     nc, 
                                     self.local_w, 
                                     self.local_h).float().to(self.device)
        self.one_step_local_map = torch.zeros(self.num_environments, 
                                              nc, 
                                              self.local_w, 
                                              self.local_h).float().to(self.device)
        
        # full_map和one_step_full_map将在需要时通过get_full_map_for_rendering()生成
        self.full_map = None
        self.one_step_full_map = None
        
        # 姿态：世界坐标（可以为负）
        self.full_pose = torch.zeros(self.num_environments, 3).float().to(self.device)
        self.local_pose = torch.zeros(self.num_environments, 3).float().to(self.device)
        self.curr_loc = torch.zeros(self.num_environments, 3).float().to(self.device)
        
        # Local Map的世界坐标原点（左上角）
        self.origins = np.zeros((self.num_environments, 3))

        # Local Map边界（世界像素坐标）
        self.lmb = np.zeros((self.num_environments, 4)).astype(int)
        
        # state: [x, y, ori, gx1, gx2, gy1, gy2]
        self.state = np.zeros((self.num_environments, 7))
        
    def _get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc # represent agent's position
        local_w, local_h = local_sizes # (240, 240)
        full_w, full_h = full_sizes # (480, 480)

        if self.global_downscaling > 1: # True, since args.global_downscaling = 2
            # calculate local map boundaries in full_map: width: (gx1, gx2); height: (gy1, gy2)
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]
        
    def init_map_and_pose(
        self,
        num_detected_classes: int,
        initial_pose: Optional[Tuple[float, float, float]] = None,
    ):
        """
        初始化分块地图和agent姿态
        
        新设计：
        1. Agent初始世界坐标为(6.0, 6.0)在Tile(0,0)中心
        2. 创建初始块tile(0, 0)，覆盖世界坐标[0, 12)m × [0, 12)m
        3. Local Map以agent为中心，范围[0, 12)m × [0, 12)m
        """
        nc = num_detected_classes + self.MAP_CHANNELS
        
        # 确保必要的tensor已初始化
        if self.local_map is None:
            self._prepare(nc)
        
        start_x_m = 6.0
        start_y_m = 6.0
        start_o_deg = 0.0
        if initial_pose is not None:
            start_x_m, start_y_m, start_o_deg = [float(value) for value in initial_pose[:3]]

        # Agent初始世界坐标：默认(6.0, 6.0, 0.0)，多楼层切换时允许从外部指定绝对相对世界位姿。
        self.full_pose.fill_(0.)
        self.full_pose[:, 0] = start_x_m
        self.full_pose[:, 1] = start_y_m
        self.full_pose[:, 2] = start_o_deg

        # 创建覆盖当前局部窗口的初始tiles，再以当前位置为中心提取Local Map。
        tiles_needed = self._get_tiles_for_region(start_x_m, start_y_m, self.TILE_SIZE_M)
        self._ensure_tiles_exist(tiles_needed or [(0, 0)], nc)

        self.local_map = self.get_local_map_from_tiles(nc)
        self.one_step_local_map = torch.zeros_like(self.local_map)

        # Local Pose：agent总是在当前Local Map中心，世界坐标通过origins/lmb保持连续。
        for e in range(self.num_environments):
            self.local_pose[e] = torch.tensor([6.0, 6.0, start_o_deg]).float().to(self.device)
            self.curr_loc[e] = self.full_pose[e].clone()
        
        # 更新state
        locs = self.full_pose.cpu().numpy()
        self.state[:, :3] = locs
        for e in range(self.num_environments):
            self.state[e, 3:] = self.lmb[e]
        
        # Full Map将在首次update_map或需要时生成
        # 这里不预先生成，避免创建不必要的tiles
        self.full_map = None
        self.one_step_full_map = None
        self.full_map_crop_offset = (0, 0)
        self.one_step_full_map_crop_offset = (0, 0)
        
                                
    def update_map(self, step: int, detected_classes: OrderedSet, current_episode_id: int) -> None:
        """
        更新分块地图
        
        关键逻辑：
        - Agent在Local Map中的位置是动态的（由forward()更新local_pose）
        - 例如：(120,120) → (140,125) → (160,130) ... 逐渐偏离中心
        - 每CENTER_RESET_STEPS步recentering：重新提取Local Map，agent回到中心
        
        步骤：
        1. 根据当前local_pose标记agent在Local Map中的位置
        2. 将更新的Local Map写回到tiles
        3. 定期recentering：重新获取以agent为中心的Local Map
        """
        if step == 0:
            self.last_loc = self.state[:, :3]
        else:
            self.last_loc = self.curr_loc
        
        # 更新state：world坐标 = local_pose（relative to Local Map origin）
        locs = self.local_pose.cpu().numpy()
        # origins保存Local Map左上角的世界坐标
        self.state[:, :3] = locs + self.origins
        self.curr_loc = self.state[:, :3]
        
        # 更新full_pose：世界坐标 + 朝向
        for e in range(self.num_environments):
            self.full_pose[e, 0] = self.state[e, 0]  # 世界X坐标（米）
            self.full_pose[e, 1] = self.state[e, 1]  # 世界Y坐标（米）
            self.full_pose[e, 2] = self.local_pose[e, 2]  # 朝向（度数）
        
        # 记录轨迹点（世界坐标列表，仅用于可视化连线）
        # 使用第一个环境的位置（单环境模式）
        agent_x_m = self.full_pose[0, 0].item()
        agent_y_m = self.full_pose[0, 1].item()
        
        # 转换为世界像素坐标
        agent_px = int(agent_y_m * 100 / self.resolution)  # Y轴像素
        agent_py = int(agent_x_m * 100 / self.resolution)  # X轴像素
        
        # 确定agent所在的tile
        tile_x = int(np.floor(agent_x_m / self.TILE_SIZE_M))
        tile_y = int(np.floor(agent_y_m / self.TILE_SIZE_M))
        
        # 计算在tile内的局部坐标
        tile_start_x_m = tile_x * self.TILE_SIZE_M
        tile_start_y_m = tile_y * self.TILE_SIZE_M
        local_px = int((agent_y_m - tile_start_y_m) * 100 / self.resolution)
        local_py = int((agent_x_m - tile_start_x_m) * 100 / self.resolution)
        
        # 限制在tile范围内
        local_px = max(0, min(self.TILE_SIZE - 1, local_px))
        local_py = max(0, min(self.TILE_SIZE - 1, local_py))
        
        # 只在位置变化时记录轨迹点（避免重复）
        current_pos = (tile_x, tile_y, local_px, local_py)
        if self.last_trajectory_pos != current_pos:
            # 保存轨迹点的世界坐标（用于渲染时连线）
            self.global_trajectory_points.append((agent_px, agent_py))  # 全局轨迹，永不清空
            self.subtask_trajectory_points.append((agent_px, agent_py))  # 子任务轨迹，可清空
            
            # print(f"[Trajectory] 记录轨迹点: ({agent_px}, {agent_py}) | 全局:{len(self.global_trajectory_points)}, 子任务:{len(self.subtask_trajectory_points)}")
            
            self.last_trajectory_pos = current_pos
        
        # 当前位置直接从 full_pose 获取；不再依赖旧的 Channel 2 waypoint 逻辑
        
        # 将更新后的Local Map写回到对应的tiles
        for e in range(self.num_environments):
            self.update_tiles_from_local_map(self.local_map[e], env_id=e, is_one_step=False)
            self.update_tiles_from_local_map(self.one_step_local_map[e], env_id=e, is_one_step=True)
        
        # 定期recentering：当agent偏离Local Map中心时，重新获取Local Map
        if ((step + 1) % self.args.CENTER_RESET_STEPS) == 0:
            nc = self.local_map.shape[1]
            for e in range(self.num_environments):
                # 从tiles获取新的Local Map（以当前agent为中心）
                self.local_map[e] = self.get_local_map_from_tiles(nc, e)
                self.one_step_local_map[e] = self.get_local_map_from_tiles(nc, e, is_one_step=True)
                
                # 更新Local Map的世界坐标边界
                agent_world_x, agent_world_y = self.full_pose[e, 0].item(), self.full_pose[e, 1].item()
                # Local Map范围：[agent - 6m, agent + 6m)
                self.lmb[e, 0] = int((agent_world_y - 6.0) * 100.0 / self.resolution)  # gx1
                self.lmb[e, 1] = int((agent_world_y + 6.0) * 100.0 / self.resolution)  # gx2
                self.lmb[e, 2] = int((agent_world_x - 6.0) * 100.0 / self.resolution)  # gy1
                self.lmb[e, 3] = int((agent_world_x + 6.0) * 100.0 / self.resolution)  # gy2
                self.state[e, 3:] = self.lmb[e]
                
                # 更新origins：Local Map左上角的世界坐标
                self.origins[e] = [
                    self.lmb[e, 2] * self.resolution / 100.0,  # X方向（米）
                    self.lmb[e, 0] * self.resolution / 100.0,  # Y方向（米）
                    0.
                ]
                
                # 重置local_pose：因为重新提取的Local Map以agent为中心
                # agent重新回到Local Map中心 = (6m, 6m) = (120px, 120px)
                # 这就是recentering：让agent回到Local Map中心，方便继续探索
                self.local_pose[e, 0] = 6.0
                self.local_pose[e, 1] = 6.0
                # 保持原有朝向
                self.local_pose[e, 2] = self.full_pose[e, 2]
        
        # 生成用于渲染的Full Map（合并所有tiles，并根据agent朝向旋转）
        self.full_map, _, self.full_map_crop_offset = self.get_full_map_for_rendering(crop_size_m=24.0, rotate_to_agent_heading=True)
        self.one_step_full_map, _, self.one_step_full_map_crop_offset = self.get_full_map_for_rendering(crop_size_m=24.0, is_one_step=True, rotate_to_agent_heading=True)
        
        if self.visualize or self.print_images:
            self._visualize(current_episode_id, 
                            id=0,
                            goal=self.goal, 
                            detected_classes=detected_classes,
                            step=step)
        
        return (self.full_map.cpu().numpy(), 
                self.full_pose.cpu().numpy(), 
                self.one_step_full_map.cpu().numpy())
    
    def _visualize(self, 
                   current_episode_id: int, 
                   id: int=0,
                   goal: Tensor=None, 
                   detected_classes: OrderedSet=None,
                   step: int=None) -> None:
        """可视化RGB图像和语义地图（Local Map + Global Map）
        
        布局：
        ┌────────────────┬──────────────┬──────────────┐
        │   RGB视图      │  Local Map   │  Global Map  │
        │   480×640      │  240×240     │  480×480     │
        └────────────────┴──────────────┴──────────────┘
        
        Args:
            id (int): 环境ID（batch中的索引）
        """
        
        # 更新检测类别
        if len(detected_classes[:-1]) > len(self.vis_classes):
            vis_classes = copy.deepcopy(self.vis_classes)
            for i in range(len(detected_classes[:-1]) - len(vis_classes)):
                self.vis_image = vu.add_class(
                    self.vis_image, 
                    5 + len(vis_classes) + i, 
                    detected_classes[i + len(vis_classes)], 
                    legend_color_palette)
                self.vis_classes.append(detected_classes[i])
        
        # ==================== 渲染 Local Map (240×240) ====================
        local_maps = self.local_map.clone()
        local_maps[:, -1, ...] = 1e-5
        local_obstacle = local_maps[id, 0, ...].cpu().numpy()
        local_explored = local_maps[id, 1, ...].cpu().numpy()
        local_semantic = local_maps[id, 4:, ...].argmax(0).cpu().numpy()
        
        # Agent世界坐标和Local Map边界
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = self.state[id]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        
        # 计算agent在Local Map中的像素位置
        agent_local_r = int(start_y * 100.0 / self.resolution - gx1)  # Y轴像素（行）
        agent_local_c = int(start_x * 100.0 / self.resolution - gy1)  # X轴像素（列）
        
        # 生成Local Map可视化
        local_map_vis = self._render_semantic_map(
            local_semantic, local_obstacle, local_explored, 
            (agent_local_r, agent_local_c), 
            (gx1, gx2, gy1, gy2),
            detected_classes
        )
        local_map_vis = cv2.resize(local_map_vis, (240, 240), interpolation=cv2.INTER_NEAREST)
        
        # 在Local Map上绘制agent箭头（居中，因为recentering后agent应该在120, 120附近）
        local_arrow_pos = (
            agent_local_c * 240.0 / local_obstacle.shape[1],
            (local_obstacle.shape[0] - agent_local_r) * 240.0 / local_obstacle.shape[0],
            np.deg2rad(-start_o)
        )
        
        # ==================== 渲染 Global Map (480×480) ====================
        # Full Map已经在update_map()中生成，以agent为中心的24m×24m区域
        if self.full_map is not None:
            global_maps = self.full_map.clone()
            global_maps[:, -1, ...] = 1e-5
            global_obstacle = global_maps[id, 0, ...].cpu().numpy()
            global_explored = global_maps[id, 1, ...].cpu().numpy()
            global_semantic = global_maps[id, 4:, ...].argmax(0).cpu().numpy()
            
            # Agent在Global Map中的位置（agent永远在中心，因为是动态裁剪的）
            global_center = global_obstacle.shape[0] // 2  # 480 // 2 = 240
            
            # 生成Global Map可视化
            global_map_vis = self._render_semantic_map(
                global_semantic, global_obstacle, global_explored,
                (global_center, global_center),  # Agent在中心
                None,  # Global Map不需要边界信息
                detected_classes
            )
            
            # Global Map上绘制agent箭头（在中心）
            global_arrow_pos = (
                global_center * 480.0 / global_obstacle.shape[1],
                (global_obstacle.shape[0] - global_center) * 480.0 / global_obstacle.shape[0],
                np.deg2rad(-start_o)
            )
        else:
            # 如果没有full_map，创建空白图
            global_map_vis = np.zeros((480, 480, 3), dtype=np.uint8)
            global_arrow_pos = None
        
        # ==================== 合成最终可视化图像 ====================
        # 布局调整为：RGB(480×640) + Local Map(240×240) + Global Map(480×480)
        self.vis_image[50:530, 15:655] = self.rgb_vis  # 左侧：RGB视图
        self.vis_image[50:290, 670:910] = local_map_vis  # 中上：Local Map (240×240)
        self.vis_image[50:530, 930:1410] = global_map_vis  # 右侧：Global Map (480×480)
        
        # 绘制Local Map的agent箭头
        if local_arrow_pos is not None:
            arrow = vu.get_contour_points(local_arrow_pos, origin=(670, 50))
            arrow_color = (int(color_palette[11] * 255),
                          int(color_palette[10] * 255),
                          int(color_palette[9] * 255))
            cv2.drawContours(self.vis_image, [arrow], 0, arrow_color, -1)
        
        # 绘制Global Map的agent箭头（永远在中心）
        if global_arrow_pos is not None:
            arrow = vu.get_contour_points(global_arrow_pos, origin=(930, 50))
            arrow_color = (int(color_palette[11] * 255),
                          int(color_palette[10] * 255),
                          int(color_palette[9] * 255))
            cv2.drawContours(self.vis_image, [arrow], 0, arrow_color, -1)
        
        # 添加标签
        cv2.putText(self.vis_image, "RGB", (15, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.vis_image, "Local Map (12m)", (670, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.vis_image, "Global Map (24m)", (930, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if self.visualize:
            # cv2.imwrite('img_debug/ref.png', self.vis_image)
            cv2.imshow("Thread 1", self.vis_image)
            cv2.waitKey(1)
            
        if self.print_images:
            result_dir = self.args.RESULTS_DIR
            save_dir = "{}/visualization/eps_{}".format(result_dir, current_episode_id)
            os.makedirs(save_dir, exist_ok=True)
            fn = "{}/step_{:04d}.png".format(save_dir, step)
            cv2.imwrite(fn, self.vis_image)
    
    def _render_semantic_map(self, semantic_map, obstacle_map, explored_map, 
                            agent_pos, lmb, detected_classes):
        """
        渲染语义地图为彩色可视化图像
        
        Args:
            semantic_map: [H, W] 语义类别索引
            obstacle_map: [H, W] 障碍物地图
            explored_map: [H, W] 已探索地图
            agent_pos: (row, col) agent位置
            lmb: Local Map边界 (gx1, gx2, gy1, gy2) or None
            detected_classes: 检测到的类别列表
            
        Returns:
            vis_map: [H, W, 3] BGR彩色图像
        """
        # 应用颜色编码
        semantic_map = semantic_map.copy()
        semantic_map += 5  # 偏移为特殊类别留空间
        
        not_cat_id = len(detected_classes) + self.MAP_CHANNELS
        not_cat_mask = (semantic_map >= not_cat_id + 5)
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        
        # 未探索区域 -> 0 (黑色)
        semantic_map[not_cat_mask] = 0
        
        # 可行走区域 -> 2 (浅色)
        free_mask = np.logical_and(not_cat_mask, explored_mask)
        semantic_map[free_mask] = 2
        
        # 障碍物 -> 1 (深色)
        obstacle_mask = np.logical_and(not_cat_mask, obstacle_mask)
        semantic_map[obstacle_mask] = 1
        
        # 如果有轨迹信息，标记轨迹 -> 3
        if lmb is not None:
            gx1, gx2, gy1, gy2 = lmb
            if hasattr(self, 'visited_vis'):
                vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
                if vis_mask.shape == semantic_map.shape:
                    semantic_map[vis_mask] = 3
        
        # 应用调色板
        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (semantic_map.shape[1], semantic_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        
        # 翻转（上下）
        sem_map_vis = np.flipud(sem_map_vis)
        # BGR格式（OpenCV）
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        
        return sem_map_vis

    def _build_explored_ray_fill_t(
        self,
        agent_view_centered_t: torch.Tensor,
        fp_exp_pred: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Approximate visible free-space by filling rays up to observed depth endpoints."""
        if not self.explored_ray_fill:
            return None
        if agent_view_centered_t is None or fp_exp_pred is None:
            return None

        try:
            xy_cells = (
                agent_view_centered_t[..., :2].detach().float().cpu().numpy()
                / float(self.resolution)
            )
            bs, _, height, width = fp_exp_pred.shape
            masks = np.zeros((bs, 1, height, width), dtype=np.float32)
            origin_x = int(round((width - 1) / 2.0))
            origin_y = 0
            min_forward_cells = max(2, int(round(0.20 * 100.0 / float(self.resolution))))

            for batch_idx in range(bs):
                xy = xy_cells[batch_idx]
                xs = np.rint(xy[..., 0]).astype(np.int32)
                ys = np.rint(xy[..., 1]).astype(np.int32)
                valid = (
                    np.isfinite(xy[..., 0])
                    & np.isfinite(xy[..., 1])
                    & (xs >= 0)
                    & (xs < width)
                    & (ys >= min_forward_cells)
                    & (ys < height)
                )
                if int(np.count_nonzero(valid)) < 4:
                    continue

                y_max = np.full((width,), -1, dtype=np.int32)
                np.maximum.at(y_max, xs[valid].reshape(-1), ys[valid].reshape(-1))

                mask_u8 = np.zeros((height, width), dtype=np.uint8)
                valid_columns = np.flatnonzero(y_max >= min_forward_cells)
                if valid_columns.size < 2:
                    continue
                max_gap_cells = max(3, int(round(width * 0.06)))
                split_points = np.flatnonzero(np.diff(valid_columns) > max_gap_cells) + 1
                column_runs = np.split(valid_columns, split_points)
                for column_run in column_runs:
                    if column_run.size < 2:
                        continue
                    boundary = [
                        (int(x), int(y_max[int(x)]))
                        for x in column_run.tolist()
                        if y_max[int(x)] >= min_forward_cells
                    ]
                    if len(boundary) < 2:
                        continue
                    points = np.asarray([(origin_x, origin_y), *boundary], dtype=np.int32)
                    cv2.fillPoly(mask_u8, [points], 255)

                if np.count_nonzero(mask_u8) == 0:
                    continue
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
                masks[batch_idx, 0] = (mask_u8 > 0).astype(np.float32)

            return torch.from_numpy(masks).to(fp_exp_pred.device)
        except Exception:
            return None

    def forward(self, obs, pose_obs, maps_last, poses_last):
        # if use CoCo the number of categories is 16(i.e. c=16), but now open-vocabulary; 
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :] # depth.shape = (bs, H, W)
        
        # cut out the needed tensor from presupposed categories dimension
        num_detected_categories = c - 4 # 4=3(RGB) + 1(Depth)
        self._dynamic_process(num_detected_categories)

        # shape: [bs, h, w, 3] 3 is (x, y, z) for each point in (h, w)
        point_cloud_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)
        
        agent_view_t = du.transform_camera_view_t(point_cloud_t, self.agent_height, 0, self.device)
        
        # point cloud in world axis
        # self.shift_loc=[250, 0, pi/2] => heading is always 90(degree), change with turn left
        # shape: [bs, h, w, 3] => (bs, 120, 160, 3)
        agent_view_centered_t = du.transform_pose_t(agent_view_t, self.shift_loc, self.device) 

        max_h = self.max_height # 72
        min_h = self.min_height # -8
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        
        # vision_range = 100(cm)
        # in sem_exp.py _preprocess_depth(), all invalid depth values are set as 100 
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float() # (bs, x, y, 3) => (bs, 120, 160, 3)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] - vision_range // 2.) / vision_range * 2. # normalize to (-1, 1)
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] - (max_h + min_h) // 2.) / (max_h - min_h) * 2. # normalize
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3]) # [bs, 3, x*y]
        
        # obs: [b, c, h*w] => [b, 17, 19200], feat is a tensor contains all predicted semantic features
        pool = nn.AvgPool2d(self.du_scale)
        # obs[:, 4, ...] = 0.
        obstacle_min_z = int(math.floor(self.obstacle_min_height_cm / z_resolution - min_h))
        obstacle_max_z = int(math.ceil(self.obstacle_max_height_cm / z_resolution - min_h))
        obstacle_min_z = max(0, min(max_h - min_h - 1, obstacle_min_z))
        obstacle_max_z = max(obstacle_min_z + 1, min(max_h - min_h, obstacle_max_z))
        self.min_z = obstacle_min_z
        self.feat[:, 1:, :] = pool(obs[:, 4:, :, :]).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        # self.init_grid: [bs, categories + 1, x=vr, y=vr, z=(max_height - min_height)] => [bs, 17, 100, 100, 80]
        # feat: average of all categories's predicted semantic features, [bs, 17, 19200]
        # XYZ_cm_std: point cloud in physical world, [bs, 3, 19200]
        # splat_feat_nd:
        assert self.init_grid.shape[1] == self.feat.shape[1], "init_grid and feat should have same number of channels!"
        
        # shape: [bs, num_detected_classes + 1, 100, 100, 80]
        voxels = du.splat_feat_nd(self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)
        agent_height_proj = voxels[..., obstacle_min_z:obstacle_max_z].sum(4) # shape: [bs, num_detected_classes + 1, 100, 100]
        all_height_proj = voxels.sum(4) # shape: [bs, num_detected_classes + 1, 100, 100]

        fp_map_pred = agent_height_proj[:, :1, :, :] # obstacle map（仅取agent高度附近）
        fp_exp_pred = all_height_proj[:, :1, :, :]   # explored map（所有高度）
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)
        ray_exp_pred = self._build_explored_ray_fill_t(agent_view_centered_t, fp_exp_pred)
        if ray_exp_pred is not None:
            fp_exp_pred = torch.maximum(fp_exp_pred, ray_exp_pred)

        pose_pred = self.local_pose

        agent_view = torch.zeros(bs, self.local_map.shape[1],
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device) # (bs, c, 480, 480) => full_map

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred # obstacle map
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred # explored area
        # 语义类别用 all_height_proj（不限高度）：障碍物只需关心通行高度，
        # 但 landmark（挂画、窗户等）可能在任意高度，用 all_height_proj 确保不漏掉
        n_mapping = 15  # mapping_classes 固定15个通道
        n_sem_total = all_height_proj.shape[1] - 1  # 去掉占位通道0
        # mapping_classes 用标准 threshold
        agent_view[:, 3:3+n_mapping, y1:y2, x1:x2] = torch.clamp(
            all_height_proj[:, 1:1+n_mapping, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)
        # landmark_classes 用更低 threshold（mask稀疏，标准threshold=5.0会全部过滤掉）
        if n_sem_total > n_mapping:
            lm_threshold = float(SEM_MAP_LANDMARK_THRESH)
            agent_view[:, 3+n_mapping:, y1:y2, x1:x2] = torch.clamp(
                all_height_proj[:, 1+n_mapping:, :, :] / lm_threshold,
                min=0.0, max=1.0)

        # [LM-DBG 2/3 PROJECT]
        _n_lm = agent_view.shape[1] - 3 - 15
        for _lm_ch in range(_n_lm):
            _ch = agent_view[0, 3 + 15 + _lm_ch, y1:y2, x1:x2]

        corrected_pose = pose_obs # sensor pose

        def get_new_pose_batch(pose, rel_pose_change):
            # pose: (bs, 3) -> x, y, ori(degree)
            # 57.29577951308232 = 180 / pi
            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose
        
        current_poses = get_new_pose_batch(self.local_pose, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        # get rotation matrix and translation matrix according to new pose (x, y, theta(degree))
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True) # shape: [bs, c, 240, 240]
        map_pred = torch.maximum(self.local_map, translated)
        one_step_map_pred = torch.maximum(self.one_step_local_map, translated)

        new_obstacle = translated[:, 0:1, :, :]
        new_explored = translated[:, 1:2, :, :]
        if self.selective_dynamic_obstacle_update:
            obstacle_observed = new_obstacle > 0.0
            free_observed = (
                (new_explored - new_obstacle)
                >= float(self.obstacle_clear_explored_threshold)
            )
            evidence_mask = obstacle_observed | free_observed
            if torch.any(evidence_mask):
                current_counts = torch.cat(
                    [
                        self._get_local_obstacle_evidence_counts(env_id=env_id)
                        for env_id in range(new_obstacle.shape[0])
                    ],
                    dim=0,
                )
                current_scores = torch.cat(
                    [
                        self._get_local_obstacle_evidence_scores(env_id=env_id)
                        for env_id in range(new_obstacle.shape[0])
                    ],
                    dim=0,
                )
                evidence_value = torch.where(
                    obstacle_observed,
                    torch.ones_like(new_obstacle),
                    torch.zeros_like(new_obstacle),
                )
                updated_counts = current_counts + evidence_mask.float()
                updated_scores = current_scores + evidence_value
                if self.obstacle_evidence_max_observations > 0:
                    max_observations = float(self.obstacle_evidence_max_observations)
                    scale = torch.where(
                        updated_counts > max_observations,
                        max_observations / torch.clamp(updated_counts, min=1.0),
                        torch.ones_like(updated_counts),
                    )
                    updated_counts = updated_counts * scale
                    updated_scores = updated_scores * scale
                averaged_obstacle = updated_scores / torch.clamp(updated_counts, min=1.0)
                thresholded_obstacle = torch.where(
                    averaged_obstacle >= float(self.obstacle_evidence_threshold),
                    averaged_obstacle,
                    torch.zeros_like(averaged_obstacle),
                )
                map_pred[:, 0:1, :, :] = torch.where(
                    evidence_mask,
                    thresholded_obstacle,
                    self.local_map[:, 0:1, :, :],
                )
                one_step_map_pred[:, 0:1, :, :] = torch.where(
                    evidence_mask,
                    thresholded_obstacle,
                    one_step_map_pred[:, 0:1, :, :],
                )
                for env_id in range(updated_counts.shape[0]):
                    self._update_obstacle_evidence_tiles(
                        updated_counts[env_id:env_id + 1],
                        updated_scores[env_id:env_id + 1],
                        env_id=env_id,
                    )
        else:
            clear_mask = (new_explored - new_obstacle) >= float(self.obstacle_clear_explored_threshold)
            if torch.any(clear_mask):
                map_pred[:, 0:1, :, :] = torch.where(clear_mask, new_obstacle, map_pred[:, 0:1, :, :])
                one_step_map_pred[:, 0:1, :, :] = torch.where(
                    clear_mask,
                    new_obstacle,
                    one_step_map_pred[:, 0:1, :, :],
                )

        self.local_map = map_pred
        self.one_step_local_map = one_step_map_pred
        self.local_pose = current_poses

        # return fp_map_pred, map_pred, pose_pred, current_poses
