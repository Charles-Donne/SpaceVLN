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
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import habitat_extensions.pose_utils as pu

from vlnce_baselines.config_system.constants import *
from vlnce_baselines.mapping.map_utils import *
import vlnce_baselines.mapping.depth_utils as du
import vlnce_baselines.visualization.rendering as vu

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
    
    坐标系统：
    - 世界坐标：agent初始位置为(0, 0)，单位：米，可以是负值
    - 块索引：(tile_x, tile_y)，可以是负值，如(-1, 0)表示左边的块
    - 块内坐标：(local_x, local_y)，范围[0, 239]，单位：像素
    
    Map结构：
    1. Obstacle Map (通道0)
    2. Explored Area (通道1)
    3. Current Agent Location (通道2)
    4. Past Agent Locations (通道3)
    5. Semantic Categories (通道4+，动态扩展)
    """
    MAP_CHANNELS = map_channels
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
        
        # 分块地图：{(tile_x, tile_y): tensor[C, 240, 240]}
        self.tiles = defaultdict(lambda: None)
        
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
        
        # 当前Local Map和Full Map（用于兼容旧接口）
        self.local_map = None
        self.one_step_local_map = None
        self.full_map = None
        self.one_step_full_map = None
        
        if self.visualize or self.print_images:
            self.vis_image = vu.init_vis_image()
            self.rgb_vis = None

        # 高度参数
        self.max_height = int(360 / self.z_resolution)  # 72
        self.min_height = int(-40 / self.z_resolution)  # -8
        self.agent_height = args.AGENT_HEIGHT * 100.  # 88cm
        self.shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, self.fov)

        # Feat通道
        self.feat = torch.ones(
            args.NUM_ENVIRONMENTS, 1, 
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)
    
    def reset(self) -> None:
        """重置地图系统"""
        self.curr_loc = None
        self.last_loc = None
        self.vis_classes = []
        self.tiles.clear()
        
        # 重置agent全局坐标到原点
        self.agent_global_x = 0.0
        self.agent_global_y = 0.0
        self.agent_orientation = 0.0
        
        # 重置pose
        self.local_pose.fill_(0.)
        self.full_pose.fill_(0.)
        self.curr_loc.fill_(0.)
        
        # 重置origins和lmb
        self.origins.fill(0.)
        self.lmb.fill(0.)
        self.state.fill(0.)
        
        # 清空local_map和full_map
        self.local_map = None
        self.one_step_local_map = None
        self.full_map = None
        self.one_step_full_map = None
        
        self.feat = torch.ones(
            self.args.NUM_ENVIRONMENTS, 1, 
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)
        
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
    
    def _ensure_tiles_exist(self, tile_indices, nc):
        """确保指定的块存在"""
        for (tile_x, tile_y) in tile_indices:
            if self.tiles[(tile_x, tile_y)] is None:
                self.tiles[(tile_x, tile_y)] = self._create_empty_tile(nc)
                print(f"🆕 创建新块: tile({tile_x}, {tile_y}) = 世界坐标({tile_x*12:.0f}m, {tile_y*12:.0f}m)")
    
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
        tile_x_min = int(np.floor(min_x / self.TILE_SIZE_M))
        tile_x_max = int(np.floor(max_x / self.TILE_SIZE_M))
        tile_y_min = int(np.floor(min_y / self.TILE_SIZE_M))
        tile_y_max = int(np.floor(max_y / self.TILE_SIZE_M))
        
        # 生成所有需要的块
        tiles = [
            (tx, ty)
            for tx in range(tile_x_min, tile_x_max + 1)
            for ty in range(tile_y_min, tile_y_max + 1)
        ]
        
        return tiles
    
    def get_local_map_from_tiles(self, nc):
        """
        从块中拼接Local Map（240×240，以agent为中心）
        
        Returns:
            local_map: [batch, C, 240, 240]
        """
        # 计算agent世界坐标（米）
        agent_x_m = self.full_pose[0, 0].item()
        agent_y_m = self.full_pose[0, 1].item()
        
        # 获取需要的块
        tiles_needed = self._get_tiles_for_region(agent_x_m, agent_y_m, self.TILE_SIZE_M)
        self._ensure_tiles_exist(tiles_needed, nc)
        
        # 创建Local Map
        local_map = torch.zeros(
            self.num_environments, nc,
            self.TILE_SIZE, self.TILE_SIZE
        ).float().to(self.device)
        
        # Local Map的世界像素范围
        half_size_px = self.TILE_SIZE // 2
        agent_px = int(agent_x_m * 100 / self.resolution)
        agent_py = int(agent_y_m * 100 / self.resolution)
        
        start_px = agent_px - half_size_px
        end_px = agent_px + half_size_px
        start_py = agent_py - half_size_px
        end_py = agent_py + half_size_px
        
        # 从各个块中提取数据
        for (tile_x, tile_y) in tiles_needed:
            tile = self.tiles[(tile_x, tile_y)]
            if tile is None:
                continue
            
            # 块的世界像素范围
            tile_start_px = tile_x * self.TILE_SIZE
            tile_end_px = tile_start_px + self.TILE_SIZE
            tile_start_py = tile_y * self.TILE_SIZE
            tile_end_py = tile_start_py + self.TILE_SIZE
            
            # 计算交集
            copy_start_px = max(start_px, tile_start_px)
            copy_end_px = min(end_px, tile_end_px)
            copy_start_py = max(start_py, tile_start_py)
            copy_end_py = min(end_py, tile_end_py)
            
            if copy_start_px >= copy_end_px or copy_start_py >= copy_end_py:
                continue
            
            # 块内坐标
            tile_x_start = copy_start_px - tile_start_px
            tile_x_end = copy_end_px - tile_start_px
            tile_y_start = copy_start_py - tile_start_py
            tile_y_end = copy_end_py - tile_start_py
            
            # Local Map坐标
            local_x_start = copy_start_px - start_px
            local_x_end = copy_end_px - start_px
            local_y_start = copy_start_py - start_py
            local_y_end = copy_end_py - start_py
            
            # 复制数据
            local_map[:, :,
                     local_x_start:local_x_end,
                     local_y_start:local_y_end] = \
                tile[:, :,
                     tile_x_start:tile_x_end,
                     tile_y_start:tile_y_end]
        
        # 更新lmb（Local Map在世界坐标系中的像素边界）
        for e in range(self.num_environments):
            self.lmb[e] = [start_px, end_px, start_py, end_py]
            # 更新origins（Local Map左上角的世界坐标）
            self.origins[e] = [
                start_py * self.resolution / 100.0,  # Y方向（米）
                start_px * self.resolution / 100.0,  # X方向（米）
                0.0
            ]
        
        return local_map
    
    def update_tiles_from_local_map(self, local_map):
        """
        将更新后的Local Map写回到块中
        
        Args:
            local_map: [batch, C, 240, 240]
        """
        # Local Map的世界像素范围（从lmb获取）
        start_px, end_px, start_py, end_py = self.lmb[0]
        
        # 计算涉及的块
        tile_x_min = start_px // self.TILE_SIZE
        tile_x_max = (end_px - 1) // self.TILE_SIZE
        tile_y_min = start_py // self.TILE_SIZE
        tile_y_max = (end_py - 1) // self.TILE_SIZE
        
        # 写回数据到各个块
        for tile_x in range(tile_x_min, tile_x_max + 1):
            for tile_y in range(tile_y_min, tile_y_max + 1):
                tile = self.tiles[(tile_x, tile_y)]
                if tile is None:
                    continue
                
                # 块的世界像素范围
                tile_start_px = tile_x * self.TILE_SIZE
                tile_end_px = tile_start_px + self.TILE_SIZE
                tile_start_py = tile_y * self.TILE_SIZE
                tile_end_py = tile_start_py + self.TILE_SIZE
                
                # 计算交集
                copy_start_px = max(start_px, tile_start_px)
                copy_end_px = min(end_px, tile_end_px)
                copy_start_py = max(start_py, tile_start_py)
                copy_end_py = min(end_py, tile_end_py)
                
                if copy_start_px >= copy_end_px or copy_start_py >= copy_end_py:
                    continue
                
                # 块内坐标
                tile_x_start = copy_start_px - tile_start_px
                tile_x_end = copy_end_px - tile_start_px
                tile_y_start = copy_start_py - tile_start_py
                tile_y_end = copy_end_py - tile_start_py
                
                # Local Map坐标
                local_x_start = copy_start_px - start_px
                local_x_end = copy_end_px - start_px
                local_y_start = copy_start_py - start_py
                local_y_end = copy_end_py - start_py
                
                # 写回数据
                tile[:, :,
                     tile_x_start:tile_x_end,
                     tile_y_start:tile_y_end] = \
                    local_map[:, :,
                             local_x_start:local_x_end,
                             local_y_start:local_y_end]
    
    def get_full_map_for_rendering(self, crop_size_m=24.0):
        """
        获取用于渲染的全局地图（以agent为中心裁剪）
        
        Args:
            crop_size_m: 裁剪尺寸（米），默认24m×24m
        
        Returns:
            full_map: [batch, C, H, W]
            map_size: (H, W) 实际尺寸
        """
        # 获取需要的块
        agent_x_m = self.full_pose[0, 0].item()
        agent_y_m = self.full_pose[0, 1].item()
        
        tiles_needed = self._get_tiles_for_region(agent_x_m, agent_y_m, crop_size_m)
        
        if not tiles_needed:
            # 没有任何块，返回空地图
            nc = self.MAP_CHANNELS + 1
            map_size_px = int(crop_size_m * 100 / self.resolution)
            return torch.zeros(self.num_environments, nc, map_size_px, map_size_px).float().to(self.device), (map_size_px, map_size_px)
        
        nc = self.tiles[tiles_needed[0]].shape[1] if self.tiles[tiles_needed[0]] is not None else self.MAP_CHANNELS + 1
        self._ensure_tiles_exist(tiles_needed, nc)
        
        # 计算裁剪区域的世界像素范围
        crop_size_px = int(crop_size_m * 100 / self.resolution)
        half_crop = crop_size_px // 2
        
        agent_px = int(agent_x_m * 100 / self.resolution)
        agent_py = int(agent_y_m * 100 / self.resolution)
        
        start_px = agent_px - half_crop
        end_px = agent_px + half_crop
        start_py = agent_py - half_crop
        end_py = agent_py + half_crop
        
        # 创建Full Map
        full_map = torch.zeros(
            self.num_environments, nc,
            crop_size_px, crop_size_px
        ).float().to(self.device)
        
        # 从各个块中拼接
        for (tile_x, tile_y) in tiles_needed:
            tile = self.tiles[(tile_x, tile_y)]
            if tile is None:
                continue
            
            # 块的世界像素范围
            tile_start_px = tile_x * self.TILE_SIZE
            tile_end_px = tile_start_px + self.TILE_SIZE
            tile_start_py = tile_y * self.TILE_SIZE
            tile_end_py = tile_start_py + self.TILE_SIZE
            
            # 计算交集
            copy_start_px = max(start_px, tile_start_px)
            copy_end_px = min(end_px, tile_end_px)
            copy_start_py = max(start_py, tile_start_py)
            copy_end_py = min(end_py, tile_end_py)
            
            if copy_start_px >= copy_end_px or copy_start_py >= copy_end_py:
                continue
            
            # 块内坐标
            tile_x_start = copy_start_px - tile_start_px
            tile_x_end = copy_end_px - tile_start_px
            tile_y_start = copy_start_py - tile_start_py
            tile_y_end = copy_end_py - tile_start_py
            
            # Full Map坐标
            full_x_start = copy_start_px - start_px
            full_x_end = copy_end_px - start_px
            full_y_start = copy_start_py - start_py
            full_y_end = copy_end_py - start_py
            
            # 复制数据
            full_map[:, :,
                    full_x_start:full_x_end,
                    full_y_start:full_y_end] = \
                tile[:, :,
                     tile_x_start:tile_x_end,
                     tile_y_start:tile_y_end]
        
        # 更新full_w和full_h（用于兼容旧代码）
        self.full_w = crop_size_px
        self.full_h = crop_size_px
        
        return full_map, (crop_size_px, crop_size_px)
    
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
        
    def init_map_and_pose(self, num_detected_classes: int):
        """
        初始化分块地图和agent姿态
        
        新设计：
        1. Agent初始世界坐标为(0, 0)，不再是(12, 12)
        2. 创建初始块tile(0, 0)，覆盖世界坐标[0, 12)m × [0, 12)m
        3. Agent在块中心(6m, 6m)
        4. Local Map以agent为中心，范围[-6, 6)m × [-6, 6)m
        """
        nc = num_detected_classes + self.MAP_CHANNELS
        
        # Agent初始世界坐标：(0, 0, 0)
        self.full_pose.fill_(0.)
        self.full_pose[:, 0] = 6.0  # X方向 = 6m（块中心）
        self.full_pose[:, 1] = 6.0  # Y方向 = 6m（块中心）
        self.full_pose[:, 2] = 0.0  # 朝向东方
        
        # 创建初始块tile(0, 0)
        self._ensure_tiles_exist([(0, 0)], nc)
        
        # 在初始块中标记agent位置（中心120, 120）
        for e in range(self.num_environments):
            tile = self.tiles[(0, 0)]
            tile[e, 2:4, 119:122, 119:122] = 1.0  # Current & Past location
        
        # 获取Local Map（240×240，以agent为中心）
        self.local_map = self.get_local_map_from_tiles(nc)
        self.one_step_local_map = self.local_map.clone()
        
        # Local Pose：agent相对Local Map的位置
        # 因为Local Map以agent为中心，所以agent在Local Map中心(6m, 6m)
        for e in range(self.num_environments):
            self.local_pose[e] = torch.tensor([6.0, 6.0, 0.0]).float().to(self.device)
            self.curr_loc[e] = self.full_pose[e].clone()
        
        # 更新state
        locs = self.full_pose.cpu().numpy()
        self.state[:, :3] = locs
        self.state[:, 3:] = self.lmb[0]  # [gx1, gx2, gy1, gy2]
        
        # 获取用于渲染的Full Map
        self.full_map, _ = self.get_full_map_for_rendering(crop_size_m=24.0)
        self.one_step_full_map = self.full_map.clone()
        
        print(f"✅ 地图初始化完成: Agent@世界(6.0m, 6.0m) = tile(0,0)@块内(120px, 120px)")
                                
    def update_map(self, step: int, detected_classes: OrderedSet, current_episode_id: int) -> None:
        """
        更新分块地图
        
        步骤：
        1. 更新agent在Local Map中的位置标记
        2. 将Local Map写回到对应的tiles
        3. 更新full_pose（世界坐标）
        4. 每CENTER_RESET_STEPS步执行recentering：重新获取Local Map
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
        
        # 清除Local Map中的当前位置标记
        self.local_map[:, 2, :, :].fill_(0.)
        self.one_step_local_map[:, 2, :, :].fill_(0.)
        
        # 标记agent当前位置
        for e in range(self.num_environments):
            r, c = locs[e, 1], locs[e, 0]  # r=Y, c=X（物理坐标，米）
            # 转换为Local Map像素坐标（240×240）
            loc_r, loc_c = [int(r * 100.0 / self.resolution),
                            int(c * 100.0 / self.resolution)]
            # 在Local Map中标记agent位置（3×3像素）
            self.local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
            self.one_step_local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
            
            # 将更新后的Local Map写回到对应的tiles
            self.update_tiles_from_local_map(self.local_map[e], self.full_pose[e])
            self.update_tiles_from_local_map(self.one_step_local_map[e], self.full_pose[e], is_one_step=True)
        
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
                
                # 更新local_pose：agent相对于新Local Map origin的位置
                # 因为重新获取的Local Map总是以agent为中心，所以local_pose应该是(6, 6, ori)
                self.local_pose[e, 0] = 6.0
                self.local_pose[e, 1] = 6.0
                # 保持原有朝向
                self.local_pose[e, 2] = self.full_pose[e, 2]
        
        # 生成用于渲染的Full Map（合并所有tiles）
        self.full_map, _ = self.get_full_map_for_rendering(crop_size_m=24.0)
        self.one_step_full_map, _ = self.get_full_map_for_rendering(crop_size_m=24.0, is_one_step=True)
        
        if self.visualize or self.print_images:
            self._visualize(current_episode_id, 
                            id=0,
                            goal=self.goal, 
                            detected_classes=detected_classes,
                            step=step)
        
        return (self.full_map.cpu().numpy(), 
                self.full_pose.cpu().numpy(), 
                # frontiers, 
                self.one_step_full_map.cpu().numpy())
    
    def _visualize(self, 
                   current_episode_id: int, 
                   id: int=0,
                   goal: Tensor=None, 
                   detected_classes: OrderedSet=None,
                   step: int=None) -> None:
        """Try to visualize RGB images with segmentation and semantic map

        Args:
            id (int): since we are running a batch of environments, 
            it's resource consuming to render all environments together,
            so please only choose one environmet to visualize.
        """
        
        # the last item of detected_class is always "not_a_cat"
        if len(detected_classes[:-1]) > len(self.vis_classes):
            vis_classes = copy.deepcopy(self.vis_classes)
            for i in range(len(detected_classes[:-1]) - len(vis_classes)):
                self.vis_image = vu.add_class(
                    self.vis_image, 
                    5 + len(vis_classes) + i, 
                    detected_classes[i + len(vis_classes)], 
                    legend_color_palette)
                self.vis_classes.append(detected_classes[i])
        
        local_maps = self.local_map.clone()
        local_maps[:, -1, ...] = 1e-5
        obstacle_map = local_maps[id, 0, ...].cpu().numpy()
        explored_map = local_maps[id, 1, ...].cpu().numpy()
        semantic_map = local_maps[id, 4:, ...].argmax(0).cpu().numpy()
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = self.state[id]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        r, c = start_y, start_x
        start = [int(r * 100.0 / self.resolution - gx1),
                 int(c * 100.0 / self.resolution - gy1)] # get agent's location in local map
        start = pu.threshold_poses(start, obstacle_map.shape)
        
        last_start_x, last_start_y = self.last_loc[id][0], self.last_loc[id][1]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / self.resolution - gx1),
                        int(c * 100.0 / self.resolution - gy1)]
        last_start = pu.threshold_poses(last_start, obstacle_map.shape)
        self.visited_vis[gx1:gx2, gy1:gy2] = vu.draw_line(last_start, start, self.visited_vis[gx1:gx2, gy1:gy2])
        
        """
        color palette:
        0: out of map
        1: obstacles
        2: agent trajectory
        3: goal
        4 ~ num_detected_class: detected objects
        """
        semantic_map += 5
        not_cat_id = local_maps.shape[1]
        not_cat_mask = (semantic_map == not_cat_id)
        obstacle_map_mask = np.rint(obstacle_map) == 1
        explored_map_mask = np.rint(explored_map) == 1
        
        semantic_map[not_cat_mask] = 0
        
        m_free = np.logical_and(not_cat_mask, explored_map_mask)
        semantic_map[m_free] = 2
        
        m_obstacle = np.logical_and(not_cat_mask, obstacle_map_mask)
        semantic_map[m_obstacle] = 1
        
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        semantic_map[vis_mask] = 3
        color_pal = [int(x * 255.) for x in color_palette]
        
        # create a new image using palette mode
        # (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes)
        # in this mode, we can map colors to picture use a color palette
        sem_map_vis = Image.new("P", (semantic_map.shape[1], semantic_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        
        # put the flattened data, so that each instance will be mapped a color according to color palette
        sem_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        
        # flip image up and down, so that agnet's turn in simulator 
        # is the same as its turn in semantic map visualization
        sem_map_vis = np.flipud(sem_map_vis)
        # sem_map_vis = np.array(sem_map_vis)
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]] # turn to bgr for opencv
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 15:655] = self.rgb_vis # 480, 640
        self.vis_image[50:530, 670:1150] = sem_map_vis # 480, 480
        
        pos = (
            (start_x * 100. / self.resolution - gy1) * 480 / obstacle_map.shape[0],
            (obstacle_map.shape[1] - start_y * 100. / self.resolution + gx1) * 480 / obstacle_map.shape[1],
            np.deg2rad(-start_o)
        )
        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        cv2.waitKey(1)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1) # draw agent arrow
        
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

    def forward(self, obs: torch.Tensor, pose_obs: torch.Tensor):
        """
        Args:
            obs: (b, c, h, w), b = batch size, c = 3(RGB) + 1(Depth) + num_detected_categories
        """
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
        self.min_z = int(25 / z_resolution - min_h) # 25 / 5 - (-8) = 13
        # self.min_z = 2 # use grounded-sam to detect floor
        self.feat[:, 1:, :] = pool(obs[:, 4:, :, :]).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        # self.init_grid: [bs, categories + 1, x=vr, y=vr, z=(max_height - min_height)] => [bs, 17, 100, 100, 80]
        # feat: average of all categories's predicted semantic features, [bs, 17, 19200]
        # XYZ_cm_std: point cloud in physical world, [bs, 3, 19200]
        # splat_feat_nd:
        assert self.init_grid.shape[1] == self.feat.shape[1], "init_grid and feat should have same number of channels!"
        
        # shape: [bs, num_detected_classes + 1, 100, 100, 80]
        voxels = du.splat_feat_nd(self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)
        max_z = int((self.agent_height + 1) / z_resolution - min_h) # int((88 + 1) / 5 - (-8))= 25
        
        agent_height_proj = voxels[..., self.min_z:max_z].sum(4) # shape: [bs, num_detected_classes + 1, 100, 100]
        all_height_proj = voxels.sum(4) # shape: [bs, num_detected_classes + 1, 100, 100]

        fp_map_pred = agent_height_proj[:, :1, :, :] # obstacle map
        fp_exp_pred = all_height_proj[:, :1, :, :] # explored map
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

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
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0) # semantic categories

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
        maps2 = torch.cat((self.local_map.unsqueeze(1), translated.unsqueeze(1)), 1)
        one_step_maps2 = torch.cat((self.one_step_local_map.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)
        one_step_map_pred, _ = torch.max(one_step_maps2, 1)
        self.local_map = map_pred
        self.one_step_local_map = one_step_map_pred
        self.local_pose = current_poses

        # return fp_map_pred, map_pred, pose_pred, current_poses