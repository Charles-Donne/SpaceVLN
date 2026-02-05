# RGBD到全局地图渲染的工作流程详解

## 📋 目录
1. [系统概述](#系统概述)
2. [核心坐标系统](#核心坐标系统)
3. [RGBD投影流程](#rgbd投影流程)
4. [地图重心化机制](#地图重心化机制)
5. [边界处理与坐标变换](#边界处理与坐标变换)
6. [可视化渲染流程](#可视化渲染流程)

---

## 系统概述

MapReAct-VLN系统使用**双层地图架构**来管理导航空间：
- **Full Map (全局地图)**: 480×480像素 (24m×24m @ 5cm分辨率)
- **Local Map (局部地图)**: 240×240像素 (12m×12m)

地图包含多个通道：
```python
Channel 0: Obstacle (障碍物)
Channel 1: Explored (已探索区域)
Channel 2: Current Location (当前位置)
Channel 3: Past Locations (历史位置)
Channel 4+: Semantic Categories (语义类别，动态扩展)
```

---

## 核心坐标系统

### 1. Habitat坐标系 (Simulator原生)
```
Y轴 (高度)
↑
|
|———→ X轴 (右)
/
Z轴 (前)
```

### 2. 地图坐标系 (Map Pixel Space)
```
map_x (行)
  ↓
  0 ————————→ map_y (列)
  |            479
  |
  |
  479
```
- `map_x`: 对应地图的**行索引** (0=顶部, 479=底部)
- `map_y`: 对应地图的**列索引** (0=左侧, 479=右侧)

### 3. 物理世界坐标系 (Meter Space)
```
   Y (北)
   ↑
   |
   |————→ X (东)
(0,0) = Agent起始位置
```

- Agent初始化时位于全局地图**中心** `(12m, 12m)`
- 朝向东方 (`orientation = 0°`)

---

## RGBD投影流程

### 步骤1: Depth → 3D点云 (Camera Space)

```python
# depth_utils.py: get_point_cloud_from_z_t()

# 输入: depth图像 [batch, H, W]  (H=224, W=224)
# 输出: 点云 [batch, H, W, 3]  (X, Y, Z)

# 相机矩阵参数
camera_matrix = {
    'xc': (W - 1) / 2,      # 中心x = 111.5
    'zc': (H - 1) / 2,      # 中心z = 111.5
    'f': (W / 2) / tan(HFOV / 2)  # 焦距
}

# 投影公式
X = (grid_x - xc) * depth / f
Y = depth
Z = (grid_z - zc) * depth / f
```

**坐标系定义**:
- X轴: 正方向向右
- Y轴: 正方向向前 (深度方向)
- Z轴: 正方向向上

### 步骤2: Camera Space → Agent Space

```python
# depth_utils.py: transform_camera_view_t()

# 旋转矩阵校正相机俯仰角 (camera_elevation_degree)
R = rotation_matrix([1, 0, 0], angle=deg2rad(camera_elevation))
XYZ = matmul(XYZ, R.T)

# 平移到agent高度
XYZ[..., 2] += agent_height  # 88cm
```

### 步骤3: Agent Space → World Space

```python
# depth_utils.py: transform_pose_t()

shift_loc = [250cm, 0, π/2]  # 固定偏移量

# 1. 旋转: 对齐agent朝向
R = rotation_matrix([0, 0, 1], angle=shift_loc[2] - π/2)
XYZ = matmul(XYZ, R.T)

# 2. 平移: 对齐agent位置
XYZ[:, :, 0] += shift_loc[0]  # X方向
XYZ[:, :, 1] += shift_loc[1]  # Y方向
```

### 步骤4: 3D Voxel化

```python
# semantic_mapping.py: forward()

# 归一化到[-1, 1]范围
XYZ_cm_std = (XYZ / xy_resolution - vision_range/2) / (vision_range/2) * 2

# 3D Voxel空间: [batch, C, X, Y, Z]
# X, Y: 100像素 (视野范围 5m @ 5cm分辨率)
# Z: 80层 (高度范围 -40cm到360cm @ 5cm分辨率)

voxels = splat_feat_nd(init_grid, feat, XYZ_cm_std)
```

### 步骤5: 高度投影 (Voxel → 2D Map)

```python
# 障碍物投影: agent高度到地面的voxel
min_z = 25 / z_resolution - min_height = 13
max_z = (agent_height + 1) / z_resolution - min_height = 25
agent_height_proj = voxels[..., min_z:max_z].sum(dim=4)

# 探索区域投影: 所有高度的voxel
all_height_proj = voxels.sum(dim=4)

# 阈值化
fp_map_pred = agent_height_proj[:, :1, :, :] / map_pred_threshold
fp_exp_pred = all_height_proj[:, :1, :, :] / exp_pred_threshold
fp_map_pred = clamp(fp_map_pred, 0.0, 1.0)
fp_exp_pred = clamp(fp_exp_pred, 0.0, 1.0)
```

---

## 地图重心化机制

### 触发条件
```python
if (step + 1) % CENTER_RESET_STEPS == 0:
    recenter_map()
```

`CENTER_RESET_STEPS`默认值通常为25-50步，防止agent移出局部地图范围。

### 重心化步骤

#### 1. 更新Full Pose (全局位姿)
```python
# 将local_map的最新数据写回full_map
full_map[:, lmb[0]:lmb[1], lmb[2]:lmb[3]] = local_map

# 更新agent的全局位置
full_pose = local_pose + origins
```

#### 2. 重新计算Local Map边界
```python
# Agent在Full Map中的像素坐标
loc_r = int(full_pose[1] * 100 / resolution)  # X方向
loc_c = int(full_pose[0] * 100 / resolution)  # Y方向

# 新的Local Map边界 (以agent为中心)
gx1 = loc_r - local_w // 2  # 240 // 2 = 120
gx2 = gx1 + local_w         # 120 + 240 = 360
gy1 = loc_c - local_h // 2
gy2 = gy1 + local_h

# 边界裁剪 (防止超出Full Map)
if gx1 < 0:
    gx1, gx2 = 0, local_w
if gx2 > full_w:
    gx1, gx2 = full_w - local_w, full_w

if gy1 < 0:
    gy1, gy2 = 0, local_h
if gy2 > full_h:
    gy1, gy2 = full_h - local_h, full_h
```

#### 3. 更新Origins (局部地图原点)
```python
# Origins定义Local Map左上角在世界坐标系中的位置
origins = [
    lmb[2] * resolution / 100.0,  # Y方向 (米)
    lmb[0] * resolution / 100.0,  # X方向 (米)
    0.0                           # Z方向 (始终为0)
]
```

#### 4. 提取新的Local Map
```python
# 从Full Map中提取以agent为中心的局部区域
local_map = full_map[:, lmb[0]:lmb[1], lmb[2]:lmb[3]]

# 更新Local Pose (相对于新的origin)
local_pose = full_pose - origins
```

### 坐标转换关系

**Full Pose ↔ Local Pose 转换**:
```python
# Local → Full
full_pose = local_pose + origins

# Full → Local  
local_pose = full_pose - origins
```

**物理坐标 → 地图像素**:
```python
# 米 → 像素
pixel_x = int(meter_y * 100 / resolution)  # 注意: meter_y → pixel_x
pixel_y = int(meter_x * 100 / resolution)  # 注意: meter_x → pixel_y

# 像素 → 米
meter_x = pixel_y * resolution / 100.0
meter_y = pixel_x * resolution / 100.0
```

---

## 边界处理与坐标变换

### 1. Agent视野投影到Full Map

```python
# semantic_mapping.py: forward()

# 创建480×480的agent_view
agent_view = zeros(batch, C, 480, 480)

# 将100×100的voxel投影放在agent_view中心
x1 = 480 // 2 - 100 // 2 = 190
x2 = x1 + 100 = 290
y1 = 480 // 2 = 240
y2 = y1 + 100 = 340

agent_view[:, :, y1:y2, x1:x2] = projected_voxels
```

### 2. 旋转和平移 (对齐agent朝向和位置)

```python
# 计算agent当前姿态的变换矩阵
st_pose = current_poses.clone()

# 归一化到[-1, 1]坐标系
st_pose[:, :2] = -(st_pose[:, :2] * 100 / resolution - 240) / 240
st_pose[:, 2] = 90 - st_pose[:, 2]  # 角度反转

# 生成grid_sample的变换矩阵
rot_mat, trans_mat = get_grid(st_pose, agent_view.size())

# 应用变换
rotated = F.grid_sample(agent_view, rot_mat)
translated = F.grid_sample(rotated, trans_mat)
```

### 3. 与Local Map融合

```python
# 最大值融合 (保留历史最大值)
maps2 = cat([local_map.unsqueeze(1), translated.unsqueeze(1)], dim=1)
local_map, _ = torch.max(maps2, dim=1)
```

### 超出边界的处理

**Full Map边界保护**:
- Full Map固定480×480，**不会动态扩展**
- 当agent接近边界时，Local Map会自动贴边
- 超出Full Map的观测数据会被**截断丢弃**

**示例**:
```python
# Agent移动到Full Map右下角 (23m, 23m)
full_pose = [23, 23, 0]  # 米
loc_r = 23 * 100 / 5 = 460  # 像素

# 计算Local Map边界
gx1 = 460 - 120 = 340
gx2 = 340 + 240 = 580  # 超出480!

# 边界修正
gx1, gx2 = 480 - 240, 480  # [240, 480]

# 此时Local Map贴在Full Map右边界
# 超出Full Map右侧的观测数据会被丢弃
```

---

## 可视化渲染流程

### 1. Global Map渲染 (visualizer.py)

```python
def render_global_map(full_map, trajectory_points, ...):
    # 阶段1: 创建语义底图 (480×480)
    obstacle_map = full_map[0, ...]  # 障碍物
    explored_map = full_map[1, ...]  # 已探索区域
    semantic_map = argmax(full_map[4:, ...])  # 语义类别
    
    # 渲染顺序 (后面的覆盖前面的):
    # 1. 未探索 (白色) - 默认背景
    # 2. 已探索 (浅灰色)
    # 3. Floor (浅绿色) - 从检测mask加载
    # 4. 障碍物 (黑色)
    # 5. 轨迹 (橙色线)
    # 6. Landmarks (紫色标注)
    # 7. Waypoints (蓝色圆圈+白色数字)
    
    sem_map_vis = apply_color_palette(semantic_map)  # 480×480 RGB
```

### 2. 坐标系转换 (Map Space → Display Space)

```python
# 原始地图坐标 (map_x, map_y) → 显示坐标 (display_x, display_y)
display_x = map_y * 480 / w
display_y = (h - 1 - map_x) * 480 / h  # Y轴翻转!
```

**为什么翻转Y轴?**
- 地图坐标: `map_x=0`在**顶部**, `map_x=479`在**底部**
- 图像坐标: `y=0`在**顶部**, `y=479`在**底部**
- **相同**,因此需要翻转: `display_y = (h-1) - map_x`

### 3. 旋转地图 (对齐Agent朝向)

```python
# 计算旋转矩阵 (使Agent朝向图像上方)
agent_orientation = current_pose[2]  # 弧度
rotation_angle = 90 - degrees(agent_orientation)  # 转为度数

M = cv2.getRotationMatrix2D(center=(240, 240), angle=rotation_angle, scale=1.0)

# 旋转轨迹点
for (map_x, map_y) in trajectory_points:
    display_point = [map_y * 480 / w, (h - 1 - map_x) * 480 / h, 1]
    rotated_point = M @ display_point  # 齐次坐标变换
    cv2.circle(global_map_rotated, (int(rotated_point[0]), int(rotated_point[1])), ...)
```

### 4. Local Map渲染 (独立构建)

```python
def render_local_map(full_map, current_pose, ...):
    # 1. 裁剪以agent为中心的400×400区域
    agent_pixel_x = int(current_pose[1] * 100 / resolution)
    agent_pixel_y = int(current_pose[0] * 100 / resolution)
    
    crop_size = 400
    x1 = agent_pixel_x - crop_size // 2
    x2 = x1 + crop_size
    y1 = agent_pixel_y - crop_size // 2
    y2 = y1 + crop_size
    
    # 边界检查
    x1 = max(0, min(x1, h - crop_size))
    y1 = max(0, min(y1, w - crop_size))
    
    # 2. 裁剪地图
    local_obstacle = obstacle_map[x1:x2, y1:y2]
    local_explored = explored_map[x1:x2, y1:y2]
    local_semantic = semantic_map[x1:x2, y1:y2]
    
    # 3. 渲染 (与Global Map相同的layer顺序)
    local_map_vis = apply_color_palette(local_semantic)
    
    # 4. 旋转使agent朝向上方
    M = cv2.getRotationMatrix2D((200, 200), rotation_angle, 1.0)
    local_map_rotated = cv2.warpAffine(local_map_vis, M, (400, 400))
    
    return local_map_rotated
```

### 5. 距离信息叠加

```python
# 使用depth测距获取障碍物距离
obstacle_distances = {
    'front': calculate_distance(depth, angle=0),
    'left_30': calculate_distance(depth, angle=-30),
    'right_30': calculate_distance(depth, angle=30),
    ...
}

# 在地图上绘制梯形距离指示器
local_map_with_distance = draw_distance_indicators(local_map, obstacle_distances)
```

---

## 常见问题

### Q1: 为什么地图坐标和像素坐标是反的?
**A**: 
- 地图逻辑: `[x, y]` = `[东, 北]` = `[列, 行]`
- NumPy数组: `array[row, col]` = `array[x, y]` 
- 因此 `map_x` 对应**行**(纵向), `map_y` 对应**列**(横向)

### Q2: 重心化后旧数据会丢失吗?
**A**: 不会。重心化前会将Local Map数据写回Full Map，确保所有探索数据持久化。

### Q3: Full Map满了怎么办?
**A**: Full Map固定24m×24m，适用于大多数室内场景。如果轨迹超出，超出部分会被截断。建议通过以下参数调整:
```python
MAP_SIZE_CM = 2400  # 增大到30m×30m
CENTER_RESET_STEPS = 25  # 更频繁重心化
```

### Q4: 为什么渲染时需要旋转?
**A**: 为了视觉直观性:
- **Global Map**: 旋转后Agent始终在中心朝向上方，便于理解空间关系
- **Local Map**: 旋转后Agent朝向上方，符合"前方=上方"的直觉

---

## 调试建议

### 打印关键坐标
```python
# semantic_mapping.py: update_map()
print(f"Full Pose: {self.full_pose[0]}")  # [x, y, ori] 米
print(f"Local Pose: {self.local_pose[0]}")  # [x, y, ori] 米
print(f"Origins: {self.origins[0]}")  # [y, x, 0] 米
print(f"LMB: {self.lmb[0]}")  # [gx1, gx2, gy1, gy2] 像素
```

### 可视化Voxel
```python
# 保存voxel投影结果
torch.save(agent_height_proj, f"voxel_proj_step{step}.pt")
plt.imshow(agent_height_proj[0, 0].cpu().numpy())
plt.savefig(f"voxel_vis_step{step}.png")
```

### 检查坐标变换
```python
# 验证Local↔Full转换
full_pose_calc = local_pose + torch.from_numpy(origins)
assert torch.allclose(full_pose, full_pose_calc, atol=1e-3)
```

---

## 参考文献

1. **Semantic Mapping**: Chaplot et al., "Object Goal Navigation using Goal-Oriented Semantic Exploration"
2. **RGBD投影**: Chaplot et al., "Active Neural SLAM"
3. **Coordinate Systems**: Habitat-Sim Documentation

---

**文档版本**: 1.0  
**最后更新**: 2026-02-05  
**维护者**: MapReAct-VLN Team
