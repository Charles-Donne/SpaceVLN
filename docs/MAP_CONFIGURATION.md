# 全局地图配置指南

## 📐 地图存储机制

### 当前设计
- **存储方式**：固定大小的 Tensor/NumPy 数组
- **默认大小**：MAP_SIZE_CM = 2400cm (24m × 24m)
- **分辨率**：5cm/pixel
- **地图尺寸**：480×480 pixels
- **问题**：有固定边界，无法动态扩展

### 为什么有边界？
1. **GPU内存限制**：预分配固定大小的Tensor在GPU上效率最高
2. **建图算法**：Semantic Mapping 模块使用固定大小的网格进行投影
3. **性能考虑**：动态扩展地图会带来性能开销

---

## 🔧 配置选项

### 1. 调整地图大小（扩展边界）

**配置文件**：`vlnce_baselines/config/exp1.yaml` 或命令行参数

```yaml
MAP:
  MAP_SIZE_CM: 3600  # 扩展到 36m × 36m
  # 或
  MAP_SIZE_CM: 4800  # 扩展到 48m × 48m
```

**效果**：
- 2400cm (24m) → 480×480 pixels
- 3600cm (36m) → 720×720 pixels
- 4800cm (48m) → 960×960 pixels

**注意**：增大地图会增加GPU内存使用和计算开销。

---

### 2. 去除 Global Map 裁剪

**默认行为**（已修改）：
- 旧版本：渲染后裁剪为 440×440
- **新版本**：默认保持 480×480（完整显示）

**配置文件**：
```yaml
MAP:
  ENABLE_GLOBAL_MAP_CROP: false  # 默认false，保持完整地图
```

**如果需要恢复裁剪**：
```yaml
MAP:
  ENABLE_GLOBAL_MAP_CROP: true  # 裁剪为 440×440
```

---

### 3. 地图使用统计

**新功能**：运行时自动显示地图使用情况

```
📊 地图使用统计:
  地图尺寸: 480×480 pixels = 24.0m × 24.0m
  轨迹范围: X=[10.2, 18.5]m, Y=[8.3, 15.7]m
  使用区域: 8.3m × 7.4m (12.8% of map)
  ⚠️  警告: 轨迹接近地图边界，建议增大 MAP_SIZE_CM
```

**作用**：
- 实时监控是否接近边界
- 评估是否需要增大 MAP_SIZE_CM
- 显示实际探索范围

---

## 📊 推荐配置

### 场景1：小房间/公寓（默认）
```yaml
MAP:
  MAP_SIZE_CM: 2400  # 24m × 24m
  ENABLE_GLOBAL_MAP_CROP: false
```

### 场景2：大型场景/多房间
```yaml
MAP:
  MAP_SIZE_CM: 3600  # 36m × 36m
  ENABLE_GLOBAL_MAP_CROP: false
```

### 场景3：超大场景
```yaml
MAP:
  MAP_SIZE_CM: 4800  # 48m × 48m
  ENABLE_GLOBAL_MAP_CROP: false
```

**注意**：增大地图会线性增加GPU内存和计算时间。

---

## ⚠️ 注意事项

### 1. 地图大小限制
- **最小建议**：1200cm (12m × 12m)
- **最大建议**：4800cm (48m × 48m)
- **超出警告**：轨迹距离边界 <10% 时会显示警告

### 2. 性能影响
- **2400cm**：480×480 = 230K pixels
- **3600cm**：720×720 = 518K pixels (+125% GPU内存)
- **4800cm**：960×960 = 922K pixels (+300% GPU内存)

### 3. 无法动态扩展
当前架构不支持运行时动态扩展地图，原因：
- Habitat Simulator 使用固定大小的网格
- GPU Tensor 预分配固定内存
- 动态扩展需要重写建图核心代码

**解决方案**：
- 提前设置足够大的 MAP_SIZE_CM
- 根据数据集选择合适的大小
- R2R Val Unseen 大部分场景在 24m×24m 内

---

## 🛠️ 实用命令

### 测试不同地图大小
```bash
# 默认 24m×24m
bash run_r2r/vlm_navigation.sh 832

# 36m×36m（命令行覆盖配置）
python vlm_navigation.py \
    --exp-config vlnce_baselines/config/exp1.yaml \
    --episode-id 832 \
    MAP.MAP_SIZE_CM 3600

# 48m×48m
python vlm_navigation.py \
    --exp-config vlnce_baselines/config/exp1.yaml \
    --episode-id 832 \
    MAP.MAP_SIZE_CM 4800
```

### 查看地图使用情况
运行任何测试后，查看终端输出中的"📊 地图使用统计"部分。

---

## 📈 未来改进（TODO）

### 1. 自适应缩放显示
- 保持存储固定大小
- 渲染时根据轨迹范围动态缩放显示区域
- 配置：`ENABLE_ADAPTIVE_ZOOM: true`

### 2. 动态扩展（长期）
- 检测边界接近时自动扩展
- 使用稀疏数据结构（如八叉树）
- 需要重写 Semantic_Mapping 核心

### 3. 多尺度表示
- 同时维护高分辨率局部地图和低分辨率全局地图
- 类似 SLAM 中的 hierarchical mapping

---

## 📞 问题排查

### Q1: 警告"轨迹接近地图边界"
**原因**：探索范围接近 MAP_SIZE_CM 限制

**解决**：
1. 增大 MAP_SIZE_CM（如 2400 → 3600）
2. 检查轨迹是否异常（如陷入循环）

### Q2: GPU 内存不足
**原因**：MAP_SIZE_CM 设置过大

**解决**：
1. 减小 MAP_SIZE_CM
2. 降低 FRAME_WIDTH / FRAME_HEIGHT
3. 使用更小的 GLOBAL_DOWNSCALING（如 4）

### Q3: Global Map 显示不完整
**原因**：启用了裁剪

**解决**：
```yaml
MAP:
  ENABLE_GLOBAL_MAP_CROP: false
```

---

## 📄 相关文件

- 配置：`vlnce_baselines/config/default.py`
- 建图：`vlnce_baselines/mapping/semantic_mapping.py`
- 可视化：`vlnce_baselines/visualization/visualizer.py`
- 控制器：`vlnce_baselines/interactive_navigation_controller.py`
