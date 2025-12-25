## VLN Navigation Evaluation System

已完成的修复和改进：

### 1. Ground Truth路径可视化 ✅

**修改文件：**
- `habitat_extensions/config/default.py`: 启用 `DRAW_REFERENCE_PATH = True`
- `habitat_extensions/maps.py`: 
  - 修复`draw_reference_path`函数，使用黑色虚线显示ground truth
  - 添加空值检查，避免崩溃
- `habitat_extensions/measures.py`: 添加reference_path存在性检查和警告

**效果：**
- Ground truth路径显示为**黑色虚线**（8px间隔）
- Agent轨迹显示为**橙色实线**
- 两条路径对比更直观

### 2. 完整的评估指标系统 ✅

**核心指标（符合VLN标准）：**
1. **Success Rate (SR)**: 成功率 - 到达目标3m内
2. **SPL**: Success weighted by Path Length - 考虑路径长度的成功率
3. **Navigation Error (NE)**: 导航误差 - 最终距离目标的距离
4. **Path Length**: 路径长度
5. **Oracle Success (OSR)**: Oracle成功率 - 轨迹中任意点到达目标3m内
6. **Oracle Navigation Error (ONE)**: Oracle导航误差 - 轨迹中最近点到目标的距离
7. **Oracle SPL**: Oracle SPL

**修改文件：**
- `vlnce_baselines/vlm_navigation_controller.py`: 修复指标收集逻辑
- `vlnce_baselines/vlm/save_manager.py`: 同时保存到详细结果和log/目录
- `analyze_results.py`: 新增结果分析脚本

### 3. 结果保存结构

```
data/vlm_navigation/
├── log/                          # 用于结果分析
│   ├── episode_701.json
│   ├── episode_832.json
│   └── summary.json             # analyze_results.py生成
└── episode_XXX/
    ├── records/
    │   └── result.json          # 详细结果（包含所有指标）
    ├── thinking/
    ├── action/
    └── ...
```

### 使用方法

#### 1. 运行导航

```bash
python vlm_navigation.py \
    --exp-config vlnce_baselines/config/exp1.yaml \
    --episode-id 701 \
    --num-episodes 5 \
    --results-dir data/vlm_navigation \
    --max-steps 500
```

#### 2. 分析结果

```bash
# 分析单次运行结果
python analyze_results.py --path data/vlm_navigation

# 输出示例：
# ============================================================
# 📊 VLN Navigation Results Summary
# ============================================================
# 
# 📁 Results Directory: data/vlm_navigation
# 📝 Total Episodes: 5
# 
# ------------------------------------------------------------
# ✅ Success Rate (SR): 3/5 (0.600)
# 🎯 SPL (Success weighted by Path Length): 0.427
# 🔮 Oracle Success Rate: 4/5 (0.800)
# 📍 Average Navigation Error (NE): 2.134m
# 📍 Average Oracle Navigation Error (ONE): 1.523m
# 📏 Average Path Length: 12.345m
# ============================================================
```

#### 3. 测试reference_path加载

```bash
python test_reference_path.py
```

### 关键配置说明

**habitat_extensions/config/default.py:**
```python
# 确保这些配置正确
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_REFERENCE_PATH = True  # 显示ground truth
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_MP3D_AGENT_PATH = True # 显示agent轨迹
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_BORDER = True           # 显示地图边界

# Success判断阈值
_C.TASK.ORACLE_SUCCESS.SUCCESS_DISTANCE = 3.0  # 3米内算成功
```

### 指标解释

**成功判断标准（3米阈值）：**
- `distance_to_goal < 3.0` → Success = 1
- 这是VLN-CE标准阈值

**SPL计算：**
```python
SPL = Success × (shortest_path_length / actual_path_length)
```
- 奖励高效率的成功导航
- 范围: [0, 1]
- 1.0 = 完美导航（成功+最短路径）

**Oracle指标：**
- 考虑整个轨迹中的最佳表现
- 用于评估"如果agent在最佳位置停止"的潜在性能

### 常见问题

**Q: 为什么看不到ground truth路径？**
A: 检查：
1. 配置: `DRAW_REFERENCE_PATH = True`
2. 数据集是否包含`reference_path`字段（运行`test_reference_path.py`检查）
3. 控制台是否有"⚠️ 缺少reference_path"警告

**Q: 指标都是0怎么办？**
A: 检查：
1. Episode是否正常完成
2. `latest_info`是否正确获取
3. 查看`log/episode_XXX.json`确认指标值

**Q: analyze_results.py报错找不到文件？**
A: 确保：
1. 指定了正确的`--path`（应该是包含`log/`子目录的父目录）
2. `log/`目录中有`.json`文件
3. 文件格式正确（包含success, spl等字段）
