# Reference Path (Ground Truth) 数据流分析

## 数据来源

两个系统都使用相同的数据集格式：**R2R_VLNCE_v1-3_preprocessed**

### 数据集文件路径
```
data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz
```
- `{split}` = `train`, `val_seen`, `val_unseen`, `test`

### JSON数据结构
```json
{
  "instruction_vocab": {...},
  "episodes": [
    {
      "episode_id": "1",
      "trajectory_id": "123",
      "scene_id": "17DRP5sb8fy",
      "start_position": [x, y, z],
      "start_rotation": [x, y, z, w],
      "goals": [{"position": [x, y, z], ...}],
      "reference_path": [
        [x1, y1, z1],
        [x2, y2, z2],
        [x3, y3, z3],
        ...
      ],
      "instruction": {...}
    },
    ...
  ]
}
```

## 数据加载流程对比

### Sub-VLM-VLN 的数据流

```
1. 配置文件：vlnce_task.yaml
   └─> DATASET.DATA_PATH = "data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"

2. 数据集类：VLNCEDatasetV1 (task.py)
   └─> from_json() 解析 JSON
       └─> 创建 VLNExtendedEpisode 对象
           └─> reference_path: Optional[List[List[float]]] = attr.ib(default=None)
               ✅ 直接从 JSON 字段 "reference_path" 加载

3. Measure类：TopDownMapVLNCE (measures.py)
   └─> reset_metric()
       └─> if self._config.DRAW_REFERENCE_PATH:
           └─> maps.draw_reference_path(
                   episode.reference_path  # 直接使用episode的reference_path
               )

4. 绘制函数：draw_reference_path (maps.py)
   └─> 遍历 episode.reference_path 中的所有点
       └─> 转换为grid坐标并绘制
           - 颜色：MAP_SHORTEST_PATH_WAYPOINT [0, 150, 0] 深绿色
           - 线条：虚线 (dashed, gap=10)
           - 粗细：0.4 * map_resolution / 128
           - 点大小：pad=0.3
```

### MapReAct-VLN 的数据流

```
1. 配置文件：habitat_extensions/config/default.py
   └─> （继承Habitat默认配置，未显式设置DATA_PATH）
       推测使用：data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz

2. 数据集类：VLNCEDatasetV1 (task.py)
   └─> from_json() 解析 JSON
       └─> 创建 VLNExtendedEpisode 对象
           └─> reference_path: Optional[List[List[float]]] = attr.ib(default=None)
               ✅ 直接从 JSON 字段 "reference_path" 加载
               + llm_reply: Dict = attr.ib(default=None)  # MapReAct额外字段

3. Measure类：TopDownMapVLNCE (measures.py)
   └─> reset_metric()
       └─> if self._config.DRAW_REFERENCE_PATH:
           └─> maps.draw_reference_path(
                   episode.reference_path  # 直接使用episode的reference_path
               )

4. 绘制函数：draw_reference_path (maps.py) ✅ 已修复为与Sub-VLM-VLN一致
   └─> 遍历 episode.reference_path 中的所有点
       └─> 转换为grid坐标并绘制
           - 颜色：MAP_SHORTEST_PATH_WAYPOINT [0, 150, 0] 深绿色
           - 线条：虚线 (dashed, gap=10)
           - 粗细：0.4 * map_resolution / 128
           - 点大小：pad=0.3
```

## 关键配置开关

### Sub-VLM-VLN
```python
# VLN_CE/habitat_extensions/config/default.py
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_REFERENCE_PATH = True
```

### MapReAct-VLN
```python
# habitat_extensions/config/default.py
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_REFERENCE_PATH = True  # 已启用
```

## 数据字段定义

两个系统都使用相同的Episode类定义：

```python
@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)  # ← Ground Truth路径
    instruction: ExtendedInstructionData = attr.ib(default=None, validator=not_none_validator)
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)
```

## 可视化参数对比（已统一）

| 参数 | Sub-VLM-VLN | MapReAct-VLN (修复后) |
|------|-------------|----------------------|
| **颜色** | `MAP_SHORTEST_PATH_WAYPOINT` [0, 150, 0] | `MAP_SHORTEST_PATH_WAYPOINT` [0, 150, 0] ✅ |
| **线条样式** | `dashed` | `dashed` ✅ |
| **虚线间隙** | `gap=10` | `gap=10` ✅ |
| **线条粗细** | `0.4 * map_resolution / 128` | `0.4 * map_resolution / 128` ✅ |
| **点大小** | `pad=0.3` | `pad=0.3` ✅ |
| **Null检查** | ❌ 无 | ❌ 无（已移除） ✅ |

## 可能的问题排查

### 1. reference_path为空或None
**原因**：
- JSON文件中缺少`reference_path`字段
- 数据集版本不匹配（v1-2 vs v1-3）
- Episode加载过程中字段被过滤

**检查方法**：
```python
import gzip, json
with gzip.open('data/datasets/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz', 'rt') as f:
    data = json.load(f)
    ep = data['episodes'][0]
    print('reference_path' in ep)
    print(len(ep.get('reference_path', [])))
```

### 2. 可视化未显示
**原因**：
- `DRAW_REFERENCE_PATH = False`
- 颜色与背景对比度不足（黑色[0,0,0]在灰色背景上不明显）
- 线条太细或被其他元素覆盖

**解决方案**：
- ✅ 使用深绿色 [0, 150, 0]（与Sub-VLM-VLN一致）
- ✅ 配置已启用 `DRAW_REFERENCE_PATH = True`
- ✅ 绘制参数已统一

### 3. 坐标转换错误
**检查点**：
- `habitat_maps.to_grid()` 坐标转换
- `[::-1]` 反转操作（从(x,y)到(y,x)）
- Grid边界检查

## 总结

两个系统的reference_path数据流**完全一致**：
1. 都从相同的JSON.gz数据集文件读取
2. 都使用相同的VLNExtendedEpisode类定义
3. 都在TopDownMapVLNCE的reset_metric()中调用draw_reference_path()
4. ✅ 现在使用相同的可视化参数（深绿色虚线）

**关键修复**：
- 将颜色从黑色改为深绿色（提高对比度）
- 统一线条粗细、间隙、点大小参数
- 移除不必要的null检查（与Sub-VLM-VLN保持一致）
