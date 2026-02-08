# Token优化建议

## 1. 🖼️ 图片输入优化（预计节省60-80%图片token）

### 当前状态
- **12个环视图**（IMAGE 1-12，每个30° FOV）
- **2张地图**（Global map + Local map）
- **总计14张图片/次** → 约2800-7000KB

### 优化方案A：降低图片分辨率
```python
# 在vlnce_baselines/vlm/api_client.py添加压缩功能
@staticmethod
def compress_image(image_path: str, max_size: int = 512, quality: int = 85) -> str:
    """
    压缩图片并返回临时路径
    
    Args:
        image_path: 原始图片路径
        max_size: 最大边长（默认512px，原始可能是1024px）
        quality: JPEG质量（85可节省50%大小而保持清晰）
    
    Returns:
        压缩后的临时文件路径
    """
    from PIL import Image
    import tempfile
    
    img = Image.open(image_path)
    
    # 等比例缩放
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    # 保存为临时文件
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.convert('RGB').save(tmp.name, 'JPEG', quality=quality, optimize=True)
    return tmp.name

# 修改encode_image_base64使用压缩
@staticmethod
def encode_image_base64(image_path: str, compress: bool = True) -> str:
    if compress:
        compressed_path = BaseAPIClient.compress_image(image_path, max_size=384, quality=80)
        with open(compressed_path, "rb") as f:
            result = base64.b64encode(f.read()).decode("utf-8")
        os.remove(compressed_path)  # 清理临时文件
        return result
    else:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
```

**预期节省：**
- 512px分辨率 → 节省 ~50% token
- 384px分辨率 → 节省 ~70% token
- Quality=80 → 额外节省 ~20% token
- **总计可节省 60-75% 图片token**

### 优化方案B：减少图片数量（最激进）
```python
# 选项1：只发送关键方向（减少到6个）
# IMAGE 1 (Front), 2 (Left 30), 4 (Left 90), 7 (Back), 10 (Right 90), 12 (Right 330)
# 节省：50%图片数量

# 选项2：只发送前方120°视野（4-5个IMAGE）
# IMAGE 12 (Right 330), 1 (Front), 2 (Left 30), 3 (Left 60) [可选4 (Left 90)]
# 节省：60-70%图片数量

# 选项3：用拼接全景图替代12张（已有PanoramaGenerator）
# 4个90°全景图（Front, Left, Back, Right）代替12张30°
# 节省：66%图片数量，但可能损失细节
```

**推荐：混合策略**
- INITIAL planning：4个全景图（90° × 4）
- VERIFICATION：保持12个图（需要精确分析）
- 分辨率：384px
- Quality：80

**预期总节省：70-80% token**

---

## 2. 📝 文本输入优化（预计节省20-40%文本token）

### A. Waypoint History压缩
**当前问题：**
```python
waypoint_summary = """
Waypoint #1 (Step 5): Kitchen - near counter
  - Coordinates: (x: 2.34, y: 1.56, z: 0.12)
  - Orientation: 45.2°
  - Description: Entered kitchen from hallway

Waypoint #2 (Step 12): Kitchen - at refrigerator
  - Coordinates: (x: 3.12, y: 2.01, z: 0.11)
  - Orientation: 90.5°
  - Description: Moved to refrigerator, facing right
...
```
**每个waypoint ~150 tokens，10个waypoint = 1500 tokens**

**优化方案：**
```python
def _get_waypoint_summary_compressed(self, max_waypoints: int = 5) -> str:
    """
    压缩的waypoint摘要，只保留最近N个
    
    格式：简化为单行
    """
    if not hasattr(self, 'waypoints') or not self.waypoints:
        return "None"
    
    # 只保留最近的N个waypoints
    recent = self.waypoints[-max_waypoints:]
    
    # 压缩格式：ID(step): room, orientation
    lines = [f"#{wp['id']}(s{wp['step']}): {wp['room']}, {wp['orientation']:.0f}°" 
             for wp in recent]
    
    return " | ".join(lines)

# 示例输出：
# "#1(s5): Kitchen, 45° | #2(s12): Kitchen, 91° | #3(s18): Living, 180°"
# 从150 tokens/个 → 20 tokens/个，节省 87%
```

### B. Detected Landmarks压缩
**当前：**
```python
detected_landmarks = "chair, table, sofa, tv, bed, desk, lamp, plant, picture, cabinet, ..."
# 可能100+ tokens
```

**优化：**
```python
def _get_landmarks_compressed(self, top_n: int = 5, min_confidence: float = 0.8) -> str:
    """
    只保留高置信度的top-N landmarks
    """
    if not self.detected_classes:
        return "None"
    
    # 如果有置信度信息，过滤低置信度
    # 否则只取前N个
    landmarks = sorted(self.detected_classes)[:top_n]
    return ", ".join(landmarks)

# 示例：从 20+ landmarks → 5个，节省 75%
```

### C. Progress Summary优化（当前已经比较精简）
保持当前格式，但限制历史步数：
```python
# 只保留最近3步，不是全部
progress_summary = "Last 3 steps: TURN_LEFT 30° | MOVE 0.5m | TURN_RIGHT 60°"
```

---

## 3. 🎯 智能采样策略（动态优化）

### 根据场景复杂度调整
```python
def get_image_sampling_strategy(self, scene_complexity: str) -> Dict:
    """
    根据场景复杂度动态调整图片采样
    
    Args:
        scene_complexity: 'simple' | 'medium' | 'complex'
            - simple: 走廊、单一房间（少障碍物）
            - medium: 客厅、厨房（中等家具）
            - complex: 多门廊hallway、家具密集区
    
    Returns:
        {"num_views": int, "resolution": int, "quality": int}
    """
    strategies = {
        'simple': {"num_views": 6, "resolution": 384, "quality": 75},
        'medium': {"num_views": 8, "resolution": 512, "quality": 80},
        'complex': {"num_views": 12, "resolution": 512, "quality": 85}
    }
    return strategies.get(scene_complexity, strategies['medium'])

# 根据地图探索率或障碍物密度判断复杂度
def estimate_scene_complexity(self) -> str:
    if not hasattr(self, 'mapper'):
        return 'medium'
    
    # 示例：基于黑色像素占比（障碍物）
    obstacle_ratio = self.mapper.get_obstacle_ratio()
    if obstacle_ratio < 0.2:
        return 'simple'
    elif obstacle_ratio < 0.4:
        return 'medium'
    else:
        return 'complex'
```

---

## 4. 📊 优先级排序（综合推荐）

### 🥇 高优先级（立即实施）
1. **图片分辨率降至384px + Quality 80** → 节省 60-70%
   - 修改`api_client.py`添加compress_image
   - 修改`encode_image_base64`默认启用压缩
   - **预计节省：~3000 tokens/call**

2. **Waypoint History限制到5个** → 节省 80%+
   - 修改`_get_waypoint_summary()`
   - **预计节省：~800 tokens/call**

3. **Detected Landmarks限制到top-5** → 节省 70%+
   - 修改landmark传递逻辑
   - **预计节省：~50 tokens/call**

### 🥈 中优先级（下一步）
4. **INITIAL阶段减少到8个图** → 节省 33%
   - 修改`get_observations_and_maps()`
   - 只在VERIFICATION保持12个
   - **预计节省：~1000 tokens/call (INITIAL)**

5. **动态采样策略** → 平均节省 20-40%
   - 根据场景复杂度调整
   - 需要添加复杂度评估逻辑

### 🥉 低优先级（激进优化）
6. **用全景图替代多张小图** → 节省 60%+
   - 需要修改prompt让VLM适应全景图格式
   - 可能影响方向判断精度

---

## 5. 💡 实施步骤

### Phase 1: 图片压缩（最快见效）
```bash
# 修改文件
vlnce_baselines/vlm/api_client.py  # 添加compress_image方法
```

### Phase 2: 历史信息压缩
```bash
# 修改文件
vlnce_baselines/vlm_navigation_controller.py  # 修改_get_waypoint_summary
```

### Phase 3: 动态采样
```bash
# 新增评估逻辑
vlnce_baselines/vlm_navigation_controller.py  # 添加complexity估计
```

---

## 6. 📈 预期收益

### 当前token消耗（估算）
- **图片**: ~7000 tokens (14张 × 500 tokens/张)
- **Prompt**: ~800 tokens
- **Waypoint History**: ~1000 tokens
- **Landmarks**: ~100 tokens
- **总计**: ~8900 tokens/call

### 优化后token消耗
- **图片**: ~2000 tokens (70%压缩)
- **Prompt**: ~800 tokens (不变)
- **Waypoint History**: ~100 tokens (90%压缩)
- **Landmarks**: ~30 tokens (70%压缩)
- **总计**: ~2930 tokens/call

**总节省：67% token消耗**

### 成本节省（按GPT-4V计价）
- 假设：$0.01/1K tokens（输入）
- 每episode平均15次thinking call
- **当前**: $1.34/episode
- **优化后**: $0.44/episode
- **节省**: $0.90/episode (67%)

---

## 7. ⚠️ 注意事项

1. **图片质量vs性能权衡**
   - 384px对于室内导航通常足够
   - 如果VLM识别物体困难，可提升到512px

2. **关键信息不能丢**
   - Waypoint history保留最近5个（不是随机采样）
   - Landmarks保留高置信度的（不是随机选择）

3. **分阶段测试**
   - 先测试图片压缩效果
   - 再逐步减少图片数量
   - 最后优化文本输入

4. **监控指标**
   - Success Rate
   - SPL (Success weighted by Path Length)
   - Navigation Error
   - 确保优化不影响导航质量
