# MapReAct-VLN 系统架构文档

## 📋 目录结构

```
MapReAct-VLN/
├── vlm_navigation.py                    # VLM自动导航入口脚本
├── interactive_navigation.py            # 交互式导航入口脚本
├── habitat_extensions/                  # Habitat环境扩展
│   ├── habitat_simulator.py            # 模拟器扩展
│   ├── sensors.py                       # 传感器定义
│   ├── task.py                          # 任务定义
│   └── config/                          # 配置文件
├── vlnce_baselines/                     # 核心功能模块
│   ├── interactive_navigation_controller.py  # 基础导航控制器
│   ├── vlm_navigation_controller.py          # VLM导航控制器
│   ├── detection/                       # 目标检测模块
│   │   ├── grounded_sam.py             # GroundedSAM检测器
│   │   └── RepViTSAM/                  # RepViT-SAM模型
│   ├── mapping/                         # 语义建图模块
│   │   ├── semantic_mapping.py         # 语义地图核心
│   │   ├── mapper.py                   # 地图管理器
│   │   ├── depth_utils.py              # 深度处理工具
│   │   ├── map_utils.py                # 地图工具函数
│   │   └── pose.py                     # 位姿处理
│   ├── visualization/                   # 可视化模块
│   │   ├── visualizer.py               # 地图可视化器
│   │   ├── panorama_generator.py       # 全景图生成器
│   │   └── rendering.py                # 渲染工具
│   ├── vlm/                             # VLM规划与执行模块
│   │   ├── thinking.py                 # LLM规划器
│   │   ├── action.py                   # VLM动作执行器
│   │   ├── prompts.py                  # 规划提示词
│   │   ├── action_prompt.py            # 动作提示词
│   │   ├── api_client.py               # API客户端
│   │   ├── navigation_visualizer.py    # 导航可视化
│   │   ├── save_manager.py             # 数据保存管理
│   │   ├── action_parser.py            # 动作解析器
│   │   └── navigation_config.py        # 导航配置
│   ├── config_system/                   # 配置系统
│   │   ├── categories.py               # 类别定义
│   │   ├── constants.py                # 常量定义
│   │   └── setup.py                    # 系统配置
│   └── common/                          # 公共工具
│       ├── environments.py             # 环境工具
│       └── utils.py                    # 通用工具
└── docs/                                # 文档
    ├── architecture.md                  # 架构文档（本文件）
    ├── step_logic_analysis.md          # 步骤逻辑分析
    └── step_numbering_logic.md         # 步骤编号逻辑
```

## 🏗️ 系统架构

### 1. 核心层次结构

```
┌─────────────────────────────────────────────┐
│         VLM Navigation Controller           │
│  ┌───────────────────────────────────────┐  │
│  │  Interactive Navigation Controller    │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
           ↓           ↓           ↓
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Detection│  │ Mapping  │  │   VLM    │
    │  Module  │  │  Module  │  │  Module  │
    └──────────┘  └──────────┘  └──────────┘
```

### 2. 模块职责

#### 2.1 控制器层（Controllers）

**InteractiveNavigationController** (基类)
- 环境管理（Habitat环境初始化和交互）
- 检测系统集成（GroundedSAM目标检测）
- 语义建图（SemanticMapping + Mapper）
- 地图可视化（MapVisualizer）
- 12步×30°环视建图流程

**VLMNavigationController** (派生类)
- 继承所有基础功能
- 添加VLM规划与执行
- 360°环视 + 4方向全景图生成
- LLM高层规划（子任务分解）
- VLM低层执行（动作决策）
- 结果保存与评估

#### 2.2 检测模块（Detection）

**GroundedSAM**
- 开放词汇目标检测
- 支持自定义类别
- 返回边界框和掩码
- 可配置置信度阈值

**RepViT-SAM**
- 轻量级SAM模型
- 移动端优化
- 快速分割推理

#### 2.3 建图模块（Mapping）

**SemanticMapping**
- 深度投影到3D空间
- 语义信息融合
- 障碍物地图构建
- 轨迹记录

**Mapper**
- 地图状态管理
- 位姿跟踪
- Waypoint管理
- 地图更新接口

#### 2.4 可视化模块（Visualization）

**MapVisualizer**
- 全局地图渲染
- 局部地图渲染
- 轨迹可视化
- Landmark标注
- Waypoint标注

**PanoramaGenerator**
- 全景图拼接（OpenCV Stitcher）
- 方向标注（跨平台字体）
- 可拔插设计

**NavigationVisualizer**
- RGB+俯视图拼接
- GIF动画生成
- 步骤记录

#### 2.5 VLM模块（VLM）

**LLMPlanner (thinking.py)**
- 初始子任务规划
- 验证与重规划
- JSON输出解析

**ActionExecutor (action.py)**
- 基于RGB+地图的动作决策
- 进度总结更新
- 动作参数解析

**NavigationVisualizer**
- 步骤可视化
- GIF生成

**SaveManager**
- Thinking输出保存
- Action输出保存
- 导航结果保存

## 🔌 可拔插设计

### 检测器（Detector）
```python
# 接口定义
class BaseDetector:
    def detect(self, image, text_prompt, box_threshold, text_threshold):
        pass

# 实现
- GroundedSAM (当前)
- YOLO-World (可替换)
- CLIP + SAM (可替换)
```

### 规划器（Planner）
```python
# 接口定义
class BasePlanner:
    def generate_initial_subtask(self, instruction, images, maps):
        pass
    
    def verify_and_replan(self, instruction, subtask, images, maps):
        pass

# 实现
- LLMPlanner (OpenAI API)
- LocalLLMPlanner (本地大模型，可扩展)
```

### 全景图生成器（PanoramaGenerator）✨
```python
# 接口定义
class BasePanoramaGenerator:
    def create_panorama(self, images, direction_name):
        pass

# 实现
- PanoramaGenerator (OpenCV Stitcher)
- CylindricalPanorama (柱面投影，可扩展)
- SphericalPanorama (球面投影，可扩展)
```

### 地图构建器（Mapper）
```python
# 接口定义
class BaseMapper:
    def update_map(self, observations, poses):
        pass

# 实现
- SemanticMapping + Mapper (当前)
- OccupancyGridMapper (可替换)
- VoxelMapper (可替换)
```

## 📊 数据流

### VLM导航主流程

```
1. 初始化
   └─> 加载环境、检测器、建图器、规划器

2. 环视建图（12步×30°）
   └─> 收集12张RGB + depth
   └─> 目标检测 + 语义融合
   └─> 生成4个90°全景图
        └─> PanoramaGenerator.create_panorama() ✨

3. LLM规划（thinking）
   └─> 输入：4全景图 + 全局地图 + 局部地图
   └─> 输出：subtask (JSON)

4. VLM执行（action）
   └─> 输入：第一人称视图 + 局部地图 + 检测结果
   └─> 输出：动作决策
   └─> 执行动作 → 更新地图

5. 验证重规划
   └─> 360°环视 + 生成全景图
   └─> LLM验证完成状态
   └─> 如已完成 → 生成下一子任务
   └─> 循环3-5直到完成

6. 结果保存
   └─> GIF动画
   └─> JSON结果文件
   └─> Thinking/Action记录
```

## 🔧 配置系统

### 环境配置
- `habitat_extensions/config/default.py`
- 传感器、动作、任务参数

### 检测配置
- `vlnce_baselines/detection/grounded_sam.py`
- 模型路径、阈值、设备

### 建图配置
- `vlnce_baselines/config/default.py`
- 地图尺寸、分辨率、投影参数

### VLM配置
- `vlnce_baselines/vlm/llm_config.yaml`
- `vlnce_baselines/vlm/vlm_config.yaml`
- API密钥、模型、提示词

## 🎯 设计原则

### 1. 模块化
- 每个模块职责单一
- 接口清晰定义
- 低耦合高内聚

### 2. 可扩展性
- 插件化设计
- 基类定义接口
- 配置驱动行为

### 3. 可维护性
- 代码注释完整
- 文档同步更新
- 遵循PEP8规范

### 4. 可复用性
- 工具函数独立
- 避免重复代码
- 统一数据格式

## 📝 最近更新（2025-12-16）

### ✨ 新增模块
- **PanoramaGenerator** (`vlnce_baselines/visualization/panorama_generator.py`)
  - 从 `VLMNavigationController` 中解耦全景图生成逻辑
  - 移动到 `visualization` 模块，与其他可视化工具统一管理
  - 支持OpenCV Stitcher拼接
  - 跨平台字体支持（macOS/Linux/Windows）
  - 可拔插设计，易于替换其他拼接算法

### 🗑️ 清理冗余
- 删除 `VLMNavigationController._create_panorama_from_3_images()`
- 删除 `VLMNavigationController._crop_panorama_view()` (已废弃)
- 删除 `VLMNavigationController._stitch_panorama()` (已废弃)
- 删除废弃的waypoint辅助方法：
  - `add_waypoint()` → 使用 `mapper.add_waypoint()`
  - `get_waypoint_summary()` → 使用 `_get_waypoint_summary()`
  - `visualize_waypoints_on_map()` → visualizer自动渲染

### 📐 架构优化
- 控制器层更轻量，专注流程控制
- 工具模块更独立，易于测试和替换
- 清晰的模块边界，降低维护成本

## 🔗 依赖关系

```
VLMNavigationController
  ├── InteractiveNavigationController
  │   ├── GroundedSAM (detection)
  │   ├── SemanticMapping + Mapper (mapping)
  │   └── MapVisualizer (visualization)
  ├── PanoramaGenerator (visualization)
  ├── LLMPlanner (vlm)
  ├── ActionExecutor (vlm)
  ├── NavigationVisualizer (vlm)
  └── SaveManager (vlm)
```

## 📚 相关文档

- [步骤逻辑分析](./step_logic_analysis.md)
- [步骤编号逻辑](./step_numbering_logic.md)
- [工作流程图](./工作流程图.md)
- [建图机制说明](./建图机制说明.md)
