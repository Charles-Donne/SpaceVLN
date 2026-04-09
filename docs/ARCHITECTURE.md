# Navigation System Architecture

`navigation_system/` 现在按功能职责划分，而不是按历史实现细节平铺。

## 1. 顶层模块

```text
navigation_system/
├── controller/
├── vlm/
├── detection/
├── space/
│   ├── map/
│   ├── topology/
│   ├── landmarks/
│   ├── geometry/
│   └── description/
├── render/
│   ├── map/
│   ├── views/
│   └── episode_visualization/
├── runtime/
│   ├── storage/
│   └── ...
├── env/
└── config/
```

## 2. 模块职责

### controller

- `base_controller.py`
  - 负责 Habitat 环境、检测、建图、渲染等基础导航能力装配。
- `navigation_controller.py`
  - 负责 thinking / action 主循环、子任务推进、停止逻辑、结果写回。
- `state/`
  - 只保留 controller 自己的运行态，例如 episode 计时和 controller 选项。

### vlm

- `api/`
  - provider 请求、图片编码、结构化返回、`vlm_info.json`。
- `planning/`
  - 高层规划和 verify / replan。
- `execution/`
  - 低层动作决策。
- `prompts/`
  - markdown prompt 模板与 builder。
  - cache prompt 继续放在 `prompts/templates/cache/`。
- `contracts/`
  - planner / action 共用 schema 与常量。
- `reporting/`
  - 缓存命中与 VLM 统计汇总。

### detection

- `grounded_sam.py`
  - 检测与分割主入口。
- `vendor/repvit_sam/`
  - 第三方 RepViT-SAM 实现隔离区。

### space

- `map/`
  - 地图本体、语义地图更新、地图层分析。
- `topology/`
  - 空间区域、空间类型、waypoint、区域连接关系。
- `landmarks/`
  - landmark world instance、landmark memory、landmark 选择与合并。
- `geometry/`
  - pose、rotation、depth、projection 等底层空间计算。
- `description/`
  - 把空间状态转成 VLM 可读的文本描述。

### render

- `map/`
  - global/local map 绘制、space area / landmark 等地图叠加层。
- `views/`
  - 给模型看的输入图，例如 thinking 12-view、action detection 视图、panorama。
- `episode_visualization/`
  - 导航回放、逐步拼接图、GIF。

### runtime

- `runner.py`
  - CLI 总入口。
- `execution.py`
  - 单 episode / 并行 worker 执行。
- `episode_selection.py`
  - episode 读取、筛选、随机采样、`skip-sr1`。
- `episode_io.py`
  - 控制台输出、stdout 重定向、结果路径解析。
- `storage/`
  - detail / log / records 的保存布局。
- `results_report.py`
  - 基于 `log/` 目录生成汇总报告。

### env

- Habitat 环境注册和 vector env 构造。

### config

- `experiments/`
  - 实验控制面板，集中管理 `DETECTION / SPACE / RENDER / OUTPUT / CONTROL`。
- `runtime/`
  - 默认值和同步逻辑，把结构化面板同步到运行时派生字段。
- `vlm/`
  - 模型接口 yaml 配置。
- `core/`
  - 静态常量、类别表和配置辅助函数。

## 3. 当前主链路

运行时主流程：

1. `runtime/runner.py` 解析参数并选择运行 profile。
2. `runtime/execution.py` 构建 `controller/navigation_controller.py`。
3. `controller/` 调度 `detection/`、`space/`、`render/`、`vlm/`。
4. `vlm/planning/` 生成子任务，`vlm/execution/` 生成动作。
5. `runtime/storage/` 保存 detail、records、log。
6. `runtime/results_report.py` 对 `log/` 做离线汇总。

## 4. 设计原则

- 控制流只放在 `controller/`，不把空间逻辑和渲染逻辑继续塞进 controller。
- prompt 与 cache 模板只放在 `vlm/prompts/`。
- landmark / waypoint / space area 全部归到 `space/`，不再散落到 controller、utils、visualization。
- 给模型看的图和给人看的回放图统一归到 `render/`。
- 结果保存与目录布局统一归到 `runtime/storage/`，不再挂在 `vlm/` 下面。
- 第三方模型实现隔离到 `detection/vendor/`，避免和自有逻辑混放。
