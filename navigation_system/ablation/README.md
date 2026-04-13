# SpaceVLN Ablation Runtime

这个目录只做**原系统减法消融**：

- 不改原始 `navigation_system/vlm/prompts/*`
- 不额外拼接新的说明 prompt
- 仍然走原来的 prompt 模板，只是把部分输入结果不再送进去
- 消融 prompt 不再运行时裁剪，而是使用从原始 `navigation_system/vlm/prompts/templates/` 完整复制出来的静态 markdown 副本：`navigation_system/ablation/templates/<ablation_name>/...`
- 每个消融目录都同时包含不开 cache 的 `planning_verify.prompt.md` / `action_execution.prompt.md`，以及 cache 版 `cache/planning_verify.*.prompt.md` / `cache/action.*.prompt.md`

## 目录结构

```text
navigation_system/ablation/
├── configs/        # 消融 yaml 预设
├── presets.py      # canonical preset 注册表与别名
├── templates/      # 从主系统完整复制并裁剪后的静态 prompt
├── prompts/        # prompt / cache prompt 构建
├── models/         # controller / planner / executor 适配层
├── render/         # thinking 图 / map 渲染适配
├── runtime/        # 运行入口、profile、结果目录/批处理规则
└── tools/          # prompt 审计等维护工具
```

## 架构约定

- 根目录只保留 `__init__.py` / `config.py` / `presets.py` 这类公共配置与注册入口
- `configs/` 只放静态 yaml
- `templates/` 只放静态 prompt 副本
- `prompts/` 只负责把输入装配成最终 prompt
- `models/` 只负责 planner / executor / controller 适配
- `render/` 只负责图片与地图最终暴露内容
- `runtime/` 负责 profile、结果目录、批量执行、入口编排
- 不保留无必要兼容 wrapper；repo 内入口全部改为新的标准模块路径

## 六种预设

- `navigation_system/ablation/configs/no_landmark.yaml`
  - 去掉 landmark 感知相关输入
- `navigation_system/ablation/configs/no_space_structure.yaml`
  - 去掉 space structure 相关输入
- `navigation_system/ablation/configs/no_planning_reasoning.yaml`
  - 只去掉 planning 侧显式分步推理 prompt 段
- `navigation_system/ablation/configs/no_action_reasoning.yaml`
  - 只去掉 action 侧显式决策推理 prompt 段
- `navigation_system/ablation/configs/no_planning_action_reasoning.yaml`
  - planning + action 两边都去掉显式分步 reasoning / process prompt 段
- `navigation_system/ablation/configs/no_landmark_no_space_structure.yaml`
  - landmark 和 space structure 都去掉

## 消融含义

### `landmark` 消融

- thinking 图里不再给 landmark detection box / landmark strip
- verify 时不再给 previous-subtask landmark summary
- action prompt 不再给 `detected_landmarks` / `landmark_map_info`
- action 图改成原始 RGB + obstacle overlay，不再用 landmark detection render

### `space structure` 消融

- verify prompt 不再给 `Space Structure` 文本摘要
- thinking 图里不再给 `Space Waypoint / Current Area / last visited marker`
- thinking 用的 global map 不再画 space-structure overlay

说明：当前主系统里，`space structure` 主要是 thinking 侧输入；action 侧本来就没有单独的 `Space Structure` 文本摘要，因此这个消融主要改 thinking prompt 和 thinking 图像。

### `planning reasoning` 消融

- 只改 planning 侧模板
- 包括：
  - 非 cache：`planning_initial.prompt.md` / `planning_verify.prompt.md`
  - cache：`cache/planning_initial.system.prompt.md` / `cache/planning_verify.system.prompt.md`
- action 侧模板保持原样

### `action reasoning` 消融

- 只改 action 侧模板
- 包括：
  - 非 cache：`action_execution.prompt.md`
  - cache：`cache/action.system.prompt.md`
- planning 侧模板保持原样

### `planning-action reasoning` 消融

- 不改输入源，不关 landmark / space structure / 图像渲染
- 只在 `ablation/templates/no-planning-action-reasoning/` 里删除 planning + action 两侧显式分步 `Reasoning` / `Process` prompt 段
- 保留原有 JSON 字段约束；`reasoning` 字段仍保留，但改成**简短任务摘要**，不再要求展开分步推理
- 同时覆盖：
  - 非 cache：`planning_initial.prompt.md` / `planning_verify.prompt.md` / `action_execution.prompt.md`
  - cache：`cache/planning_initial.*.prompt.md` / `cache/planning_verify.*.prompt.md` / `cache/action.*.prompt.md`
- 兼容旧别名：`navigation_system/ablation/configs/no_reasoning.yaml` 仍可用，但会解析到同一个 `no-planning-action-reasoning` 预设

## 入口

- `vlm_navigation_ablation.py`
- `run_r2r/vlm_navigation_ablation.sh`
- `vlm_navigation_ablation_qwen_cache.py`
- `run_r2r/vlm_navigation_ablation_qwen_cache.sh`

## 用法

```bash
bash run_r2r/vlm_navigation_ablation.sh landmark 1 10
```

```bash
bash run_r2r/vlm_navigation_ablation.sh space_structure 1 10
```

```bash
bash run_r2r/vlm_navigation_ablation.sh planning_reasoning 1 10
```

```bash
bash run_r2r/vlm_navigation_ablation.sh action_reasoning 1 10
```

```bash
bash run_r2r/vlm_navigation_ablation.sh planning_action_reasoning 1 10
```

```bash
bash run_r2r/vlm_navigation_ablation.sh both 1 10
```

也支持显式参数写法：

```bash
bash run_r2r/vlm_navigation_ablation.sh --ablation no_landmark 1 10
```

也兼容旧的环境变量写法：

```bash
ABLATION_CONFIG=navigation_system/ablation/configs/no_landmark.yaml \
bash run_r2r/vlm_navigation_ablation.sh 1 10
```

查看脚本提示：

```bash
bash run_r2r/vlm_navigation_ablation.sh --help
bash run_r2r/vlm_navigation_ablation_qwen_cache.sh --help
```

检查 ablation prompt 树是否完整：

```bash
python navigation_system/ablation/tools/prompt_audit.py
```

## 结果隔离

默认结果目录会写到：

- `.../vlnce/ablation/<ablation_name>/<model_name>/`

并自动保存：

- `result_dir/ablation/manifest.json`
- `result_dir/ablation/config.yaml`
