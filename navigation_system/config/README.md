# Config Layout

`navigation_system/config/` 按职责固定分成 5 层，建议按下面的优先级修改。

## 1. `experiments/`

这里放实验覆盖项，是**最推荐的日常入口**。

当前主入口：

- `navigation_system/config/experiments/r2r_eval.yaml`

主要分区：

- `TASK`
- `PATHS`
- `OUTPUT`
- `CONTROL`
- `EVAL`

## 2. `system/`

这里放部署相关的系统默认值，不放算法细节。

- `navigation_system/config/system/00_runtime.yaml`
  - 任务入口、GPU、并行环境、结果目录默认值。
- `navigation_system/config/system/10_detection_models.yaml`
  - GroundingDINO / SAM / RepViT-SAM 路径与开关。
- `navigation_system/config/system/20_space_sensor.yaml`
  - 相机 HFOV、分辨率、agent height 等传感器参数。

## 3. `core/`

这里放静态类别、默认阈值和配置辅助函数。

- `navigation_system/config/core/params/`
- `navigation_system/config/core/categories.py`
- `navigation_system/config/core/constants.py`
- `navigation_system/config/core/setup.py`

## 4. `runtime/`

这里是运行时装配层，通常不作为实验主入口直接修改。

- `navigation_system/config/runtime/panels.py`
  - 结构化面板定义与代码级默认值。
- `navigation_system/config/runtime/default.py`
  - 加载 `system/*.yaml + experiments/*.yaml` 并装配完整配置。
- `navigation_system/config/runtime/sync.py`
  - 把结构化面板同步到 Habitat 字段和内部派生字段。

## 5. `vlm/`

这里放统一的 VLM / LLM API 配置模板与本地配置文件。

- `navigation_system/config/vlm/vlm_api_config.yaml.template`
  - 标准模板。
- `navigation_system/config/vlm/vlm_api_config.yaml`
  - 本地实际运行配置，不入 git。
- `navigation_system/config/vlm/vlm_api_config_qwen_cache.yaml.template`
  - Qwen 显式缓存模板。
- `navigation_system/config/vlm/vlm_api_config_qwen_cache.yaml`
  - 本地 Qwen 缓存配置，不入 git。

## 推荐修改原则

- 改实验行为：优先改 `experiments/r2r_eval.yaml`
- 改 GPU、模型路径、传感器：改 `system/*.yaml`
- 改 detection / map / render 默认算法参数：改 `core/params/*`
- 改 VLM 服务商与模型：改 `vlm/*.yaml`
- 改配置结构与同步逻辑：改 `runtime/*`
