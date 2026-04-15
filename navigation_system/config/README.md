# Config Layout

`navigation_system/config/` 现在只保留一套结构化配置，不再保留旧的 flat 兼容字段，也不再有 `panels.py` / `sync.py` 这类历史同步层。

建议按下面的优先级修改。

## 1. `experiments/`

这里放实验覆盖项，是**最推荐的日常入口**。

当前主入口：

- `navigation_system/config/experiments/r2r_eval.yaml`

主要分区：

- `OUTPUT`
- `CONTROL`
- `EVAL`

这里现在只放**实验行为覆盖项**，不再重复写任务路径、结果根目录、GPU 之类系统默认值。

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

- `navigation_system/config/runtime/default.py`
  - 唯一装配入口：加载 `system/*.yaml + experiments/*.yaml`，并生成 Habitat task / map runtime 派生字段。

## 5. `vlm/`

这里放统一的 VLM / LLM API 配置模板与本地配置文件。

- `navigation_system/config/vlm/vlm_api_config.yaml.template`
  - 标准模板。
- `navigation_system/config/vlm/vlm_api_config.yaml`
  - 本地实际运行配置，不入 git。
- `navigation_system/config/vlm/vlm_api_config_context_cache.yaml.template`
  - 显式缓存模板。
- `navigation_system/config/vlm/vlm_api_config_context_cache.yaml`
  - 本地显式缓存配置，不入 git。

## 推荐修改原则

- 改实验行为：优先改 `experiments/r2r_eval.yaml`
- 改 GPU、模型路径、传感器：改 `system/*.yaml`
- 改 detection / map / render 默认算法参数：改 `core/params/*`
- 改 VLM 服务商与模型：改 `vlm/*.yaml`
- 改配置结构与派生逻辑：改 `runtime/default.py`

## 结果路径规则

- 默认结果总根目录只在 `navigation_system/config/system/00_runtime.yaml` 的 `PATHS.RESULTS_ROOT` 定义一次。
- 普通 VLN-CE 运行会自动保存到 `<RESULTS_ROOT>/vlnce/<model-dir>`。
- 消融运行会自动保存到 `<RESULTS_ROOT>/vlnce/ablation/<ablation-slug>/<model-dir>`。
- `PATHS.RESULTS_DIR` 不写在 YAML 里；它是运行时解析后的最终目录。
- `--results-root DIR` 只覆盖总根目录，仍保留 `vlnce/...` 结构。
- `--results-dir DIR` 是高级最终目录覆盖项，指定后会跳过自动结构化子目录。

## 当前原则

- 结构化字段是唯一真源：如 `config.RUNTIME.TORCH_GPU_ID`、`config.PATHS.RESULTS_DIR`
- 不再维护旧式顶层字段：如 `config.TORCH_GPU_ID`、`config.RESULTS_DIR`
- 不再保留为兼容旧代码而存在的冗余配置层
