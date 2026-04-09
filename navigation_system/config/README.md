# Config Layout

`navigation_system/config/` 现在固定分成 5 层，边界如下。

## 1. `experiments/`

这里只放实验覆盖项，是日常最应该改的入口。

当前主入口是 [`r2r_eval.yaml`](/home/charlesdonne/project/nav_ws/SpaceVLN/navigation_system/config/experiments/r2r_eval.yaml)，按功能分区：

- `TASK`: 当前实验覆盖的任务入口
- `PATHS`: 当前实验结果目录等路径覆盖
- `OUTPUT`: 保存策略
- `CONTROL`: 恢复、停滞、停止逻辑
- `EVAL`: split、success distance 等评测参数

## 2. `system/`

这里只放外置系统参数，也就是部署层默认值，不放算法阈值。

- `00_runtime.yaml`: 任务入口、GPU/并行环境、默认目录
- `10_detection_models.yaml`: GroundedSAM / RepViT-SAM 模型路径与开关
- `20_space_sensor.yaml`: 相机 HFOV、分辨率、agent height 等传感器参数

## 3. `core/`

这里只放静态定义和算法默认值，不保存运行时状态。

- `params/`: detection / spatial / rendering / semantic / thresholds 等默认行为参数
- `categories.py`: 静态类别定义
- `constants.py`: 对旧调用点保留的常量导出层
- `setup.py`: 配置二次设置助手

## 4. `runtime/`

运行时装配层，不作为实验面板直接修改。

- [`panels.py`](/home/charlesdonne/project/nav_ws/SpaceVLN/navigation_system/config/runtime/panels.py)
  定义结构化面板和代码级回退默认值。
- [`default.py`](/home/charlesdonne/project/nav_ws/SpaceVLN/navigation_system/config/runtime/default.py)
  负责加载 `system/*.yaml + experiments/*.yaml` 并组装总配置。
- [`sync.py`](/home/charlesdonne/project/nav_ws/SpaceVLN/navigation_system/config/runtime/sync.py)
  把结构化面板同步到 Habitat 字段和内部 `MAP` 派生字段。

## 5. `vlm/`

VLM / LLM API 统一配置层。

- [`vlm_api_config.yaml.template`](/home/charlesdonne/project/nav_ws/SpaceVLN/navigation_system/config/vlm/vlm_api_config.yaml.template)
  标准模板
- [`vlm_api_config.yaml`](/home/charlesdonne/project/nav_ws/SpaceVLN/navigation_system/config/vlm/vlm_api_config.yaml)
  实际运行配置
- [`vlm_api_config_qwen_cache.yaml`](/home/charlesdonne/project/nav_ws/SpaceVLN/navigation_system/config/vlm/vlm_api_config_qwen_cache.yaml)
  Qwen 显式缓存专用配置

## 推荐修改原则

- 改实验行为：优先改 `experiments/r2r_eval.yaml`
- 改 GPU、模型路径、传感器：改 `system/*.yaml`
- 改 detection/map/render 默认算法参数：改 `core/params/*`
- 改 VLM 服务商与模型：改 `vlm/*.yaml`
- 改配置结构和同步逻辑：改 `runtime/*`
