# SpaceVLN

SpaceVLN 是一个在 Habitat 上运行的分层视觉语言导航系统。当前仓库已经按“单机单项目”使用场景收缩过一轮，主链路集中在规划、动作、建图、结果保存和报告生成这几层，不再保留旧的交互式入口和历史兼容脚本。

## 1. 环境准备

仓库默认按 `conda + Python 3.8` 使用，建议新建独立环境 `spacevln`：

```bash
conda create -n spacevln python=3.8 -y
conda activate spacevln
```

先安装与你机器 CUDA 匹配的 PyTorch。当前工作环境使用的是 CUDA 12.1 版本组合：

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2+cu121 \
  torchvision==0.16.2+cu121 \
  torchaudio==2.1.2+cu121
```

再安装仓库依赖：

```bash
cd /home/charlesdonne/project/nav_ws/SpaceVLN
pip install -r requirements.txt
```

`requirements.txt` 里默认依赖本地 Habitat：

```text
-e ../habitat-lab
```

所以请确认同级目录存在可用的 [`habitat-lab`](/home/charlesdonne/project/nav_ws/habitat-lab)。

## 2. 模型与数据

运行前需要准备两类外部资源：

- Habitat 数据与 R2R/SpaceVLN 评测数据。
- GroundedSAM / RepViT-SAM 权重。

默认路径定义在 [`default.py`](/home/charlesdonne/project/nav_ws/SpaceVLN/navigation_system/config/runtime/default.py)：

- `MAP.GROUNDING_DINO_CONFIG_PATH`
- `MAP.GROUNDING_DINO_CHECKPOINT_PATH`
- `MAP.SAM_CHECKPOINT_PATH`
- `MAP.RepViTSAM_CHECKPOINT_PATH`

如果你的权重路径不同，直接改这一个文件即可。

## 3. API 配置

标准运行使用统一配置文件：

[`vlm_api_config.yaml.template`](/home/charlesdonne/project/nav_ws/SpaceVLN/navigation_system/config/api/vlm_api_config.yaml.template)

复制一份后填写：

```bash
cp navigation_system/config/api/vlm_api_config.yaml.template \
   navigation_system/config/api/vlm_api_config.yaml
```

当前支持三类 provider：

- `dashscope`
- `openrouter`
- `openai`

OpenAI 兼容接口建议填写：

```yaml
provider: "openai"

openai:
  api_key: "env:OPENAI_API_KEY"
  base_url: "https://your-openai-compatible-endpoint"
  wire_api: "responses"
  llm_model: "gpt-5.4"
  vlm_model: "gpt-5.4-nano"
  reasoning_effort: "none"
```

Qwen 显式上下文缓存单独使用：

- [`vlm_api_config_qwen_cache.yaml`](/home/charlesdonne/project/nav_ws/SpaceVLN/navigation_system/config/api/vlm_api_config_qwen_cache.yaml)
- [`vlm_navigation_qwen_cache.py`](/home/charlesdonne/project/nav_ws/SpaceVLN/vlm_navigation_qwen_cache.py)
- [`vlm_navigation_qwen_cache.sh`](/home/charlesdonne/project/nav_ws/SpaceVLN/run_r2r/vlm_navigation_qwen_cache.sh)

缓存配置块示例：

```yaml
qwen_context_cache:
  enabled: true
  cache_type: "ephemeral"
  print_usage: false
  save_usage_json: true
  results_dir_suffix: "_cache"
```

## 4. 运行方式

标准运行：

```bash
bash run_r2r/vlm_navigation.sh 1 10 260 4
```

含义：

- 从 episode `1` 开始
- 连续跑 `10` 个 episode
- 每个 episode 最多 `260` 步
- 并行 `4` 个 worker

缓存版运行：

```bash
bash run_r2r/vlm_navigation_qwen_cache.sh 1 10 260 4
```

也支持直接传完整 CLI 参数：

```bash
python vlm_navigation.py \
  --exp-config navigation_system/config/experiments/r2r_eval.yaml \
  --episode-id 1 \
  --num-episodes 10 \
  --max-steps 260 \
  --parallel-workers 4 \
  --vlm-api-config navigation_system/config/api/vlm_api_config.yaml
```

报告生成：

```bash
bash run_r2r/vlm_report_range.sh 1 100
```

## 5. 结果目录

结果根目录会按模型组合自动生成：

```text
data/result/vlnce/<planner>__<action>/
```

显式缓存版会额外带后缀：

```text
data/result/vlnce/<planner>__<action>_cache/
```

当前结果布局是：

```text
data/result/vlnce/planner__action/
├── detail/
│   └── 1-100/
│       └── episode_1/
│           ├── thinking/
│           ├── action/
│           ├── visualization/
│           └── records/
│               └── result_latest.json
└── log/
    └── 1-100/
        └── episode_1.json
```

`detail/episode_x` 下保留的重点产物：

- `thinking/subtask_*`
  - `prompt.md` 或 `system_prompt.md + user_prompt.md`
  - 模型真实看到的图片副本
  - `provider_response.json`
  - 解析后的 `response.json`
  - 缓存版额外有 `cache_usage.json`
- `action/subtask_*`
  - 动作模型输入输出
- `visualization/`
  - 导航逐步可视化
  - `navigation.gif`
- `records/result_latest.json`
  - 当前这次 episode 的完整结果

`log/episode_x.json` 保存每个 episode 的最佳摘要结果，报告程序只依赖这一层。

## 6. 保存策略

当前默认策略已经收缩过：

- 保留模型输入输出调试产物。
- 保留导航可视化和 GIF。
- 保留 `result_latest.json` 与 `log/episode_x.json`。
- 默认不再额外保存整段 stdout 文本日志。
- 默认不再保存 `waypoint_memory.json`。
- 默认不再在 `records/` 下复制一份 `result.json`。

这些开关统一在 [`default.py`](/home/charlesdonne/project/nav_ws/SpaceVLN/navigation_system/config/runtime/default.py) 的 `MAP` 配置里：

```python
_C.MAP.SAVE_API_REQUEST_ARTIFACTS = True
_C.MAP.SAVE_NAVIGATION_STEP_IMAGES = True
_C.MAP.SAVE_NAVIGATION_GIF = True
_C.MAP.SAVE_EPISODE_STDOUT_LOG = False
_C.MAP.SAVE_WAYPOINT_MEMORY = False
_C.MAP.SAVE_BEST_RESULT_COPY = False
```

如果你想重新打开 stdout 文本日志，只需要把 `SAVE_EPISODE_STDOUT_LOG` 改成 `True`。

## 7. 当前架构

主链路目录如下：

```text
SpaceVLN/
├── run_r2r/
│   ├── common.sh
│   ├── vlm_navigation.sh
│   ├── vlm_navigation_qwen_cache.sh
│   └── vlm_report_range.sh
├── navigation_system/
│   ├── controllers/
│   │   ├── base_navigation_controller.py
│   │   ├── runtime_state.py
│   │   ├── vlm_navigation_controller.py
│   │   └── vlm_navigation_controller_qwen_cache.py
│   ├── runtime/
│   │   ├── runner.py
│   │   ├── runner_qwen_cache.py
│   │   └── results_report.py
│   ├── vlm/
│   │   ├── api/
│   │   ├── planning/
│   │   ├── execution/
│   │   ├── prompts/
│   │   └── support/
│   ├── mapping/
│   ├── detection/
│   └── visualization/
├── vlm_navigation.py
└── vlm_navigation_qwen_cache.py
```

模块职责：

- `controllers/`
  负责 episode 调度、thinking/action 切换、状态维护。
- `runtime/`
  负责 CLI、并行 worker、结果汇总和报告。
- `vlm/api/`
  负责 provider 配置解析与请求发送。
- `vlm/planning/`
  负责高层规划模型接入。
- `vlm/execution/`
  负责低层动作模型接入。
- `vlm/prompts/`
  负责 prompt 模板。
- `vlm/support/`
  负责保存、thinking 视图渲染和导航 GIF。
- `mapping/`
  负责语义建图和空间拓扑。
- `detection/`
  负责 GroundedSAM / RepViT-SAM 检测。
- `visualization/`
  负责地图、检测和观测可视化。

## 8. 运行流程

系统主循环是：

1. 环视并更新地图。
2. Planner 根据多视角图像和 global map 生成下一子任务。
3. Action model 根据 detection 视图与结构化上下文输出动作。
4. 执行动作并更新地图。
5. 到达子任务边界后重新 verify / replan。
6. 直到任务完成或 episode 预算耗尽。

缓存版与标准版的差异只有两点：

- planner/action adapter 不同。
- 结果目录后缀不同，并且缓存版会额外保存缓存命中统计。

## 9. 当前清理结论

这轮整理后，仓库里已经去掉了几类明显无效的内容：

- 已失效的 `interactive_navigation.sh`
- 无引用的 `audit_result_layout.py`
- 无引用的 `observation_collector.py`

如果后面还要继续做更深一轮瘦身，优先建议沿着这三条继续做：

- 继续拆 `vlm_navigation_controller.py`
- 继续收紧结果产物种类
- 继续把 provider 特定逻辑从通用 API client 分离
