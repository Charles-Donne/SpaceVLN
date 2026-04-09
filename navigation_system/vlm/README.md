# VLM Layer

`navigation_system/vlm/` 现在只保留和模型交互直接相关的模块。

```text
vlm/
├── api/
│   ├── api_client.py
│   ├── base_client.py
│   ├── config.py
│   └── qwen_context_cache_client.py
├── planning/
│   ├── planner.py
│   └── planner_qwen_cache.py
├── execution/
│   ├── executor.py
│   └── executor_qwen_cache.py
├── prompts/
│   ├── builders.py
│   ├── cache_builders.py
│   ├── common.py
│   └── templates/
├── contracts/
│   └── schema.py
├── reporting/
│   └── cache_report.py
├── interfaces.py
└── runtime_factory.py
```

## 职责边界

- `api/`
  - 负责 provider 通信、图片编码、结构化解析、`vlm_info.json`。
- `planning/`
  - 负责高层规划与 verify / replan。
- `execution/`
  - 负责低层动作决策。
- `prompts/`
  - 所有 prompt 模板与 builder。
  - cache 模板继续放在 `templates/cache/`。
- `contracts/`
  - 放 planner / action 共用 schema 和常量。
- `reporting/`
  - 汇总缓存命中和请求时延。
- `runtime_factory.py`
  - 统一创建标准版与 Qwen 显式缓存版模型栈。

## 明确不放在 VLM 里的内容

这些内容已经移出 `vlm/`：

- 结果目录和 artifact 路径
  - 现在在 `navigation_system/runtime/storage/`
- thinking 12-view 渲染
  - 现在在 `navigation_system/render/views/`
- 导航 GIF / 回放可视化
  - 现在在 `navigation_system/render/episode_visualization/`

这样 `vlm/` 的边界更干净：

- 只管模型怎么收输入、怎么发请求、怎么解析输出。
- 不再混放存储、渲染、回放逻辑。
