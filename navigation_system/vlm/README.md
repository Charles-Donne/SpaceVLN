# VLM Layer

`navigation_system/vlm` 现在按职责拆成 6 层：

```text
vlm/
├── api/
│   ├── base_client.py
│   ├── client.py
│   ├── config.py
│   └── qwen_cache.py
├── execution/
│   ├── executor.py
│   └── executor_qwen_cache.py
├── planning/
│   ├── planner.py
│   └── planner_qwen_cache.py
├── prompts/
│   ├── builders.py
│   ├── cache_builders.py
│   ├── common.py
│   ├── preview_cache_prompts.py
│   └── templates/
├── reporting/
│   └── cache_report.py
└── support/
    ├── artifacts.py
    ├── navigation_visualizer.py
    ├── schema.py
    └── thinking_view_renderer.py
```

## 职责边界

- `api/`
  - 只负责模型配置、图片编码、请求发送、provider 响应解析。
  - `client.py` 是通用入口。
  - `qwen_cache.py` 只保留 Qwen 显式上下文缓存逻辑。
- `planning/`
  - 只负责 thinking / verify / replan 模型调用。
- `execution/`
  - 只负责 action 模型调用和动作解析。
  - 已删除未使用的旧 `ActionParser`。
- `prompts/`
  - `builders.py` 管普通 prompt 构建。
  - `cache_builders.py` 管显式缓存 prompt 构建。
  - `common.py` 管模板加载和 cache prompt bundle。
  - `templates/` 只放 markdown 模板。
- `reporting/`
  - 只负责基于 `vlm_info.json` 汇总缓存命中与速度统计。
- `support/`
  - `artifacts.py` 管结果目录和落盘规则。
  - `schema.py` 管方向配置、动作映射、subtask payload 规范。
  - 其余两个文件只负责渲染与可视化。

## 约束

- 不再保留同义旧文件名，例如 `thinking.py`、`action.py`、`api_client.py`、`save_manager.py`。
- 不再从普通 prompt 运行时提取 cache prompt。
- 不通过 `navigation_system.vlm` 顶层做重型聚合导入，直接按子模块引用。
- 新增 provider / runtime 时，优先在现有层内扩展，不再增加平行的旧式包装层。
