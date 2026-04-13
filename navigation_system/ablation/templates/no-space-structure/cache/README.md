# Cache Prompt Split

这里的文件是 Qwen 显式上下文缓存专用 prompt，不再从普通 prompt 运行时裁剪生成。

## 设计原则

- `*.system.prompt.md`
  - 放稳定、长文本、希望被显式缓存命中的规则和输出格式。
  - 只允许保留少量固定阈值占位符，例如 `{obs_blocked_m}`。
- `*.user.prompt.md`
  - 只放每次调用都会变化的动态文本。
  - 不再放长规则、输出格式、示例说明等稳定内容。
- 普通非缓存 prompt 与缓存 prompt 独立维护。
  - 修改缓存行为时，只改本目录文件即可。
  - 不需要再从普通 prompt 中“提取”可缓存部分。

## 文件对应关系

- `planning_initial.system.prompt.md`
  - 初始 thinking 的固定规则、输入说明、输出 JSON 结构、示例。
- `planning_initial.user.prompt.md`
  - 初始 thinking 的动态输入，目前只包含 `Global Task`。
- `planning_verify.system.prompt.md`
  - verify / replan 的固定规则、固定输出结构、示例。
  - 这里使用泛化表述 `Surrounding Views`，不再用动态的 `{verify_view_count}` 标题。
- `planning_verify.user.prompt.md`
  - verify / replan 的动态输入：`Stuck Notice`、`Global Task`、`Previous Subtask`、`Space Structure`。
- `action.system.prompt.md`
  - action 的固定规则、输出 JSON 结构、示例。
  - `Output Format` 固定放在 system prompt，不再放到动态 user prompt。
- `action.user.prompt.md`
  - action 的动态输入：当前 subtask、环境感知、空间结构、黄框候选检测、当前 action space。

## 最终发送给模型的结构

显式缓存调用最终会发成两条消息：

```json
{
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "<system_prompt>",
          "cache_control": {
            "type": "ephemeral"
          }
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "<user_prompt>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:..."
          }
        }
      ]
    }
  ]
}
```

## 调试产物

每次 cache 调用会在对应 step 目录下保存：

- `system_prompt.md`
- `user_prompt.md`
- 输入图片副本
- `vlm_info.json`
- `response.json`

其中：

- `system_prompt.md` 是被显式缓存的稳定文本。
- `user_prompt.md` 是每次变化的动态文本。
- 输入图片副本就是模型实际看到的压缩后版本，便于直接 debug。
- `vlm_info.json` 统一记录模型、token、耗时和缓存统计。
- `response.json` 是系统解析后的最终输出。
