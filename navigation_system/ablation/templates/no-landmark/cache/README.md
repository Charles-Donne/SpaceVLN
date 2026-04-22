# Cache Prompt Split

These files define the explicit-context-cache prompt split used for Qwen-style cached calls. They are maintained directly in this directory rather than being extracted at runtime from the non-cache prompt.

## Design Principles

- `*.system.prompt.md`
  - Stores the stable long-form rules, formatting constraints, and examples that should benefit from explicit cache reuse.
  - May keep a small number of fixed threshold placeholders such as `{obs_blocked_m}`.
- `*.user.prompt.md`
  - Stores only the per-call dynamic text.
  - Should not duplicate long stable rules, output schemas, or examples.
- Non-cache prompts and cache prompts are maintained independently.
  - Update files in this directory when changing explicit-cache behavior.
  - No runtime extraction step is required from the non-cache prompt.

## File Mapping

- `planning_initial.system.prompt.md`
  - Stable rules, input legend, JSON schema, and examples for initial planning.
- `planning_initial.user.prompt.md`
  - Dynamic initial-planning input, usually the current `Global Task`.
- `planning_verify.system.prompt.md`
  - Stable rules, JSON schema, and examples for verify / replan.
  - Uses the generalized `Surrounding Views` wording instead of a runtime-specific view-count title.
- `planning_verify.user.prompt.md`
  - Dynamic verify / replan inputs such as the current task, previous subtask, optional notice block, and any task-variant runtime summaries.
- `action.system.prompt.md`
  - Stable action rules, JSON schema, and examples.
  - Keeps the `Output Format` in the cached system prompt.
- `action.user.prompt.md`
  - Dynamic action inputs such as the current subtask, perception summaries, optional runtime summaries, candidate detections, and the current action space.

## Final Message Structure

An explicit-cache call is sent as two messages:

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

## Debug Artifacts

Each cache call saves prompt/debug artifacts in the corresponding step directory, typically including:

- `system_prompt.md`
- `user_prompt.md`
- input image copies
- `vlm_info.json`
- `response.json`

Where:

- `system_prompt.md` is the stable text intended for explicit cache reuse.
- `user_prompt.md` is the dynamic per-call text.
- Input image copies reflect the actual compressed image payload sent to the model.
- `vlm_info.json` records model name, tokens, latency, and cache stats.
- `response.json` stores the parsed final model output.
