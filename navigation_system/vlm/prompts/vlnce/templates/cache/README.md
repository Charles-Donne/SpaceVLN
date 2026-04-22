# VLNCE Cache Prompt Split

These files define the explicit-context-cache prompt split for the main VLNCE prompt family.

## Design Principles

- `*.system.prompt.md`
  - Stores stable long-form rules, formatting constraints, and examples that benefit from cache reuse.
  - May keep a small set of fixed threshold placeholders such as `{obs_blocked_m}`.
- `*.user.prompt.md`
  - Stores only the dynamic text that changes on each call.
  - Should not duplicate long stable rules or examples.

## File Mapping

- `planning_initial.system.prompt.md`
  - Stable initial-planning rules, input legend, output schema, and examples.
- `planning_initial.user.prompt.md`
  - Dynamic initial-planning input, typically the current `Global Task`.
- `planning_verify.system.prompt.md`
  - Stable verify / replanning rules, schema, and examples.
- `planning_verify.user.prompt.md`
  - Dynamic verify / replanning input such as the current task, previous subtask, optional notice block, and current structure summary.
- `action.system.prompt.md`
  - Stable action-execution rules, schema, and examples.
- `action.user.prompt.md`
  - Dynamic action input such as the current subtask, perception summaries, candidate detections, and current action space.

## Final Message Structure

An explicit-cache call is sent as separate `system` and `user` messages:

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

Cache calls typically save:

- `system_prompt.md`
- `user_prompt.md`
- input image copies
- `vlm_info.json`
- `response.json`

Where `system_prompt.md` is the stable cached text, `user_prompt.md` is the per-call dynamic text, and `vlm_info.json` / `response.json` record runtime metadata and the parsed model output.
