# VLNCE Prompt Module

This directory contains the prompt definitions and renderers for the main VLNCE task.

## Structure

- `builders.py`
  - Renders the shared `system` + `user` prompt bundles used by both standard and context-cache runtimes.
- `templates/`
  - Stores the single source of truth for prompt markdown files.
- `navgbench/`
  - Adds NavGBench-specific planner prompt overlays. Complex/grounded
    NavGBench instructions get an extra route-compression policy; R2R prompts
    and simple NavGBench instructions stay on the base VLNCE wording.

## Prompt Files

- `templates/planning_initial.system.prompt.md`
  - Stable initial-planning system instructions.
- `templates/planning_initial.user.prompt.md`
  - Dynamic initial-planning user content.
- `templates/planning_verify.system.prompt.md`
  - Stable verify / replanning system instructions.
- `templates/planning_verify.user.prompt.md`
  - Dynamic verify / replanning user content.
- `templates/executor.system.prompt.md`
  - Stable executor system instructions.
- `templates/executor.user.prompt.md`
  - Dynamic executor user content.

## Runtime Behavior

Standard and context-cache runtimes now use the same prompt builders and the same markdown files. The context-cache runtime only changes transport/cache behavior; it does not use a separate prompt version.
