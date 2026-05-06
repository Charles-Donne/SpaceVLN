# OVON Prompt Module

This directory contains the prompt definitions and renderers for the main object-navigation / OVON task.

## Structure

- `builders.py`
  - Renders the shared `system` + `user` prompt bundles used by both standard and context-cache runtimes.
  - Injects OVON-specific thresholds and task wording into the prompt text.
- `common.py`
  - Task-local template loading helpers for the OVON prompt family.
- `templates/`
  - Stores the single source of truth for prompt markdown files.

## Prompt Files

- `templates/planning_initial.system.prompt.md`
  - Stable initial search / planning system instructions.
- `templates/planning_initial.user.prompt.md`
  - Dynamic initial search / planning user content.
- `templates/planning_verify.system.prompt.md`
  - Stable verify / replanning system instructions.
- `templates/planning_verify.user.prompt.md`
  - Dynamic verify / replanning user content.
- `templates/action.system.prompt.md`
  - Stable action-execution system instructions.
- `templates/action.user.prompt.md`
  - Dynamic action-execution user content.

## Runtime Behavior

Standard and context-cache runtimes now use the same prompt builders and the same markdown files. The context-cache runtime only changes transport/cache behavior; it does not use a separate prompt version.
