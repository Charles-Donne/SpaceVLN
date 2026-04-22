# VLNCE Prompt Module

This directory contains the prompt definitions and renderers for the main VLNCE task.

## Structure

- `builders.py`
  - Renders the standard non-cache prompts from the combined markdown templates in `templates/`.
- `cache_builders.py`
  - Renders the explicit-cache prompt bundles used by runtimes that send stable `system` text plus dynamic `user` text separately.
- `templates/`
  - Stores the prompt markdown files used by the standard renderer.
- `templates/cache/`
  - Stores the explicit-cache prompt split for the same task family.

## Prompt Files

- `templates/planning_initial.prompt.md`
  - Combined initial-planning prompt for non-cache calls.
- `templates/planning_verify.prompt.md`
  - Combined verify / replanning prompt for non-cache calls.
- `templates/action_execution.prompt.md`
  - Combined action-execution prompt for non-cache calls.

## Cache Split

The explicit-cache version keeps stable long-form instructions in `templates/cache/*.system.prompt.md` and per-call dynamic text in `templates/cache/*.user.prompt.md`.

See `templates/cache/README.md` for the exact split and message structure.
