# SpaceVLN Ablation Runtime

`navigation_system/ablation/` contains subtractive ablations for the VLNCE
pipeline only. The goal is to disable specific inputs or reasoning stages
without mutating the canonical main-system prompt templates in place.

## Scope

- Keep the shared runtime and controller stack intact.
- Reuse the original prompt families as the source of truth.
- Materialize ablation prompt variants as static Markdown under
  `navigation_system/ablation/templates/`.
- Route ablation runs through dedicated runtime/config/model adapters under
  `navigation_system/ablation/`.

## Layout

```text
navigation_system/ablation/
├── configs/      # ablation YAML presets
├── prompts/      # prompt assembly for ablation variants
├── models/       # planner / executor / controller adapters
├── render/       # ablation-specific render exposure
├── runtime/      # batch rules, profiles, result routing
├── templates/    # static copied-and-trimmed prompt templates
├── tools/        # maintenance helpers such as prompt audit
└── presets.py    # preset registry
```

## Supported Presets

- `landmark`
- `space_structure`
- `both`
- `planning_reasoning` / `thinking_reasoning`
- `action_reasoning`
- `planning_action_reasoning` / `thinking_action_reasoning`

Each preset also has a canonical YAML file under
`navigation_system/ablation/configs/`.

## Behavioral Summary

- `landmark`
  - removes landmark detections and landmark-focused render inputs from
    planning and action prompts.
- `space_structure`
  - removes space-structure text and related thinking-map overlays.
- `planning_reasoning`
  - removes the planning-side reasoning scaffold while preserving output schema,
    examples, and safety constraints.
- `action_reasoning`
  - removes the action-side reasoning scaffold while preserving action-space and
    output-format constraints.
- `planning_action_reasoning`
  - removes both planning-side and action-side reasoning scaffolds while
    keeping the structured response contract stable.

## Entrypoints

- `vlm_navigation.py --ablation ...`
- `bash run_navigation/vlnce.sh --ablation ...`

Examples:

```bash
bash run_navigation/vlnce.sh --ablation landmark 1 10
bash run_navigation/vlnce.sh --ablation planning_reasoning 1 10
bash run_navigation/vlnce.sh --runtime context_cache --ablation space_structure 1 10
```

You can also pass a YAML path directly:

```bash
bash run_navigation/vlnce.sh \
  --ablation navigation_system/ablation/configs/no_landmark.yaml \
  1 10
```

## Validation

Audit the ablation prompt tree with:

```bash
python navigation_system/ablation/tools/prompt_audit.py
```

## Result Layout

Ablation runs are stored under:

```text
result/vlnce/ablation/<ablation_name>/<model_name>/
```

The runtime also writes:

- `ablation/manifest.json`
- `ablation/config.yaml`
