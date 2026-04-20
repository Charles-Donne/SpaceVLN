# Navigation System Architecture

`navigation_system/` is organized by functional responsibility first, and by
task-specific overlays second.

## 1. Top-Level Layout

```text
navigation_system/
├── controller/
│   └── object_navigation/
├── env/
│   └── object_navigation/
├── runtime/
│   ├── storage/
│   └── object_navigation/
├── vlm/
│   ├── planning/
│   │   └── object_navigation/
│   ├── execution/
│   │   └── object_navigation/
│   └── prompts/
│       └── object_navigation/
├── detection/
├── space/
├── render/
└── config/
```

The default VLNCE / CE pipeline remains in the shared functional modules.
OVON-specific behavior now lives under task-scoped subpackages inside the same
functional areas.

## 2. Functional Responsibilities

### `controller/`

- Owns control flow, episode lifecycle, stopping logic, and integration across
  detection, spatial memory, rendering, and VLM calls.
- Task-specific controller overlays live in subpackages such as
  `controller/object_navigation/`.

### `env/`

- Owns Habitat environment wiring and environment adapters.
- Task-specific episode facades and instruction adapters live in
  `env/object_navigation/`.

### `runtime/`

- Owns CLI entrypoints, episode scheduling, result directory selection, and
  evaluation/report generation.
- Task-specific launch/runtime helpers live in `runtime/object_navigation/`.
- Shared artifact layout lives in `runtime/storage/`.

### `vlm/`

- `planning/`: high-level subtask planning and replanning.
- `execution/`: low-level action selection.
- `prompts/`: prompt builders and Markdown templates.
- `api/`: provider clients and request persistence.
- `contracts/`: shared response schema and parsing helpers.
- `reporting/`: cache and API reporting.

Task-specific prompt/planner/executor variants live under matching subpackages,
for example `vlm/planning/object_navigation/`.

### `detection/`

- Owns GroundingDINO / SAM integration and detection-related vendor code.

### `space/`

- Owns geometry, semantic mapping, topology, landmarks, and textual spatial
  summaries.

### `render/`

- Owns model-facing views, top-down maps, and human-facing episode
  visualizations.

### `config/`

- `experiments/`: experiment-level YAML defaults.
- `runtime/`: derived-field sync and runtime mutation helpers.
- `vlm/`: API/provider configuration.
- `core/`: static constants and reusable parameter defaults.

## 3. Main Execution Path

1. A launcher script under `run_navigation/` selects the task entrypoint.
2. `runtime/*` resolves configs, runtime profile, results directories, and
   episode selection.
3. `controller/*` drives the navigation loop.
4. `vlm/planning/*` proposes subtasks and `vlm/execution/*` chooses actions.
5. `runtime/storage/*` saves `detail/`, `records/`, `log/`, and summaries.
6. `runtime/results_report.py` aggregates offline evaluation metrics.

## 4. Design Rules

- Shared logic stays shared; only true task-specific behavior goes into
  task-scoped subpackages.
- Control flow stays in `controller/`; spatial logic stays in `space/`;
  rendering stays in `render/`.
- Prompt templates live only under `vlm/prompts/`.
- Result-path policy is centralized in `runtime/storage/`.
- `run_navigation/` is the canonical shell entry surface for both VLNCE and
  OVON.
- Task-specific code should import canonical module paths directly; the tree no
  longer keeps compatibility wrappers for the old object-navigation layout.
