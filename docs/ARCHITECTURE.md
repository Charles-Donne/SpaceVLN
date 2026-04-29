# Navigation System Architecture

`navigation_system/` is organized by functional responsibility first, and by
task-specific overlays second.

## 1. Top-Level Layout

```text
navigation_system/
├── controller/
│   └── agent/
├── env/
│   └── vlnce/
│   └── object_navigation/
├── runtime/
│   ├── vlnce/
│   ├── storage/
│   └── object_navigation/
├── vlm/
│   ├── vlnce/
│   ├── planning/
│   │   └── vlnce/
│   │   └── object_navigation/
│   ├── execution/
│   │   └── vlnce/
│   │   └── object_navigation/
│   └── prompts/
│       └── vlnce/
│       └── object_navigation/
├── detection/
├── space/
├── render/
└── config/
```

Shared logic remains in the functional root. Task-specific logic now lives in
explicit task overlays such as `vlnce/` and `object_navigation/` inside the
same functional areas.

## 2. Functional Responsibilities

### `controller/`

- Owns control flow, episode lifecycle, stopping logic, and integration across
  detection, spatial memory, rendering, and VLM calls.
- The reusable navigation loop lives in `controller/agent/`.
- Task/benchmark overlays, such as OVON-specific stop-gate behavior, live
  under their runtime plugin packages instead of the shared controller tree.

### `env/`

- Owns Habitat environment wiring and benchmark adapters.
- VLNCE benchmark adapters live under `env/vlnce/r2r/` and
  `env/vlnce/navgbench/`; OVON lives under `env/object_navigation/ovon/`.

### `runtime/`

- Owns CLI entrypoints, episode scheduling, result directory selection, and
  evaluation/report generation.
- VLNCE task profiles live in `runtime/vlnce/`; benchmark runners live under
  `runtime/vlnce/r2r/` and `runtime/vlnce/navgbench/`.
- OVON runtime orchestration lives in `runtime/object_navigation/ovon/`.
- Shared artifact layout lives in `runtime/storage/`.

### `vlm/`

- `planning/`: high-level subtask planning and replanning.
- `execution/`: low-level action selection.
- `prompts/`: prompt builders and Markdown templates.
- `api/`: provider clients and request persistence.
- `contracts/`: shared response schema and parsing helpers.
- `reporting/`: cache and API reporting.

Task-specific prompt/planner/executor variants live under matching subpackages,
for example `vlm/planning/vlnce/` and `vlm/planning/object_navigation/`.

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

1. A launcher script under `run_navigation/` calls `navigation_agent.py`.
2. `runtime/<task>/<benchmark>/*` resolves benchmark-specific episode
   selection, environment adapters, results directories, and reports.
3. `controller/agent/*` drives the shared navigation loop; benchmark plugins
   wrap it only when behavior truly differs.
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
- `navigation_agent.py` is the only Python entrypoint; `run_navigation/`
  launchers are thin argument translators.
- Task-specific code should import canonical module paths directly; the tree no
  longer keeps compatibility wrappers for the old object-navigation layout.
