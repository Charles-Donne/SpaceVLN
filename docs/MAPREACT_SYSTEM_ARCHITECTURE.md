# MapReAct-VLN System Architecture

## 1. System Goal

MapReAct-VLN is a layered navigation system that combines:

1. Habitat simulation and episode control
2. RGB-D detection and semantic mapping
3. LLM high-level subtask planning
4. VLM low-level action selection
5. Visualization, logging, and result export

The design target is: keep the CLI thin, keep perception/mapping/planning/action/rendering separate, and reuse the same spatial transforms across mapping, prompting, and visualization.

---

## 2. Layered Structure

### 2.1 Entry Layer

- `vlm_navigation.py`
  - CLI entry only
  - parses arguments
  - delegates runtime orchestration to `vlnce_baselines/vlm/runner.py`

- `vlnce_baselines/vlm/runner.py`
  - batch episode selection
  - controller construction
  - per-episode execution
  - summary/report generation

Rule: no mapping, planning, detection, or rendering logic should live in the CLI entry.

### 2.2 Control Layer

- `vlnce_baselines/interactive_navigation_controller.py`
  - simulator stepping
  - observation collection
  - mapper / detector / visualizer wiring
  - shared navigation primitives

- `vlnce_baselines/vlm_navigation_controller.py`
  - MapReAct runtime loop
  - look-around -> think -> auto-rotate -> act -> verify/replan
  - prompt input assembly
  - subtask state and retry policy

Rule: controller coordinates modules; it should not duplicate geometry or rendering math already owned elsewhere.

### 2.3 Perception + Mapping Layer

- `vlnce_baselines/detection/`
  - Grounded-SAM / detector wrappers

- `vlnce_baselines/mapping/mapper.py`
  - high-level mapping facade
  - updates map state from observation batches
  - owns floor extraction and full/local map state packaging

- `vlnce_baselines/mapping/semantic_mapping.py`
  - dense map tensor update
  - RGB-D projection to map
  - semantic accumulation
  - rotated render map export

Rule: world/map state is the source of truth for obstacle distance, landmark distance, and landmark angle.

### 2.4 VLM / LLM Layer

- `vlnce_baselines/vlm/thinking.py`
  - planning API client
  - initial subtask generation
  - verification / replanning

- `vlnce_baselines/vlm/action.py`
  - action API client
  - low-level action decision

- `vlnce_baselines/vlm/prompts.py`
  - thinking prompt template

- `vlnce_baselines/vlm/action_prompt.py`
  - action prompt template

Rule: prompts describe policy and output format; they should not contain implementation-specific geometry that can be computed directly in code.

### 2.5 Visualization Layer

- `vlnce_baselines/visualization/visualizer.py`
  - global/local map rendering
  - detection overlay rendering
  - step asset saving

- `vlnce_baselines/visualization/landmark_overlay.py`
  - action-view partition guides
  - bbox label ordering and rendering
  - bottom-strip line building and rendering

- `vlnce_baselines/visualization/map_projection.py`
  - shared world -> rotated map -> display projection

- `vlnce_baselines/visualization/obstacle_analysis.py`
  - shared rotated obstacle mask build
  - shared depth-sampled action obstacle distance scan
  - shared panorama distance scan utilities

Rule: all renderers and prompt inputs must reuse the same projection and obstacle analysis utilities.

### 2.6 Persistence Layer

- `vlnce_baselines/vlm/save_manager.py`
  - JSON / image / thinking record saving

- `vlnce_baselines/vlm/navigation_visualizer.py`
  - composite navigation output
  - GIF / final visualization packaging

---

## 3. Runtime Data Flow

### 3.1 Episode Lifecycle

1. `vlm_navigation.py` parses CLI args
2. `runner.py` resolves episode IDs and builds per-episode config
3. `VLMNavigationController` resets episode and initializes modules
4. controller performs 12-view look-around
5. detector + mapper update semantic map
6. planner produces the nearest executable subtask
7. controller auto-rotates if needed
8. action model consumes current RGB + map + obstacle distances + history summary
9. executed action updates observation and map
10. verify/replan loop repeats until stop or episode end

### 3.2 Map Pipeline

1. RGB-D observation enters detector + semantic mapper
2. semantic channels are accumulated in `Semantic_Mapping`
3. `mapper.py` exposes:
   - full rotated render map
   - floor mask
   - waypoint history
   - crop offset / pose metadata
4. visualization and prompt formatting read from the same map state

### 3.3 Landmark Pipeline

1. detector outputs bbox / mask / label / score
2. depth-guided projection estimates world position
3. world position is stored as landmark instance state
4. rendering converts world position back into current rotated map coordinates
5. action / thinking inputs use map-derived distance and angle, not raw bbox angle

---

## 4. Current Module Boundaries

### 4.1 What `main` Should Do

- parse CLI args
- call runner
- exit with status code

### 4.2 What `runner` Should Do

- batch orchestration
- episode selection
- controller creation
- summary/report generation

### 4.3 What controller should do

- state machine / orchestration
- retries
- module interaction
- result bookkeeping

### 4.4 What controller should not do

- duplicate render-space transforms
- duplicate obstacle raycasting math
- duplicate reporting code already handled by runner/save modules

---

## 5. Refactor Rules

### 5.1 Shared Geometry

Any logic that converts among:

- world pixels
- rotated map pixels
- global display coordinates
- local display coordinates

must live in `map_projection.py`.

### 5.2 Shared Obstacle Scans

Any logic that computes:

- action-view obstacle distances
- 12-direction panorama obstacle distances
- rotated obstacle masks

must live in `obstacle_analysis.py`.

### 5.3 Prompt Inputs

Prompt inputs should consume:

- map-derived obstacle distances
- map-derived waypoint summaries
- map-derived landmark distance/angle

They should not recompute geometry from rendered images unless used only as a fallback display hint.

---

## 6. Recent Cleanup in This Pass

### 6.1 Decoupled Entry Runtime

- moved batch runtime orchestration out of `vlm_navigation.py`
- added reusable runner module: `vlnce_baselines/vlm/runner.py`

### 6.2 Slimmed Visualization Responsibilities

- kept `visualizer.py` as the rendering coordinator
- moved action landmark overlay rendering details into `vlnce_baselines/visualization/landmark_overlay.py`
- kept geometry in `map_projection.py` and depth-distance sampling in `obstacle_analysis.py`

### 6.2 Removed Duplicate Geometry / Scan Logic

- added `vlnce_baselines/visualization/map_projection.py`
- added `vlnce_baselines/visualization/obstacle_analysis.py`
- updated controller and visualizer to reuse them

### 6.3 Removed Redundant / Dead Paths

- removed unreachable legacy waypoint rendering code in `vlnce_baselines/vlm_navigation_controller.py`
- removed duplicated manual projection blocks in `vlnce_baselines/visualization/visualizer.py`
- removed unused map-usage helper path from `visualizer.py`

---

## 7. Performance Notes

### 7.1 Reduced Repeated Math

- one shared projector replaces repeated trig-heavy coordinate transforms
- one shared obstacle mask builder replaces repeated `flipud + resize` code
- one shared raycast utility serves both action and thinking modes

### 7.2 Safer State Reuse

- controller, prompts, and rendering now align more tightly around the same map state
- fewer duplicated code paths means less risk of display-space / prompt-space mismatch

---

## 8. Extension Points

### 8.1 If adding a new planner

- keep API logic inside `vlnce_baselines/vlm/`
- let controller call a stable planner interface

### 8.2 If adding a new render mode

- reuse `MapVisualizer`
- reuse `RotatedMapProjector`
- avoid adding transform code in controllers

### 8.3 If adding a new spatial feature

- store world-space truth first
- derive view-space / prompt-space data from that source

---

## 9. Recommended Next Refactor Targets

1. split `VLMNavigationController` into smaller services:
   - planning coordinator
   - action coordinator
   - landmark state manager
2. formalize a `NavigationState` data object instead of passing many loose dict fields
3. move prompt input assembly into dedicated formatter modules
4. add lightweight tests for:
   - world/local/global projection consistency
   - obstacle distance scan consistency
   - landmark instance merge/update logic
