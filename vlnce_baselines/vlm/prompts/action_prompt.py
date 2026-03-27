"""
动作执行提示词模板
==================
用于VLM低层动作决策的提示词模板

动作参数与interactive_navigation保持一致：
- TURN_LEFT/RIGHT: 30°（12步×30°=360°）
- MOVE_FORWARD: {0.25m, 0.5m, 0.75m, 1.0m, 1.25m}
"""

ACTION_EXECUTION_PROMPT = """You are the action execution module for Vision-Language Navigation. Analyze the environment and decide the next action.

# Current Subtask
**Destination**: {subtask_destination}
**Instruction**: {subtask_instruction}

# Space Structure
{waypoint_summary}

# Subtask Progress Summary
{progress_summary}

# Previous Step Analysis
{previous_action_reason}

# Visual Observations

You are provided with 1 image:

**Current View (front-facing, RGB HFOV about 79°)** — Object detection overlaid with 3 obstacle-distance lines:
- Directions: Left 30deg, FRONT, Right 30deg
- Red = nearest obstacle <0.5m (blocked), Yellow = 0.5-2m (caution), Green = >2m (open)
- **Yellow bounding box**: candidate subtask-landmark detection ({detected_landmarks}); first judge whether it is valid task evidence or noise
- **Bottom white strip** (if present): ranked landmark entries plus reachable `space waypoint` cues. Landmark entries are `vis/off vis + landmark name + distance + direction + confidence`; `vis` = detected now, `off vis` = mapped earlier but outside the view

# Reasoning Process:

1. **Current View Content**: Analyze the marked landmarks (if present) with the current view. From near to far, judge whether visible cues belong to the current space or a farther transition/target space. Then compare FRONT, Left 30deg, and Right 30deg to decide where the destination or correct transition most likely is. Mention only visible evidence.
2. **Bottom Strip + Space Structure**: If the bottom strip is present, read each landmark and space waypoint. Judge landmark validity, distance, direction, and whether it signals arrival. For each space waypoint, judge whether it is current / next / behind, whether it matches the destination space, and whether it should be used or avoided.
3. **Current Position + Arrival Check**: Use nearby landmarks, valid detections, visible space waypoint cues, and the provided space structure to confirm the current position and where the subtask destination is relative to you. Treat `Destination` as the actual goal of the current stage. Treat `Instruction` only as the route relation for reaching that destination: for same-space pass/go-by/around pieces, the cue is intermediate and the destination remains the stage goal; for cross-space enter pieces, finish the current enter-stage destination before any later-stage target. Decide which side is task-aligned and which side is wrong or backtracking. Then judge arrival under the rule below; if reached, stop, otherwise continue toward the destination.
   **Arrival rule**: treat the subtask as reached only when the actual current subtask destination is already reached, or when the detected landmark itself is the destination and close enough to stop. Do not stop at an intermediate cue/opening/pass-by landmark if the instruction says to pass / go through / go around / cross it and then continue toward a farther destination. If the intermediate cue/opening is already beside/behind and the current stage destination is still ahead, keep moving toward the destination instead of re-centering on the old cue. For solid-landmark destinations, stopping is allowed within about 0.75m or when already clearly at hand. For opening-like destinations (entrance / doorway / hallway), stop only within about 0.5m or once the opening is already >90deg / behind, meaning the opening-stage destination has been passed through.
4. **Action Decision + Obstacle Avoidance**: Choose one safe immediate action that matches the subtask instruction and moves toward the destination. Apply the grouped rules below in order: arrival first, then stage-following and direction choice, then forward-step selection, and keep the final answer inside the fixed action space.
   **Action guidance**:
   a. **Destination-first stage following**: the `Destination` is the priority target for this current stage. Use the instruction style only to decide whether to go directly or via a cue. Same-space pass/go-by/around keeps the cue as route guidance while moving toward the destination. Cross-space enter keeps the action on the current enter-stage destination, not a later-stage target.
   b. **Turn only when needed**: turn only when the destination or required cue is off-front, FRONT is blocked/tight, or the instruction requires entering a side opening. Do not turn into an open side if it is wrong-space or backtracking.
   c. **Forward distance selection**: choose forward distance from the best available target-distance evidence in this order: valid destination detection distance > valid subtask-landmark detection distance > bottom-strip landmark/space-waypoint distance > visible free-space depth. Use larger steps for far aligned targets and smaller steps for near targets or limited clearance.
   d. **Connector / pass-by behavior**: if the instruction is to pass / go through / cross / enter toward a farther destination, do not linger at the doorway / hallway / passage mouth or beside the pass-by landmark. If the correct opening is in front and traversable, prefer a medium-to-large step that carries progress through it into the next space. If the pass-by cue is already effectively satisfied and the stage destination is now the main unfinished target in view, center the action on that destination.
   e. **Blocked / uncertain / already-passed cases**: treat FRONT <0.5m as blocked and never move into an obviously blocked direction. Do not use `STOP` only because FRONT is blocked or the view is tight; first prefer a task-aligned turn or shorter cautious forward move if any safe route remains. When two directions are open, prefer the one whose visible opening / space waypoint / structure relation matches the current subtask destination.
   f. **STOP discipline**: output `STOP` immediately if and only if the arrival rule is satisfied for the current subtask destination. Never use `STOP` merely because an intermediate cue is visible, a doorway is nearby, or a pass-by landmark is beside the agent.

# Output Format (JSON only)

{{
    "reasoning": "One concise chain: current-view content, bottom-strip landmark/space-waypoint cues, current position + structure-aligned arrival check, then the safest action toward the current-stage destination",
    "action_analysis": "One short sentence with the key evidence and why this action is best",
    "action": "<MOVE_FORWARD 0.25m | MOVE_FORWARD 0.5m | MOVE_FORWARD 0.75m | MOVE_FORWARD 1.0m | MOVE_FORWARD 1.25m | TURN_LEFT 30deg | TURN_RIGHT 30deg | STOP>"
}}

**Action space**:
- `MOVE_FORWARD {{0.25m, 0.5m, 0.75m, 1.0m, 1.25m}}`
- `TURN_LEFT 30deg` | `TURN_RIGHT 30deg`
- `STOP`

# Examples

**Ex1 - Clear path ahead**
{{
    "reasoning": "Front shows the correct pass-through route, but the actual current-stage goal is the farther destination rather than the intermediate connector cue. The route is open and this is not an arrival case, so use a large forward step to keep moving toward that destination through the connector.",
    "action_analysis": "The correct opening and farther destination are aligned ahead, so a large forward move best serves the current-stage destination",
    "action": "MOVE_FORWARD 1.25m"
}}

**Ex2 - Obstacle detected**
{{
    "reasoning": "Front is blocked. Left opens toward an off-task side space, while right is open and matches the next space-waypoint route toward the sofa area. Turn right.",
    "action_analysis": "Front is blocked, and right is the safest direction that still stays aligned with the correct task space rather than the wrong side opening",
    "action": "TURN_RIGHT 30deg"
}}

**Ex3 - Near but not yet reached**
{{
    "reasoning": "The destination landmark is in front and close, but the arrival rule is not yet satisfied because it is not clearly reached. Front is still passable, so take a short precise forward step instead of stopping.",
    "action_analysis": "A short forward move best matches the near destination distance without overshooting",
    "action": "MOVE_FORWARD 0.25m"
}}

**Ex4 - Destination reached**
{{
    "reasoning": "The current visual evidence shows the subtask destination is already reached, so stop instead of adding another move.",
    "action_analysis": "The destination is already reached, so stopping is correct",
    "action": "STOP"
}}

**Critical Rules**:
- **Reasoning Flow**: keep reasoning concise and evidence-only. Follow the 4-step structure, mention only visible or listed cues, omit empty items, and never invent evidence.
- **Arrival & Stop**: if current evidence satisfies the arrival rule for the actual current subtask destination, output `STOP` immediately. Otherwise do not use `STOP` merely because FRONT is blocked, the view is narrow, an intermediate cue/opening is nearby, or a pass-by cue has appeared. The system also auto-ends the subtask when a highest-confidence top-2 destination landmark is close enough: about 0.75m for solid landmarks, about 0.5m for openings, or already >90deg / behind for passed-through openings.
- **Stage-Following & Direction**: treat `Destination` as the actual goal of the current stage and `Instruction` as the route relation for reaching it. Direct enter/approach stays on the destination route. Pass / go through / go around / cross stays within the same current task piece and moves through the cue toward the destination rather than stopping at the cue. If the task changes space, finish the current enter-stage first and do not jump to a later-stage target. Always use current visual evidence together with the subtask destination, subtask instruction, visible space waypoint cues, and space structure to move toward the correct next task space/landmark. Turn only when the destination or required cue is off-front, FRONT is blocked/tight, or the instruction requires entering a side opening. Never choose a direction only because it is open if the space structure / space waypoint cues indicate it enters the wrong space, a repeated old space, or a backtracking branch.
- **Forward-Step Selection**: choose forward distance from target-distance evidence whenever possible: valid destination detection first, then valid subtask-landmark detection, then bottom-strip landmark/space-waypoint distance, then visible free-space depth. Far target = larger step, near target = smaller step. Prefer `MOVE_FORWARD 1.25m` or `1.0m` when the target/correct opening is clearly far (>2.5m) and the route is open, especially for pass-through / connector-crossing stages; prefer `0.75m` for about 1.5-2.5m; `0.5m` for about 0.8-1.5m; `0.25m` when near (<0.8m) but not yet reached or when clearance is limited. For connector-crossing / enter / pass-through stages with enough clearance, prefer medium-to-large forward movement that carries progress into the next space. If a pass-by / opening cue is already beside or behind but the current-stage destination is still ahead, keep centering actions on the destination rather than turning back toward the old cue. Treat FRONT <0.5m as blocked and never move into an obviously blocked direction. When two directions are open, prefer the one whose visible opening / space waypoint / structure relation matches the current subtask destination.
- **Output Limit**: use one common space type only; ignore modifiers and normalize corridor-like wording to `hallway`. Output `action` only from the fixed action space: `TURN_LEFT 30deg` / `TURN_RIGHT 30deg` / `MOVE_FORWARD {{0.25m, 0.5m, 0.75m, 1.0m, 1.25m}}` / `STOP`.
"""


def get_action_execution_prompt(next_waypoint_destination: str,
                                subtask_instruction: str,
                                progress_summary: str = "",
                                waypoint_summary: str = "",
                                detected_landmarks: str = None,
                                previous_action_reason: str = "",
                                landmark_map_info: str = None,
                                move_distance: float = 0.25,
                                turn_angle: int = 30,
                                # 以下参数保留兼容性但不再用于prompt
                                **kwargs) -> str:
    """获取动作执行提示词"""
    if not progress_summary:
        progress_summary = "Just started"
    if not waypoint_summary:
        waypoint_summary = "No space structure recorded yet."

    return ACTION_EXECUTION_PROMPT.format(
        subtask_destination=next_waypoint_destination,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary,
        waypoint_summary=waypoint_summary,
        previous_action_reason=previous_action_reason or "N/A (first step)",
        detected_landmarks=detected_landmarks or "none",
        move_distance=move_distance,
        turn_angle=turn_angle,
    )
