"""
动作执行提示词模板
==================
用于VLM低层动作决策的提示词模板

动作参数与interactive_navigation保持一致：
- TURN_LEFT/RIGHT: 30°（12步×30°=360°）
- MOVE_FORWARD: 0.25m
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

**Current View (front-facing, RGB HFOV about 79°)** — Object detection overlaid with 3 depth-sampled obstacle-distance lines:
- Directions: Left 30deg, FRONT, Right 30deg
- Red = nearest obstacle <0.5m (blocked), Yellow = 0.5-2m (caution), Green = >2m (open)
- **Yellow bounding box**: candidate subtask-landmark detection ({detected_landmarks}); first judge whether it is valid task-relevant evidence or just duplicate/noisy evidence
- **Bottom white strip** (if present): ranked landmark entries plus reachable `space waypoint` cues. Landmark entries are `vis/off vis + landmark name + distance + direction + confidence`; `vis` = detected now, `off vis` = mapped earlier but outside the current view

# Reasoning Process:

1. **Current View Content**: First analyze the marked landmarks together (if present) with the current Image Content. From near to far, judge whether the visible objects belong to the current space or a farther transition/target space. Then compare FRONT, Left 30deg, and Right 30deg using visible evidence plus the subtask destination to judge where the destination or the correct transition space most likely is. If something is not visible, do not mention it and do not write filler like `none`.
2. **Bottom Strip + Space Structure**: If the bottom white strip is present, read it item by item. For each landmark, know its confidence and whether it matches the image content, how far it is, which direction it is in, and whether it already means arrival. For each space waypoint, judge whether it is the current/next/behind waypoint for this subtask, whether it matches the destination space, and whether it is directly reachable now or indicates another space that should be avoided.
3. **Current Position + Arrival Check**: Use nearby objects, valid landmarks, the visible space waypoint cues, and the provided space structure to confirm your current position and where the subtask destination is relative to you. Decide which side leads into the correct next task space and which side would enter a wrong/backtracking space. If you have already reached the subtask destination, stop immediately; otherwise continue moving toward that destination.
   **Arrival rule**: Treat the subtask as reached when you are already in the destination place, or when the detected landmark itself is the destination and is close enough to stop. For solid-object destinations, stopping is allowed once it is within about 1.0m or already clearly at hand. For opening-like destinations (entrance / doorway / hallway), stop only when it is within about 0.5m or when that opening has moved to >90deg or the back side, meaning you have already passed through it.
4. **Action Decision + Obstacle Avoidance**: Choose one safe immediate action that follows the subtask instruction style and moves toward the subtask destination.
   **Action guidance**:
   a. **Direct approach / enter instruction**: if the instruction is like `move/enter/approach toward [destination]`, keep the destination or its opening/target cue in the chosen forward route. If FRONT matches the destination and is open, prefer `MOVE_FORWARD`. If another open side more likely leads to a wrong side space according to the space structure / waypoint cues, reject it even if it looks open.
   b. **Via visible cue / path-following instruction**: if the instruction is like `pass/go through/go around/cross [visible cue], then continue to [destination]`, first act toward the visible intermediate cue/opening that makes the route feasible, then continue toward the destination after that cue is passed. If that cue and destination belong to the same current task piece / same space, keep following that one instruction and do not treat the cue itself as a separate stage or stop target. If the task changes into another space, first finish the current enter-stage before following the later in-room destination stage. Do not rotate just to center a pass-by cue if the forward route already safely goes past it.
   c. **Turn only when needed**: prefer `TURN_LEFT 30deg` / `TURN_RIGHT 30deg` mainly when the destination or required intermediate cue is off-front, when FRONT is blocked or too tight, or when the current instruction explicitly requires entering a side opening. Do not turn into a side opening just because it is open if the space structure says it is the wrong space or a repeated/backtracking route.
   d. **Forward distance selection**: when choosing `MOVE_FORWARD`, match distance to visible free space and target depth. Prefer about `1.0m-1.25m` for clearly open far advance (>2m), `0.5m-0.75m` for medium advance (about 1-2m), and `0.25m-0.5m` when the cue/destination is already near or clearance is limited.
   e. **Blocked / uncertain / already-passed cases**: treat FRONT <0.5m as blocked. Never move into an obviously blocked direction. Do not choose `STOP` just because FRONT is blocked or the current view is tight; first prefer a task-aligned turn or a shorter cautious forward move if any safe route remains. Use `STOP` only when the arrival rule is satisfied, or when the destination is already passed / current evidence clearly says the subtask should end and hand control back for replanning. When two directions are both open, prefer the one whose visible opening / space waypoint / structure relation matches the current subtask destination and avoid the other.
  

# Output Format (JSON only)

{{
    "reasoning": "One concise chain: current-view content, bottom-strip landmark/space-waypoint cues, current position + structure-aligned arrival check, then safe action toward the correct task space/destination",
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
    "reasoning": "Front is open, the target-area cue stays ahead, and the space structure says the next task waypoint is still forward rather than in the side spaces. Left and right do not better match the route, so move forward.",
    "action_analysis": "Forward best matches the visible target cue and the correct next space-waypoint direction with a clear front path",
    "action": "MOVE_FORWARD 0.75m"
}}

**Ex2 - Obstacle detected**
{{
    "reasoning": "Front is blocked. Left opens toward an off-task side space, while right is open and matches the next space-waypoint route toward the sofa area. Turn right.",
    "action_analysis": "Front is blocked, and right is the safest direction that still stays aligned with the correct task space rather than the wrong side opening",
    "action": "TURN_RIGHT 30deg"
}}

**Ex3 - Near destination cue**
{{
    "reasoning": "The current visual evidence shows the subtask destination is already reached, so stop immediately instead of adding another move.",
    "action_analysis": "The destination is already reached in the current view, so stopping is the correct action",
    "action": "STOP"
}}

**Critical Rules**:
- if current evidence satisfies the arrival rule, output `STOP` immediately; otherwise do not use `STOP` merely because FRONT is blocked or the view is narrow. The system also auto-ends the subtask when a highest-confidence top-2 destination landmark is close enough: about 1.0m for solid objects, about 0.5m for openings, or already >90deg / behind for passed-through openings
- keep reasoning concise and evidence-only: follow the 4-step structure, mention only visible or listed cues, omit empty items, and never invent evidence
- use one common room/space type only; ignore modifiers and normalize corridor-like wording to `hallway`
- output `action` only from the fixed action space: `TURN_LEFT 30deg` / `TURN_RIGHT 30deg` / `MOVE_FORWARD {{0.25m, 0.5m, 0.75m, 1.0m, 1.25m}}` / `STOP`
- prefer the action that best matches the instruction style while staying safe: for direct enter/approach instructions, go forward when FRONT clearly matches and is open; for pass/go through/go around/cross instructions, follow the cue-to-destination path within the same current task piece without splitting it into a fake extra stage; if the task changes space, finish the current enter-stage first; if FRONT is blocked, choose the nearer safe side instead of a wide detour
- always use current visual evidence together with the subtask destination, subtask instruction, visible space waypoint cues, and space structure to keep moving toward the correct next task space/object
- never choose a direction only because it is open if the space structure / space waypoint cues indicate it enters the wrong space, a repeated old space, or a backtracking branch
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
