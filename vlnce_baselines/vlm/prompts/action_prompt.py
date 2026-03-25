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

# Progress Summary
{progress_summary}

# Space Structure
{waypoint_summary}

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

1. **Current View Content**: First analyze the marked landmarks together (if present) with the current Image Content. From near to far, judge whether the visible objects belong to the current space or a farther transition/target space. Then compare FRONT, Left 30deg, and Right 30deg using only visible evidence to judge where the subtask destination most likely is. If something is not visible, do not mention it and do not write filler like `none`.
2. **Bottom Strip: Detected Landmarks / Space Waypoint**: If the bottom white strip is present, read it item by item. For each landmark, know its confidence and whether it matches the image content, how far it is, which direction it is in, and whether it already means arrival. For each space waypoint, judge whether it is the waypoint you need to go to, how far it is, which direction it is in.
3. **Current Position + Arrival Check**: Use nearby objects, valid landmarks, and the space structure to confirm your current position and where the subtask destination is relative to you. If you have already reached the subtask destination, stop immediately; otherwise continue moving toward that destination.
   **Arrival rule**: Treat the subtask as reached when you are already in the destination place, or when the detected landmark itself is the destination and is close enough to stop. For solid-object destinations, stopping is allowed once it is within about 1.0m or already clearly at hand. For opening-like destinations (entrance / doorway / hallway), stop only when it is within about 0.5m or when that opening has moved to >90deg or the back side, meaning you have already passed through it.
4. **Action Decision + Obstacle Avoidance**: Choose the action from the subtask destination/instruction, while avoiding obstacles. If the destination is clearly ahead and FRONT is safe, prefer moving forward with a distance that matches the visible depth. If the destination is not visible ahead or is more consistent with Left 30deg / Right 30deg, prefer turning toward that side. If FRONT has obstacle distance <0.5m, treat it as blocked and choose the safer side whose image content is more likely to lead to the subtask destination. Never move into an obviously blocked direction.
   **Action guidance**:
   a. If the destination or task-relevant landmark is visible and still clearly far, prefer moving forward first.
   b. Do not overuse turns. Turn mainly when FRONT is blocked, or when the subtask destination is already near but clearly off-front / out of view.
   c. If you only need to pass by a landmark before entering the true destination, and that landmark is already near but not centered ahead, do not force a turn just to face it.
   d. If you are unsure where the destination is, or the destination already appears very near / beside you / already passed, stop immediately.
   e. When choosing `MOVE_FORWARD`, select 0.25m-1.25m to match the visible depth to the destination or key landmark.

# Output Format (JSON only)

{{
    "reasoning": "One concise chain: current-view content, bottom-strip landmark/space cues if present, current position + arrival check, then safe action toward the subtask destination",
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
    "reasoning": "Front is open, the target-area cue stays ahead, and the space structure still aligns forward. Left and right do not better match the route, so move forward.",
    "action_analysis": "Forward best matches the visible target cue and the space structure with a clear front path",
    "action": "MOVE_FORWARD 0.75m"
}}

**Ex2 - Obstacle detected**
{{
    "reasoning": "Front is blocked, right is open, and right aligns better than left with the space waypoint route toward the sofa area. Turn right.",
    "action_analysis": "Front is blocked, and right is the safest direction that still stays aligned with the target route",
    "action": "TURN_RIGHT 30deg"
}}

**Ex3 - Near destination cue**
{{
    "reasoning": "The current visual evidence shows the subtask destination is already reached, so stop immediately instead of adding another move.",
    "action_analysis": "The destination is already reached in the current view, so stopping is the correct action",
    "action": "STOP"
}}

**Critical Rules**:
- if current evidence satisfies the arrival rule, output `STOP` immediately; the system also auto-ends the subtask when a highest-confidence top-2 destination landmark is close enough: about 1.0m for solid objects, about 0.5m for openings, or already >90deg / behind for passed-through openings
- keep reasoning concise and evidence-only: follow the 4-step structure, mention only visible or listed cues, omit empty items, and never invent evidence
- use one common room/space type only; ignore modifiers and normalize corridor-like wording to `hallway`
- output `action` only from the fixed action space: `TURN_LEFT 30deg` / `TURN_RIGHT 30deg` / `MOVE_FORWARD {{0.25m, 0.5m, 0.75m, 1.0m, 1.25m}}` / `STOP`
- prefer the action that best matches the destination and instruction while staying safe: go forward if FRONT clearly matches and is open, turn toward off-screen or side cues first, and if FRONT is blocked choose the closer safe side instead of a wide detour
- always use current visual evidence together with the subtask destination, subtask instruction, and space structure to keep moving toward the most likely relevant space/object
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
