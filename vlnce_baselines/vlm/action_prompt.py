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

# Previous Step Analysis
{previous_action_reason}

# Visual Observations

You are provided with 1 image:

**Current View (front-facing)** — Object detection results with bounding boxes (target landmark: {detected_landmarks}), overlaid with 7-direction lines showing the distance to the nearest obstacle in each direction:
- Directions: FRONT, Left/Right 30deg, Left/Right 60deg, Left/Right 90deg (from bottom center)
- Red = nearest obstacle <0.5m (blocked), Yellow = 0.5–2m (caution), Green = >2m (open)
- **Bottom strip** (cyan text, if present): mapped landmarks currently **off-screen** — same as Map-offscreen entries below

# Known Landmark Map (from semantic map, sorted by distance)
{landmark_map_info}

# Your Task

**Decision Process**:
1. **Detection View**: Are there relevant landmarks (yellow bbox)? Where is the destination relative to current view?
2. **Landmark Map**: Check the Known Landmark Map section above — each entry shows distance and direction.
   - `[Visible]` → visible in current view, move toward its yellow bbox
   - `[Map-offscreen R/Ldeg]` → turn RIGHT/LEFT by that many degrees first, then move
   - Distance < 0.5m → **STOP immediately**
3. **Distance Lines**: Which directions are blocked (red) vs safe (green/yellow)?
4. **Distance Estimation**: How far to destination? (e.g., "~3m", "<0.5m")
5. **Action Decision**: Choose safest action toward destination, avoiding blocked directions

**STOP Condition** — You must get as close as possible to the destination or fully complete the subtask instruction, then STOP immediately — do not move past it or take unnecessary extra steps.

**Safety Priority**: Avoid directions with red distance lines (obstacle <0.5m)

# Output Format (JSON only)

{{
    "reasoning": "Logic: (1) Destination location and distance (2) Movement count (3) Action decision",
    "action_analysis": "One-sentence analysis of why this action was chosen",
    "action": "MOVE_FORWARD" | "TURN_LEFT" | "TURN_RIGHT" | "STOP",
    "value": 0,
    "progress_summary": "Updated summary: actions taken, current facing direction, locations entered/bypassed"
}}

**Parameter rules**:
- MOVE_FORWARD: "value" = meters (0.25 ~ 1.5)
- TURN_LEFT / TURN_RIGHT: "value" = degrees (30 ~ 90, multiples of 30)
- STOP: "value" = 0

# Examples

**Ex1 - Clear path ahead**
{{
    "reasoning": "Destination doorway visible ahead (yellow bbox). Landmark Map shows 2.1m ahead. Front distance line is green (>2m open). Move forward.",
    "action_analysis": "Destination visible ahead with clear path, moving forward",
    "action": "MOVE_FORWARD",
    "value": 0.75,
    "progress_summary": "Facing the hallway entrance; moved forward 0.5m toward doorway; no obstacles bypassed yet"
}}

**Ex2 - Obstacle detected**
{{
    "reasoning": "Front distance line is red (<0.5m blocked). Right 30° is green. Turn right to find clear path toward sofa.",
    "action_analysis": "Obstacle blocking forward path, turning right toward open direction",
    "action": "TURN_RIGHT",
    "value": 30,
    "progress_summary": "Bypassed wall on left; now facing right corridor; moved ~1m total"
}}

**Ex3 - At destination**
{{
    "reasoning": "Landmark Map: sofa 0.3m [Visible]. Under 0.5m threshold. STOP.",
    "action_analysis": "Destination within 0.5m, stopping immediately",
    "action": "STOP",
    "value": 0,
    "progress_summary": "Entered living room from hallway; bypassed table on right; now facing sofa at ~0.3m"
}}

**Critical Rules**:
- **STOP immediately** if destination is < 0.5m or subtask instruction is fulfilled
- For off-screen landmarks, **always turn toward the indicated direction first**
- progress_summary must describe orientation, locations entered/passed, obstacles bypassed
"""


def get_action_execution_prompt(next_waypoint_destination: str,
                                subtask_instruction: str,
                                progress_summary: str = "",
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
    if not landmark_map_info:
        landmark_map_info = "No landmarks mapped yet"

    return ACTION_EXECUTION_PROMPT.format(
        subtask_destination=next_waypoint_destination,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary,
        previous_action_reason=previous_action_reason or "N/A (first step)",
        detected_landmarks=detected_landmarks or "none",
        landmark_map_info=landmark_map_info,
        move_distance=move_distance,
        turn_angle=turn_angle,

