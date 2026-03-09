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

**Current View (front-facing)** — Object detection results with bounding boxes (target landmark: {detected_landmarks}), overlaid with 7-direction lines showing obstacle distances:
- Directions: FRONT, Left/Right 30deg, Left/Right 60deg, Left/Right 90deg (from bottom center)
- Red = nearest obstacle <0.5m (blocked), Yellow = 0.5–2m (caution), Green = >2m (open)
- **Bottom strip** (cyan text, if present): mapped landmarks currently off-screen — same as Map-offscreen entries below
- **Yellow bbox label**: `name Xm Ydeg` — distance and angle from semantic map

# Known Landmark Map (from semantic map, sorted by distance)
{landmark_map_info}

  - `[Visible]`: landmark is in current view — navigate toward its yellow bbox
  - `[Map-offscreen Rdeg]`: off-screen to the RIGHT — TURN RIGHT that many degrees first
  - `[Map-offscreen Ldeg]`: off-screen to the LEFT — TURN LEFT that many degrees first
  - Distance < 0.5m → **STOP immediately**

# Your Task

**Decision Process**:
1. **Check Landmark Map**: Is the destination listed?
   - `[Visible]` → it's in view, move toward the yellow bbox
   - `[Map-offscreen]` → **execute the TURN hint first** (e.g., `R60deg` = TURN_RIGHT 60)
   - Listed distance < 0.5m → **STOP immediately**
2. **Confirm in Detection View**: Visible yellow bbox present? How close?
3. **Obstacle Check**: Which directions are blocked (red) vs safe (green/yellow)?
4. **Action Decision**: Safest action toward destination.

**Safety**: Never move into a red-line direction.

# Output Format (JSON only)

{{
    "reasoning": "1) Landmark location+distance from map  2) Turn needed?  3) Action decision",
    "action_analysis": "One-sentence analysis of why this action was chosen",
    "action": "MOVE_FORWARD" | "TURN_LEFT" | "TURN_RIGHT" | "STOP",
    "value": 0,
    "progress_summary": "Updated: actions taken, current facing direction, locations entered/bypassed"
}}

**Parameter rules**:
- MOVE_FORWARD: "value" = meters (0.25 ~ 1.5)
- TURN_LEFT / TURN_RIGHT: "value" = degrees (30 ~ 90, multiples of 30)
- STOP: "value" = 0

# Examples

**Ex1 - Off-screen landmark to the right**
{{
    "reasoning": "Landmark Map: cabinet 3.2m [Map-offscreen R60deg]. Need TURN_RIGHT 60 to face it. Front is green.",
    "action_analysis": "Destination off-screen to right, turning right 60deg to face it",
    "action": "TURN_RIGHT",
    "value": 60,
    "progress_summary": "Had turned right 60deg toward cabinet"
}}

**Ex2 - Visible landmark ahead, clear path**
{{
    "reasoning": "Landmark Map: doorway 2.1m [Visible]. Yellow bbox visible ahead. Front line green (>2m). Move forward.",
    "action_analysis": "Destination visible ahead with clear path, moving forward",
    "action": "MOVE_FORWARD",
    "value": 0.75,
    "progress_summary": "Facing hallway entrance; moved 0.5m toward doorway"
}}

**Ex3 - At destination**
{{
    "reasoning": "Landmark Map: sofa 0.3m [Visible]. Under 0.5m threshold. STOP.",
    "action_analysis": "Destination within 0.5m, stopping immediately",
    "action": "STOP",
    "value": 0,
    "progress_summary": "Entered living room; now at sofa ~0.3m"
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
                                distance_front: str = "Unknown",
                                distance_left_30: str = "Unknown",
                                distance_right_30: str = "Unknown",
                                distance_left_60: str = "Unknown",
                                distance_right_60: str = "Unknown",
                                distance_left_90: str = "Unknown",
                                distance_right_90: str = "Unknown",
                                landmark_map_info: str = None,
                                move_distance: float = 0.25,
                                turn_angle: int = 30) -> str:
    """获取动作执行提示词（精简版）"""
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
        distance_front=distance_front,
        distance_left_30=distance_left_30,
        distance_left_60=distance_left_60,
        distance_right_30=distance_right_30,
        distance_right_60=distance_right_60,
        distance_left_90=distance_left_90,
        distance_right_90=distance_right_90
    )
