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

  Legend:
  - `[Visible]`: landmark appears in current view — navigate toward it directly
  - `[Map-offscreen]`: landmark is mapped but outside FOV — **follow the TURN hint**, then move forward toward it
  - If the destination landmark is within **0.5m** → **STOP immediately**

# Your Task

**Decision Process**:
1. **Check Landmark Map**: Is the destination listed? If `[Visible]` → move toward it. If `[Map-offscreen]` → execute the TURN hint first to face it.
2. **Detection View**: Confirm landmark position in image (yellow bbox). Is it close (<0.5m)?
3. **Obstacle Check**: Which directions are blocked (red) vs safe (green/yellow)?
4. **Action Decision**: Choose the safest action that makes progress toward destination.

**STOP Rules** — STOP immediately if ANY of:
- Destination landmark bbox is visible and within ~0.5m
- Landmark Map shows destination at <0.5m distance
- Subtask instruction is clearly fulfilled

**Safety**: Never move in a red-line direction.

# Output Format (JSON only)

{{
    "reasoning": "Logic: (1) Landmark location and distance from map (2) Is turn needed? (3) Action decision",
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

**Ex1 - Landmark off-screen to the right**
{{
    "reasoning": "Landmark Map shows destination cabinet 3.2m R60deg [Map-offscreen]. Need to turn right ~60deg to face it. Front distance line is green so safe.",
    "action_analysis": "Destination is off-screen to the right, turning right 60deg to face it",
    "action": "TURN_RIGHT",
    "value": 60,
    "progress_summary": "Had turned right 60deg toward cabinet"
}}

**Ex2 - Clear path ahead toward visible landmark**
{{
    "reasoning": "Destination doorway visible ahead (yellow bbox). Landmark Map shows 2.1m ahead. Front distance line is green (>2m open). Move forward.",
    "action_analysis": "Destination visible ahead with clear path, moving forward",
    "action": "MOVE_FORWARD",
    "value": 0.75,
    "progress_summary": "Facing the hallway entrance; moved forward 0.5m toward doorway"
}}

**Ex3 - At destination**
{{
    "reasoning": "Landmark Map shows destination sofa at 0.3m. Yellow bbox is close. STOP condition met.",
    "action_analysis": "Destination within 0.5m, stopping",
    "action": "STOP",
    "value": 0,
    "progress_summary": "Entered living room; now at sofa ~0.3m"
}}

**Critical Rules**:
- **STOP immediately** if destination is within 0.5m or subtask instruction is fulfilled
- For off-screen landmarks, **always turn toward the indicated direction first**
- progress_summary must describe orientation, locations entered/passed, and obstacles bypassed
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
