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

# Visual Observations

You are provided with 1 image:

**IMAGE 1: Object Detection View** - Detected objects with bounding boxes and 7-direction obstacle distance lines (landmark: {detected_landmarks})
- Distance lines from bottom center: FRONT (up), Left/Right 30°/60°/90°
- Red line = obstacle <0.5m (blocked), Yellow = 0.5-2m, Green = >2m (open)

# Your Task

Analyze the detection image to decide the next action.

**Decision Process**:
1. **Detection View**: Are there relevant landmarks (yellow bbox)? Where is the destination?
2. **Distance Lines**: Which directions are blocked (red) vs safe (green/yellow)?
3. **Distance Estimation**: How far to destination? (e.g., "~3m", "<0.5m")
4. **Action Decision**: Choose safest action toward destination

**STOP Conditions** (ALL required):
- Moved ≥2 times
- Destination within 0.5m

**Safety Priority**: Avoid obstacles shown as black regions on local map

# Output Format (JSON only)

{{
    "reasoning": "Logic: (1) Destination location and distance (2) Movement count (3) Action decision",
    "action_analysis": "One-sentence analysis of why this action was chosen",
    "action": "MOVE_FORWARD" | "TURN_LEFT" | "TURN_RIGHT" | "STOP",
    "value": 0,
    "progress_summary": "Updated action history for current subtask"
}}

**Parameter rules**:
- MOVE_FORWARD: "value" = meters (0.25 ~ 1.5)
- TURN_LEFT / TURN_RIGHT: "value" = degrees (30 ~ 90, multiples of 30)
- STOP: "value" = 0

# Examples

**Ex1 - Clear path ahead**
{{
    "reasoning": "Local map shows safe green floor ahead. Destination visible closed. Move forward.",
    "action_analysis": "Clear path ahead on local map, destination visible in detection view",
    "action": "MOVE_FORWARD",
    "value": 0.5,
    "progress_summary": "Moved forward 1x toward doorway"
}}

**Ex2 - Obstacle detected**
{{
    "reasoning": "Local map shows black obstacle directly ahead. Must turn right to find clear path.",
    "action_analysis": "Obstacle blocking forward path, turning right to find clear route",
    "action": "TURN_RIGHT",
    "value": 30,
    "progress_summary": "Turned right 30 to avoid obstacle"
}}

**Ex3 - At destination**
{{
    "reasoning": "Movement: 4 (>=2), Distance: <0.5m. ALL STOP conditions met.",
    "action_analysis": "All STOP criteria met: moved >=2 times, distance <0.5m",
    "action": "STOP",
    "value": 0,
    "progress_summary": "Reached destination after forward moves 3m"
}}

**Critical Rules**:
- Move ≥2 times before STOP
- STOP only when distance ≤0.5m
- When uncertain, MOVE_FORWARD
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
                                move_distance: float = 0.25,
                                turn_angle: int = 30) -> str:
    """获取动作执行提示词（精简版）"""
    if not progress_summary:
        progress_summary = "Just started"
        
    return ACTION_EXECUTION_PROMPT.format(
        subtask_destination=next_waypoint_destination,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary,
        detected_landmarks=detected_landmarks or "none",
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
