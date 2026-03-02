"""
动作执行提示词模板
==================
用于VLM低层动作决策的提示词模板

动作参数与interactive_navigation保持一致：
- TURN_LEFT/RIGHT: 30°（12步×30°=360°）
- MOVE_FORWARD: 0.25m
"""

ACTION_EXECUTION_PROMPT = """Navigate to {next_waypoint_destination}.

# Task
**Destination**: {next_waypoint_destination}
**Instruction**: {subtask_instruction}
**Progress**: {progress_summary}

# Detection Image
Detection view with distance lines:
**Distances**: FRONT {distance_front} | L30° {distance_left_30} | R30° {distance_right_30} | L90° {distance_left_90} | R90° {distance_right_90}
(<0.5m blocked | >1.0m safe)

# Decision
1. Find destination: Location? Distance? (NEAR<1m/FAR>1m/NONE)
2. Action: NEAR→STOP | NONE→STOP | Left→TURN_LEFT | Right→TURN_RIGHT | Front+clear→MOVE
3. Safety check
# Actions
TURN_LEFT/RIGHT (30-180°), MOVE_FORWARD (0.25-1.0m), STOP

# Output JSON
{{
    "reasoning": "1. Destination: [location][distance] 2. Action: [why] 3. Safety: [check]",
    "action_analysis": "Brief summary",
    "action": "STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT",
    "degrees": 30,
    "meters": 0.25
}}

Example: {{"reasoning": "1. Table: Front 2m 2. Front+clear→MOVE 3. Safe", "action_analysis": "Table ahead→move 0.5m", "action": "MOVE_FORWARD", "meters": 0.5}}

Rules: NEAR<1m→STOP | Left→TURN_LEFT | Right→TURN_RIGHT | Front+clear→MOVE | NONE→STOP"""


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
                                distance_right_90: str = "Unknown") -> str:
    """获取动作执行提示词（精简版）"""
    if not progress_summary:
        progress_summary = "Just started"
        
    return ACTION_EXECUTION_PROMPT.format(
        next_waypoint_destination=next_waypoint_destination,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary,
        distance_front=distance_front,
        distance_left_30=distance_left_30,
        distance_right_30=distance_right_30,
        distance_left_90=distance_left_90,
        distance_right_90=distance_right_90
    )
