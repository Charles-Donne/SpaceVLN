"""
动作执行提示词模板
==================
用于VLM低层动作决策的提示词模板

动作参数与interactive_navigation保持一致：
- TURN_LEFT/RIGHT: 30°（12步×30°=360°）
- MOVE_FORWARD: 0.25m
"""

ACTION_EXECUTION_PROMPT = """You are executing a navigation sub-task. Analyze environment and decide next action to complete this sub-task.

# Current Sub-Task
**Sub-Destination**: {subtask_destination}
**Sub-Instruction**: {subtask_instruction}

# Progress Summary
{progress_summary}

# Visual Observations

**3 Images Provided**:
- **IMAGE 1**: First-person RGB view (current facing direction)
- **IMAGE 2**: Object detection view with bounding boxes (landmarks: {detected_landmarks})
- **IMAGE 3**: Local semantic map (nearby region, landmarks shown as purple markers)

# Local Map Guide

**Orientation**: Top = Front, map rotates with agent, agent at center

**Colors**:
- White: Unexplored | Black: Obstacles (AVOID) | Green: Safe floor
- Orange line: Recent trajectory | Red arrow: Agent position & facing direction
- Purple markers: Detected landmarks | Blue semi-circle: Current field of view (opening = Front)

# Task

Decide next action to complete the sub-task based on:
1. **Progress Summary**: Review past actions (rotation, movement)
2. **Orange Trajectory** on local map: Shows completed movement path
3. **Current View**: Identify sub-destination location and obstacles

**STOP Conditions**
- Sub-destination within 0.5m or Sub-instruction completed
- Must have moved (verify via progress & trajectory)
- Landmark detected (if applicable)

**Decision Strategy**: Use progress history + trajectory to determine next action

**Safety**: Avoid black obstacles on local map

# Available Actions

**Rotation** (30° increments): TURN_LEFT / TURN_RIGHT (30, 60, 90, 120, 150, 180)
**Movement** (0.25m increments): MOVE_FORWARD (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
**Arrival**: STOP

# Output Format (JSON only)

{{
    "reasoning": "<Single-paragraph: destination location, distance estimate, obstacle check, action decision>",
    "action": "TURN_LEFT" | "TURN_RIGHT" | "MOVE_FORWARD" | "STOP",
    "degrees": <30|60|90|120|150|180> (for TURN only),
    "meters": <0.25|0.5|0.75|1.0|1.25|1.5> (for MOVE_FORWARD only),
    "progress_summary": "<Cumulative: rotation total, facing direction, distance total>"
}}

# Examples

**Ex1 - Orientation**
{{
    "reasoning": "Sub-destination sofa at left ~3m. No movement yet. Rotate 90° left to face sofa.",
    "action": "TURN_LEFT",
    "degrees": 90,
    "progress_summary": "Rotated left 90°, facing sofa"
}}

**Ex2 - Approaching**
{{
    "reasoning": "Sofa ahead ~2m. Orange trajectory shows rotated 90°. Clear path. Move 0.5m toward sofa.",
    "action": "MOVE_FORWARD",
    "meters": 0.5,
    "progress_summary": "Rotated left 90°, moved 0.5m, approaching sofa"
}}

**Ex3 - Obstacle Avoidance**
{{
    "reasoning": "Table sub-destination ahead-right ~2.5m. Black obstacle blocks path. Turn 60° right to navigate around.",
    "action": "TURN_RIGHT",
    "degrees": 60,
    "progress_summary": "Rotated left 90°, moved 0.5m, turned right 60°, navigating to table"
}}

**Ex4 - Destination Reached**
{{
    "reasoning": "Refrigerator fills view <0.5m. Moved 1.0m total (trajectory visible). Sub-destination reached.",
    "action": "STOP",
    "progress_summary": "Rotated left 30°, moved 1.0m, reached refrigerator"
}}

**Rules**:
- Specify degrees (30-180) for TURN | meters (0.25-1.5) for MOVE_FORWARD
- STOP when ANY: sub-destination <0.5m OR landmark detected OR sub-instruction done (must have moved)
- Use progress + orange trajectory to understand past actions and decide next
- Progress must be cumulative and precise
"""


def get_action_execution_prompt(subtask_destination: str,
                                subtask_instruction: str,
                                turn_angle: float,
                                move_distance: float,
                                progress_summary: str = "",
                                detected_landmarks: str = None) -> str:
    """
    获取动作执行提示词
    
    Args:
        subtask_destination: 子任务目的地
        subtask_instruction: 子任务指令
        turn_angle: 转向角度（度）- 默认30°
        move_distance: 前进距离（米）- 默认0.25m
        progress_summary: 当前子任务进度摘要
        detected_landmarks: 已检测到的landmark类别字符串
        
    Returns:
        格式化的提示词字符串
    """
    if not detected_landmarks:
        detected_landmarks = "No landmarks detected yet"
        
    return ACTION_EXECUTION_PROMPT.format(
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        turn_angle=turn_angle,
        move_distance=move_distance,
        progress_summary=progress_summary if progress_summary else "(Just started - no actions yet)",
        detected_landmarks=detected_landmarks
    )
