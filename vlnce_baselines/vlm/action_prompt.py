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
- **IMAGE 1**: First-person RGB view - observe environment, landmarks, spatial layout
- **IMAGE 2**: Object detection with bounding boxes - identify landmarks: {detected_landmarks}
- **IMAGE 3**: Local semantic map - spatial relationships, obstacles, path planning

**Use all 3 images together**: RGB shows what you see, detection identifies objects, map shows spatial layout.

# Local Map Guide

**Orientation**: Top = Front, map rotates with agent, agent at center

**Colors**:
- **White**: Unexplored areas
- **Black**: Obstacles (walls/furniture) - AVOID
- **Green**: Safe floor
- **Orange line**: Movement trajectory
- **Red arrow**: Current position & facing direction
- **Purple markers**: Instruction-related landmarks 
- **Blue semi-circle**: Field of view (opening = Front)

# Task

**Analyze all 3 images together to decide next action**:
- **RGB (IMAGE 1)**: What environment/landmarks/obstacles visible?
- **Detection (IMAGE 2)**: Which landmarks detected and where?
- **Map (IMAGE 3)**: Your position (red arrow), instruction-related landmarks (purple), safe paths (green), obstacles (black), trajectory (orange)
- **Progress**: What actions completed?
- **Next Action**: Follow sub-instruction, adapt to environment, avoid obstacles

# Available Actions

**Rotation** (30° increments): TURN_LEFT / TURN_RIGHT (30, 60, 90, 120, 150, 180)
**Movement** (0.25m increments): MOVE_FORWARD (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
**Arrival**: STOP

# Output Format (JSON only)

{{
    "reasoning": "<Combine: (1) RGB view: what visible in IMAGE 1, (2) Detection: landmarks in IMAGE 2, (3) Map: position/destination/obstacles/path in IMAGE 3, (4) Progress, (5) Action decision>",
    "action": "TURN_LEFT" | "TURN_RIGHT" | "MOVE_FORWARD" | "STOP",
    "degrees": <30|60|90|120|150|180> (for TURN only),
    "meters": <0.25|0.5|0.75|1.0|1.25|1.5> (for MOVE_FORWARD only),
    "progress_summary": "<Cumulative: rotation total, facing direction, distance total>"
}}

# Examples

**Ex1 - Safe Movement**
{{
    "reasoning": "RGB: Open space visible ahead. Detection: Sofa detected ahead-left. Map: Green clear 2m ahead, no black obstacles; purple marker (sofa) 3m ahead-left. Progress: started. Decision: Move 0.5m safely.",
    "action": "MOVE_FORWARD",
    "meters": 0.5,
    "progress_summary": "Moved 0.5m forward, approaching sofa"
}}

**Ex2 - Obstacle Avoidance**
{{
    "reasoning": "RGB: Wall visible ahead, table visible at right. Detection: Table detected. Map: Black obstacle ahead; green opening right 60°; purple marker (table) right 2.5m. Decision: Turn right 60° to avoid wall and approach table.",
    "action": "TURN_RIGHT",
    "degrees": 60,
    "progress_summary": "Moved 0.5m, turned right 60°, avoiding obstacle"
}}

**Ex3 - Dead-End Escape**
{{
    "reasoning": "RGB: Walls on front/left/right. Detection: No destination landmarks visible. Map: Black walls front/left/right; green opening behind 180°. Progress: 1.0m into corner. Decision: Turn 180° to escape dead-end.",
    "action": "TURN_LEFT",
    "degrees": 180,
    "progress_summary": "Moved 1.0m, encountered dead-end, turning 180° to backtrack"
}}

**Ex4 - Destination Reached**
{{
    "reasoning": "RGB: Refrigerator fills view. Detection: Refrigerator detected center <0.5m. Map: Purple marker (refrigerator) at center <0.5m; orange trajectory 1.5m. Progress: completed all steps. Decision: Destination reached, STOP.",
    "action": "STOP",
    "progress_summary": "Rotated left 30°, moved 1.5m, reached refrigerator"
}}


**CRITICAL EXECUTION RULES** (MUST FOLLOW):

1. **MULTIMODAL UNDERSTANDING** - Combine all 3 images for every decision:
   - **RGB (IMAGE 1)**: Observe visible environment, landmarks, obstacles
   - **Detection (IMAGE 2)**: Confirm which landmarks detected and positions
   - **Map (IMAGE 3)**: Your position (red arrow), instruction-related landmarks (purple), safe areas (green floor or unexplored white), obstacles (black)

2. **MAP NAVIGATION**:
   - Locate instruction landmarks: Purple markers show instruction-related objects, estimate distance/angle from red arrow
   - Plan safe path: Avoid black obstacles
   - If trapped by black: Turn toward nearest green/white opening

3. **FOLLOW INSTRUCTION**: Execute sub-instruction step-by-step - do not skip or reorder

4. **STOP CONDITIONS** - Only STOP when ALL met:
   - Completed ALL sub-instruction steps
   - Destination landmark visible in RGB + detected in IMAGE 2 + within <0.5m
   - Orange trajectory on map confirms arrival at destination area
   - Must have moved (verify via progress & trajectory)

5. **ACTION PARAMETERS**:
   - Specify degrees (30-180) for TURN | meters (0.25-1.5) for MOVE_FORWARD
   - Progress must be cumulative and precise (track total rotation, distance, status)
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
