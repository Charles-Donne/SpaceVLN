"""
动作执行提示词模板
==================
用于VLM低层动作决策的提示词模板

动作参数与interactive_navigation保持一致：
- TURN_LEFT/RIGHT: 30°（12步×30°=360°）
- MOVE_FORWARD: 0.25m
"""

ACTION_EXECUTION_PROMPT = """You are executing a navigation sub-task. Follow the sub-instruction guidance while adapting to actual environment.

# Current Sub-Task
**Sub-Destination**: {subtask_destination}
**Sub-Instruction**: {subtask_instruction}
**Progress**: {progress_summary}

# Visual Inputs (Analyze Together)

**IMAGE 1 - RGB View**: Environment, landmarks, obstacles
**IMAGE 2 - Detection**: Landmark identification: {detected_landmarks}
**IMAGE 3 - Local Map** (Bird's-eye view): Spatial layout around you
- **Red arrow**: Your position & facing direction (arrow points FRONT, map top = FRONT)
- **Purple markers**: Destination landmarks
- **Black**: Obstacles - **MUST AVOID**
- **Green/White**: Safe paths
- **Orange line**: Trajectory history
- **Blue arc**: Current field of view (90° HFOV)
- **Orientation labels**: FRONT (top) / BACK (bottom) / LEFT / RIGHT marked on map edges


# Execution Strategy

**Follow sub-instruction to complete key actions (turn/move/stop)**, BUT:
1. **Adapt parameters**: Fine-tune angles/distances based on map and RGB - not rigidly bound to exact values
2. **Avoid obstacles**: NEVER move into black areas - detour if instruction path blocked
3. **Adjust as needed**: If action result incorrect, make corrective adjustments immediately

**Decision Priority**: Complete key action(sub-instruction goal) → Obstacle avoidance → Parameter refinement

# Actions Available

**Turn**: TURN_LEFT/RIGHT (30°, 60°, 90°, 120°, 150°, 180°)
**Move**: MOVE_FORWARD (0.25m, 0.5m, 0.75m, 1.0m, 1.25m, 1.5m)
**Arrive**: STOP (when <0.5m from destination)

# Output Format (JSON)

{{
    "reasoning": "<(1) Sub-instruction goal, (2) Map check: purple marker position vs instruction, obstacles blocking path, (3) RGB/Detection validation, (4) Action decision: follow instruction OR adapt (specify adjustments or detours)>",
    "action": "TURN_LEFT" | "TURN_RIGHT" | "MOVE_FORWARD" | "STOP",
    "degrees": <30-180> (TURN only),
    "meters": <0.25-1.5> (MOVE_FORWARD only),
    "progress_summary": "<Total rotation, facing direction, total distance>"
}}

# Examples

**Ex1 - Follow Instruction (Path Clear)**
{{
    "reasoning": "Sub-instruction: 'Move 0.5m toward sofa'. Map: Purple marker (sofa) 3m ahead, green path clear, no obstacles. RGB: Open space. Detection: Sofa detected. Action: Follow instruction, move 0.5m.",
    "action": "MOVE_FORWARD",
    "meters": 0.5,
    "progress_summary": "Moved 0.5m toward sofa"
}}

**Ex2 - Adapt Angle (Refine Direction)**
{{
    "reasoning": "Sub-instruction: 'Turn left 90° to table'. Map: Purple marker (table) at left 75° (not 90°), green path clear. RGB: Table visible left. Detection: Table detected. Action: Adjust to 90° (close enough to instruction).",
    "action": "TURN_LEFT",
    "degrees": 90,
    "progress_summary": "Turned left 90°, facing table"
}}

**Ex3 - Detour Obstacle**
{{
    "reasoning": "Sub-instruction: 'Move forward 1m'. Map: Black wall directly ahead, purple marker (door) accessible via right 60°, green path opens right. RGB: Wall ahead. Action: Detour right to avoid obstacle.",
    "action": "TURN_RIGHT",
    "degrees": 60,
    "progress_summary": "Moved 0.5m, detouring right 60° around wall"
}}

**Ex4 - Stop at Destination**
{{
    "reasoning": "Sub-instruction: 'Reach refrigerator'. Map: Red arrow overlaps purple marker <0.5m. RGB: Refrigerator fills view. Detection: Refrigerator detected <0.5m. Action: Destination reached, STOP.",
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

3. **FOLLOW INSTRUCTION & ADAPT**: Complete key actions (turn/move/stop) specified in sub-instruction, but fine-tune angles/distances based on map and RGB - not rigidly bound to exact values

4. **STOP CONDITIONS** - Only STOP when ALL met:
   - Completed key actions in sub-instruction
   - Destination landmark detected in IMAGE 2 + within <0.5m + visible in FRONT RGB view (maximized proximity before stop)
   - Facing toward subtask destination (landmark in FRONT view, NOT left/right/back) AND arrived at destination area
   - Must have moved - orange trajectory on map confirms arrival at destination area

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
