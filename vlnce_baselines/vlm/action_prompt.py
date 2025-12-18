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
**Previous Progress**: {progress_summary}

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
- Avoid obstacles: NEVER move into black areas - detour if instruction path blocked

**Decision Priority**: Complete key action(sub-instruction goal) → Obstacle avoidance → Parameter refinement(optional) → Progress update

# Actions Available

**Turn**: TURN_LEFT/RIGHT (30°, 60°, 90°, 120°, 150°, 180°)
**Move**: MOVE_FORWARD (0.25m, 0.5m, 0.75m, 1.0m, 1.25m, 1.5m)
**Arrive**: STOP (when <0.5m from destination)

# Output Format (JSON)

{{
    "reasoning": "<(1) Subtask goal. (2) Finding of observation. (3) Map check: your position, orientation, landmark, obstacles. (4) Action: follow instruction OR adaptive fine-tuning>",
    "action": "TURN_LEFT" | "TURN_RIGHT" | "MOVE_FORWARD" | "STOP",
    "degrees": <30-180> (TURN only),
    "meters": <0.25-1.5> (MOVE_FORWARD only),
    "progress_summary": "<Update motion trajectory and the observed object>"
}}

# Examples

## Ex1 - Start turning to face the target:
**Sub-Instruction**: Turn left 90° to face the oven, then move forward 0.5m, Stop in front of oven.
**Previous Progress**: None
**Current Observation**: Oven is not in front view; need to turn to face it.
{{
    "reasoning": "The subtask goal is to face the oven first. RGB: No oven visible in current front view. Map: Purple marker (oven) is to the left, need to rotate. Action: Follow instruction - turn left 90° to align with oven direction.",
    "action": "TURN_LEFT",
    "degrees": 90,
    "progress_summary": "Turned left 90° to face oven"
}}

## Ex2 - Continue with the instruction action:
**Sub-Instruction**: Turn left 90° to face the oven, then move forward 0.5m, Stop in front of oven.
**Previous Progress**: Turned left 90°, moved 0.5m toward oven.
**Current Observation**: Facing the oven, but the distance is still too far.
{{
    "reasoning": "The subtask goal is to stop at the oven. RGB & Detection: The oven is ahead, and there's space to move. Map: Purple marker (oven) ahead, green path clear, no obstacles. Action: Move a little closer to the oven.",
    "action": "MOVE_FORWARD",
    "meters": 0.5,
    "progress_summary": "Turned left 90°, moved 1.0m (0.5+0.5m) toward oven"
}}

## Ex3 - Arrive at destination and stop:
**Sub-Instruction**: Turn left 90° to face the oven, then move forward 0.5m, Stop in front of oven.
**Previous Progress**: Turned left 90°, moved 1.5m toward oven.
**Current Observation**: The oven is directly in front, very close (within 0.5m).
{{
    "reasoning": "The subtask goal is to stop at the oven. RGB: Oven clearly visible in front view. Detection: Oven detected. Map: Red arrow overlaps purple marker (oven), distance <0.5m, orange trajectory confirms arrival. All key actions completed. Action: STOP.",
    "action": "STOP",
    "progress_summary": "Turned left 90°, moved 1.5m toward oven, arrived and stopped"
}}

## Ex4 - Detour around obstacle:
**Sub-Instruction**: Turn left 90° to face the oven, then move forward 0.5m, Stop in front of oven.
**Previous Progress**: Turned left 90°.
**Current Observation**: Oven is at front-left 30°, but straight ahead has a wall (black obstacle on map).
{{
    "reasoning": "The subtask goal is to reach the oven. RGB: Wall/obstacle blocking direct path ahead. Detection: Oven detected at left side. Map: Purple marker (oven) at front-left 30°, black obstacle directly ahead, green path to the left. Action: Turn left 30° to avoid obstacle and align toward oven.",
    "action": "TURN_LEFT",
    "degrees": 30,
    "progress_summary": "Turned left 120° (90°+30° detour) to avoid wall and face oven"
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

3. **Strictly FOLLOW INSTRUCTION & ADAPT**: Complete key actions (turn/move) specified in sub-instruction, but you can fine-tune angles/distances based on map and RGB - not rigidly bound to exact values

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
                                progress_summary: str = "",
                                detected_landmarks: str = None) -> str:
    """
    获取动作执行提示词
    
    Args:
        subtask_destination: 子任务目的地
        subtask_instruction: 子任务指令
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
        progress_summary=progress_summary if progress_summary else "(Just started - no actions yet)",
        detected_landmarks=detected_landmarks
    )
