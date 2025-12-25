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
**Previous Total Progress**: {progress_summary}
**Last One Action Reason**: {previous_action_reason}

# Visual Inputs (Analyze Together)

**IMAGE 1 - RGB View**: Environment, landmarks, obstacles
**IMAGE 2 - Detection**: Detected Landmark: {detected_landmarks}
**IMAGE 3 - Local Map** (Bird's-eye view): Spatial layout around you
- **Red arrow**: Your position & facing direction (arrow points FRONT, map top = FRONT)
- **Dark red dashed line**: Extends from red arrow upward, indicating exact Forward direction - **MUST align with destination, NOT obstacles**
- **Dark green circle**: 0.5m radius nearby area around current position
- **Purple markers**: Destination landmarks: {detected_landmarks}
- **Black**: Obstacles - **MUST AVOID**
- **Green areas**: Floor (Safe to move)
- **White areas**: Unexplored space
- **Orange line**: Trajectory history
- **Blue filled area**: Current visible navigable area (90° FOV, blocked by obstacles)
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
    "reasoning": "<(1) Subtask goal. (2) Finding of observation. (3) Map check: your position, orientation, landmark, obstacles.>",
    "action_analysis": "<Action decision: follow instruction OR adaptive fine-tuning with detailed reason>",
    "action": "TURN_LEFT" | "TURN_RIGHT" | "MOVE_FORWARD" | "STOP",
    "degrees": <30-180> (TURN only),
    "meters": <0.25-1.5> (MOVE_FORWARD only)
}}

# Examples

## Ex1 - Start turning to face the target:
**Sub-Instruction**: Turn left 90° to face the oven, then move forward 0.5m, Stop in front of oven.
**Previous Progress**: None
**Previous Action Reason**: None
**Current Observation**: Oven is not in front view; need to turn to face it.
{{
    "reasoning": "The subtask goal is to face the oven first. RGB: No oven visible in current front view. Map: Purple marker (oven) is to the left, far outside the dark green circle (0.5m radius), need to rotate first to face it.",
    "action_analysis": "Follow instruction - turn left 90° to align with oven direction.",
    "action": "TURN_LEFT",
    "degrees": 90
}}

## Ex2 - Continue with the instruction action:
**Sub-Instruction**: Turn left 90° to face the oven, then move forward 0.5m, Stop in front of oven.
**Previous Progress**: Had turned left 90°, then moved forward 0.5m.
**Previous Action Reason**: Continue moving forward to get closer to the oven.
**Current Observation**: Facing the oven, but the distance is still too far.
{{
    "reasoning": "The subtask goal is to stop at the oven. RGB & Detection: The oven is ahead, and there's space to move. Map: Purple marker (oven) is ahead but still outside the dark green circle (0.5m radius), meaning the destination is not yet reached. The path is clear with no obstacles.",
    "action_analysis": "Continue moving forward to get closer to the oven.",
    "action": "MOVE_FORWARD",
    "meters": 0.5
}}

## Ex3 - Arrive at destination and stop:
**Sub-Instruction**: Turn left 90° to face the oven, then move forward 0.5m, Stop in front of oven.
**Previous Progress**: Had turned left 90°, then moved forward 1.5m.
**Previous Action Reason**: Continue moving forward to get closer to the oven.
**Current Observation**: The oven is directly in front, very close (within 0.5m).
{{
    "reasoning": "The subtask goal is to stop at the oven. RGB: Oven clearly visible in front view. Detection: Oven detected. Map: Purple marker (oven) is now inside the dark green circle (0.5m radius), meaning destination has been reached. Red arrow overlaps with the destination, orange trajectory confirms arrival. All key actions completed.",
    "action_analysis": "All conditions met for STOP: oven detected, within 0.5m radius, visible in front view, and key actions completed.",
    "action": "STOP"
}}

## Ex4 - Detour around obstacle:
**Sub-Instruction**: Turn left 90° to face the oven, then move forward 0.5m, Stop in front of oven.
**Previous Progress**: Had turned left 90°.
**Previous Action Reason**: Follow instruction - turn left 90° to align with oven direction.
**Current Observation**: Oven is at front-left 30°, but straight ahead has a wall (black obstacle on map).
{{
    "reasoning": "The subtask goal is to reach the oven. RGB: Wall/obstacle blocking direct path ahead. Detection: Oven detected at left side. Map: Purple marker (oven) at front-left 30°, still outside the dark green circle (0.5m radius), black obstacle directly ahead blocking the path, green path to the left.",
    "action_analysis": "Adaptive adjustment: Turn left 30° to avoid obstacle and align toward oven, instead of moving forward into wall.",
    "action": "TURN_LEFT",
    "degrees": 30
}}



**CRITICAL EXECUTION RULES** (MUST FOLLOW):

1. **MULTIMODAL UNDERSTANDING** - Combine all 3 images for every decision:
   - **RGB (IMAGE 1)**: Observe visible environment, landmarks, obstacles
   - **Detection (IMAGE 2)**: Confirm which landmarks detected and positions
   - **Map (IMAGE 3)**: Your position (red arrow), instruction-related landmarks (purple), safe areas (green floor), obstacles (black)

2. **MAP NAVIGATION**:
   - Locate instruction landmarks: Purple markers show instruction-related objects, estimate distance/angle from red arrow
   - Plan safe path: Avoid black obstacles
   - If trapped by black: Turn toward nearest green floor area to escape, then re-orient toward destination

3. **ALIGNMENT REQUIREMENTS**:
   - **Map Forward Direction**: Dark red dashed line shows exact Forward direction - must point toward open paths/destination on map, NOT obstacles/walls.
   - **View Direction**: RGB view (Front view) should face toward visible destinations, NOT blocked by obstacles/walls. If facing blocked view, turn immediately to align toward destination using map guidance.
   - **Path Alignment**: Keep agent centered in corridors/paths, parallel to walls/boundaries with equal distance to both sides
   - **Target Alignment**: Keep destination/landmark centered in Front view (0°), face it directly without angular deviation

4. **Strictly FOLLOW INSTRUCTION & ADAPT**: Complete key actions (turn/move) specified in sub-instruction, but you can fine-tune angles/distances based on map and RGB - not rigidly bound to exact values

5. **STOP CONDITIONS** - Only STOP when ALL met:
   - Completed key actions in sub-instruction
   - Destination landmark detected in View + within <0.5m(destination is within map dark green circle) + visible in FRONT RGB view (maximized proximity before stop)
   - Arrived at destination area (destination is within map dark green circle)
   - Must have moved - orange trajectory on map confirms arrival at destination area

6. **ACTION PARAMETERS**:
   - Specify degrees (30-180) for TURN | meters (0.25-1.5) for MOVE_FORWARD
"""


def get_action_execution_prompt(subtask_destination: str,
                                subtask_instruction: str,
                                progress_summary: str = "",
                                detected_landmarks: str = None,
                                previous_action_reason: str = "") -> str:
    """
    获取动作执行提示词
    
    Args:
        subtask_destination: 子任务目的地
        subtask_instruction: 子任务指令
        progress_summary: 当前子任务进度摘要
        detected_landmarks: 已检测到的landmark类别字符串
        previous_action_reason: 上一步动作的action_analysis
        
    Returns:
        格式化的提示词字符串
    """
    if not detected_landmarks:
        detected_landmarks = "No landmarks detected yet"
    if not previous_action_reason:
        previous_action_reason = "None"
        
    return ACTION_EXECUTION_PROMPT.format(
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary if progress_summary else "(Just started - no actions yet)",
        detected_landmarks=detected_landmarks,
        previous_action_reason=previous_action_reason
    )
