"""
动作执行提示词模板
==================
用于VLM低层动作决策的提示词模板

动作参数与interactive_navigation保持一致：
- TURN_LEFT/RIGHT: 30°（12步×30°=360°）
- MOVE_FORWARD: 0.25m
"""

ACTION_EXECUTION_PROMPT = """You are executing navigation to reach {next_waypoint_destination}. Analyze view + map to decide: arrived OR move toward destination OR adjust pose to avoid obstacles.

# Current Sub-Task
**Destination**: {next_waypoint_destination}
**Sub-Instruction**: {subtask_instruction}
**Previous Progress**: {progress_summary}
**Last Action Reason**: {previous_action_reason}

# Visual Inputs

**IMAGE 1 - RGB View**: First-person view
- **Primary focus**: Is destination visible? Where is it? (front/left/right)

**IMAGE 2 - Detection View + Distance Labels**: Obstacle distances from your position
- **FRONT**: {distance_front} | **Left/Right 30°**: {distance_left_30}/{distance_right_30}
- **Left/Right 60°**: {distance_left_60}/{distance_right_60} | **Left/Right 90°**: {distance_left_90}/{distance_right_90}
- **Critical for obstacle avoidance**: < 0.5m = blocked, > 1.5m = clear

**IMAGE 3 - Local Map**: Bird's-eye view
- **Red arrow**: Your position/direction (top = FRONT)
- **Dark green circle**: 0.5m arrival radius
- **Black**: Obstacles (AVOID) | **Green**: Safe floor
- **Orange line**: Movement trajectory
- **Blue area**: 90° FOV cone

# Decision Process (Execute in Order)

**Step 0: Confirm Current Position** (FOUNDATION)
- What room/space am I in? What's around me? (RGB view + Map position)
- This establishes context for all decisions

**Step 1: Check Arrival** (HIGHEST PRIORITY)
- **Is destination visible in RGB view?** 
  * **Room destination**: Am I inside? See room features around?
  * **Object destination**: Is object in FRONT/nearby view and very close?
- **Check distance**: Map dark green circle (< 0.5m) OR RGB shows extremely close
- **If destination visible AND very close** → **STOP immediately** (avoid overshooting)

**Step 2: Check Obstacles** (if not arrived)
- **FRONT > 1.0m** = clear to move | **< 0.5m** = blocked, must turn first
- If blocked: Turn toward clearer side (check Left/Right 30-60°)

**Step 3: Navigate**
- **Path clear**: MOVE_FORWARD (0.25-1.0m toward destination)
- **Blocked**: TURN toward clear side (30-60°), then bypass and realign

# Available Actions

- **TURN_LEFT/RIGHT**: degrees = 30, 60, 90, 120, 150, 180
- **MOVE_FORWARD**: meters = 0.25, 0.5, 0.75, 1.0 (max 1.0m)
- **STOP**: (when < 0.5m at destination)

# Output (JSON only)

{{
    "reasoning": "<4 sentences: 0) Where am I? (current position/space) 1) Arrived? (check view+map) 2) Destination location? Obstacles? 3) Action decision>",
    "action_analysis": "<1-2 sentences: Why this action?>",
    "action": "STOP" | "MOVE_FORWARD" | "TURN_LEFT" | "TURN_RIGHT",
    "degrees": <30-180> (TURN only),
    "meters": <0.25-1.0> (MOVE only)
}}

# Examples
- **MOVE_FORWARD meters: MUST be 0.25-1.0 ONLY (Maximum 1.0m)**

## Ex1 - Arrived at destination (object):
**Destination**: Kitchen table
**Sub-Instruction**: Move forward to the kitchen table
**Progress**: Moved forward 1.5m
**Observation**: RGB shows table directly in front view very close, Detection FRONT 0.3m, Map shows trajectory approaching table

{{
    "reasoning": "Step 0: Currently in kitchen area, see kitchen features around. Step 1: Table visible in FRONT view, very close. Map confirms < 0.5m (inside dark green circle). Arrived. Step 2-3: Not needed.",
    "action_analysis": "Destination reached (table in front < 0.5m). Stop to avoid overshooting.",
    "action": "STOP"
}}

## Ex2 - Move forward (path clear):
**Destination**: Kitchen table
**Sub-Instruction**: Move forward to the kitchen table
**Progress**: Just started
**Observation**: RGB shows table ahead, Detection FRONT 1.5m clear, Left-90 0.8m, Right-90 1.2m

{{
    "reasoning": "Step 0: Currently in living room area near kitchen. Step 1: Table visible ahead but not close enough (> 0.5m). Not arrived. Step 2: FRONT 1.5m clear, no obstacles. Step 3: Move forward 0.75m toward table.",
    "action_analysis": "Path clear to destination. Advance toward table.",
    "action": "MOVE_FORWARD",
    "meters": 0.75
}}

## Ex3 - Obstacle blocking, adjust pose:
**Destination**: Bedroom doorway
**Sub-Instruction**: Walk to bedroom doorway
**Progress**: Moved 0.5m
**Observation**: RGB shows doorway ahead-right but furniture blocking, Detection FRONT 0.4m blocked, Right-30 1.8m clear

{{
    "reasoning": "Step 0: In hallway corridor, see walls on both sides. Step 1: Doorway visible but not reached. Step 2: FRONT 0.4m blocked by furniture, Right-30 1.8m clear. Step 3: Turn right 30° to bypass obstacle on clearer side.",
    "action_analysis": "Obstacle blocks direct path (FRONT < 0.5m). Adjust pose right to avoid.",
    "action": "TURN_RIGHT",
    "degrees": 30
}}

## Ex4 - Bypass obstacle then realign:
**Destination**: Bedroom doorway
**Progress**: Turned right 30°, moved 0.5m past obstacle
**Observation**: RGB shows doorway now at left side, Detection FRONT 2.0m clear, Left-30 1.5m

{{
    "reasoning": "Step 0: Still in hallway, furniture now behind me. Step 1: Doorway visible at left, not close yet. Bypassed obstacle. Step 2: FRONT clear now. Step 3: Turn left 30° to realign toward doorway.",
    "action_analysis": "Obstacle bypassed. Realign toward destination direction.",
    "action": "TURN_LEFT",
    "degrees": 30
}}

## Ex5 - Arrived at room (entered new space):
**Destination**: Exercise room
**Sub-Instruction**: Enter exercise room
**Progress**: Moved 2.0m, passed doorway
**Observation**: RGB shows gym equipment inside room (treadmill, weights), Map shows red arrow in new expanded green area

{{
    "reasoning": "Step 0: Now inside exercise room, surrounded by gym equipment. Step 1: Room destination confirmed - see treadmill, weights around. Map shows entered new room space (green area expanded). Arrived.",
    "action_analysis": "Room destination reached (RGB shows inside room + map confirms space transition). Stop.",
    "action": "STOP"
}}

**Critical Rules**:
1. **Arrival Check First**: Always check if arrived before any other action. Look at RGB view - if destination visible and very close, STOP immediately (avoid overshooting)
2. **Use Detection Distances**: < 0.5m = blocked, > 1.0m = safe to move
3. **Obstacle Avoidance**: If FRONT blocked, turn toward clearer side (check Left/Right 30-60° distances)
4. **Map Confirmation**: Use dark green circle (0.5m radius) to verify arrival distance
5. **Subtask Completion**: When destination appears in RGB view AND very close (< 0.5m on map or visually obvious), STOP - subtask complete"""


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
    """
    获取动作执行提示词
    
    Args:
        next_waypoint_destination: 下一个waypoint目的地
        subtask_instruction: 子任务指令
        progress_summary: 当前子任务进度摘要
        detected_landmarks: 已检测到的landmark类别字符串（可选，不强制要求）
        previous_action_reason: 上一步动作的action_analysis
        distance_front: 前方(0°)障碍物距离
        distance_left_30: 左前方(30°)障碍物距离
        distance_right_30: 右前方(30°)障碍物距离
        distance_left_60: 左前方(60°)障碍物距离
        distance_right_60: 右前方(60°)障碍物距离
        distance_left_90: 左侧(90°)障碍物距离
        distance_right_90: 右侧(90°)障碍物距离
        
    Returns:
        格式化的提示词字符串
    """
    if not previous_action_reason:
        previous_action_reason = "None"
    
    # 如果progress_summary为空，说明是刚开始
    if not progress_summary:
        progress_summary = "Just started"
        
    return ACTION_EXECUTION_PROMPT.format(
        next_waypoint_destination=next_waypoint_destination,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary,
        previous_action_reason=previous_action_reason,
        distance_front=distance_front,
        distance_left_30=distance_left_30,
        distance_right_30=distance_right_30,
        distance_left_60=distance_left_60,
        distance_right_60=distance_right_60,
        distance_left_90=distance_left_90,
        distance_right_90=distance_right_90
    )
