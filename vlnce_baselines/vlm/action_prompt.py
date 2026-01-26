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
    "reasoning": "<MAX 4 sentences: 1) Current Position: Where am I NOW? (Based on NEAR objects < 0.5m in RGB view + Local Map position - NOT distant objects) 2) Destination: What is my destination? Where is it located? (In which direction/view? How far?) 3) Arrival Check: Have I reached the destination? (Is destination beside me or am I AT the destination? If yes, must STOP) 4) Navigation Action: If not arrived, how should I move to reach destination? (Are there obstacles ahead? How to avoid obstacles and move toward destination?)>",
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
    "reasoning": "1) Current Position: In kitchen area - see kitchen counters, cabinets surrounding me (NEAR objects in RGB view). Local Map shows red arrow in kitchen space. 2) Destination: Kitchen table is my destination. RGB view shows table directly in FRONT, very close - large in view, occupying significant area. 3) Arrival Check: YES, reached destination. Table is right in front < 0.5m (inside Local Map dark green circle). Am AT the table position. Must STOP. 4) Navigation Action: Not needed - already arrived at destination.",
    "action_analysis": "Destination reached (table in front < 0.5m). Stop to avoid overshooting.",
    "action": "STOP"
}}

## Ex2 - Turn toward open path (doorway with obstacle):
**Destination**: Living room
**Sub-Instruction**: Walk straight into the living room
**Progress**: Standing at doorway threshold
**Observation**: RGB shows living room visible ahead-left through doorway, wall on right. Detection FRONT 0.70m (wall), Left-30 >2.0m open, Right-30 0.78m, Left-60 1.65m, Right-60 1.00m, Left-90 >2.0m open, Right-90 0.85m

{{
    "reasoning": "1) Current Position: Standing at doorway threshold - see wall on right side (NEAR object in RGB view), floor tiles visible. Local Map shows red arrow at doorway entrance. 2) Destination: Living room is my destination. RGB view shows living room visible ahead-left through doorway opening. 3) Arrival Check: NO, not arrived. Living room visible but not entered yet (still at doorway threshold). 4) Navigation Action: FRONT obstacle distance 0.70m - NOT enough clearance (close to wall). Left-30° shows >2.0m open (very clear) - this is the doorway opening direction toward living room. Right side has walls (0.78m-0.85m). Must turn LEFT 30° first to face the open doorway path toward destination, then can advance.",
    "action_analysis": "FRONT blocked by wall (0.70m). Left-30° is open (>2.0m) toward living room. Turn left to align with doorway opening.",
    "action": "TURN_LEFT",
    "degrees": 30
}}

## Ex3 - Move forward (path clear):
**Destination**: Kitchen table
**Sub-Instruction**: Move forward to the kitchen table
**Progress**: Just started
**Observation**: RGB shows table ahead, Detection FRONT 1.5m clear, Left-30 1.2m, Right-30 1.0m, Left-60 0.9m, Right-60 1.1m, Left-90 0.8m, Right-90 1.2m

{{
    "reasoning": "1) Current Position: In living room area near kitchen entrance - see living room furniture around me (NEAR). Local Map red arrow shows position in living room. 2) Destination: Kitchen table is my destination. RGB view shows table visible ahead in distance - small in view, FAR away (not NEAR yet). 3) Arrival Check: NO, not arrived. Table visible ahead but > 0.5m away (outside Local Map green circle). Not AT destination yet. 4) Navigation Action: FRONT obstacle distance 1.5m - clear, safe to move. All directions show adequate clearance (> 0.8m). No obstacles blocking path. Move forward 0.75m toward table to approach destination.",
    "action_analysis": "Path clear to destination (FRONT 1.5m). Advance toward table.",
    "action": "MOVE_FORWARD",
    "meters": 0.75
}}

## Ex4 - Obstacle blocking, adjust pose:
**Destination**: Bedroom doorway
**Sub-Instruction**: Walk to bedroom doorway
**Progress**: Moved 0.5m
**Observation**: RGB shows doorway ahead-right but furniture blocking, Detection FRONT 0.4m blocked, Left-30 0.6m, Right-30 1.8m clear, Left-60 0.8m, Right-60 2.0m, Left-90 1.0m, Right-90 1.5m

{{
    "reasoning": "1) Current Position: In hallway corridor - see hallway walls on both sides (NEAR objects in RGB view), narrow passage. Local Map shows red arrow in corridor space. 2) Destination: Bedroom doorway is my destination. RGB view shows doorway visible ahead-right direction but furniture blocking direct path. 3) Arrival Check: NO, not arrived. Doorway visible but not reached yet, still in hallway (not bedroom). 4) Navigation Action: FRONT obstacle distance 0.4m - BLOCKED by furniture (< 0.5m). Right-30° shows 1.8m clear (safe direction), Right-60° shows 2.0m even clearer. Turn right 30° to bypass obstacle on clearer side, then can move toward destination.",
    "action_analysis": "Obstacle blocks direct path (FRONT 0.4m < 0.5m). Right side clear (1.8m). Adjust pose right to avoid.",
    "action": "TURN_RIGHT",
    "degrees": 30
}}

## Ex5 - Bypass obstacle then realign:
**Destination**: Bedroom doorway
**Progress**: Turned right 30°, moved 0.5m past obstacle
**Observation**: RGB shows doorway now at left side, Detection FRONT 2.0m clear, Left-30 1.5m, Right-30 1.2m, Left-60 1.8m, Right-60 1.0m, Left-90 1.6m, Right-90 0.9m

{{
    "reasoning": "1) Current Position: Still in hallway corridor - furniture now behind me (NEAR obstacles passed). Local Map shows red arrow still in hallway but moved past previous obstacle. 2) Destination: Bedroom doorway is my destination. RGB view shows doorway now visible at LEFT side - need to realign toward it. 3) Arrival Check: NO, not arrived yet. Doorway visible at left but not reached, still in hallway corridor. Bypassed obstacle successfully. 4) Navigation Action: FRONT obstacle distance clear now (2.0m). Obstacle bypassed, but destination (doorway) now at left direction. Left-30° shows 1.5m clear. Turn left 30° to realign toward doorway direction, then can continue moving toward destination.",
    "action_analysis": "Obstacle bypassed. Destination at left. Realign toward doorway direction.",
    "action": "TURN_LEFT",
    "degrees": 30
}}

## Ex6 - Arrived at room (entered new space):
**Destination**: Exercise room
**Sub-Instruction**: Enter exercise room
**Progress**: Moved 2.0m, passed doorway
**Observation**: RGB shows gym equipment inside room (treadmill, weights), Map shows red arrow in new expanded green area

{{
    "reasoning": "1) Current Position: Inside exercise room - see gym equipment surrounding me (treadmill, weights visible as NEAR objects in RGB view, large, < 1.0m, occupying view). Local Map red arrow shows position in expanded green area (new room space). 2) Destination: Exercise room is my destination. RGB view confirms inside exercise room - gym equipment all around defining this space. 3) Arrival Check: YES, arrived at destination. AM IN exercise room (room destination), see room features surrounding. Local Map shows space transition - entered new room area (green area expanded). Must STOP. 4) Navigation Action: Not needed - already arrived at room destination.",
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
