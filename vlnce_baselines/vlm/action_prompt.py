"""
动作执行提示词模板
==================
用于VLM低层动作决策的提示词模板

动作参数与interactive_navigation保持一致：
- TURN_LEFT/RIGHT: 30°（12步×30°=360°）
- MOVE_FORWARD: 0.25m
"""

ACTION_EXECUTION_PROMPT = """You are executing a navigation sub-task. Follow the sub-instruction to navigate toward the destination while observing the RGB view and avoiding obstacles.

# Current Sub-Task
**Destination**: {next_waypoint_destination}
**Sub-Instruction**: {subtask_instruction}
**Previous Progress**: {progress_summary}
**Last Action Reason**: {previous_action_reason}

# Visual Inputs

**IMAGE 1 - RGB View**: Your first-person view showing the environment
- **Look for your destination** ({next_waypoint_destination}) in this view
- Identify landmarks, room features, doorways, furniture
- Determine where the destination is located (front, left, right)

**IMAGE 2 - Detection View with Distance Labels**: Same view with obstacle distance warnings
- Shows 7 direction distance measurements from bottom center
- **FRONT**: {distance_front}
- **Left/Right 30°**: {distance_left_30} / {distance_right_30}
- **Left/Right 60°**: {distance_left_60} / {distance_right_60}
- **Left/Right 90°**: {distance_left_90} / {distance_right_90}
- **Use these distances to avoid obstacles** when planning movement

**IMAGE 3 - Local Map** (Bird's-eye view):
- **Red arrow**: Your current position and facing direction (map top = FRONT)
- **Dark green circle**: 0.5m arrival radius around you
- **Orange line**: Your trajectory history
- **Black areas**: Obstacles (MUST AVOID)
- **Green areas**: Safe floor areas
- **Blue area**: Your current 90° field of view
- **Watch for space changes**: Entering a new room shows as green area expansion on map

# Navigation Strategy

**Initial State**: You are already facing toward the destination direction (rotated automatically)

1. **Follow Sub-Instruction**: The sub-instruction guides you on how to reach the destination
   - Parse the instruction for key actions (turn left/right, move forward, etc.)
   - Execute these actions in sequence while adapting for obstacles

2. **Locate Destination**: Look at RGB view (IMAGE 1) - where is {next_waypoint_destination}?
   - Should be visible in front or front-side view (you're already rotated toward it)
   - If not visible yet: Follow sub-instruction navigation guidance

3. **Check Space Change**: Compare RGB view and Local Map together
   - **For room destinations**: If RGB shows you're inside the target room (see room features like furniture, walls) AND map shows new green space → You've arrived
   - **For object destinations**: Need to get close (<0.5m) to the specific object

4. **Check Obstacles**: Look at Detection view (IMAGE 2) distance labels in all 7 directions
   - If path has ">2.0m open" → Safe to move
   - If path has "<0.5m WARNING" → Must detour around obstacle

5. **Navigate**:
   - **If path clear**: Move forward toward destination (follow sub-instruction)
   - **If obstacle blocks**: Detour (turn 30-60° to clear side, move, then turn back)
   - **If arrived**: STOP (see arrival conditions below)

# Decision Priority

1. **Am I at destination?** → Check arrival condition:
   - **Room destination** (e.g., "kitchen", "bedroom", "exercise room"): RGB view shows inside the room + Map shows entered new space → STOP
   - **Object destination** (e.g., "table", "chair", "bed"): Object in FRONT view + <0.5m → STOP
2. **Can I move forward?** → Check FRONT distance label for obstacles
3. **Execute action** → Move forward toward destination OR detour around obstacle (turn to clear side, move, turn back)

# Actions Available

**Turn**: TURN_LEFT/RIGHT (30°, 60°, 90°, 120°, 150°, 180°)
**Move**: MOVE_FORWARD - **MUST be 0.25m to 1.0m only** (e.g., 0.25m, 0.5m, 0.75m, 1.0m)
**Arrive**: STOP (when destination in front AND <0.5m)

# Output Format (JSON)

**CRITICAL**: You MUST output ONLY valid JSON. No extra text before or after.
**Word Limits**: 
- "reasoning": MAX 120 words (4 sentences max)
- "action_analysis": MAX 50 words (2 sentences max)

{{
    "reasoning": "<4 sentences max: (1) Where am I? (2) Destination location in view? (3) Obstacles blocking? (4) Action plan.>",
    "action_analysis": "<2 sentences max: Arrived? OR Next action?>",
    "action": "TURN_LEFT" | "TURN_RIGHT" | "MOVE_FORWARD" | "STOP",
    "degrees": <30-180> (TURN only),
    "meters": <0.25-1.0 ONLY, MAX 1.0m> (MOVE_FORWARD only)
}}

# Examples

**FORMAT REQUIREMENTS**:
- Output ONLY the JSON object, no additional text
- Keep reasoning to 4 sentences (120 words max) - NO distance estimation needed
- Keep action_analysis to 2 sentences (50 words max)
- Use exact action names: "TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD", "STOP"
- **MOVE_FORWARD meters: MUST be 0.25-1.0 ONLY (Maximum 1.0m)**

## Ex1 - Move forward toward destination:
**Destination**: Kitchen table
**Sub-Instruction**: Move forward to the kitchen table
**Progress**: Just started
**Observation**: RGB view shows table directly ahead, Detection shows FRONT 1.5m
{{
    "reasoning": "Current: Living room, facing kitchen table. Destination: Table visible in front view (already rotated toward it). Obstacles: Front 1.5m allows movement. Plan: Move forward 0.75m toward table.",
    "action_analysis": "Not at destination (table ahead). Move forward to approach table.",
    "action": "MOVE_FORWARD",
    "meters": 0.75
}}

## Ex2 - Arrived at destination:
**Destination**: Kitchen table
**Sub-Instruction**: Move forward to the kitchen table
**Progress**: Moved forward 1.5m total
**Observation**: RGB view shows table very close in front, Detection shows FRONT 0.3m
{{
    "reasoning": "Current: Kitchen area. Destination: Table in front view, very close (<0.5m arrival radius). Obstacles: None. Plan: Stop, arrived at destination.",
    "action_analysis": "Arrived at destination (table in front view, <0.5m). Stop here.",
    "action": "STOP"
}}

## Ex3 - Detour around obstacle:
**Destination**: Bedroom doorway
**Sub-Instruction**: Walk straight to the bedroom doorway
**Progress**: Moved forward 0.5m
**Observation**: RGB view shows doorway ahead but furniture/wall blocking direct path, Detection shows FRONT <0.5m WARNING, Right-30 1.5m clear
{{
    "reasoning": "Current: Hallway. Destination: Bedroom doorway visible ahead-right but blocked. Obstacles: FRONT <0.5m blocked, Right-30 1.5m clear. Plan: Turn right 30° to bypass obstacle.",
    "action_analysis": "Direct path blocked (FRONT <0.5m). Detour: Turn right 30° to use clear path, will turn back toward doorway after bypassing.",
    "action": "TURN_RIGHT",
    "degrees": 30
}}

## Ex4 - Resume toward destination after detour:
**Destination**: Bedroom doorway  
**Sub-Instruction**: Walk straight to the bedroom doorway
**Progress**: Moved forward 0.5m, turned right 30°, moved forward 0.5m
**Observation**: RGB view shows doorway now at left 30°, Detection shows Left-30 1.2m, FRONT >2.0m open
{{
    "reasoning": "Current: Hallway, bypassed obstacle. Destination: Doorway at left 30° in view. Obstacles: All clear. Plan: Turn left 30° to face doorway again.",
    "action_analysis": "Bypassed obstacle. Doorway now at left 30°. Turn back toward destination.",
    "action": "TURN_LEFT",
    "degrees": 30
}}

## Ex5 - Arrived at room destination (entered new space):
**Destination**: Exercise room
**Sub-Instruction**: Enter the exercise room through the doorway
**Progress**: Moved forward 2.0m total, passed through doorway
**Observation**: RGB view shows exercise equipment inside room (treadmill, weights visible), Detection shows FRONT >2.0m, Left-90 1.65m, Right-90 >2.0m. Local map shows red arrow now in expanded green area (new room space).
{{
    "reasoning": "Current: Inside exercise room (can see gym equipment around me). Destination: Exercise room - already entered. Space change: Map shows I've moved from hallway into new room (green area expanded). Plan: Stop, destination reached.",
    "action_analysis": "Arrived at destination (room destination). RGB view confirms inside exercise room with equipment visible. Map confirms space transition from hallway to room. Stop here.",
    "action": "STOP"
}}

**CRITICAL RULES**:

1. **Already Facing Destination**: You start already rotated toward the destination direction - destination should be in front or front-side view
2. **Prioritize RGB View + Map Together**: Look at IMAGE 1 (RGB) to see what's around you, and IMAGE 3 (Local Map) to understand space changes
3. **Room vs Object Destinations**:
   - **Room** (kitchen, bedroom, exercise room, etc.): STOP when RGB shows you're inside the room + Map shows new space entered
   - **Object** (table, chair, bed, etc.): STOP when object in FRONT view + <0.5m
4. **Use Distance Labels**: Check IMAGE 2 (Detection view) distance warnings before moving
5. **Turn Only for Obstacles**: Only turn to detour around obstacles (<0.5m blocking front path), then turn back to face destination"""


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
