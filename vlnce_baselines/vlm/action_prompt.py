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
- **CRITICAL**: Identify room context and objects with format: [room]'s [object] distance

**IMAGE 2 - Detection View + Distance Labels**: Obstacle distances from your position
- **FRONT**: {distance_front} | **Left/Right 30°**: {distance_left_30}/{distance_right_30}
- **Left/Right 60°**: {distance_left_60}/{distance_right_60} | **Left/Right 90°**: {distance_left_90}/{distance_right_90}
- **Critical for obstacle avoidance**: < 0.5m = blocked, > 1.0m = safe, > 1.5m = very clear

**IMAGE 3 - Local Map**: Bird's-eye view
- **Red arrow**: Your position/direction (top = FRONT)
- **Dark green circle**: 0.5m arrival radius
- **Black**: Obstacles (AVOID) | **Green**: Safe floor
- **Orange line**: Movement trajectory
- **Blue area**: 90° FOV cone

# Decision Process (Execute in Order)

**Step 0: Observation Analysis** (What Do I See?)
- **RGB View**: What objects/rooms visible? NEAR (<1.0m, large) vs FAR (>1.5m, small)? Where is destination?
- **Distance Labels**: 7 directions obstacle distances - which clear (>1.0m)? which blocked (<0.5m)?
- **Local Map**: Red arrow position? Green safe areas? Black obstacles? Orange trajectory path?
- **NO HALLUCINATION**: Only describe what's ACTUALLY visible in images

**Step 1: Position Determination** (Where Am I? - Detailed Location)
- Analyze NEAR objects (<1.0m, large in RGB) + map position → identify current room AND specific position within room
- **Format**: "I am in [ROOM] near [specific object]" - Example: "kitchen's counter 0.8m, kitchen's sink 1.0m, kitchen's refrigerator 1.5m → I am in KITCHEN near counter and sink"
- **Be Specific**: Not just "in living room", but "in living room near sofa", "in living room between coffee table and TV"
- **Compare with Destination**: Current position vs destination position - Example: "I am in living room near sofa, destination is living room's dining table"

**Step 2: Arrival & Destination Check** (Critical Decision Point)
- **Case A - Arrived**: Destination FILLING VIEW (close, prominent, "in my face") → **STOP immediately**
- **Case B - Destination Far**: Destination visible but SMALL/DISTANT in view → continue navigation
- **Case C - Cannot Find Destination**: Destination not visible OR confused about location → **STOP immediately** (avoid wandering)
- **Visual judgment is primary**, map is auxiliary

**Step 3: Navigation Decision** (if Case B - destination far, continue navigation)
- Analyze 7 directions: "DIRECTION: [room]'s [object] distance" 
- Navigation logic:
  * FRONT ≥1.0m clear + destination ahead → MOVE_FORWARD
  * FRONT <0.5m blocked → TURN toward clearer side
  * Destination at side angles → TURN toward it
- Distance rules: <0.5m blocked, 1.0-2.0m safe, >2.0m very clear

# Available Actions

- **TURN_LEFT/RIGHT**: degrees = 30, 60, 90, 120, 150, 180
- **MOVE_FORWARD**: meters = 0.25, 0.5, 0.75, 1.0 (max 1.0m)
- **STOP**: (when < 0.5m at destination)

# Output (JSON only)

{{
    "reasoning": "<4-step reasoning: 0) Observation Analysis: RGB shows what? Distance labels show which directions clear/blocked? Map shows position where? Destination visible where? NO HALLUCINATION - only describe actual observations. 1) Position Determination: NEAR objects (<1.0m, large in RGB) indicate room AND specific position. Format: 'I am in [ROOM] near [specific object]'. Compare with destination position. Example: 'I am in living room near sofa, destination is living room's dining table'. 2) Arrival & Destination Check: Case A (Arrived) - Destination FILLING VIEW (prominent, close) → STOP. Case B (Far) - Destination SMALL/DISTANT → continue. Case C (Cannot Find) - Destination not visible OR confused → STOP immediately. 3) Navigation Decision (if Case B): Analyze 7 directions with room context. FRONT: [room]'s [object] distance. Decision: move forward / turn / why?>",
    "action_analysis": "<1-2 sentences: Why this action with room context?>",
    "action": "STOP" | "MOVE_FORWARD" | "TURN_LEFT" | "TURN_RIGHT",
    "degrees": <30-180> (TURN only),
    "meters": <0.25-1.0> (MOVE only)
}}

# Examples
- **MOVE_FORWARD meters: MUST be 0.25-1.0 ONLY (Maximum 1.0m)**
- **CRITICAL**: Every object MUST include room context: [room]'s [object] distance

## Ex1 - Arrived at destination (object):
**Destination**: Kitchen table
**Sub-Instruction**: Move forward to the kitchen table
**Progress**: Moved forward 1.5m
**Observation**: RGB shows table directly in front view very close, Detection FRONT 0.3m, Map shows trajectory approaching table

{{
    "reasoning": "0) Observation Analysis: RGB shows kitchen's table directly ahead VERY CLOSE (large, filling most of view), kitchen's counter 0.9m, kitchen's cabinets 1.0m. Distance labels: FRONT 0.3m. Map: red arrow inside green circle near table. NO HALLUCINATION - only actual objects visible. 1) Position Determination: NEAR objects kitchen's counter 0.9m, kitchen's cabinets 1.0m, kitchen's sink 1.2m → I am in KITCHEN near counter and cabinets. Destination: kitchen's table 0.3m directly ahead. Position comparison: very close to destination object. 2) Arrival & Destination Check: Destination kitchen's table FILLING ENTIRE VIEW (extremely close, RIGHT IN FRONT, prominent) → Case A (Arrived). Visual judgment: table IN MY FACE. Map confirms inside green circle. STOP immediately. 3) Navigation: Not needed - arrived.",
    "action_analysis": "Arrived at kitchen's table (NEAR 0.3m, filling RGB view, inside green circle). Stop.",
    "action": "STOP"
}}

## Ex2 - Turn toward open path (doorway with obstacle):
**Destination**: Living room
**Sub-Instruction**: Walk straight into the living room
**Progress**: Standing at doorway threshold
**Observation**: RGB shows living room visible ahead-left through doorway, wall on right. Detection FRONT 0.70m (wall), Left-30 >2.0m open, Right-30 0.78m, Left-60 1.65m, Right-60 1.00m, Left-90 >2.0m open, Right-90 0.85m

{{
    "reasoning": "0) Observation Analysis: RGB shows doorway's frame 0.7m ahead, hallway's wall on right 0.78m, living room's furniture visible ahead-left through opening. Distance labels: FRONT 0.70m, Left-30° >2.0m open, Right-30° 0.78m. Map: red arrow at doorway threshold. 1) Position Determination: Doorway's frame 0.7m, hallway's wall 0.78m → I am at DOORWAY THRESHOLD between hallway and living room, standing at entrance. Destination: inside living room (need to pass through doorway). Position comparison: at doorway entrance, destination is inside room beyond. 2) Arrival & Destination Check: Living room's furniture SMALL/DISTANT through opening (not filling view, far away) → Case B (Far). Not inside room yet, destination distant. Continue navigation. 3) Navigation Decision: FRONT doorway's wall 0.70m (partially blocked). Left-30° living room's opening >2.0m clear (toward destination). Right-30° hallway's wall 0.78m. Turn LEFT 30° to align with living room's doorway opening.",
    "action_analysis": "Doorway's wall blocks FRONT (0.70m). Left-30° shows living room's opening (>2.0m). Turn left to align.",
    "action": "TURN_LEFT",
    "degrees": 30
}}

## Ex3 - Move forward (path clear):
**Destination**: Kitchen table
**Sub-Instruction**: Move forward to the kitchen table
**Progress**: Just started
**Observation**: RGB shows table ahead, Detection FRONT 1.5m clear, Left-30 1.2m, Right-30 1.0m, Left-60 0.9m, Right-60 1.1m, Left-90 0.8m, Right-90 1.2m

{{
    "reasoning": "0) Observation Analysis: RGB shows kitchen's table ahead FAR (small, ~1.5m), living room's sofa 1.1m, living room's coffee table 1.3m. Distance labels: FRONT 1.5m clear, Left-30° 1.2m, Right-30° 1.0m. Map: red arrow in living room near kitchen entrance. 1) Position Determination: Living room's sofa 1.1m, living room's coffee table 1.3m → I am in LIVING ROOM near sofa and coffee table area, close to kitchen entrance. Destination: kitchen's table ahead. Position comparison: in living room near sofa, destination is kitchen's table (different area/room). 2) Arrival & Destination Check: Kitchen's table SMALL/DISTANT ahead (not filling view, far away) → Case B (Far). Not at destination, continue navigation. 3) Navigation Decision: FRONT kitchen's table direction 1.5m clear (>1.0m safe). Left-30° living room's wall 1.2m. Right-30° kitchen's doorway 1.0m. Path clear ahead. Move forward 0.75m toward kitchen's table.",
    "action_analysis": "Kitchen's table ahead FAR. FRONT clear 1.5m. Advance toward destination.",
    "action": "MOVE_FORWARD",
    "meters": 0.75
}}

## Ex4 - Obstacle blocking, adjust pose:
**Destination**: Bedroom doorway
**Sub-Instruction**: Walk to bedroom doorway
**Progress**: Moved 0.5m
**Observation**: RGB shows doorway ahead-right but furniture blocking, Detection FRONT 0.4m blocked, Left-30 0.6m, Right-30 1.8m clear, Left-60 0.8m, Right-60 2.0m, Left-90 1.0m, Right-90 1.5m

{{
    "reasoning": "0) Observation Analysis: RGB shows hallway's furniture 0.4m ahead blocking, hallway's wall 0.6m left, bedroom's doorway visible ahead-right FAR. Distance labels: FRONT 0.4m blocked, Left-30° 0.6m, Right-30° 1.8m clear, Right-60° 2.0m. Map: red arrow in hallway. 1) Position Determination: Hallway's wall 0.6m, hallway's furniture 0.4m → I am in HALLWAY corridor, blocked by furniture obstacle ahead. Destination: bedroom's doorway (ahead-right direction). Position comparison: in narrow hallway with furniture blocking, destination is bedroom doorway beyond obstacle. 2) Arrival & Destination Check: Bedroom's doorway SMALL/DISTANT in side view (not filling view, far away) → Case B (Far). Still in hallway, continue. 3) Navigation Decision: FRONT hallway's furniture 0.4m BLOCKED (<0.5m). Right-30° hallway's open path 1.8m clear. Right-60° bedroom's doorway direction 2.0m clear. Turn RIGHT 30° to bypass furniture obstacle.",
    "action_analysis": "Hallway's furniture blocks FRONT (0.4m). Right-30° clear 1.8m toward bedroom's doorway. Turn right to avoid.",
    "action": "TURN_RIGHT",
    "degrees": 30
}}

## Ex5 - Bypass obstacle then realign:
**Destination**: Bedroom doorway
**Progress**: Turned right 30°, moved 0.5m past obstacle
**Observation**: RGB shows doorway now at left side, Detection FRONT 2.0m clear, Left-30 1.5m, Right-30 1.2m, Left-60 1.8m, Right-60 1.0m, Left-90 1.6m, Right-90 0.9m

{{
    "reasoning": "0) What Room Am I In?: RGB view shows hallway's wall 1.5m, hallway's furniture 0.9m behind (passed) - I am in HALLWAY. 1) Where Am I Relative to Destination?: Current room: hallway. Destination: bedroom's doorway. RGB view shows bedroom's doorway at LEFT side (~1.5m). 2) Arrival Check - VISUAL ANALYSIS: RGB view shows bedroom's doorway at LEFT side, SMALL/DISTANT (not filling view, not right in front, not prominent). Visually NOT at destination - doorway appears distant at side. Still in hallway. 3) Navigation Decision: FRONT: hallway's wall 2.0m clear (not toward destination). Left-30°: bedroom's doorway direction 1.5m. Turn LEFT 30° to realign toward bedroom's doorway.",
    "action_analysis": "Obstacle bypassed. Destination at left. Realign toward doorway direction.",
    "action": "TURN_LEFT",
    "degrees": 30
}}

## Ex5 - Bypass obstacle then realign:
**Destination**: Bedroom doorway
**Progress**: Turned right 30°, moved 0.5m past obstacle
**Observation**: RGB shows doorway now at left side, Detection FRONT 2.0m clear, Left-30 1.5m, Right-30 1.2m, Left-60 1.8m, Right-60 1.0m, Left-90 1.6m, Right-90 0.9m

{{
    "reasoning": "0) Observation Analysis: RGB shows bedroom's doorway at LEFT side (~1.5m), hallway's wall 1.5m left, hallway's furniture 0.9m behind (passed). Distance labels: FRONT 2.0m, Left-30° 1.5m, Left-60° 1.8m. Map: red arrow in hallway past obstacle. 1) Position Determination: Hallway's wall 1.5m, hallway's furniture 0.9m behind → I am in HALLWAY corridor, past the furniture obstacle. Destination: bedroom's doorway at left side. Position comparison: in hallway clear area, destination doorway now visible to left. 2) Arrival & Destination Check: Bedroom's doorway SMALL/DISTANT at left side (not filling view, not prominent) → Case B (Far). Still in hallway, continue. 3) Navigation Decision: FRONT hallway's wall 2.0m (not toward destination). Left-30° bedroom's doorway direction 1.5m clear. Turn LEFT 30° to realign toward bedroom's doorway.",
    "action_analysis": "Hallway's furniture bypassed. Bedroom's doorway at left 1.5m. Turn left to realign toward destination.",
    "action": "TURN_LEFT",
    "degrees": 30
}}

## Ex6 - Arrived at room (entered new space):
**Destination**: Exercise room
**Sub-Instruction**: Enter exercise room
**Progress**: Moved 2.0m, passed doorway
**Observation**: RGB shows gym equipment inside room (treadmill, weights), Map shows red arrow in new expanded green area

{{
    "reasoning": "0) Observation Analysis: RGB shows exercise room's treadmill 0.8m NEAR (large), exercise room's weights 0.9m, exercise room's mat 1.0m - gym equipment SURROUNDING. Distance labels confirm close objects. Map: red arrow in expanded green area (new room). 1) Position Determination: Exercise room's treadmill 0.8m, exercise room's weights 0.9m, exercise room's mat 1.0m → I am INSIDE EXERCISE ROOM, surrounded by gym equipment (treadmill and weights nearby). Destination: exercise room. Position comparison: inside exercise room with equipment all around - at destination. 2) Arrival & Destination Check: Destination exercise room. Equipment SURROUNDING ME (large, filling view, prominent in all directions) → Case A (Arrived). Room destination achieved, INSIDE exercise room. STOP immediately. 3) Navigation: Not needed - arrived.",
    "action_analysis": "Exercise room destination reached (inside room, exercise room's equipment surrounding, map confirms space transition). Stop.",
    "action": "STOP"
}}

**Critical Rules**:
1. **Room Context MANDATORY**: Every object → [room]'s [object] distance (prevents confusion: kitchen's chair vs living room's chair)
2. **4-Step Process**: 0) Observe (what's visible?) → 1) Position (where am I?) → 2) Arrival (arrived/far/cannot find?) → 3) Navigate (if far)
3. **NO HALLUCINATION**: Only describe what's ACTUALLY visible in RGB + map
4. **Arrival Decision (Step 2)**:
   - Case A: Destination FILLING VIEW (close, prominent) → STOP immediately
   - Case B: Destination SMALL/DISTANT → continue navigation
   - Case C: Cannot find destination OR confused about location → STOP immediately (avoid wandering)
5. **Focus on Visual + Map**: Analyze RGB view and local map carefully to determine position and destination status
6. **Distance Rules**: <0.5m blocked, 1.0-2.0m safe, >2.0m very clear"""


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
