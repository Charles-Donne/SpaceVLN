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

**Step 0: What Room Am I In?** (Position Foundation)
- Analyze NEAR objects (< 1.0m, large in RGB) to identify current room
- Format: "[room]'s [object] distance" - Example: "kitchen's counter 0.8m, kitchen's sink 1.0m - I am in KITCHEN"

**Step 1: Where Am I Relative to Destination?** (Distance Check)
- Current room vs destination room? Same or different?
- Is destination visible in RGB? If yes: NEAR (< 1.0m, large) or FAR (> 1.5m, small)?
- Format destination as: "[room]'s [object] distance"

**Step 2: Arrival Check** (VISUAL ANALYSIS PRIMARY)
- **Look at RGB view**: Is destination FILLING VIEW or OCCUPYING MAJOR PORTION?
  * Filling/occupying view (close, prominent, "in my face") → **STOP immediately**
  * Small/distant in view (far away) → continue navigation
- **Local map is auxiliary only** - Visual judgment dominates decision

**Step 3: Detailed Obstacle Analysis & Navigation Decision** (if not arrived)
- Analyze ALL 7 directions: "DIRECTION: [room]'s [object] distance"
- Navigation logic:
  * FRONT ≥ 1.0m clear + destination ahead → MOVE_FORWARD
  * FRONT < 0.5m blocked → TURN toward clearer side (compare Left/Right 30-60°)
  * Destination at side angles → TURN toward it first
- Distance rules: < 0.5m blocked, 1.0-2.0m safe, > 2.0m very clear

# Available Actions

- **TURN_LEFT/RIGHT**: degrees = 30, 60, 90, 120, 150, 180
- **MOVE_FORWARD**: meters = 0.25, 0.5, 0.75, 1.0 (max 1.0m)
- **STOP**: (when < 0.5m at destination)

# Output (JSON only)

{{
    "reasoning": "<4-step reasoning: 0) What Room Am I In?: Analyze NEAR objects (< 1.0m, large in RGB) to identify room. Format: '[room]'s [object] distance'. Example: 'kitchen's counter 0.8m, kitchen's sink 1.0m - I am in KITCHEN'. 1) Where Am I Relative to Destination?: Current room? Destination room? Destination visible? Format destination as '[room]'s [object]'. If visible: NEAR (< 1.0m, large) or FAR (> 1.5m, small)? 2) Arrival Check - VISUAL ANALYSIS: Look at RGB view - Is destination RIGHT IN FRONT filling/occupying major portion of view? If destination PROMINENT and CLOSE in RGB (not small/distant) → AT destination, STOP. If destination small/distant in RGB → NOT arrived. Local map is auxiliary only. 3) Navigation Decision (if not arrived): Analyze obstacles with room context. FRONT: [room]'s [object] distance. All 7 directions. Decision: move forward / turn left/right / why?>",
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
    "reasoning": "0) What Room Am I In?: RGB view shows kitchen's counter 0.9m NEAR (large in view), kitchen's cabinets 1.0m, kitchen's sink 1.2m - I am in KITCHEN room. 1) Where Am I Relative to Destination?: Current room: KITCHEN. Destination: kitchen's table. RGB view shows kitchen's table directly in FRONT VERY NEAR (0.3m, large, occupying most of view). 2) Arrival Check - VISUAL ANALYSIS: RGB view shows kitchen's table FILLING ENTIRE VIEW (extremely close, occupying most of visual field, RIGHT IN FRONT of me, not small/distant). Visual judgment: table is IN MY FACE - AT destination! Local map confirms (inside green circle) but visual analysis is primary. Must STOP. 3) Navigation: Not needed - visually AT kitchen's table (filling view).",
    "action_analysis": "Arrived at kitchen's table (NEAR 0.3m, filling RGB view, inside green circle). Stop.",
    "action": "STOP"
}}

## Ex2 - Turn toward open path (doorway with obstacle):
**Destination**: Living room
**Sub-Instruction**: Walk straight into the living room
**Progress**: Standing at doorway threshold
**Observation**: RGB shows living room visible ahead-left through doorway, wall on right. Detection FRONT 0.70m (wall), Left-30 >2.0m open, Right-30 0.78m, Left-60 1.65m, Right-60 1.00m, Left-90 >2.0m open, Right-90 0.85m

{{
    "reasoning": "0) What Room Am I In?: RGB view shows doorway's frame 0.7m ahead, hallway's wall 0.78m on right - I am at DOORWAY THRESHOLD between hallway and living room. 1) Where Am I Relative to Destination?: Current position: doorway threshold. Destination: living room (through doorway opening). RGB view shows living room's furniture visible ahead-left FAR (through opening), living room not entered yet. 2) Arrival Check - VISUAL ANALYSIS: RGB view shows living room SMALL/DISTANT through doorway opening (not filling view, not prominent, far away). Visually NOT at destination yet - living room space is distant, not surrounding me. Not inside room yet. 3) Navigation Decision: Analyze obstacles - FRONT: doorway's wall edge 0.70m (partially blocked, not clear). Left-30°: living room's open floor >2.0m (very clear, doorway opening toward destination!). Right-30°: hallway's wall 0.78m. Left-60°: living room's floor 1.65m clear. Right-60°: hallway's wall 1.00m. Left-90°: living room's space >2.0m open. Right-90°: hallway's wall 0.85m. Decision: FRONT not clear (0.70m). LEFT directions show living room's open space (>2.0m clear) - this is doorway opening toward destination. Turn LEFT 30° to align with living room's doorway opening path.",
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
    "reasoning": "0) What Room Am I In?: RGB view shows living room's sofa 1.1m, living room's coffee table 1.3m - I am in LIVING ROOM near kitchen entrance. 1) Where Am I Relative to Destination?: Current room: living room. Destination: kitchen's table. RGB view shows kitchen's table visible ahead FAR (~1.5m, small in view, not NEAR). 2) Arrival Check - VISUAL ANALYSIS: RGB view shows kitchen's table SMALL and DISTANT ahead (not filling view, not prominent, far away). Visually NOT at destination - table appears small/distant in view, not close. Not AT destination. 3) Navigation Decision: FRONT: kitchen's table direction 1.5m clear (safe, > 1.0m). Left-30°: living room's wall 1.2m. Right-30°: kitchen's doorway 1.0m. Left-60°: living room's furniture 0.9m. Right-60°: kitchen's counter 1.1m. Left-90°: living room's cabinet 0.8m. Right-90°: kitchen's appliances 1.2m. Path clear ahead (FRONT 1.5m > 1.0m), no obstacles blocking. Move forward 0.75m toward kitchen's table.",
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
    "reasoning": "0) What Room Am I In?: RGB view shows hallway's wall 0.6m on left, hallway's furniture 0.4m ahead blocking - I am in HALLWAY corridor. 1) Where Am I Relative to Destination?: Current room: hallway. Destination: bedroom's doorway. RGB view shows bedroom's doorway visible ahead-right FAR (~2.0m) but hallway's furniture blocking direct path. 2) Arrival Check - VISUAL ANALYSIS: RGB view shows bedroom's doorway SMALL/DISTANT in side view (not filling view, not prominent, far away). Visually NOT at destination - doorway appears distant. Still in hallway. 3) Navigation Decision: FRONT: hallway's furniture 0.4m BLOCKED (< 0.5m, too close). Left-30°: hallway's wall 0.6m (not good). Right-30°: hallway's open path 1.8m clear (safe, > 1.0m). Left-60°: hallway's wall 0.8m. Right-60°: bedroom's doorway direction 2.0m very clear. Left-90°: hallway's picture 1.0m. Right-90°: hallway's corner 1.5m. Decision: FRONT blocked by hallway's furniture (0.4m < 0.5m). RIGHT directions clearer (Right-30° 1.8m, Right-60° 2.0m toward bedroom's doorway). Turn RIGHT 30° to bypass hallway's furniture obstacle.",
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
    "reasoning": "0) What Room Am I In?: RGB view shows hallway's wall 1.5m on left, hallway's furniture 0.9m behind on right (passed) - I am still in HALLWAY corridor. 1) Where Am I Relative to Destination?: Current room: hallway. Destination: bedroom's doorway. RGB view shows bedroom's doorway now visible at LEFT side (~1.5m). Hallway's furniture obstacle now behind me (bypassed). 2) Arrival Check - VISUAL ANALYSIS: RGB view shows bedroom's doorway at LEFT side, SMALL/DISTANT (not filling view, not right in front, not prominent). Visually NOT at destination - doorway appears distant at side angle. Still in hallway. 3) Navigation Decision: FRONT: hallway's wall 2.0m clear (not toward destination). Left-30°: bedroom's doorway direction 1.5m (destination here, clear). Right-30°: hallway's corner 1.2m. Left-60°: bedroom's door frame 1.8m (destination closer). Right-60°: hallway's furniture 1.0m (obstacle). Left-90°: bedroom's entrance 1.6m. Right-90°: hallway's furniture 0.9m (passed obstacle). Decision: Hallway's furniture obstacle bypassed successfully. Bedroom's doorway now at LEFT angles (Left-30° 1.5m, Left-60° 1.8m). FRONT goes to hallway's wall (not destination). Turn LEFT 30° to realign toward bedroom's doorway direction.",
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
    "reasoning": "0) What Room Am I In?: RGB view shows exercise room's treadmill 0.8m NEAR (large in view), exercise room's weights 0.9m, exercise room's mat 1.0m - I am INSIDE EXERCISE ROOM. 1) Where Am I Relative to Destination?: Current room: exercise room (just entered). Destination: exercise room. RGB view confirms exercise room's gym equipment all around me defining this space. 2) Arrival Check - VISUAL ANALYSIS: RGB view shows exercise room's equipment SURROUNDING ME (large, close, filling view, prominent in all directions). Visually INSIDE exercise room - equipment occupies major portion of view, not small/distant. Room destination achieved! AT destination. 3) Navigation: Not needed - visually confirmed inside exercise room (equipment surrounding).",
    "action_analysis": "Exercise room destination reached (inside room, exercise room's equipment surrounding, map confirms space transition). Stop.",
    "action": "STOP"
}}

**Critical Rules**:
1. **Room Context MANDATORY**: Every object → [room]'s [object] distance (prevents confusion: kitchen's chair vs living room's chair)
2. **Position Awareness First**: Always identify "What Room Am I In?" using NEAR objects (< 1.0m, large in RGB)
3. **Arrival = Visual Analysis**: Is destination filling/occupying major portion of RGB view?
   - Filling view (close, prominent) → STOP immediately
   - Small/distant in view → continue navigation
   - Local map is auxiliary only, visual dominates
4. **Obstacle Analysis**: Analyze all 7 directions with [room]'s [object] format
5. **Real-time Tracking**: State current room → destination room → visual distance → arrived or not
6. **Distance Rules**: < 0.5m blocked, 1.0-2.0m safe, > 2.0m very clear"""


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
