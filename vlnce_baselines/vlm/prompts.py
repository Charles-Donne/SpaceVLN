"""
VLM规划提示词模板
================
用于LLM高层规划的提示词模板
"""

# 初始规划提示词 - 在任务开始时生成第一个子任务
INITIAL_PLANNING_PROMPT = """You are a Vision-Language Navigation planning module. Analyze the environment and Global Task to design the next navigation subtask.

# Navigation Global Task:
{instruction}

# Visual Observations
12 directional views (30° FOV each, covering full 360°) + 2 top-down maps:

**IMAGE 1-12**: 12 independent directional views, each labeled with its angle (0°, 30°, 60°, ..., 330°)
- IMAGE 1 = Front (0°) is the current forward direction
- Angles increase counterclockwise: 30°, 60°, 90° (Left), 180° (Back), 270° (Right), etc.
- **Each view shows obstacle distance** (e.g., "1.5m", "0.3m"): Distance to nearest obstacle in that direction

**Direction Selection Strategy**:
- Analyze all 12 views to determine which direction contains the waypoint/landmark
- **Avoid obstacle directions**: If distance < 0.5m, path is blocked - choose another direction
- Choose direction where: 1) Waypoint visible, 2) Obstacle distance > 0.5m (safe to navigate)
- **System will automatically rotate to face the waypoint_direction you specify**
- After rotation, waypoint will be in Front view (IMAGE 1) for step-by-step navigation

**Action Origin**: All actions start from Front (IMAGE 1, 0°) **after automatic rotation**

**Global Map** - Full explored area
**Local Map** - Nearby region (agent-centered, FOV cone shown)

# Map Interpretation Guide

**Map Orientation**: 
- Top of map = Agent's current Front direction (Front is always up)

**Global Map**:
- **White**: Unexplored/unknown areas
- **Black**: Obstacles (walls, furniture, barriers) - **MUST AVOID in planning**
- **Green**: Confirmed floor areas (safe to navigate)
- **Orange line**: Complete trajectory from navigation start to current position (all subtasks)
  - Shows your entire navigation history - **AVOID revisiting same areas unless necessary**
- **Red arrow**: Current position, arrow points to Front direction
- **Dark red dashed line**: Extends upward from red arrow, indicating exact Forward direction - should align with destination/safe paths
  
**Local Map** (zoomed view around agent, same color legend as Global Map):
- Shows finer details in immediate vicinity for precise navigation
- **Red arrow**: Current position, arrow points to Front direction
- **Dark red dashed line**: Extends upward from red arrow, indicating exact Forward direction
- **Dark green circle**: 0.5m radius nearby area around current position
- **Orange line**: Current subtask trajectory (shorter than global map trajectory)
- **Blue filled area**: Current visible navigable area (90° FOV, blocked by obstacles)
- Better for planning nearby movements and obstacle avoidance

**Use Maps for Planning**:
- **Identify obstacles**: Black areas and space behind black areas(unexplored).
- **Spatial awareness**: Use global map for overall layout, local map for immediate surroundings

# Your Task

1. **Analyze environment**: Use 12 directional views + global and local map to identify your position, related landmarks and obstacles
2. **Determine waypoint direction**: Analyze which of the 12 views contains the next waypoint/landmark (choose angle where waypoint is most centered/visible)
3. **Plan navigation instruction**: 
   - **System will auto-rotate** to face your specified waypoint_direction
   - After rotation, waypoint will be in Front view (IMAGE 1)
   - Write step-by-step instructions assuming agent is already **facing the waypoint**

# Actions Available

**Turn**: TURN_LEFT/RIGHT (30°, 60°, 90°, 120°, 150°, 180°)
**Move**: MOVE_FORWARD (0.25m, 0.5m, 0.75m, 1.0m, 1.25m, 1.5m)
**Arrive**: STOP (when <0.5m from destination)
- Use key actions (turn/move/stop) to navigate, but use fewer precise parameters (meters/degrees)

# Output Format (JSON only)

**CRITICAL**: Output ONLY valid JSON. No extra text before or after.

{{
    "current_waypoint": "<Current Area Type> - <Key Surrounding Landmarks and Relationships>",
    "task_progress": "<Global task with completed parts marked with ✓. E.g.: 'Turn around(✓) walk through exercise room into living room. Wait by Table.'>",
    "waypoint_sequence": "<Current Location> → <Next Waypoint> → ... → <Final Waypoints>",
    "next_waypoint_direction": "<IMAGE number where next waypoint appears most centered/visible (1-12)>",
    "next_waypoint_destination": "<Next immediate waypoint name>",
    "subtask_instruction": "<Step-by-step navigation instructions starting from Front view>",
    "next_waypoint_landmark": "<Single landmark to detect (common, e.g. door, table, painting, cabinet)>",
    "completion_criteria": "<Detection: what detected + distance | Location: position/area | Map: space + trajectory>",
    "global_task_finish": <true ONLY when final destination of global task is visible in current views (any IMAGE 1-12) and close enough to reach - Global task complete, stop navigating immediately. false otherwise>,
    "reasoning": "<Max 200 words: Brief explanation of observation and analysis leading to this subtask planning>"
}}

#Examples:

## Ex1: 
**Global Task**: Turn around walk through the exercise room into the living room. Wait by the Table.
**Current Observation:** IMAGE 1 (Front 0°): Bookshelf visible at distance. IMAGE 5 (Left 120°): Exercise room doorway visible with gym equipment inside. IMAGE 10 (Right 270°): Toilet and washbasin visible

{{
    "current_waypoint": "Restroom - beside exercise room doorway, toilet and washbasin nearby.",
    "task_progress": "Turn around(✓) walk through the exercise room into the living room. Wait by the Table.",
    "waypoint_sequence": "Restroom(Current) → Exercise Room → Living Room → Living Room's Table(Goal)",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "exercise room",
    "subtask_instruction": "Move forward through doorway to enter the exercise room",
    "next_waypoint_landmark": "exercise equipment",
    "completion_criteria": "Detection: Exercise equipment surrounding < 1.0m | Location: Exercise Room interior | Map: Inside gym area (green space), trajectory entered from restroom",
    "global_task_finish": false,
    "reasoning": "IMAGE 5 (Left 120°) shows exercise room doorway with gym equipment - next waypoint. System will auto-rotate 120° left to face this direction. After rotation, move forward through doorway into exercise room. Local map shows dark green circle (0.5m range) clear. Global map shows red arrow (current position) in small room, orange trajectory shows arrival path, exercise room (larger green area) is to the left. Using 'exercise equipment' as landmark."
}}

## Ex2:
**Global Task**: Exit the room and turn left, head toward the kitchen and turn right. Go through the kitchen and out. Wait right at the bathroom.
**Current Observation:** IMAGE 1 (Front 0°): Open space ahead. IMAGE 2 (Left 30°): Bedroom exit doorway visible, corridor with pictures beyond. IMAGE 4 (Left 90°): Wall nearby

{{
    "current_waypoint": "Bedroom - near exit",
    "task_progress": "Exit the room and turn left, head toward the kitchen and turn right. Go through the kitchen and out. Wait right at the bathroom.",
    "waypoint_sequence": "Bedroom(Current) → Corridor → Kitchen Entrance → Kitchen → Kitchen Exit → Bathroom(Goal)",
    "next_waypoint_direction": "IMAGE 2 (Left 30°)",
    "next_waypoint_destination": "corridor with pictures",
    "subtask_instruction": "Move forward through doorway to reach corridor",
    "next_waypoint_landmark": "picture",
    "completion_criteria": "Detection: Pictures on corridor wall < 0.5m | Location: Corridor | Map: Corridor space along bedroom exit, trajectory moved forward 1.5m",
    "global_task_finish": false,
    "reasoning": "IMAGE 2 (Left 30°) shows bedroom exit with corridor and pictures visible - next waypoint. System will auto-rotate 30° left to face this direction. After rotation, move forward 1.5m through doorway. Global map shows red arrow in bedroom (enclosed green area), corridor extends to the left with green floor area. Orange trajectory short (just started). Local map shows doorway ahead. Blue filled area (90° FOV) will show doorway centered after rotation. Using 'picture' as landmark."
}}

**Critical Requirements**:
- **12-Direction Analysis**: Analyze all 12 views to locate position and next waypoint
- **Auto-Rotation**: System rotates to waypoint_direction. Write instructions from Front view after rotation.
- **Sequential Navigation**: Follow waypoint_sequence progressively. Don't return to previous waypoints.
- **Path Safety**: Avoid obstacles (black areas). Keep centered in paths. Use maps for safe routes.
- **Distance Judgment**: Dark green circle on local map = 0.5m radius (objects inside < 0.5m away)
- **Landmark Priority**: Use objects from Global Task. Prefer specific items (chair, table, bed). Avoid ambiguous terms (door, entrance).
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """You are a Vision-Language Navigation verification and planning module. Verify previous subtask completion and plan the next navigation step.

# Navigation Global Task:
{instruction}

# Previous Subtask Context:
**Previous Waypoint Sequence**: {waypoint_sequence}
**Previous Subtask Destination**: {subtask_destination}
**Previous Subtask Instruction**: {subtask_instruction}
**Previous Subtask Completion Criteria**: {completion_criteria}

# Visual Observations
12 directional views (30° FOV, full 360°) + 2 maps:

**IMAGE 1-12**: Independent views labeled with angles. IMAGE 1 = Front (0°)
- **Obstacle distances**: Each view shows distance to nearest obstacle (e.g., "1.5m", "0.3m")
- **Waypoint markers**: White circles (ID) + boxes (room type) show visited locations
- Example: Circle "3" + "Kitchen" box in IMAGE 7 = Kitchen waypoint behind

**Direction Strategy**: 
- Observe all views for next waypoint location and obstacle distances
- **Avoid blocked directions**: Distance < 0.5m = blocked, choose clearer path (> 1.0m)
- Move toward NEXT waypoint (not previous ones with markers) via safe direction
- System auto-rotates to your chosen direction

**Maps**:
- **Global Map**: Full area (trajectory, waypoints, landmarks) - Top = Front
- **Local Map**: Nearby region (agent-centered, FOV cone, dark green = 0.5m radius)

# Map Guide

**Colors**: White=unexplored, Black=obstacles (AVOID), Green=safe floor, Orange line=trajectory, Red arrow=current position/direction, Purple=landmarks, Blue circles=waypoints

**Global Map**: 
- Red arrow points to Front; dashed line extends forward (should align with safe paths, NOT obstacles)
- Orange line = full navigation history (avoid revisiting)
- Blue circles with numbers = waypoint history

**Local Map**: 
- Dark green circle = 0.5m radius (objects inside are < 0.5m away)
- Orange line = current subtask trajectory
- Blue filled = visible 90° FOV area

**Use Maps**: Verify position via trajectory/landmarks. Identify obstacles (black). Global for layout, local for immediate surroundings.

# Spatial Memory (Waypoint History):
{waypoint_summary}
- Numbered waypoints = previous 360° scan locations

# Your Task

**Step 1: Localize Current Position**
- Analyze 12 views: What's visible in each? Where are waypoint markers (white circles)?
- Analyze Global Map: Where's red arrow? Which waypoint am I closest to?
- Synthesize: Where am I in environment? Which waypoint in sequence?

**Step 2: Determine Next Direction**
- Identify next waypoint from sequence
- Find which IMAGE (1-12) shows it most clearly
- Verify on map: Does this direction lead there?
- Avoid waypoint markers (visited areas) and orange trajectory path

**Step 3: Plan Navigation**
- Choose next_waypoint_direction (IMAGE 1-12)
- System auto-rotates to face it → next waypoint becomes Front (IMAGE 1)
- Write instructions assuming already facing waypoint: "Move forward..."

# Actions
TURN_LEFT/RIGHT (30-180°), MOVE_FORWARD (0.25-1.5m), STOP (<0.5m from goal)

# Output Format (JSON only)

**CRITICAL**: Output ONLY valid JSON. No extra text before or after.
**Word Limits**: reasoning MAX 200 words, subtask_instruction MAX 100 words, others 20-50 words

{{
    "current_waypoint": "<Current Area Type> - <Key Surrounding Landmarks and Relationships>",
    "task_progress": "<Global task with completed parts marked with ✓. E.g.: 'Turn around(✓) walk through exercise room(✓) into living room. Wait by Table.'>",
    "waypoint_sequence": "<Completed Waypoints(✓)> → <Current Position> → <Next Waypoint> → <Remaining Waypoints> → <Goal>",
    "next_waypoint_direction": "<IMAGE number where next waypoint appears most centered/visible (1-12)>",
    "next_waypoint_destination": "<Next waypoint in sequence to navigate toward>",
    "subtask_instruction": "<Step-by-step navigation instructions from current position to next waypoint>",
    "next_waypoint_landmark": "<Single landmark name at next waypoint for detection>",
    "completion_criteria": "<Detection: what detected + distance | Location: position/area | Map: space + trajectory>",
    "global_task_finish": <true ONLY when final destination of global task is visible in current 12 views and close (< 1.0m) - You have arrived, stop immediately. false otherwise>,
    "reasoning": "<MAX 200 words: 1) Where am I? (views + map + waypoints) 2) What's around? (front/back/left/right) 3) Task progress: Which parts of global task completed? 4) Which waypoint? Where's last one? 5) Next direction? (IMAGE # + map direction) 6) Plan>"
}}

## Example 1:
**Global Task**: Turn around walk through the exercise room into the living room. Wait by the Table.
**Previous Subtask**: Navigate to exercise room
**Current Observation:** IMAGE 1 (Front 0°): Exercise equipment surrounding. IMAGE 7 (Back 180°): Restroom visible behind

{{
    "current_waypoint": "Exercise Room (entrance area) - gym equipment surrounding, restroom behind",
    "task_progress": "Turn around(✓) walk through the exercise room(✓) into the living room. Wait by the Table.",
    "waypoint_sequence": "Restroom(✓) → Exercise Room(Current) → Living Room → Living Room's Table(Goal)",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "living room",
    "subtask_instruction": "Move forward through the exercise room to reach the living room exit",
    "next_waypoint_landmark": "arched doorway",
    "completion_criteria": "Detection: Arched doorway in Front view | Location: Living Room entrance | Map: Doorway space leading to living room, trajectory moved through exercise room",
    "global_task_finish": false,
    "reasoning": "Spatial localization: 12 views show gym equipment in Front/Sides (IMAGE 1-6, 8-12), restroom visible in Back (IMAGE 7). Global map shows red arrow at Exercise Room entrance, blue circle #1 (Restroom) behind me. Waypoint history confirms last waypoint = Restroom. Task progress: Turned around(✓), walked through exercise room(✓), now need to enter living room and reach table. Environment awareness: Gym equipment surrounding (front/sides), restroom behind (completed). Waypoint progress: Currently at Exercise Room entrance (Waypoint #2), last waypoint (Restroom) is directly behind on map. Next direction: IMAGE 1 (Front 0°) shows path forward through gym area toward living room exit. Map shows living room ahead of current position. Plan: Already facing Front, move forward through exercise room to reach living room."
}}

## Example 2:
**Global Task**: Exit the bedroom and turn left. Walk straight passing the gray couch and stop near the rug.
**Previous Subtask**: Navigate past gray couch toward rug
**Current Observation:** IMAGE 1 (Front 0°): Rug directly ahead < 0.5m, gray couch visible beside rug. IMAGE 10 (Right 270°): Gray couch right beside. IMAGE 7 (Back 180°): Hallway visible behind

{{
    "current_waypoint": "Rug area - standing near rug, gray couch right beside",
    "task_progress": "Exit the bedroom(✓) and turn left(✓). Walk straight passing the gray couch(✓) and stop near the rug(✓).",
    "waypoint_sequence": "Bedroom(✓) → Hallway(✓) → Living Room(✓) → Rug(Current = Goal)",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "rug",
    "subtask_instruction": "Stop. Already near the rug within 0.5m",
    "next_waypoint_landmark": "rug",
    "completion_criteria": "Detection: Rug in Front < 0.5m, gray couch beside | Location: Rug area - goal position | Map: At rug landmark, trajectory ends here",
    "global_task_finish": true,
    "reasoning": "Spatial localization: 12 views show rug in Front (IMAGE 1) < 0.5m with gray couch visible right beside rug - final destination reached. Gray couch at Right (IMAGE 10), hallway in Back (IMAGE 7). Global map confirms red arrow at rug position. Task progress: Exited bedroom(✓), turned left(✓), walked straight passing gray couch(✓), stopped near rug(✓) - all parts completed. Global task finish condition met: Final destination (rug) is visible in current views (IMAGE 1) and very close (< 0.5m), gray couch visible beside rug confirming correct location. Task complete - stop immediately."
}}

## Example 3:
**Global Task**: Walk to the kitchen through the hallway, then enter the bedroom on your left.
**Previous Subtask**: Navigate through hallway
**Current Observation:** IMAGE 1 (Front 0°): Hallway continues ahead 3.0m. IMAGE 5 (Left 120°): Bedroom doorway visible at distance (~2.5m), bed partially visible inside. IMAGE 7 (Back 180°): Kitchen visible behind

{{
    "current_waypoint": "Hallway - bedroom doorway at left, kitchen behind",
    "task_progress": "Walk to the kitchen(✓) through the hallway(✓), then enter the bedroom on your left.",
    "waypoint_sequence": "Kitchen(✓) → Hallway(Current) → Bedroom(Goal)",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "bedroom",
    "subtask_instruction": "Move forward through the doorway to enter the bedroom",
    "next_waypoint_landmark": "bed",
    "completion_criteria": "Detection: Bed in Front < 1.0m | Location: Bedroom interior | Map: Inside bedroom space, trajectory entered from hallway",
    "global_task_finish": false,
    "reasoning": "Spatial localization: 12 views show bedroom doorway with bed visible at Left (IMAGE 5 at 120°) - final destination visible but not yet reached. Hallway ahead (IMAGE 1), kitchen in Back (IMAGE 7). Task progress: Walked to kitchen(✓), through hallway(✓), now need to enter bedroom. Global task finish condition NOT yet met: Although final destination (bedroom) is visible in current views (IMAGE 5), I'm still in hallway outside the bedroom. Need to enter bedroom interior to complete task. Plan: Rotate to IMAGE 5, move forward through doorway into bedroom, then stop."
}}

**Critical Requirements**:
- **Spatial Awareness**: Analyze 12 views + Global Map + waypoint history to determine current position. Show reasoning explicitly.
- **Obstacle Avoidance**: Check distance labels on 12 views. Avoid directions with distance < 0.5m (blocked). Choose next_waypoint_direction with clear path (> 1.0m).
- **Task Progress Tracking**: Mark completed parts of global instruction with ✓ to maintain awareness of overall task completion.
- **Waypoint Markers**: White circles + boxes show visited areas - avoid backtracking
- **Auto-Rotation**: System rotates to next_waypoint_direction. Write instructions from Front view.
- **Sequential Navigation**: Follow waypoint_sequence progressively. Don't return to previous waypoints (marked circles).
- **Global Task Completion**: global_task_finish=true when final destination visible in 12 views and close (< 1.0m). Once you see the goal in current views and it's nearby, task is complete - STOP immediately, don't continue navigating.
- **Path Safety**: Avoid black areas (obstacles). Keep centered in paths. Use maps to verify trajectory and plan safe routes.
- **Landmark Priority**: Use objects from Global Task. Prefer specific items (chair, table, bed). Avoid ambiguous terms (door, entrance).
"""


def get_initial_planning_prompt(instruction: str, 
                               action_space: str) -> str:
    """
    获取初始规划提示词
    
    Args:
        instruction: 完整导航指令
        action_space: 动作空间描述
        
    Returns:
        格式化的提示词字符串
    """
    return INITIAL_PLANNING_PROMPT.format(
        instruction=instruction,
        action_space=action_space
    )

def get_verification_replanning_prompt(instruction: str,
                                       waypoint_sequence: str,
                                       subtask_destination: str,
                                       subtask_instruction: str,
                                       completion_criteria: str,
                                       action_space: str,
                                       detected_landmarks: str = None,
                                       waypoint_summary: str = None) -> str:
    """
    获取验证和重规划提示词
    
    Args:
        instruction: 完整导航指令
        waypoint_sequence: 当前路径点序列
        subtask_destination: 当前子任务目的地
        subtask_instruction: 当前子任务指令
        completion_criteria: 完成条件
        action_space: 动作空间描述
        detected_landmarks: 已检测到的landmark类别字符串
        waypoint_summary: 路径点历史记录字符串
        
    Returns:
        格式化的提示词字符串
    """
    if not detected_landmarks:
        detected_landmarks = "No landmarks detected yet"
    if not waypoint_summary:
        waypoint_summary = "No waypoints recorded yet"
    
    return VERIFICATION_REPLANNING_PROMPT.format(
        instruction=instruction,
        waypoint_sequence=waypoint_sequence,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        completion_criteria=completion_criteria,
        action_space=action_space,
        detected_landmarks=detected_landmarks,
        waypoint_summary=waypoint_summary
    )