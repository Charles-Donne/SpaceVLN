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

**Direction Selection Strategy**:
- Analyze all 12 views to determine which direction contains the waypoint/landmark
- Choose the angle where waypoint appears centered in view (or most visible)
- If waypoint is in Front view (IMAGE 1), no turn needed
- If waypoint is in other views, instruction must **first turn to face waypoint, then move**

**Action Origin**: All actions start from Front (IMAGE 1, 0°)

# Obstacle Distances (5 Directions)

Distance to nearest obstacles from current position (calculated from map):

- **FRONT (0°)**: {distance_front}
- **LEFT-30 (30°)**: {distance_left_30}
- **RIGHT-30 (-30°)**: {distance_right_30}
- **LEFT-90 (90°)**: {distance_left_90}
- **RIGHT-90 (-90°)**: {distance_right_90}

**Distance Rules**:
- ">2.0m open": Safe distance, no immediate obstacles
- "X.XXm": Specific distance < 2m, caution needed
- "<0.5m WARNING": Critical proximity, immediate turn/stop required

**Use distances for safe planning**: Choose turns/moves that avoid obstacles in travel direction

**IMAGE 13: Global Map** - Full explored area
**IMAGE 14: Local Map** - Nearby region (agent-centered, FOV cone shown)

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
2. **Determine waypoint direction**: Analyze which of the 12 views contains the next waypoint/landmark (choose angle where waypoint is most centered/possibleb
3. **Plan navigation instruction**: 
   - **If waypoint in Front**: Step by step to Next-Waypoint
   - **If waypoint in other Views**: **First turn to face Next-Waypoint direction, then step by step to Next-Waypoint**

# Actions Available

**Turn**: TURN_LEFT/RIGHT (30°, 60°, 90°, 120°, 150°, 180°)
**Move**: MOVE_FORWARD (0.25m, 0.5m, 0.75m, 1.0m, 1.25m, 1.5m)
**Arrive**: STOP (when <0.5m from destination)
- Use key actions (turn/move/stop) to navigate, but use fewer precise parameters (meters/degrees)

# Output Format (JSON only)

{{
    "waypoint": "<Current Area Type> - <Key Surrounding Landmarks and Relationships>",
    "waypoint_sequence": "<Current Location> → <Next Waypoint> → ... → <Final Waypoints>",
    "waypoint_direction": "<IMAGE number where next waypoint appears most centered/visible (1-12)>",
    "subtask_destination": "<Next immediate waypoint name>",
    "subtask_instruction": "<Step-by-step navigation instructions starting from Front view>",
    "subtask_landmark": "<Single landmark to detect (common, e.g. door, table, painting, cabinet)>",
    "completion_criteria": {{
        "Panoramic_Detection": "<Destination detected in which view>. <Other objects detected in which view>",
        "Spatial_relationship": "<Destination position and distance> (map verification). <Other objects relationships> (map verification). <Trajectory description>",
        "Location": "<Current Area Type> - <relative position descriptions>"
    }},
    "global_task_finish": <true if completing this subtask will finish the entire global task, false otherwise>,
    "reasoning": "<Brief explanation of observation and analysis leading to this subtask planning>"
}}

#Examples:

## Ex1: 
**Global Task**: Turn around walk through the exercise room into the living room. Wait by the Table.
**Current Observation:** IMAGE 1 (Front 0°): Bookshelf visible at distance. IMAGE 5 (Left 120°): Exercise room doorway visible with gym equipment inside. IMAGE 10 (Right 270°): Toilet and washbasin visible
**Obstacle Distances**: FRONT: >2.0m open | LEFT-30: 0.8m | RIGHT-30: >2.0m open | LEFT-90: 0.3m (<0.5m WARNING) | RIGHT-90: >2.0m open

{{
    "waypoint": "Restroom - beside exercise room doorway, toilet and washbasin nearby.",
    "waypoint_sequence": "Restroom(Current) → Exercise Room Entrance → Exercise Room → Living Room → Living Room's Table(Goal)",
    "waypoint_direction": "IMAGE 5 (Left 120°)",
    "subtask_destination": "exercise room entrance",
    "subtask_instruction": "Turn left 120° to face exercise room doorway, then move forward to enter the exercise room",
    "subtask_landmark": "exercise equipment",
    "completion_criteria": {{
        "Panoramic_Detection": "Exercise equipment detected in Front. Restroom fixtures detected in Back",
        "Spatial_relationship": "Exercise equipment ahead < 0.5m (map shows gym equipment at entrance position). Restroom far behind (map shows previous location). Orange trajectory shows left turn 120° and forward movement into gym entrance",
        "Location": "Exercise Room Entrance - exercise equipment ahead < 0.5m, restroom behind"
    }},
    "global_task_finish": false,
    "reasoning": "IMAGE 5 (Left 120°) shows exercise room doorway with gym equipment - next waypoint. Since NOT in IMAGE 1, turn left 120° first to align with IMAGE 5 direction, then move forward. Local map shows dark green circle (0.5m range) clear, LEFT-90 obstacle at 0.3m confirms wall nearby. Global map shows red arrow (current position) in small room, orange trajectory shows arrival path, exercise room (larger green area) is to the left. Dark red dashed line currently points away from exercise room, needs 120° left turn to align with doorway. Using 'exercise equipment' as landmark."
}}

## Ex2:
**Global Task**: Exit the room and turn left, head toward the kitchen and turn right. Go through the kitchen and out. Wait right at the bathroom.
**Current Observation:** IMAGE 1 (Front 0°): Open space ahead. IMAGE 2 (Left 30°): Bedroom exit doorway visible, corridor with pictures beyond. IMAGE 4 (Left 90°): Wall nearby
**Obstacle Distances**: FRONT: >2.0m open | LEFT-30: >2.0m open | RIGHT-30: 1.5m | LEFT-90: 0.6m | RIGHT-90: >2.0m open

{{
    "waypoint": "Bedroom - near exit",
    "waypoint_sequence": "Bedroom(Current) → Corridor → Kitchen Entrance → Kitchen → Kitchen Exit → Bathroom(Goal)",
    "waypoint_direction": "IMAGE 2 (Left 30°)",
    "subtask_destination": "corridor with pictures",
    "subtask_instruction": "Turn left 30° to face bedroom exit, then move forward through doorway to reach corridor",
    "subtask_landmark": "picture",
    "completion_criteria": {{
        "Directional_Detection": "Pictures detected on corridor wall in Front. Bedroom interior detected in Back",
        "Spatial_relationship": "Pictures on corridor wall < 0.5m (map shows decorative objects along corridor). Bedroom interior far behind (map shows previous area). Orange trajectory shows left turn 30° and forward 1.5m movement to corridor",
        "Location": "Corridor - pictures on wall < 1.0m, bedroom behind"
    }},
    "global_task_finish": false,
    "reasoning": "IMAGE 2 (Left 30°) shows bedroom exit with corridor and pictures visible - next waypoint. Since NOT in IMAGE 1, turn left 30° first to align with IMAGE 2 direction, then move forward 1.5m. Global map shows red arrow in bedroom (enclosed green area), corridor extends to the left with green floor area. Orange trajectory short (just started). Local map shows dark red dashed line pointing slightly left of doorway, 30° adjustment needed. Blue filled area (90° FOV) shows doorway at left edge. Distances confirm LEFT-30 >2m open, no obstacles blocking path. Using 'picture' as landmark."
}}

**Critical Requirements**:
- **12-Direction Analysis**: Analyze all 12 directional views (IMAGE 1-12) to locate current position and next waypoint
- **Turn-First Strategy**: If Next-Waypoint NOT in IMAGE 1 (Front 0°), turn to face it first; if in IMAGE 1, move forward directly
- **Sequential Navigation**: Treat waypoint_sequence as a chain to follow progressively. Identify current position → plan to next waypoint. Do NOT turn back to previous waypoints
- **Off-Path Recovery**: If deviated from sequence, identify current location and plan route to nearest upcoming waypoint, using turn-first strategy if needed
- **Forward Direction Alignment**: Dark red dashed line shows exact Forward direction - must align with destination/safe paths, NOT obstacles. Turn immediately if misaligned
- **Path Alignment**: Keep agent centered in corridors/paths, parallel to walls/boundaries with equal distance to both sides
- **Target Alignment**: Keep destination/landmark centered in Front view (0°), face it directly without angular deviation
- **Distance Judgment**: Use dark green circle on local map to determine if destination/landmark is nearby - objects within the circle are < 0.5m from current position
- **Landmark Selection**: Priority: landmarks from Global Task (e.g., "Wait by the Table" → "table"). Use specific objects (chair, table, bed, cabinet, sofa, painting). Avoid ambiguous terms (door, doorway, entrance, wall).
- **Logical Analysis**: Ensure reasoning and output aligns with inputs - All the content must not contain any contradictions.
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
12 directional views (30° FOV each, covering full 360°) + 2 top-down maps:

**IMAGE 1-12**: 12 independent directional views, each labeled with its angle
- IMAGE 1 = Front (0°) is the current forward direction

**Direction Selection Strategy**:
- Analyze all 12 views to determine which direction contains the Next Waypoint
- Choose the angle where Next Waypoint appears centered in view (or most possible)
- If waypoint is in IMAGE 1 (Front 0°), no turn needed
- If waypoint is in other images, instruction must **first turn to face waypoint, then move**

**Action Origin**: All actions start from IMAGE 1 (Front 0°)

# Obstacle Distances (5 Directions)

Distance to nearest obstacles from current position (calculated from map after 360° scan):

- **FRONT (0°)**: {distance_front}
- **LEFT-30 (30°)**: {distance_left_30}
- **RIGHT-30 (-30°)**: {distance_right_30}
- **LEFT-90 (90°)**: {distance_left_90}
- **RIGHT-90 (-90°)**: {distance_right_90}

**Distance Rules**:
- ">2.0m open": Safe distance, no immediate obstacles
- "X.XXm": Specific distance < 2m, caution needed
- "<0.5m WARNING": Critical proximity, immediate turn/stop required

**Use distances for safe planning**: Choose turns/moves that avoid obstacles in travel direction

**IMAGE 13: Global Map** - Full explored area (updated trajectory, waypoints, landmarks)
**IMAGE 14: Local Map** - Nearby region (agent-centered, FOV cone shown)

# Map Interpretation Guide

**Map Orientation**: 
- Top of map = Agent's current Front direction (Front is always up)

**Global Map**:
- **White**: Unexplored/unknown areas
- **Black**: Obstacles (walls, furniture, barriers) - **MUST AVOID in next planning**
- **Green**: Confirmed floor areas (safe to navigate)
- **Orange line**: Complete trajectory from navigation start to current position (all subtasks)
  - Shows your entire navigation history - **AVOID revisiting same areas unless necessary**
- **Red arrow**: Current position, arrow points to Front direction
- **Dark red dashed line**: Extends from red arrow upward, indicating exact Forward direction
  - **Should align with destination and safe paths**, **NOT face obstacles**
- **Purple markers with labels**: Previous Detected Landmark: {detected_landmarks}
- **Blue circles with white numbers**: Historical waypoints (see below)

# Spatial Memory (Waypoint History):
{waypoint_summary}
- Each numbered waypoint indicates a previously 360° scan and thinking location
  
**Local Map** (zoomed view around agent, same color legend as Global Map):
- Shows finer details in immediate vicinity for precise navigation
- **Red arrow**: Current position, arrow points to Front direction
- **Dark red dashed line**: Extends from red arrow upward, indicating exact Forward direction
- **Dark green circle**: 0.5m radius nearby area around current position
- **Orange line**: Current subtask trajectory (shorter than global map trajectory)
- **Blue filled area**: Current visible navigable area (90° FOV, blocked by obstacles)
- Better for planning nearby movements and obstacle avoidance

**Use Maps for Verification & Planning**:
- **Verify Previous Subtask**: Detect current position, previous landmarks({detected_landmarks})'s direction and distance, trajectory history to confirm if previous subtask completed.
- **Identify obstacles**: Black areas and space behind black areas(unexplored) - MUST AVOID in next planning.
- **Spatial awareness**: Use global map for overall layout, local map for immediate surroundings.

# Your Task

1. **Locate Current Position in Waypoint Sequence**: 
   - Analyze all 12 directional views + map trajectory + landmarks to identify current position
   - Determine which waypoint in sequence you are closest to or have reached
   - If off-path: Identify current location and nearest waypoint in sequence

2. **Determine Next Waypoint Direction**:
   - Analyze which of the 12 views (IMAGE 1-12) contains the Next Waypoint
   - Choose the angle where Next Waypoint is most Centered/Possible
   
3. **Plan Navigation to Next Waypoint**:
   - **On-path (before/at waypoint)**: Plan instruction to next waypoint in sequence
   - **Off-path (deviated)**: Plan instruction to return to nearest waypoint in sequence, then continue
   - Always move forward in waypoint_sequence chain, do NOT turn back to previous waypoints

# Actions Available

**Turn**: TURN_LEFT/RIGHT (30°, 60°, 90°, 120°, 150°, 180°)
**Move**: MOVE_FORWARD (0.25m, 0.5m, 0.75m, 1.0m, 1.25m, 1.5m)
**Arrive**: STOP (when <0.5m from destination)
- Use key actions (turn/move/stop) to navigate, but use fewer precise parameters (meters/degrees)

# Output Format (JSON only)

{{
    "waypoint": "<Current Area Type> - <Key Surrounding Landmarks and Relationships>",
    "waypoint_sequence": "<Completed Waypoints(✓)> → <Current Position> → <Next Waypoint> → <Remaining Waypoints> → <Goal>",
    "waypoint_direction": "<IMAGE number where next waypoint appears most centered/visible (1-12)>",
    "subtask_destination": "<Next waypoint in sequence to navigate toward>",
    "subtask_instruction": "<Step-by-step navigation instructions from current position to next waypoint>",
    "subtask_landmark": "<Single landmark name at next waypoint for detection>",
    "completion_criteria": {{
        "Panoramic_Detection": "<Next waypoint detected in which view>. <Other objects detected in which view>",
        "Spatial_relationship": "<Next waypoint position and distance> (map verification). <Other objects relationships> (map verification). <Trajectory description>",
        "Location": "<Next Waypoint Area> - <relative position descriptions>"
    }},
    "global_task_finish": <true if completing this subtask will finish the entire global task, false otherwise>,
    "reasoning": "<Brief explanation of completion verification, progress, and next plan>"
}}

## Example 1:
**Global Task**: Turn around walk through the exercise room into the living room. Wait by the Table.
**Previous Subtask**: Navigate to exercise room entrance
**Current Observation:** IMAGE 1 (Front 0°): Exercise equipment directly ahead. IMAGE 7 (Back 180°): Restroom visible behind

{{
    "waypoint": "Exercise Room Entrance - facing gym equipment ahead, restroom behind",
    "waypoint_sequence": "Restroom(✓) → Exercise Room Entrance(Current) → Exercise Room(Next) → Living Room Arched Doorway → Living Room's Table(Goal)",
    "waypoint_direction": "IMAGE 1 (Front 0°)",
    "subtask_destination": "exercise room interior",
    "subtask_instruction": "Move forward into the interior of the gym",
    "subtask_landmark": "exercise equipment",
    "completion_criteria": {{
        "Directional_Detection": "Exercise equipment detected surrounding agent in multiple IMAGEs (1, 2, 12). Restroom detected in IMAGE 7 (Back 180°)",
        "Spatial_relationship": "Exercise equipment surrounding agent < 1.0m in multiple directions (map shows inside gym area). Restroom far behind (map shows previous waypoint far back). Orange trajectory shows entered gym room interior",
        "Location": "Exercise Room Interior - exercise equipment surrounding agent, restroom far behind"
    }},
    "global_task_finish": false,
    "reasoning": "Current at Exercise Room Entrance, gym equipment ahead in IMAGE 1, restroom behind in IMAGE 7. Previous subtask (reach entrance) completed. Global map shows red arrow at entrance threshold between small room (restroom) and larger room (gym), orange trajectory shows 120° turn and forward movement from restroom. Local map shows dark red dashed line aligned with gym interior, dark green circle overlaps gym entrance, blue filled area shows gym equipment visible ahead. Next waypoint is gym interior - in IMAGE 1, no turn needed. Move forward directly into gym."
}}

## Example 2:
**Global Task**: Turn around and navigate to refrigerator in kitchen
**Previous Subtask**: Navigate through kitchen center
**Current Observation:** IMAGE 1 (Front 0°): Refrigerator directly ahead < 0.5m. IMAGE 10 (Right 270°): Counter visible. IMAGE 7 (Back 180°): Kitchen island visible

{{
    "waypoint": "Kitchen Center - refrigerator ahead < 0.5m, counter to right, kitchen island behind",
    "waypoint_sequence": "Bedroom(✓) → Hallway(✓) → Kitchen Center(✓) → Refrigerator(Current + Goal)",
    "waypoint_direction": "IMAGE 1 (Front 0°)",
    "subtask_destination": "refrigerator in kitchen",
    "subtask_instruction": "Stop. The refrigerator is directly ahead within 0.5m",
    "subtask_landmark": "refrigerator",
    "completion_criteria": {{
        "Directional_Detection": "Refrigerator detected in Front centered ahead. Counter detected in Right. Kitchen island detected in Back",
        "Spatial_relationship": "Refrigerator ahead < 0.5m (map shows refrigerator landmark within dark green circle). Counter at right. Kitchen island behind. Orange trajectory shows direct forward movement through kitchen to refrigerator",
        "Location": "Refrigerator Area - refrigerator ahead < 0.5m, counter at right, kitchen island behind"
    }},
    "global_task_finish": true,
    "reasoning": "Current at Kitchen Center, refrigerator < 0.5m ahead filling IMAGE 1. Previous subtask (kitchen center) completed. Global map shows red arrow in kitchen (large green area), orange trajectory extends from bedroom through hallway to kitchen center, purple marker likely shows refrigerator landmark ahead. Local map shows refrigerator within dark green circle (< 0.5m), dark red dashed line points directly at refrigerator, blue filled area shows refrigerator dominates front view. Refrigerator is final destination in IMAGE 1. Global task complete. Execute STOP."
}}

## Example 3:
**Global Task**: Walk toward the oven. Go through the archway on your right that is past the painting of the girl in a blue bonnet. Go through the doorway on your left. Stop in front of the small sink, before you reach the grill. 
**Previous Subtask**: Approach oven area
**Current Observation:** IMAGE 1 (Front 0°): Oven visible ahead but distance > 1.0m. IMAGE 7 (Back 180°): Kitchen island visible behind

{{
    "waypoint": "Kitchen - Oven ahead > 1.0m, kitchen island behind",
    "waypoint_sequence": "Starting Point(✓) → Kitchen(Current) → Oven Area(Next) → Archway Past Painting → Left Doorway → Small Sink(Goal)",
    "waypoint_direction": "IMAGE 1 (Front 0°)",
    "subtask_destination": "oven area",
    "subtask_instruction": "Continue moving forward to approach oven until oven is directly ahead < 0.5m (target: oven centered in Front view, very close)",
    "subtask_landmark": "oven",
    "completion_criteria": {{
        "Directional_Detection": "Oven detected in Front centered ahead occupying large portion. Kitchen island detected far away in Back",
        "Spatial_relationship": "Oven ahead < 0.5m (map shows oven landmark within dark green circle). Kitchen island far behind. Orange trajectory shows forward movement toward oven",
        "Location": "Oven Area - oven ahead < 0.5m, kitchen island far behind"
    }},
    "global_task_finish": false,
    "reasoning": "Current approaching oven - IMAGE 1 shows oven visible, IMAGE 7 shows kitchen island behind. Target is oven area. Position: BEFORE target - oven detected but > 1.0m. Global map shows red arrow in kitchen, orange trajectory shows forward progress toward oven area. Local map shows oven outside dark green circle (> 0.5m away), dark red dashed line aligned with oven direction, blue filled area shows oven visible but not yet close. No black obstacles blocking path ahead. Completion requires oven < 0.5m (inside dark green circle). Oven in IMAGE 1, no turn needed. Continue forward 0.75m until oven enters dark green circle."
}}

**Critical Requirements**:
- **12-Direction Analysis**: Analyze all 12 directional views (IMAGE 1-12) to locate current position and next waypoint
- **Turn-First Strategy**: If Next-Waypoint NOT in IMAGE 1 (Front 0°), turn to face it first; if in IMAGE 1, move forward directly
- **Sequential Navigation**: Treat waypoint_sequence as a chain to follow progressively. Identify current position → plan to next waypoint. Do NOT turn back to previous waypoints
- **Off-Path Recovery**: If deviated from sequence, identify current location and plan route to nearest upcoming waypoint, using turn-first strategy if needed
- **Forward Direction Alignment**: Dark red dashed line shows exact Forward direction - must align with destination/safe paths, NOT obstacles. Turn immediately if misaligned
- **Path Alignment**: Keep agent centered in corridors/paths, parallel to walls/boundaries with equal distance to both sides
- **Target Alignment**: Keep destination/landmark centered in Front view (0°), face it directly without angular deviation
- **Distance Judgment**: Use dark green circle on local map to determine if destination/landmark is nearby - objects within the circle are < 0.5m from current position
- **Planning**: Start all actions from Front view (0°). If subtask completed, plan NEXT waypoint; if not, adjust CURRENT subtask
- **Map**: Use maps to verify trajectory, identify obstacles and plan safe paths for next subtask
- **Landmark Selection**: Priority: landmarks from Global Task (e.g., "Wait by the Table" → "table"). Use specific objects (chair, table, bed, cabinet, sofa, painting). Avoid ambiguous terms (door, doorway, entrance, wall).
- **Logical Analysis**: Ensure reasoning and output aligns with inputs - All the content must not contain any contradictions.
- **Explore Unseen Areas**: If the destination is invisible, explore more places but avoiding areas with too many history waypoints, and understand the spatial relationships.
"""


def get_initial_planning_prompt(instruction: str, 
                               action_space: str,
                               distance_front: str = "Unknown",
                               distance_left_30: str = "Unknown",
                               distance_right_30: str = "Unknown",
                               distance_left_90: str = "Unknown",
                               distance_right_90: str = "Unknown") -> str:
    """
    获取初始规划提示词
    
    Args:
        instruction: 完整导航指令
        action_space: 动作空间描述
        distance_front: 前方障碍物距离
        distance_left_30: 左前30°障碍物距离
        distance_right_30: 右前30°障碍物距离
        distance_left_90: 左侧90°障碍物距离
        distance_right_90: 右侧90°障碍物距离
        
    Returns:
        格式化的提示词字符串
    """
    return INITIAL_PLANNING_PROMPT.format(
        instruction=instruction,
        action_space=action_space,
        distance_front=distance_front,
        distance_left_30=distance_left_30,
        distance_right_30=distance_right_30,
        distance_left_90=distance_left_90,
        distance_right_90=distance_right_90
    )

def get_verification_replanning_prompt(instruction: str,
                                       waypoint_sequence: str,
                                       subtask_destination: str,
                                       subtask_instruction: str,
                                       completion_criteria: str,
                                       action_space: str,
                                       detected_landmarks: str = None,
                                       waypoint_summary: str = None,
                                       distance_front: str = "Unknown",
                                       distance_left_30: str = "Unknown",
                                       distance_right_30: str = "Unknown",
                                       distance_left_90: str = "Unknown",
                                       distance_right_90: str = "Unknown") -> str:
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
        distance_front: 前方障碍物距离
        distance_left_30: 左前30°障碍物距离
        distance_right_30: 右前30°障碍物距离
        distance_left_90: 左侧90°障碍物距离
        distance_right_90: 右侧90°障碍物距离
        
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
        waypoint_summary=waypoint_summary,
        distance_front=distance_front,
        distance_left_30=distance_left_30,
        distance_right_30=distance_right_30,
        distance_left_90=distance_left_90,
        distance_right_90=distance_right_90
    )