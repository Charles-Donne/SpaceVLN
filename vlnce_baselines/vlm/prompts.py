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
4 panoramic views (90° FOV each) + 2 top-down maps:

**Direction Usage**: e.g., "To Right-View's left-portion: turn right 60°".
- Determine the location and orientation of subtask-destination using orientation indicated on panoramic view (left, right or center portion).

**Action Origin**: All actions start from Front (IMAGE 1 center)

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

**IMAGE 5: Global Map** - Full explored area
**IMAGE 6: Local Map** - Nearby region (agent-centered, FOV cone shown)

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

1. **Analyze environment**: Use 4-directional views + global and local map to identify your position, related landmarks and obstacles
2. **Plan subtask**: Break down global task into achievable intermediate waypoints
3. **Provide instructions**: Action instructionn starting from Front view using concrete landmarks

# Actions Available

**Turn**: TURN_LEFT/RIGHT (30°, 60°, 90°, 120°, 150°, 180°)
**Move**: MOVE_FORWARD (0.25m, 0.5m, 0.75m, 1.0m, 1.25m, 1.5m)
**Arrive**: STOP (when <0.5m from destination)
- Use key actions (turn/move/stop) to navigate, but use fewer precise parameters (meters/degrees)

# Output Format (JSON only)

{{
    "waypoint": "<Current Area Type> - <Key Surrounding Landmarks and Relationships>",
    "waypoint_sequence": "<Current Location> → <Next Waypoint> → ... → <Final Waypoints>",
    "subtask_destination": "<Next immediate waypoint name>",
    "subtask_instruction": "<Step-by-step navigation instructions starting from Front view>",
    "subtask_landmark": "<Single landmark to detect (common, e.g. door, table, painting, cabinet)>",
    "completion_criteria": {{
        "Panoramic_Detection": "<Destination detected in which view>. <Other objects detected in which view>",(from panoramic view detection)
        "Spatial_relationship": "<Destination position and distance> (map verification). <Other objects relationships> (map verification). <Trajectory description>",
        "Location": "<Current Area Type> - <relative position descriptions>"
    }},
    "global_task_finish": <true if completing this subtask will finish the entire global task, false otherwise>,
    "reasoning": "<Brief explanation of observation and analysis leading to this subtask planning>"
}}

#Examples:

## Ex1: 
**Global Task**: Turn around walk through the exercise room into the living room. Wait by the Table.
**Current Observation**: Far front is a bookshelf. Toilet and Sink can be seen from right view. Left is a wall but left 120° is doorway to gym.
**Obstacle Distances**: FRONT: >2.0m open | LEFT-30: 0.8m | RIGHT-30: >2.0m open | LEFT-90: 0.3m (<0.5m WARNING) | RIGHT-90: >2.0m open
{{
    "waypoint": "Restroom - beside exercise room doorway, toilet and washbasin nearby.",
    "waypoint_sequence": "Restroom(Current) → Exercise Room Entrance → Exercise Room → Living Room → Living Room's Table(Goal)",
    "subtask_destination": "exercise room entrance",
    "subtask_instruction": "Turn left 120° to face doorway opening, then move forward to enter the gym area.",
    "subtask_landmark": "treadmill",
    "completion_criteria": {{
        "Panoramic_Detection": "Treadmill detected ahead in Front view. Restroom fixtures (toilet, sink) detected in Back view.",
        "Spatial_relationship": "Treadmill ahead < 1.0m (map shows exercise equipment near current position). Restroom far behind (map shows away from last waypoint). Orange trajectory shows left turn and approach into gym area",
        "Location": "Exercise Room Entrance - treadmill ahead < 1.0m, restroom behind"
    }},
    "global_task_finish": false,
    "reasoning": "Agent currently in Restroom (toilet and washbasin visible from right view, bookshelf at far front). Exercise room doorway visible at left 120° (left portion of Left-View). Distances: LEFT-90 has wall WARNING (0.3m), must turn more than 90° to clear obstacle. Map: Left 90° is wall obstacle (black), green floor path clear after turning left 120° leading to doorway opening, no black obstacles blocking approach. Global task requires: 1) pass through exercise room (use treadmill as landmark), 2) enter living room, 3) reach table (final goal landmark). First waypoint is entering exercise room, using treadmill as specific, unambiguous landmark (better than generic 'door')."
}}

## Ex2:
**Global Task**: Exit the room and turn left, head toward the kitchen and turn right. Go through the kitchen and out. Wait right at the bathroom.
**Current Observation**: Bedroom exit visible at left 30°. Walls on left side. Corridor with pictures visible beyond the exit at left 30°.
**Obstacle Distances**: FRONT: >2.0m open | LEFT-30: >2.0m open | RIGHT-30: 1.5m | LEFT-90: 0.6m | RIGHT-90: >2.0m open
{{
    "waypoint": "Bedroom - near exit.",
    "waypoint_sequence": "Bedroom(Current) → Corridor → Kitchen Entrance → Kitchen → Kitchen Exit → Bathroom(Goal)",
    "subtask_destination": "corridor with pictures",
    "subtask_instruction": "Turn left 30° to face the bedroom exit, move forward 1.5m to reach the corridor, then turn left 90° to face along the corridor toward pictures.",
    "subtask_landmark": "picture",
    "completion_criteria": {{
        "Panoramic_Detection": "Pictures visible on corridor wall in Front view. Bedroom bed visible in Back view.",
        "Spatial_relationship": "Pictures on corridor wall < 1.0m (map shows decorative objects along corridor near current position). Bedroom interior far behind (map shows previous bedroom area away from current position). Orange trajectory shows left turn 30°, forward 1.5m to corridor, then left turn 90°.",
        "Location": "Corridor - pictures on wall < 1.0m, bedroom behind"
    }},
    "global_task_finish": false,
    "reasoning": "Agent currently in bedroom near exit. Bedroom exit at left 30° leading to corridor with pictures on walls (distinctive visual landmark). Distances: LEFT-30 >2m open (safe to turn), LEFT-90 0.6m (wall nearby, explains left side wall observation). Map: Left side has wall obstacle (black) near current position, but left 30° shows clear green floor path through bedroom exit. Global task requires: 1) exit to corridor, 2) navigate to kitchen, 3) reach bathroom (final goal). First subtask: turn left 30° to face bedroom exit, move forward 1.5m to reach corridor threshold, then turn left 90° to orient along corridor. Use 'picture' as landmark (specific visual feature on corridor wall, better than ambiguous 'door/doorway')."
}}

**Critical Requirements**:
- **Panoramic View Content**: Detect each portion of panoramic view for comprehensive spatial understanding and precise directional descriptions.
- **Planing**: Start all actions from Front view (0°).
- **Map**: Use maps to identify your location, landmarks, obstacles and plan safe paths.
- **Forward Direction Alignment**: Dark red dashed line shows exact Forward direction - must align with destination/safe paths, NOT obstacles. Turn immediately if misaligned.
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
4 panoramic views (90° FOV each) + 2 top-down maps:

**Direction Usage**: e.g., "To Right-View's left-portion: turn right 60°".
- Determine the location and orientation of next subtask-destination using orientation indicated on panoramic view.

**Action Origin**: All actions start from Front (IMAGE 1 center)

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

**IMAGE 5: Global Map** - Full explored area (updated trajectory, waypoints, landmarks)
**IMAGE 6: Local Map** - Nearby region (agent-centered, FOV cone shown)

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
   - Analyze surroundings, map trajectory, landmarks to identify current position
   - Determine which waypoint in sequence you are closest to or have reached
   - If off-path: Identify current location and nearest waypoint in sequence

2. **Plan Navigation to Next Waypoint**:
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
    "subtask_destination": "<Next waypoint in sequence to navigate toward>",
    "subtask_instruction": "<Step-by-step navigation instructions from current position to next waypoint>",
    "subtask_landmark": "<Single landmark name at next waypoint for detection>",
    "completion_criteria": {{
        "Panoramic_Detection": "<Next waypoint detected in which view>. <Other objects detected in which view>",(from panoramic view detection) 
        "Spatial_relationship": "<Next waypoint position and distance> (map verification). <Other objects relationships> (map verification). <Trajectory description>",
        "Location": "<Next Waypoint Area> - <relative position descriptions>"
    }},
    "global_task_finish": <true if completing this subtask will finish the entire global task, false otherwise>,
    "reasoning": "<Brief explanation of completion verification, progress, and next plan>"
}}

## Example 1:
**Global Task**: Turn around walk through the exercise room into the living room. Wait by the Table.
**Previous Subtask**: Navigate to exercise room entrance
**Current Observation:** Exercise equipment directly ahead in Front view. Already turned 120° and facing exercise room doorway. Restroom visible behind in Back view.

{{
    "waypoint": "Exercise Room Entrance - facing gym equipment ahead, restroom behind",
    "waypoint_sequence": "Restroom(✓) → Exercise Room Entrance(Current) → Exercise Room(Next) → Living Room Arched Doorway → Living Room's Table(Goal)",
    "subtask_destination": "exercise room interior",
    "subtask_instruction": "Move forward into exercise room, continue until exercise equipment visible around agent",
    "subtask_landmark": "exercise equipment",
    "completion_criteria": {{
        "Panoramic_Detection": "Exercise equipment detected surrounding agent in multiple views. Restroom far behind in Back view.",
        "Spatial_relationship": "Exercise equipment surrounding agent < 1.0m in multiple directions (map shows inside gym area). Restroom far behind (map shows previous waypoint far back). Orange trajectory shows entered gym room interior",
        "Location": "Exercise Room Interior - exercise equipment surrounding agent, restroom far behind"
    }},
    "global_task_finish": false,
    "reasoning": "Progress Check: Current - Exercise Room Entrance, facing gym equipment after 120° left turn (exercise equipment ahead in Front view, restroom behind in Back view). Target - Exercise Room Entrance (previous subtask). Waypoint Position: AT target - already at entrance facing doorway, completion criteria met (gym equipment ahead < 1.0m, restroom behind). Evidence: Map shows orange trajectory reached entrance position, panoramic view matches. Decision: Mark completed - entrance reached. Next Action: Plan NEXT waypoint to enter exercise room interior."
}}

## Example 2:
**Global Task**: Turn around and navigate to refrigerator in kitchen
**Previous Subtask**: Navigate through kitchen center
**Current Observation:** Agent in kitchen center, refrigerator visible in Front view ahead, counter to right, kitchen island behind

{{
    "waypoint": "Kitchen Center - refrigerator ahead < 0.5m, counter to right, kitchen island behind",
    "waypoint_sequence": "Bedroom(✓) → Hallway(✓) → Kitchen Center(✓) → Refrigerator(Current + Goal)",
    "subtask_destination": "refrigerator in kitchen",
    "subtask_instruction": "Stop. The refrigerator is directly ahead within 0.5m.",
    "subtask_landmark": "refrigerator",
    "completion_criteria": {{
        "Panoramic_Detection": "Refrigerator detected in Front view centered ahead. Counter detected in Right view. Kitchen island detected in Back view",
        "Spatial_relationship": "Refrigerator ahead < 0.5m (map shows refrigerator landmark within dark green circle around Red arrow). Counter at right. Kitchen island behind. Orange trajectory shows direct forward movement through kitchen to refrigerator",
        "Location": "Refrigerator Area - refrigerator ahead < 0.5m, counter at right, kitchen island behind"
    }},
    "global_task_finish": true,
    "reasoning": "Progress Check: Current - Kitchen Center in front of refrigerator (refrigerator centered in Front view IMAGE 1, counter at right IMAGE 4, kitchen island behind IMAGE 3). Target - Kitchen Center (previous subtask). Waypoint Position: AT target - refrigerator within 0.5m (inside dark green circle on local map IMAGE 6). Evidence: Refrigerator fills Front view, orange trajectory shows movement through hallway → kitchen center → refrigerator, spatial relationships match completion criteria. Decision: Mark completed - at kitchen center near refrigerator. Next Action: Global task complete - this is final destination. Execute STOP."
}}

## Example 3:
**Global Task**: Walk toward the oven.  Go through the archway on your right that is past the painting of the girl in a blue bonnet.  Go through the doorway on your left.  Stop in front of the small sink, before you reach the grill. 
**Previous Subtask**: Approach oven area
**Current Observation:** Oven visible in Front view but distance still > 1.0m. Kitchen island visible behind. Orange trajectory shows progress but hasn't reached oven yet.

{{
    "waypoint": "Kitchen - Oven ahead > 1.0m, kitchen island behind",
    "waypoint_sequence": "Starting Point(✓) → Kitchen(Current) → Oven Area(Next) → Archway Past Painting → Left Doorway → Small Sink(Goal)",
    "subtask_destination": "oven area",
    "subtask_instruction": "Continue moving forward to approach oven until oven is directly ahead < 0.5m (target: oven centered in Front view, very close)",
    "subtask_landmark": "oven",
    "completion_criteria": {{
        "Panoramic_Detection": "Oven detected in Front view centered ahead occupying large portion. Kitchen island detected far away in Back view",
        "Spatial_relationship": "Oven ahead < 0.5m (map shows oven landmark near Red arrow within dark green circle). Kitchen island far behind (map shows it away from waypoint). Orange trajectory shows forward movement toward oven",
        "Location": "Oven Area - oven ahead < 0.5m, kitchen island far behind"
    }},
    "global_task_finish": false,
    "reasoning": "Progress Check: Current - Kitchen area approaching oven (oven visible in Front view IMAGE 1 but not filling screen, kitchen island visible behind IMAGE 3). Target - Oven area (subtask destination). Waypoint Position: BEFORE target - oven detected but distance > 1.0m on map, outside dark green circle. Evidence: Oven visible but small in Front view, orange trajectory shows forward progress but waypoint marker nearby indicates previous stop location, haven't reached oven area yet. Completion criteria requires oven < 0.5m filling Front view. Decision: Mark NOT completed - still approaching target. Next Action: Continue SAME subtask (oven area), move forward until oven is very close (< 0.5m, inside dark green circle, filling entire Front view)."
}}

**Critical Requirements**:
- **Panoramic View Content**: Detect each portion of panoramic view for comprehensive spatial understanding and precise directional descriptions.
- **Sequential Navigation**: Treat waypoint_sequence as a chain to follow progressively. Identify current position → plan to next waypoint. Do NOT turn back to previous waypoints.
- **Off-Path Recovery**: If deviated from sequence, identify current location and plan route to nearest upcoming waypoint in sequence.
- **Forward Direction Alignment**: Dark red dashed line shows exact Forward direction - must align with destination/safe paths, NOT obstacles. Turn immediately if misaligned.
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