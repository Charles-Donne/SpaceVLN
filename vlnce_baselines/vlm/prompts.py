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
    "waypoint_sequence": "<Current Location> → <Next Waypoint> → ... → <Final Waypoints>",
    "next_waypoint_direction": "<IMAGE number where next waypoint appears most centered/visible (1-12)>",
    "next_waypoint_destination": "<Next immediate waypoint name>",
    "subtask_instruction": "<Step-by-step navigation instructions starting from Front view>",
    "next_waypoint_landmark": "<Single landmark to detect (common, e.g. door, table, painting, cabinet)>",
    "completion_criteria": {{
        "Surrounding_Detection": "<Destination detected in which view>. <Other objects detected in which view>",
        "Spatial_relationship": "<Destination position and distance> (map verification). <Other objects relationships> (map verification). <Trajectory description>",
        "Location": "<Current Area Type> - <relative position descriptions>"
    }},
    "global_task_finish": <true if completing this subtask will finish the entire global task, false otherwise>,
    "reasoning": "<Max 200 words: Brief explanation of observation and analysis leading to this subtask planning>"
}}

#Examples:

## Ex1: 
**Global Task**: Turn around walk through the exercise room into the living room. Wait by the Table.
**Current Observation:** IMAGE 1 (Front 0°): Bookshelf visible at distance. IMAGE 5 (Left 120°): Exercise room doorway visible with gym equipment inside. IMAGE 10 (Right 270°): Toilet and washbasin visible

{{
    "current_waypoint": "Restroom - beside exercise room doorway, toilet and washbasin nearby.",
    "waypoint_sequence": "Restroom(Current) → Exercise Room → Living Room → Living Room's Table(Goal)",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "exercise room",
    "subtask_instruction": "Move forward through doorway to enter the exercise room",
    "next_waypoint_landmark": "exercise equipment",
    "completion_criteria": {{
        "Surrounding_Detection": "Exercise equipment detected surrounding in multiple views. Restroom fixtures detected in Back",
        "Spatial_relationship": "Exercise equipment surrounding agent < 1.0m (map shows inside gym area). Restroom far behind (map shows previous location). Orange trajectory shows entered exercise room interior",
        "Location": "Exercise Room - exercise equipment surrounding, restroom behind"
    }},
    "global_task_finish": false,
    "reasoning": "IMAGE 5 (Left 120°) shows exercise room doorway with gym equipment - next waypoint. System will auto-rotate 120° left to face this direction. After rotation, move forward through doorway into exercise room. Local map shows dark green circle (0.5m range) clear. Global map shows red arrow (current position) in small room, orange trajectory shows arrival path, exercise room (larger green area) is to the left. Using 'exercise equipment' as landmark."
}}

## Ex2:
**Global Task**: Exit the room and turn left, head toward the kitchen and turn right. Go through the kitchen and out. Wait right at the bathroom.
**Current Observation:** IMAGE 1 (Front 0°): Open space ahead. IMAGE 2 (Left 30°): Bedroom exit doorway visible, corridor with pictures beyond. IMAGE 4 (Left 90°): Wall nearby

{{
    "current_waypoint": "Bedroom - near exit",
    "waypoint_sequence": "Bedroom(Current) → Corridor → Kitchen Entrance → Kitchen → Kitchen Exit → Bathroom(Goal)",
    "next_waypoint_direction": "IMAGE 2 (Left 30°)",
    "next_waypoint_destination": "corridor with pictures",
    "subtask_instruction": "Move forward through doorway to reach corridor",
    "next_waypoint_landmark": "picture",
    "completion_criteria": {{
        "Surrounding_Detection": "Pictures detected on corridor wall in Front. Bedroom interior detected in Back",
        "Spatial_relationship": "Pictures on corridor wall < 0.5m (map shows decorative objects along corridor). Bedroom interior far behind (map shows previous area). Orange trajectory shows forward 1.5m movement to corridor",
        "Location": "Corridor - pictures on wall < 1.0m, bedroom behind"
    }},
    "global_task_finish": false,
    "reasoning": "IMAGE 2 (Left 30°) shows bedroom exit with corridor and pictures visible - next waypoint. System will auto-rotate 30° left to face this direction. After rotation, move forward 1.5m through doorway. Global map shows red arrow in bedroom (enclosed green area), corridor extends to the left with green floor area. Orange trajectory short (just started). Local map shows doorway ahead. Blue filled area (90° FOV) will show doorway centered after rotation. Using 'picture' as landmark."
}}

**Critical Requirements**:
- **12-Direction Analysis**: Analyze all 12 directional views (IMAGE 1-12) to locate current position and next waypoint
- **Auto-Rotation System**: Specify waypoint_direction, system will auto-rotate to face it. Write instructions assuming agent is already facing the waypoint
- **Sequential Navigation**: Treat waypoint_sequence as a chain to follow progressively. Identify current position → plan to next waypoint. Do NOT turn back to previous waypoints
- **Off-Path Recovery**: If deviated from sequence, identify current location and plan route to nearest upcoming waypoint
- **Forward Direction Alignment**: After auto-rotation, agent faces waypoint directly. Verify alignment and move forward
- **Path Alignment**: Keep agent centered in corridors/paths, parallel to walls/boundaries with equal distance to both sides
- **Target Alignment**: After auto-rotation, destination/landmark will be centered in Front view (0°)
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
- **Observe all 12 views**: Previous waypoint markers (blue circles with numbers + room type) show visited locations
- **Logical reasoning**: Determine what next waypoint should be and which IMAGE (1-12) it appears in
- **Avoid backtracking**: Don't return to visited waypoints unless necessary
- **System auto-rotates** to your specified waypoint_direction → Next Waypoint becomes Front view (IMAGE 1)

**Global Map** - Full explored area (updated trajectory, waypoints, landmarks)
**Local Map** - Nearby region (agent-centered, FOV cone shown)

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

2. **Determine Next Waypoint**: 
   - Observe 12 views + previous waypoint markers (blue circles with room types)
   - Logical reasoning: What should next waypoint be? Which IMAGE (1-12)?
   - Choose most centered/visible angle, avoid backtracking

3. **Plan Navigation**: 
   - Specify waypoint_direction → **system auto-rotates** → Next Waypoint in Front view (IMAGE 1)
   - Write instructions assuming already facing Next Waypoint
   - On-path: navigate to next waypoint | Off-path: return to nearest waypoint → continue
   - Move forward in sequence, do NOT turn back

# Actions Available

**Turn**: TURN_LEFT/RIGHT (30°, 60°, 90°, 120°, 150°, 180°)
**Move**: MOVE_FORWARD (0.25m, 0.5m, 0.75m, 1.0m, 1.25m, 1.5m)
**Arrive**: STOP (when <0.5m from destination)
- Use key actions (turn/move/stop) to navigate, but use fewer precise parameters (meters/degrees)

# Output Format (JSON only)

**CRITICAL**: Output ONLY valid JSON. No extra text before or after.
**Word Limits**:
- "reasoning": MAX 200 words (concise completion check + next plan)
- "subtask_instruction": MAX 100 words (clear, actionable steps)
- Other fields: Keep concise (20-50 words each)

{{
    "current_waypoint": "<Current Area Type> - <Key Surrounding Landmarks and Relationships>",
    "waypoint_sequence": "<Completed Waypoints(✓)> → <Current Position> → <Next Waypoint> → <Remaining Waypoints> → <Goal>",
    "next_waypoint_direction": "<IMAGE number where next waypoint appears most centered/visible (1-12)>",
    "next_waypoint_destination": "<Next waypoint in sequence to navigate toward>",
    "subtask_instruction": "<Step-by-step navigation instructions from current position to next waypoint>",
    "next_waypoint_landmark": "<Single landmark name at next waypoint for detection>",
    "completion_criteria": {{
        "Surrounding_Detection": "<Next waypoint detected in which view>. <Other objects detected in which view>",
        "Spatial_relationship": "<Next waypoint position and distance> (map verification). <Other objects relationships> (map verification). <Trajectory description>",
        "Location": "<Next Waypoint Area> - <relative position descriptions>"
    }},
    "global_task_finish": <true if completing this subtask will finish the entire global task, false otherwise>,
    "reasoning": "<Max 200 words: Brief explanation of completion verification, progress, and next plan>"
}}

## Example 1:
**Global Task**: Turn around walk through the exercise room into the living room. Wait by the Table.
**Previous Subtask**: Navigate to exercise room
**Current Observation:** IMAGE 1 (Front 0°): Exercise equipment surrounding. IMAGE 7 (Back 180°): Restroom visible behind

{{
    "current_waypoint": "Exercise Room (entrance area) - gym equipment surrounding, restroom behind",
    "waypoint_sequence": "Restroom(✓) → Exercise Room(Current) → Living Room → Living Room's Table(Goal)",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "living room",
    "subtask_instruction": "Move forward through the exercise room toward the living room exit",
    "next_waypoint_landmark": "arched doorway",
    "completion_criteria": {{
        "Surrounding_Detection": "Arched doorway detected in Front. Exercise equipment detected in Back",
        "Spatial_relationship": "Arched doorway ahead (map shows exit to living room). Exercise equipment behind (map shows gym area). Orange trajectory shows movement through exercise room toward living room",
        "Location": "Living Room Entrance - arched doorway ahead, exercise room behind"
    }},
    "global_task_finish": false,
    "reasoning": "Current in Exercise Room, gym equipment surrounding in multiple images, restroom behind in IMAGE 7. Previous subtask (enter exercise room) completed. Global map shows red arrow inside gym area, orange trajectory shows entered from restroom. Local map shows inside exercise room with equipment around. Next waypoint is Living Room - need to cross exercise room to find living room exit. Move forward through gym to locate arched doorway leading to living room."
}}

## Example 2:
**Global Task**: Turn around and navigate to refrigerator in kitchen
**Previous Subtask**: Navigate through kitchen center
**Current Observation:** IMAGE 1 (Front 0°): Refrigerator directly ahead < 0.5m. IMAGE 10 (Right 270°): Counter visible. IMAGE 7 (Back 180°): Kitchen island visible

{{
    "current_waypoint": "Kitchen Center - refrigerator ahead < 0.5m, counter to right, kitchen island behind",
    "waypoint_sequence": "Bedroom(✓) → Hallway(✓) → Kitchen Center(✓) → Refrigerator(Current + Goal)",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "refrigerator in kitchen",
    "subtask_instruction": "Stop. The refrigerator is directly ahead within 0.5m",
    "next_waypoint_landmark": "refrigerator",
    "completion_criteria": {{
        "Surrounding_Detection": "Refrigerator detected in Front centered ahead. Counter detected in Right. Kitchen island detected in Back",
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
    "current_waypoint": "Kitchen - Oven ahead > 1.0m, kitchen island behind",
    "waypoint_sequence": "Starting Point(✓) → Kitchen(Current) → Oven Area(Next) → Archway Past Painting → Left Doorway → Small Sink(Goal)",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "oven area",
    "subtask_instruction": "Continue moving forward to approach oven until oven is directly ahead < 0.5m (target: oven centered in Front view, very close)",
    "next_waypoint_landmark": "oven",
    "completion_criteria": {{
        "Surrounding_Detection": "Oven detected in Front centered ahead occupying large portion. Kitchen island detected far away in Back",
        "Spatial_relationship": "Oven ahead < 0.5m (map shows oven landmark within dark green circle). Kitchen island far behind. Orange trajectory shows forward movement toward oven",
        "Location": "Oven Area - oven ahead < 0.5m, kitchen island far behind"
    }},
    "global_task_finish": false,
    "reasoning": "Current approaching oven - IMAGE 1 shows oven visible, IMAGE 7 shows kitchen island behind. Target is oven area. Position: BEFORE target - oven detected but > 1.0m. Global map shows red arrow in kitchen, orange trajectory shows forward progress toward oven area. Local map shows oven outside dark green circle (> 0.5m away), dark red dashed line aligned with oven direction, blue filled area shows oven visible but not yet close. No black obstacles blocking path ahead. Completion requires oven < 0.5m (inside dark green circle). Oven in IMAGE 1, no turn needed. Continue forward 0.75m until oven enters dark green circle."
}}

**Critical Requirements**:
- **12-Direction Analysis**: Analyze all 12 directional views (IMAGE 1-12) to locate current position and next waypoint
- **Auto-Rotation System**: Specify waypoint_direction, system will auto-rotate to face it. Write instructions assuming agent is already facing the Next Waypoint
- **Sequential Navigation**: Treat waypoint_sequence as a chain to follow progressively. Identify current position → plan to next waypoint. Do NOT turn back to previous waypoints
- **Off-Path Recovery**: If deviated from sequence, identify current location and plan route to nearest upcoming waypoint
- **Forward Direction Alignment**: After auto-rotation, agent faces Next Waypoint directly. Verify alignment and move forward
- **Path Alignment**: Keep agent centered in corridors/paths, parallel to walls/boundaries with equal distance to both sides
- **Target Alignment**: After auto-rotation, destination/landmark will be centered in Front view (0°)
- **Distance Judgment**: Use dark green circle on local map to determine if destination/landmark is nearby - objects within the circle are < 0.5m from current position
- **Planning**: Start all actions from Front view (0°) after auto-rotation. If subtask completed, plan NEXT waypoint; if not, adjust CURRENT subtask
- **Map**: Use maps to verify trajectory, identify obstacles and plan safe paths for next subtask
- **Landmark Selection**: Priority: landmarks from Global Task (e.g., "Wait by the Table" → "table"). Use specific objects (chair, table, bed, cabinet, sofa, painting). Avoid ambiguous terms (door, doorway, entrance, wall).
- **Logical Analysis**: Ensure reasoning and output aligns with inputs - All the content must not contain any contradictions.
- **Explore Unseen Areas**: If the destination is invisible, explore more places but avoiding areas with too many history waypoints, and understand the spatial relationships.
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