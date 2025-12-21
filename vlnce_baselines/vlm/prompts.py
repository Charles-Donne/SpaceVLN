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

**IMAGE 5: Global Map** - Full explored area
**IMAGE 6: Local Map** - Nearby region (agent-centered, FOV cone shown)

# Map Interpretation Guide

**Map Orientation**: 
- Top of map = Agent's current Front direction (Front is always up)

**Global Map**:
- **White**: Unexplored/unknown areas
- **Black**: Obstacles (walls, furniture, barriers) - **MUST AVOID in planning**
- **Green**: Confirmed floor areas (safe to navigate)
- **Orange line**: Trajectory from subtask start to current position
- **Red arrow**: Current position, arrow points to Front direction
  
**Local Map** (zoomed view around agent, same color legend as Global Map):
- Shows finer details in immediate vicinity for precise navigation
- **Red arrow**: Current position, arrow points to Front direction
- **Dark green circle**: 0.5m radius nearby area around current position
- **Blue semi-circle**: Agent's current field of view (Front direction visibility range)
  - The opening of the semi-circle indicates Front view direction
  - Objects within this blue region are currently visible in IMAGE 1 Front (Front View)
- Better for planning nearby movements and obstacle avoidance

**Use Maps for Planning**:
- **Identify obstacles**: Black areas and space behind black areas(unexplored).
- **Spatial awareness**: Use global map for overall layout, local map for immediate surroundings

# Your Task

1. **Analyze environment**: Use 4-directional views + global and local map to identify your position, related landmarks and obstacles
2. **Plan subtask**: Break down global task into achievable intermediate waypoints
3. **Provide instructions**: Action instructionn starting from Front view using concrete landmarks

**Available Actions**: {action_space}

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
**Current Observation: Far front is a bookshelf. Toilet and Sink can be seen from right view. Left is a wall but left 120° is doorway to gym. 
{{
    "waypoint": "Restroom - beside exercise room door, toilet and washbasin nearby.",
    "waypoint_sequence": "Restroom(Current) → Exercise Room Entrance → Exercise Room → Living Room → Living Room's Table(Goal)",
    "subtask_destination": "exercise room entrance",
    "subtask_instruction": "Turn left 120° to face doorway, then move forward 0.5m to stop at gym's entrance.",
    "subtask_landmark": "door",
    "completion_criteria": {{
        "Panoramic_Detection": "Door detected in Front view centered ahead.",
        "Spatial_relationship": "Door ahead < 0.5m (map shows doorway at Red arrow). Restroom far behind (map shows away from last waypoint). Orange trajectory shows approach to doorway",
        "Location": "Exercise Room Entrance - doorway ahead < 0.5m, restroom far behind"
    }},
    "global_task_finish": false,
    "reasoning": "Agent currently in Restroom (toilet and washbasin visible from right view, bookshelf at far front). Exercise room door visible at left 120° (left portion of Left-View). Map: Left 90° is wall obstacle (black), green floor path clear after turning left 120° leading to doorway, no black obstacles blocking approach to doorway. Global task requires passing through exercise room to reach living room table, so first waypoint is exercise room entrance."
}}

## Ex2:
**Global Task**: Walk to the kitchen, stop at the refrigerator.
**Current Observation:** Living Room visible with Bar in front. Kitchen visible beyond Bar. Chair and sofa nearby.
{{
    "waypoint": "Living Room - near chair, sofa and Bar, kitchen far ahead.",
    "waypoint_sequence": "Living Room(Current) → Bar → Kitchen → Kitchen's Refrigerator(Goal)",
    "subtask_destination": "Bar area",
    "subtask_instruction": "Turn right 30° to avoid Bar obstacle, move forward 1.5m to pass by Bar",
    "subtask_landmark": "Bar",
    "completion_criteria": {{
        "Panoramic_Detection": "Kitchen entrance detected in Front view ahead. Bar detected in Left view. Sofa and chair detected far away in Back view",
        "Spatial_relationship": "Bar at left < 0.5m (map shows bar landmark is on left and at Red arrow). Kitchen entrance ahead (map shows green path leading to kitchen). Sofa and chair far behind (map shows starting position). Orange trajectory shows right turn to bypass bar",
        "Location": "Bar Area - Bar at left < 0.5m, kitchen entrance ahead, living room far behind"
    }},
    "global_task_finish": false,
    "reasoning": "Agent currently in Living Room (chair and sofa visible). Bar blocking front path, kitchen entrance visible beyond at right 30° (right portion of Front-View). Map: Front direction blocked by black Bar obstacle, green floor path clear at right 30° leading around Bar toward kitchen. Global task requires passing through living room to kitchen and reaching refrigerator, so first waypoint is Bar area."
}}

**Critical Requirements**:
- **Panoramic View Content**: Detect each portion of panoramic view for comprehensive spatial understanding and precise directional descriptions.
- **Planing**: Start all actions from Front view (0°).
- **Map**: Use maps to identify your location, landmarks, obstacles and plan safe paths.
- **Distance Judgment**: Use dark green circle on local map to determine if destination/landmark is nearby - objects within the circle are < 0.5m from current position
- **Landmark Selection**: Choose common furniture items with simple nouns for easy detection (e.g., door, chair, table, bed, cabinet, refrigerator, sofa)
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

**IMAGE 5: Global Map** - Full explored area (updated trajectory, waypoints, landmarks)
**IMAGE 6: Local Map** - Nearby region (agent-centered, FOV cone shown)

# Map Interpretation Guide

**Map Orientation**: 
- Top of map = Agent's current Front direction (Front is always up)

**Global Map**:
- **White**: Unexplored/unknown areas
- **Black**: Obstacles (walls, furniture, barriers) - **MUST AVOID in next planning**
- **Green**: Confirmed floor areas (safe to navigate)
- **Orange line**: Trajectory from previous subtask start to current position
- **Red arrow**: Current position, arrow points to Front direction
- **Purple markers with labels**: Previous Detected Landmark: {detected_landmarks}
- **Blue circles with white numbers**: Historical waypoints (see below)

# Spatial Memory (Waypoint History):
{waypoint_summary}
- Each numbered waypoint indicates a previously 360° scan and thinking location
  
**Local Map** (zoomed view around agent, same color legend as Global Map):
- Shows finer details in immediate vicinity for precise navigation
- **Red arrow**: Current position, arrow points to Front direction
- **Dark green circle**: 0.5m radius nearby area around current position
- **Blue semi-circle**: Agent's current field of view (Front direction visibility range)
  - The opening of the semi-circle indicates Front view direction
  - Objects within this blue region are currently visible in IMAGE 1 (Front View)
- Better for planning nearby movements and obstacle avoidance

**Use Maps for Verification & Planning**:
- **Verify Previous Subtask**: Detect current position, previous landmarks({detected_landmarks})'s direction and distance, trajectory history to confirm if previous subtask completed.
- **Identify obstacles**: Black areas and space behind black areas(unexplored) - MUST AVOID in next planning.
- **Spatial awareness**: Use global map for overall layout, local map for immediate surroundings.

# Your Task

1. **Verify completion**: Compare current visual observations and map with previous subtask completion_criteria (Panoramic_Detection, Spatial_relationship, Location)
2. **Make decision**: 
   - **is_completed = true**: Subtask finished → plan NEXT waypoint
   - **is_completed = false**: Not finished → continue SAME subtask
3. **Plan next step**: If completed, update waypoint_sequence and define new subtask; if not, adjust current subtask instruction

**Available Actions**: {action_space}

# Output Format (JSON only)

{{
    "is_completed": <true if previous subtask completed, false if not>,
    "waypoint": "<Current Area Type> - <Key Surrounding Landmarks and Relationships>",
    "waypoint_sequence": "<Completed Waypoints(✓)> → <Current> → <Remaining Waypoints> → <Final Waypoints>",
    "subtask_destination": "<Next waypoint if completed, same waypoint if not>",
    "subtask_instruction": "<Step-by-step navigation instructions from Front view>",
    "subtask_landmark": "<Single landmark name for map marking>",
    "completion_criteria": {{
        "Panoramic_Detection": "<Destination detected in which view>. <Other objects detected in which view>",(from panoramic view detection)
        "Spatial_relationship": "<Destination position and distance> (map verification). <Other objects relationships> (map verification). <Trajectory description>",
        "Location": "<Current Area Type> - <relative position descriptions>"
    }},
    "global_task_finish": <true if completing this subtask will finish the entire global task, false otherwise>,
    "reasoning": "<Brief explanation of completion verification, progress, and next plan>"
}}

## Example 1:
**Global Task**: Turn around walk through the exercise room into the living room. Wait by the Table.
**Previous Subtask**: Navigate to exercise room entrance
**Current Observation:** Living room visible at left 30°-60° (right portion of Left-View and left portion of Front-View) beyond arched doorway. Exercise equipment blocking direct left path in Front view.

{{
    "is_completed": true,
    "waypoint": "Exercise Room - exercise equipment nearby, living room and bathroom in far away",
    "waypoint_sequence": "Restroom(✓) → Exercise Room Entrance(✓) → Exercise Room(Current) → Living Room Arched Doorway → Living Room's Table(Goal)",
    "subtask_destination": "living room arched doorway",
    "subtask_instruction": "Move forward to bypass exercise equipment, then turn left to face arched doorway when left side is no obstacle, move forward and stop in front of sofa",
    "subtask_landmark": "sofa",
    "completion_criteria": {{
        "Panoramic_Detection": "Sofa and arched doorway detected in Front view ahead. Exercise equipment detected far away in Back view",
        "Spatial_relationship": "Sofa ahead < 0.5m (map shows sofa landmark is at Red arrow). Exercise equipment far behind (map shows black obstacles at previous waypoint). Orange trajectory shows forward movement then turn to living room",
        "Location": "Living Room Arched Doorway - sofa and doorway ahead < 0.5m, exercise room far behind"
    }},
    "global_task_finish": false,
    "reasoning": "Previous subtask completed: reached exercise room entrance, orange trajectory confirms entry. Current position: inside exercise room near entrance, exercise equipment visible at left blocking direct path. Living room visible at left 30°-60° (right portion of Left-View + left portion of Front-View) through arched doorway. Map: exercise equipment is black obstacle at left, green floor path clear straight ahead to bypass equipment, then green path clear at left 30° after bypass leading to arched doorway. Global task requires passing through exercise room to reach living room table, so next waypoint is living room arched doorway area with sofa."
}}

## Example 2:
**Global Task**: Turn around and navigate to refrigerator in kitchen
**Previous Subtask**: Navigate through kitchen center
**Current Observation:** Agent in kitchen center, refrigerator visible in Front view ahead, counter to right, kitchen island behind

{{
    "is_completed": true,
    "waypoint": "Kitchen Center - refrigerator ahead, counter to right, kitchen island behind",
    "waypoint_sequence": "Bedroom(✓) → Hallway(✓) → Kitchen Center(✓) → Refrigerator(Current, Goal)",
    "subtask_destination": "refrigerator in kitchen",
    "subtask_instruction": "Stop. The refrigerator is directly ahead within 0.5m.",
    "subtask_landmark": "refrigerator",
    "completion_criteria": {{
        "Panoramic_Detection": "Refrigerator detected in Front view centered ahead. Counter detected in Right view. Kitchen island detected in Back view",
        "Spatial_relationship": "Refrigerator ahead < 0.5m (map shows refrigerator landmark is within the dark green circle around Red arrow). Counter at right (map shows counter landmark is to the right of the agent). Kitchen island behind (map shows island landmark is behind the agent). Orange trajectory shows direct forward movement through kitchen to refrigerator",
        "Location": "Refrigerator Area - refrigerator ahead < 0.5m, counter at right, kitchen island behind"
    }},
    "global_task_finish": true,
    "reasoning": "Previous subtask completed: orange trajectory shows entered kitchen center and moved forward. Current position: in front of refrigerator. The refrigerator is centered in the Front view (IMAGE 1) and is within the dark green circle on the local map (IMAGE 6), confirming it is < 0.5m away. The counter is visible in the Right view (IMAGE 4), and the kitchen island is visible in the Back view (IMAGE 3), matching the spatial relationships. The orange trajectory on the global map (IMAGE 5) confirms the agent moved through hallway and kitchen center to reach the refrigerator. Since the global task was to 'Turn around and navigate to refrigerator in kitchen', and the refrigerator is now directly ahead within 0.5m, the entire task is complete. No further navigation is required."
}}

## Example 3:
**Global Task**: Walk toward the oven.  Go through the archway on your right that is past the painting of the girl in a blue bonnet.  Go through the doorway on your left.  Stop in front of the small sink, before you reach the grill. 
**Previous Subtask**: Approach oven area
**Current Observation:** Oven visible in Front view but distance still > 1.0m. Kitchen island visible behind. Orange trajectory shows progress but hasn't reached oven yet.

{{
    "is_completed": false,
    "waypoint": "Kitchen - Oven is far ahead, kitchen island nearby.",
    "waypoint_sequence": "Kitchen(Current) → Oven Area → Archway Past Painting → Left Doorway → Small Sink(Goal)",
    "subtask_destination": "oven area",
    "subtask_instruction": "Continue moving forward 1.0m to approach oven until oven is directly ahead < 0.5m (target: oven centered in Front view, very close)",
    "subtask_landmark": "oven",
    "completion_criteria": {{
        "Panoramic_Detection": "Oven detected in Front view centered ahead occupying large portion. Kitchen island detected far away in Back view",
        "Spatial_relationship": "Oven ahead < 0.5m (map shows oven landmark is at Red arrow). Kitchen far behind (map shows it away from last waypoint). Orange trajectory shows forward movement toward oven",
        "Location": "Oven Area - oven ahead < 0.5m, kitchen far behind"
    }},
    "global_task_finish": false,
    "reasoning": "Previous subtask NOT completed. Current position: kitchen area far from oven. Oven detected in Front view (IMAGE 1) but distance still > 1.0m on map (orange trajectory shows progress but hasn't reached oven yet, waypoint 1 nearby shows previous stop location). Completion criteria requires distance < 0.5m with oven filling Front view. Kitchen visible behind (IMAGE 3, Back 180°). Local map: clear green floor path ahead toward oven, no black obstacles blocking. Root cause: insufficient forward movement. Corrective subtask: same destination (oven area) with instruction to continue forward until very close."
}}

**Critical Requirements**:
- **Panoramic View Content**: Detect each portion of panoramic view for comprehensive spatial understanding and precise directional descriptions.
- **Verification**: Compare current observations against all three completion_criteria fields (Panoramic_Detection, Spatial_relationship, Location)
- **Distance Judgment**: Use dark green circle on local map to determine if destination/landmark is nearby - objects within the circle are < 0.5m from current position
- **Planning**: Start all actions from Front view (0°). If subtask completed, plan NEXT waypoint; if not, adjust CURRENT subtask
- **Map**: Use maps to verify trajectory, identify obstacles and plan safe paths for next subtask
- **Landmark Selection**: Choose common furniture items with simple nouns for easy detection (e.g., door, chair, table, bed, cabinet, refrigerator, sofa)
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