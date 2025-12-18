"""
VLM规划提示词模板
================
用于LLM高层规划的提示词模板
"""

# 初始规划提示词 - 在任务开始时生成第一个子任务
INITIAL_PLANNING_PROMPT = """You are a Vision-Language Navigation planning module. Analyze the environment and design the next navigation subtask.

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
- **Red circle with arrow**: Current position, arrow points to Front direction
  
**Local Map** (zoomed view around agent, same color legend as Global Map):
- Shows finer details in immediate vicinity for precise navigation
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
    "waypoint": "<Current Area Type> - <Key Surrounding Landmarks>",
    "waypoint_sequence": "<Current Location> → <Next Waypoint> → ... → <Final Waypoints>",
    "subtask_destination": "<Next immediate waypoint name>",
    "subtask_instruction": "<Step-by-step navigation instructions starting from Front view>",
    "subtask_landmark": "<Single landmark to detect (common, e.g. door, table, painting, cabinet)>",
    "completion_criteria": {{
        "Object_Detection": "<What landmark should be detected in 4-directional views>",
        "Location": "<Current Area Type> - <Key Surrounding Landmarks>",
        "Spatial_relationship": "<Destination should be directly ahead. Distance. Other Objects Relationships>"
    }},
    "is_final_subtask": <true if this is the final subtask to finish global task (next waypoint is final waypoint), false otherwise>,
    "reasoning": "<Brief explanation of analysis, waypoint selection, and action plan>"
}}

#Examples:

## Ex1: 
**Global Task**: Turn around walk through the exercise room into the living room. Wait by the Table.
**Current Observation: Far front is a bookshelf. Toilet and Sink can be seen from right view. Left is a wall but left 120° is doorway to gym. 
{{
    "waypoint": "Restroom - beside exercise room door, toilet and washbasin.",
    "waypoint_sequence": "Restroom(Current) → Exercise Room Entrance → Exercise Room → Living Room → Living Room's Table(Goal)",
    "subtask_destination": "exercise room entrance",
    "subtask_instruction": "Turn left 120° to face doorway, then move forward 0.5m to stop at gym's entrance.",
    "subtask_landmark": "door",
    "completion_criteria": {{
        "Object_Detection": "Door detected in Front view",
        "Location": "Exercise Room Entrance - doorway centered in front",
        "Spatial_relationship": "Door directly ahead < 0.5m. Restroom is behind."
    }},
    "is_final_subtask": false,
    "reasoning": "Agent currently in Restroom (toilet and washbasin visible from right view, bookshelf at far front). Exercise room door visible at left 120° (left portion of Left-View). Map: Left 90° is wall obstacle (black), green floor path clear after turning left 120° leading to doorway, no black obstacles blocking approach to doorway. Global task requires passing through exercise room to reach living room table, so first waypoint is exercise room entrance. Action plan: turn left 120° to face doorway in Front view, move forward 0.5m to stop at entrance (distance < 0.5m confirms arrival at doorway, Restroom will be behind)."
}}

## Ex2:
**Global Task**: Walk to the kitchen, stop at the refrigerator.
**Current Observation:** Living Room visible with Bar in front. Kitchen visible beyond Bar. Chair and sofa nearby.
{{
    "waypoint": "Living Room - near chair, sofa and Bar",
    "waypoint_sequence": "Living Room(Current) → Bar → Kitchen → Kitchen's Refrigerator(Goal)",
    "subtask_destination": "Bar area",
    "subtask_instruction": "Turn right 30° to avoid Bar obstacle, move forward 1.5m to pass by Bar",
    "subtask_landmark": "Bar",
    "completion_criteria": {{
        "Object_Detection": "Bar detected in Left view. Kitchen entrance visible ahead",
        "Location": "Bar Area - Bar at left, path to kitchen clear ahead",
        "Spatial_relationship": "Bar at Left < 0.5m. Kitchen in front. Sofa and chair behind."
    }},
    "is_final_subtask": false,
    "reasoning": "Agent currently in Living Room (chair and sofa visible). Bar blocking front path, kitchen entrance visible beyond at right 30° (right portion of Front-View). Map: Front direction blocked by black Bar obstacle, green floor path clear at right 30° leading around Bar toward kitchen. Global task requires passing through living room to kitchen and reaching refrigerator, so first waypoint is Bar area. Action plan: turn right 30° to bypass Bar (avoid black obstacle), move forward 1.5m to pass by Bar (distance < 0.5m confirms passing Bar area)."
}}

**Critical Requirements**:
- **Panoramic View Content**: Detect each portion of panoramic view for comprehensive spatial understanding and precise directional descriptions.
- **Planing**: Start all actions from Front view (0°).
- **Map**: Use maps to identify your location, landmarks, obstacles and plan safe paths.
- **Landmark Selection**: Choose common furniture items with simple nouns for easy detection (e.g., door, chair, table, bed, cabinet, refrigerator, sofa)
- **Logical Analysis**: Ensure reasoning and output aligns with inputs - All the content must not contain any contradictions.
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """You are a Vision-Language Navigation verification module. Verify subtask completion and plan the next navigation step.

# Navigation Global Task:
{instruction}

# Previous Subtask Context:
**Waypoint Sequence**: {waypoint_sequence}
**Subtask Destination**: {subtask_destination}
**Subtask Instruction**: {subtask_instruction}
**Completion Criteria**: {completion_criteria}

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
- **Orange line**: Trajectory from subtask start to current position
- **Red circle with arrow**: Current position, arrow points to Front direction
- **Purple markers with labels**: Detected landmark objects: {detected_landmarks}
- **Dark red circles with white numbers**: Historical waypoints (see below)
  
**Local Map** (zoomed view around agent, same color legend as Global Map):
- Shows finer details in immediate vicinity for precise navigation
- **Blue semi-circle**: Agent's current field of view (Front direction visibility range)
  - The opening of the semi-circle indicates Front view direction
  - Objects within this blue region are currently visible in IMAGE 1 (Front View)
- Better for planning nearby movements and obstacle avoidance

# Spatial Memory (Waypoint History)
{waypoint_summary}

**Note**: Waypoint markers appear as **dark red circles with white numbers** on the Global Map.

**Use Maps for Verification & Planning**:
- **Verify trajectory**: Check orange line shows expected movement toward previous subtask destination
- **Identify obstacles**: Black areas and space behind black areas(unexplored) - MUST AVOID in next planning
- **Spatial awareness**: Use global map for overall layout, local map for immediate surroundings

# Your Task

1. **Verify completion**: Compare current observations with completion_criteria (Object_Detection, Location, Spatial_relationship)
2. **Make decision**: 
   - **is_completed = true**: Subtask finished → plan NEXT waypoint
   - **is_completed = false**: Not finished → continue SAME subtask
3. **Plan next step**: If completed, update waypoint_sequence and define new subtask; if not, adjust current subtask instruction

**Available Actions**: {action_space}

# Output Format (JSON only)

{{
    "is_completed": <true if previous subtask completed, false if not>,
    "waypoint": "<Current Area Type> - <Key Surrounding Landmarks>",
    "waypoint_sequence": "<Completed Waypoints(✓)> → <Current> → <Remaining Waypoints> → <Final Waypoints>",
    "subtask_destination": "<Next waypoint if completed, same waypoint if not>",
    "subtask_instruction": "<Step-by-step navigation instructions from Front view>",
    "subtask_landmark": "<Single landmark name for map marking>",
    "completion_criteria": {{
        "Object_Detection": "<What landmark/object should be detected in which view>",
        "Location": "<Current Area Type> - <Key landmarks and spatial context>",
        "Spatial_relationship": "<Distance/position/trajectory/orientation description>"
    }},
    "is_final_subtask": <true if next destination is final goal, false otherwise>,
    "reasoning": "<Brief explanation of completion verification, progress, and next plan>"
}}

## Example 1:
**Global Task**: Walk through hallway to kitchen entrance
**Previous Subtask**: Enter bedroom doorway
**Current Observation:** Agent at doorway, door detected, trajectory shows entry completed

{{
    "is_completed": true,
    "waypoint": "Doorway - between bedroom and hallway",
    "waypoint_sequence": "Bedroom(✓) → Doorway(Current) → Hallway → Kitchen Entrance(Goal)",
    "subtask_destination": "hallway midpoint",
    "subtask_instruction": "Move forward 1.5m through hallway center toward kitchen entrance",
    "subtask_landmark": "wall",
    "completion_criteria": {{
        "Object_Detection": "Hallway walls visible on both sides in Front view",
        "Location": "Hallway Midpoint - walls left/right, kitchen entrance ahead",
        "Spatial_relationship": "Destination ahead. Orange trajectory extends through hallway center toward kitchen"
    }},
    "is_final_subtask": false,
    "reasoning": "Previous subtask completed: door detected in Front view, orange trajectory confirms reached doorway. Current position: doorway between bedroom and hallway, hallway extending ahead (IMAGE 1). Map: green floor path clear straight through hallway center, black walls both sides, no obstacles. Global task requires reaching kitchen entrance at hallway end, so next waypoint is hallway midpoint. Action plan: move forward 1.5m straight through hallway center toward kitchen."
}}

## Example 2:
**Global Task**: Turn around and navigate to refrigerator in kitchen
**Previous Subtask**: Navigate through kitchen center
**Current Observation:** Agent in kitchen center, refrigerator visible at Front-Left 30°

{{
    "is_completed": true,
    "waypoint": "Kitchen Center - between refrigerator and counter",
    "waypoint_sequence": "Bedroom(✓) → Doorway(✓) → Hallway(✓) → Kitchen Center(Current) → Refrigerator(Goal)",
    "subtask_destination": "refrigerator in kitchen",
    "subtask_instruction": "Turn left 30° to face refrigerator directly, move forward 1.0m to approach refrigerator",
    "subtask_landmark": "refrigerator",
    "completion_criteria": {{
        "Object_Detection": "Refrigerator detected in Front view center",
        "Location": "Refrigerator Area - refrigerator ahead, counter at right",
        "Spatial_relationship": "Destination ahead. Distance < 1.0m. Orange trajectory shows left turn and approach to refrigerator"
    }},
    "is_final_subtask": true,
    "reasoning": "Previous subtask completed: orange trajectory shows entered kitchen center. Current position: kitchen center, refrigerator visible at left 30° (Front-Left 30°, left portion of IMAGE 1). Map: green floor path clear between agent and refrigerator. Global task requires reaching refrigerator (final destination), so next waypoint is refrigerator itself. Action plan: turn left 30° to center refrigerator in Front view, move forward 1.0m to approach (distance < 1.0m confirms arrival)."
}}

## Example 3:
**Global Task**: Enter the bedroom through the doorway
**Previous Subtask**: Approach bedroom doorway
**Current Observation:** Agent facing slightly left of doorway, distance still > 1.0m, trajectory shows movement but orientation misaligned

{{
    "is_completed": false,
    "waypoint": "Hallway - approaching bedroom doorway",
    "waypoint_sequence": "Living Room(✓) → Hallway(Current) → Bedroom Doorway → Bedroom(Goal)",
    "subtask_destination": "bedroom doorway",
    "subtask_instruction": "First turn right 30° to align with doorway center (currently facing left edge), then move forward 0.75m to reach doorway (target distance < 0.5m)",
    "subtask_landmark": "door",
    "completion_criteria": {{
        "Object_Detection": "Door detected in Front view center (not left/right portion)",
        "Location": "Bedroom Doorway - doorway centered ahead",
        "Spatial_relationship": "Destination ahead. Distance < 0.5m. Orange trajectory shows approach with final alignment. Red arrow centered on doorway"
    }},
    "is_final_subtask": false,
    "reasoning": "Previous subtask NOT completed. Current observation analysis: (1) Door detected but in Front-Left portion of IMAGE 1 (not centered) - orientation misaligned by ~30°. (2) Distance still > 1.0m on map (completion requires < 0.5m) - orange trajectory shows progress but insufficient. (3) Local map: red arrow pointing slightly left of doorway. Root cause: narrow hallway caused drift left during movement. Current position: hallway approaching bedroom doorway. Corrective action plan: first turn right 30° to center doorway in Front view, then move forward 0.75m to reach doorway (target distance < 0.5m). Map: clear green path, no black obstacles blocking corrected path."
}}

**Critical Requirements**:
- **Panoramic View Content**: Detect each portion of panoramic view for comprehensive spatial understanding and precise directional descriptions.
- **Verification**: Compare current observations against all three completion_criteria fields (Object_Detection, Location, Spatial_relationship)
- **Planning**: Start all actions from Front view (0°). If subtask completed, plan NEXT waypoint; if not, adjust CURRENT subtask
- **Map**: Use maps to verify trajectory, identify obstacles and plan safe paths for next subtask
- **Landmark Selection**: Choose common furniture items with simple nouns for easy detection (e.g., door, chair, table, bed, cabinet, refrigerator, sofa)
- **Logical Analysis**: Ensure reasoning and output aligns with inputs - All the content must not contain any contradictions.

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