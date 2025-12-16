"""
VLM规划提示词模板
================
用于LLM高层规划的提示词模板
"""

# 初始规划提示词 - 在任务开始时生成第一个子任务
INITIAL_PLANNING_PROMPT = """You are a Vision-Language Navigation planning module. Analyze the environment and design the next navigation subtask.

# Navigation Task
{instruction}

# Visual Observations
4 panoramic views (90° FOV each) + 2 top-down maps:

**IMAGE 5: Global Map** - Full explored area
**IMAGE 6: Local Map** - Nearby region (agent-centered, FOV cone shown)

**Direction Usage**: e.g., "Right View right portion: turn right 120°".
- Determine the destination of the sub-task based on the image content and 
- Determine the location and orientation of the destination using the orientation indicators on the panoramic image.
**Action Origin**: All actions start from Front (IMAGE 1 center)

# Map Interpretation Guide

**Map Orientation**: 
- Top of map = Agent's current Front direction (IMAGE 1)
- Map rotates with agent - front is always up

**Global Map**:
- **White**: Unexplored/unknown areas
- **Black**: Obstacles (walls, furniture, barriers) - **MUST AVOID in planning**
- **Green**: Confirmed floor areas
- **Orange line**: Trajectory from subtask start to current position
- **Red circle with arrow**: Current position, arrow points to Front direction
  
**Local Map** (zoomed view around agent, same color legend as Global Map):
- Shows finer details in immediate vicinity for precise navigation
- **Blue semi-circle**: Agent's current field of view (Front direction visibility range)
  - The opening of the semi-circle indicates Front view direction
  - Objects within this blue region are currently visible in IMAGE 1 (Front View)
- Better for planning nearby movements and obstacle avoidance

**Use Maps for Planning**:
1. **Identify obstacles (black areas)**: Look for walls, furniture, barriers blocking paths
2. **Spatial awareness**: Use global map for overall layout, local map for immediate surroundings

# Your Task

1. **Analyze environment**: Use 4-directional views + global and local map to identify landmarks and obstacles
2. **Plan subtask**: Break down global task into achievable intermediate waypoints
3. **Provide instructions**: Action sequence starting from Front view using concrete landmarks

**Available Actions**: {action_space}

# Output Format (JSON only)

{{
    "waypoint": "<Current Area Type> - <Key Surrounding Landmarks>",
    "waypoint_sequence": "<Current Location> → <Next Waypoint> → ... → <Final Waypoints>",
    "subtask_destination": "<Next immediate waypoint name>",
    "subtask_instruction": "<Step-by-step navigation instructions starting from Front view>",
    "subtask_landmark": "<Single landmark name for map marking>",
    "completion_criteria": {{
        "landmark_detection": "<What landmark should be detected in 4 views>",
        "destination_reached": "<Distance/position requirement>",
        "spatial_relationship": "<Trajectory and orientation check on map>"
    }},
    "is_final_subtask": <true if this is the final destination, false otherwise>,
    "reasoning": "<Brief explanation of analysis, waypoint selection, and action plan>"
}}

## Example 1: Instruction "Turn around and enter the exercise room"
{{
    "waypoint": "Bathroom - beside exercise room door",
    "waypoint_sequence": "Bathroom(Current) → Exercise Room Entrance → Exercise Room Center(Goal)",
    "subtask_destination": "exercise room entrance",
    "subtask_instruction": "Turn left 90° to face the open doorway, then move forward 0.5m to enter exercise room",
    "subtask_landmark": "door",
    "completion_criteria": {{
        "landmark_detection": "exercise room doorway visible in front view after turning",
        "destination_reached": "entered exercise room, distance to doorway < 0.5m",
        "spatial_relationship": "orange trajectory shows 90° left turn and entry into exercise room on map"
    }},
    "is_final_subtask": false,
    "reasoning": "Agent in bathroom with exercise room door visible in left view (IMAGE 2). Map shows green floor path available after turning left, with no black obstacles blocking entry to exercise room. Task requires entering exercise room, so first waypoint is to enter through doorway: turn left 90° to align with doorway, then move forward 0.5m to enter room."
}}

## Example 2: Instruction "Go back to the kitchen and find the microwave"
{{
    "waypoint": "Living Room - near kitchen entrance and chair",
    "waypoint_sequence": "Living Room(Current) → Kitchen Entrance → Microwave(Goal)",
    "subtask_destination": "kitchen entrance",
    "subtask_instruction": "Turn left 120° to face kitchen entrance behind, move forward 1.5m to enter kitchen",
    "subtask_landmark": "door",
    "completion_criteria": {{
        "landmark_detection": "kitchen doorway visible in Front panorama after turning, chair visible at Back 180°",
        "destination_reached": "entered kitchen area, distance to entrance < 0.5m",
        "spatial_relationship": "orange trajectory shows 120° left turn and backward movement toward kitchen on map"
    }},
    "is_final_subtask": false,
    "reasoning": "Agent in living room with kitchen entrance visible at Back-Left 120° (right portion of IMAGE 2). Map shows green floor path clear between current position and kitchen entrance behind agent, with chair obstacle at Back 180° (center of IMAGE 3). Task requires returning to kitchen to find microwave, so first waypoint is kitchen entrance: turn left 120° to face entrance, move forward 1.5m to enter kitchen through doorway."
}}

**Critical Requirements**:
- Use panorama portions for precise directional descriptions (e.g., "Front-Right 30°", "Back-Left 120°")
- Start all actions from Front view (0°)
- Use maps to identify obstacles and plan safe paths
- **Select landmark**: Choose common, visually distinct objects (door, wall, chair, table, bed, cabinet, window) that detection models can easily recognize
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """You are a Vision-Language Navigation verification module. Verify subtask completion and plan the next navigation step.

# Navigation Task
{instruction}

# Previous Subtask Context
**Waypoint Sequence**: {waypoint_sequence}
**Subtask Destination**: {subtask_destination}
**Subtask Instruction**: {subtask_instruction}
**Completion Criteria**: {completion_criteria}

# Visual Observations
4 panoramic views (90° FOV each) + 2 top-down maps:

**IMAGE 5: Global Map** - Full explored area (updated trajectory, waypoints, landmarks)
**IMAGE 6: Local Map** - Nearby region (agent-centered, FOV cone shown)

**Direction Usage**: e.g., "Right View right portion: turn right 120°".
- Determine the destination of the sub-task based on the image content and 
- Determine the location and orientation of the destination using the orientation indicators on the panoramic image.
**Action Origin**: All actions start from Front (IMAGE 1 center)

# Map Interpretation Guide

**Map Orientation**: 
- Top of map = Agent's current Front direction (IMAGE 1)
- Map rotates with agent - front is always up

**Global Map**:
- **White**: Unexplored/unknown areas
- **Black**: Obstacles (walls, furniture, barriers) - **MUST AVOID in next planning**
- **Green**: Confirmed floor areas
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

**Use Maps for Planning**:
1. **Verify previous trajectory**: Check orange line shows expected movement
2. **Identify obstacles (black areas)**: Look for walls, furniture, barriers blocking future paths
3. **Spatial awareness**: Use global map for overall layout, local map for immediate surroundings

**Note**: Each waypoint shows ID and location description. Waypoint markers appear as **dark red circles with white numbers** on the Global Map, marking past observation positions.

# Your Task

1. **Verify completion**: Compare current observations with completion_criteria (landmark detection, destination arrival, trajectory/orientation)
2. **Make decision**: 
   - **is_completed = true**: Subtask finished → plan NEXT waypoint
   - **is_completed = false**: Not finished → continue SAME subtask
3. **Plan next step**: If completed, update waypoint_sequence and define new subtask; if not, adjust current subtask

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
        "landmark_detection": "<What landmark should be detected in views>",
        "destination_reached": "<Distance/position requirement>",
        "spatial_relationship": "<Trajectory and orientation check on map>"
    }},
    "is_final_subtask": <true if next destination is final goal, false otherwise>,
    "reasoning": "<Brief explanation of completion verification, progress, and next plan>"
}}

## Example 1: Instruction "Walk through hallway to kitchen entrance"
**Previous Subtask**: Enter bedroom doorway
**Current Observation**: Agent at doorway, door detected, trajectory shows entry completed

{{
    "is_completed": true,
    "waypoint": "Doorway - between bedroom and hallway",
    "waypoint_sequence": "Bedroom(✓) → Doorway(Current) → Hallway → Kitchen Entrance(Goal)",
    "subtask_destination": "hallway midpoint",
    "subtask_instruction": "Move forward 1.5m through hallway center toward kitchen entrance",
    "subtask_landmark": "wall",
    "completion_criteria": {{
        "landmark_detection": "hallway walls visible on both sides in views",
        "destination_reached": "progressed through hallway, closer to kitchen entrance",
        "spatial_relationship": "orange trajectory extends through hallway center on map toward kitchen"
    }},
    "is_final_subtask": false,
    "reasoning": "Previous subtask completed: door detected in multiple views, orange trajectory confirms reached doorway on map. Now at doorway with hallway extending ahead in front view (IMAGE 1). Map shows green floor path clear straight through hallway center with black walls on both sides, no obstacles blocking path. Task requires reaching kitchen entrance at hallway end, so next waypoint is to progress through hallway: move forward 1.5m straight through hallway center."
}}

## Example 2: Instruction "Turn around and navigate to refrigerator in kitchen"
**Previous Subtask**: Navigate through kitchen center
**Current Observation**: Agent in kitchen center, refrigerator visible at Front-Left 30°

{{
    "is_completed": true,
    "waypoint": "Kitchen Center - between refrigerator and counter",
    "waypoint_sequence": "Bedroom(✓) → Doorway(✓) → Hallway(✓) → Kitchen Center(Current) → Refrigerator(Goal)",
    "subtask_destination": "refrigerator in kitchen",
    "subtask_instruction": "Turn left 30° to face refrigerator directly, move forward 1.0m to approach refrigerator",
    "subtask_landmark": "refrigerator",
    "completion_criteria": {{
        "landmark_detection": "refrigerator visible in Front panorama center",
        "destination_reached": "reached refrigerator area, distance to refrigerator < 1.0m",
        "spatial_relationship": "orange trajectory shows left turn and approach to refrigerator on map"
    }},
    "is_final_subtask": true,
    "reasoning": "Previous subtask completed: orange trajectory shows entered kitchen center on map. Refrigerator visible at Front-Left 30° (left portion of IMAGE 1), with green floor path clear between agent and refrigerator on map. Task requires reaching refrigerator (final destination), so next waypoint is refrigerator itself: turn left 30° to align front view with refrigerator, move forward 1.0m."
}}

## Example 3: Instruction "Enter the bedroom through the doorway"
**Previous Subtask**: Approach bedroom doorway
**Current Observation**: Agent facing slightly left of doorway, distance still > 1.0m, trajectory shows movement but orientation misaligned

{{
    "is_completed": false,
    "waypoint": "Hallway - approaching bedroom doorway",
    "waypoint_sequence": "Living Room(✓) → Hallway(Current) → Bedroom Doorway → Bedroom(Goal)",
    "subtask_destination": "bedroom doorway",
    "subtask_instruction": "First turn right 30° to align with doorway center (currently facing left edge), then move forward 0.75m to reach doorway (target distance < 0.5m)",
    "subtask_landmark": "door",
    "completion_criteria": {{
        "landmark_detection": "bedroom doorway visible and centered in Front view (not left/right portion)",
        "destination_reached": "distance to doorway < 0.5m, ready to enter",
        "spatial_relationship": "orange trajectory shows approach with final alignment, red arrow centered on doorway on map"
    }},
    "is_final_subtask": false,
    "reasoning": "Previous subtask NOT completed: Analysis shows three issues: (1) Doorway detected but in Front-Left portion of IMAGE 1 (not centered), indicating orientation misalignment by ~30°. (2) Map shows distance still > 1.0m (completion criteria requires < 0.5m), orange trajectory shows progress but insufficient. (3) Local map shows red arrow pointing slightly left of doorway entrance. Root cause: Initial approach instruction didn't account for narrow hallway causing drift left during movement. Corrective strategy: Two-step adjustment - first realign orientation (turn right 30° to center doorway in front view), then approach remaining distance (0.75m forward). Map confirms clear green path with no black obstacles blocking corrected path."
}}

**Critical Requirements**:
- Use panorama portions for precise directional descriptions (e.g., "Front-Left 30°", "Back 180°", "Back-Right 210°")
- Verify **completion_criteria** 3 checks: (1) landmark detected, (2) destination arrived, (3) trajectory/orientation matches
- Analyze all 4 panoramas and Maps for complete 360° understanding
- Start all actions from Front view (0°)
- Use maps to identify obstacles and plan safe paths
- **Select landmark**: Choose common, visually distinct objects (door, wall, chair, table, bed, cabinet, window) that detection models can easily recognize

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