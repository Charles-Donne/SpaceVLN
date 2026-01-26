"""
VLM规划提示词模板
================
用于LLM高层规划的提示词模板
"""

# 初始规划提示词 - 在任务开始时生成第一个子任务
INITIAL_PLANNING_PROMPT = """You are a Vision-Language Navigation planning module. Analyze the environment and Global Task to design the next navigation subtask.

# System Role Definition
**IMPORTANT**: You are a NAVIGATION system. Your mission: Follow the Global Task instruction to navigate step-by-step through waypoint sequences, completing each waypoint until reaching the final navigation goal. Use TURN/MOVE_FORWARD/STOP actions. NOT responsible for physical actions (opening doors, picking objects, etc.). Task completes when reaching final navigation waypoint. Example: "Stop near chair and open doors" → Navigation ends at "near chair".

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

1. **Analyze environment**: 
   - **Identify current position**: Focus on NEAR objects (< 1.0m, large in views, occupying significant view area). Use Local Map's dark green circle (0.5m radius) - objects inside define your current location.
   - **Identify next waypoint**: Look for objects in distance (small, far ahead, separate space) - these determine navigation direction, NOT current position.
   - Use 12 directional views + global and local map to distinguish immediate surroundings from distant targets
2. **Determine waypoint direction**: 
   - **CRITICAL**: Choose IMAGE where next waypoint appears **MOST CENTERED** in the view
   - **CRITICAL**: Verify obstacle distance > 0.5m in that direction (safe to navigate)
   - **CRITICAL**: Use Global Map to confirm this direction leads toward waypoint along safe path (green areas, avoiding black obstacles)
   - If waypoint direction has obstacles, choose alternative route that bypasses obstacles
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
    "waypoint_sequence": "<Current Location> → <Next Waypoint> → ... → <Final Waypoints>. Note: Mark (✓) only for waypoints you've passed through.",
    "task_progress": "<Global task with completed parts marked with ✓. CRITICAL: Only mark stages TRULY completed. E.g.: 'Turn around(✓) walk through exercise room into living room. Wait by Table.'>",
    "next_waypoint_direction": "<IMAGE number where next waypoint appears most centered/visible (1-12)>",
    "next_waypoint_destination": "<Next immediate waypoint name>",
    "subtask_instruction": "<Step-by-step navigation instructions starting from Front view>",
    "next_waypoint_landmark": "<Single landmark to detect (common, e.g. door, table, painting, cabinet)>",
    "completion_criteria": "<Detection: what detected + distance | Location: position/area | Map: space + trajectory>",
    "global_task_finish": <true ONLY when final destination of global task is visible in current views (any IMAGE 1-12) and close enough to reach - Global task complete, stop navigating immediately. false otherwise>,
    "reasoning": "<Max 250 words: CRITICAL: Base ALL reasoning on actual observations from 12 views and maps. Maintain logical consistency. 1) Current Position & Waypoint: Where am I ACTUALLY? (Identify NEAR objects < 1.0m in 12 views, use Local Map green circle 0.5m). What is my current waypoint? 2) Next Waypoint & Direction: Based on global task, what is the next waypoint? In which IMAGE (1-12) is it most centered and visible? Why this direction (obstacle distance > 0.5m, map-verified safe path)? 3) Task Progress: Which stages completed (✓)? Which stage am I currently at? (Be honest - only mark truly completed stages) 4) Near-term Plan: How to reach the next waypoint? (Explain your subtask_instruction step-by-step reasoning) 5) Long-term Plan: What remaining tasks after next waypoint? How to reach subsequent waypoints and ultimately complete the global task?>"
}}

#Examples:

## Ex1: 
**Global Task**: Turn around walk through the exercise room into the living room. Wait by the Table.
**Current Observation:** IMAGE 1 (Front 0°): Bookshelf visible at distance. IMAGE 5 (Left 120°): Exercise room doorway visible with gym equipment inside. IMAGE 10 (Right 270°): Toilet and washbasin visible

{{
    "current_waypoint": "Restroom - beside exercise room doorway, toilet and washbasin nearby.",
    "waypoint_sequence": "Restroom(Current) → Exercise Room → Living Room → Living Room's Table(Goal)",
    "task_progress": "Turn around(✓) walk through the exercise room into the living room. Wait by the Table.",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "exercise room",
    "subtask_instruction": "Move forward through doorway to enter the exercise room",
    "next_waypoint_landmark": "exercise equipment",
    "completion_criteria": "Detection: Exercise equipment surrounding < 1.0m | Location: Exercise Room interior | Map: Inside gym area (green space), trajectory entered from restroom",
    "global_task_finish": false,
    "reasoning": "Current observations: IMAGE 5 (Left 120°) shows exercise room doorway with gym equipment visible inside - next waypoint. IMAGE 1 (Front) shows bookshelf. IMAGE 10 (Right) shows toilet/washbasin. Global map shows red arrow in small restroom, exercise room (larger green area) to the left. Current position: In restroom beside exercise room. Global task progress: 'Turn around'(✓ completed) - facing exercise room now. Current stage: About to enter exercise room. Remaining stages: walk through exercise room → enter living room → reach table. Future plan from current to goal: Step 1) Enter exercise room through doorway at Left 120° → Step 2) Walk through exercise room interior → Step 3) Exit exercise room into living room → Step 4) Locate and approach table in living room → Step 5) Stop at table (final goal). Next immediate action: Rotate 120° left to face exercise room doorway (IMAGE 5), then move forward to enter. Using 'exercise equipment' as landmark for verification."
}}

## Ex2:
**Global Task**: Exit the room and turn left, head toward the kitchen and turn right. Go through the kitchen and out. Wait right at the bathroom.
**Current Observation:** IMAGE 1 (Front 0°): Open space ahead. IMAGE 2 (Left 30°): Bedroom exit doorway visible, corridor with pictures beyond. IMAGE 4 (Left 90°): Wall nearby

{{
    "current_waypoint": "Bedroom - near exit",
    "waypoint_sequence": "Bedroom(Current) → Corridor → Kitchen Entrance → Kitchen  → Bathroom(Goal)",
    "task_progress": "Exit the room and turn left, head toward the kitchen and turn right. Go through the kitchen and out. Wait right at the bathroom.",
    "next_waypoint_direction": "IMAGE 2 (Left 30°)",
    "next_waypoint_destination": "corridor with pictures",
    "subtask_instruction": "Move forward through doorway to reach corridor",
    "next_waypoint_landmark": "picture",
    "completion_criteria": "Detection: Pictures on corridor wall < 0.5m | Location: Corridor | Map: Corridor space along bedroom exit, trajectory moved forward 1.5m",
    "global_task_finish": false,
    "reasoning": "Current observations: IMAGE 2 (Left 30°) shows bedroom exit doorway with corridor and pictures beyond. IMAGE 1 (Front) shows open space. Global map shows red arrow in bedroom (enclosed green area), corridor extends left. Current position: Inside bedroom near exit. Global task progress: No stages completed yet - at starting point. Current stage: About to exit bedroom. Remaining stages: exit room → turn left → head to kitchen → turn right → go through kitchen → exit kitchen → reach bathroom. Future plan from current to goal: Step 1) Exit bedroom through doorway into corridor (IMAGE 2, Left 30°) → Step 2) Turn left in corridor → Step 3) Navigate toward kitchen entrance → Step 4) Turn right to enter kitchen → Step 5) Walk through kitchen → Step 6) Exit kitchen → Step 7) Turn right to reach bathroom (final goal). Next immediate action: Rotate 30° left to face corridor doorway (IMAGE 2), move forward 1.5m to enter corridor. Using 'picture' as landmark for corridor verification."
}}

**Critical Requirements**:
- **Accurate Position Awareness**: Determine TRUE current location by analyzing ALL 12 views + maps. Base current_waypoint on where you ACTUALLY are, not where you see in the distance.
- **Task Progress Accuracy**: Only mark stages (✓) that are TRULY completed. Be honest about current progress - don't mark future steps as done.
- **Reasoning Consistency**: Base ALL reasoning on actual visual observations. Ensure current position, task progress, and next action are logically consistent.
- **12-Direction Analysis**: Analyze all 12 views to locate position and next waypoint
- **Direction Selection Logic**: Choose next_waypoint_direction where: 1) Waypoint is MOST CENTERED in view, 2) Obstacle distance > 0.5m, 3) Map shows safe green path toward waypoint
- **Waypoint Sequence Logic**: waypoints marked (✓) = completed/passed, Current = current position, unmarked = not yet reached. Maintain logical consistency.
- **Auto-Rotation**: System rotates to waypoint_direction. Write instructions from Front view after rotation.
- **Sequential Navigation**: Follow waypoint_sequence progressively. Don't return to previous waypoints.
- **Obstacle Bypass**: If direct path to waypoint blocked (distance < 0.5m), choose alternative direction bypassing obstacles while moving toward waypoint.
- **Path Safety**: Avoid obstacles (black areas). Keep centered in paths. Use maps for safe routes.
- **Distance Judgment**: Dark green circle on local map = 0.5m radius (objects inside < 0.5m away)
- **Landmark Priority**: Use objects from Global Task. Prefer specific items (chair, table, bed). Avoid ambiguous terms (door, entrance).
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """You are a Vision-Language Navigation verification and planning module. Verify previous subtask completion and plan the next navigation step.

# System Role Definition
**IMPORTANT**: You are a NAVIGATION system. Your mission: Follow the Global Task instruction to navigate through the planned waypoint chain, completing each waypoint sequentially until reaching the final navigation goal. Use TURN/MOVE_FORWARD/STOP actions. NOT responsible for physical actions (opening doors, manipulating objects, etc.). Task completes when reaching final navigation waypoint. Example: "Go to kitchen and open refrigerator" → Navigation ends at "kitchen near refrigerator".

# Navigation Global Task:
{instruction}

# Previous Subtask Context:
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

**Step 1: Localize Current Position & Verify Waypoint Chain**
- **CRITICAL**: Determine TRUE current location - focus on NEAR objects (< 1.0m, large in views, occupying significant area). FAR objects (small/distant) belong to next spaces, NOT current position.
- **Map assistance**: Local Map green circle (0.5m radius) = immediate surroundings. Objects inside define your current location. Black obstacles outside circle = far away.
- **Don't assume arrival from distance**: Living room visible ahead but you're in hallway → Current = Hallway, NOT Living Room.
- Analyze 12 views + map: What's NEAR you vs. far ahead? Where's red arrow? What's in green circle?
- **Waypoint Chain Logic (CRITICAL)**: Based on your ACTUAL current position, determine where you are in waypoint_sequence:
  * Waypoints BEFORE current position = mark (✓) (you've passed through them)
  * Current position = mark (Current) (where you are NOW - must match NEAR objects)
  * Waypoints AFTER current position = unmarked (haven't reached yet)
  * **Example**: If currently in Hallway, sequence should be "Bedroom(✓) → Hallway(Current) → Living Room → Rug(Goal)" - Living Room and Rug are NOT (✓) because you haven't reached them yet!
- Synthesize: Where am I based on NEAR objects and map position? Which waypoints have I truly passed?

**Step 2: Determine Next Waypoint & Verify Task Progress**
- **Task Progress Check (CRITICAL)**: task_progress MUST align with waypoint_sequence using proper markers:
  * Use (✓) for stages corresponding to COMPLETED waypoints (you've passed through them)
  * Use (Current) for stage corresponding to CURRENT waypoint position (where you are NOW)
  * Leave unmarked for stages corresponding to FUTURE waypoints (haven't reached yet)
  * **Example**: If waypoint_sequence = "Kitchen(✓) → Hallway(Current) → Bedroom(Goal)", then task_progress = "Walk to kitchen(✓) through hallway(Current) enter bedroom"
  * **Logic Chain**: Current Position (Step 1) → Waypoint Chain (✓/Current/unmarked) → Task Progress (✓/Current/unmarked) - All must be consistent!
- **Dynamic Planning**: Infer next waypoint based on global task + current position. Use SPACE names (Bedroom, Hallway, Living Room, Sofa)
- **Intermediate Waypoints**: If final destination not visible, infer intermediate waypoints (e.g., Bedroom → Hallway → Living Room → Sofa). Choose next immediate reachable waypoint
- **Direction Selection**: Find IMAGE (1-12) where next waypoint is MOST CENTERED + obstacle distance > 0.5m + map shows safe green path
- **Convenience Priority**: 1) Centered/aligned, 2) Front path clear, 3) Shortest route. Bypass obstacles if needed
- **Avoid**: Waypoint markers (visited areas) and orange trajectory (no backtracking)

**Step 3: Plan Navigation**
- Choose next_waypoint_direction (IMAGE 1-12) based on: 1) Waypoint most centered/aligned, 2) Front direction has no obstacles after rotation, 3) Map-verified safe path, 4) Most convenient route
- **IMPORTANT**: System will AUTO-ROTATE to face next_waypoint_direction → Next waypoint becomes Front (IMAGE 1) after rotation
- **Write subtask_instruction ASSUMING you are ALREADY FACING the waypoint after rotation**: e.g., "Move forward toward sofa" (NOT "Turn 30° then move" - rotation already done by system)

# Actions
TURN_LEFT/RIGHT (30-180°), MOVE_FORWARD (0.25-1.5m), STOP (<0.5m from goal)

# Output Format (JSON only)

**CRITICAL**: Output ONLY valid JSON. No extra text before or after.
**Word Limits**: reasoning MAX 200 words, subtask_instruction MAX 100 words, others 20-50 words

{{
    "current_waypoint": "<Current Area Type> - <Key Surrounding Landmarks and Relationships>",
    "waypoint_sequence": "<Your DYNAMICALLY INFERRED waypoint chain: Completed(✓) → Current → Next Immediate Waypoint → Intermediate Waypoints → Final Goal. Use SPACE names (e.g., Bedroom, Hallway, Living Room's Sofa). Infer intermediate waypoints if final destination not directly reachable. Mark (✓) only waypoints you've PASSED THROUGH or ARE AT (<0.5m). Example: Bedroom(✓) → Hallway(Current) → Living Room → Sofa(Goal)>",
    "task_progress": "<Global task with completed parts marked with ✓, current stage marked with (Current). CRITICAL: ✓ = truly completed stages, (Current) = stage you are executing NOW, unmarked = future stages. Example: 'Exit bedroom(✓) through hallway(Current) to kitchen. Enter bedroom.' - Only mark ✓ for completed, (Current) for ongoing>",
    "next_waypoint_direction": "<IMAGE number where next waypoint appears most centered/visible (1-12)>",
    "next_waypoint_destination": "<Next waypoint in sequence to navigate toward>",
    "subtask_instruction": "<Step-by-step navigation instructions from current position to next waypoint>",
    "next_waypoint_landmark": "<Single landmark name at next waypoint for detection>",
    "completion_criteria": "<Detection: what detected + distance | Location: position/area | Map: space + trajectory>",
    "global_task_finish": <true ONLY when you have completed ALL waypoints and reached the final destination (visible and close < 0.5m) - This ENDS entire navigation immediately. false otherwise>,
    "reasoning": "<MAX 250 words: CRITICAL: Base ALL reasoning on actual visual observations - maintain logical consistency. LOGIC CHAIN: First determine current position, then waypoint sequence status, then task progress alignment. 1) Current Position & Waypoint: Where am I ACTUALLY? (Identify NEAR objects < 1.0m in 12 views, use Local Map green circle 0.5m). What is my current waypoint (use SPACE name: Bedroom, Hallway, Living Room, etc.)? 2) Waypoint Sequence Status: Based on my current position, which waypoints have I passed (✓)? Which is current (Current)? Which are future (unmarked)? 3) Task Progress Alignment: Based on waypoint sequence, which task stages are completed (✓), current (Current), or future (unmarked)? Ensure task_progress markers align with waypoint_sequence. 4) Next Waypoint & Direction: What is the next immediate reachable waypoint? If final destination not visible, infer intermediate waypoints. In which IMAGE (1-12) is it most centered/aligned? Why this direction (centered, obstacle distance > 0.5m, convenient path, map-verified)? 5) Navigation Plan: System AUTO-ROTATES to next_waypoint_direction. Explain subtask_instruction AFTER rotation - assume facing waypoint, describe movement from Front view. What remaining tasks/waypoints to complete global task? Ensure internal consistency.>"
}}

## Example 1:
**Global Task**: Turn around walk through the exercise room into the living room. Wait by the Table.
**Previous Subtask**: Navigate to exercise room
**Current Observation:** IMAGE 1 (Front 0°): Exercise equipment surrounding. IMAGE 7 (Back 180°): Restroom visible behind

{{
    "current_waypoint": "Exercise Room (entrance area) - gym equipment surrounding, restroom behind",
    "waypoint_sequence": "Restroom(✓) → Exercise Room(Current) → Living Room → Living Room's Table(Goal)",
    "task_progress": "Turn around(✓) walk through the exercise room(✓) into the living room. Wait by the Table.",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "living room",
    "subtask_instruction": "Move forward through the exercise room to reach the living room exit",
    "next_waypoint_landmark": "arched doorway",
    "completion_criteria": "Detection: Arched doorway in Front view | Location: Living Room entrance | Map: Doorway space leading to living room, trajectory moved through exercise room",
    "global_task_finish": false,
    "reasoning": "1) Current Position & Waypoint: NEAR objects in 12 views show gym equipment surrounding me in Front/Sides (IMAGE 1-6, 8-12) - large, < 1.0m, occupying significant view. Local Map green circle confirms gym equipment inside 0.5m radius. Current waypoint: Exercise Room (Waypoint #2). Restroom (Waypoint #1, completed) visible FAR behind in IMAGE 7 - small/distant, separate space. 2) Next Waypoint & Dynamic Reasoning: Global task requires reaching Living Room's Table. From Exercise Room, I cannot directly see the table. Infer intermediate waypoints: Exercise Room → Living Room (enter room first) → Table (locate inside). Next immediate waypoint: Living Room. IMAGE 1 (Front 0°) shows open path ahead through exercise room interior toward living room entrance - living room entrance CENTERED in this view. Choose IMAGE 1: a) Living room path most centered/aligned, b) Front obstacle distance > 0.5m (safe, clear), c) Global Map confirms living room ahead via green path, d) Most convenient/direct route. 3) Task Progress: Stage 'Turn around'(✓ completed) - turned from restroom to face exercise room. Stage 'walk through exercise room' - IN PROGRESS (at entrance, need to continue through). 4) Near-term Plan: Next waypoint (Living Room) in IMAGE 1 (Front 0°) - already facing correct direction, no rotation needed. Subtask_instruction AFTER auto-rotation (already aligned): Move forward straight through exercise room interior. Living room entrance will be in Front view - gym equipment on sides, clear path ahead. Use arched doorway as landmark for living room entrance. 5) Long-term Plan: After reaching Living Room entrance → Enter living room → Scan to locate table → Navigate to table position → Stop at table (final goal). 2-3 waypoints remaining."
}}

## Example 2:
**Global Task**: Exit the bedroom and turn left. Walk straight passing the gray couch and stop near the rug.
**Previous Subtask**: Navigate past gray couch toward rug
**Current Observation:** IMAGE 1 (Front 0°): Rug directly ahead < 0.5m, gray couch visible beside rug. IMAGE 10 (Right 270°): Gray couch right beside. IMAGE 7 (Back 180°): Hallway visible behind

{{
    "current_waypoint": "Living Room - Rug area, standing near rug with gray couch beside",
    "waypoint_sequence": "Bedroom Exit(✓) → Hallway(✓) → Living Room with Gray Couch(✓) → Living Room's Rug(Current = Goal)",
    "task_progress": "Exit the bedroom(✓) and turn left(✓). Walk straight passing the gray couch(✓) and stop near the rug(✓).",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "Living Room's Rug",
    "subtask_instruction": "Stop. Already at rug within 0.5m - goal reached",
    "next_waypoint_landmark": "rug",
    "completion_criteria": "Detection: Rug in Front < 0.5m, gray couch beside | Location: Living Room - Rug area (final goal) | Map: At rug landmark, trajectory ends at goal",
    "global_task_finish": true,
    "reasoning": "1) Current Position & Waypoint: NEAR objects in 12 views show rug directly in Front (IMAGE 1) < 0.5m - large in view, occupying significant area. Gray couch visible right beside at IMAGE 10 - also NEAR, within 1.0m. Local Map green circle (0.5m radius) confirms rug inside immediate surroundings. Currently AT Rug position (final goal waypoint). Hallway visible FAR behind in IMAGE 7 - completed area. Waypoint chain logic: Bedroom(✓ passed) → Hallway(✓ passed) → Living Room(✓ passed, inside now) → Rug(Current = at rug now, final goal). All previous waypoints truly completed. 2) Next Waypoint & Direction: No next waypoint - already at final destination (Rug). IMAGE 1 shows rug directly ahead < 0.5m, confirming arrival. Task progress: All stages completed because I'm AT the final destination. Stage 1 'Exit bedroom'(✓ COMPLETED - passed Bedroom waypoint), Stage 2 'turn left'(✓ COMPLETED - passed Hallway waypoint), Stage 3 'walk straight passing gray couch'(✓ COMPLETED - passed through Living Room), Stage 4 'stop near rug'(✓ COMPLETED - AT rug now). All (✓) because all waypoints reached. 3) Task Progress: ALL stages finished - no (Current) marker because task complete. 4) Near-term Plan: No action needed - already at final destination. Rug visible in Front < 0.5m, gray couch beside confirms correct location. STOP immediately to complete task. 5) Long-term Plan: NO remaining tasks - global task fully completed. All navigation stages finished. Rug (final destination) reached and confirmed via visual + map. Ready to end navigation."
}}

## Example 3:
**Global Task**: Walk to the kitchen through the hallway, then enter the bedroom on your left.
**Previous Subtask**: Navigate through hallway
**Current Observation:** IMAGE 1 (Front 0°): Hallway continues ahead 3.0m. IMAGE 5 (Left 120°): Bedroom doorway visible at distance (~2.5m), bed partially visible inside. IMAGE 7 (Back 180°): Kitchen visible behind

{{
    "current_waypoint": "Hallway - bedroom doorway at left, kitchen behind",
    "waypoint_sequence": "Kitchen(✓) → Hallway(Current) → Bedroom(Goal)",
    "task_progress": "Walk to the kitchen(✓) through the hallway(Current), then enter the bedroom on your left.",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "bedroom",
    "subtask_instruction": "Move forward through the doorway to enter the bedroom",
    "next_waypoint_landmark": "bed",
    "completion_criteria": "Detection: Bed in Front < 1.0m | Location: Bedroom interior | Map: Inside bedroom space, trajectory entered from hallway",
    "global_task_finish": false,
    "reasoning": "1) Current Position & Waypoint: NEAR objects in 12 views show hallway walls on both sides - occupying Front view (IMAGE 1), defining current space. Local Map green circle (0.5m radius) shows hallway corridor surroundings. Current waypoint: Hallway (Waypoint #2). Bedroom doorway visible at Left (IMAGE 5, 120°) - FAR, ~2.5m away, bed partially visible inside = next target, NOT current location. Kitchen visible FAR behind (IMAGE 7) - completed area. Waypoint chain logic: Kitchen(✓ passed through earlier) → Hallway(Current = where I am NOW, in hallway corridor) → Bedroom(unmarked, haven't reached yet). 2) Next Waypoint & Dynamic Reasoning: Global task requires entering Bedroom (final destination). From Hallway, bedroom is visible and directly reachable. Next immediate waypoint: Bedroom. IMAGE 5 (Left 120°) shows bedroom doorway MOST CENTERED/ALIGNED in frame with bed partially visible inside. Choose IMAGE 5: a) Bedroom doorway centered in this view (easiest approach), b) After rotating to IMAGE 5, Front direction will have obstacle distance > 0.5m (safe, clear path through doorway), c) Global Map shows bedroom as green area to the left (safe path verified), d) Most convenient route to bedroom. Task progress logic: Stage 1 'Walk to kitchen'(✓ COMPLETED - Kitchen waypoint passed), Stage 2 'through hallway'(Current - I'm IN Hallway NOW, executing this stage), Stage 3 'enter bedroom' (unmarked - haven't started yet, bedroom not reached). (Current) aligns with Hallway(Current) in waypoint_sequence. 3) Task Progress: Only mark (✓) for completed waypoints. Currently executing 'through hallway'(Current) because I'm in Hallway. Bedroom stage unmarked because not reached yet. 4) Near-term Plan: Next waypoint (Bedroom) is in IMAGE 5 (Left 120°). System will AUTO-ROTATE 120° left to face bedroom doorway first. Subtask_instruction AFTER auto-rotation (assuming already facing doorway in Front view): Move forward straight through doorway to enter bedroom interior. Doorway will be centered in Front view after rotation. Bed will become visible ahead for arrival verification. Use bed as landmark to confirm bedroom entry. 5) Long-term Plan: After entering Bedroom (final waypoint) → Confirm inside bedroom space → STOP (task complete). Only 1 remaining waypoint. Bedroom is final destination - once entered and bed confirmed nearby, global task finished."
}}

## Example 4:
**Global Task**: Walk out of the bedroom through the open door into the hallway. Turn the corner and walk into the dining area. Pass the dining table and walk into the living room area towards the television. Stop near the chair and open sliding doors to outside.
**Previous Subtask**: Navigate to living room near chair
**Current Observation:** IMAGE 1 (Front 0°): Chair visible ahead < 0.5m. IMAGE 12 (Right 330°): Sliding doors to outside visible at right. IMAGE 7 (Back 180°): Dining area visible behind

{{
    "current_waypoint": "Living Room - near chair, sliding doors to outside visible at right",
    "waypoint_sequence": "Bedroom(✓) → Hallway(✓) → Dining Area(✓) → Living Room with TV(✓) → Chair(Current = Final Navigation Goal)",
    "task_progress": "Walk out of the bedroom(✓) through the open door into the hallway(✓). Turn the corner and walk into the dining area(✓). Pass the dining table and walk into the living room area towards the television(✓). Stop near the chair(✓) and open sliding doors to outside.",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "chair",
    "subtask_instruction": "Stop. Already at chair within 0.5m - navigation goal reached",
    "next_waypoint_landmark": "chair",
    "completion_criteria": "Detection: Chair in Front < 0.5m, sliding doors visible at right | Location: Living Room - near chair (final navigation position) | Map: At chair landmark, trajectory ends at navigation goal",
    "global_task_finish": true,
    "reasoning": "1) Current Position & Waypoint: NEAR objects in 12 views show chair directly in Front (IMAGE 1) < 0.5m - large in view, occupying significant area, defining immediate location. Sliding doors visible at Right (IMAGE 12) - also NEAR. Local Map green circle (0.5m radius) confirms chair inside immediate surroundings. Currently AT Chair position (final navigation waypoint). Dining area visible FAR behind in IMAGE 7 - completed area, separate space. 2) Next Waypoint & Direction: No next waypoint - already at final NAVIGATION destination (Chair). IMAGE 1 shows chair directly ahead < 0.5m, confirming arrival. All navigation waypoints completed: Bedroom(✓) → Hallway(✓) → Dining Area(✓) → Living Room with TV(✓) → Chair(Current = Final Navigation Goal). NOTE: 'open sliding doors to outside' is an ACTION task (manipulation), NOT navigation - out of scope. 3) Task Progress: Navigation stages: Stage 1 'Walk out of bedroom'(✓ COMPLETED), Stage 2 'into hallway'(✓ COMPLETED), Stage 3 'Turn corner into dining area'(✓ COMPLETED), Stage 4 'Pass dining table'(✓ COMPLETED), Stage 5 'walk into living room towards TV'(✓ COMPLETED), Stage 6 'Stop near chair'(✓ COMPLETED) - ALL NAVIGATION stages finished. 4) Near-term Plan: No navigation action needed - already at final navigation destination. Chair visible in Front < 0.5m, sliding doors at right confirm correct living room location. STOP immediately to complete navigation task. 5) Long-term Plan: NO remaining NAVIGATION tasks - navigation mission fully completed. As navigation system, responsibility ends at reaching chair position. Manipulation tasks (opening doors) should be handled by other systems/modules. Ready to end navigation and hand over to action/manipulation modules."
}}

**Critical Requirements**:
- **Accurate Position Awareness**: Determine TRUE current location by analyzing ALL 12 views + Global Map. Don't assume arrival at distant waypoints - must be INSIDE or IMMEDIATELY AT (<0.5m) to claim current position there.
- **Waypoint Chain Logic (CRITICAL)**: waypoint_sequence (✓) markers MUST match your TRUE current position:
  * Only waypoints you've PASSED THROUGH can be marked (✓)
  * Current position gets (Current) marker - must match NEAR objects in views
  * Future waypoints stay unmarked
  * **Example of CORRECT logic**: In Hallway → "Bedroom(✓) → Hallway(Current) → Living Room → Rug(Goal)"
  * **Example of WRONG logic**: In Hallway but marking "Living Room(✓)" - impossible! Haven't reached it yet!
- **Task Progress Consistency (CRITICAL)**: task_progress markers MUST align with waypoint_sequence:
  * Use (✓) for stages corresponding to completed waypoints
  * Use (Current) for stage corresponding to current waypoint position
  * Leave unmarked for stages corresponding to future waypoints
  * **Example**: If waypoint_sequence = "Kitchen(✓) → Hallway(Current) → Bedroom(Goal)", then task_progress = "Walk to kitchen(✓) through hallway(Current) enter bedroom"
  * **Logic Chain**: Current Position → Waypoint Chain (✓/Current/unmarked) → Task Progress (✓/Current/unmarked) - All must be consistent!
- **Spatial Awareness**: Analyze 12 views + Global Map + waypoint history to determine current position. Show reasoning explicitly in 5-part structure.
- **Reasoning Consistency**: Base ALL reasoning on actual visual observations. Maintain logical consistency: current position → task progress → waypoint status → next action must all align. Don't contradict yourself.
- **Direction Selection Logic**: Choose next_waypoint_direction where: 1) Next waypoint is MOST CENTERED in that IMAGE, 2) Obstacle distance > 0.5m (safe), 3) Global Map confirms safe green path toward waypoint.
- **Obstacle Bypass Planning**: If direct path blocked (distance < 0.5m), use Global Map to plan alternative route bypassing black obstacle areas while progressing toward next waypoint.
- **Waypoint Markers**: White circles + boxes show visited areas - avoid backtracking
- **Auto-Rotation**: System rotates to next_waypoint_direction. Write instructions from Front view.
- **Sequential Navigation**: Follow waypoint_sequence progressively. Don't return to previous waypoints (marked circles).
- **Global Task Completion**: When you have completed ALL navigation waypoints and reached the final destination (visible in 12 views and close < 0.5m), output global_task_finish=true. This immediately ENDS the entire navigation - YOU make this final decision, no further verification.
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
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        completion_criteria=completion_criteria,
        action_space=action_space,
        detected_landmarks=detected_landmarks,
        waypoint_summary=waypoint_summary
    )