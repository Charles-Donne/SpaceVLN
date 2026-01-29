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
   - **CRITICAL - Entrance vs Room Interior**: When global task says "Wait at entrance" or "Stop at entrance", the goal waypoint is THE ENTRANCE itself, NOT inside the room. Example: "Wait in the entrance to the bedroom" → Goal = "Entrance to bedroom", NOT "Bedroom interior". Plan to stop at the entrance/doorway position.
   - If waypoint direction has obstacles, choose alternative route that bypasses obstacles
3. **Plan navigation instruction**: 
   - **IMPORTANT**: System will AUTO-ROTATE to face your specified next_waypoint_direction → Next waypoint becomes Front (IMAGE 1) after rotation
   - **Write subtask_instruction ASSUMING you are ALREADY FACING the waypoint after rotation**: e.g., "Move forward toward corridor" (NOT "Turn 30° left then move" - rotation already done by system)
   - Describe navigation from Front view perspective after automatic rotation

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
    "task_progress": "<Global task with completed stages marked with ✓, current executing stage marked with (Current), future stages unmarked. CRITICAL: Use (Current) for the stage you are CURRENTLY WORKING ON. When ALL stages are (✓) with no (Current), task is complete. E.g.: 'Turn around(✓) walk through exercise room(Current) into living room. Wait by Table.'>",
    "next_waypoint_direction": "<IMAGE number where next waypoint appears most centered/visible (1-12)>",
    "next_waypoint_destination": "<Next immediate waypoint name>",
    "subtask_instruction": "<Step-by-step navigation instructions from Front view AFTER system auto-rotates to next_waypoint_direction. Example: If you chose IMAGE 2 (Left 30°) as next_waypoint_direction for corridor, write 'Move forward through doorway to reach corridor' - NOT 'Turn 30° left then move'>",
    "next_waypoint_landmark": "<Single landmark to detect (common, e.g. door, table, painting, cabinet)>",
    "completion_criteria": "<Detection: what detected + distance | Location: position/area | Map: space + trajectory>",
    "global_task_finish": <true ONLY when ALL task stages are completed (all marked with ✓, no (Current) remaining) AND you are at the final destination. false otherwise>,
    "reasoning": "<Max 250 words: CRITICAL: FIRST determine current position, THEN mark waypoints/tasks. 1) Current Position Analysis: WHERE AM I? Analyze ALL 12 views - what's NEAR (< 1.0m, large) vs FAR (small/distant)? Check obstacle distances from image labels! Current position = space with NEAR objects surrounding me. Global Map: which room? Synthesis: Current waypoint = TRUE position. CRITICAL: Seeing ≠ At destination! 2) Task Chain Position: Which waypoints passed (✓)? Current (Current)? Future (unmarked)? Only ONE (Current). If all (✓), verify at destination! 3) Next Waypoint & Distance Analysis (CRITICAL - CHECK ALL 12 DIRECTIONS): What's next waypoint? Which IMAGE (1-12)? ANALYZE obstacle distance from EACH direction's image label! List key directions: IMAGE 1: ?m, IMAGE 3: ?m, IMAGE 5: ?m, etc. Decision: If destination FAR (>1m OR small in views) → continue. If destination NEAR (<1m in MULTIPLE directions, surrounding me, visible in several IMAGEs, occupying views) AND cannot get closer (front <0.5m blocked) → arrived. Must be SURROUNDED by destination objects in multiple directions, not just one view! Eliminate: walls (<0.5m). 4) Near-term Plan: Rotate. Subtask. 5) Long-term Plan: Remaining?>"
}}

#Examples:

## Ex1: 
**Global Task**: Turn around walk through the exercise room into the living room. Wait by the Table.
**Current Observation:** IMAGE 1 (Front 0°): Bookshelf visible at distance. IMAGE 5 (Left 120°): Exercise room doorway visible with gym equipment inside. IMAGE 10 (Right 270°): Toilet and washbasin visible

{{
    "current_waypoint": "Restroom - beside exercise room doorway, toilet and washbasin nearby.",
    "waypoint_sequence": "Restroom(Current) → Exercise Room → Living Room → Living Room's Table(Goal)",
    "task_progress": "Turn around(✓) walk through the exercise room(Current) into the living room. Wait by the Table.",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "exercise room",
    "subtask_instruction": "Move forward through doorway to enter the exercise room",
    "next_waypoint_landmark": "exercise equipment",
    "completion_criteria": "Detection: Exercise equipment surrounding < 1.0m | Location: Exercise Room interior | Map: Inside gym area (green space), trajectory entered from restroom",
    "global_task_finish": false,
    "reasoning": "1) Current Position & Waypoint Analysis: Analyze 12 views - IMAGE 10 (Right 270°) shows toilet/washbasin NEAR (< 1.0m, large in view, defining current space). IMAGE 1 (Front) shows bookshelf FAR. IMAGE 5 (Left 120°) shows exercise room doorway FAR with gym equipment inside. Global Map analysis: Red arrow positioned in small restroom space (enclosed green area), exercise room (larger green area) to the left. Local Map: Green circle shows restroom surroundings. Synthesis: Current waypoint = Restroom (starting position). 2) Task Chain Position: waypoint_sequence shows Restroom(Current) - starting point, no waypoints completed yet. Task progress: 'Turn around'(✓ completed). Current stage: 'walk through exercise room'(Current - about to start executing). Remaining: into living room → Wait by Table. 3) Direction Selection & Analysis: Check all 12 directions for obstacles: IMAGE 1 (Front): bookshelf visible, distance ~2.0m (safe). IMAGE 2-4 (Left 30-90°): walls, distance <0.5m (TOO CLOSE - avoid!). IMAGE 5 (Left 120°): exercise room doorway, distance ~1.5m (safe, destination visible). IMAGE 6-8 (Left 150-210°): walls/corners <0.5m (avoid). IMAGE 9-12 (Right): restroom fixtures/walls <1.0m (backwards/blocked). Decision: Choose IMAGE 5 (Left 120°) - exercise room doorway centered, obstacle distance >0.5m safe, not facing wall, forward progress toward goal. 4) Near-term Plan: System rotates to IMAGE 5. After rotation, exercise room doorway becomes Front. Subtask: Move forward through doorway to enter exercise room. 5) Long-term Plan: Enter exercise room → walk through → exit to living room → locate table → stop at table (final goal)."
}}

## Ex2:
**Global Task**: Exit the room and turn left, head toward the kitchen and turn right. Go through the kitchen and out. Wait right at the bathroom.
**Current Observation:** IMAGE 1 (Front 0°): Open space ahead. IMAGE 2 (Left 30°): Bedroom exit doorway visible, corridor with pictures beyond. IMAGE 4 (Left 90°): Wall nearby

{{
    "current_waypoint": "Bedroom - near exit",
    "waypoint_sequence": "Bedroom(Current) → Corridor → Kitchen Entrance → Kitchen  → Bathroom(Goal)",
    "task_progress": "Exit the room(Current) and turn left, head toward the kitchen and turn right. Go through the kitchen and out. Wait right at the bathroom.",
    "next_waypoint_direction": "IMAGE 2 (Left 30°)",
    "next_waypoint_destination": "corridor with pictures",
    "subtask_instruction": "Move forward through doorway to reach corridor",
    "next_waypoint_landmark": "picture",
    "completion_criteria": "Detection: Pictures on corridor wall < 0.5m | Location: Corridor | Map: Corridor space along bedroom exit, trajectory moved forward 1.5m",
    "global_task_finish": false,
    "reasoning": "1) Current Position & Waypoint Analysis: Analyze 12 views - IMAGE 4 (Left 90°) and surrounding views show bedroom walls/furniture NEAR (< 1.0m, large in views, defining current space). IMAGE 2 (Left 30°) shows bedroom exit doorway with corridor FAR (pictures visible beyond doorway). Global Map analysis: Red arrow inside bedroom space (enclosed green area), corridor green area extends to the left. Local Map: Green circle shows bedroom interior. Synthesis: Current waypoint = Bedroom (starting position, near exit). 2) Task Chain Position: waypoint_sequence shows Bedroom(Current) - starting point, no waypoints completed. Task progress: No stages completed yet. Current stage: 'Exit the room'(Current - about to execute). Remaining: turn left → head to kitchen → turn right → go through kitchen → exit → reach bathroom. 3) Direction Selection & Analysis: Analyze all directions: IMAGE 1 (Front): open space, distance ~1.0m (safe but no clear destination). IMAGE 2 (Left 30°): corridor doorway with pictures, distance ~1.2m (safe, destination visible - BEST choice). IMAGE 3-4 (Left 60-90°): bedroom walls, distance <0.5m (TOO CLOSE - facing wall, avoid!). IMAGE 5-8 (Left/Back): bedroom furniture/walls <0.8m (blocked). IMAGE 9-12 (Right): bed/furniture, distance <1.0m (not toward goal). Decision: Choose IMAGE 2 (Left 30°) - corridor doorway centered, obstacle distance 1.2m (safe, >0.5m), pictures visible (landmark), not facing wall, forward to goal. 4) Near-term Plan: System rotates to IMAGE 2. After rotation, corridor doorway becomes Front. Subtask: Move forward through doorway to reach corridor. 5) Long-term Plan: Enter corridor → turn left → navigate to kitchen → walk through kitchen → reach bathroom (final goal)."
}}

**Critical Requirements**:
- **CRITICAL - Position First, Then Mark**: ALWAYS determine TRUE current position FIRST (Step 1), THEN mark waypoint_sequence and task_progress based on that position. If current position matches a previous waypoint (blue circle on map), you went BACKWARDS - must ROLLBACK waypoint markers (✓) and task_progress to match current reality.
- **Accurate Position Awareness (CRITICAL - Surrounded = Arrived)**: Current position = space with NEAR objects (< 1.0m) surrounding me. Arrival judgment: 1) Seeing destination FAR (>1m, small, in one direction only) → NOT arrived, continue. 2) Seeing destination NEAR (<1m) but only in one direction → NOT arrived yet, continue approaching. 3) AT destination: destination visible in MULTIPLE directions (front + sides), obstacle distances <1m in SEVERAL IMAGEs (check image labels!), destination occupying significant view areas, cannot advance (front <0.5m blocked). Must be SURROUNDED by destination, not just seeing it!
- **Task Progress Markers (CRITICAL)**: Completed stages=(✓), Current executing stage=(Current) [only ONE], Future stages=unmarked. When all stages are (✓) with no (Current), set global_task_finish=true.
- **Reasoning Consistency**: Base ALL reasoning on actual visual observations. Ensure current position, task progress, and next action are logically consistent.
- **Entrance vs Room Interior (CRITICAL)**: When task says "Wait at entrance" or "Stop at entrance to [room]", goal is THE ENTRANCE/DOORWAY position itself, NOT inside the room. Don't enter - stop at doorway. Example: "Wait in the entrance to the bedroom" → Stop at bedroom doorway, don't go inside bedroom.
- **12-Direction Analysis (CRITICAL for Part 3 - MANDATORY)**: In reasoning Part 3, MUST analyze EACH direction (IMAGE 1-12) with: what objects visible? obstacle distance for that direction? Then list key directions considered with their distances. Example: "IMAGE 1: corridor, 1.5m (safe). IMAGE 3-4: walls <0.5m (TOO CLOSE-avoid). IMAGE 5: doorway, 2.0m (safe, best)." This analysis is MANDATORY - don't skip!
- **Direction Selection Logic (CRITICAL - Never Face Walls)**: Choose next_waypoint_direction by: 1) Check ALL 12 views' obstacle distances first, 2) Eliminate directions with walls/obstacles <0.5m (unsafe, facing wall), 3) Eliminate backwards directions (toward blue circle waypoints), 4) From remaining safe directions, choose where destination most visible/centered AND obstacle distance >0.5m AND forward progress toward goal.
- **Waypoint Naming (CRITICAL - Include Spatial Context)**: Name waypoints with spatial context, NOT isolated objects! BAD: "Chair" (which chair?). GOOD: "Living room sofa area's coffee table", "Living room near gray couch", "Kitchen dining table". Include: 1) Room/area name, 2) Spatial relationship (near/beside/by), 3) Landmark objects. This prevents stopping at wrong objects (e.g., stopping at random chair instead of living room's specific chair by sofa).
- **Next Waypoint Distance Check (CRITICAL - Analyze ALL 12 Directions)**: Analyze obstacle distances from ALL 12 image labels: IMAGE 1: ?m, IMAGE 2: ?m, ..., IMAGE 12: ?m. Decision: 1) If destination >1m in any view OR small in views → continue. 2) If destination <1m in MULTIPLE directions (surrounding, visible in several IMAGEs) AND occupying views → check if can approach. 3) If cannot (front <0.5m blocked) AND surrounded by destination → arrived. Must be SURROUNDED by destination objects in multiple directions!
- **Waypoint Sequence Logic**: waypoints marked (✓) = completed/passed, Current = current position, unmarked = not yet reached. If you're at a previous location, ROLLBACK markers.
- **Auto-Rotation**: System rotates to waypoint_direction. Write instructions from Front view after rotation.
- **Sequential Navigation**: Follow waypoint_sequence progressively. Don't return to previous waypoints unless backtracked - then adjust markers accordingly.
- **Obstacle Bypass**: If direct path to waypoint blocked (distance < 0.5m), choose alternative direction bypassing obstacles while moving toward waypoint.
- **Global Task Completion (CRITICAL - Must Be SURROUNDED!)**: Set global_task_finish=true when: 1) ALL stages (✓), no (Current), 2) Final destination SURROUNDING me: visible in MULTIPLE IMAGEs (not just front), obstacle distances <1m in multiple directions (check ALL image labels!), destination occupying significant view areas, 3) Cannot advance: front obstacle distance <0.5m (blocked). CRITICAL: Don't stop just seeing destination in one direction! Must be surrounded - destination visible in front AND sides, distances <1m in SEVERAL IMAGEs. DON'T stop if: destination only in 1-2 IMAGEs, destination >1m in any direction, can advance 0.25m (front distance >0.5m).
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
- **CRITICAL - Entrance vs Room Interior**: When global task says "Wait at entrance" or "Stop at entrance", the goal waypoint is THE ENTRANCE itself, NOT inside the room. Example: "Wait in the entrance to the other room" → Goal = "Entrance to other room", NOT "Other room interior". Stop at the entrance/doorway position, don't enter the room.
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
    "task_progress": "<Global task with completed stages marked with ✓, current executing stage marked with (Current), future stages unmarked. CRITICAL: Only ONE stage should have (Current). When ALL stages are (✓) with NO (Current), task is complete. Example: 'Exit bedroom(✓) through hallway(Current) to kitchen. Enter bedroom.' OR when complete: 'Exit bedroom(✓) through hallway(✓) to kitchen(✓). Enter bedroom(✓).'>",
    "next_waypoint_direction": "<IMAGE number where next waypoint appears most centered/visible (1-12)>",
    "next_waypoint_destination": "<Next waypoint in sequence to navigate toward>",
    "subtask_instruction": "<Step-by-step navigation instructions from current position to next waypoint>",
    "next_waypoint_landmark": "<Single landmark name at next waypoint for detection>",
    "completion_criteria": "<Detection: what detected + distance | Location: position/area | Map: space + trajectory>",
    "global_task_finish": <true ONLY when ALL task stages are completed (all marked with ✓, no (Current) remaining) AND you are at the final destination. false otherwise>,
    "reasoning": "<MAX 250 words: CRITICAL: FIRST determine current position, THEN mark waypoints/tasks. 1) Current Position Analysis: WHERE AM I? Analyze ALL 12 views - what's NEAR (< 1.0m, large) vs FAR (small/distant)? Check obstacle distances from image labels! Current position = space with NEAR objects surrounding me. Global Map: which room? Synthesis: Current waypoint = TRUE position. CRITICAL: Seeing ≠ At destination! If at previous waypoint, BACKWARDS - ROLLBACK! 2) Task Chain Position & Historical Waypoint Analysis: CRITICAL - Analyze Global Map blue circles (historical waypoints): Where is each blue circle relative to current red arrow? List: Blue Circle #1 (Kitchen) in IMAGE 7 direction ~2.0m (BACKWARDS), Blue Circle #2 in IMAGE 9 direction ~1.5m (BACKWARDS). These directions = already visited - AVOID! Which waypoints passed (✓)? Current (Current)? Future (unmarked)? Only ONE (Current). ROLLBACK if needed. If all (✓), verify at destination! 3) Next Waypoint & Distance Analysis (CRITICAL - ALL 12 DIRECTIONS): What's next waypoint? Which IMAGE? ANALYZE obstacle distance from EACH direction's image label! List: IMAGE 1: ?m, IMAGE 2: ?m, etc. If destination >1m OR small in views → continue. If destination <1m in MULTIPLE directions (surrounding, visible in several IMAGEs) AND occupying views → check if can approach. If cannot (front <0.5m blocked) AND surrounded by destination → arrived. Must be SURROUNDED in multiple directions, not just one! Eliminate: blue circle directions (backwards), walls <0.5m. 4) Near-term Plan: Rotate. Subtask. 5) Long-term Plan: Remaining?>"
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
    "reasoning": "1) Current Position & Waypoint Analysis: Analyze 12 views - IMAGE 1 (Front), IMAGE 2-6 (Left sides), IMAGE 8-12 (Right sides) show gym equipment NEAR (< 1.0m, large, occupying significant view area). IMAGE 7 (Back 180°) shows restroom FAR (small/distant, separate space). Global Map analysis: Red arrow inside exercise room space (green area), trajectory shows moved from restroom into exercise room. Local Map: Green circle shows gym equipment inside 0.5m. Synthesis: Current waypoint = Exercise Room interior (Waypoint #2). 2) Task Chain Position & Historical Waypoint Analysis: Global Map blue circles - Blue Circle #1 (Restroom) visible in IMAGE 7 direction ~2.0m behind (BACKWARDS - AVOID this direction!). waypoint_sequence: Restroom(✓ passed) → Exercise Room(Current = inside now) → Living Room → Table(Goal). Task progress alignment: 'Turn around'(✓ completed), 'walk through exercise room'(Current - executing now, in middle of room), future: 'into living room. Wait by Table'. One (Current) marker for current stage. 3) Direction Selection & Analysis: Check all 12 directions: IMAGE 1 (Front 0°): open path toward living room, distance ~2.5m (safe, destination direction). IMAGE 2-3 (Left 30-60°): gym equipment, distance ~1.0m (passable but not optimal). IMAGE 4-6 (Left 90-150°): walls/equipment <0.5m (TOO CLOSE - avoid). IMAGE 7 (Back 180°): restroom direction with Blue Circle #1, distance 2.0m (BACKWARDS - already visited, avoid!). IMAGE 8-12 (Right): gym equipment/walls 0.5-1.0m (not toward goal). Decision: Choose IMAGE 1 (Front 0°) - living room path centered, obstacle distance 2.5m (safe, >0.5m), not facing wall, not backwards to blue circle, forward to next waypoint. 4) Near-term Plan: System rotates to IMAGE 1 (already aligned). Subtask: Move forward through exercise room to reach living room entrance. 5) Long-term Plan: Reach living room entrance → enter living room → locate table → navigate to table → stop at table (final goal). 2 waypoints remaining."
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
    "reasoning": "1) Current Position & Waypoint Analysis: Analyze 12 views - IMAGE 1 (Front 0°) shows rug VERY NEAR (< 0.5m, large in view, occupying significant area). IMAGE 10 (Right 270°) shows gray couch NEAR (within 1.0m). IMAGE 7 (Back 180°) shows hallway FAR (completed area). Global Map analysis: Red arrow at rug position in living room, trajectory extends from bedroom → hallway → living room → rug. Local Map: Green circle (0.5m) confirms rug inside immediate surroundings. Synthesis: Current waypoint = Rug (final destination, arrived). 2) Task Chain Position & Historical Waypoint Analysis: Global Map blue circles - Blue Circle #1 (Bedroom Exit) in IMAGE 7 direction ~4.0m (BACKWARDS), Blue Circle #2 (Hallway) in IMAGE 7 direction ~3.0m (BACKWARDS), Blue Circle #3 (Gray Couch) in IMAGE 10 direction ~1.0m (BACKWARDS). All behind - AVOID these directions! waypoint_sequence: Bedroom Exit(✓ passed) → Hallway(✓ passed) → Living Room with Gray Couch(✓ passed) → Rug(Current = AT rug now, final goal). Task progress: ALL stages completed (all marked ✓, no (Current) remaining) - 'Exit bedroom'(✓), 'turn left'(✓), 'walk straight passing gray couch'(✓), 'stop near rug'(✓). All stages finished! 3) Final Destination Check: Global task destination = rug. Current observation: Rug VERY NEAR in Front < 0.5m. All task stages are (✓) with no (Current). Conclusion: AT final destination AND all tasks complete, stop immediately (global_task_finish=true). 4) Near-term Plan: No navigation needed. Already at final destination with all stages completed. STOP to complete task. 5) Long-term Plan: NO remaining tasks. Global navigation mission fully completed."
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
    "reasoning": "1) Current Position & Waypoint Analysis: Analyze 12 views - IMAGE 1 (Front), IMAGE 2-4 (Left), IMAGE 9-12 (Right) show hallway walls NEAR on both sides (< 1.0m, defining current space). IMAGE 5 (Left 120°) shows bedroom doorway FAR (~2.5m away, bed partially visible inside). IMAGE 7 (Back 180°) shows kitchen FAR (completed area). Global Map analysis: Red arrow in hallway corridor, trajectory shows moved from kitchen into hallway. Local Map: Green circle shows hallway surroundings. Synthesis: Current waypoint = Hallway corridor (Waypoint #2). 2) Task Chain Position & Historical Waypoint Analysis: Global Map blue circles - Blue Circle #1 (Kitchen) visible in IMAGE 7 direction ~2.5m behind (BACKWARDS - AVOID!). waypoint_sequence: Kitchen(✓ passed earlier) → Hallway(Current = in hallway NOW) → Bedroom(Goal, unmarked - not reached). Task progress: 'Walk to kitchen'(✓ completed - passed kitchen waypoint), 'through hallway'(Current - executing now, in middle of hallway), 'enter bedroom' (unmarked - future stage). One (Current) marker for current stage being executed. 3) Next Waypoint & Direction: Next immediate waypoint = Bedroom (final destination). IMAGE 5 (Left 120°) shows bedroom doorway most centered with bed visible inside. Verify: obstacle distance > 0.5m (safe), map shows green path to bedroom. IMAGE 7 has Blue Circle #1 (backwards - avoid). Not at final destination yet (bedroom doorway far ~2.5m, need to reach it). 4) Near-term Plan: System rotates to IMAGE 5 (Left 120°). After rotation, bedroom doorway will be Front. Subtask: Move forward through doorway to enter bedroom. Use bed as landmark. 5) Long-term Plan: Enter bedroom (final waypoint) → confirm inside bedroom space → STOP (task complete). 1 waypoint remaining."
}}

## Example 4:
**Global Task**: Walk out of the bedroom through the open door into the hallway. Turn the corner and walk into the dining area. Pass the dining table and walk into the living room area towards the television. Stop near the chair and open sliding doors to outside.
**Previous Subtask**: Navigate through dining area towards living room
**Current Observation:** IMAGE 1 (Front 0°): Dining table edge visible 0.8m. IMAGE 2 (Left 30°): Living room sofa and chair visible FAR away ~3.5m (small in view). IMAGE 5 (Left 120°): Living room entrance opening at distance ~2.8m. IMAGE 7 (Back 180°): Hallway behind ~2.5m (completed area). IMAGE 12 (Right 330°): Dining area wall/furniture NEAR.

{{
    "current_waypoint": "Dining Area - next to dining table, living room visible far left",
    "waypoint_sequence": "Bedroom(✓) → Hallway(✓) → Dining Area(Current) → Living Room with TV → Living Room Chair(Goal)",
    "task_progress": "Walk out of the bedroom(✓) through the open door into the hallway(✓). Turn the corner and walk into the dining area(Current). Pass the dining table and walk into the living room area towards the television. Stop near the chair and open sliding doors to outside.",
    "next_waypoint_direction": "IMAGE 2 (Left 30°)",
    "next_waypoint_destination": "living room chair",
    "subtask_instruction": "Move forward towards living room entrance to approach the chair",
    "next_waypoint_landmark": "living room sofa and chair",
    "completion_criteria": "Detection: Chair visible approaching | Location: Progress from dining area through living room entrance | Map: Trajectory moving towards living room area",
    "global_task_finish": false,
    "reasoning": "1) Current Position & Waypoint Analysis: Analyze 12 views - IMAGE 1 (Front 0°) shows dining table edge NEAR (0.8m). IMAGE 12 (Right 330°) shows dining furniture NEAR. IMAGE 2 (Left 30°) shows living room sofa + chair FAR away (~3.5m, small in view - not reached yet!). IMAGE 5 (Left 120°) shows living room entrance opening FAR (~2.8m). IMAGE 7 (Back 180°) shows hallway FAR (completed). Global Map: Red arrow in dining area near table. Local Map green circle: dining table inside. Synthesis: Current waypoint = Dining Area (near table). CRITICAL: Living room chair visible but FAR (>3m) - SEEING ≠ AT destination! 2) Task Chain Position & Historical Waypoint Analysis: Global Map blue circles - Blue Circle #1 (Bedroom) in IMAGE 8 direction ~5.0m (BACKWARDS), Blue Circle #2 (Hallway) in IMAGE 7 direction ~2.5m (BACKWARDS - AVOID!). waypoint_sequence: Bedroom(✓ passed), Hallway(✓ passed), Dining Area(Current = at dining table now), Living Room + Chair (unmarked - not reached, still far). Task progress: 'bedroom'(✓), 'hallway'(✓), 'into dining area'(Current - just entered), 'Pass dining table'(Current - executing), 'living room towards TV'(unmarked - future), 'chair'(unmarked - final goal far away). 3) Next Waypoint & Distance Analysis: ANALYZE ALL 12 DIRECTIONS - IMAGE 1: dining table 0.8m (current position). IMAGE 2: living room chair ~3.5m (GOAL - FAR, small in view). IMAGE 3-4: walls/dining area. IMAGE 5: living room entrance ~2.8m. IMAGE 6-11: dining area surroundings. IMAGE 12: dining furniture NEAR. IMAGE 7: hallway with Blue Circle #2 backwards (AVOID). Destination check: Chair visible in IMAGE 2 but ~3.5m away (FAR, small), NOT surrounded (only visible in 1 direction), NOT occupying views. Cannot stop - must continue! Eliminate: IMAGE 7 backwards to blue circles. Choose IMAGE 2 direction towards living room entrance. 4) Near-term Plan: Rotate to IMAGE 2 (Left 30°). After rotation: move forward through living room entrance towards chair. Pass through entrance opening first. 5) Long-term Plan: Enter living room → approach TV area → reach chair (<1m, surrounded) → STOP."
}}

## Example 5:
**Global Task**: Walk out of the bedroom through the open door into the hallway. Turn the corner and walk into the dining area. Pass the dining table and walk into the living room area towards the television. Stop near the chair and open sliding doors to outside.
**Previous Subtask**: Navigate to living room near chair
**Current Observation:** IMAGE 1 (Front 0°): Living room chair visible 0.4m, filling view. IMAGE 2 (Left 30°): Chair side visible 0.6m. IMAGE 11 (Right 300°): Chair back visible 0.7m. IMAGE 12 (Right 330°): Sliding doors to outside visible NEAR 0.9m. IMAGE 7 (Back 180°): Living room open area behind ~2.0m.

{{
    "current_waypoint": "Living Room - AT chair (surrounded), sliding doors visible at right",
    "waypoint_sequence": "Bedroom(✓) → Hallway(✓) → Dining Area(✓) → Living Room with TV(✓) → Living Room Chair(Current = Final Navigation Goal)",
    "task_progress": "Walk out of the bedroom(✓) through the open door into the hallway(✓). Turn the corner and walk into the dining area(✓). Pass the dining table and walk into the living room area towards the television(✓). Stop near the chair(✓) and open sliding doors to outside.",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "living room chair",
    "subtask_instruction": "Stop. Already at chair, SURROUNDED by destination - navigation goal reached",
    "next_waypoint_landmark": "chair",
    "completion_criteria": "Detection: Chair <1m in MULTIPLE directions (Front 0.4m, Left 0.6m, Right 0.7m) - SURROUNDED | Location: Living Room - AT chair (final navigation position) | Map: At chair landmark, trajectory ends at navigation goal",
    "global_task_finish": true,
    "reasoning": "1) Current Position & Waypoint Analysis: Analyze 12 views - IMAGE 1 (Front 0°) shows chair VERY NEAR (0.4m, filling view). IMAGE 2 (Left 30°) shows chair side NEAR (0.6m). IMAGE 11 (Right 300°) shows chair back NEAR (0.7m). IMAGE 12 (Right 330°) shows sliding doors NEAR (0.9m). IMAGE 7 (Back 180°) shows living room open space behind. CRITICAL: Chair visible in MULTIPLE IMAGEs (1, 2, 11) with distances ALL <1m - SURROUNDED by destination! Global Map: Red arrow at chair position in living room, trajectory through bedroom → hallway → dining → living room → chair. Local Map green circle: chair inside 0.5m. Synthesis: Current waypoint = Living Room Chair (SURROUNDED, arrived). 2) Task Chain Position & Historical Waypoint Analysis: Global Map blue circles - Blue Circle #1 (Bedroom) in IMAGE 8 direction ~6.0m (BACKWARDS), Blue Circle #2 (Hallway) in IMAGE 7 direction ~4.5m (BACKWARDS), Blue Circle #3 (Dining Area) in IMAGE 7 direction ~3.0m (BACKWARDS), Blue Circle #4 (Living Room TV area) in IMAGE 6 direction ~2.0m (BACKWARDS). All blue circles are behind - AVOID all these directions! waypoint_sequence: Bedroom(✓) → Hallway(✓) → Dining Area(✓) → Living Room with TV(✓) → Living Room Chair(Current = AT chair, SURROUNDED). Task progress: ALL NAVIGATION stages completed - 'bedroom'(✓), 'hallway'(✓), 'dining area'(✓), 'pass dining table'(✓), 'living room towards TV'(✓), 'Stop near chair'(✓). All (✓) because reached final navigation destination. NOTE: 'open sliding doors' = manipulation, not navigation. 3) Final Destination Check: ANALYZE ALL 12 DIRECTIONS - IMAGE 1: chair 0.4m. IMAGE 2: chair side 0.6m. IMAGE 3-10: living room surroundings. IMAGE 11: chair back 0.7m. IMAGE 12: sliding doors 0.9m. Destination (chair) <1m in MULTIPLE directions (IMAGEs 1, 2, 11) - SURROUNDED! Chair occupying significant view areas in multiple IMAGEs. Front blocked by chair 0.4m. Conclusion: SURROUNDED by final destination, cannot advance - AT navigation goal, stop immediately (global_task_finish=true). 4) Near-term Plan: No navigation needed. SURROUNDED by chair (<1m in multiple directions). STOP to complete navigation. 5) Long-term Plan: NO remaining NAVIGATION tasks. Navigation mission completed. Manipulation tasks (doors) handled by other systems."
}}

**Critical Requirements**:
- **CRITICAL - Position First, Then Mark**: ALWAYS determine TRUE current position FIRST (Step 1 of reasoning), THEN mark waypoint_sequence and task_progress. If current position matches a previous waypoint (blue circle), you WENT BACKWARDS - must ROLLBACK all markers to match current reality.
- **Accurate Position Awareness (CRITICAL - Seeing ≠ Arriving)**: Current position = space with NEAR objects (< 1.0m, large in views, inside Local Map green circle). DON'T confuse: 1) Seeing destination FAR away (small in view, >0.5m) with 2) Being AT destination (VERY NEAR < 0.5m, inside green circle). Before marking waypoint as (Current), verify: Is it inside Local Map green circle? Are its objects NEAR and large in view? Can I see it clearly occupying significant area?
- **Entrance vs Room Interior (CRITICAL)**: When task says "Wait at entrance" or "Stop at entrance to [room]", goal is THE ENTRANCE/DOORWAY position itself, NOT inside the room. Don't enter - stop at doorway. Example: "Wait in the entrance to the bedroom" → Goal = "Entrance to bedroom" (doorway), NOT "Bedroom interior". Once at doorway/entrance position, stop immediately (global_task_finish=true).
- **Waypoint Chain Logic (CRITICAL)**: waypoint_sequence (✓) markers MUST match your TRUE current position:
  * Only waypoints you've PASSED THROUGH can be marked (✓)
  * Current position gets (Current) marker - must match NEAR objects in views
  * Future waypoints stay unmarked
  * **If you're at a previous waypoint location, ROLLBACK**: Remove (✓) from waypoints you haven't reached anymore
  * **Example of CORRECT logic**: In Hallway → "Bedroom(✓) → Hallway(Current) → Living Room → Rug(Goal)"
  * **Example of WRONG logic**: In Hallway but marking "Living Room(✓)" - impossible! Haven't reached it yet!
  * **Backtracking example**: If current_waypoint="Bedroom" but waypoint_sequence shows "Bedroom(✓) → Hallway(✓)", you went BACK. Correct to: "Bedroom(Current) → Hallway → ..."
- **Task Progress Consistency (CRITICAL)**: task_progress markers MUST align with waypoint_sequence: Completed stages=(✓), Current stage=(Current) [only ONE], Future stages=unmarked. When all stages are (✓) with no (Current), task complete - set global_task_finish=true! Logic: Current Position → Waypoint Chain (✓/Current/unmarked) → Task Progress (✓/Current/unmarked) must be consistent.
- **Spatial Awareness**: Analyze 12 views + Global Map + waypoint history to determine current position. Show reasoning explicitly in 5-part structure.
- **Reasoning Consistency**: Base ALL reasoning on actual visual observations. Maintain logical consistency: current position → task progress → waypoint status → next action must all align. Don't contradict yourself.
- **Direction Selection (CRITICAL for Part 3 - MANDATORY)**: In reasoning Part 3, MUST analyze EACH direction (IMAGE 1-12) stating: what's visible? obstacle distance? Then list key directions with their distances. Example: "IMAGE 1: living room path, 2.5m. IMAGE 4-6: walls <0.5m (avoid). IMAGE 7: restroom, 2.0m (backwards-avoid)." MANDATORY - don't skip! Then eliminate unsafe directions: walls/obstacles <0.5m (facing wall - dangerous!), blue circles (backwards), orange trajectory (revisit). From remaining safe options, choose: destination visible/likely, distance >0.5m, forward progress.
- **Obstacle Bypass Planning**: If direct path blocked (distance < 0.5m), use Global Map to plan alternative route bypassing black obstacle areas while progressing toward next waypoint.
- **Waypoint Markers**: White circles + boxes show visited areas - avoid backtracking unless necessary, then adjust markers.
- **Auto-Rotation**: System rotates to next_waypoint_direction. Write instructions from Front view.
- **Sequential Navigation**: Follow waypoint_sequence progressively. If you backtracked, adjust sequence and progress markers to reflect reality.
- **Global Task Completion (CRITICAL - Must Be SURROUNDED by Destination!)**: Set global_task_finish=true when: 1) ALL task stages (✓), no (Current), 2) At final destination (SURROUNDED): Check obstacle distances from EACH direction's image label! Destination visible in MULTIPLE IMAGEs (not just one direction), distance <1m in SEVERAL directions (front, left, right, etc.), destination occupying significant view areas. This means you're INSIDE or IMMEDIATELY AT the destination area (surrounded by it). 3) Cannot advance further: front obstacle distance <0.5m (blocked). DON'T stop if: destination only in one direction (not surrounded!), destination >1m away in most directions, small in views (continue approaching!).
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