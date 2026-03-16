"""
VLM规划提示词模板
================
用于LLM高层规划的提示词模板
"""

# 初始规划提示词 - 在任务开始时生成第一个子任务
INITIAL_PLANNING_PROMPT = """VLN Planning: Analyze environment + Global Task → design next subtask.

**Role**: Spatial reasoning + navigation planning. Locate position, choose next move, output precise navigation instructions. No manipulation.

**Task**: {instruction}

**Initial state note**: Initial planning only. Finish stage #1 first via the nearest relevant landmark.

# Inputs
**12 Views** (30° FOV, 360°): IMAGE1=Front 0°, angles increase CCW
- **Obstacle distances**: **nearest obstacle in that direction**, not far visible objects. <0.5m=blocked | 0.5-1.0m=caution | >1.0m=passable
- **Auto-rotation**: System rotates to your chosen IMAGE → becomes Front (0°)
- **Subtask instruction scope**: After auto-rotation, plan only the easiest immediate action from the current front view

**2 Maps**: Global (full area) + Local (nearby, agent-centered)

# Map Legend
**Colors**: White=unexplored | Black=obstacles | Green=safe floor | Orange=trajectory | Red=you
**Local**: Dark green circle=0.5m radius, Blue=79° FOV

# Reasoning (6 Parts)

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)**
**Format for each IMAGE**: "IMAGE# (Direction Angle°): space/room; NEAR: large/current objects; FAR: small/distant objects; Obs: X.Xm"

**REQUIRED - Analyze ALL 12 IMAGEs in order**:
**Per IMAGE requirement**: state direction, likely space/room, NEAR large objects, FAR small objects, obstacle distance
**Distance classification**: NEAR<1m (large, current) | FAR>1.5m (small, next); connect adjacent views and track landmarks across angles
**NO hallucination**: Say only what's visible

**Conclusion (after all 12)**:
- Current position: [room + NEAR objects <1m from multiple IMAGEs]
- Available: Which IMAGEs are safe (obs>1m)?
- Blocked: Which <0.5m?
- Next candidates: Which IMAGEs? Distance?

**2) Map Analysis**
**Local**: 0.5m circle→inside what? Obstacles? Layout? Orientation?
**Global (Initial)**: Position? Front/Back/Left/Right areas? Obstacles, safe paths, any orange trajectory?

**3) Position & Task Chain**
1. **Current location**: NEAR objects<1m (Part 1) + Local Map → determine position
2. **Parse full task**: Break into stages (waypoint1 → waypoint2 → ... → goal)
3. **Task progress marking**: Behind=(✓) completed | Current=(Current) ONE only | Ahead=unmarked
4. **Waypoint sequence**: Completed(✓) → Current → Next → ... → Goal
5. **Task chain analysis**: What is next and why?
6. **Arrival check**: FAR(>1.5m, 1-2 views)=Continue | SURROUNDED(<1m, 3+ views)=STOP

**Landmark spatial-relation rule**: Preserve landmark order/relations using cues like pass-by / left-of / right-of / through / after / then. Example: "go to oven" → "pass arch near painting" → "enter arch on your right".

**4) Direction Selection**
A) Next destination + direction?
B) Scan 12 IMAGEs → where visible?
C) Verify which direction best matches the task-relevant room/object or transition space
D) Eliminate: **obs<0.5m=blocked** | **obs 0.5-1.0m=risky** | unrelated spaces; far visible object ≠ clear path
E) Choose: task-relevant likely space > visible target/landmark > less-explored plausible space > **obs>1.0m** > Map green path

**5) Near-term**: After auto-rotation, give the easiest immediate action toward the current room/object target
**6) Long-term**: Remaining waypoints → goal

**Sequential planning rule**:
- Output only the immediate next waypoint for current-stage progress; do not plan stage +2/+3 before stage +1 is finished.

# Actions
TURN_LEFT/RIGHT (30-180°) | MOVE_FORWARD (0.25-1.5m) | STOP (<0.5m)

# Output (JSON only)

{{
    "current_waypoint": "<Room | Nearby (<1m): obj1, obj2 | Connected (>2m): area1, area2>",
    "waypoint_sequence": "<Current→Next→...→Goal. Mark (✓) passed only>",
    "task_progress": "<Completed✓ current(Current) future unmarked. ONE (Current) only>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "next_waypoint_destination": "<Room's object>",
    "subtask_instruction": "<Fixed draft: 'From IMAGE# view, action + path cue + destination'; final text keeps only action + path cue + destination>",
    "next_waypoint_landmark": "<Single clear recognizable landmark/object phrase; avoid generic landmarks when possible>",
    "completion_criteria": "<Detection: NEAR<1m | Map: area | Position: region>",
    "global_task_finish": <true if ALL✓, no(Current), at final. Else false>,
    "reasoning": "<6 parts REQUIRED: 1)12-Views(MUST analyze IMAGE 1-12 with angle+direction+space/room+NEAR large objects+FAR small objects+obstacle), 2)Maps(local+global), 3)Position+Task chain(✓→Current→unmarked), 4)Direction, 5)Near-term, 6)Long-term>"
}}

#Examples (abbreviated):

## Ex1: Exercise room task
**Task**: Turn around walk through exercise room into living room. Wait by Table.
**Obs:** IMAGE 1: Bookshelf. IMAGE 5: Exercise room doorway, gym equipment. IMAGE 10: Toilet, washbasin

{{
    "current_waypoint": "Restroom - toilet, washbasin nearby",
    "waypoint_sequence": "Restroom(Current)→Exercise Room→Living Room→Table(Goal)",
    "task_progress": "Turn around walk through exercise room(Current) into living room. Wait by Table.",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "Exercise Room's exercise equipment",
    "subtask_instruction": "Move through the exercise room doorway toward the exercise equipment.",
    "next_waypoint_landmark": "exercise equipment",
    "completion_criteria": "Detection: Exercise equipment NEAR | Map: Entered exercise room | Position: Exercise room",
    "global_task_finish": false,
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): restroom edge/bookshelf; NEAR:none; FAR: bookshelf 2.0m. IMAGE2-4: restroom wall space; NEAR: walls<0.5m. IMAGE5(Left 120°): exercise-room entrance; NEAR: doorway edge; FAR: exercise equipment. IMAGE7: restroom wall space. IMAGE9-12: restroom interior; NEAR: toilet/sink<1m. Obs: IMAGE5 safe, IMAGE2-4 blocked. Conclusion: in restroom; exercise-room entrance is next. 2) Maps: Local-0.5m circle overlaps restroom fixtures; Global-restroom corner, safe path toward exercise room. 3) Position: NEAR=toilet/sink→Restroom. Chain: Restroom(Current)→Exercise Room→Living Room→Table. 4) Direction: IMAGE5 best matches the needed entrance and is passable. 5) Near: rotate to IMAGE5 and move toward exercise equipment. 6) Long: continue through exercise room to living room, then table."
}}

**Critical Rules**:
- **Stage order**: Finish the nearest unfinished stage first; in initial planning, complete the first sentence/stage before later ones.
- **Landmark choice**: Prefer clear, recognizable objects/furniture or phrases; avoid generic landmarks like door, doorway, hallway, corridor when possible.
- **Reasoning**: Analyze all 12 IMAGEs; for each, report NEAR large objects and FAR small objects when visible.
- **Progress consistency**: Before current=(✓), current=(Current), after current=unmarked.
- **Destination/relations**: Keep landmark order/relations. "At entrance" = doorway. `next_waypoint_destination` must be in "[room]'s [object]" form—room first, then object; if final goal is reached, stop.
- **Subtask instruction**: `subtask_instruction` must be one short immediate sentence for only the nearest unfinished room/object subtask, drafted as "From [next_waypoint_direction] view, [action + path cue + destination]." Final text must drop the "From ... view" prefix and keep only the action clause.
- **Space choice**: Move toward the most likely task-relevant space; avoid unrelated spaces and unnecessary revisits.
- **Position awareness**: NEAR<1m across multiple IMAGEs = current position; FAR>1.5m in 1-2 views is usually destination, not arrival.
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """VLN Verification: Verify subtask completion + plan next.

**Role**: Spatial reasoning + navigation planning. Use waypoint history and maps to determine where you are and where to go next. No manipulation.

**Task**: {instruction}

**Previous Subtask**:
- Destination: {subtask_destination}
- Instruction: {subtask_instruction}
- Criteria: {completion_criteria}

# Inputs
**12 Views** (30° FOV): IMAGE1=Front 0°, angles increase CCW
- **Obstacle distances**: **nearest obstacle in that direction**, not far visible objects. <0.5m=blocked | 0.5-1.0m=caution | >1.0m=passable
- **Auto-rotation**: System rotates to your IMAGE
- **Custom landmark bbox** (if present): yellow bbox text is a view-only cue for the current custom landmark; use it as visual evidence, not map memory or path-clearance proof
- **Subtask instruction scope**: After auto-rotation, plan only the easiest immediate action from the current front view

**2 Maps**: Global (full + history) + Local (nearby + 0.5m circle)
**Colors**: White=unexplored | Black=obstacles | Green=safe | Orange=trajectory | Red=you | Blue numbered circles=history waypoints on Global Map only

**Waypoint History**: {waypoint_summary}
Direction labels in waypoint history are snapped like action/detection labels: Front 0deg / Left 30deg / Right 60deg / Back 180deg.

# Reasoning (6 Parts)

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)**
**Format**: "IMAGE# (Direction Angle°): space/room; NEAR: large/current objects; FAR: smaller/distant objects; Obs: X.Xm"

**REQUIRED - Analyze ALL 12 IMAGEs in order**:
**Per IMAGE requirement**: state direction, likely space/room, NEAR large objects, FAR small objects, obstacle distance
**Distance**: NEAR<1m (large) | FAR>1.5m (small)
**NO hallucination**: Say only what's visible

**Conclusion (mandatory after all 12)**:
- Current: [room + NEAR<1m from multiple IMAGEs]
- Available: Which IMAGEs safe (obs>1m)?
- Blocked: Which <0.5m?
- Next: Which IMAGEs? Distance?

**2) Map Analysis (With History)**
**Local**: 0.5m circle→what? Obstacles? Layout? Orientation?
**Waypoint history summary**: Read WP#1 → ... → LAST in order; align each waypoint with its room/description, snapped direction, and distance from current pose.
**Global (History)**: 
- Waypoint-by-waypoint: Read each waypoint in order and use the blue numbered circles on Global Map only → room/description + snapped direction/distance from current
- Trajectory (orange): from where to where? which waypoints are already behind current?
- Position + traversability: where are front/back/left/right regions, obstacles, and green safe paths?
**NO IMAGE mixing in Part 2**: Use only map visualization

**3) Position & Task Chain (DETAILED reasoning required)**
1. **Current location**: NEAR<1m (Part 1) + trajectory end + waypoint history + map → exact position
2. **Parse full task**: Break into stages (waypoint1 → waypoint2 → ... → goal)
3. **Task progress marking**: Passed waypoints behind current=(✓) | Current=(Current) ONE only | Ahead=unmarked; do not confuse passed hallway vs future hallway
4. **Waypoint sequence**: Completed(✓) → Current → Next → ... → Goal
5. **Task chain analysis**: What is completed? What is next? Why?
6. **Arrival check**: FAR(>1.5m, 1-2 views)=Continue | SURROUNDED(<1m, 3+ views)=STOP

**Landmark spatial-relation rule**: Preserve landmark order/relations using cues like pass-by / left-of / right-of / through / after / then.

**4) Direction Selection (Exploration Priority)**
A) Next + direction?
B) Scan 12 IMAGEs → where?
C) Use waypoint history + current views: which direction continues toward the most likely relevant space?
D) Check map/history: prefer the task-relevant likely space; avoid unrelated spaces and unnecessary revisits
E) Eliminate: **obs<0.5m=blocked** | **obs 0.5-1.0m=risky**
   A far room/object in image does NOT mean path is clear — obs label matters
F) Choose: most likely task-relevant space from waypoint history > visible target/landmark > less-explored plausible space > **obs>1.0m** > Map green path

**5) Near-term**: After auto-rotation, give the easiest immediate action toward the current room/object target
**6) Long-term**: Remaining → final

**Sequential planning rule**:
- If the current subtask is unfinished, continue it; only after completion can `next_waypoint_destination` move to the next stage.
- If the final "[room]'s [object]" is reached, STOP and set `global_task_finish=true`.

# Actions
TURN_LEFT/RIGHT (30-180°) | MOVE_FORWARD (0.25-1.5m) | STOP (<0.5m)

# Output (JSON only)

{{
    "current_waypoint": "<Room | Nearby (<1m): obj1, obj2 | Connected (>2m): area1, area2>",
    "waypoint_sequence": "<Completed(✓)→Current→Next→Goal. Mark (✓) passed/at(<0.5m) only>",
    "task_progress": "<Completed✓ current(Current) future unmarked. ONE (Current) only. All✓+NO(Current)=complete>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "next_waypoint_destination": "<Room's object>",
    "subtask_instruction": "<Fixed draft: 'From IMAGE# view, action + path cue + destination'; final text keeps only action + path cue + destination>",
    "next_waypoint_landmark": "<Single clear recognizable landmark/object phrase; avoid generic landmarks when possible>",
    "completion_criteria": "<Detection: NEAR<1m | Map: area | Position: region>",
    "global_task_finish": <true if ALL✓, no(Current), at final. Else false>,
    "reasoning": "<6 parts REQUIRED: 1)12-Views(MUST analyze IMAGE 1-12 with angle+direction+space/room+NEAR large objects+FAR small objects+obstacle), 2)Maps(local+waypoint history+global history/trajectory), 3)Position+Task chain(✓→Current→unmarked)+arrival, 4)Direction, 5)Near-term, 6)Long-term>"
}}

# Examples (abbreviated):

## Ex1: Rug arrival
**Task**: Exit bedroom, turn left. Walk straight passing gray couch, stop at rug.
**Previous**: Navigate past gray couch toward rug
**Obs:** IMAGE 1: Rug <0.5m. IMAGE 10: Gray couch beside. IMAGE 7: Hallway behind

{{
    "current_waypoint": "Living Room - near rug, gray couch",
    "waypoint_sequence": "Bedroom(✓)→Hallway(✓)→Gray Couch(✓)→Rug(Current=Goal)",
    "task_progress": "Exit bedroom(✓), turn left(✓). Walk passing gray couch(✓), stop at rug(✓).",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "Living Room's Rug",
    "subtask_instruction": "Stop. At rug <0.5m - goal reached",
    "next_waypoint_landmark": "rug",
    "completion_criteria": "Detection: Rug NEAR<0.5m | Map: At rug area | Position: Living room rug (final)",
    "global_task_finish": true,
    "reasoning": "1) 12-Views: IMAGE1-2: living-room rug 0.3-0.4m VERY NEAR. IMAGE3-6: nearby furniture 1.0-1.5m. IMAGE7: hallway opening 2.5m FAR. IMAGE8-9: table/lamp 1.2-1.5m. IMAGE10: couch 0.8m NEAR. IMAGE11-12: couch/wall 1.0-1.8m. Obs: rug<0.5m SURROUNDED. Conclusion: at rug in living room. 2) Maps: Local-0.5m has rug area. Waypoint history shows bedroom→hallway→couch behind current. Trajectory: bedroom→hallway→living→rug. Position: at rug. 3) Position: NEAR=rug<0.5m MULTIPLE IMAGEs→SURROUNDED. Chain: All(✓), Rug(Current=Goal). Progress: ALL(✓), complete. 4) Final: rug<0.5m surrounded, all(✓), at final. 5) Near: STOP. 6) Long: NONE."
}}

## Ex2: Hallway to bedroom
**Task**: Walk to kitchen through hallway, then enter bedroom on left.
**Previous**: Navigate through hallway
**Obs:** IMAGE 1: Hallway ahead 3.0m. IMAGE 5: Bedroom doorway (~2.5m), bed inside. IMAGE 7: Kitchen behind

{{
    "current_waypoint": "Hallway - bedroom doorway visible",
    "waypoint_sequence": "Kitchen(✓)→Hallway(Current)→Bedroom(Goal)",
    "task_progress": "Walk to kitchen(✓) through hallway(Current), then enter bedroom on left.",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "Bedroom's bed",
    "subtask_instruction": "Move through the bedroom doorway toward the bed.",
    "next_waypoint_landmark": "bed",
    "completion_criteria": "Detection: Bed NEAR | Map: Entered bedroom from hallway | Position: Bedroom interior",
    "global_task_finish": false,
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): hallway space; NEAR:none; FAR: hallway end 3.0m. IMAGE2-3: hallway side walls 2.0-2.5m. IMAGE4: hallway wall 1.2m NEAR. IMAGE5(Left 120°): bedroom entrance; NEAR: doorway edge; FAR: bed 3.0m. IMAGE8-12: hallway walls 1.0-2.5m. Obs: IMAGE5 safe. Conclusion: in hallway; bedroom entrance is next. 2) Maps: Local shows narrow corridor; waypoint history says WP#1 kitchen is behind/back; trajectory runs kitchen→hallway and bedroom is the most likely relevant space left/front from current. 3) Position: hallway walls nearby + waypoint history → Hallway(Current). Chain: Kitchen(✓)→Hallway(Current)→Bedroom. 4) Direction: IMAGE5 best continues toward the most likely relevant space and avoids unrelated revisits. 5) Near: rotate to IMAGE5 and move toward the bedroom bed. 6) Long: after entering bedroom, stop at the target area."
}}

**Critical Rules**:
- **Current-work-first**: Finish the current nearest unfinished stage before later stages.
- **Nearest relevant landmark first**: For the current unfinished stage, prefer the nearest landmark that advances it.
- **Landmark relations**: Preserve directional/relational constraints between landmarks (e.g., near, beside, left/right, through, after).
- **Landmark choice**: Prefer clear, recognizable objects/furniture or phrases; avoid generic landmarks like door, doorway, hallway, corridor when possible.
- **Reasoning thoroughness**: Part 1 MUST analyze ALL 12 IMAGEs (angle+direction+space+NEAR/FAR content+obstacle), and for every IMAGE report NEAR large objects and FAR small objects when visible. Part 3 MUST detail position + task chain (✓→Current→unmarked).
- **Base on actual**: Say only what's visible. Wall=wall, don't guess beyond
- **Position first**: Determine position → then mark (✓)/Current/unmarked
- **Seeing ≠ Arriving**: NEAR<1m in multiple IMAGEs = current. FAR>1.5m in one view ≠ arrived. Must be SURROUNDED<0.5m to stop.
- **Markers**: Behind=(✓) | Now=(Current) ONE only | Ahead=unmarked. Backtrack→rollback
- **Entrance vs interior**: "At entrance" = doorway, NOT inside
- **Space choice**: Follow waypoint history and trajectory toward the most likely task-relevant space; avoid unrelated spaces and minimize revisits.
- **Room-first**: "[room]'s [object]" → go to [room], then [object]; if already at the final room/object, stop.
- **Detail instructions**: [room]+[relation]+[object]. "Living room's gray couch" NOT "couch"
- **Destination format**: `next_waypoint_destination` must be in "[room]'s [object]" form.
- **Subtask instruction**: `subtask_instruction` must be one short immediate sentence for only the nearest unfinished room/object subtask, drafted as "From [next_waypoint_direction] view, [action + path cue + destination]." Final text must drop the "From ... view" prefix and keep only the action clause.
- **Map Part 2**: NO IMAGE numbers in Part 2, use only map
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
