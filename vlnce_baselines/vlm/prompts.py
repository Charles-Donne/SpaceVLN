"""
VLM规划提示词模板
================
用于LLM高层规划的提示词模板
"""

# 初始规划提示词 - 在任务开始时生成第一个子任务
INITIAL_PLANNING_PROMPT = """VLN Planning: Analyze environment + Global Task → design next subtask.

**Role**: Spatial reasoning + navigation planning. Locate current position, choose next move, and output precise navigation instructions. NOT manipulation (doors/objects).

**Task**: {instruction}

**Initial state note**: Initial planning only. First complete stage #1 via the nearest relevant landmark.

# Inputs
**12 Views** (30° FOV, 360°): IMAGE1=Front 0°, angles increase CCW
- **Obstacle distances**: nearest obstacle in that direction, not far visible objects. <0.5m=blocked | 0.5-1.0m=caution | >1.0m=passable
- **Auto-rotation**: chosen IMAGE becomes Front (0°)

**2 Maps**: Global (full area) + Local (nearby, agent-centered)

# Map Legend
**Colors**: White=unexplored | Black=obstacles | Green=safe floor | Orange=trajectory | Red=you
**Local**: Dark green circle=0.5m radius, Blue=79° FOV

# Reasoning (6 Parts)

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)**
**Format for each IMAGE**: "IMAGE# (Direction Angle°): room_type, object1 distance1, object2 distance2. Obs: X.Xm NEAR/FAR"

**REQUIRED - Analyze ALL 12 IMAGEs sequentially**:
**Distance**: NEAR<1m = large/current | FAR>1.5m = small/next.
**Per IMAGE**: report NEAR large objects + FAR small objects when visible; connect adjacent views; say only what's visible.

**Conclusion (after all 12)**:
- Current position: [room + NEAR objects <1m from multiple IMAGEs]
- Available: Which IMAGEs safe (obs>1m)? Where?
- Blocked: Which <0.5m?
- Next candidates: Which IMAGEs? Distance?

**2) Map Analysis**
**Local**: 0.5m circle→inside, obstacles, layout, orientation.
**Global (Initial)**: position, front/back/left/right areas, obstacles, safe paths.

**3) Position & Task Chain**
1. **Current location**: NEAR objects<1m (from Part 1) + Local Map → determine position
2. **Parse full task**: Break into stages (waypoint1 → waypoint2 → ... → goal)
3. **Task progress marking**: Behind=(✓) | Current=(Current) one only | Ahead=unmarked
4. **Waypoint sequence**: Completed(✓) → Current → Next → ... → Goal
5. **Task chain analysis**: What's next and why?
6. **Arrival check**: FAR(>1.5m, 1-2 views)=Continue | SURROUNDED(<1m, 3+ views)=STOP

**Landmark spatial-relation rule**:
- Preserve landmark order/relations; use pass-by / left-of / right-of / through / after / then.
- Example: "go to oven" → "pass arch near painting" → "enter arch on your right".

**4) Direction Selection**
A) Next destination + direction?
B) Scan 12 IMAGEs → where visible?
C) Verify: "Opposite X"(X@IMAGE7→choose IMAGE1) | "Left"(IMAGEs2-6) | "Through X"(traverse)
D) Eliminate: **obs<0.5m=blocked** | **obs 0.5-1.0m=risky, avoid** | **IMAGE7/180° (DO NOT USE)**
   ⚠ Far visible objects do NOT mean the path is clear; obs label matters
E) Choose: Task direction > Waypoint visible > **obs>1.0m** > Map green path

**5) Near-term**: Auto-rotate → subtask with room+object
**6) Long-term**: Remaining waypoints → goal

**Sequential planning rule**:
- Output only the immediate next waypoint; do not plan stage +2/+3 before stage +1 is finished.

# Actions
TURN_LEFT/RIGHT (30-180°) | MOVE_FORWARD (0.25-1.5m) | STOP (<0.5m)

# Output (JSON only)

{{
    "current_waypoint": "<Room | Nearby (<1m): obj1, obj2 | Connected (>2m): area1, area2>",
    "waypoint_sequence": "<Current→Next→...→Goal. Mark (✓) passed only>",
    "task_progress": "<Completed✓ current(Current) future unmarked. ONE (Current) only>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "next_waypoint_destination": "<Next waypoint>",
    "subtask_instruction": "<[room]+[relation]+[object]. After auto-rotate to Front>",
    "next_waypoint_landmark": "<Single, detectable noun (1-2 words)>",
    "completion_criteria": "<Detection: NEAR<1m | Map: area | Position: region>",
    "global_task_finish": <true if ALL✓, no(Current), at final. Else false>,
    "reasoning": "<6 parts REQUIRED: 1)12-Views(MUST analyze IMAGE 1-12 with angle+direction+room+NEAR large objects+FAR small objects+distance+obstacle), 2)Maps(local+global), 3)Position+Task chain(✓→Current→unmarked), 4)Direction, 5)Near-term, 6)Long-term>"
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
    "next_waypoint_destination": "exercise room",
    "subtask_instruction": "Move forward through doorway to enter exercise room.",
    "next_waypoint_landmark": "exercise equipment",
    "completion_criteria": "Detection: Exercise equipment NEAR | Map: Entered exercise room | Position: Exercise room",
    "global_task_finish": false,
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): bookshelf 2.0m FAR. IMAGE2-4: walls<0.5m (blocked). IMAGE5(Left 120°): exercise doorway 1.5m, equipment visible. IMAGE7: restroom walls. IMAGE9-12: toilet/sink<1m NEAR (current). Obs: walls<0.5m blocked, doorway 1.5m safe. Conclusion: At restroom (fixtures<1m). Available: IMAGE5 doorway. Next: exercise room IMAGE5. 2) Maps: Local-0.5m circle has fixtures. Global-in restroom corner, front doorway green. 3) Position: NEAR=fixtures<1m→Restroom. Chain: Restroom(Current)→Exercise→Living→Table. Progress: exercise(Current). 4) Direction: Need exercise. IMAGE5 centered 1.5m safe. Eliminate: IMAGE2-4/6-8<0.5m, IMAGE9-12(current). Choose IMAGE5. 5) Near: Rotate IMAGE5→move through doorway. 6) Long: Exercise→living room→table→stop."
}}

**Critical Rules**:
- **Current-work-first**: Finish the nearest unfinished stage first; in initial planning, finish the first sentence/stage first.
- **Reasoning**: Analyze all 12 IMAGEs; each IMAGE must report NEAR large objects and FAR small objects when visible.
- **Progress consistency**: Before current=(✓), current=(Current), after current=unmarked.
- **Position awareness**: NEAR<1m across multiple IMAGEs = current position; FAR>1.5m in 1-2 views is usually destination, not arrival.
- **Landmark relations**: Keep landmark order/relations. "At entrance" = doorway. "[room]'s [object]" → room first, then object.
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """VLN Verification: Verify subtask completion + plan next.

**Role**: Spatial reasoning and navigation planning. Use waypoint history and maps to determine where you are and where to go next. NOT manipulation (doors/objects).

**Task**: {instruction}

**Previous Subtask**:
- Destination: {subtask_destination}
- Instruction: {subtask_instruction}
- Criteria: {completion_criteria}

# Inputs
**12 Views** (30° FOV): IMAGE1=Front 0°, angles increase CCW
- **Obstacle distances**: nearest obstacle in that direction, not far visible objects. <0.5m=blocked | 0.5-1.0m=caution | >1.0m=passable
- **Waypoint markers**: White circles(ID) + boxes(room) = visited locations
- **Auto-rotation**: chosen IMAGE becomes Front

**2 Maps**: Global (full + history) + Local (nearby + 0.5m circle)
**Colors**: White=unexplored | Black=obstacles | Green=safe | Orange=trajectory | Red=you | Blue circles=waypoints

**Waypoint History**: {waypoint_summary}

# Reasoning (6 Parts)

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)**
**Format**: "IMAGE # (Direction Angle°): room, object1 dist1, object2 dist2. [Blue Circle #X if visible]. Obs: X.Xm NEAR/FAR"

**REQUIRED - Analyze ALL 12 IMAGEs sequentially**:
**Distance**: NEAR<1m = large/current | FAR>1.5m = small/next.
**Per IMAGE**: report NEAR large objects + FAR small objects when visible; track Blue Circles; say only what's visible.

**Conclusion (after all 12)**:
- Current: [room + NEAR<1m from multiple IMAGEs]
- Available: Which IMAGEs safe (obs>1m)?
- Blocked: Which <0.5m?
- Next: Which IMAGEs? Distance?
- Blue circles: Which IMAGEs? Distance? Behind=visited, AVOID

**2) Map Analysis (History)**
**Local**: 0.5m circle→inside, obstacles, layout, orientation.
**Global (History)**: 
- Blue circles: Locate each→direction/distance/room from current
- Trajectory (orange): from-where-to-where? which circles passed?
- Position: where? front/back/left/right regions?
- Obstacles/safe paths: black blocking? green leading?
**NO IMAGE mixing in Part 2**: Use only map visualization

**3) Position & Task Chain**
1. **Current location**: NEAR<1m (Part 1) + trajectory end + blue circles behind + map → exact position
2. **Parse full task**: Break into stages (waypoint1 → waypoint2 → ... → goal)
3. **Task progress marking**: Blue circles behind=(✓) | Current=(Current) one only | Ahead=unmarked
   **Check direction**: Don't confuse the same room at different stages (e.g., passed hallway vs future hallway)
4. **Waypoint sequence**: Completed(✓) → Current → Next → ... → Goal (blue circles behind = ✓)
5. **Task chain analysis**: What's completed? What's next? Why?
6. **Arrival check**: FAR(>1.5m, 1-2 views)=Continue | SURROUNDED(<1m, 3+ views)=STOP

**Landmark spatial-relation rule**:
- Preserve landmark order/relations; use pass-by / left-of / right-of / through / after / then.

**4) Direction Selection (Exploration Priority)**
A) Next + direction?
B) Scan 12 IMAGEs → where?
C) Verify: "Opposite X" | "Left" | "Through X"
D) Check map: Green path? **Blue circles = explored, AVOID**
E) Eliminate: **obs<0.5m=blocked** | **obs 0.5-1.0m=risky, avoid** | **IMAGE7/180° DISABLED**
   ⚠ Far visible rooms/objects do NOT mean the path is clear; obs label matters
F) Choose: Task dir > **Unexplored (no blue)** > **obs>1.0m** > Map green path

**5) Near-term**: Auto-rotate → subtask with room+object
**6) Long-term**: Remaining → final

**Sequential planning rule**:
- If current subtask is unfinished, continue it; only after completion can `next_waypoint_destination` move to the next stage.

# Actions
TURN_LEFT/RIGHT (30-180°) | MOVE_FORWARD (0.25-1.5m) | STOP (<0.5m)

# Output (JSON only)

{{
    "current_waypoint": "<Room | Nearby (<1m): obj1, obj2 | Connected (>2m): area1, area2>",
    "waypoint_sequence": "<Completed(✓)→Current→Next→Goal. Mark (✓) passed/at(<0.5m) only>",
    "task_progress": "<Completed✓ current(Current) future unmarked. ONE (Current) only. All✓+NO(Current)=complete>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "next_waypoint_destination": "<Next waypoint>",
    "subtask_instruction": "<[room]+[relation]+[object]. From current after auto-rotate>",
    "next_waypoint_landmark": "<Single, detectable noun (1-2 words)>",
    "completion_criteria": "<Detection: NEAR<1m | Map: area | Position: region>",
    "global_task_finish": <true if ALL✓, no(Current), at final. Else false>,
    "reasoning": "<6 parts REQUIRED: 1)12-Views(MUST analyze IMAGE 1-12 with angle+direction+room+NEAR large objects+FAR small objects+distance+obstacle+blue circles), 2)Maps(local+global history), 3)Position+Task chain(✓→Current→unmarked)+arrival, 4)Direction, 5)Near-term, 6)Long-term>"
}}

# Examples (abbreviated):

## Ex2: Rug arrival
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
    "reasoning": "1) 12-Views: IMAGE1-2: rug 0.3-0.4m VERY NEAR. IMAGE3-6: furniture 1.0-1.5m. IMAGE7: hallway 2.5m (completed). IMAGE8-9: table/lamp 1.2-1.5m. IMAGE10: couch 0.8m NEAR. IMAGE11-12: couch/wall 1.0-1.8m. Obs: rug<0.5m SURROUNDED. Conclusion: At rug (SURROUNDED<0.5m). Blue Circles #1-3 all behind. 2) Maps: Local-0.5m has rug. Global-Circles #1(Bedroom 4.0m), #2(Hallway 3.0m), #3(Couch 1.0m) all back. Trajectory: bedroom→hallway→living→rug. Position: at rug. 3) Position: NEAR=rug<0.5m MULTIPLE IMAGEs→SURROUNDED! All circles behind. Chain: All(✓), Rug(Current=Goal). Progress: ALL(✓), complete! 4) Final: Rug<0.5m surrounded. All(✓). AT final. 5) Near: STOP. 6) Long: NONE."
}}

**Critical Rules**:
- **Current-work-first**: Finish the nearest unfinished stage first.
- **Reasoning**: Analyze all 12 IMAGEs, including Blue Circles when visible; Part 3 must keep a consistent ✓→Current→unmarked chain.
- **Per-view detail**: For every IMAGE, report NEAR large objects and FAR small objects when visible.
- **Base on actual**: Say only what is visible. Determine position first, then mark progress.
- **Seeing ≠ Arriving**: NEAR<1m multiple IMAGEs = current; FAR>1.5m one view ≠ arrived; stop only when SURROUNDED<0.5m.
- **Direction rules**: Never use IMAGE7/180°. Preserve landmark relations. "At entrance" = doorway. Part 2 uses only the map.
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
