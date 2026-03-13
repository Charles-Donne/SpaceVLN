"""
VLM规划提示词模板
================
用于LLM高层规划的提示词模板
"""

# 初始规划提示词 - 在任务开始时生成第一个子任务
INITIAL_PLANNING_PROMPT = """VLN Planning: Analyze environment + Global Task → design next subtask.

**Role**: Spatial reasoning and navigation planning — you are the cognitive decision-making core. Build a mental spatial model of the environment, determine exactly where you are, decide where to go next, and issue precise navigation instructions. NOT manipulation (doors/objects). End at final waypoint.
Example: "Stop at chair and open doors" → Navigation ends at "chair".

**Task**: {instruction}

**Execution Order (MANDATORY)**:
- Complete the nearest CURRENT unfinished work first.
- Do NOT skip current work and jump to a later subtask.
- If completion of current work is uncertain, continue current work (do not jump).
- Initial planning starts right after reset: begin from the FIRST unfinished task stage.

# Inputs
**12 Views** (30° FOV, 360°): IMAGE 1=Front 0°, angles increase CCW (30°, 60°, ..., 330°)
- **Obstacle distances**: label = **nearest obstacle in that direction** (NOT the distance to far objects visible in the scene). <0.5m=blocked | 0.5-1.0m=caution | >1.0m=passable
- **Auto-rotation**: System rotates to your chosen IMAGE → becomes Front (0°)

**2 Maps**: Global (full area) + Local (nearby, agent-centered)

# Map Legend
**Colors**: White=unexplored | Black=obstacles (AVOID) | Green=safe floor | Orange=trajectory (avoid revisit) | Red arrow=you (points Front) | Red dash=Forward direction

**Global**: Full area, shows history
**Local**: Zoomed, Dark green circle=0.5m radius, Blue=79° FOV

# Reasoning (6 Parts)

**Concise-but-complete style (MANDATORY)**:
- Keep all 6 parts, do not omit any part.
- Use short bullet/checklist style, avoid long prose repetition.
- Prefer compact evidence statements: "IMAGE # → object, distance, obstacle".
- Keep reasoning focused on decisions; no redundant narration.

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)** 
**Format for each IMAGE**: "IMAGE # (Angle°): room, key objects+distance, Obs X.Xm"

**REQUIRED - Analyze ALL 12 IMAGEs sequentially**:
**Distance classification**: NEAR<1m (large, current position) | FAR>1.5m (small, next destination)
**Connect adjacent views**: Group similar IMAGEs, track landmark across angles
**NO hallucination**: Say only what's visible

**Conclusion (mandatory after all 12)**:
- Current position: [room + NEAR objects <1m from multiple IMAGEs]
- Available: Which IMAGEs safe (obs>1m)? Where lead?
- Blocked: Which <0.5m?
- Next candidates: Where? Which IMAGEs? Distance?

**2) Map Analysis**
**Local**: 0.5m circle→what inside? Obstacles? Layout? Orientation? Blue FOV?
**Global (Initial)**: Position where? Front/Back/Left/Right→what areas? Obstacles (black)? Safe paths (green)?

**3) Position & Task Chain (DETAILED reasoning required)**
1. **Current location**: NEAR objects<1m (from Part 1) + Local Map → determine exact position
2. **Parse full task**: Break into stages (waypoint1 → waypoint2 → ... → goal)
3. **Task progress marking** (MUST be consistent):
   - Behind current = (✓) completed
   - Current location = (Current) ONE only
   - Ahead of current = unmarked (not yet reached)
4. **Waypoint sequence**: Write complete chain: Completed(✓) → Current → Next → ... → Goal
5. **Task chain analysis**: What's next step? Which waypoint to navigate to? Why?
6. **Arrival check**: FAR(>1.5m, 1-2 views)=Continue | SURROUNDED(<1m, 3+ views)=STOP

**4) Direction Selection**
A) Next destination + task direction?
B) Scan 12 IMAGEs → where visible?
C) Verify: "Opposite X"(X@IMAGE7→choose IMAGE1) | "Left"(IMAGEs2-6) | "Through X"(traverse)
D) Eliminate: **obs<0.5m=blocked** | **obs 0.5-1.0m=risky, avoid** | **IMAGE7/180° (DO NOT USE)**
   ⚠ Seeing a far object in image does NOT mean path is clear — obs label is what matters
E) Choose: Task direction > Waypoint visible > **obs>1.0m** > Map green path

**5) Near-term**: Auto-rotate → detailed subtask with room+object
**6) Long-term**: Remaining waypoints → goal

**Sequential planning rule (strict)**:
- Output only the immediate next waypoint for current-stage progress.
- Forbidden: planning stage +2/+3 destination while stage +1 is unfinished.

# Actions
TURN_LEFT/RIGHT (30-180°) | MOVE_FORWARD (0.25-1.5m) | STOP (<0.5m)

# Output (JSON only)

{{
    "current_waypoint": "<Room | Nearby (<1m): obj1, obj2 | Connected (>2m): area1, area2>",
    "waypoint_sequence": "<Current→Next→...→Goal. Mark (✓) passed only>",
    "task_progress": "<Completed✓ current(Current) future unmarked. ONE (Current) only>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "next_waypoint_destination": "<Next waypoint>",
    "subtask_instruction": "<DETAILED: [room]+[relation]+[object]. After auto-rotate to Front view>",
    "next_waypoint_landmark": "<Single, detectable noun (1-2 words)>",
    "completion_criteria": "<Detection: NEAR<1m | Map: area | Position: region>",
    "global_task_finish": <true if ALL✓, no(Current), at final. Else false>,
    "reasoning": "<6 parts REQUIRED, concise checklist style: 1)12-Views(all IMAGE1-12), 2)Maps, 3)Position+Task chain, 4)Direction decision, 5)Near-term, 6)Long-term. Clear evidence, no omission, no redundancy>"
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

## Ex2: Corridor task
**Task**: Exit room, turn left, head to kitchen, turn right. Through kitchen, out. Wait at bathroom.
**Obs:** IMAGE 1: Open space. IMAGE 2: Bedroom exit, corridor, pictures. IMAGE 4: Wall

{{
    "current_waypoint": "Bedroom - exit doorway",
    "waypoint_sequence": "Bedroom(Current)→Corridor→Kitchen→Bathroom(Goal)",
    "task_progress": "Exit room(Current), turn left, to kitchen, right. Through kitchen, out. Wait at bathroom.",
    "next_waypoint_direction": "IMAGE 2 (Left 30°)",
    "next_waypoint_destination": "corridor with pictures",
    "subtask_instruction": "Move forward through bedroom exit to corridor.",
    "next_waypoint_landmark": "picture",
    "completion_criteria": "Detection: Pictures NEAR | Map: Bedroom→corridor | Position: Corridor",
    "global_task_finish": false,
    "reasoning": "1) 12-Views: IMAGE1: bedroom space 1.0m. IMAGE2: corridor doorway 1.2m, pictures beyond. IMAGE3-4: walls<0.5m. IMAGE5-8: furniture<0.8m. IMAGE9-12: bed<1m NEAR. Obs: walls<0.5m blocked, corridor 1.2m safe. Conclusion: At bedroom (bed<1m). Available: IMAGE2 safe. Next: corridor IMAGE2. 2) Maps: Local-0.5m has furniture. Global-bedroom near exit, front doorway green. 3) Position: NEAR=bed<1m→Bedroom. Chain: Bedroom(Current)→Corridor→Kitchen→Bathroom. Progress: Exit(Current). 4) Direction: Need corridor. IMAGE2 doorway 1.2m, pictures landmark. Eliminate: IMAGE3-4<0.5m. Choose IMAGE2. 5) Near: Rotate IMAGE2→move to corridor. 6) Long: Corridor→kitchen→bathroom."
}}

**Critical Rules**:
- **Current-work-first**: Must finish current nearest unfinished stage before any later stage.
- **Reasoning thoroughness**: Part 1 MUST analyze ALL 12 IMAGEs with angle+direction+content. Part 3 MUST detail position and complete task chain (✓→Current→unmarked)
- **Initial position**: Determine from NEAR<1m + Local Map 0.5m circle. Mark current=(Current), future=unmarked
- **Position awareness**: NEAR<1m (multiple IMAGEs) = current position ≠ destination FAR>1.5m (1-2 views)
- **Markers**: Behind=(✓) | Now=(Current) ONE only | Ahead=unmarked
- **Task chain consistency**: Before current=(✓), current=(Current), after current=unmarked. Chain determines next direction
- **Entrance vs interior**: "Wait at entrance" = doorway, NOT inside room
- **Room-first strategy**: "[room]'s [object]" → navigate to [room] first, then [object]
- **Auto-rotation**: System rotates to your IMAGE → write from Front view after
- **Detail instructions**: Use [room]+[relation]+[object]. "Living room's gray couch" NOT "couch"
- **IMAGE-angle match**: IMAGE1=0°, IMAGE2=30°, ..., IMAGE7=180°, ..., IMAGE12=330°
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """VLN Verification: Verify subtask completion + plan next.

**Role**: Spatial reasoning and navigation planning — you are the cognitive decision-making core. Build a mental topology of explored space using waypoint history and maps, determine exactly where you are, decide where to go next, and issue precise navigation instructions. NOT manipulation (doors/objects). End at final waypoint.
Example: "Stop at chair and open doors" → Navigation ends at "chair".

**Task**: {instruction}

**Execution Order (MANDATORY)**:
- Complete the nearest CURRENT unfinished work first.
- Do NOT skip current work and directly start a later subtask.
- Verification must decide current subtask completion first; only then move to the next stage.

**Previous Subtask**:
- Destination: {subtask_destination}
- Instruction: {subtask_instruction}
- Criteria: {completion_criteria}

# Inputs
**12 Views** (30° FOV): IMAGE1=Front 0°, angles increase CCW
- **Obstacle distances**: label = **nearest obstacle in that direction** (NOT the distance to far objects visible in the scene). <0.5m=blocked | 0.5-1.0m=caution | >1.0m=passable
- **Waypoint markers**: White circles(ID) + boxes(room) = visited locations
- **Auto-rotation**: System rotates to your IMAGE

**2 Maps**: Global (full + history) + Local (nearby + 0.5m circle)
**Colors**: White=unexplored | Black=obstacles | Green=safe | Orange=trajectory | Red=you | Blue circles=waypoints

**Waypoint History**: {waypoint_summary}
> Each entry shows direction and distance from your **current pose** (Front=0deg, Right=positive, Left=negative). The **last entry is your most recently visited waypoint** — confirms where you came from.

# Reasoning (6 Parts)

**Concise-but-complete style (MANDATORY)**:
- Keep all 6 parts, do not omit any part.
- Use short bullet/checklist style, avoid long prose repetition.
- Prefer compact evidence statements: "IMAGE # → object, distance, obstacle, blue-circle".
- Keep reasoning focused on verification decision + next immediate step.

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)**
**Format**: "IMAGE # (Angle°): room, key objects+distance, [Blue Circle #X if visible], Obs X.Xm"

**REQUIRED - Analyze ALL 12 IMAGEs sequentially**:
**Distance**: NEAR<1m (large) | FAR>1.5m (small)
**Track waypoint markers**: Note which IMAGEs show Blue Circles (visited locations)
**NO hallucination**: Say only what's visible

**Conclusion (mandatory after all 12)**:
- Current: [room + NEAR<1m from multiple IMAGEs]
- Available: Which IMAGEs safe (obs>1m)?
- Blocked: Which <0.5m?
- Next: Where? Which IMAGEs? Distance?
- Blue circles: Which IMAGEs? Distance? Behind=visited, AVOID

**2) Map Analysis (With History)**
**Local**: 0.5m circle→what? Obstacles? Layout? Orientation? Blue FOV?
**Global (History)**: 
- Blue circles: Locate each→direction/distance/room from current
- Trajectory (orange): from-where-to-where? which circles passed?
- Position: where? Spatial structure: front/back/left/right regions?
- Obstacles/safe paths: black blocking? green leading?
**NO IMAGE mixing in Part 2**: Use only map visualization

**3) Position & Task Chain (DETAILED reasoning required)**
1. **Current location**: NEAR<1m (Part 1) + trajectory end + blue circles behind + map → exact position
2. **Parse full task**: Break into stages (waypoint1 → waypoint2 → ... → goal)
3. **Task progress marking** (MUST be consistent):
   - Blue circles behind = (✓) completed, already passed
   - Current location = (Current) ONE only
   - Ahead of current = unmarked (not yet reached)
   **Check direction**: Don't confuse same room at different stages (e.g., passed hallway vs future hallway)
4. **Waypoint sequence**: Completed(✓) → Current → Next → ... → Goal (blue circles behind = ✓)
5. **Task chain analysis**: What's completed? What's next? Which waypoint now? Why?
6. **Arrival check**: FAR(>1.5m, 1-2 views)=Continue | SURROUNDED(<1m, 3+ views)=STOP

**4) Direction Selection (Exploration Priority)**
A) Next + task direction?
B) Scan 12 IMAGEs → where?
C) Verify: "Opposite X" | "Left" | "Through X"
D) Check map: Green path? **Blue circles = explored, AVOID**
E) Eliminate: **obs<0.5m=blocked** | **obs 0.5-1.0m=risky, avoid** | **IMAGE7/180° DISABLED**
   ⚠ Seeing a room/object far in image does NOT mean path is clear — obs label is what matters
F) Choose: Task dir > **Unexplored (no blue)** > **obs>1.0m** > Map green path

**5) Near-term**: Auto-rotate → detailed subtask with room+object
**6) Long-term**: Remaining → final

**Sequential planning rule (strict)**:
- If current subtask is unfinished, next output must continue current subtask.
- Only after current subtask is complete can `next_waypoint_destination` move to the next stage.

# Actions
TURN_LEFT/RIGHT (30-180°) | MOVE_FORWARD (0.25-1.5m) | STOP (<0.5m)

# Output (JSON only)

{{
    "current_waypoint": "<Room | Nearby (<1m): obj1, obj2 | Connected (>2m): area1, area2>",
    "waypoint_sequence": "<Completed(✓)→Current→Next→Goal. Mark (✓) passed/at(<0.5m) only>",
    "task_progress": "<Completed✓ current(Current) future unmarked. ONE (Current) only. All✓+NO(Current)=complete>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "next_waypoint_destination": "<Next waypoint>",
    "subtask_instruction": "<DETAILED: [room]+[relation]+[object]. From current after auto-rotate>",
    "next_waypoint_landmark": "<Single, detectable noun (1-2 words)>",
    "completion_criteria": "<Detection: NEAR<1m | Map: area | Position: region>",
    "global_task_finish": <true if ALL✓, no(Current), at final. Else false>,
    "reasoning": "<6 parts REQUIRED, concise checklist style: 1)12-Views(all IMAGE1-12), 2)Maps+history, 3)Position+Task chain+completion check, 4)Direction decision, 5)Near-term, 6)Long-term. Clear evidence, no omission, no redundancy>"
}}

# Examples (abbreviated):

## Ex1: Exercise room
**Task**: Turn around walk through exercise room into living room. Wait by Table.
**Previous**: Navigate to exercise room
**Obs:** IMAGE 1: Exercise equipment. IMAGE 7: Restroom behind

{{
    "current_waypoint": "Exercise Room - gym equipment",
    "waypoint_sequence": "Restroom(✓)→Exercise Room(Current)→Living Room→Table(Goal)",
    "task_progress": "Turn around walk through exercise room(Current) into living room. Wait by Table.",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "living room",
    "subtask_instruction": "Move forward through exercise room to living room exit",
    "next_waypoint_landmark": "arched doorway",
    "completion_criteria": "Detection: Doorway NEAR | Map: Through exercise→living room | Position: Living room entrance",
    "global_task_finish": false,
    "reasoning": "1) 12-Views: IMAGE1: treadmill 2.5m, doorway 3.0m FAR. IMAGE2-3: bike 1.2m, dumbbells 1.0m. IMAGE4-6: mirror/mats 0.5-0.8m NEAR. IMAGE7: Blue Circle #1 (Restroom) 2.0m BACK. IMAGE8-12: bench/weights 0.6-1.2m NEAR. Obs: equipment<1m sides, doorway 3.0m safe. Conclusion: At exercise (equipment<1m). Available: IMAGE1 doorway. Blue Circle #1 behind. 2) Maps: Local-0.5m has equipment. Global-Blue #1 (Restroom) behind 2.0m. Trajectory: Restroom→exercise. Position: exercise center. Structure: restroom(passed)→exercise(current)→living(ahead). 3) Position: NEAR=equipment<1m→Exercise Room. Circle #1 behind=passed. Chain: Restroom(✓)→Exercise(Current)→Living→Table. Progress: exercise(Current). 4) Direction: Need living. IMAGE1 doorway 3.0m centered safe. IMAGE7 Circle #1 BACK. Eliminate: IMAGE7 (backwards). Choose IMAGE1. 5) Near: Already facing IMAGE1→move forward. 6) Long: Living entrance→find table→stop."
}}

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

## Ex3: Hallway to bedroom
**Task**: Walk to kitchen through hallway, then enter bedroom on left.
**Previous**: Navigate through hallway
**Obs:** IMAGE 1: Hallway ahead 3.0m. IMAGE 5: Bedroom doorway (~2.5m), bed inside. IMAGE 7: Kitchen behind

{{
    "current_waypoint": "Hallway - bedroom doorway visible",
    "waypoint_sequence": "Kitchen(✓)→Hallway(Current)→Bedroom(Goal)",
    "task_progress": "Walk to kitchen(✓) through hallway(Current), then enter bedroom on left.",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "bedroom",
    "subtask_instruction": "Move forward through doorway to enter bedroom",
    "next_waypoint_landmark": "bed",
    "completion_criteria": "Detection: Bed NEAR | Map: Entered bedroom from hallway | Position: Bedroom interior",
    "global_task_finish": false,
    "reasoning": "1) 12-Views: IMAGE1: hallway wall 3.0m. IMAGE2-3: hallway walls 2.0-2.5m. IMAGE4: wall 1.2m NEAR. IMAGE5: bedroom doorway 2.5m FAR, bed 3.0m visible. IMAGE6: wall 1.5m. IMAGE7: Blue Circle #1 (Kitchen) ~2.5m BACK - AVOID. IMAGE8-12: hallway walls 1.0-2.5m. Obs: doorway 2.5m safe. Conclusion: In hallway. Available: IMAGE5 doorway. Blue #1 behind. 2) Maps: Local-0.5m narrow corridor. Global-Blue #1 (Kitchen) behind 2.5m. Trajectory: kitchen→hallway. Position: hallway center. Structure: kitchen(passed)→hallway(current)→bedroom(ahead). 3) Position: NEAR=walls→Hallway. Circle #1 behind=passed. Chain: Kitchen(✓)→Hallway(Current)→Bedroom. Progress: hallway(Current). 4) Direction: Need bedroom. IMAGE5 doorway 2.5m LEFT. IMAGE7 has Circle #1 BACK-NO! Eliminate: IMAGE7 (backward), IMAGE4/walls<1.5m. Choose IMAGE5 LEFT. 5) Near: Rotate IMAGE5→move through doorway. 6) Long: Enter bedroom→stop."
}}

**Critical Rules**:
- **Current-work-first**: Must finish current nearest unfinished stage before planning later stages.
- **Reasoning thoroughness**: Part 1 MUST analyze ALL 12 IMAGEs with angle+direction+content+blue circles. Part 3 MUST detail position and task chain (✓→Current→unmarked, blue behind=✓)
- **Base on actual**: Say only what's visible. Wall=wall, don't guess beyond
- **IMAGE-angle**: IMAGE1=0°, IMAGE2=30°, ..., IMAGE7=180°, ..., IMAGE12=330°
- **Position first**: Determine position → then mark (✓)/Current/unmarked
- **Seeing ≠ Arriving**: NEAR<1m multiple IMAGEs = current. FAR>1.5m one view ≠ arrived. Must be SURROUNDED<0.5m to stop
- **Markers**: Behind=(✓) | Now=(Current) ONE only | Ahead=unmarked. Backtrack→rollback
- **Progress**: Completed✓ | Current(Current) ONE | Future unmarked. ALL✓+NO(Current)=complete→global_task_finish=true
- **Task chain consistency**: Blue circles behind=(✓), current=(Current), ahead=unmarked. Chain analysis determines next direction
- **Entrance vs interior**: "At entrance" = doorway, NOT inside
- **NO IMAGE7/180° EVER**: Turning 180° backward is DISABLED. Goals are ahead. Only use IMAGEs 1-6, 8-12 (Front/Left/Right). If FRONT blocked, go LEFT or RIGHT, NEVER back.
- **Room-first**: "[room]'s [object]" → navigate to [room], then [object]
- **Detail instructions**: [room]+[relation]+[object]. "Living room's gray couch" NOT "couch"
- **Auto-rotation**: System rotates to your IMAGE → write from Front after
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
