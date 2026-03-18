"""
VLM规划提示词模板
================
用于LLM高层规划的提示词模板
"""

# 初始规划提示词 - 在任务开始时生成第一个子任务
INITIAL_PLANNING_PROMPT = """**Role**: You are a VLN planning module. Analyze the environment and task, determine position, choose the next move, and output precise navigation instructions. No manipulation.

**Task**: {instruction}

**Initial state**: You are at the task start. From your current location, strictly follow the Task from the beginning and complete the first subtask.

# Inputs
**12 Views** (sampled every 30° around 360°; each RGB view HFOV is about 79°): IMAGE1=Front 0°, angles increase CCW
- **Obstacle distance**: nearest obstacle only. <0.5m=blocked | 0.5-1.0m=caution | >1.0m=passable
- **Custom landmark bbox** (if present): current-view cue only; use shown name + distance/angle only as room/object evidence, not map memory or path-clearance proof
**2 Maps**: Global (full area) + Local (nearby, agent-centered)
**Map colors**: White=unexplored | Black=obstacles | Green=safe floor | Orange=trajectory | Red=you | Local: dark green circle=0.5m, blue=79° FOV

# Reasoning (6 Parts)

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)**
**Format for each IMAGE**: "IMAGE# (Direction Angle°): [space/room] + visible evidence". Mention only what is actually visible and useful, such as NEAR objects, FAR objects, landmark name + shown distance/angle, obstacle distance.

**REQUIRED - Analyze ALL 12 IMAGEs in order**:
**Per IMAGE requirement**: direction, likely space/room, and only the visible evidence that exists in that IMAGE
**Distance classification**: NEAR<1m (large/current) | FAR>1.5m (small/distant); connect adjacent views and track landmarks across angles
**NO hallucination**: Say only what is visible. If an IMAGE only shows a near wall, say that only. Do not write `none`, do not fill empty slots, and do not invent landmarks / FAR objects / waypoint cues.

**Conclusion (after all 12)**:
- Current position: [room + NEAR objects <1m from multiple IMAGEs]
- Available: Which IMAGEs are safe (obs>1m)?
- Blocked: Which IMAGEs are <0.5m?
- Next candidates: Which IMAGEs best continue the task?

**2) Map Analysis**
**Local**: 0.5m circle→inside what? Obstacles? Layout? Orientation?
**Global (Initial)**: Position? Front/Back/Left/Right areas? Obstacles, safe paths, any orange trajectory?

**3) Position & Task Chain**
1. **Current location**: NEAR objects<1m (Part 1) + Local Map → determine position
2. **Parse full task**: Break into stages (waypoint1 → waypoint2 → ... → goal)
3. **Task progress marking**: Behind=(✓) completed | Current=(Current) ONE only | Ahead=unmarked
4. **Waypoint sequence**: Completed(✓) → Current → Next → ... → Goal
5. **Task chain analysis**: What is next and why?
6. **Arrival check**: Room first, then target object in that room. Wrong room or FAR(>1.5m, 1-2 views)=Continue | Correct room + target object within ~1m=STOP

**Landmark spatial-relation rule**: Preserve landmark order/relations using cues like pass-by / left-of / right-of / through / after / then. Example: "go to oven" → "pass arch near painting" → "enter arch on your right".

**4) Direction Selection**
A) Next room/object target + direction?
B) Scan 12 IMAGEs → where is the target or its transition space?
C) Verify which direction best matches the task-relevant room/object or transition space
D) Eliminate: **obs<0.5m=blocked** | **obs 0.5-1.0m=risky** | unrelated spaces; far visible object ≠ clear path
E) Choose: most likely task-relevant space/object > visible target/landmark > plausible continuation > **obs>1.0m** > Map green path

**5) Near-term**: After auto-rotation, give the easiest immediate action toward the current room/object target; if already in the correct room and the target object is within ~1m, STOP
**6) Long-term**: Remaining waypoints → goal

**Sequential planning rule**:
- Output only the immediate next waypoint for current-stage progress; do not plan stage +2/+3 before stage +1 is finished.

# Output (JSON only)

{{
    "reasoning": "<6 parts REQUIRED, concise: 1)12-Views(IMAGE 1-12 in order; mention only visible evidence, omit empty items), 2)Maps(local+global), 3)Position+Task chain(✓→Current→unmarked), 4)Direction, 5)Near-term, 6)Long-term>",
    "current_waypoint": "<Room | Nearby (<1m): obj1, obj2 | Connected (>2m): area1, area2>",
    "waypoint_sequence": "<Current→Next→...→Goal. Mark (✓) passed only>",
    "task_progress": "<Completed✓ current(Current) future unmarked. ONE (Current) only>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "next_waypoint_destination": "<Room's object; if the subtask target is the landmark itself, explicitly include that landmark word>",
    "subtask_instruction": "<Exactly one short sentence: 'From [next_waypoint_direction] view, start, [action + optional pass-by/path cue + destination]'>",
    "next_waypoint_landmark": "<Single clear recognizable landmark/object phrase; NEVER use door/doorway/hallway/corridor>",
    "global_task_finish": <true if the final room-object target is reached and the target object is within ~1m. Else false>
}}

#Examples (abbreviated):

## Ex1: Exercise room task
**Task**: Turn around walk through exercise room into living room. Wait by Table.
**Obs:** IMAGE 1: Bookshelf. IMAGE 5: Exercise room doorway, gym equipment. IMAGE 10: Toilet, washbasin

{{
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): restroom edge, bookshelf FAR. IMAGE2-4: restroom walls very near, blocked. IMAGE5(Left 120°): exercise-room entrance, exercise equipment FAR, passable. IMAGE9-12: toilet and sink near. Conclusion: currently in restroom; IMAGE5 best continues the task. 2) Maps: Local-0.5m circle overlaps restroom fixtures; Global-restroom corner with safe path toward exercise room. 3) Position: toilet/sink near across multiple views + Local Map -> Restroom(Current). Chain: Restroom(Current)→Exercise Room→Living Room→Table. 4) Direction: IMAGE5 best matches the needed transition and is passable. 5) Near: rotate to IMAGE5 and move toward exercise equipment. 6) Long: continue through exercise room to living room, then table.",
    "current_waypoint": "Restroom - toilet, washbasin nearby",
    "waypoint_sequence": "Restroom(Current)→Exercise Room→Living Room→Table(Goal)",
    "task_progress": "Turn around walk through exercise room(Current) into living room. Wait by Table.",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "Exercise Room's exercise equipment",
    "subtask_instruction": "From IMAGE 5 (Left 120deg) view, start, move toward the exercise room's exercise equipment.",
    "next_waypoint_landmark": "exercise equipment",
    "global_task_finish": false
}}

**Critical Rules**:
- **Planning priority**: Finish the nearest unfinished stage first; in initial planning, finish the first stage before later ones. Follow current views and waypoint history toward the most likely task-relevant space/object, and preserve landmark order/relations.
- **Reasoning discipline**: Part 1 must cover all 12 IMAGEs in order, but keep it evidence-only and concise: mention only what is actually visible/relevant in each IMAGE, omit empty fields, and never write fake filler such as `none`. Part 3 must still detail position + task chain (✓→Current→unmarked). NEAR<1m across multiple IMAGEs = current position; FAR>1.5m in 1-2 views is destination evidence, not arrival.
- **Progress and arrival**: Before current=(✓), current=(Current), after current=unmarked. Judge arrival in two steps: confirm the room/space, then confirm the room's target object is within ~1m; do not STOP before both hold. If the current room-object target is already within ~1m, STOP immediately.
- **Output constraints**: Use a single common room/space type, remove modifiers, and normalize corridor-like wording to `hallway`. `next_waypoint_landmark` must be a clear recognizable object/furniture phrase, never door/doorway/hallway/corridor. `next_waypoint_destination` must be "[room]'s [object]"; if the subtask target is the same landmark, explicitly include that landmark word in `next_waypoint_destination`. "At entrance" means doorway. `subtask_instruction` must be one short immediate sentence for only the nearest unfinished room/object subtask, in the fixed form "From [next_waypoint_direction] view, start, [action + optional pass-by/path cue + destination]." Use direct verbs such as move/go/walk/enter/pass/follow/cross/approach/continue/head/climb/ascend/descend/stop. The action module automatically drops the "From ... view, start," prefix.
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """**Role**: You are a VLN verification and replanning module. Use waypoint history, views, and maps to verify subtask completion, determine current position, and plan the next subtask. No manipulation.

**Task**: {instruction}

**Previous Subtask**:
- Destination: {subtask_destination}
- Instruction: {subtask_instruction}

**Waypoint History**: {waypoint_summary}

# Inputs
**12 Views** (sampled every 30°; each RGB view HFOV is about 79°): IMAGE1=Front 0°, angles increase CCW
- **Obstacle distance**: nearest obstacle only. <0.5m=blocked | 0.5-1.0m=caution | >1.0m=passable
- **Custom landmark bbox** (if present): current-view cue only; use shown name + distance/angle only as room/object evidence, not map memory or path-clearance proof
**2 Maps**: Global (full + history) + Local (nearby + 0.5m circle)
**Map colors**: White=unexplored | Black=obstacles | Green=safe | Orange=trajectory | Red=you | Blue numbered circles=history waypoints on Global only

# Reasoning (6 Parts)

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)**
**Format**: "IMAGE# (Direction Angle°): [space/room] + visible evidence". Mention only what is actually visible and useful, such as NEAR objects, FAR objects, landmark name + shown distance/angle, obstacle distance.

**REQUIRED - Analyze ALL 12 IMAGEs in order**:
**Per IMAGE requirement**: direction, likely space/room, and only the visible evidence that exists in that IMAGE
**Distance**: NEAR<1m (large/current) | FAR>1.5m (small/distant)
**NO hallucination**: Say only what is visible. If an IMAGE only shows a near wall, say that only. Do not write `none`, do not fill empty slots, and do not invent landmarks / FAR objects / waypoint cues.

**Conclusion (mandatory after all 12)**:
- Current: [room + NEAR<1m from multiple IMAGEs]
- Available: Which IMAGEs are safe (obs>1m)?
- Blocked: Which IMAGEs are <0.5m?
- Next: Which IMAGEs best continue the task?

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
6. **Arrival check**: Room first, then target object in that room. Wrong room or FAR(>1.5m, 1-2 views)=Continue | Correct room + target object within ~1m=STOP

**4) Direction Selection (Exploration Priority)**
A) Next + direction?
B) Scan 12 IMAGEs → where?
C) Use waypoint history + current views: which direction continues toward the most likely relevant space and object?
D) Check map/history: prefer the task-relevant likely space/object combination
E) Eliminate: **obs<0.5m=blocked** | **obs 0.5-1.0m=risky**
   A far room/object in image does NOT mean path is clear — obs label matters
F) Choose: most likely task-relevant space/object from waypoint history > visible target/landmark > plausible continuation > **obs>1.0m** > Map green path

**5) Near-term**: After auto-rotation, give the easiest immediate action toward the current room/object target; if already in the correct room and the target object is within ~1m, STOP
**6) Long-term**: Remaining → final

**Sequential planning rule**:
- If the current subtask is unfinished, continue it; only after completion can `next_waypoint_destination` move to the next stage.
- If the final "[room]'s [object]" is reached and that object is within ~1m in the correct room, STOP and set `global_task_finish=true`.

# Output (JSON only)

{{
    "reasoning": "<6 parts REQUIRED, concise: 1)12-Views(IMAGE 1-12 in order; mention only visible evidence, omit empty items), 2)Maps(local+waypoint history+global history/trajectory), 3)Position+Task chain(✓→Current→unmarked)+arrival, 4)Direction, 5)Near-term, 6)Long-term>",
    "current_waypoint": "<Room | Nearby (<1m): obj1, obj2 | Connected (>2m): area1, area2>",
    "waypoint_sequence": "<Completed(✓)→Current→Next→Goal. Mark (✓) passed/at(<0.5m) only>",
    "task_progress": "<Completed✓ current(Current) future unmarked. ONE (Current) only. All✓+NO(Current)=complete>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "next_waypoint_destination": "<Room's object; if the subtask target is the landmark itself, explicitly include that landmark word>",
    "subtask_instruction": "<Exactly one short sentence: 'From [next_waypoint_direction] view, start, [action + optional pass-by/path cue + destination]'>",
    "next_waypoint_landmark": "<Single clear recognizable landmark/object phrase; NEVER use door/doorway/hallway/corridor>",
    "global_task_finish": <true if the final room-object target is reached and the target object is within ~1m. Else false>
}}

# Examples (abbreviated):

## Ex1: Rug arrival
**Task**: Exit bedroom, turn left. Walk straight passing gray couch, stop at rug.
**Previous**: Navigate past gray couch toward rug
**Obs:** IMAGE 1: Rug <0.5m. IMAGE 10: Gray couch beside. IMAGE 7: Hallway behind

{{
    "reasoning": "1) 12-Views: IMAGE1-2 show the rug very near. IMAGE3-6 show nearby living-room furniture. IMAGE7 shows the hallway opening behind. IMAGE10-12 show the gray couch near. Conclusion: at the rug in the living room. 2) Maps: Local-0.5m has rug area. Waypoint history and trajectory run bedroom→hallway→living room→rug. 3) Position: rug very near across multiple views + map/history -> Living Room, at Rug(Current=Goal). Chain: Bedroom(✓)→Hallway(✓)→Gray Couch(✓)→Rug(Current=Goal). 4) Direction: no new direction needed because the goal is already reached. 5) Near: STOP. 6) Long: task complete.",
    "current_waypoint": "Living Room - near rug, gray couch",
    "waypoint_sequence": "Bedroom(✓)→Hallway(✓)→Gray Couch(✓)→Rug(Current=Goal)",
    "task_progress": "Exit bedroom(✓), turn left(✓). Walk passing gray couch(✓), stop at rug(✓).",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "Living Room's Rug",
    "subtask_instruction": "From IMAGE 1 (Front 0°) view, start, stop at the living room's rug.",
    "next_waypoint_landmark": "rug",
    "global_task_finish": true
}}

## Ex2: Hallway to bedroom
**Task**: Walk to kitchen through hallway, then enter bedroom on left.
**Previous**: Navigate through hallway
**Obs:** IMAGE 1: Hallway ahead 3.0m. IMAGE 5: Bedroom doorway (~2.5m), bed inside. IMAGE 7: Kitchen behind

{{
    "reasoning": "1) 12-Views: IMAGE1 shows hallway ahead. IMAGE2-4 show side walls nearby. IMAGE5 shows the bedroom entrance with the bed farther inside and is passable. IMAGE8-12 keep showing hallway walls. Conclusion: currently in hallway; IMAGE5 best continues the task. 2) Maps: Local shows a narrow corridor; waypoint history and trajectory run kitchen→hallway, with bedroom left/front from current. 3) Position: hallway walls nearby + waypoint history -> Hallway(Current). Chain: Kitchen(✓)→Hallway(Current)→Bedroom. 4) Direction: IMAGE5 best continues toward the most likely relevant room/object. 5) Near: rotate to IMAGE5 and move toward the bedroom bed. 6) Long: after entering bedroom, stop at the target area.",
    "current_waypoint": "Hallway - bedroom doorway visible",
    "waypoint_sequence": "Kitchen(✓)→Hallway(Current)→Bedroom(Goal)",
    "task_progress": "Walk to kitchen(✓) through hallway(Current), then enter bedroom on left.",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "Bedroom's bed",
    "subtask_instruction": "From IMAGE 5 (Left 120°) view, start, move toward the bedroom's bed.",
    "next_waypoint_landmark": "bed",
    "global_task_finish": false
}}

**Critical Rules**:
- **Planning priority**: Finish the current nearest unfinished stage before later stages, and for that stage prefer the nearest landmark that advances it. Follow waypoint history and trajectory toward the most likely task-relevant space/object.
- **Reasoning discipline**: Base reasoning only on visible evidence. Part 1 must analyze all 12 IMAGEs in order, but keep it concise: mention only what is actually visible/relevant in each IMAGE, omit empty fields, and never write fake filler such as `none`. Part 3 must still detail position + task chain (✓→Current→unmarked). Determine position first, then mark progress.
- **Progress and arrival**: Behind=(✓) | Now=(Current) ONE only | Ahead=unmarked; backtrack→rollback. Judge arrival in two steps: first confirm the room/space, then confirm the room's target object is within ~1m; far visibility alone is not arrival. "[room]'s [object]" means go to the room first, then the object; if the current/final room-object target is already within ~1m, STOP immediately.
- **Output constraints**: Use a single common room/space type, remove modifiers, and normalize corridor-like wording to `hallway`. `next_waypoint_landmark` must be a clear recognizable object/furniture phrase, not door/doorway/hallway/corridor. `next_waypoint_destination` must stay in "[room]'s [object]" form, and detail phrases should use [room]+[relation]+[object], e.g. "Living room's gray couch", not "couch"; if the subtask target is the same landmark, explicitly include that landmark word in `next_waypoint_destination`. `subtask_instruction` must be one short immediate sentence for only the nearest unfinished room/object subtask, in the fixed form "From [next_waypoint_direction] view, start, [action + optional pass-by/path cue + destination]." Use direct verbs such as move/go/walk/enter/pass/follow/cross/approach/continue/head/climb/ascend/descend/stop. The action module automatically drops the "From ... view, start," prefix.
- **Spatial wording**: "At entrance" = doorway, NOT inside.
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
        action_space=action_space,
    )

def get_verification_replanning_prompt(instruction: str,
                                       subtask_destination: str,
                                       subtask_instruction: str,
                                       action_space: str,
                                       detected_landmarks: str = None,
                                       waypoint_summary: str = None) -> str:
    """
    获取验证和重规划提示词
    
    Args:
        instruction: 完整导航指令
        subtask_destination: 当前子任务目的地
        subtask_instruction: 当前子任务指令
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
        action_space=action_space,
        detected_landmarks=detected_landmarks,
        waypoint_summary=waypoint_summary,
    )
