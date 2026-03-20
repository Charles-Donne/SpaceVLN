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
**12 Views** (sampled every 30° around 360°; each RGB view HFOV is about 79°): 
- **Obstacle distance**: nearest obstacle only. <0.5m=blocked | 0.5-1.0m=caution | >1.0m=passable
**Map**: full explored area + obstacles + current pose
- **Map colors**: White=unexplored | Black=obstacles | Green=safe floor | Dark red=trajectory | Red Arrow=you position

# Reasoning (5 Parts)

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)**
**Format**: "IMAGE# (Direction Angle°): [object] or [space] + visible evidence". For each IMAGE, analyze only the evidence that is actually shown, in this order when present: NEAR large/current objects and what space they indicate; FAR small/distant objects and what space they indicate; obstacle distance.
**No hallucination**: for each IMAGE state direction, likely space, and only the visible evidence that exists in that IMAGE. If an IMAGE only shows a near wall, say that only. Do not write `none`, do not fill empty slots, do not mention NEAR / FAR objects if unvisible, and do not merge IMAGE1-2 / IMAGE3-4 together: each IMAGE must be analyzed separately.
**Conclusion**: Current=[current space + NEAR objects from multiple IMAGEs] | Available=[which IMAGEs are safe to other spaces or FAR objects] | Blocked=[which IMAGEs are <0.5m] | Next=[which IMAGEs best continue the task]

**2) Current Position + Final Task Goal + Task Chain**
1. **Current Start Position**: you are in the beginning position of the Task. Localize yourself from the current observations, especially NEAR evidence, together with the map pose. State both the current space and the local place inside it in the form "[space] - near/by/at [local objects/place]".
2. **Final Task Goal**: know the final task destination is what objects in what space.
3. **Parse Full Task**: break the task into stages from the current start position to the final goal.
4. **Task Waypoint Chain**: state the start(Current) → next → ... → goal chain, and explain which first stage should be executed now and why.
5. **Initial Progress marking**: because this is the task start, nothing is completed yet. Mark only the start subtask as `(Current)` and all later stages as unmarked; do not use `(✓)` unless the final target is already reached at the start.
6. **Global Goal Arrival Check**: judge whether the final task goal is already reached at the task start. If the final goal "[space]'s [object]" is already within ~1m in the correct space, you must STOP and set `global_task_finish=true`.
**Careful analysis**: Part 2 must reason carefully and explicitly, not just name a room. It should explain current place, final goal place, task chain, current stage, and stop/not-stop judgment in clear detail.

**3) Subtask Destination + Subtask Instruction**
1. Based on the current position, waypoint/task chain, and 12-view observations, identify the most appropriate immediate next subtask destination as what object in what space.
2. Based on the Task Instruction and Progress, write the next subtask instruction for only the nearest unfinished stage.
3. The next subtask must be the most direct basic subtask that advances the task now without skipping a nearer unfinished stage, while preserving landmark order/relations using cues like pass-by / toward / enter / after / then.

**4) Direction Selection + Summary**
1. From the 12 IMAGEs plus the map, choose the direction that best matches the current subtask destination, transition space, and subtask instruction.
2. Avoid obstacles: eliminate **obs<0.5m=blocked** directions, avoid **obs 0.5-1.0m=risky** directions when better options exist, and remember that a far room/object in an image does not mean the path is clear. Prefer: task-relevant space/object from the space structure > visible target/landmark > plausible continuation > **obs>1.0m** > Map green path.
3. **Short-term**: summarize the immediate next move, chosen direction, and short-term target, and make sure they match the subtask instruction.
4. **Long-term**: summarize the remaining stages from after this subtask to the final goal.

**Sequential planning rule**:
- Output only the immediate next task stage/subtask for current-stage progress; do not plan stage +2/+3 before stage +1 is finished.

# Output (JSON only)

{{
    "reasoning": "<Follow the reasoning flow above exactly. Part 1 must analyze IMAGE1, IMAGE2, ... IMAGE12 separately without merging; Part 2 must reason carefully about current place, final goal, task chain, and global stop/not-stop. Then continue with Parts 3-5.>",
    "current_waypoint": "<Room - current local place from nearby observations>",
    "waypoint_chain": "<Task chain from task start: Start(Current)→Next→...→Goal. Do not mark (✓) in initial planning unless already at goal>",
    "task_progress": "<Original task text with (Current) inserted at the current stage, e.g. 'Exit bedroom(Current), turn left. Walk straight passing gray couch, stop at rug.'>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "next_waypoint_destination": "<Room's object; if the subtask target is the landmark itself, explicitly include that landmark word>",
    "subtask_instruction": "<Exactly one short sentence: 'From [next_waypoint_direction] view, start, [action + optional pass-by/path cue + destination]'>",
    "next_waypoint_landmark": "<Single clear recognizable landmark/object phrase; NEVER use door/doorway/hallway/corridor>",
    "global_task_finish": <true if the final room-object target is reached and the target object is within ~1m. Else false>
}}

#Examples (abbreviated):

## Ex1: Bedroom to rug
**Task**: Exit bedroom, turn left. Walk straight passing gray couch, stop at rug.
**Obs:** IMAGE 2-5: hallway entrance. IMAGE 6-9: bedroom interior with bed / tripod / couch / curtains.

{{
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): wall edge NEAR, blocked. IMAGE2(Left 30°): hallway entrance FAR, passable. IMAGE3(Left 60°): hallway entrance FAR, passable. IMAGE4(Left 90°): hallway entrance FAR, passable. IMAGE5(Left 120°): hallway entrance FAR, passable. IMAGE6(Left 150°): bedroom doorway FAR, bed and tripod NEAR, passable. IMAGE7(Back 180°): bedroom interior FAR, couch and curtains NEAR, open. IMAGE8(Right 150°): bedroom interior FAR, couch and curtains NEAR, passable. IMAGE9(Right 120°): bedroom interior FAR, couch and curtains NEAR, passable. IMAGE10(Right 90°): wall edge NEAR, blocked. IMAGE11(Right 60°): wall edge NEAR, blocked. IMAGE12(Right 30°): wall edge NEAR, blocked. Conclusion: current start place is the bedroom entrance beside the hallway, and IMAGE2 best continues the task. 2) Current position + final goal + task chain: the nearby bedroom furniture in IMAGE6-9 and the hallway opening in IMAGE2-5 show the current start is the bedroom entrance facing the hallway. The final goal is the living room rug near the gray couch. This is still the task start, so nothing is completed and the final goal is not reached. Chain: Bedroom(Start/Current)→Hallway→Living Room→Rug(Goal). 3) Subtask destination + instruction: the immediate next subtask destination is the hallway's entrance opening, so the next direct basic subtask is to move into the hallway. 4) Direction selection + summary: choose IMAGE2 because it best matches the hallway transition and avoids blocked views; short-term move into the hallway, long-term continue to the living room and stop near the rug.",
    "current_waypoint": "Bedroom - at entrance",
    "waypoint_chain": "Bedroom(Current)→Hallway→Living Room→Rug(Goal)",
    "task_progress": "Exit bedroom(Current), turn left. Walk straight passing gray couch, stop at rug.",
    "next_waypoint_direction": "IMAGE 2 (Left 30°)",
    "next_waypoint_destination": "Hallway's entrance opening",
    "subtask_instruction": "From IMAGE 2 (Left 30deg) view, start, move into the hallway.",
    "next_waypoint_landmark": "Hallway",
    "global_task_finish": false
}}

**Critical Rules**:
- **Planning priority**: Finish the nearest unfinished stage first; in initial planning, finish the first stage before later ones. Follow current views and the map toward the most likely task-relevant space/object, and preserve landmark order/relations.
- **Reasoning discipline**: Part 1 must cover all 12 IMAGEs in order, but keep it evidence-only and concise: in each IMAGE, state the likely space, then only the shown NEAR large objects and what space they indicate, FAR small objects and what space they indicate, and obstacle distance; omit empty fields and never write fake filler such as `none`. Part 2 must state that this is the task start, localize the current start place from NEAR observations plus map pose, then state the final goal, the start(Current)→next→goal task chain, and whether the final goal is already reached. Part 3 must identify only the immediate next direct basic subtask destination/instruction, not a later-stage shortcut. Part 4 must choose the direction that best matches that subtask while avoiding obstacles. Part 5 must briefly summarize the short-term move/target and the remaining stages.
- **Progress and stop**: In initial planning, nothing is completed yet; mark only the current/start stage as `(Current)` and leave later stages unmarked unless the final target is already reached at the start. If the final "[space]'s [object]" is already within ~1m in the correct space at the start, STOP immediately and set `global_task_finish=true`.
- **Output constraints**: Use a single common room/space type, remove modifiers, and normalize corridor-like wording to `hallway`. `next_waypoint_landmark` must be a clear recognizable object/furniture phrase, never door/doorway/hallway/corridor. `next_waypoint_destination` must be "[room]'s [object]"; if the subtask target is the same landmark, explicitly include that landmark word in `next_waypoint_destination`. "At entrance" means doorway. `subtask_instruction` must be one short immediate sentence for only the nearest unfinished room/object subtask, in the fixed form "From [next_waypoint_direction] view, start, [action + optional pass-by/path cue + destination]." Use direct verbs such as move/go/walk/enter/pass/follow/cross/approach/continue/head/climb/ascend/descend/stop. The action module automatically drops the "From ... view, start," prefix.
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """**Role**: You are a VLN verification and replanning module. Use the space structure, views, and maps to verify subtask completion, determine current position, and plan the next subtask. No manipulation.

**Task**: {instruction}

**Previous Subtask**:
- Destination: {subtask_destination}
- Instruction: {subtask_instruction}

**Space Structure**: {waypoint_summary}

# Inputs
**12 Views** (sampled every 30°; each RGB view HFOV is about 79°): 
- **Obstacle distance**: nearest obstacle only. <0.5m=blocked | 0.5-1.0m=caution | >1.0m=passable
- **Custom landmark bbox** (if present): current-view cue only; use shown name + distance/angle only as room/object evidence, not map memory or path-clearance proof
**Global Map**: full explored area + obstacles + trajectory + current pose + space structure
- **Map colors**: White=unexplored | Black=obstacles | Green=safe | Dark red=trajectory | Red Arrow=you position | Colored regions + blue tags=space structure on Global

# Reasoning (6 Parts)

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)**
**Format**: "IMAGE# (Direction Angle°): [object] or [space] + visible evidence". For each IMAGE, analyze only the evidence that is actually shown, in this order when present: NEAR large/current objects and what space they indicate; FAR small/distant objects and what space they indicate; landmark name + shown distance; obstacle distance; reachable space waypoint marker + shown distance.
**No hallucination**: for each IMAGE state direction, likely space, and only the visible evidence that exists in that IMAGE. If an IMAGE only shows a near wall, say that only. Do not write `none`, do not fill empty slots, do not invent landmarks / FAR objects / space waypoint cues, do not analyze space waypoints unless that IMAGE explicitly shows a space waypoint entry, and do not merge IMAGE1-2 / IMAGE3-4 together: each IMAGE must be analyzed separately.
**Conclusion**: Current=[current space + NEAR objects from multiple IMAGEs] | Available=[which IMAGEs are safe to other spaces or FAR objects] | Blocked=[which IMAGEs are <0.5m] | Next=[which IMAGEs best continue the task]

**2) Map + Space Structure Analysis**
Use only the Map and Space Structure in this part.
1. **Identify Current Area**: read the current area from the map and space structure; if the current area is `Unknown`, infer the current space from Part 1 NEAR observations.
2. **Read Space Waypoint list**: for each `Space WP#...`, identify which space it belongs to, what local objects/place cues it has, its direction/distance relative to current, and whether it is directly reachable; if not directly reachable, state which other space waypoint(s) it should be reached through.
3. **Read Space Waypoint Path**: use the Space Waypoint Path to know which spaces/waypoints were already visited, and whether the current waypoint is backtracking/repeating old space or already entering the next space waypoint area.
4. **Read Map**: check your current position on the map and other spaces in which directions.

**3) Current Position + Final Task Goal + Task Chain**
1. **Current Position**: Localize yourself from the current observations, especially NEAR evidence, together with the map pose. State both the current space and the local place inside it in the form "[space] - near/by/at [local objects/place]".
2. **Final Task Goal**: know the final task destination is what objects in what space.
3. **Parse Full Task**: break the task into stages from the current start position to the final goal. 
4. **Space Waypoint Chain**: Completed(✓) → Current → Next → ... → Goal. Analyze: what is completed? what is the current stage? what are the next or future stages?
5. **Task Progress marking**: Passed space waypoints behind current=(✓) | Current subtask =(Current) ONE only | Ahead=unmarked.
6. **Global Goal Arrival Check**: judge whether the final task goal is already reached now. If the final goal "[space]'s [object]" is already within ~1m in the correct space, you must STOP and set `global_task_finish=true`.
**Careful analysis**: Part 3 must reason carefully and explicitly, not just name a room. It should explain current place, final goal place, task chain, completed/current/future stages, and stop/not-stop judgment in clear detail.

**4) Subtask Destination + Subtask Instruction**
1. Based on the current position, waypoint chain, and 12-view observations, identify the most appropriate immediate next subtask destination as what space and what object/local place in that space.
2. Based on the task chain and current progress, write the next subtask instruction for only the nearest unfinished stage.
3. If the current subtask is unfinished, continue it; if it is finished, advance to the next unfinished stage. Do not skip to a later stage while a nearer stage is still unfinished.
4. The next subtask must be the most direct basic subtask that advances the task now without skipping a nearer unfinished stage, while preserving landmark order/relations using cues like pass-by / toward / enter / after / then.

**5) Direction Selection + Summary**
1. From the 12 IMAGEs plus the space structure, choose the direction that best matches the current subtask destination, transition space, and subtask instruction.
2. Avoid obstacles: eliminate **obs<0.5m=blocked** directions, avoid **obs 0.5-1.0m=risky** directions when better options exist, and remember that a far room/object in an image does not mean the path is clear. Prefer: task-relevant space/object from the space structure > visible target/landmark > plausible continuation > **obs>1.0m** > Map green path.
3. **Short-term**: summarize the immediate next move, chosen direction, and short-term target, and make sure they match the subtask instruction.
4. **Long-term**: summarize the remaining stages from after this subtask to the final goal.

**Sequential planning rule**:
- If the current subtask is unfinished, continue it; only after completion can `next_waypoint_destination` move to the next stage.
- If the final "[room]'s [object]" is reached and that object is within ~1m in the correct room, STOP and set `global_task_finish=true`.

# Output (JSON only)

{{
    "reasoning": "<Follow the reasoning flow above exactly. Part 1 must analyze IMAGE1, IMAGE2, ... IMAGE12 separately without merging; Part 3 must reason carefully about current place, final goal, task chain, completed/current/future stages, and global stop/not-stop. Then continue with Parts 4-6.>",
    "current_waypoint": "<Room - current local place from nearby observations>",
    "waypoint_chain": "<Space waypoint chain: Completed(✓)→Current→Next→Goal. Mark (✓) passed/at(<0.5m) only>",
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
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): rug NEAR. IMAGE2(Right 30°): rug edge NEAR. IMAGE3(Right 60°): nearby living-room furniture NEAR. IMAGE4(Right 90°): nearby living-room furniture NEAR. IMAGE5(Left 120°): living-room furniture NEAR. IMAGE6(Left 150°): living-room furniture NEAR. IMAGE7(Back 180°): hallway opening FAR. IMAGE8(Right 210°): living-room side wall/furniture NEAR. IMAGE9(Left 240°): gray couch NEAR. IMAGE10(Left 270°): gray couch NEAR. IMAGE11(Left 300°): rug/couch area NEAR. IMAGE12(Left 330°): rug edge NEAR. Conclusion: at the rug in the living room. 2) Map + Space Structure: the space structure and trajectory run bedroom→hallway→living room→rug, and the current pose sits at the end of that path. 3) Current position + final goal + task chain: the NEAR rug and gray couch evidence across IMAGE1-2 and IMAGE9-12 shows the current place is the living room rug area beside the gray couch. The final task goal is also the living room's rug area. Chain: Bedroom(✓)→Hallway(✓)→Gray Couch(✓)→Rug(Current=Goal). Since the goal space and goal object are already reached now, STOP and set global_task_finish=true. 4) Subtask destination + instruction: the immediate destination remains the living room's rug, and no new movement subtask is needed because the goal is already reached. 5) Direction selection + summary: no new direction is needed because the goal is already reached; short-term STOP at the current rug position; long-term task complete.",
    "current_waypoint": "Living Room - at rug beside gray couch",
    "waypoint_chain": "Bedroom(✓)→Hallway(✓)→Gray Couch(✓)→Rug(Current=Goal)",
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
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): hallway ahead FAR. IMAGE2(Right 30°): hallway side wall NEAR. IMAGE3(Right 60°): hallway side wall NEAR. IMAGE4(Right 90°): hallway side wall NEAR. IMAGE5(Left 120°): bedroom entrance FAR, bed FAR, passable. IMAGE6(Left 150°): bedroom-side opening FAR. IMAGE7(Back 180°): kitchen behind FAR. IMAGE8(Right 210°): hallway wall NEAR. IMAGE9(Left 240°): hallway wall/doorframe NEAR. IMAGE10(Left 270°): side wall NEAR. IMAGE11(Left 300°): side wall NEAR. IMAGE12(Left 330°): hallway front/side transition. Conclusion: currently in hallway; IMAGE5 best continues the task. 2) Map + Space Structure: the space structure and trajectory run kitchen→hallway, and the map shows the bedroom transition left/front from current. 3) Current position + final goal + task chain: the NEAR hallway walls and the FAR bedroom entrance/bed evidence show the current place is the hallway by the bedroom doorway on the left/front side. The final goal is the bedroom's bed deeper inside the bedroom. Chain: Kitchen(✓)→Hallway(Current)→Bedroom. The final goal is not reached yet because the bedroom bed is still far and the current place is still hallway, so continue. 4) Subtask destination + instruction: the immediate next subtask destination is the bedroom's bed, and the direct basic subtask is to enter the bedroom and move toward the bed. 5) Direction selection + summary: choose IMAGE5 because it best matches the subtask destination/transition space and avoids blocked views; short-term rotate to IMAGE5 and move toward the bedroom entrance/bed direction; long-term after entering the bedroom, stop at the target area.",
    "current_waypoint": "Hallway - by bedroom doorway",
    "waypoint_chain": "Kitchen(✓)→Hallway(Current)→Bedroom(Goal)",
    "task_progress": "Walk to kitchen(✓) through hallway(Current), then enter bedroom on left.",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "Bedroom's bed",
    "subtask_instruction": "From IMAGE 5 (Left 120°) view, start, move toward the bedroom's bed.",
    "next_waypoint_landmark": "bed",
    "global_task_finish": false
}}

**Critical Rules**:
- **Planning priority**: Finish the current nearest unfinished stage before later stages, and for that stage prefer the nearest landmark that advances it. Follow the space structure and trajectory toward the most likely task-relevant space/object.
- **Reasoning discipline**: Base reasoning only on visible evidence. Part 1 must analyze all 12 IMAGEs in order, but keep it concise: in each IMAGE, state the likely space, then only the shown NEAR large objects and what space they indicate, FAR small objects and what space they indicate, landmark distance cue, obstacle distance, and any shown reachable space waypoint cue; omit empty fields and never write fake filler such as `none`. Part 2 must first read map/space-structure current place; if the current area is `Unknown`, resolve it from Part 1 NEAR observations, then read the space waypoint list and history path/trajectory before later reasoning. Part 3 must localize the current position, state the final goal and task chain, and explicitly judge whether the final goal is already reached. Part 4 must identify only the immediate next direct basic subtask destination/instruction. Part 5 must choose the direction that best matches that subtask while avoiding obstacles. Part 6 must briefly summarize the short-term move/target and the remaining stages.
- **Progress and stop**: Behind=(✓) | Now=(Current) ONE only | Ahead=unmarked; backtrack→rollback. "[room]'s [object]" means go to the room first, then the object. If the final "[space]'s [object]" is already within ~1m in the correct space, STOP immediately and set `global_task_finish=true`; otherwise continue with the current nearest unfinished stage.
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
        waypoint_summary: 空间结构字符串
        
    Returns:
        格式化的提示词字符串
    """
    if not detected_landmarks:
        detected_landmarks = "No landmarks detected yet"
    if not waypoint_summary:
        waypoint_summary = "No space structure recorded yet"
    
    return VERIFICATION_REPLANNING_PROMPT.format(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        action_space=action_space,
        detected_landmarks=detected_landmarks,
        waypoint_summary=waypoint_summary,
    )
