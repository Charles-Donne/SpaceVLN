"""
VLM规划提示词模板
================
用于LLM高层规划的提示词模板
"""

# 初始规划提示词 - 在任务开始时生成第一个子任务
INITIAL_PLANNING_PROMPT = """**Role**: You are a VLN planning module. Use the observations and map to localize the start position, identify the first reachable task stage, and output precise navigation instructions. No manipulation.

**Task**: {instruction}

**Initial state**: You are at the task start. Follow the Task strictly from the beginning and complete only the first subtask.

# Inputs
**12 Views** (sampled every 30° around 360°; each RGB view HFOV is about 79°):
- **Obstacle distance**: nearest obstacle only. <0.5m=blocked | 0.5-1.0m=caution | >1.0m=passable
**Map**: explored area + obstacles + current pose
- **Map colors**: White=unexplored | Black=obstacles | Green=safe floor | Dark red=trajectory | Red Arrow=you position

# Reasoning (4 Parts)

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)**
**Format**: For each IMAGE, use `IMAGE# (Direction Angle°): likely [space]; NEAR: ...; FAR: ...; Obstacle: ...; Landmark: ...; Space Waypoint: ...`, and omit any field not actually visible there.
**Evidence order**: When present, analyze only visible evidence in this order: NEAR current/large objects and implied space; FAR distant objects/openings and implied space; obstacle distance + blocked/caution/passable judgment; landmark name + shown distance; reachable space waypoint marker + shown distance.
**No hallucination**: analyze each IMAGE separately. If an IMAGE only shows a near wall or close furniture, say only that. Do not write `none`, fill empty slots, invent spaces / FAR objects / landmarks / space waypoint cues, or mention landmark / space waypoint unless explicitly shown.
**Conclusion**: Identify from the 12-view content: Current Position Guess: [current space + NEAR objects + surrounding scene across adjacent views] | Reachable Far Area/Object: [which IMAGEs show FAR spaces/objects/openings that are likely reachable, what space/object each one leads to, and which task-related transition/route point each one may support] | Blocked: [which IMAGEs have obstacle distance <0.5m]

**2) Current Position + Final Task Goal + Task Chain**
1. **Current Start Position**: localize the start position explicitly from Part 1. The reasoning must clearly answer both **which space** you are in and **where inside that space** you are now. Use nearby objects, nearby walls/openings/furniture layout, and continuity across adjacent views first. A room name alone is not enough. State it as "[space] - near/by/at [local objects]".
2. **Final Task Goal**: state the final task goal explicitly and only in the form "[space]'s [object/local place]".
3. **Parse Full Task**: break the task into ordered fixed stages from the current start position to the final goal. Each stage must use `"[space]'s [object/local place]" -> "[space]'s [object/local place]"`. Stage boundaries follow **cross-space transitions**: if the task moves into another space, split there and make entering that next space its own stage; if the task says pass by / through / around one object to reach another object in the **same space**, keep that as one stage and keep the route relation inside that stage instead of splitting it.
4. **Task Waypoint Chain**: output the ordered stage-anchor chain that matches those stages: Start(Current) → Stage1 destination → Stage2 destination → ... → Goal. Each node must be one `"[space]'s [object/local place]"`, and the chain must preserve task space order with no skipped intermediate space/object.
5. **Task Progress start**: because this is the task start, nothing is completed yet. Write `task_progress` as task-ordered natural-language sub-instructions expanded from the original Task, not node-to-node arrows. You may refine/supplement the task wording into clearer sub-instructions, but it must still follow the task order and the stage split above. Same-space pass-by / through / around relations stay inside one task piece; cross-space movement becomes separate task pieces. Separate sub-instructions with commas. Mark only the current task piece `(Current)`; later task pieces stay unmarked; do not use `(✓)` unless the final target is already reached at the start.
6. **Task Goal Arrival Check**: this must be reasoned, not guessed. First use the current localized position to determine where you are in the Task Waypoint Chain and task progress. Then compare that current node/current task piece with the final goal node/final task piece. If you are not yet in the final goal space, do not stop. If you are already in the final goal space, use the 12 views to judge whether the final goal object/local place in that same space is already at/near/beside you. Only if this comparison shows that the final goal "[space]'s [object/local place]" is already reached within about 1.5m, or the 12 views clearly show arrival, set `global_task_finish=true`; otherwise false.
**Careful analysis**: localize first, then state the final goal, parse fixed stages, explain chain/progress, and reason Task Goal Arrival Check by comparing current position + current chain/progress state against the final goal.

**3) Subtask Destination + Direction + Subtask Instruction + Landmark**
1. In initial planning, use only the first task stage. From the current localized position, the task's ordered spaces/objects/route points, Part 1's reachable FAR spaces/objects, and the known space transitions, identify the first subtask destination: the next task-relevant `"[space]'s [object]"` that should be reached now. The task-mentioned spaces / route points / room transitions must be advanced one by one in order. If the first task stage is a same-space pass-by / through relation, keep its final object/local place as the destination and keep the intermediate cue inside the instruction; if the task moves into another space, the destination should be that next-space entry/object for the current stage, not a later-stage target. Use FAR analysis to infer which distant space/object/opening is the correct next transition toward that task destination. Do not skip spaces/objects or choose a destination behind clearly blocked directions.
2. Choose the direction from the 12 IMAGEs that best reaches that destination. This must follow the task and the destination chosen above: prefer where the destination itself appears, or the FAR space/object/opening that most directly and safely leads toward the needed next space/route point. Use the 12-view FAR evidence together with the space layout to reason which side leads to which space and which task-related relation (enter / pass-by / through / toward) should be followed. Prefer open directions with obstacle distance >1.0m, ideally >2.0m; avoid blocked or tight views, and do not backtrack unless clearly required.
3. Write one short immediate subtask instruction for that same first stage in the fixed form `From [next_waypoint_direction] view, start, ...`. It must directly match the chosen destination and direction, and it must describe only the nearest unfinished task piece. Use one of two concise styles: direct approach / enter for direct stages, or pass-by / through / around + destination for same-stage path relations. Do not split a same-space pass-by relation into another stage, and do not mention a later-stage destination.
4. Choose `next_waypoint_landmark` only for that first stage. **Definition**: it is the most useful visible concrete object cue for executing the current subtask now. If the subtask destination itself is a visible object/local target, prefer the destination itself as `next_waypoint_landmark`. Use another visible intermediate object only when the destination is not clearly visible as an object, or when a pass-by / through / around relation is necessary to execute the subtask. Keep it short; if no useful visible object landmark is needed, use an empty string.

**4) Plan**
1. **Short-term Plan**: explain why this destination, direction, and subtask instruction are the correct immediate plan. Summarize the next move, turn if any, short-term target, and landmark choice if any, and make sure they match.
2. **Long-term Plan**: summarize the remaining stage order after this subtask as a forward plan.

**Sequential planning rule**:
- Output only the immediate next task stage/subtask for current-stage progress; do not plan stage +2/+3 before stage +1 is finished.

# Output (JSON only)

{{
    "reasoning": "<Follow the reasoning flow above exactly. Part 1 must analyze IMAGE1, IMAGE2, ... IMAGE12 separately without merging and conclude with Current Position Guess / Reachable Far Area/Object / Blocked. Part 2 must reason: where the current start position is, what the final goal is in '[space]'s [object/local place]' form, how the task splits into fixed stages, what the waypoint chain and task-progress sub-instructions are, and whether Task Goal Arrival Check is satisfied now by comparing current localized position/current chain-progress state against the final goal. Then continue with Parts 3-4.>",
    "current_waypoint": "<Room - current local place from nearby observations>",
    "waypoint_chain": "<Ordered stage-anchor chain using one [room]'s [object/local place] node per fixed stage endpoint: Start(Current)→Next→...→Goal. Do not mark (✓) in initial planning unless already at goal>",
    "task_progress": "<Task-ordered natural-language sub-instructions expanded from the original Task, not waypoint arrows. Use commas to separate sub-instructions. Same-space pass-by/through relations stay in one task piece; cross-space transitions are separate task pieces. In initial planning only the current task piece is (Current); later task pieces are unmarked unless already at goal>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "next_waypoint_destination": "<Room's object/local place; if the subtask target is the landmark itself, explicitly include that landmark word>",
    "subtask_instruction": "<Exactly one short sentence in one of two forms: direct `From [next_waypoint_direction] view, start, [move/enter/approach] toward [destination].` or same-stage path `From [next_waypoint_direction] view, start, [pass/go through/go around/cross] [visible cue], then [enter/approach/continue toward] [destination].` Use the path form only when that cue-destination relation belongs to the same current task piece>",
    "next_waypoint_landmark": "<Single clear recognizable visible object phrase if useful; otherwise empty string. This is the visible object cue for the current subtask. If the subtask destination itself is a visible object/local target, prefer the destination itself. Prefer specific objects like bed/rug/chair/table/lamp, not broad areas/spaces. NEVER use door/doorway/hallway/corridor>",
    "global_task_finish": "<true only if Task Goal Arrival Check reasons from current position + current waypoint chain/task progress + visible goal evidence that you have already reached the correct final goal space and final goal object/local place; otherwise false>"
}}

#Examples (abbreviated):

## Ex1: Bedroom to rug
**Task**: Exit bedroom, turn left. Walk straight passing gray couch, stop at rug.
**Obs:** IMAGE 2-5: hallway entrance. IMAGE 6-9: bedroom interior with bed / tripod / couch / curtains.

{{
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): likely bedroom; NEAR wall edge; Obstacle blocked. IMAGE2(Left 30°): likely hallway transition; FAR hallway entrance opening; Obstacle passable. IMAGE3(Left 60°): likely hallway transition; FAR hallway entrance opening; Obstacle passable. IMAGE4(Left 90°): likely hallway transition; FAR hallway entrance opening; Obstacle passable. IMAGE5(Left 120°): likely hallway transition; FAR hallway entrance opening; Obstacle passable. IMAGE6(Left 150°): likely bedroom; NEAR bed and tripod; FAR bedroom interior; Obstacle passable. IMAGE7(Back 180°): likely bedroom; NEAR couch and curtains; FAR bedroom interior; Obstacle passable. IMAGE8(Right 150°): likely bedroom; NEAR couch and curtains; FAR bedroom interior; Obstacle passable. IMAGE9(Right 120°): likely bedroom; NEAR couch and curtains; FAR bedroom interior; Obstacle passable. IMAGE10(Right 90°): likely bedroom boundary; NEAR wall edge; Obstacle blocked. IMAGE11(Right 60°): likely bedroom boundary; NEAR wall edge; Obstacle blocked. IMAGE12(Right 30°): likely bedroom boundary; NEAR wall edge; Obstacle blocked. Conclusion: Current Position Guess: bedroom entrance beside the hallway opening. Reachable Far Area/Object: IMAGE2-5 lead toward the hallway's entrance opening and support the next hallway transition. Blocked: IMAGE1, IMAGE10-12. 2) Current Position + Final Task Goal + Task Chain: from the bedroom furniture around IMAGE6-9 and the open hallway views in IMAGE2-5, the current start position is Bedroom - at entrance by bed/tripod side. The final task goal is Living room's rug. Fixed stages: Bedroom's entrance -> Hallway's entrance opening | Hallway's entrance opening -> Living room's gray couch | Living room's gray couch -> Living room's rug. Task Waypoint Chain: Bedroom's entrance(Current) -> Hallway's entrance opening -> Living room's gray couch -> Living room's rug(Goal). Task Progress start: Exit bedroom(Current), turn left, walk straight passing gray couch, stop at rug. Cross-space movement stays split by space change, so the current piece is only the bedroom-to-hallway transition. Task Goal Arrival Check: compare the current localized position and current chain/progress state against the final goal. The current node/task piece is still the bedroom start stage, while the final goal node/task piece is the living room's rug, so the goal is not reached and global_task_finish=false. 3) Subtask Destination + Direction + Subtask Instruction + Landmark: the first-stage destination is Hallway's entrance opening because it is the next task-relevant place in order and the reachable FAR opening in IMAGE2-5 supports that hallway transition. IMAGE2 is the best direction because it most directly and safely leads into the next task-required space. The immediate instruction is the direct first-stage instruction to move toward the hallway's entrance opening. No visible object landmark is needed for this direct first-stage transition. 4) Plan: short-term plan is to turn toward IMAGE2 and move to the hallway's entrance opening. Long-term plan is to reach the hallway first, then continue to the living room's gray couch, and finally stop at the living room's rug.",
    "current_waypoint": "Bedroom - at entrance by bed and tripod",
    "waypoint_chain": "Bedroom's entrance(Current)→Hallway's entrance opening→Living room's gray couch→Living room's rug(Goal)",
    "task_progress": "Exit bedroom(Current), turn left, walk straight passing gray couch, stop at rug",
    "next_waypoint_direction": "IMAGE 2 (Left 30°)",
    "next_waypoint_destination": "Hallway's entrance opening",
    "subtask_instruction": "From IMAGE 2 (Left 30deg) view, start, move toward the hallway's entrance opening.",
    "next_waypoint_landmark": "",
    "global_task_finish": false
}}

**Critical Rules**:
- **Planning priority**: finish the nearest unfinished stage first. In initial planning, finish only the first stage, preserve task space/object/route-point order, and do not skip intermediate spaces.
- **Reasoning discipline**: Part 1 must stay evidence-only and conclude with Current Position Guess / Reachable Far Area/Object / Blocked. Use Part 1 first to localize exactly where you are, then parse fixed stages, build the chain, and judge Task Goal Arrival Check. For destination/direction, use the current localized position + task order + reachable FAR evidence + known space transitions to choose the next reachable task-relevant destination and the best non-backtracking direction.
- **Progress, stop, and output**: `task_progress` must be task-ordered natural-language sub-instructions expanded from the original Task, separated by commas, not waypoint arrows: completed pieces in front, current piece `(Current)`, later pieces unmarked unless already at goal. Same-space pass-by / through relations stay inside one task piece; cross-space transitions split into separate task pieces. Task Goal Arrival Check must compare current localized position + current chain/progress state against the final goal, then confirm with visible final-goal evidence. `next_waypoint_destination` must stay in "[room]'s [object/local place]" form and match the reachable destination of the current stage. `next_waypoint_landmark` is the visible object cue for the current subtask: if the destination itself is a visible object/local target, prefer the destination itself; otherwise use a necessary visible intermediate object or empty string. Never use door/doorway/hallway/corridor. `subtask_instruction` must be one short sentence in the fixed form "From [next_waypoint_direction] view, start, ...", using the direct form for direct stages and the path form only when the current task piece itself is a same-stage pass-by / through / around relation. "At entrance" means doorway.
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """**Role**: You are a VLN verification and replanning module. Use the space structure, views, and maps to verify subtask completion, localize the current position, and plan the next subtask. No manipulation.

**Task**: {instruction}

**Previous Subtask**:
- Destination: {subtask_destination}
- Instruction: {subtask_instruction}

**Space Structure**: {waypoint_summary}

# Inputs
**12 Views** (sampled every 30°; each RGB view HFOV is about 79°):
- **Obstacle distance**: nearest obstacle only. <0.5m=blocked | 0.5-1.0m=caution | >1.0m=passable
- **Custom landmark bbox** (if present): current-view cue only; use shown name + distance/angle only as room/object evidence, not map memory or path-clearance proof
**Global Map**: explored area + obstacles + trajectory + current pose + space structure
- **Map colors**: White=unexplored | Black=obstacles | Green=safe | Dark red=trajectory | Red Arrow=you position | Colored regions + blue tags=space structure on Global

# Reasoning (5 Parts)

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)**
**Format**: For each IMAGE, use `IMAGE# (Direction Angle°): likely [space]; NEAR: ...; FAR: ...; Obstacle: ...; Landmark: ...; Space Waypoint: ...`, and omit any field not actually visible there.
**Evidence order**: When present, analyze only visible evidence in this order: NEAR current/large objects and implied space; FAR distant objects/openings and implied space; obstacle distance + blocked/caution/passable judgment; landmark name + shown distance; reachable space waypoint marker + shown distance.
**No hallucination**: analyze each IMAGE separately. If an IMAGE only shows a near wall or close furniture, say only that. Do not write `none`, fill empty slots, invent spaces / FAR objects / landmarks / space waypoint cues, or mention landmark / space waypoint unless explicitly shown.
**Conclusion**: Identify from the 12-view content: Current Position Guess: [current space + NEAR objects + surrounding scene across adjacent views] | Reachable Far Area/Object: [which IMAGEs show FAR spaces/objects/openings that are likely reachable, what space/object each one leads to, and which task-related transition/route point each one may support] | Blocked: [which IMAGEs have obstacle distance <0.5m]

**2) Map + Space Structure Analysis**
Use only the Map and Space Structure in this part.
1. **Identify Current Area**: read the current area from the map and space structure, but always cross-check it with Part 1 nearby evidence. If the map current area is `Unknown` or conflicts with nearby-view evidence, resolve the current space/local place from the strongest consistent 12-view nearby evidence.
2. **Read Space Waypoint list**: for each `Space WP#...`, identify its space, local objects/place cues, direction/distance relative to current, and whether it is directly reachable; if not, state which other space waypoint(s) it should be reached through.
3. **Read Space Waypoint Chain**: use it to know which spaces/waypoints were already visited and whether the current waypoint is backtracking/repeating old space or entering the next waypoint area.
4. **Read Map**: check the current position on the map and where other spaces lie.

**3) Current Position + Final Task Goal + Task Chain**
1. **Current Position**: localize the current position explicitly from Part 1. The reasoning must clearly answer both **which space** you are in and **where inside that space** you are now. Use nearby objects, nearby walls/openings/furniture layout, and continuity across neighboring views first. Map pose and space structure are only for support/disambiguation, not the sole basis. A room name alone is not enough. State it as "[space] - near/by/at [local objects/place]".
2. **Final Task Goal**: state the final task goal explicitly and only in the form "[space]'s [object/local place]".
3. **Parse Full Task**: break the task into ordered fixed stages from the current start position to the final goal. Each stage must use `"[space]'s [object/local place]" -> "[space]'s [object/local place]"`. Stage boundaries follow **cross-space transitions**: if the task moves into another space, split there and make entering that next space its own stage; if the task says pass by / through / around one object to reach another object in the **same space**, keep that as one stage and keep the route relation inside that stage instead of splitting it.
4. **Task Waypoint Chain**: build the ordered stage-anchor chain that matches those stages and anchor it to the already-judged **Current Position**. The matching node must be `(Current)`; nodes before it are `(✓)`; nodes after it are unmarked; if the current node is already the final goal, mark it `(Current, Goal)`. Each node must be one `"[space]'s [object/local place]"`, and the chain must preserve task space order with no skipped intermediate space/object.
5. **Task Progress marking**: write `task_progress` as task-ordered natural-language sub-instructions expanded from the original Task, not node-to-node arrows. You may refine/supplement the task wording into clearer sub-instructions, but it must still follow the task order and the stage split above. Same-space pass-by / through / around relations stay inside one task piece; cross-space movement becomes separate task pieces. Separate sub-instructions with commas. Completed task pieces stay in front with `(✓)`, the current unfinished task piece is `(Current)` ONE only, and later unfinished task pieces remain after it unmarked.
6. **Task Goal Arrival Check**: this must be reasoned, not guessed. First use the current localized position to determine where you are in the Task Waypoint Chain and task progress. Then compare that current node/current task piece with the final goal node/final task piece. If you are not yet in the final goal space, do not stop. If you are already in the final goal space, use the 12 views to judge whether the final goal object/local place in that same space is already at/near/beside you. Only if this comparison shows that the final goal "[space]'s [object/local place]" is already reached within about 1.5m, or the 12 views clearly show arrival, set `global_task_finish=true`; otherwise false.
**Careful analysis**: localize first, then state the final goal, parse fixed stages, explain chain/progress, and reason Task Goal Arrival Check by comparing current position + current chain/progress state against the final goal.

**4) Subtask Destination + Direction + Subtask Instruction + Landmark**
1. Based on the current position, waypoint chain, 12-view observations, visible space waypoint cues, space structure, and the full task, identify the immediate next subtask destination: the nearest unfinished task-relevant space/object/local place that should be advanced now. The task-mentioned spaces / route points / room transitions must be advanced one by one in order. If the current task piece is a same-space pass-by / through relation, keep its final object/local place as the destination and keep the intermediate cue inside the instruction; if the task moves into another space, the current destination should be that next-space entry/object for the current stage, not a later-stage target. Use FAR analysis to infer which distant space/object/opening is the correct next transition toward that task destination. Choose a destination that is visibly approachable now; do not choose one hidden behind clearly blocked directions.
2. If the current subtask is unfinished, continue it; if it is finished, advance to the next unfinished stage. Do not skip a nearer unfinished stage. Preserve waypoint order and task relations such as pass-by / toward / enter / after / then.
3. Choose the direction from the 12 IMAGEs plus the space structure that best reaches that destination. This must follow the task and the destination chosen above: judge it from where the destination itself appears, or which reachable FAR space/object/opening/space-waypoint view most directly and safely leads toward the needed next space/route point. Use the 12-view FAR evidence together with the space structure to reason which side leads to which space and which task-related relation (enter / pass-by / through / toward) should be followed. Prefer open directions with obstacle distance >1.0m, ideally >2.0m; avoid blocked or tight low-clearance views, and do not backtrack unless clearly required.
4. Write the next subtask instruction for only that nearest unfinished stage. It must directly match the chosen destination and direction. If the current task piece is a same-space pass-by / through / around relation, keep that cue + destination in one instruction; if the task changes to another space, keep the instruction only on the current cross-space stage and leave later-stage targets for later. Prefer one of these concise forms when appropriate:
   - Direct approach / enter: `From [next_waypoint_direction] view, start, [move/enter/approach] toward [destination].`
   - Via visible cue / obstacle bypass: `From [next_waypoint_direction] view, start, [pass/go through/go around/cross] [visible cue], then [enter/approach/continue toward] [destination].`
5. Choose the most relevant visible subtask landmark. **Definition**: it is the most useful visible concrete object cue for executing the current subtask now. If the subtask destination itself is a visible object/local target, prefer the destination itself as `next_waypoint_landmark`. Use another visible intermediate object only when the destination is not clearly visible as an object, or when a pass-by / through / around relation is necessary to execute the subtask. Keep `next_waypoint_landmark` as a short object noun phrase, not a broad space or vague area description. If no useful visible object landmark is needed, use an empty string.

**5) Plan**
1. **Short-term Plan**: explain why this destination, direction, and subtask instruction are the correct immediate plan. Summarize the next move, turn if any, short-term target, and landmark choice if any, and make sure they match.
2. **Long-term Plan**: summarize the remaining stage order after this subtask as a forward plan.

**Sequential planning rule**:
- If the current subtask is unfinished, continue it; only after completion can `next_waypoint_destination` move to the next stage.
- If you are already inside the correct final goal space and the final goal object/local place is within about 1.5m, or the 12 views clearly show you have already arrived there, STOP and set `global_task_finish=true`.

# Output (JSON only)

{{
    "reasoning": "<Follow the reasoning flow above exactly. Part 1 must analyze IMAGE1, IMAGE2, ... IMAGE12 separately without merging and conclude with Current Position Guess / Reachable Far Area/Object / Blocked. Part 3 must reason: where the current position is, what the final goal is in '[space]'s [object/local place]' form, how the task splits into fixed stages, what the waypoint chain and task-progress sub-instructions are, and whether Task Goal Arrival Check is satisfied now by comparing current localized position/current chain-progress state against the final goal. Then continue with Parts 4-5.>",
    "current_waypoint": "<Room - current local place from nearby observations>",
    "waypoint_chain": "<Ordered stage-anchor chain using one [room]'s [object/local place] node per fixed stage endpoint, anchored to the judged current position: nodes before current=(✓), the matched current node=(Current), nodes after current=unmarked, and if current is the goal use (Current, Goal)>",
    "task_progress": "<Task-ordered natural-language sub-instructions expanded from the original Task, not waypoint arrows. Use commas to separate sub-instructions. Same-space pass-by/through relations stay in one task piece; cross-space transitions are separate task pieces. Completed task pieces stay in front with (✓), current task piece=(Current) ONE only, later task pieces after it are unmarked. All task pieces (✓)=complete>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "next_waypoint_destination": "<Room's object/local place; if the subtask target is the landmark itself, explicitly include that landmark word>",
    "subtask_instruction": "<Exactly one short sentence in one of two forms: direct `From [next_waypoint_direction] view, start, [move/enter/approach] toward [destination].` or same-stage path `From [next_waypoint_direction] view, start, [pass/go through/go around/cross] [visible cue], then [enter/approach/continue toward] [destination].` Use the path form only when that cue-destination relation belongs to the same current task piece>",
    "next_waypoint_landmark": "<Single clear recognizable visible object phrase if useful; otherwise empty string. This is the visible object cue for the current subtask. If the subtask destination itself is a visible object/local target, prefer the destination itself. Prefer specific objects like bed/rug/chair/table/lamp, not broad areas/spaces. NEVER use door/doorway/hallway/corridor>",
    "global_task_finish": "<true only if Task Goal Arrival Check reasons from current position + current waypoint chain/task progress + visible goal evidence that you have already reached the correct final goal space and final goal object/local place; otherwise false>"
}}

# Examples (abbreviated):

## Ex1: Rug arrival
**Task**: Exit bedroom, turn left. Walk straight passing gray couch, stop at rug.
**Previous**: Navigate past gray couch toward rug
**Obs:** IMAGE 1: Rug <0.5m. IMAGE 10: Gray couch beside. IMAGE 7: Hallway behind

{{
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): likely living room; NEAR rug; Obstacle passable. IMAGE2(Right 30°): likely living room; NEAR rug edge; Obstacle passable. IMAGE3(Right 60°): likely living room; NEAR nearby furniture; Obstacle passable. IMAGE4(Right 90°): likely living room; NEAR nearby furniture; Obstacle passable. IMAGE5(Left 120°): likely living room; NEAR living-room furniture; Obstacle passable. IMAGE6(Left 150°): likely living room; NEAR living-room furniture; Obstacle passable. IMAGE7(Back 180°): likely hallway transition; FAR hallway opening; Obstacle passable. IMAGE8(Right 210°): likely living room; NEAR side wall and furniture; Obstacle passable. IMAGE9(Left 240°): likely living room; NEAR gray couch; Obstacle passable. IMAGE10(Left 270°): likely living room; NEAR gray couch; Obstacle passable. IMAGE11(Left 300°): likely living room; NEAR rug and couch area; Obstacle passable. IMAGE12(Left 330°): likely living room; NEAR rug edge; Obstacle passable. Conclusion: Current Position Guess: living room rug area beside the gray couch. Reachable Far Area/Object: IMAGE7 leads back toward the hallway opening. Blocked: none critical. 2) Map + Space Structure Analysis: the current pose and space waypoints place the agent at the end of the bedroom -> hallway -> living room route, which matches the near-view rug/couch evidence. 3) Current Position + Final Task Goal + Task Chain: the current position is Living room - at rug beside gray couch. The final task goal is Living room's rug. Fixed stages: Bedroom's entrance -> Hallway's entrance opening | Hallway's entrance opening -> Living room's gray couch | Living room's gray couch -> Living room's rug. Task Waypoint Chain: Bedroom's entrance(✓) -> Hallway's entrance opening(✓) -> Living room's gray couch(✓) -> Living room's rug(Current, Goal). Task Progress: Exit bedroom(✓), turn left(✓), walk straight passing gray couch(✓), stop at rug(Current, Goal). Task Goal Arrival Check: compare the current localized position and current chain/progress state against the final goal. The current node is already the goal node, the current task piece is already the final task piece, and the rug is directly beside the agent in the 12 views, so the goal is satisfied now and global_task_finish=true. 4) Subtask Destination + Direction + Subtask Instruction + Landmark: the immediate destination remains Living room's rug because it is the final target and it is already reached. IMAGE1 can remain the reference direction because the rug is directly present there. The instruction is to stop at the living room's rug, and the rug is the clearest visible landmark. 5) Plan: short-term plan is to stop at the current rug position with no further movement. Long-term plan is complete because all fixed stages are already finished.",
    "current_waypoint": "Living Room - at rug beside gray couch",
    "waypoint_chain": "Bedroom's entrance(✓)→Hallway's entrance opening(✓)→Living room's gray couch(✓)→Living room's rug(Current, Goal)",
    "task_progress": "Exit bedroom(✓), turn left(✓), walk straight passing gray couch(✓), stop at rug(Current, Goal)",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "Living room's rug",
    "subtask_instruction": "From IMAGE 1 (Front 0°) view, start, stop at the living room's rug.",
    "next_waypoint_landmark": "rug",
    "global_task_finish": true
}}

## Ex2: Hallway to bedroom
**Task**: Walk through hallway, then enter bedroom on left and go to bed.
**Previous**: Navigate through hallway
**Obs:** IMAGE 1: Hallway ahead 3.0m. IMAGE 5: Bedroom doorway (~2.5m), bed inside. IMAGE 7: Kitchen behind

{{
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): likely hallway; FAR hallway ahead; Obstacle passable. IMAGE2(Right 30°): likely hallway; NEAR side wall; Obstacle passable. IMAGE3(Right 60°): likely hallway; NEAR side wall; Obstacle passable. IMAGE4(Right 90°): likely hallway; NEAR side wall; Obstacle passable. IMAGE5(Left 120°): likely bedroom transition; FAR bedroom doorway and bed; Obstacle passable. IMAGE6(Left 150°): likely bedroom transition; FAR left-side opening; Obstacle passable. IMAGE7(Back 180°): likely kitchen transition; FAR kitchen-side opening; Obstacle passable. IMAGE8(Right 210°): likely hallway; NEAR wall; Obstacle passable. IMAGE9(Left 240°): likely hallway; NEAR wall and doorframe; Obstacle passable. IMAGE10(Left 270°): likely hallway; NEAR side wall; Obstacle passable. IMAGE11(Left 300°): likely hallway; NEAR side wall; Obstacle passable. IMAGE12(Left 330°): likely hallway; FAR front-side transition; Obstacle passable. Conclusion: Current Position Guess: hallway by the left-side bedroom doorway. Reachable Far Area/Object: IMAGE5-6 lead into the bedroom and support the next bedroom-entry transition toward the bed. Blocked: no critical blocked direction on the left entry route. 2) Map + Space Structure Analysis: the map and space waypoints place the agent in the hallway with the next branch opening into the bedroom on the left, which matches IMAGE5-6. 3) Current Position + Final Task Goal + Task Chain: the current position is Hallway - by left bedroom doorway. The final task goal is Bedroom's bed. Fixed stages: Hallway's forward section -> Bedroom's doorway | Bedroom's doorway -> Bedroom's bed. Task Waypoint Chain: Hallway's forward section(Current) -> Bedroom's doorway -> Bedroom's bed(Goal). Task Progress: Walk through hallway(✓), enter bedroom on left(Current), go to bed. The cross-space entry into the bedroom stays its own current stage, so the bed remains the later in-room target. Task Goal Arrival Check: compare the current localized position and current chain/progress state against the final goal. The current node/task piece is still the bedroom-entry stage, while the final goal node/task piece is the bed stage, so the goal is not yet reached and global_task_finish=false. 4) Subtask Destination + Direction + Subtask Instruction + Landmark: the immediate next destination is Bedroom's doorway because it is the nearest unfinished stage destination and the reachable FAR opening in IMAGE5-6 supports that next task transition. IMAGE5 is the best direction because it most directly and safely leads into the next task-required space. The instruction is to enter toward the bedroom's doorway, and the visible bed is the most relevant concrete landmark for this stage. 5) Plan: short-term plan is to turn toward IMAGE5 and enter the bedroom doorway using the visible bed as the cue. Long-term plan is to finish this doorway transition first, then continue inside the bedroom to the bed.",
    "current_waypoint": "Hallway - by bedroom doorway",
    "waypoint_chain": "Hallway's forward section(Current)→Bedroom's doorway→Bedroom's bed(Goal)",
    "task_progress": "Walk through hallway(✓), enter bedroom on left(Current), go to bed",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "Bedroom's doorway",
    "subtask_instruction": "From IMAGE 5 (Left 120°) view, start, enter toward the bedroom's doorway.",
    "next_waypoint_landmark": "bed",
    "global_task_finish": false
}}

**Critical Rules**:
- **Planning priority**: finish the current nearest unfinished stage before later ones. Preserve task space/object/route-point order, and do not skip intermediate spaces.
- **Reasoning discipline**: Part 1 must stay evidence-only and conclude with Current Position Guess / Reachable Far Area/Object / Blocked. Use Part 1 first to localize exactly where you are, then parse fixed stages, build the chain, and judge Task Goal Arrival Check. For destination/direction, use the current localized position + task order + reachable FAR evidence + space structure to choose the nearest unfinished reachable destination and the best non-backtracking direction.
- **Progress, stop, and output**: `task_progress` must be task-ordered natural-language sub-instructions expanded from the original Task, separated by commas, not waypoint arrows: completed pieces in front with `(✓)`, current piece `(Current)`, later pieces unmarked; if current is already the final task piece, use `(Current, Goal)`. Same-space pass-by / through relations stay inside one task piece; cross-space transitions split into separate task pieces. Task Goal Arrival Check must compare current localized position + current chain/progress state against the final goal, then confirm with visible final-goal evidence. `next_waypoint_destination` must stay in "[room]'s [object/local place]" form and match the current stage only. `next_waypoint_landmark` is the visible object cue for the current subtask: if the destination itself is a visible object/local target, prefer the destination itself; otherwise use a necessary visible intermediate object or empty string. Never use door/doorway/hallway/corridor. `subtask_instruction` must be one short sentence in the fixed form "From [next_waypoint_direction] view, start, ...", using the direct form for direct stages and the path form only when the current task piece itself is a same-stage pass-by / through / around relation. "At entrance" means doorway.
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
