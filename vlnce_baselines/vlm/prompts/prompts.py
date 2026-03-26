"""
VLM规划提示词模板
================
用于LLM高层规划的提示词模板
"""

# 初始规划提示词 - 在任务开始时生成第一个子任务
INITIAL_PLANNING_PROMPT = """**Role**: You are a VLN planning module. Use the observations and map to localize the start position, identify the first reachable task stage, and output precise navigation instructions. No manipulation.

**Global Task**: {instruction}

**Initial state**: You are at the task start. Follow the Global Task strictly from the beginning and complete only the first subtask.

# Inputs
**12 Views** (sampled every 30° around 360°; each RGB view HFOV is about 79°):
- **Obstacle distance**: nearest obstacle only. <0.5m=blocked | 0.5-1.0m=caution | >1.0m=passable
**Map**: explored area + obstacles + current pose
- **Map colors**: White=unexplored | Black=obstacles | Green=safe floor | Dark red=trajectory | Red Arrow=you position

# Reasoning (4 Parts)

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)**
**Format**: For each IMAGE, use `IMAGE# (Direction Angle°): likely [space]; NEAR: ...; FAR: ...; Obstacle: ...; Landmark: ...; Space Waypoint: ...`, and omit any field not actually visible there.
**Evidence order**: When present, analyze only visible evidence in this order: NEAR current/large objects + implied space; FAR objects/openings + implied space; obstacle distance + blocked/caution/passable; landmark name + shown distance; reachable space waypoint + shown distance.
**No hallucination**: analyze each IMAGE separately. If an IMAGE only shows a near wall or close furniture, say only that. Do not write `none`, fill empty slots, invent spaces / FAR objects / landmarks / space waypoint cues, or mention landmark / space waypoint unless explicitly shown.
**Conclusion**: Identify from the 12-view content: Current Position Guess: [current space + NEAR objects + surrounding scene across adjacent views] | Reachable Far Area/Object: [which IMAGEs show reachable FAR spaces/objects/openings, what space/object each one leads to, and which task-relevant transition/route point each one may support] | Blocked: [which IMAGEs have obstacle distance <0.5m]

**2) Current Position + Global Task Goal + Task Chain**
1. **Current Start Position**: localize the start position from Part 1. State both **which space** and **where inside that space** you are now; a room name alone is not enough. Use nearby objects, walls/openings, furniture layout, and continuity across adjacent views first. For `current_waypoint`, prefer a common room type or task-mentioned / task-required space name plus a nearby distinctive local object/place. Prefer task-mentioned objects, route cues, or the clearest nearby anchor. If local surroundings change, update the local object/place wording rather than reusing the same suffix. State it as "[space] - near/by/at [local objects/place]".
2. **Global Task Goal**: state the global task goal only in the form "[space]'s [object/local place]". Also state the nearby objects/local layout that would confirm true arrival there, especially if the final object may be partly hidden.
3. **Parse Full Task**: reason over the natural-language Global Task and break it into ordered fixed stages from the current start position to the global task goal. Each stage must use `"[space]'s [object/local place]" -> "[space]'s [object/local place]"`. Split at cross-space transitions; keep same-space pass by / through / around relations inside one stage. Use task words such as pass / through / enter / toward / after / then to infer order and stage boundaries.
4. **Task Waypoint Chain**: output the ordered stage-anchor chain that matches those stages. The current/start node must also be a full `"[space]'s [object/local place](Current)"`, followed by later stage nodes in the same `"[space]'s [object/local place]"` form until Goal. Never output bare `Start`, `Current`, `WP#`, or room-only nodes. Preserve task order and do not skip intermediate space/object.
5. **Task Progress start**: because this is the task start, nothing is completed yet unless the global task goal is already reached at the start. Write `task_progress` as task-ordered natural-language sub-instructions expanded from the Global Task, separated by commas, not arrows. Mark only the current task piece `(Current)`; later task pieces stay unmarked; do not use `(✓)` unless the global task goal is already reached at the start.
6. **Task Goal Arrival Check**: reason from (a) the current localized position, (b) the global task goal's expected local cues, (c) current 12-view nearby evidence, and (d) the waypoint chain / task progress state. Judge how much of the Global Task is already complete from the ordered stage sequence. If the task says to enter or pass through another space before reaching the goal, do not stop early just because a later object is visible. If you are not yet in the global task goal space, do not stop. Only set `global_task_finish=true` if these cues consistently show that the global task goal `"[space]'s [object/local place]"` is already reached within about 1.5m or the 12 views clearly show arrival there; otherwise false.
**Order**: localize first, then global task goal and expected cues, then fixed stages, then chain/progress, then Task Goal Arrival Check.

**3) Subtask Destination + Direction + Subtask Instruction + Landmark**
1. In initial planning, use only the first task stage. From the current localized position, the Global Task's ordered spaces/objects/route points, Part 1's reachable FAR spaces/objects, and the known space transitions, identify the first subtask destination: the next task-relevant `"[space]'s [object]"` that should be reached now. Advance spaces / route points / room transitions one by one in order. Use natural-language task reasoning to decide what is current versus later: if the task first requires entering another space, keep the destination on that cross-space stage; if the first stage is a same-space pass-by / through relation, keep its final object/local place as the destination; do not jump to a later-stage target just because it is visible.
2. Choose the direction from the 12 IMAGEs that best reaches that destination. Prefer where the destination itself appears, or the FAR space/object/opening that most directly and safely leads to the needed next space/route point. Prefer open directions with obstacle distance >1.0m, ideally >2.0m; avoid blocked/tight views and backtracking unless clearly required. The chosen direction must match task order and the current -> next relation, not just the most open view.
3. Write one short immediate subtask instruction for that same first stage. It must directly match the chosen destination and direction, and describe only the nearest unfinished task piece. Reflect the inferred task relation, such as enter / pass through / go around / continue toward, without leaking later-stage goals into the current stage. Prefer one of these concise forms when appropriate:
   - Direct approach / enter: `From [next_waypoint_direction] view, start, [move/enter/approach] toward [destination].`
   - Via visible cue / obstacle bypass: `From [next_waypoint_direction] view, start, [pass/go through/go around/cross] [visible cue], then [enter/approach/continue toward] [destination].`
4. Choose `subtask_landmark` only for that first stage. It is the most useful visible concrete object cue for executing the current subtask now. If the destination itself is visible, prefer it; otherwise use a necessary visible intermediate object; if none is useful, use an empty string.

**4) Plan**
1. **Short-term Plan**: explain why this destination, direction, and subtask instruction are the correct immediate plan. Summarize the next move, turn if any, short-term target, and landmark choice if any, and make sure they match. State why this is the current unfinished stage from the Global Task order rather than a later stage.
2. **Long-term Plan**: summarize the remaining stage order after this subtask as a forward plan. Keep task-completion reasoning active: if a doorway / hallway / entrance crossing would place you into the next required space, that cross-space stage can finish there and the following in-room stage becomes next; if that new space is also the global task goal space and the goal cues are already satisfied, the task can end.

**Sequential planning rule**:
- Output only the immediate next task stage/subtask for current-stage progress; do not plan stage +2/+3 before stage +1 is finished.

# Output (JSON only)

{{
    "reasoning": "<Follow the reasoning flow above exactly. Part 1 must analyze IMAGE1, IMAGE2, ... IMAGE12 separately without merging and conclude with Current Position Guess / Reachable Far Area/Object / Blocked. Part 2 must reason: where the current start position is, what the global task goal is in '[space]'s [object/local place]' form, what local cues define that goal place, how the Global Task splits into fixed stages, what the waypoint chain and task-progress sub-instructions are, how much of the Global Task is already complete, and whether Task Goal Arrival Check is satisfied now by comparing current localized position + current 12-view local evidence + chain/progress state against the global task goal. Then continue with Parts 3-4.>",
    "current_waypoint": "<Room - current local place from nearby observations. Prefer a common or task-mentioned space name plus a nearby distinctive task-relevant/local object or place. Do not keep reusing the same object suffix when the local surroundings have changed.>",
    "waypoint_chain": "<Ordered stage-anchor chain using one [room]'s [object/local place] node per fixed stage endpoint. The current/start node must also be a full [room]'s [object/local place](Current), not a bare Start/Current/WP# label. Do not mark (✓) in initial planning unless already at goal>",
    "task_progress": "<Task-ordered natural-language sub-instructions expanded from the original Global Task, not waypoint arrows. Use commas to separate sub-instructions. Same-space pass-by/through relations stay in one task piece; cross-space transitions are separate task pieces. In initial planning only the current task piece is (Current); later task pieces are unmarked unless already at goal>",
    "next_waypoint": "<Room's object/local place; if the subtask target is the landmark itself, explicitly include that landmark word>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "subtask_instruction": "<Exactly one short sentence in one of two forms: direct `From [next_waypoint_direction] view, start, [move/enter/approach] toward [destination].` or same-stage path `From [next_waypoint_direction] view, start, [pass/go through/go around/cross] [visible cue], then [enter/approach/continue toward] [destination].` Use the path form only when that cue-destination relation belongs to the same current task piece>",
    "subtask_landmark": "<Single clear recognizable visible object phrase if useful; otherwise empty string. This is the visible object cue for the current subtask. If the subtask destination itself is a visible object/local target, prefer the destination itself. Prefer specific objects like bed/rug/chair/table/lamp, not broad areas/spaces. NEVER use door/doorway/hallway/corridor>",
    "global_task_finish": "<true only if Task Goal Arrival Check reasons from current position + current waypoint chain/task progress + visible goal evidence that you have already reached the correct global task goal space and global task goal object/local place; otherwise false>"
}}

#Examples (abbreviated):

## Ex1: Bathroom to Giraffi via exercise room
**Global Task**: Turn around, walk through the exercise room into the living room. Wait by the Giraffi.
**Obs:** IMAGE 4-6: exercise-room opening and exercise equipment. IMAGE 8-11: bathroom sink / vanity / toilet. IMAGE 5: clearest open route toward the exercise room.

{{
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): likely hallway/transition edge; NEAR bookshelf; Obstacle passable. IMAGE2(Left 30°): likely bathroom boundary; NEAR window and wall; Obstacle caution. IMAGE3(Left 60°): likely bathroom boundary; NEAR window; Obstacle caution. IMAGE4(Left 90°): likely exercise-room transition; FAR exercise equipment and opening; Obstacle caution. IMAGE5(Left 120°): likely exercise room; FAR exercise equipment and deeper opening toward the living room; Obstacle passable. IMAGE6(Left 150°): likely exercise-room entrance; NEAR door frame; FAR exercise-room interior; Obstacle caution. IMAGE7(Back 180°): likely bathroom; NEAR wooden door; Obstacle caution. IMAGE8(Right 150°): likely bathroom; NEAR vanity and sink; Obstacle caution. IMAGE9(Right 120°): likely bathroom; NEAR vanity and mirror; Obstacle caution. IMAGE10(Right 90°): likely bathroom; NEAR toilet and vanity; Obstacle caution. IMAGE11(Right 60°): likely bathroom; NEAR toilet and wall; Obstacle caution. IMAGE12(Right 30°): likely bathroom boundary; NEAR wall and bookshelf edge; Obstacle caution. Conclusion: Current Position Guess: bathroom doorway by the sink/vanity side, facing the exercise-room opening. Reachable Far Area/Object: IMAGE4-6 open toward the exercise room; IMAGE5 most clearly shows the exercise equipment and the onward route toward the living room, so it best supports the next task transition. Blocked: none strictly <0.5m, but most bathroom-side views are tight and only the forward-left route is clearly usable. 2) Current Position + Global Task Goal + Task Chain: from the sink/toilet/vanity evidence in IMAGE8-11 and the exercise-room opening in IMAGE4-6, the current start position is Bathroom - at doorway by sink and vanity, facing the exercise room. The global task goal is Living room's Giraffi. The expected goal-place cues are living-room local evidence around the Giraffi, not bathroom fixtures or the exercise-room transition. Fixed stages: Bathroom's doorway -> Exercise room's exercise equipment | Exercise room's exercise equipment -> Living room's entrance | Living room's entrance -> Living room's Giraffi. Task Waypoint Chain: Bathroom's doorway(Current) -> Exercise room's exercise equipment -> Living room's entrance -> Living room's Giraffi(Goal). Task Progress start: Turn around and enter the exercise room toward the exercise equipment(Current), continue through the exercise room into the living room, wait by the Giraffi. Cross-space movement stays split by space change, so the current piece is only the bathroom-to-exercise-room transition. Task Goal Arrival Check: compare the current localized position, the expected living-room Giraffi cues, and the current chain/progress state against the global task goal. The current nearby evidence is still bathroom sink/toilet/vanity plus the exercise-room opening, and the current node/task piece is still the bathroom start stage, so the goal is not reached and global_task_finish=false. 3) Subtask Destination + Direction + Subtask Instruction + Landmark: the first-stage destination is Exercise room's exercise equipment because the task must advance into the next space first, and IMAGE4-6 show that reachable transition with IMAGE5 giving the clearest open route. IMAGE5 is the best direction because it most directly and safely enters the exercise room toward the first task-relevant object. The immediate instruction is to enter toward the exercise room's exercise equipment. The landmark is exercise equipment because the destination itself is a visible task-relevant object. 4) Plan: short-term plan is to turn toward IMAGE5 and enter the exercise room using the visible exercise equipment as the cue because that is still the current unfinished stage from the Global Task order. Long-term plan is to reach the exercise room first, then continue into the living room, and finally wait by the Giraffi.",
    "current_waypoint": "Bathroom - at doorway by sink and vanity",
    "waypoint_chain": "Bathroom's doorway(Current)→Exercise room's exercise equipment→Living room's entrance→Living room's Giraffi(Goal)",
    "task_progress": "Turn around and enter the exercise room toward the exercise equipment(Current), continue through the exercise room into the living room, wait by the Giraffi",
    "next_waypoint": "Exercise room's exercise equipment",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "subtask_instruction": "From IMAGE 5 (Left 120deg) view, start, enter toward the exercise room's exercise equipment.",
    "subtask_landmark": "exercise equipment",
    "global_task_finish": false
}}

**Critical Rules**:
- **Planning priority**: finish the nearest unfinished stage first. In initial planning, finish only the first stage, preserve task space/object/route-point order, and do not skip intermediate spaces.
- **Reasoning discipline**: Part 1 must stay evidence-only and conclude with Current Position Guess / Reachable Far Area/Object / Blocked. Use Part 1 first to localize exactly where you are, then parse the natural-language Global Task into fixed stages, build the chain, judge task completion, and then judge Task Goal Arrival Check. For destination/direction/instruction, use current localized position + task order + reachable FAR evidence + known space transitions to choose the next reachable non-backtracking destination.
- **Current waypoint naming**: `current_waypoint` must be tied to the actual current local evidence, not copied mechanically from earlier waypoints. Prefer a common or task-mentioned space name, and choose a nearby distinctive local object/place that is task-mentioned, task-relevant, or immediately beside the agent. When position/local surroundings change, the `- object/place` part should also change so nearby waypoints stay distinguishable.
- **Progress, stop, and output**: `task_progress` must be task-ordered natural-language sub-instructions separated by commas, not waypoint arrows: completed pieces in front, current piece `(Current)`, later pieces unmarked unless already at goal. Same-space pass-by / through relations stay inside one task piece; cross-space transitions split into separate task pieces. `waypoint_chain` must use the full `"[room]'s [object/local place]"` form for every node, including the current/start node and the goal node; never output bare `Start`, `Current`, `WP#`, or room-only nodes. `next_waypoint` must stay in "[room]'s [object/local place]" form and match the current stage only. `subtask_landmark` is the visible object cue for the current subtask: prefer the destination itself when visible; otherwise use a necessary visible intermediate object or empty string. Never use door/doorway/hallway/corridor. `subtask_instruction` must be one short sentence in the fixed output form below. "At entrance" means doorway.
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """**Role**: You are a VLN verification and replanning module. Use the space structure, views, and maps to verify subtask completion, localize the current position, and plan the next subtask. No manipulation.
{verify_replan_prompt_notice}

**Global Task**: {instruction}

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
**Evidence order**: When present, analyze only visible evidence in this order: NEAR current/large objects + implied space; FAR objects/openings + implied space; obstacle distance + blocked/caution/passable; landmark name + shown distance; reachable space waypoint + shown distance.
**No hallucination**: analyze each IMAGE separately. If an IMAGE only shows a near wall or close furniture, say only that. Do not write `none`, fill empty slots, invent spaces / FAR objects / landmarks / space waypoint cues, or mention landmark / space waypoint unless explicitly shown.
**Conclusion**: Identify from the 12-view content: Current Position Guess: [current space + NEAR objects + surrounding scene across adjacent views] | Reachable Far Area/Object: [which IMAGEs show reachable FAR spaces/objects/openings, what space/object each one leads to, and which task-relevant transition/route point each one may support] | Blocked: [which IMAGEs have obstacle distance <0.5m]

**2) Map + Space Structure Analysis**
Use only the Map and Space Structure in this part.
1. **Identify Current Area**: read the current area from the map and space structure, but always cross-check it with Part 1 nearby evidence. If the map current area is `Unknown` or conflicts with nearby evidence, resolve it from the strongest consistent 12-view nearby evidence.
2. **Read Space Waypoint list**: for each `Space WP#...`, identify its space, local cues, direction/distance relative to current, whether it is directly reachable now, and which waypoint(s) it should be reached through if not.
3. **Read Space Waypoint Chain**: treat each chain node as a full space waypoint plus its local object/place meaning, not a bare room/WP id. Use it to know which spaces/waypoints are already visited, which are current / next / behind, and which visible openings would cause wrong-space entry, repetition, or backtracking.
4. **Read Map**: check current position, where other spaces lie, and which side/opening corresponds to which waypoint/space.
5. **Structure Conclusion**: conclude `Current space/area from structure | current/next/behind space waypoint(s) | reachable task-aligned space transition(s) | wrong-space / backtracking transition(s) to avoid`.

**3) Current Position + Global Task Goal + Task Chain**
1. **Current Position**: localize the current position from Part 1. State both **which space** and **where inside that space** you are now. Use nearby objects, walls/openings, furniture layout, and continuity across neighboring views first; map pose and space structure are support only. For `current_waypoint`, prefer a common room type or task-mentioned / task-required space name plus a nearby distinctive local object/place. Prefer task-mentioned objects, route cues, or the clearest nearby anchor. If local surroundings change, update the local object/place wording rather than reusing the same suffix. State it as "[space] - near/by/at [local objects/place]".
2. **Global Task Goal**: state the global task goal only in the form "[space]'s [object/local place]". Also state the nearby objects/local layout that would confirm true arrival there, especially if the final object may be partly hidden.
3. **Parse Full Task**: reason over the natural-language Global Task and break it into ordered fixed stages from the current start position to the global task goal. Each stage must use `"[space]'s [object/local place]" -> "[space]'s [object/local place]"`. Split at cross-space transitions; keep same-space pass by / through / around relations inside one stage. Use task words such as pass / through / enter / toward / after / then to infer order and stage boundaries.
4. **Task Waypoint Chain**: build the ordered stage-anchor chain and anchor it to the already-judged **Current Position** plus Part 2's current/next space-waypoint state. Every node, including the matched current node and the goal node, must stay in full `"[space]'s [object/local place]"` form. The matching node must be `(Current)`; nodes before it are `(✓)`; nodes after it are unmarked; if the current node is already the global task goal, mark it `(Current, Goal)`. Never output bare `Current`, `Goal`, `WP#`, or room-only nodes. Use the Space Waypoint Chain to judge whether a cross-space stage has already finished, for example if a doorway / hallway / entrance has already been passed and the agent is now in the next required space.
5. **Task Progress marking**: write `task_progress` as task-ordered natural-language sub-instructions expanded from the Global Task, separated by commas, not arrows. Completed task pieces stay in front with `(✓)`, the current unfinished task piece is `(Current)` ONE only, and later unfinished task pieces remain after it unmarked. Judge completion from ordered task meaning plus current space evidence, not single landmark matches alone.
6. **Task Goal Arrival Check**: reason from (a) current localized position, (b) expected global-task-goal local cues, (c) current 12-view nearby evidence, and (d) structure / waypoint list / Space Waypoint Chain / task waypoint chain / task progress. Use the current node and goal node as complete space + object/local-place anchors when checking whether the goal is already reached. Reason how much of the Global Task is complete and whether the current stage should finish, advance, or stop. If you are not yet in the global task goal space, do not stop. Only set `global_task_finish=true` if these cues consistently show that the global task goal `"[space]'s [object/local place]"` is already reached within about 1.5m or the 12 views clearly show arrival there; otherwise false.
**Order**: localize first, then Part 2 structure state, then global task goal and expected cues, then fixed stages, then chain/progress, then Task Goal Arrival Check.

**4) Subtask Destination + Direction + Subtask Instruction + Landmark**
1. Based on the current position, Part 2's structure conclusion, Space Waypoint Chain, waypoint chain, 12-view observations, visible space waypoint cues, and the full Global Task, identify the immediate next subtask destination: the nearest unfinished task-relevant space/object/local place that should be advanced now. Advance spaces / route points / room transitions one by one in order. Use natural-language task reasoning plus the Space Waypoint Chain to judge whether the current stage is still unfinished or has already switched to the next stage. If the current task piece is a same-space pass-by / through relation, keep its final object/local place as the destination; if the task moves into another space, use the next-space entry/object for the current stage, not a later-stage target.
2. If the current subtask is unfinished, continue it; if it is finished, advance to the next unfinished stage. Do not skip a nearer unfinished stage. Preserve waypoint order and task relations such as pass-by / toward / enter / after / then. For example, if the task said to enter another room first and the Space Waypoint Chain plus current evidence show that doorway / hallway / entrance has already been passed and you are now in that required next room, mark that cross-space stage finished and move to the next stage; if that new room/place already satisfies the global task goal, stop.
3. Choose the direction from the 12 IMAGEs plus the space structure that best reaches that destination. Judge it from where the destination itself appears, or which reachable FAR space/object/opening/space-waypoint view most directly and safely leads toward the needed next space/route point. Use Part 2 and the Space Waypoint Chain to decide which view leads into the **correct next space** and which open views would enter a wrong, repeated, or backtracking space. Prefer open directions with obstacle distance >1.0m, ideally >2.0m; avoid blocked/tight views and backtracking unless clearly required. The chosen direction must match the current unfinished stage from the Global Task order.
4. Write the next subtask instruction for only that nearest unfinished stage. It must directly match the chosen destination and direction. It must also match the same current -> next step implied by the Space Waypoint Chain and task waypoint chain. If the current task piece is a same-space pass-by / through / around relation, keep that cue + destination in one instruction; if the task changes to another space, keep the instruction only on the current cross-space stage and leave later-stage targets for later. Reflect the task relation inferred from the Global Task rather than a generic move command. Prefer one of these concise forms when appropriate:
   - Direct approach / enter: `From [next_waypoint_direction] view, start, [move/enter/approach] toward [destination].`
   - Via visible cue / obstacle bypass: `From [next_waypoint_direction] view, start, [pass/go through/go around/cross] [visible cue], then [enter/approach/continue toward] [destination].`
5. Choose the most relevant visible subtask landmark. If the destination itself is a visible object/local target, prefer the destination itself as `subtask_landmark`. Otherwise use a necessary visible intermediate object; if none is useful, use an empty string.

**5) Plan**
1. **Short-term Plan**: explain why this destination, direction, and subtask instruction are the correct immediate plan. Summarize the next move, turn if any, short-term target, and landmark choice if any, and make sure they match. State why this is the current unfinished stage from the Global Task order, and why alternative visible openings/directions would be later-stage, wrong-space, repeated, or backtracking choices.
2. **Long-term Plan**: summarize the remaining stage order after this subtask as a forward plan. Keep task-completion reasoning active: if the current cross-space stage finishes upon entering the next required room, the next in-room stage should become current; if the current room/place already satisfies the global task goal cues, the plan is complete.

**Sequential planning rule**:
- If the current subtask is unfinished, continue it; only after completion can `next_waypoint` move to the next stage.
- If you are already inside the correct global task goal space and the global task goal object/local place is within about 1.5m, or the 12 views clearly show you have already arrived there, STOP and set `global_task_finish=true`.

# Output (JSON only)

{{
    "reasoning": "<Follow the reasoning flow above exactly. Part 1 must analyze IMAGE1, IMAGE2, ... IMAGE12 separately without merging and conclude with Current Position Guess / Reachable Far Area/Object / Blocked. Part 2 must explicitly analyze the space structure: current/next/behind space waypoint(s), directly reachable vs via-which-waypoint transitions, and wrong-space/backtracking options to avoid. Part 3 must reason: where the current position is, what the global task goal is in '[space]'s [object/local place]' form, what local cues define that goal place, how the Global Task splits into fixed stages, how much of that ordered task is already complete, what the waypoint chain and task-progress sub-instructions are, and whether Task Goal Arrival Check is satisfied now by comparing current localized position + current 12-view local evidence + structure + waypoint list + Space Waypoint Chain + waypoint chain/task progress state against the global task goal. Then continue with Parts 4-5, using Part 2 and the Space Waypoint Chain to justify destination, direction, and whether the current stage should continue, advance, or stop.>",
    "current_waypoint": "<Room - current local place from nearby observations. Prefer a common or task-mentioned space name plus a nearby distinctive task-relevant/local object or place. Do not keep reusing the same object suffix when the local surroundings have changed.>",
    "waypoint_chain": "<Ordered stage-anchor chain using one [room]'s [object/local place] node per fixed stage endpoint, anchored to the judged current position. Every node, including current and goal, must stay in full [room]'s [object/local place] form; never output bare Current/Goal/WP# labels. Nodes before current=(✓), the matched current node=(Current), nodes after current=unmarked, and if current is the goal use (Current, Goal)>",
    "task_progress": "<Task-ordered natural-language sub-instructions expanded from the original Global Task, not waypoint arrows. Use commas to separate sub-instructions. Same-space pass-by/through relations stay in one task piece; cross-space transitions are separate task pieces. Completed task pieces stay in front with (✓), current task piece=(Current) ONE only, later task pieces after it are unmarked. All task pieces (✓)=complete>",
    "next_waypoint": "<Room's object/local place; if the subtask target is the landmark itself, explicitly include that landmark word>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "subtask_instruction": "<Exactly one short sentence in one of two forms: direct `From [next_waypoint_direction] view, start, [move/enter/approach] toward [destination].` or same-stage path `From [next_waypoint_direction] view, start, [pass/go through/go around/cross] [visible cue], then [enter/approach/continue toward] [destination].` Use the path form only when that cue-destination relation belongs to the same current task piece>",
    "subtask_landmark": "<Single clear recognizable visible object phrase if useful; otherwise empty string. This is the visible object cue for the current subtask. If the subtask destination itself is a visible object/local target, prefer the destination itself. Prefer specific objects like bed/rug/chair/table/lamp, not broad areas/spaces. NEVER use door/doorway/hallway/corridor>",
    "global_task_finish": "<true only if Task Goal Arrival Check reasons from current position + Space Waypoint Chain + current waypoint chain/task progress + visible goal evidence that you have already reached the correct global task goal space and global task goal object/local place; otherwise false>"
}}

# Examples (abbreviated):

## Ex1: Rug arrival
**Global Task**: Exit bedroom, turn left. Walk straight passing gray couch, stop at rug.
**Previous**: Navigate past gray couch toward rug
**Obs:** IMAGE 1: Rug <0.5m. IMAGE 10: Gray couch beside. IMAGE 7: Hallway behind

{{
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): likely living room; NEAR rug; Obstacle passable. IMAGE2(Right 30°): likely living room; NEAR rug edge; Obstacle passable. IMAGE3(Right 60°): likely living room; NEAR nearby furniture; Obstacle passable. IMAGE4(Right 90°): likely living room; NEAR nearby furniture; Obstacle passable. IMAGE5(Left 120°): likely living room; NEAR living-room furniture; Obstacle passable. IMAGE6(Left 150°): likely living room; NEAR living-room furniture; Obstacle passable. IMAGE7(Back 180°): likely hallway transition; FAR hallway opening; Obstacle passable. IMAGE8(Right 210°): likely living room; NEAR side wall and furniture; Obstacle passable. IMAGE9(Left 240°): likely living room; NEAR gray couch; Obstacle passable. IMAGE10(Left 270°): likely living room; NEAR gray couch; Obstacle passable. IMAGE11(Left 300°): likely living room; NEAR rug and couch area; Obstacle passable. IMAGE12(Left 330°): likely living room; NEAR rug edge; Obstacle passable. Conclusion: Current Position Guess: living room rug area beside the gray couch. Reachable Far Area/Object: IMAGE7 leads back toward the hallway opening. Blocked: none critical. 2) Map + Space Structure Analysis: the current pose lies in the living room area. The visible and mapped space waypoint state shows the hallway waypoint is behind/backtracked, the living room waypoint area is current, and no further task-relevant transition ahead is needed. The only notable farther transition is the hallway opening behind in IMAGE7, which is a previous-space route and should not be chosen. Structure Conclusion: current space/area=living room rug zone | current waypoint=living room | behind waypoint=hallway opening | reachable task-aligned transition=none needed because already at goal | avoid=backtracking to hallway. 3) Current Position + Global Task Goal + Task Chain: the current position is Living room - at rug beside gray couch. The global task goal is Living room's rug. The expected goal-place cues are the rug itself plus the nearby gray couch / living-room rug zone. Fixed stages: Bedroom's entrance -> Hallway's entrance opening | Hallway's entrance opening -> Living room's gray couch | Living room's gray couch -> Living room's rug. Task Waypoint Chain: Bedroom's entrance(✓) -> Hallway's entrance opening(✓) -> Living room's gray couch(✓) -> Living room's rug(Current, Goal). Task Progress: Exit bedroom(✓), turn left(✓), walk straight passing gray couch(✓), stop at rug(Current, Goal). Task Goal Arrival Check: compare the current localized position, the expected rug-area cues, the current 12-view nearby evidence, and the structure/chain/progress state against the global task goal. The current node is already the goal node, the current task piece is already the final task piece, and the rug with its nearby couch context is directly beside the agent in the 12 views, so the goal is satisfied now and global_task_finish=true. 4) Subtask Destination + Direction + Subtask Instruction + Landmark: the immediate destination remains Living room's rug because it is the global task goal and it is already reached. The structure analysis also says the hallway transition behind is old space and must not be used. IMAGE1 can remain the reference direction because the rug is directly present there. The instruction is to stop at the living room's rug, and the rug is the clearest visible landmark. 5) Plan: short-term plan is to stop at the current rug position with no further movement because the Global Task is already complete. Long-term plan is complete because all fixed stages are already finished.",
    "current_waypoint": "Living Room - at rug beside gray couch",
    "waypoint_chain": "Bedroom's entrance(✓)→Hallway's entrance opening(✓)→Living room's gray couch(✓)→Living room's rug(Current, Goal)",
    "task_progress": "Exit bedroom(✓), turn left(✓), walk straight passing gray couch(✓), stop at rug(Current, Goal)",
    "next_waypoint": "Living room's rug",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "subtask_instruction": "From IMAGE 1 (Front 0°) view, start, stop at the living room's rug.",
    "subtask_landmark": "rug",
    "global_task_finish": true
}}

## Ex2: Hallway to bedroom
**Global Task**: Walk through hallway, then enter bedroom on left and go to bed.
**Previous**: Navigate through hallway
**Obs:** IMAGE 1: Hallway ahead 3.0m. IMAGE 5: Bedroom doorway (~2.5m), bed inside. IMAGE 7: Kitchen behind

{{
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): likely hallway; FAR hallway ahead; Obstacle passable. IMAGE2(Right 30°): likely hallway; NEAR side wall; Obstacle passable. IMAGE3(Right 60°): likely hallway; NEAR side wall; Obstacle passable. IMAGE4(Right 90°): likely hallway; NEAR side wall; Obstacle passable. IMAGE5(Left 120°): likely bedroom transition; FAR bedroom doorway and bed; Obstacle passable. IMAGE6(Left 150°): likely bedroom transition; FAR left-side opening; Obstacle passable. IMAGE7(Back 180°): likely kitchen transition; FAR kitchen-side opening; Obstacle passable. IMAGE8(Right 210°): likely hallway; NEAR wall; Obstacle passable. IMAGE9(Left 240°): likely hallway; NEAR wall and doorframe; Obstacle passable. IMAGE10(Left 270°): likely hallway; NEAR side wall; Obstacle passable. IMAGE11(Left 300°): likely hallway; NEAR side wall; Obstacle passable. IMAGE12(Left 330°): likely hallway; FAR front-side transition; Obstacle passable. Conclusion: Current Position Guess: hallway by the left-side bedroom doorway. Reachable Far Area/Object: IMAGE5-6 lead into the bedroom and support the next bedroom-entry transition toward the bed. Blocked: no critical blocked direction on the left entry route. 2) Map + Space Structure Analysis: the map and space waypoint list place the agent in the hallway area with a reachable bedroom waypoint on the left and another kitchen-side transition behind. The space waypoint chain says hallway is current, bedroom is next, and the kitchen-side branch is not the current task target. IMAGE5-6 align with the next bedroom waypoint, while IMAGE7 is a wrong-space/backtracking option for this task. Structure Conclusion: current space/area=hallway | current waypoint=hallway section | next waypoint=bedroom entry on left | behind/wrong-side transition=kitchen opening behind | reachable task-aligned transition=left bedroom entry via IMAGE5-6. 3) Current Position + Global Task Goal + Task Chain: the current position is Hallway - by left bedroom doorway. The global task goal is Bedroom's bed. The expected goal-place cues are bedroom-local evidence around the bed, not hallway walls plus an entry opening. Fixed stages: Hallway's forward section -> Bedroom's doorway | Bedroom's doorway -> Bedroom's bed. Task Waypoint Chain: Hallway's forward section(Current) -> Bedroom's doorway -> Bedroom's bed(Goal). Task Progress: Walk through hallway(✓), enter bedroom on left(Current), go to bed. The cross-space entry into the bedroom stays its own current stage, so the bed remains the later in-room target. Task Goal Arrival Check: compare the current localized position, the expected bedroom-bed cues, the current 12-view nearby evidence, and the structure/waypoint/chain/progress state against the global task goal. The current area/node is still hallway-side entry rather than the final bedroom-bed node, and the nearby evidence is still hallway boundary plus doorway transition, so the goal is not yet reached and global_task_finish=false. 4) Subtask Destination + Direction + Subtask Instruction + Landmark: the immediate next destination is Bedroom's doorway because it is the nearest unfinished stage destination and the reachable FAR opening in IMAGE5-6 supports that next task transition. The space structure also confirms the left bedroom waypoint is next and the kitchen-side opening behind is wrong for this stage. IMAGE5 is the best direction because it most directly and safely leads into the next task-required space. The instruction is to enter toward the bedroom's doorway, and the visible bed is the most relevant concrete landmark for this stage. 5) Plan: short-term plan is to turn toward IMAGE5 and enter the bedroom doorway using the visible bed as the cue because the cross-space entry is still the current unfinished stage. Long-term plan is to finish this doorway transition first, then continue inside the bedroom to the bed.",
    "current_waypoint": "Hallway - by bedroom doorway",
    "waypoint_chain": "Hallway's forward section(Current)→Bedroom's doorway→Bedroom's bed(Goal)",
    "task_progress": "Walk through hallway(✓), enter bedroom on left(Current), go to bed",
    "next_waypoint": "Bedroom's doorway",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "subtask_instruction": "From IMAGE 5 (Left 120°) view, start, enter toward the bedroom's doorway.",
    "subtask_landmark": "bed",
    "global_task_finish": false
}}

**Critical Rules**:
- **Planning priority**: finish the current nearest unfinished stage before later ones. Preserve task space/object/route-point order, and do not skip intermediate spaces.
- **Reasoning discipline**: Part 1 must stay evidence-only and conclude with Current Position Guess / Reachable Far Area/Object / Blocked. Part 2 must explicitly analyze the space structure: which waypoint(s) are current / next / behind, which are directly reachable or require via-waypoint transitions, and which visible/open directions would enter the wrong space. Use Part 1 first to localize exactly where you are, then use Part 2 plus the natural-language Global Task to anchor chain/progress, judge ordered task completion, and decide whether the current stage should continue, advance, or stop.
- **Current waypoint naming**: `current_waypoint` must be tied to the actual current local evidence, not copied mechanically from earlier waypoints. Prefer a common or task-mentioned space name, and choose a nearby distinctive local object/place that is task-mentioned, task-relevant, or immediately beside the agent. When position/local surroundings change, the `- object/place` part should also change so nearby waypoints stay distinguishable.
- **Progress, stop, and output**: `task_progress` must be task-ordered natural-language sub-instructions separated by commas, not waypoint arrows: completed pieces in front with `(✓)`, current piece `(Current)`, later pieces unmarked; if current is already the final task piece, use `(Current, Goal)`. Same-space pass-by / through relations stay inside one task piece; cross-space transitions split into separate task pieces. Task Goal Arrival Check must compare current localized position + expected global-task-goal local cues + current 12-view nearby evidence + structure + waypoint list + Space Waypoint Chain + waypoint chain/progress state against the global task goal. For current progress, arrival, next destination, next direction, and next instruction, always reason with the Space Waypoint Chain together with the current task stage/progress. If these comparisons show the global task goal area is already reached, stop. If the task required crossing a doorway / hallway / entrance and the Space Waypoint Chain plus current evidence show that crossing is already finished, advance to the next stage; if the resulting current room/place already satisfies the global task goal cues, stop. `waypoint_chain` must use the full `"[room]'s [object/local place]"` form for every node, including current and goal; never output bare `Current`, `Goal`, `Start`, `WP#`, or room-only nodes. `next_waypoint` must stay in "[room]'s [object/local place]" form and match the current stage only. `next_waypoint_direction` must match the task-aligned current/next space waypoint relation from Part 2 and reject wrong-space/backtracking openings even if they are open. `subtask_landmark` is the visible object cue for the current subtask: prefer the destination itself when visible; otherwise use a necessary visible intermediate object or empty string. Never use door/doorway/hallway/corridor. `subtask_instruction` must be one short sentence in the fixed output form below. "At entrance" means doorway.
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
                                       waypoint_summary: str = None,
                                       verify_replan_prompt_notice: str = None) -> str:
    """
    获取验证和重规划提示词
    
    Args:
        instruction: 完整导航指令
        subtask_destination: 当前子任务目的地
        subtask_instruction: 当前子任务指令
        action_space: 动作空间描述
        detected_landmarks: 已检测到的landmark类别字符串
        waypoint_summary: 空间结构字符串
        verify_replan_prompt_notice: verify/replan 顶部附加提示
        
    Returns:
        格式化的提示词字符串
    """
    if not detected_landmarks:
        detected_landmarks = "No landmarks detected yet"
    if not waypoint_summary:
        waypoint_summary = "No space structure recorded yet"
    if verify_replan_prompt_notice:
        verify_replan_prompt_notice = f"**Stuck Notice**: {verify_replan_prompt_notice.strip()}"
    else:
        verify_replan_prompt_notice = ""
    
    return VERIFICATION_REPLANNING_PROMPT.format(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        action_space=action_space,
        detected_landmarks=detected_landmarks,
        waypoint_summary=waypoint_summary,
        verify_replan_prompt_notice=verify_replan_prompt_notice,
    )
