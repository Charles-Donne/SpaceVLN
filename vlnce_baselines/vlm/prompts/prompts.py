"""
VLM规划提示词模板
================
用于LLM高层规划的提示词模板
"""

from vlnce_baselines.config.core.params.thresholds import (
    ARRIVAL_NEAR_M,
    OBS_BLOCKED_M,
    OBS_OPEN_M,
    OBS_RISKY_M,
)


def _fmt_threshold_m(value: float) -> str:
    text = f"{float(value):.2f}"
    if text.endswith("00"):
        return f"{float(value):.1f}"
    return text.rstrip("0").rstrip(".")

# 初始规划提示词 - 在任务开始时生成第一个子任务
INITIAL_PLANNING_PROMPT = """**Role**: You are a VLN planning module. Use the views and map to localize the start position, identify the first reachable task stage, and output precise navigation instructions. No manipulation.

**Global Task**: {instruction}

**Initial state**: You are at the task start. Follow the Global Task from the beginning and complete only the first subtask.

# Inputs
**12 Views** (sampled every 30° around 360°; each RGB view HFOV is about 79°):
- **Obstacle distance**: nearest obstacle only. <{obs_blocked_m}m=blocked | {obs_blocked_m}-{obs_risky_m}m=caution | >{obs_risky_m}m=passable
- **In-view distance labels**: when shown, `Obstacle` and `Landmark` display meters; use only the shown value.
**Map**: explored area + obstacles + current pose
- **Map colors**: White=unexplored | Black=obstacles | Green=safe floor | Dark red=trajectory | Red Arrow=you position

# Reasoning (4 Parts)

**1) 12-View Analysis (MUST analyze EACH IMAGE 1-12)**
**Format**: For each IMAGE, use `IMAGE# (Direction Angle°): likely [space]; NEAR: ...; FAR: ...; Obstacle: ...; Landmark: ...`, omitting any field not visible there.
**Distance reading**: `Obstacle` and `Landmark` refer only to that IMAGE's shown value; do not infer hidden values.
**Near rule**: treat only cues within about {arrival_near_m}m as truly NEAR/current-position evidence. This is for localization/progress only, not auto-stop. Farther cues may support direction/future destination but do not prove arrival.
**Evidence order**: read each IMAGE in this order: NEAR current/large objects + implied space; FAR objects/openings + implied space; obstacle distance + blocked/caution/passable; landmark + shown distance. Judge RGB/layout first and use labels/distances only as support. If a label conflicts with the scene, trust the scene. Use layout, openings, furniture relations, and adjacent-view consistency. In stair scenes, infer upstairs/downstairs from stair edges, rise/drop trend, landing continuity, railings, adjacent views, and any open no-floor/drop side; infer connector from layout, not landmark detections.
**No hallucination**: analyze each IMAGE separately. If an IMAGE shows only a wall or nearby furniture, say only that. Do not write `none`, fill empty slots, invent spaces/FAR objects/landmarks, or mention landmark unless explicitly shown.
**Conclusion**: Identify from the 12-view content: Current Position Guess: [current space + NEAR landmarks + adjacent-view context] | Reachable Far Area/Landmark: [which IMAGEs show reachable FAR spaces/landmarks/openings, what each leads to, and which task-relevant transition/route point it may support] | Blocked: [which IMAGEs have obstacle distance <{obs_blocked_m}m]

**2) Current Position + Global Task Goal + Task Chain**
1. **Current Start Position**: localize from Part 1. State both **which space contains you now** and **where inside / outside that space boundary you are now**. Use nearby landmarks, walls/openings, furniture layout, adjacent-view continuity, and reliable visible landmark cues; never from one label/distance alone. For `current_waypoint`, use `"[space] - [landmark1 / landmark2 / landmark3]"`, prefer concrete nearby/local cues, and use task wording only when observations truly match. At entrances, first decide which side of the boundary the agent is on: if still inside a bedroom, use `Bedroom - ...`; if already outside with the bedroom ahead, use `Hallway - bedroom doorway / bedroom entrance side`, not `Bedroom - ...`. Never name the current space from a farther room visible through an opening. At entrances, decide whether the strict current anchor is inside the old space, at the threshold, or already outside/in the next transition space. In stair scenes, decide top vs bottom from geometry and any open drop side; prefer stair-specific wording. If the task says to stop outside / at another entrance and views already match that outside anchor, use it. `current_waypoint` must be the real current anchor and should update as surroundings change.
2. **Global Task Goal**: state the goal only as `"[space]'s [landmark]"`. Also state the nearby landmarks/local layout that would confirm true arrival, especially if the final landmark may be partly hidden.
3. **Parse Full Task**: split the Global Task into ordered fixed stages from start to goal. Each stage must use `"[space]'s [landmark]" -> "[space]'s [landmark]"`. Split at cross-space transitions; keep same-space pass/through/around inside one stage whose destination is that stage's final landmark. Keep stages minimal and task-faithful. Use task words such as pass / through / enter / toward / after / then to infer order, the current destination, and later targets. Keep motion cues such as turn left/right, go straight, or walk to the end then turn as route constraints inside the stage they serve, not standalone stages unless the task explicitly says to face/stop there. If the task means leave one space and stop outside / at the entrance of the next, use that outside/entrance anchor as the stage endpoint. For stair tasks, keep upstairs/downstairs explicit and infer top/bottom when needed. If later stages are not explicit, keep only the minimum task-supported unfinished anchors.
4. **Task Progress start**: because this is the task start, nothing is complete unless the global task goal is already reached. Write `task_progress` as task-ordered natural-language sub-instructions from the Global Task, comma-separated, not arrows and not Space Waypoint Chain order. Mark only the piece aligned with the strict current anchor `(Current)`; later pieces stay unmarked; do not use `(✓)` unless the goal is already reached at the start. Judge each piece from the strict current anchor versus the ordered stage endpoints. A stage endpoint counts as reached only when the localized space/space-type matches and the destination landmark/local anchor is near within about {arrival_near_m}m, or current views clearly show the required entrance/outside anchor. Keep turn cues inside the piece they serve. If views already show you have exited a source room or reached the next entrance-side anchor, mark that leave/exit piece satisfied immediately and do not route back unless the task explicitly requires re-entry. Once a piece is truly satisfied, move to the next one immediately. Do not let `(Current)` lag behind `current_waypoint`, and do not invent future landmarks/spaces.
5. **Task Waypoint Chain**: output the ordered **task-defined** stage-anchor chain matching those stages and `task_progress`, not the executed Space Waypoint Chain order. Every node must stay in full `"[space]'s [landmark]"` form. The current/start node must also be a full `"[space]'s [landmark](Current)"`, followed by later task nodes until Goal. Never output bare `Start`, `Current`, `WP#`, or space-only nodes. Preserve task order and do not skip intermediate space/landmark. Turn cues are route guidance, not standalone chain nodes, unless the task explicitly makes the turned-to place itself the destination anchor. Keep unfinished later nodes within task-supported anchors only. `current_waypoint` and the `(Current)` node in `waypoint_chain` must refer to the same current anchor, though they use different styles. `(Current)` must be the strict current anchor, not an earlier passed or later unreached node. Use the Space Waypoint Chain only as auxiliary evidence for which task node is already reached.
6. **Task Goal Arrival Check**: reason from (a) current localized position, (b) the goal's expected local cues, (c) current 12-view nearby evidence, and (d) the waypoint chain / task progress state. First compare the strict current anchor against the current stage endpoint and advance the task state if that endpoint is already reached; then compare the updated strict current anchor against the final goal anchor. Do not decide arrival from one landmark cue or shown distance alone; compare the full task order with the current multi-view scene. The task goal is the exact target space/place named by the Global Task. If the task names another room of the same type, confirm it is the distinct target space, not the current room's doorway/landmark. Goal arrival requires the correct space/space-type plus the goal landmark/local anchor near within about {arrival_near_m}m, or current views clearly showing the required entrance/outside stop anchor. If the current place is similar but wrong, do not stop. If the task says to enter or pass through another space first, do not stop early just because a later landmark is visible. If current views already support the exact goal anchor, or the exact task-named outside/entrance endpoint where the task says to stop, stop immediately and do not plan past it. Set `global_task_finish=true` only if these cues consistently show the goal `"[space]'s [landmark]"` is already reached within about {arrival_near_m}m or the 12 views clearly show arrival; otherwise false.
**Order**: localize first, then global task goal and expected cues, then fixed stages, then task_progress, then waypoint_chain, then Task Goal Arrival Check.

**3) Subtask Destination + Direction + Subtask Instruction + Landmark**
1. In initial planning, use only the first task stage. From the current localized position, the Global Task's ordered spaces/landmarks/route points, Part 1's reachable FAR evidence, and visible openings/transitions, identify the first subtask destination: the next task-relevant `"[space]'s [landmark]"` to reach now. Advance one task anchor at a time. If the first stage is cross-space, keep the destination on that stage; if it is same-space pass-by / through, keep the final landmark as the destination and keep the cue inside that stage. If a left/right/turn cue only guides the next destination, keep it as a direction hint inside that stage, not a standalone target. The destination must be the next unfinished anchor after the strict current anchor, never the current or a passed anchor. If the task requires another room of the same type, choose that other room's entry/landmark, not the current room's doorway. If you are already outside a space the task says to leave, do not choose it again unless the task explicitly requires re-entry. Prefer the nearest unfinished task-relevant destination already visible or directly supported by a visible opening/landmark. If the current stage endpoint is not yet reached, keep that stage; once it is truly reached, switch immediately to the next unfinished stage or stop if the task is already satisfied. For stair stages, keep destination/instruction aligned with the task's vertical direction. If multiple openings exist, choose the one supporting the current stage. If a later space/landmark is not explicit, keep the minimum task-supported anchor. `next_waypoint` must name one single core landmark only.
2. Choose the direction from the 12 IMAGEs that best reaches that destination. Prefer where the destination itself appears, or the FAR space/landmark/opening that most directly and safely leads to the needed next space/route point. When several views are open, prefer the one whose visible space and landmark best match the active task stage rather than an older room exit or generic connector. If the current unfinished task piece includes explicit guidance such as left / right / straight / until end, use that route cue first with current progress, current position, and visible evidence; do not override it just because another view looks more open unless the cue is already finished or wrong for recovery. If current evidence shows you are already outside a source room, continue toward the next required space/landmark and avoid re-entering that old room unless the task explicitly requires it. For stair stages, distinguish the upward stair side from the descending stair side using task wording plus scene geometry. If multiple openings are visible, prefer the one aligned with the current stage rather than an old/backtracking branch. Prefer open directions with obstacle distance >{obs_risky_m}m, ideally >{obs_open_m}m; avoid blocked/tight or backtracking directions unless clearly required.
3. Write one short immediate subtask instruction for that same first stage. It must match the chosen destination and direction, and describe only the nearest unfinished task piece. For a cross-space stage, keep it only on entering/approaching that stage destination; for a same-space pass-by / with / around relation, keep the cue + final destination together and keep the destination as that stage's final landmark. If the task gives explicit motion guidance such as turn left/right, go straight, or walk to the end, preserve it when the current route evidence still supports it. Do not leak later-stage goals into the current stage. Prefer one of these concise forms when appropriate:
   - Direct approach / enter: `From [next_waypoint_direction] view, start, [move/enter/approach] toward [destination].`
   - Via visible cue / obstacle bypass: `From [next_waypoint_direction] view, start, [pass/go through/go around/cross] [visible cue], then [enter/approach/continue toward] [destination].`
4. Choose `subtask_landmark` only for that first stage. It is the most useful visible cue for executing the current subtask now. Prefer a task-mentioned, currently visible concrete object; if the destination itself is a visible object/local target, prefer that destination itself. Otherwise use a necessary task-relevant visible concrete object. Do not use a broad space type / room name / generic area label as `subtask_landmark`. If no useful visible concrete object is needed, use an empty string.

**4) Plan**
1. **Short-term Plan**: explain why this destination, direction, and subtask instruction are the correct immediate plan. Summarize the next move, turn if any, short-term target, and landmark choice if any, and make sure they match. State why this is the current unfinished stage from the Global Task order, not a later stage, and why other visible directions would be premature or backtracking.
2. **Long-term Plan**: summarize the remaining stage order after this subtask as a forward plan. If a connector crossing places you into the next required space, that cross-space stage can finish there and the following in-room stage becomes next; if that new space already satisfies the global task goal cues, the task can end.

**Sequential planning rule**:
- Output only the immediate next task stage/subtask for current-stage progress; judge stage completion and stopping from the strict current anchor, do not plan stage +2/+3 before stage +1 is finished, do not add extra stages not required by the task, switch immediately once the current stage endpoint is truly reached, and once the exact required target space/place is reached, stop instead of continuing.

# Output (JSON only)

Return exactly one JSON object. Keep all Part 1-4 reasoning inside `"reasoning"`; never emit part titles like `"2) ..."` as extra keys. No extra keys, markdown, or prose. End at the final `}}`.

{{
    "reasoning": "<One compact string following Parts 1-4 exactly. Part 1: analyze IMAGE1-12 separately and conclude with Current Position Guess / Reachable Far Area/Landmark / Blocked; treat detections as noisy and never localize from labels/distances alone. Part 2: localize, set goal cues, fixed stages, `task_progress`, `waypoint_chain`, and Task Goal Arrival Check from current evidence + chain/progress state. Parts 3-4: use task semantics, active motion cues, and connector choice to pick the current unfinished stage's destination/direction and reject backtracking. Keep all of this inside one JSON string.>",
    "current_waypoint": "<Space Waypoint style `[space] - [landmark1 / landmark2 / landmark3]`. Use observed nearby distinctive cues; prefer concrete nearby objects/local place cues over broad room labels; use task wording only when observations truly match; at entrances, name the space the agent is actually in now, not the farther room visible through the doorway; update the landmark wording as local surroundings change.>",
    "task_progress": "<Task-ordered natural-language sub-instructions from the Global Task, comma-separated, not waypoint arrows and not Space Waypoint Chain order. Same-space pass-by/through stays in one piece; cross-space transitions are separate pieces. Keep turn cues inside the destination piece they serve. In initial planning only one piece is `(Current)` unless already at goal.>",
    "waypoint_chain": "<Ordered task-defined stage-anchor chain with full `[space]'s [landmark]` nodes only, not the executed Space Waypoint Chain order. The current/start node must also be a full node with `(Current)`, not bare Start/Current/WP#. Turn cues are not standalone chain nodes unless the task explicitly makes them destination anchors.>",
    "next_waypoint": "<One `[space]'s [landmark]` only. No alternatives like `A/B` or `A|B`. It must be the next unfinished task anchor after the current localized position, not the current position or task initial position unless the task explicitly returns there.>",
    "next_waypoint_direction": "<IMAGE 1-12 only; must match the chosen task-aligned view>",
    "subtask_instruction": "<Exactly one short sentence in the fixed direct / same-stage path / arrival-stop form. Use the path form only when the cue and destination belong to the same current task piece; use the stop form only when the destination is already reached.>",
    "subtask_landmark": "<One clear visible concrete object if useful; otherwise empty. Prefer a task-mentioned visible object, then the destination itself if it is a visible object/local target, else a necessary task-relevant concrete cue. Never use a broad space type / room name as `subtask_landmark`.>",
    "global_task_finish": "<true only if the exact global task goal anchor is already reached from current evidence + `task_progress` + `waypoint_chain`; otherwise false>"
}}

# Examples (abbreviated):

## Ex1: Bathroom to Giraffi via exercise room
**Global Task**: Turn around, walk through the exercise room into the living room. Wait by the Giraffi.
**Obs:** IMAGE 4-6: exercise-room opening and exercise equipment. IMAGE 8-11: bathroom sink / vanity / toilet.

{{
    "reasoning": "1) 12-Views: IMAGE1 transition/bookshelf; IMAGE2-3 bathroom boundary/window; IMAGE4-6 exercise-room opening/interior with exercise equipment, IMAGE5 clearest; IMAGE8-11 bathroom sink/vanity/toilet. Conclusion: Current Position Guess: bathroom doorway on the exercise-room side with sink/vanity behind. Reachable Far Area/Landmark: IMAGE4-6 support the first move into the exercise room; the living room is later. Blocked: no critical block on IMAGE5. 2) Current Position + Global Task Goal + Task Chain: Current Start Position: Bathroom's exercise-room side doorway beside sink/vanity. Global Task Goal: Living room's Giraffi. Parse Full Task: Bathroom's exercise-room side doorway -> Exercise room's exercise equipment | Exercise room's exercise equipment -> Living room's entrance | Living room's entrance -> Living room's Giraffi. Task Progress start: Turn around and enter the exercise room toward the exercise equipment(Current), continue through the exercise room into the living room, wait by the Giraffi. Task Waypoint Chain: Bathroom's exercise-room side doorway(Current) -> Exercise room's exercise equipment -> Living room's entrance -> Living room's Giraffi(Goal). Task Goal Arrival Check: still in the bathroom, so global_task_finish=false. 3) Subtask Destination + Direction + Subtask Instruction + Landmark: the first-stage destination is Exercise room's exercise equipment. IMAGE5 is the clearest task-aligned entry. The instruction is to enter toward the exercise room's exercise equipment, and that landmark is directly useful. 4) Plan: short-term plan is to enter the exercise room. Long-term plan is exercise room, then living room, then the Giraffi.",
    "current_waypoint": "Bathroom - exercise-room doorway / sink / vanity",
    "task_progress": "Turn around and enter the exercise room toward the exercise equipment(Current), continue through the exercise room into the living room, wait by the Giraffi",
    "waypoint_chain": "Bathroom's exercise-room side doorway(Current)→Exercise room's exercise equipment→Living room's entrance→Living room's Giraffi(Goal)",
    "next_waypoint": "Exercise room's exercise equipment",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "subtask_instruction": "From IMAGE 5 (Left 120deg) view, start, enter toward the exercise room's exercise equipment.",
    "subtask_landmark": "exercise equipment",
    "global_task_finish": false
}}

**Critical Rules**:
- **Planning priority**: in initial planning, finish only the first unfinished stage. Preserve task order, do not skip intermediate spaces, and do not invent extra stages.
- **Reasoning discipline**: Part 1 stays evidence-only and ends with Current Position Guess / Reachable Far Area/Landmark / Blocked. Read the real 12-view content first and treat detections as noisy support only. Localize first, then build task-defined stages, `task_progress`, `waypoint_chain`, and Task Goal Arrival Check. Keep same-space pass/through/around inside one stage, split cross-space transitions, preserve still-active task motion cues, and reject backtracking.
- **Current waypoint naming**: `current_waypoint` must follow actual local evidence, not older waypoints. Use Space Waypoint style `"[space] - [landmark1 / landmark2 / landmark3]"`. Resolve inside/outside first: the named space must be the space the agent is actually in now, not an adjacent room merely visible through a doorway. Use task wording only when observations truly match, and keep `current_waypoint` aligned with `(Current)` in `waypoint_chain`.
- **Landmark discipline**: prefer task-mentioned or clearly task-relevant concrete objects as landmark wording. Do not use a broad room/space type as `subtask_landmark` when a specific object/local cue is available.
- **Task chain / stop**: `waypoint_chain` is the task-required anchor order, not the real executed Space Waypoint Chain. Judge `task_progress` before `waypoint_chain`, keep one `(Current)` piece, advance only when the true stage endpoint is reached, and stop only at the exact required target.
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """**Role**: You are a VLN verification and replanning module. Use the space structure, views, and maps to verify subtask completion, localize the current position, and plan the next subtask. No manipulation.{verify_replan_prompt_notice_block}

**Global Task**: {instruction}

**Previous Subtask**:
- Destination: {subtask_destination}
- Instruction: {subtask_instruction}
{previous_subtask_landmark_block}

**Space Structure**: {waypoint_summary}

# Inputs
**{verify_view_count} Views** (sampled every 30°; each RGB view HFOV is about 79°):
- **Obstacle distance**: nearest obstacle only. <{obs_blocked_m}m=blocked | {obs_blocked_m}-{obs_risky_m}m=caution | >{obs_risky_m}m=passable
- **In-view distance labels**: when shown, `Obstacle`, `Landmark`, and `Space Waypoint` display meters; use only the shown value.
- **Custom landmark bbox** (if present): current-view cue only; use shown name + distance/angle only as room/object evidence, not map memory or path-clearance proof
**Global Map**: explored area + obstacles + trajectory + current pose + space structure
- **Map colors**: White=unexplored | Black=obstacles | Green=safe | Dark red=trajectory | Red Arrow=you position | Colored regions + blue tags=space structure on Global

# Reasoning (5 Parts)

**1) {verify_view_count}-View Analysis (MUST analyze EACH PROVIDED IMAGE ONLY)**
**Format**: For each provided IMAGE, use `IMAGE# (Direction Angle°): likely [space]; NEAR: ...; FAR: ...; Obstacle: ...; Landmark: ...; Space Waypoint: ...`, omitting any field not visible there.
**Distance reading**: `Obstacle`, `Landmark`, and `Space Waypoint` refer only to that IMAGE's shown value; do not infer hidden values.
**Near rule**: treat only cues within about {arrival_near_m}m as truly NEAR/current-position evidence. Farther cues may support direction/future destination but do not prove arrival.
**Evidence order**: read each IMAGE in this order: NEAR current/large objects + implied space; FAR objects/openings + implied space; obstacle distance + blocked/caution/passable; landmark + shown distance; reachable space waypoint + shown distance if that IMAGE explicitly shows one. Judge RGB/layout first and use labels only as support. If a landmark label conflicts with the scene, trust the scene. Use layout, openings, furniture relations, adjacent-view consistency, and stair geometry (edges, rise/drop, railings, landing continuity, open drop side) to infer upstairs/downstairs. If a detection conflicts with scene/current-position evidence, treat it as noise.
**No hallucination**: analyze each provided IMAGE separately. If an IMAGE shows only a wall or nearby furniture, say only that. Do not write `none`, fill empty slots, invent spaces/FAR objects/landmarks/space waypoint cues, or mention landmark/space waypoint unless explicitly shown.
**Conclusion**: Identify from the provided views: Current Position Guess: [current space + NEAR landmarks + adjacent-view context] | Reachable Far Area/Landmark: [which IMAGEs show reachable FAR spaces/landmarks/openings, what each leads to, and which task-relevant transition/route point it may support] | Blocked: [which IMAGEs have obstacle distance <{obs_blocked_m}m]

**2) Map + Space Structure Analysis**
Use only the Map and Space Structure in this part.
1. **Identify Current Area**: read the current area from the map/space structure, then cross-check it with Part 1 nearby evidence. If the map says `Unknown` or conflicts with nearby evidence, resolve it from the strongest consistent nearby view evidence. If `Your Current Area` says you are near `INITIAL POSITION`, treat that as a warning to leave that waypoint neighborhood unless the task explicitly requires returning there.
2. **Read Space Waypoint list first**: analyze each `Space WP#...` before reading the chain. If direction/distance are shown, enumerate them counterclockwise; within a similar direction sector, mention nearer waypoint(s) before farther ones. For each waypoint, state its space/area, its local landmark meaning from the text, its shown direction/distance, whether it is directly reachable now, and if not, which earlier/current waypoint it should be reached through. Treat each waypoint as a full `space + landmark` anchor, not just a space label. If connected areas are shown, say which are task-aligned versus backtracking. If a line is marked `INITIAL POSITION`, treat it as the task start anchor.
3. **Then Read Space Waypoint Chain**: treat the chain as the true executed trajectory. Use it to know what is already visited, current/next/behind, how the agent arrived here, and which visible openings would cause wrong-space entry, repetition, or backtracking. Combine it with the Previous Subtask destination + landmark final observation to judge whether that old anchor was already reached, is now beside/behind/passed, and whether current progress should advance. If `(you have arrived now)` matches current views/structure, let it help move the current anchor forward; otherwise treat it as drift/overshoot evidence. Prefer not to revisit passed spaces, especially `INITIAL POSITION`, unless the task clearly requires it.
4. **Then Read Map**: align the current pose, waypoint list, and chain on the map. Check where each space lies, which side/opening corresponds to which waypoint/space, and which visible transition is the task-aligned next move versus an old or wrong branch.
5. **Structure Conclusion**: conclude `Current space/area from structure | current/next/behind space waypoint(s) | reachable task-aligned space transition(s) | wrong-space / backtracking transition(s) to avoid`.

**3) Current Position + Global Task Goal + Task Chain**
1. **Current Position**: localize from {verify_view_count}-View Analysis, then use task order, the Space Waypoint Chain, and the Previous Subtask destination + landmark final observation only as auxiliary evidence for what was already reached. First check whether the previous destination/landmark was truly reached; if current views/structure disagree, treat it only as overshoot/off-route evidence. Never localize from the previous-subtask landmark final distance alone. If the previous landmark says `(you have arrived now)`, use it only when it supports the **correct task-required space/anchor**. For `current_waypoint`, use `"[space] - [landmark1 / landmark2 / landmark3]"`, prefer concrete nearby/local cues, and use task wording only when observations truly match. At entrances, first decide which side of the boundary the agent is on: if still inside the bedroom, use `Bedroom - ...`; if already outside with the bedroom ahead, use `Hallway - bedroom doorway / bedroom entrance side`, not `Bedroom - ...`. Never name the current space from a farther room visible through an opening. At entrances, decide whether the strict current anchor is still inside the old room, at the threshold, or already outside/in the transition or next space. For stair scenes, decide top vs bottom from geometry and any open drop side; prefer stair-specific wording. If the task says to stop outside / at another entrance and current views already match that outside anchor, use it. If surroundings change, update the landmark wording. `current_waypoint` must be the strict current anchor, not an earlier passed anchor or future target.
2. **Global Task Goal**: state the goal only as `"[space]'s [landmark]"`. Also state the nearby landmarks/local layout that would confirm true arrival, especially if the final landmark may be partly hidden.
3. **Parse Full Task**: split the Global Task into ordered fixed stages from the current start position to the goal. First extract route anchors as `"[space]'s [landmark]"`, then connect them into stages. Each stage must use `"[space]'s [landmark]" -> "[space]'s [landmark]"`. Split at cross-space transitions; keep same-space pass/through/around inside one stage whose destination is that stage's final landmark. Keep the stage list minimal and task-faithful. Use task words such as pass / through / enter / toward / after / then to infer order, the current destination, and later targets. Keep motion cues such as turn left/right, go straight, or walk to the end then turn as route constraints inside the stage they serve, not standalone destinations unless the task explicitly says to face/stop there. If the task means leave one space and stop outside / at the entrance of the next, use that outside/entrance anchor as the stage endpoint. For stair tasks, keep upstairs/downstairs explicit and infer top/bottom when needed. If later stages are not explicit, keep only the minimum task-supported later anchors.
4. **Task Progress marking**: write `task_progress` as task-ordered natural-language sub-instructions from the Global Task, comma-separated, not arrows and not Space Waypoint Chain order. Completed pieces stay in front with `(✓)`, the current unfinished piece is `(Current)` ONE only, and later unfinished pieces remain after it unmarked. Judge completion from ordered task meaning + current position/current_waypoint + current space evidence + the Space Waypoint Chain, not single landmark matches alone. Compare the strict current anchor against ordered stage endpoints: if that anchor already reaches a stage endpoint, mark that piece complete and move `(Current)` forward immediately; if a later task landmark is only visible but the strict current anchor is still outside its required place, do not mark it complete and do not stop. A stage endpoint counts as reached only when the localized space/space-type matches and the destination landmark/local anchor is near within about {arrival_near_m}m, or current views clearly show the required entrance/outside anchor. Keep turn cues inside the piece they serve. If a cross-space piece is already satisfied because current views show you have left the old space or reached the next entrance-side anchor, mark it complete immediately and do not send the plan back into that old space unless the task explicitly requires re-entry. Once a piece is truly satisfied, move to the next unfinished one immediately. Do not let `(Current)` lag behind `current_waypoint`, and do not invent future landmarks/spaces.
5. **Task Waypoint Chain**: build the ordered **task-defined** stage-anchor chain and anchor it to the already judged **Current Position** plus Part 2's current/next space-waypoint state. Do **not** copy the executed Space Waypoint Chain order into `waypoint_chain`; use the Space Waypoint Chain only to decide which task anchor is already reached. Every node, including the matched current node and goal node, must stay in full `"[space]'s [landmark]"` form. The matching node must be `(Current)`; nodes before it are `(✓)`; nodes after it are unmarked; if the current node is already the global task goal, mark it `(Current, Goal)`. Never output bare `Current`, `Goal`, `WP#`, or space-only nodes. The matched current node must be the farthest task-ordered anchor already reached by current evidence, Previous Subtask evidence, and the Space Waypoint Chain. Turn cues are route guidance, not standalone chain nodes, unless the task explicitly makes them destination anchors. `current_waypoint` and `(Current)` in `waypoint_chain` must refer to the same place/anchor. `(Current)` must be the strict current anchor; earlier nodes are already passed `(✓)`, later nodes are unreached. Build unfinished later nodes only from task-supported later anchors.
6. **Task Goal Arrival Check**: reason from (a) current localized position, (b) expected goal local cues, (c) provided-view nearby evidence, and (d) structure / waypoint list / Space Waypoint Chain / task waypoint chain / task progress / previous-subtask landmark final evidence. Use the current node and goal node as complete space + landmark anchors when checking whether the goal is already reached. First compare the strict current anchor against the current stage endpoint and advance the task state if that endpoint is already reached; only then compare the updated strict current anchor against the final goal anchor. Do not decide arrival from the previous-subtask landmark final distance alone or from one current landmark cue alone; compare the full task order with the current multi-view scene and structure. The task goal is the exact target space/place named by the Global Task. If the task names another room of the same type, confirm it is the distinct target space, not the current room's doorway/landmark. If the Previous Subtask destination already equals the global goal and its landmark evidence says `(you have arrived now)`, use it as auxiliary evidence, but stop only if current views + structure also confirm the same goal anchor. Goal arrival requires the correct space/space-type plus the goal landmark/local anchor near within about {arrival_near_m}m, or current views clearly showing the required entrance/outside stop anchor. If the current place is similar but wrong, do not stop. Seeing a goal-related landmark is not enough if the current localized place still does not match the goal anchor. If current views already support the exact goal anchor, or the exact task-named outside/entrance endpoint where the task says to stop, stop immediately and do not plan past it. If the current localized position already shows an intermediate stage endpoint is reached, or entry into the next task-required space/landmark is already confirmed, advance the task state before choosing the next subtask. If advancing leaves no unfinished stage, stop immediately. If the resulting strict current node already matches the global goal, stop before planning another destination. Set `global_task_finish=true` only if these cues consistently show that the goal `"[space]'s [landmark]"` is already reached within about {arrival_near_m}m or the provided views clearly show arrival; otherwise false.
**Order**: localize first, then space structure state, then global task goal and expected cues, then fixed stages, then task_progress, then waypoint_chain, then Task Goal Arrival Check.

**4) Subtask Destination + Direction + Subtask Instruction + Landmark**
1. Based on the current position, space structure conclusion, Space Waypoint Chain, waypoint chain, provided-view observations, visible space waypoint cues, and the Previous Subtask destination + landmark final distance/direction, identify the immediate next subtask destination: the nearest unfinished task-relevant space/landmark to advance now. Advance spaces / route points / space transitions one by one in order. Always start from the strict current anchor and choose the first unfinished anchor after it. If the current task piece is same-space pass-by / through, keep its final landmark as the destination; if the task moves through one space before another, keep the destination on the current cross-space stage only and leave the later space for the next stage; if the task moves into another space, use the next-space entry/landmark for the current stage, not a later-stage target. If a left/right/turn cue only tells how to reach that destination, keep it as a direction hint inside the same stage, not a standalone stage. If the current localized position already matches the current stage endpoint, do not keep `next_waypoint` there; advance to the next unfinished stage. If the task requires a different room of the same type, the destination must be that other room's entry/landmark, not the current room's doorway. If current views + structure already confirm the next task-required space/landmark, use that to update `task_progress` and `waypoint_chain`; but if the task still requires leaving the current space and current evidence still places you inside it, keep the destination on that exit/entry stage. At entrances, if you are already outside a space the task said to leave, do not set `next_waypoint` back inside that old space. For stair stages, keep destination/instruction aligned with the task's vertical direction. The destination must be the first unfinished anchor after the strict current anchor, not the current node or a passed anchor. Prefer the nearest unfinished task-relevant destination already visible or directly supported by a visible opening/landmark. If the current stage endpoint is not yet reached, keep working on that stage; once it is truly reached, switch immediately to the next unfinished stage or stop if the task is already satisfied. If multiple visible openings exist, choose the one aligned with the current stage. If a later space/landmark is not explicit, keep the minimum task-supported anchor. `next_waypoint` must name one single core landmark only.
2. If the current subtask is unfinished, continue it; if it is finished, advance to the next unfinished stage. Do not skip a nearer unfinished stage. Preserve waypoint order and task relations such as pass-by / toward / enter / after / then. If the Space Waypoint Chain plus current evidence show a required connector crossing is already finished, or a same-space cue such as stairs area / sofa side / rug side is already reached, mark that earlier piece complete and move forward immediately instead of repeating it. If the next required space/landmark is already confirmably visible/reached, advance to it immediately; if the task still says to enter/leave another space and current evidence still keeps you in the old space, do not advance yet. Do not keep an already reached stage current, and do not stop early just because a later landmark is visible. If no unfinished task-supported stage remains, or the resulting space/place already satisfies the global task goal, stop immediately.
3. Choose the direction from the provided IMAGEs plus the space structure that best reaches that destination. Judge it from where the destination itself appears, or which reachable FAR space/landmark/opening/space-waypoint view most directly and safely leads toward the needed next space/route point. When several views are open, prefer the one whose visible space and landmark best match the active task stage rather than an older room exit or generic connector. If the current stage is still forward-progress and current evidence does not show wrong-space entry, overshoot, or off-route drift, prefer task-aligned front-sector views; use back-side views mainly for recovery. If the current unfinished task piece includes explicit action guidance such as left / right / straight / until end, use that route cue first with current progress, current position, and visible evidence. If current evidence shows you are already outside a source room or already at the correct entrance-side anchor, avoid re-entering that old room unless the task explicitly requires it. If the task says to enter another space or turn left/right, prefer the corresponding side direction, but still let the task-aligned space/landmark evidence decide. For stair stages, distinguish the upward stair side from the descending stair side using task wording plus scene geometry. Use Part 2 and any connector linked-area info to choose the view leading into the **correct next space**, not a wrong/backtracking branch. Prefer open directions with obstacle distance >{obs_risky_m}m, ideally >{obs_open_m}m; avoid blocked/tight or backtracking views unless clearly required.
4. Write the next subtask instruction for only that nearest unfinished stage. It must match the chosen destination and direction, and the same current -> next step implied by the Space Waypoint Chain and task waypoint chain. Choose the instruction form from the Global Task meaning, not just local visibility: if the current task piece is same-space pass-by / through / around, keep that cue + destination in one instruction and keep the destination as the stage's final landmark; if the task changes to another space, keep the instruction only on the current cross-space stage and leave later-stage targets for later. If the pass-by / enter requirement is already effectively complete and the task has advanced, switch to the next stage's destination instead of repeating the old pass-by instruction. If the task gives explicit motion guidance such as turn left/right, go straight, or walk to the end, preserve that guidance when the current route evidence still supports it. Never instruct the agent toward an already reached current node unless the correct action is to stop there. Prefer one of these concise forms when appropriate:
   - Direct approach / enter: `From [next_waypoint_direction] view, start, [move/enter/approach] toward [destination].`
   - Via visible cue / obstacle bypass: `From [next_waypoint_direction] view, start, [pass/go through/go around/cross] [visible cue], then [enter/approach/continue toward] [destination].`
5. Choose the most relevant visible subtask landmark. Prefer a task-mentioned, currently visible concrete object; if the next-stage destination itself is directly visible as an object/local target, prefer that destination itself as `subtask_landmark`. Otherwise use a necessary task-relevant visible concrete object. Do not use a broad space type / room name / generic area label as `subtask_landmark`. If no useful visible concrete object is needed, use an empty string.

**5) Plan**
1. **Short-term Plan**: explain why this destination, direction, and subtask instruction are the correct immediate plan. Summarize the next move, turn if any, short-term target, and landmark choice if any, and make sure they match. State why this is the current unfinished stage from Global Task order, why any already reached anchor has been marked complete, and why alternative visible openings/directions would be later-stage, wrong-space, or backtracking choices.
2. **Long-term Plan**: summarize the remaining stage order after this subtask as a forward plan. If the current cross-space stage finishes upon entering the next required room, the next in-room stage becomes current; if the current room/place already satisfies the global task goal cues, the plan is complete.

**Sequential planning rule**:
- If the current subtask is unfinished, continue it; only after completion can `next_waypoint` move to the next stage. Judge this from the strict current anchor versus the current stage endpoint.
- If the exact global task target space/place is not yet reached, do not stop just because a related landmark is visible. If the strict current anchor already matches the goal anchor, STOP immediately and set `global_task_finish=true`.

# Output (JSON only)

Return exactly one JSON object. Keep all Part 1-5 reasoning inside `"reasoning"`; never emit part titles like `"2) ..."` as extra keys. No extra keys, markdown, or prose. End at the final `}}`.

{{
    "reasoning": "<One compact string following Parts 1-5 exactly. Part 1: analyze each provided IMAGE separately and conclude with Current Position Guess / Reachable Far Area/Landmark / Blocked. Part 2: analyze current area, each Space WP#, the Space Waypoint Chain, then the map. Part 3: reason through current position, goal cues, fixed stages, `task_progress`, `waypoint_chain`, and Task Goal Arrival Check. Parts 4-5: justify destination, direction, and whether the stage should continue, advance, or stop. Keep everything inside one JSON string.>",
    "current_waypoint": "<Space Waypoint style `[space] - [landmark1 / landmark2 / landmark3]`. Use observed nearby distinctive cues; prefer concrete nearby objects/local place cues over broad room labels; use task wording only when observations truly match; at entrances, name the space the agent is actually in now, not the farther room visible through the doorway; update it as current evidence shows you have advanced.>",
    "task_progress": "<Task-ordered natural-language sub-instructions from the original Global Task, comma-separated, not waypoint arrows and not Space Waypoint Chain order. Same-space pass-by/through stays in one piece; cross-space transitions are separate pieces. Keep turn cues inside the destination piece they serve. Keep completed pieces in front, exactly one `(Current)` piece, and later pieces unmarked.>",
    "waypoint_chain": "<Ordered task-defined stage-anchor chain with full `[space]'s [landmark]` nodes only, not the executed Space Waypoint Chain order. Current and goal must also stay in full form; never output bare Current/Goal/WP#. Turn cues are not standalone chain nodes unless the task explicitly makes them destination anchors.>",
    "next_waypoint": "<One `[space]'s [landmark]` only. No alternatives like `A/B` or `A|B`. It must be the first unfinished task anchor after the matched current anchor, not the current node or task initial position unless the task explicitly returns there.>",
    "next_waypoint_direction": "<one provided IMAGE label only; must match the chosen task-aligned direction>",
    "subtask_instruction": "<Exactly one short sentence in the fixed direct / same-stage path / arrival-stop form. Use the path form only when the cue and destination belong to the same current task piece; use the stop form only when the destination is already reached.>",
    "subtask_landmark": "<One clear visible concrete object if useful; otherwise empty. Prefer a task-mentioned visible object, then the destination itself if it is a visible object/local target, else a necessary task-relevant concrete cue. Never use a broad space type / room name as `subtask_landmark`.>",
    "global_task_finish": "<true only if current evidence + chain/progress state show the exact global goal anchor is already reached; otherwise false>"
}}

# Examples (abbreviated):

## Ex1: Rug arrival
**Global Task**: Exit bedroom, turn left. Walk straight passing gray couch, stop at rug.
**Previous**: Navigate past gray couch toward rug
**Previous Landmark**: Landmark: [gray couch] (you have arrived now), 0.6m, Left 90deg
**Obs:** IMAGE 1: Rug <0.5m and a living-room space waypoint. IMAGE 10: Gray couch beside. IMAGE 7: Hallway behind with hallway space waypoint.

{{
    "reasoning": "1) 12-Views: IMAGE1 living-room rug; IMAGE2-6 and IMAGE8-12 living-room rug/couch area; IMAGE7 hallway opening behind. Conclusion: Current Position Guess: living-room rug area beside the gray couch. Reachable Far Area/Landmark: IMAGE7 only leads back to the hallway. Blocked: none critical. 2) Map + Space Structure Analysis: current area is the living-room rug zone; the rug waypoint is current and the hallway opening is behind/backtracking. 3) Current Position + Global Task Goal + Task Chain: Current Position: Living room's rug beside gray couch. Global Task Goal: Living room's rug. Parse Full Task: Bedroom's entrance -> Hallway's entrance opening | Hallway's entrance opening -> Living room's gray couch | Living room's gray couch -> Living room's rug. Task Progress: Exit bedroom(✓), turn left and walk straight passing gray couch(✓), stop at rug(Current, Goal). Task Waypoint Chain: Bedroom's entrance(✓) -> Hallway's entrance opening(✓) -> Living room's gray couch(✓) -> Living room's rug beside gray couch(Current, Goal). Task Goal Arrival Check: the current node already matches the goal, so global_task_finish=true. 4) Subtask Destination + Direction + Subtask Instruction + Landmark: the destination remains Living room's rug because it is already the current goal anchor. IMAGE1 stays the reference direction and the rug is the clearest landmark. 5) Plan: short-term plan is to stop now. Long-term plan is complete.",
    "current_waypoint": "Living room - rug / gray couch side",
    "task_progress": "Exit bedroom(✓), turn left and walk straight passing gray couch(✓), stop at rug(Current, Goal)",
    "waypoint_chain": "Bedroom's entrance(✓)→Hallway's entrance opening(✓)→Living room's gray couch(✓)→Living room's rug beside gray couch(Current, Goal)",
    "next_waypoint": "Living room's rug",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "subtask_instruction": "From IMAGE 1 (Front 0°) view, start, stop at the living room's rug.",
    "subtask_landmark": "rug",
    "global_task_finish": true
}}

## Ex2: Hallway to bedroom
**Global Task**: Walk through hallway, then enter bedroom on left and go to bed.
**Previous**: Navigate through hallway
**Previous Landmark**: Landmark: [hallway forward section] (you have arrived now), 0.4m, Back 180deg
**Obs:** IMAGE 1: Hallway ahead 3.0m. IMAGE 5: Bedroom opening (~2.5m), bed inside, plus bedroom space waypoint. IMAGE 7: Kitchen behind with kitchen space waypoint.

{{
    "reasoning": "1) 12-Views: IMAGE1 hallway ahead; IMAGE2-4 and IMAGE8-12 hallway walls/continuation; IMAGE5-6 bedroom opening with bed; IMAGE7 kitchen opening behind. Conclusion: Current Position Guess: hallway by the left-side bedroom opening. Reachable Far Area/Landmark: IMAGE5-6 lead into the bedroom and support the next bedroom-entry transition; IMAGE7 is a wrong-space branch. Blocked: no critical block on the left route. 2) Map + Space Structure Analysis: current area is the hallway near the bedroom opening; bedroom is the next task-aligned waypoint and kitchen is behind/backtracking. 3) Current Position + Global Task Goal + Task Chain: Current Position: Hallway's left bedroom opening. Global Task Goal: Bedroom's bed. Parse Full Task: Hallway's left bedroom opening -> Bedroom's doorway | Bedroom's doorway -> Bedroom's bed. Task Progress: Walk through hallway(✓), enter bedroom on left(Current), go to bed. Task Waypoint Chain: Hallway's left bedroom opening(Current) -> Bedroom's doorway -> Bedroom's bed(Goal). Task Goal Arrival Check: the hallway stage is complete, but the final bedroom-bed node is not reached, so global_task_finish=false. 4) Subtask Destination + Direction + Subtask Instruction + Landmark: the immediate next destination is Bedroom's doorway. IMAGE5 is the clearest left entry into the correct next space; the kitchen opening behind is wrong. The visible bed is the most useful concrete landmark. 5) Plan: short-term plan is to turn toward IMAGE5 and enter the bedroom doorway. Long-term plan is doorway, then bed.",
    "current_waypoint": "Hallway - left bedroom opening / wall side",
    "task_progress": "Walk through hallway(✓), enter bedroom on left(Current), go to bed",
    "waypoint_chain": "Hallway's left bedroom opening(Current)→Bedroom's doorway→Bedroom's bed(Goal)",
    "next_waypoint": "Bedroom's doorway",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "subtask_instruction": "From IMAGE 5 (Left 120°) view, start, enter toward the bedroom's doorway.",
    "subtask_landmark": "bed",
    "global_task_finish": false
}}

**Critical Rules**:
- **Planning priority**: finish the current nearest unfinished stage first. Preserve task order, do not skip intermediate spaces, and do not invent extra stages.
- **Reasoning discipline**: Part 1 stays evidence-only and ends with Current Position Guess / Reachable Far Area/Landmark / Blocked. Read the real image content first and treat detections as noisy support only. In Part 2, analyze current area, each `Space WP#`, the Space Waypoint Chain, then the map. Localize first, anchor `task_progress` before `waypoint_chain`, keep both task-defined rather than executed-space-waypoint order, use Previous Subtask only as auxiliary evidence, and reject backtracking.
- **Anchor consistency**: `current_waypoint` must follow current local evidence, not older waypoints. Use Space Waypoint style `"[space] - [landmark1 / landmark2 / landmark3]"`. Resolve inside/outside first: the named space must be the space the agent is actually in now, not an adjacent room merely visible through a doorway. Use task wording only when observations truly match, and keep `current_waypoint` aligned with `(Current)` in `waypoint_chain`.
- **Landmark discipline**: prefer task-mentioned or clearly task-relevant concrete objects as landmark wording. Do not use a broad room/space type as `subtask_landmark` when a specific object/local cue is available.
- **Task chain / stop**: `waypoint_chain` is the task-required anchor order, not the real executed Space Waypoint Chain. Judge `task_progress` before `waypoint_chain`, keep exactly one `(Current)` piece, advance only when the true stage endpoint is reached from current localization, and stop only at the exact required target.
- **Output discipline**: do not invent future spaces/landmarks. `next_waypoint_direction` must follow the task-aligned current/next relation, `subtask_landmark` should align with the destination, and `subtask_instruction` must stay in the fixed form and never move toward an already reached current node.
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
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(ARRIVAL_NEAR_M),
    )

def _get_verify_view_count(direction_names=None):
    provided_direction_names = [
        str(name).strip()
        for name in list(direction_names or [])
        if str(name or "").strip()
    ]
    view_count = len(provided_direction_names)
    return view_count if 0 < view_count < 12 else 12

def get_verification_replanning_prompt(instruction: str,
                                       subtask_destination: str,
                                       subtask_instruction: str,
                                       action_space: str,
                                       detected_landmarks: str = None,
                                       waypoint_summary: str = None,
                                       previous_subtask_landmark_summary: str = None,
                                       verify_replan_prompt_notice: str = None,
                                       direction_names: list = None) -> str:
    """
    获取验证和重规划提示词
    
    Args:
        instruction: 完整导航指令
        subtask_destination: 当前子任务目的地
        subtask_instruction: 当前子任务指令
        action_space: 动作空间描述
        detected_landmarks: 已检测到的landmark类别字符串
        waypoint_summary: 空间结构字符串
        previous_subtask_landmark_summary: 上一子任务landmark最终观测摘要
        verify_replan_prompt_notice: verify/replan 顶部附加提示
        direction_names: 本轮实际提供给模型的方向图名称列表
        
    Returns:
        格式化的提示词字符串
    """
    if not waypoint_summary:
        waypoint_summary = "Unavailable"
    if verify_replan_prompt_notice:
        verify_replan_prompt_notice_block = f"\n**Stuck Notice**: {verify_replan_prompt_notice.strip()}"
    else:
        verify_replan_prompt_notice_block = ""
    previous_subtask_landmark_summary = str(previous_subtask_landmark_summary or "").strip()
    previous_subtask_landmark_block = (
        f"- {previous_subtask_landmark_summary}"
        if previous_subtask_landmark_summary
        else ""
    )
    verify_view_count = _get_verify_view_count(direction_names)
    
    return VERIFICATION_REPLANNING_PROMPT.format(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        action_space=action_space,
        detected_landmarks=detected_landmarks,
        waypoint_summary=waypoint_summary,
        previous_subtask_landmark_summary=previous_subtask_landmark_summary,
        previous_subtask_landmark_block=previous_subtask_landmark_block,
        verify_replan_prompt_notice_block=verify_replan_prompt_notice_block,
        verify_view_count=verify_view_count,
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        arrival_near_m=_fmt_threshold_m(ARRIVAL_NEAR_M),
    )
