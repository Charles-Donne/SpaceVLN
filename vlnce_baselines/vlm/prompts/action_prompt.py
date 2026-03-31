"""
动作执行提示词模板
==================
用于VLM低层动作决策的提示词模板

动作参数与interactive_navigation保持一致：
- TURN_LEFT/RIGHT: 30°（12步×30°=360°）
- MOVE_FORWARD: {0.25m, 0.5m, 0.75m, 1.0m, 1.25m}
"""

ACTION_EXECUTION_PROMPT = """You are the action execution module for Vision-Language Navigation. Analyze the environment and choose the next action.

# Current Subtask
**Destination**: {subtask_destination}
**Instruction**: {subtask_instruction}
**Subtask Progress**: {progress_summary}
**Previous Step Analysis**: {previous_action_reason}

# Space Structure
{waypoint_summary}

# Visual Observations

You have 1 image:

**Current View (front-facing, RGB HFOV about 79°)** — object detections plus 3 obstacle-distance lines:
- Directions: Left 30deg, FRONT, Right 30deg
- Red = nearest obstacle <0.5m (blocked), Yellow = 0.5-2m (caution), Green = >2m (open)
- If this stage is upstairs/downstairs and FRONT shows the needed stair run, or stair edge + rise/drop geometry shows the continuation, treat that short stair-facing distance as stair geometry, not a wall/block test. For downstairs, the needed run may look like an open drop / missing-floor direction beyond a stair edge or railing rather than clear lower steps
- **Yellow bounding box**: candidate subtask-landmark detection ({detected_landmarks}); first judge whether it is valid task evidence or noise
- **Bottom white strip** (if present): ranked landmark entries plus reachable `space waypoint` cues. Landmark entries are `vis/off vis + landmark name + distance + direction + confidence`; `vis` = detected now, `off vis` = mapped earlier but outside view

# Reasoning Process:

1. **Current View Content**: Analyze the current view from near to far. Judge whether visible cues belong to the current space or a farther transition/target space. Then compare FRONT, Left 30deg, and Right 30deg to decide where the destination or correct transition most likely is. Mention only visible evidence.
2. **Bottom Strip + Space Structure**: If the bottom strip is present, read each landmark and space waypoint. Judge landmark validity, distance, direction, and whether it signals arrival. For each space waypoint, judge whether it is current / next / behind, whether it matches the destination space, and whether it should be used or avoided.
3. **Current Position + Progress + Arrival Check**: Use nearby landmarks, valid detections, visible space waypoint cues, `Subtask Progress`, `Previous Step Analysis`, and space structure to confirm current position, stage progress, and the destination's relative position. Treat `Destination` as the stage goal and `Instruction` only as route relation: same-space pass/go-by/around keeps the cue intermediate; cross-space enter must finish the current enter-stage before any later target. Use `Subtask Progress` and `Previous Step Analysis` to avoid repeating a finished reorientation or intermediate cue. If `Subtask Progress` contains `(warning: front route blocked; forced stop)` or `Previous Step Analysis` says the last forward step was blocked, treat that as a one-call hard constraint: do not keep pushing straight into that same FRONT route; choose `STOP` only if arrival is already satisfied, otherwise choose a side turn. On that blocked-front retry, the effective action space is only `TURN_LEFT 30deg`, `TURN_RIGHT 30deg`, or valid `STOP`. For stair stages, judge stair bottom/top and the required up/down run from the current view; partly hidden stair edge, rise/drop trend, railing, landing geometry, and any open no-floor/drop side still count. If the task is downstairs, prefer the descending side/open drop rather than the upward run. Thinking usually hands off the task-aligned heading, so unless current evidence shows blocking or clear off-front target evidence, keep following it. Decide which side is task-aligned and which is wrong/backtracking. Then apply the arrival rule below: if reached, stop; otherwise continue.
   **Arrival rule**: `Destination` is the exact target space/place. If it is not yet reached, do not stop. Stop immediately once it is reached, or when the detected landmark itself is the destination and close enough to count as reached. Do not stop at an intermediate cue/opening/pass-by landmark if the instruction says to pass / go through / go around / cross it and continue to a farther destination. If that intermediate cue/opening is already beside/behind and the stage destination is still ahead, keep moving toward the destination instead of re-centering on the old cue. For solid-landmark destinations, stopping is allowed within about 0.75m or when clearly at hand. For opening-like destinations (entrance / doorway / hallway), stop only within about 0.5m or once the opening is already >90deg / behind, meaning the opening-stage destination has been passed through.
4. **Action Decision + Obstacle Avoidance**: Choose one safe immediate action that matches the subtask instruction and moves toward the destination. Apply the grouped rules below in order: arrival first, then stage-following and direction choice, then forward-step selection. Keep the final answer inside the fixed action space.
   **Action guidance**:
   a. **Destination-first stage following**: `Destination` is the priority target. Use `Instruction` only to decide whether to go directly or via a cue. Same-space pass/go-by/around keeps the cue as route guidance; cross-space enter stays on the current enter-stage destination, not a later target. Always read `Subtask Progress` and `Previous Step Analysis` to judge what is already finished and whether the last turn already aligned the agent.
   b. **Turn only when needed**: turn only when the destination or required cue is off-front, FRONT is blocked/tight, the previous-step notice says recent forward movement failed, or the instruction requires a side entry. If the stage is upstairs/downstairs and FRONT is the required stair run, or clearly enters it from a stair edge/top/bottom landing or descending open-drop side, do not side-turn just because the stair-facing distance is short. When FRONT is blocked (<0.5m) and is not the correct stair run, turn toward the side that still matches the destination/required cue; if one side is also blocked/tight, use the other open side. If the blocked-front warning is present, treat straight movement into that same FRONT route as forbidden on this call unless FRONT is clearly the required stair run. Do not turn into an open side if it is wrong-space or backtracking. After a detour, re-center once the destination becomes reachable again.
   c. **Forward-first when already aligned**: thinking usually hands off a destination-aligned heading. If `Subtask Progress` is empty / `Just started` and `Previous Step Analysis` is empty / `N/A (first step)`, treat the current facing as the default aligned heading; if FRONT is passable and no stronger evidence puts the destination off-front, prefer a forward move over exploratory turning. More generally, when FRONT is passable and task-aligned, prefer forward progress; use turns mainly for obstacle avoidance, task-required side entry, or clear off-front destination evidence. This does not apply to a blocked-front retry. Use small turns only for micro-adjustment or avoidance.
   d. **Forward distance selection**: choose forward distance from the best available target-distance evidence in this order: valid destination detection > valid subtask-landmark detection > bottom-strip landmark/space-waypoint distance > visible free-space depth. Use larger steps for far aligned targets and smaller steps for near targets or tight clearance. If the current heading is already task-aligned, choose the forward distance instead of spending the first step on extra turning. For upstairs/downstairs stages, if FRONT is the correct stair run, keep moving along it instead of treating that stair-facing distance as ordinary blocking. After a side detour, clear the obstacle with forward moves and turn back toward the destination as soon as the aligned route reappears.
   e. **Avoid turn oscillation**: do not repeat left/right reorientation without new evidence. If `Previous Step Analysis` shows the last action already turned toward the destination, especially as a forced obstacle-avoidance turn, and FRONT is now passable, follow with forward progress instead of another in-place turn unless obstacle avoidance or clearer off-front destination evidence requires it.
   f. **Connector / pass-by behavior**: if the instruction is to pass / go through / cross / enter toward a farther destination, do not linger at the doorway / hallway / passage mouth or beside the pass-by landmark. If the correct opening is in front and traversable, prefer a medium-to-large step that carries progress through it into the next space. If the pass-by cue is already effectively satisfied and the stage destination is now the main unfinished target in view, center the action on that destination.
   g. **Blocked / uncertain / already-passed cases**: treat FRONT <0.5m as blocked and never move into an obviously blocked direction, unless FRONT is clearly the task-aligned stair run for an upstairs/downstairs stage. Do not use `STOP` only because FRONT is blocked or the view is tight. If FRONT is blocked, compare Left 30deg and Right 30deg: prefer the side that matches the destination/required cue, or the only open side if just one is open. If the previous-step notice says forward just failed or `Subtask Progress` carries the blocked-front warning, do not answer with another forward into that same blocked FRONT route on this retry unless current evidence clearly shows it has reopened or it is the correct stair run. When two directions are open, prefer the one whose visible opening / space waypoint / structure relation matches the current subtask destination.
   h. **STOP discipline**: output `STOP` immediately iff the arrival rule above is satisfied for the current subtask destination. Otherwise do not stop merely because an intermediate cue is visible, a doorway is nearby, or a pass-by landmark is beside the agent.

# Output Format (JSON only)

Return exactly one JSON object and nothing else. Keep all reasoning inside the single `"reasoning"` string; never emit step titles or extra analysis blocks as new JSON keys. Do not add extra keys, markdown, or prose after the JSON. End immediately after the final `}`.

{{
    "reasoning": "One concise chain: current-view content, bottom-strip landmark/space-waypoint cues, current position + progress + structure-aligned arrival check, then the safest action toward the current-stage destination",
    "action_analysis": "One short sentence with the key evidence and why this action is best",
    "action": "<{allowed_action_output}>"
}}

**Action space**:
{allowed_action_bullets}

# Examples

**Ex1 - Clear path ahead**
{{
    "reasoning": "The heading is already aligned with the correct route, FRONT is open, and there is no progress or previous-step evidence requiring reorientation. The current-stage goal is the farther destination, not the intermediate connector cue, so use a large forward step.",
    "action_analysis": "FRONT is task-aligned and open, so forward progress is better than turning",
    "action": "MOVE_FORWARD 1.25m"
}}

**Ex2 - Obstacle detected**
{{
    "reasoning": "FRONT is blocked. Left leads to an off-task side space, while right is open and matches the next space-waypoint route, so use a temporary right detour and re-center afterward.",
    "action_analysis": "FRONT is blocked, and right is the destination-aligned open side",
    "action": "TURN_RIGHT 30deg"
}}

**Ex3 - Near but not yet reached**
{{
    "reasoning": "The destination is in front and close, but the arrival rule is not yet satisfied. FRONT is still passable, so take a short forward step.",
    "action_analysis": "A short forward move fits the near destination distance without overshooting",
    "action": "MOVE_FORWARD 0.25m"
}}

**Ex4 - Destination reached**
{{
    "reasoning": "The current visual evidence shows the subtask destination is already reached, so stop instead of adding another move.",
    "action_analysis": "The destination is already reached, so stopping is correct",
    "action": "STOP"
}}

**Critical Rules**:
- **Reasoning Flow**: keep reasoning concise and evidence-only. Follow the 4-step structure, mention only visible/listed cues, omit empty items, and never invent evidence.
- **Progress Awareness**: always read `Subtask Progress` and `Previous Step Analysis` to judge stage completion, satisfied route relation, and whether the last turn already aligned the agent. If `Subtask Progress` contains `(warning: front route blocked; forced stop)`, treat it as a one-call blocked-front retry and do not keep pushing into the same FRONT route. Do not repeat a completed reorientation or treat a passed intermediate cue as unfinished.
- **Arrival & Stop**: `Destination` is the exact stop target. If it is not yet reached, do not output `STOP`; if it is reached, output `STOP` immediately. Do not stop merely because FRONT is blocked, the view is narrow, or an intermediate cue/opening has appeared. The system also auto-ends the subtask when a highest-confidence top-2 destination landmark is close enough: about 0.75m for solid landmarks, about 0.5m for openings, or already >90deg / behind for passed-through openings.
- **Stage-Following & Direction**: treat `Destination` as the current-stage goal and `Instruction` as the route relation. Direct enter/approach stays on the destination route; pass / go through / go around / cross keeps moving through the cue toward the destination. If the task changes space, finish the current enter-stage before any later target. Use current visual evidence together with `Destination`, `Instruction`, `Subtask Progress`, `Previous Step Analysis`, visible space waypoint cues, and space structure to choose the correct next task space/landmark. For stairs, judge from the current view plus stage meaning whether the agent is at stair bottom or top, and whether the visible or partly hidden stair run is the required up/down route. For downstairs, the correct route may be an open drop / missing-floor direction rather than the clearest visible upward steps, so choose the descending side when task + geometry support it. Thinking usually hands off the correct facing, so prefer forward if FRONT stays task-aligned and passable. If FRONT is blocked and is not the correct stair run, turn toward the destination-aligned open side. If the blocked-front warning is present, do not output another straight move into that same FRONT route on this retry. Never choose a direction only because it is open if structure / space waypoint cues show a wrong, repeated, or backtracking branch.
- **Forward-Step Selection**: choose forward distance from target-distance evidence whenever possible: valid destination detection first, then valid subtask-landmark detection, then bottom-strip landmark/space-waypoint distance, then visible free-space depth. Far target = larger step, near target = smaller step. Prefer `MOVE_FORWARD 1.25m` or `1.0m` when the target/correct opening is clearly far (>2.5m) and the route is open, especially for pass-through / connector-crossing stages; prefer `0.75m` for about 1.5-2.5m; `0.5m` for about 0.8-1.5m; `0.25m` when near (<0.8m) but not yet reached or when clearance is limited. When `Subtask Progress` / `Previous Step Analysis` are empty or first-step defaults and the heading is already task-aligned, prefer forward and choose the distance directly. For connector-crossing / enter / pass-through stages with enough clearance, prefer medium-to-large forward movement that carries progress into the next space. For upstairs/downstairs stages, if FRONT is the correct stair run or clearly enters it from the top/bottom edge, keep moving along it instead of treating the stair-facing distance as a normal collision cue. If a pass-by / opening cue is already beside or behind but the stage destination is still ahead, keep centering on the destination rather than turning back. If you detour around an obstacle, clear it and turn back once the aligned route reappears. If the blocked-front warning is present, side-turn first; after that turn, if FRONT becomes passable and aligned, use forward instead of turning back. Avoid left-right oscillation when the last turn already aligned the route.
- **Output Limit**: use one common space type only; ignore modifiers and normalize corridor-like wording to `hallway`. Output `action` only from the fixed action space: `TURN_LEFT 30deg` / `TURN_RIGHT 30deg` / `MOVE_FORWARD {{0.25m, 0.5m, 0.75m, 1.0m, 1.25m}}` / `STOP`.
"""


DEFAULT_ALLOWED_ACTION_NAMES = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")


def _normalize_allowed_action_names(allowed_action_names=None):
    if not allowed_action_names:
        return list(DEFAULT_ALLOWED_ACTION_NAMES)

    allowed = {str(name or "").strip().upper() for name in allowed_action_names if str(name or "").strip()}
    ordered = [name for name in DEFAULT_ALLOWED_ACTION_NAMES if name in allowed]
    return ordered or list(DEFAULT_ALLOWED_ACTION_NAMES)


def _build_allowed_action_output(allowed_action_names=None) -> str:
    ordered = _normalize_allowed_action_names(allowed_action_names)
    choices = []
    if "MOVE_FORWARD" in ordered:
        choices.extend([
            "MOVE_FORWARD 0.25m",
            "MOVE_FORWARD 0.5m",
            "MOVE_FORWARD 0.75m",
            "MOVE_FORWARD 1.0m",
            "MOVE_FORWARD 1.25m",
        ])
    if "TURN_LEFT" in ordered:
        choices.append("TURN_LEFT 30deg")
    if "TURN_RIGHT" in ordered:
        choices.append("TURN_RIGHT 30deg")
    if "STOP" in ordered:
        choices.append("STOP")
    return " | ".join(choices)


def _build_allowed_action_bullets(allowed_action_names=None) -> str:
    ordered = _normalize_allowed_action_names(allowed_action_names)
    lines = []
    if "MOVE_FORWARD" in ordered:
        lines.append("- `MOVE_FORWARD {0.25m, 0.5m, 0.75m, 1.0m, 1.25m}`")
    turn_parts = []
    if "TURN_LEFT" in ordered:
        turn_parts.append("`TURN_LEFT 30deg`")
    if "TURN_RIGHT" in ordered:
        turn_parts.append("`TURN_RIGHT 30deg`")
    if turn_parts:
        lines.append("- " + " | ".join(turn_parts))
    if "STOP" in ordered:
        lines.append("- `STOP`")
    return "\n".join(lines)


def get_action_execution_prompt(next_waypoint_destination: str,
                                subtask_instruction: str,
                                progress_summary: str = "",
                                waypoint_summary: str = "",
                                detected_landmarks: str = None,
                                previous_action_reason: str = "",
                                landmark_map_info: str = None,
                                allowed_action_names = None,
                                move_distance: float = 0.25,
                                turn_angle: int = 30,
                                # 以下参数保留兼容性但不再用于prompt
                                **kwargs) -> str:
    """获取动作执行提示词"""
    if not progress_summary:
        progress_summary = "Just started"
    if not waypoint_summary:
        waypoint_summary = "No space structure recorded yet."

    return ACTION_EXECUTION_PROMPT.format(
        subtask_destination=next_waypoint_destination,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary,
        waypoint_summary=waypoint_summary,
        previous_action_reason=previous_action_reason or "N/A (first step)",
        detected_landmarks=detected_landmarks or "none",
        allowed_action_output=_build_allowed_action_output(allowed_action_names),
        allowed_action_bullets=_build_allowed_action_bullets(allowed_action_names),
        move_distance=move_distance,
        turn_angle=turn_angle,
    )
