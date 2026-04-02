"""
动作执行提示词模板
==================
用于VLM低层动作决策的提示词模板

动作参数与interactive_navigation保持一致：
- TURN_LEFT/RIGHT: 30°（12步×30°=360°）
- MOVE_FORWARD: {0.25m, 0.5m, 0.75m, 1.0m, 1.25m}
"""

import re

from vlnce_baselines.config.core.params.actions import (
    ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M,
    ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M,
)
from vlnce_baselines.config.core.params.thresholds import (
    OBS_BLOCKED_M,
    OBS_OPEN_M,
    OBS_RISKY_M,
)


def _fmt_threshold_m(value: float) -> str:
    text = f"{float(value):.2f}"
    if text.endswith("00"):
        return f"{float(value):.1f}"
    return text.rstrip("0").rstrip(".")


def _parse_distance_m(distance_text) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)", str(distance_text or ""))
    if not match:
        return -1.0
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return -1.0


def _format_obstacle_state(distance_text) -> str:
    distance_m = _parse_distance_m(distance_text)
    if distance_m < 0.0:
        return ""
    if distance_m < float(OBS_BLOCKED_M):
        return "warning"
    if distance_m > float(OBS_OPEN_M):
        return "open"
    return ""


def _build_obstacle_perception_summary(obstacle_distances=None) -> str:
    distances = dict(obstacle_distances or {})
    items = []
    for label, key in (
        ("FRONT", "front"),
        ("Left 30deg", "left_30"),
        ("Right 30deg", "right_30"),
    ):
        distance_text = distances.get(key, "Unknown")
        state_text = _format_obstacle_state(distance_text)
        lower_text = str(distance_text or "").strip().lower()
        has_state_text = any(token in lower_text for token in ("warning", "open"))
        if state_text and not has_state_text:
            items.append(f"{label} {distance_text} {state_text}")
        else:
            items.append(f"{label} {distance_text}")
    return " | ".join(items)


def _build_landmark_perception_summary(detected_landmarks=None, landmark_map_info=None) -> str:
    lines = []
    landmark_map_text = str(landmark_map_info or "").strip()
    detected_text = str(detected_landmarks or "").strip()
    if landmark_map_text:
        for line in landmark_map_text.splitlines():
            clean_line = str(line).rstrip()
            if clean_line:
                lines.append(clean_line)
    elif detected_text and not detected_text.lower().startswith("no "):
        lines.append(f"- raw detected landmarks: {detected_text}")

    if not lines:
        return "- no valid visible landmark entries"
    return "\n".join(lines)

ACTION_EXECUTION_PROMPT = """You are the action execution module for Vision-Language Navigation. Analyze the environment and choose the next action.

# Current Subtask
**Destination**: {subtask_destination}
**Instruction**: {subtask_instruction}
**Subtask Progress**: {progress_summary}
**Previous Step Analysis**: {previous_action_reason}

# Environment Perception
**Obstacle**: {obstacle_perception_summary}
**Landmark**:
{landmark_perception_summary}

# Space Structure
{waypoint_summary}

# Visual Observations
You have 1 image.
**Current View (front-facing, RGB HFOV about 79°)** — object detections plus 3 obstacle-distance lines:
- Directions: Left 30deg, FRONT, Right 30deg
- Read `Environment Perception` first: `Obstacle` is the current depth-based 3-direction summary in one line; `Landmark` lists the current-view top visible entries
- Red = nearest obstacle <{obs_blocked_m}m (blocked), Yellow = {obs_blocked_m}-{obs_risky_m}m or {obs_risky_m}-{obs_open_m}m (not open), Green = >{obs_open_m}m (open)
- If this stage is a connector-like route such as doorway / hallway / passage / stairs, analyze current RGB geometry and prefer the visible opening or stair run middle / centerline instead of hugging a side wall, door frame, railing, or corner. If this stage is upstairs/downstairs and FRONT shows the needed stair run, or stair edge + rise/drop geometry shows the continuation, treat that short stair-facing distance as stair geometry, not a wall/block test. For downstairs, the needed run may look like an open drop / missing-floor direction beyond a stair edge or railing
- **Yellow bounding box**: candidate current-view landmark detection ({detected_landmarks}); first judge whether it is valid task evidence or noise. If the label/box conflicts with the RGB scene, local geometry, obstacle layout, or task/space context, downweight or ignore it
- **Bottom white strip** (if present): auxiliary only; if it is missing or hard to read, trust the text landmark entries in `Environment Perception`

# Reasoning Process:

1. **Environment Perception + Current View**: read FRONT / Left 30deg / Right 30deg obstacle sensing first, then analyze the image from near to far. Judge whether visible cues belong to the current space or a farther transition/target space. Compare the three directions to decide where the destination or correct transition most likely is. For doorway / hallway / passage / stairs stages, identify the traversable middle / centerline of the correct route from current image geometry, not just the nearest edge. Mention only visible/listed evidence, especially obstacle distance, visible landmark distance/direction, whether the destination or task-relevant landmark is visible, and whether a shown landmark looks plausible or noisy.
2. **Landmark Perception + Space Structure**: read the top visible landmark entries from `Environment Perception`, then use space structure. Judge landmark validity, distance, direction, confidence, arrival value, and whether each space waypoint is current / next / behind / avoid. Use landmark distance/direction with image content; never decide from name alone. If a landmark label/box conflicts with the RGB scene, local geometry, obstacle layout, destination semantics, or space structure, treat it as weak/noisy evidence rather than the main target cue.
3. **Current Position + Progress + Arrival Check**: use nearby landmarks, valid detections, visible space waypoint cues, `Subtask Progress`, `Previous Step Analysis`, space structure, and current image content to confirm current position, stage progress, and the destination's relative position. Treat `Destination` as the stage goal and `Instruction` only as route relation: same-space pass/go-by/around keeps the cue intermediate; cross-space enter must finish the current enter-stage first. Judge whether the destination / subtask landmark is far, already near / beside, or not reliably visible. If the destination landmark is near and clearly side-offset, face it before forward; if it is far and only mildly side-front while FRONT stays passable and task-aligned, keep forward progress first. If a detected landmark looks wrong or weak, do not turn toward it just because it is labeled; fall back to the safer task-aligned opening / space structure / obstacle-supported route. Use `Subtask Progress` and `Previous Step Analysis` to avoid repeating a finished reorientation or intermediate cue. If the last step already turned to avoid an obstacle or face the destination and FRONT is now passable and destination-aligned, go forward rather than turning again. If `Subtask Progress` contains `(warning: front route blocked; forced stop)` or `Previous Step Analysis` says the last forward step was blocked, treat that as a one-call hard constraint: do not push into that same FRONT route; choose `STOP` only if arrival is already satisfied, otherwise choose a side turn. On that retry, the effective action space is only `TURN_LEFT 30deg`, `TURN_RIGHT 30deg`, or valid `STOP`. For stair stages, judge stair bottom/top and the needed up/down run from the current view; partly hidden stair edge, rise/drop trend, railing, landing geometry, and open no-floor/drop side still count. If the task is downstairs, prefer the descending side/open drop. Thinking usually hands off the task-aligned heading, so unless current evidence shows blocking or clear off-front target evidence, keep following it. Then apply the arrival rule below.
   **Arrival rule**: `Destination` is the exact target space/place. If it is not yet reached, do not stop. Stop immediately once it is reached, including when the detected landmark itself is the destination and close enough to count as reached. Do not stop at an intermediate cue/opening/pass-by landmark if the instruction says to pass / go through / go around / cross it and continue to a farther destination. If that intermediate cue/opening is already beside/behind and the stage destination is still ahead, keep moving toward the destination. For solid-landmark destinations, stopping is allowed within about {solid_autocomplete_m}m or when clearly at hand. For opening-like destinations (entrance / doorway / hallway), stop only within about {open_autocomplete_m}m or once the opening is already >90deg / behind, meaning the opening-stage destination has been passed through.
4. **Action Decision + Obstacle Avoidance**: choose one safe immediate action that follows the subtask and moves toward the destination. Jointly analyze image content, landmark distance/direction, obstacle distance, task instruction, and arrival state. Apply the grouped rules below in order: arrival, then stage-following/direction, then forward-step selection. Keep the final answer inside the fixed action space.
   **Action guidance**:
   a. **Destination-first stage following**: `Destination` is the priority target. Use `Instruction` only to decide whether to go directly or via a cue. Same-space pass/go-by/around keeps the cue as route guidance; cross-space enter stays on the current enter-stage destination. Always read `Subtask Progress` and `Previous Step Analysis` to judge what is already finished and whether the last turn already aligned the agent. Treat landmark detections as supporting evidence unless the current view and task context also support them. Choose the road/opening that actually leads toward the correct destination, not just the nearest open side.
   b. **Turn only when needed**: turn only when the destination or required cue is off-front, FRONT is blocked/tight, recent forward movement failed, or the instruction requires a side entry. If the destination landmark is already near and clearly off-front / beside, turn to face it before forward, but only when it looks visually valid and task-aligned. If a landmark is only a pass-by / intermediate cue toward a farther destination, do **not** turn to face it when near. For doorway / hallway / passage / stairs stages, face the correct route middle / centerline, not a side edge or frame. If the stage is upstairs/downstairs and FRONT is the required stair run or clearly enters it from stair geometry, do not side-turn just because the stair-facing distance is short; keep advancing along the needed up/down direction. Otherwise, when FRONT is blocked or warning/tight, rule out blocked/tight side routes first, prefer the non-warning open/passable side, and use destination alignment only to break ties between similarly safe sides. If one side is open/passable and the other is warning/blocked, choose the open/passable side even if the landmark lies slightly toward the worse side, then re-center later. If the blocked-front warning is present, treat straight movement into that same FRONT route as forbidden on this call unless FRONT is clearly the required stair run. Do not turn into a wrong-space or backtracking open side. After a detour, re-center once the destination is reachable again.
   c. **Forward-first when already aligned**: thinking usually hands off a destination-aligned heading. If `Subtask Progress` is empty / `Just started` and `Previous Step Analysis` is empty / `N/A (first step)`, treat the current facing as aligned; if FRONT is passable and no stronger evidence puts the destination off-front, prefer forward over exploratory turning. If the destination or most relevant task landmark is clearly and plausibly visible ahead and still far, prefer forward. If the destination landmark is still far and only mildly off-front, such as in the left/right 30deg side-front sector, prefer forward first and adjust later. More generally, when FRONT is task-aligned, prefer forward; use turns mainly for obstacle avoidance, task-required side entry, or clear off-front destination evidence.
   d. **Forward distance selection**: choose forward distance from the best available target-distance evidence in this order: valid destination detection > valid subtask-landmark detection > bottom-strip landmark/space-waypoint distance > visible free-space depth. Use larger steps for far aligned targets and smaller steps for near targets or tight clearance. If the destination / landmark is visible and still far, keep moving forward. If a near landmark is **not** the destination, keep moving toward the true destination rather than centering the cue. If the destination itself is near and clearly off-front / beside, adjust heading first; if it is far and only slightly side-front, keep forward progress first. If the current heading is already task-aligned, choose forward distance instead of extra turning. For upstairs/downstairs stages, if FRONT is the correct stair run, keep moving along the stair middle / centerline in the needed up/down direction instead of treating that stair-facing distance as ordinary blocking. After a side detour, clear the obstacle with forward moves and turn back once the aligned route reappears.
   e. **Avoid turn oscillation**: do not repeat left/right reorientation without new evidence. If `Previous Step Analysis` shows the last action already turned toward the destination, or already turned to avoid an obstacle and re-align the route, treat that reorientation as finished. If FRONT is now passable and still task-aligned, follow with forward progress instead of another in-place turn. Only turn again if new evidence shows the destination is still off-front, the front route is still blocked, or the previous turn was clearly insufficient.
   f. **Connector / pass-by behavior**: if the instruction is to pass / go through / cross / enter toward a farther destination, do not linger at the doorway / hallway / passage mouth or beside the pass-by landmark. Use real-time image geometry to decide the correct opening/run and its traversable middle. If the correct opening is in front and traversable, align to its middle / centerline and prefer a medium-to-large step that carries progress through it into the next space. If the pass-by cue is already effectively satisfied and the stage destination is now the main unfinished target in view, center the action on that destination.
   g. **Blocked / uncertain / already-passed cases**: treat FRONT <{obs_blocked_m}m as blocked and never move into an obviously blocked direction, unless FRONT is clearly the task-aligned stair run for an upstairs/downstairs stage. Treat {obs_blocked_m}-{obs_risky_m}m as warning/risky: if a better open/passable task-aligned side exists, do not keep pushing FRONT. Do not use `STOP` only because FRONT is blocked or the view is tight. If FRONT is blocked or warning/tight for the intended advance, compare Left 30deg and Right 30deg and analyze their image content too: eliminate blocked/tight sides first, prefer the non-warning open/passable side, and use destination / landmark direction and instruction only to break ties between similarly safe choices. If one side is open/passable and the other is warning/blocked, choose the open/passable side even if the landmark is slightly closer to the worse side; re-center after clearing the obstacle. If the previous-step notice says forward just failed or `Subtask Progress` carries the blocked-front warning, do not answer with another forward into that same blocked FRONT route on this retry unless current evidence clearly shows it has reopened or it is the correct stair run. When two directions are open, prefer the one whose visible opening / space waypoint / landmark relation best matches the current subtask destination.
   h. **STOP discipline**: output `STOP` immediately iff the arrival rule above is satisfied for the current subtask destination. Otherwise do not stop merely because an intermediate cue is visible, a doorway is nearby, or a pass-by landmark is beside the agent.

# Output Format (JSON only)

Return exactly one JSON object. Keep all reasoning inside `"reasoning"`; never emit step titles or extra analysis blocks as new keys. No extra keys, markdown, or prose. End at the final `}}`.

{{
    "reasoning": "One concise chain: environment perception + current-view content, landmark/space-structure cues, position/progress/arrival check, then the safest action toward the current-stage destination",
    "action_analysis": "One short sentence with the key evidence and why this action is best",
    "action": "<{allowed_action_output}>"
}}

**Action space**:
{allowed_action_bullets}

# Examples

**Ex1 - Clear path ahead**
{{
    "reasoning": "The heading is aligned, FRONT is open, and the current-stage goal is still far and only mildly side-front, so forward progress is better than early turning.",
    "action_analysis": "FRONT is task-aligned and open, so keep moving forward before refining angle",
    "action": "MOVE_FORWARD 1.25m"
}}

**Ex2 - Obstacle detected**
{{
    "reasoning": "FRONT is warning/blocked and the right side is also warning, while left is the only open/passable side. Do not turn into the warning side just because the target is slightly right-biased; clear the obstacle via the safer left side, then re-center.",
    "action_analysis": "FRONT is not safely traversable, and left is the only open side",
    "action": "TURN_LEFT 30deg"
}}

**Ex3 - Near but not yet reached**
{{
    "reasoning": "The destination is in front and close, but the arrival rule is not yet satisfied. FRONT is still passable, so take a short forward step.",
    "action_analysis": "A short forward move fits the near destination distance without overshooting",
    "action": "MOVE_FORWARD 0.25m"
}}

**Ex4 - Destination reached**
{{
    "reasoning": "The subtask destination is already reached, so stop instead of adding another move.",
    "action_analysis": "The destination is already reached, so stopping is correct",
    "action": "STOP"
}}

**Critical Rules**:
- **Reasoning Flow**: keep reasoning concise and evidence-only. Follow the 4-step structure, mention only visible/listed cues, omit empty items, and never invent evidence.
- **Progress Awareness**: always read `Subtask Progress` and `Previous Step Analysis` to judge stage completion, satisfied route relation, and whether the last turn already aligned the agent. If `Subtask Progress` contains `(warning: front route blocked; forced stop)`, treat it as a one-call blocked-front retry and do not keep pushing into the same FRONT route.
- **Turn-Then-Go Rule**: if `Previous Step Analysis` shows the previous action already turned for obstacle avoidance or destination alignment, and FRONT is now passable and task-aligned, prefer `MOVE_FORWARD` on this call. Do not alternate left/right turns in place without new evidence.
- **Visible-Target Rule**: if the destination or most relevant task landmark is visibly plausible and still far, especially if only mildly side-front, prefer forward. If it is already near and clearly off-front / beside, adjust angle first. A dubious detection label alone is never enough reason to turn.
- **Landmark Validity**: landmark detections are candidate evidence, not ground truth. Validate them against RGB appearance, local geometry, obstacle layout, task destination, and space structure. If an entry looks noisy, inconsistent, or wrong-space, downweight or ignore it and follow the safer task-aligned route instead.
- **Connector Centerline**: for doorway / hallway / passage / stairs stages, analyze the current image and move through the correct route middle / centerline. Do not hug side walls, door frames, or railings, and do not choose a side branch just because it looks open if it is not the correct destination route.
- **Arrival & Stop**: `Destination` is the exact stop target. If it is not yet reached, do not output `STOP`; if it is reached, output `STOP` immediately. Do not stop merely because FRONT is blocked or an intermediate cue/opening appears.
- **Stage-Following & Direction**: treat `Destination` as the current-stage goal and `Instruction` as the route relation. Finish the current enter-stage before any later target. For stairs, infer the required up/down run from the current view + stage meaning; for downstairs, choose the descending side when task + geometry support it. Once the correct stair run is identified, prefer its middle / centerline and keep advancing. If FRONT is blocked or warning/tight and is not the correct stair run, prefer a destination-supporting non-warning side over a warning/blocked side.
- **Forward-Step Selection**: choose forward distance from target-distance evidence whenever possible: destination detection, then subtask-landmark detection, then bottom-strip landmark/space-waypoint distance, then visible free-space depth. If FRONT is task-aligned and passable, prefer forward. If the blocked-front warning is present, side-turn first; after that turn, if FRONT becomes passable and aligned, use forward instead of turning back. If the destination is near but clearly off-front, turn first; if it is still far and only mildly side-front, keep forward. Avoid left-right oscillation when the last turn already aligned the route.
- **Output Limit**: use one common space type only and normalize corridor-like wording to `hallway`. Output `action` only from the fixed action space: `TURN_LEFT 30deg` / `TURN_RIGHT 30deg` / `MOVE_FORWARD {{0.25m, 0.5m, 0.75m, 1.0m, 1.25m}}` / `STOP`.
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


def get_action_execution_prompt(next_waypoint: str,
                                subtask_instruction: str,
                                progress_summary: str = "",
                                waypoint_summary: str = "",
                                detected_landmarks: str = None,
                                previous_action_reason: str = "",
                                obstacle_distances = None,
                                landmark_map_info: str = None,
                                allowed_action_names = None,
                                move_distance: float = 0.25,
                                turn_angle: int = 30) -> str:
    """获取动作执行提示词"""
    if not progress_summary:
        progress_summary = "Just started"
    if not waypoint_summary:
        waypoint_summary = "No space structure recorded yet."

    return ACTION_EXECUTION_PROMPT.format(
        subtask_destination=next_waypoint,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary,
        waypoint_summary=waypoint_summary,
        previous_action_reason=previous_action_reason or "N/A (first step)",
        detected_landmarks=detected_landmarks or "none",
        obstacle_perception_summary=_build_obstacle_perception_summary(obstacle_distances),
        landmark_perception_summary=_build_landmark_perception_summary(
            detected_landmarks=detected_landmarks,
            landmark_map_info=landmark_map_info,
        ),
        allowed_action_output=_build_allowed_action_output(allowed_action_names),
        allowed_action_bullets=_build_allowed_action_bullets(allowed_action_names),
        obs_blocked_m=_fmt_threshold_m(OBS_BLOCKED_M),
        obs_risky_m=_fmt_threshold_m(OBS_RISKY_M),
        obs_open_m=_fmt_threshold_m(OBS_OPEN_M),
        open_autocomplete_m=_fmt_threshold_m(ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M),
        solid_autocomplete_m=_fmt_threshold_m(ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M),
        move_distance=move_distance,
        turn_angle=turn_angle,
    )
