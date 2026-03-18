"""
动作执行提示词模板
==================
用于VLM低层动作决策的提示词模板

动作参数与interactive_navigation保持一致：
- TURN_LEFT/RIGHT: 30°（12步×30°=360°）
- MOVE_FORWARD: 0.25m
"""

ACTION_EXECUTION_PROMPT = """You are the action execution module for Vision-Language Navigation. Analyze the environment and decide the next action.

# Current Subtask
**Destination**: {subtask_destination}
**Instruction**: {subtask_instruction}

# Progress Summary
{progress_summary}

# Space Structure
{waypoint_summary}

# Previous Step Analysis
{previous_action_reason}

# Visual Observations

You are provided with 1 image:

**Current View (front-facing, RGB HFOV about 79°)** — Object detection overlaid with 3 depth-sampled obstacle-distance lines:
- Directions: Left 30deg, FRONT, Right 30deg
- Red = nearest obstacle <0.5m (blocked), Yellow = 0.5–2m (caution), Green = >2m (open)
- **Yellow bounding box**: candidate subtask-landmark detection ({detected_landmarks}); first judge whether it is valid task-relevant evidence or just duplicate/noisy evidence
- **Bottom white strip** (if present): ranked landmark entries plus reachable `space waypoint` cues. Landmark entries are `vis/off vis + landmark name + distance + direction + confidence`; `vis` = detected now, `off vis` = mapped earlier but outside the current view

# Your Task

**Decision Process**:
1. **Detection + View Analysis**: Read detection + space cues in the current view first, ordered from near to far. Check whether each current detection is valid subtask evidence, duplicate evidence, or weak/noisy evidence. Also read visible `space waypoint` / space-area cues in the strip if present. Then analyze FRONT, Left 30deg, and Right 30deg using only visible evidence: likely room/space, near objects, farther objects, valid landmark(s), valid space cue(s), and where the subtask destination most likely is. If something is not visible, do not mention it and do not write filler like `none`.
2. **Space Structure**: Analyze two parts under Space Structure. First read **space areas**: merged same-type regions such as Bedroom1 / Hallway1 with any shown links. Then read **space waypoints**: `Space WP#...` entries with snapped direction/distance. Use both to infer which way continues toward the most likely task-relevant room/object.
3. **Current Position + Destination Relation**: From the current view plus the space structure, infer where you are and where the target room/object most likely is.
4. **Arrival Check**: If the current evidence shows you are already at the subtask destination, stop immediately. During normal action steps, the system will also automatically end the current subtask and switch to thinking if one of the highest-confidence top-2 displayed destination landmarks becomes near enough. Opening-like cues (entrance / doorway / hallway / passage) use about 0.5m; solid object cues use about 1.0m.
5. **Depth + Avoidance**: Check Left 30 / Front / Right 30 for obstacle safety. Avoid blocked directions and prefer safe progress toward the destination.
6. **Action Decision**: Base the action on the visual analysis first, then the subtask destination, then the subtask instruction and space structure. Prefer the direction that best matches the destination and instruction while staying safe. If FRONT is blocked, choose the safest side that still stays closest to the destination.

**Safety Priority**: Avoid directions with red distance lines (<0.5m). Never move into an obviously blocked direction.

# Output Format (JSON only)

{{
    "reasoning": "Brief chain: view analysis (detection + space cues, near to far), current position vs destination, arrival check, depth safety / obstacle avoidance, final action",
    "action_analysis": "One short sentence with the key evidence and why this action is best",
    "action": "<MOVE_FORWARD 0.25m | MOVE_FORWARD 0.5m | MOVE_FORWARD 0.75m | MOVE_FORWARD 1.0m | MOVE_FORWARD 1.25m | TURN_LEFT 30deg | TURN_RIGHT 30deg | STOP>"
}}

**Action space**:
- `MOVE_FORWARD {{0.25m, 0.5m, 0.75m, 1.0m, 1.25m}}`
- `TURN_LEFT 30deg` | `TURN_RIGHT 30deg`
- `STOP`

# Examples

**Ex1 - Clear path ahead**
{{
    "reasoning": "Front is open, the target-area cue stays ahead, and the space structure still aligns forward. Left and right do not better match the route, so move forward.",
    "action_analysis": "Forward best matches the visible target cue and the space structure with a clear front path",
    "action": "MOVE_FORWARD 0.75m"
}}

**Ex2 - Obstacle detected**
{{
    "reasoning": "Front is blocked, right is open, and right aligns better than left with the space waypoint route toward the sofa area. Turn right.",
    "action_analysis": "Front is blocked, and right is the safest direction that still stays aligned with the target route",
    "action": "TURN_RIGHT 30deg"
}}

**Ex3 - Near destination cue**
{{
    "reasoning": "The current visual evidence shows the subtask destination is already reached, so stop immediately instead of adding another move.",
    "action_analysis": "The destination is already reached in the current view, so stopping is the correct action",
    "action": "STOP"
}}

**Critical Rules**:
- if the current evidence already shows the subtask destination is reached, output `STOP` immediately
- during normal action steps the system also automatically ends the current subtask when one of the highest-confidence top-2 displayed destination landmarks is within its stop range: about 0.5m for opening-like cues and about 1.0m for solid objects
- reasoning must stay concise and evidence-only: cover FRONT/LEFT30/RIGHT30, visible/off-screen landmarks if present, visible space cues if present, current position, destination room/object relation, space-area and space-waypoint alignment, and depth safety / obstacle avoidance before choosing an action; omit empty items and never invent evidence
- use one common room/space type only; ignore modifiers and normalize corridor-like wording to `hallway`
- output `action` must stay inside the fixed action space: `TURN_LEFT 30deg` / `TURN_RIGHT 30deg` / `MOVE_FORWARD {{0.25m, 0.5m, 0.75m, 1.0m, 1.25m}}` / `STOP` (`STOP` is compatibility fallback only)
- If the destination is ahead and FRONT is clear, prefer MOVE_FORWARD
- If FRONT is blocked, choose the closest safe side direction toward the destination, not a wider detour
- For off-screen landmarks, **always turn toward the indicated direction first**
- Use the current visual evidence together with the subtask destination, subtask instruction, and space structure to keep moving toward the most likely relevant space/object
"""


def get_action_execution_prompt(next_waypoint_destination: str,
                                subtask_instruction: str,
                                progress_summary: str = "",
                                waypoint_summary: str = "",
                                detected_landmarks: str = None,
                                previous_action_reason: str = "",
                                landmark_map_info: str = None,
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
        move_distance=move_distance,
        turn_angle=turn_angle,
    )
