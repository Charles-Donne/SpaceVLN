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

# Waypoint History
{waypoint_summary}

# Previous Step Analysis
{previous_action_reason}

# Visual Observations

You are provided with 1 image:

**Current View (front-facing, RGB HFOV about 79°)** — Object detection overlaid with 3 depth-sampled obstacle-distance lines:
- Directions: Left 30deg, FRONT, Right 30deg
- Red = nearest obstacle <0.5m (blocked), Yellow = 0.5–2m (caution), Green = >2m (open)
- **Yellow bounding box**: candidate subtask-landmark detection ({detected_landmarks}); first judge whether it is valid task-relevant evidence or just duplicate/noisy evidence
- **Bottom white strip** (if present): top-3 landmark entries only, ranked by confidence then distance, in the form `vis/off vis + landmark name + distance + direction + confidence`; `vis` = detected now, `off vis` = mapped earlier but outside the current view

# Your Task

**Decision Process**:
1. **Detection + View Analysis**: Read `vis/off vis` first. Check whether each current detection is a valid subtask landmark, a duplicate of the same object, or weak/noisy evidence. Then analyze FRONT, Left 30deg, and Right 30deg using only visible evidence: likely room/space, near objects, far objects, and valid landmark(s) with distance/angle/confidence. If something is not visible, do not mention it and do not write filler like `none`.
2. **Waypoint History**: Read WP#1 -> ... -> LAST. Use snapped direction/distance to infer which way continues toward the most likely task-relevant room/object.
3. **Current Position + Destination Relation**: From the current view plus waypoint history, infer where you are and where the target room/object most likely is.
4. **System Auto-Complete**: During normal action steps, the system will automatically end the current subtask and switch to thinking if a displayed top-3 destination landmark enters about 0.5m. Do not rely on `STOP` for that; focus on the best immediate movement.
5. **Depth Lines**: Which of Left 30 / Front / Right 30 is blocked vs safe?
6. **Action Decision**: Prefer the direction that best matches current position + destination relation + waypoint history + current room/object evidence. If FRONT is blocked, choose the safest side that stays closest to the destination.

**Safety Priority**: Avoid directions with red distance lines (<0.5m)

# Output Format (JSON only)

{{
    "reasoning": "Brief chain: view + landmarks, waypoint history, current position vs destination, system auto-complete note, depth safety, final action",
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
    "reasoning": "Front is open, the target-area cue stays ahead, and waypoint history still aligns forward. Left and right do not better match the route, so move forward.",
    "action_analysis": "Forward best matches the visible target cue and waypoint history with a clear front path",
    "action": "MOVE_FORWARD 0.75m"
}}

**Ex2 - Obstacle detected**
{{
    "reasoning": "Front is blocked, right is open, and right aligns better than left with the waypoint route toward the sofa area. Turn right.",
    "action_analysis": "Front is blocked, and right is the safest direction that still stays aligned with the target route",
    "action": "TURN_RIGHT 30deg"
}}

**Ex3 - Near destination cue**
{{
    "reasoning": "The target cue is already very near, waypoint history stays aligned, and front is still safe. Keep the action minimal; the system will auto-end the current subtask once the displayed destination landmark enters about 0.5m.",
    "action_analysis": "The target cue is already close, so a minimal aligned move is better while the system handles subtask completion",
    "action": "MOVE_FORWARD 0.25m"
}}

**Critical Rules**:
- the system automatically ends the current subtask during normal action steps when a displayed top-3 destination landmark is within about 0.5m, so do not use `STOP` just because the destination cue looks near
- reasoning must stay concise and evidence-only: cover FRONT/LEFT30/RIGHT30, visible/off-screen landmarks if present, current position, destination room/object relation, waypoint-history alignment, and depth safety before choosing an action; omit empty items and never invent evidence
- use one common room/space type only; ignore modifiers and normalize corridor-like wording to `hallway`
- output `action` must stay inside the fixed action space: `TURN_LEFT 30deg` / `TURN_RIGHT 30deg` / `MOVE_FORWARD {{0.25m, 0.5m, 0.75m, 1.0m, 1.25m}}` / `STOP` (`STOP` is compatibility fallback only)
- If the destination is ahead and FRONT is clear, prefer MOVE_FORWARD
- If FRONT is blocked, choose the closest safe side direction toward the destination, not a wider detour
- For off-screen landmarks, **always turn toward the indicated direction first**
- Use waypoint history and current room/object evidence to keep moving toward the most likely relevant space/object
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
        waypoint_summary = "No waypoints visited yet."

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
