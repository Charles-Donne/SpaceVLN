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
- **Yellow bounding box**: marks the subtask **landmark reference** ({detected_landmarks}) — a recognizable object near the destination area, **not the destination itself**; use it as a visual anchor to navigate into the right area
- **Bottom white strip** (if present): all landmark names, distances and directions — `[Visible]` = detected in current frame, `[Off-screen]` = mapped but outside current view

{known_landmark_section}

# Your Task

**Decision Process**:
1. **Detection View**: Check the **bottom strip** for all landmark names and distances. Is a yellow bbox visible? It marks the **subtask landmark reference** (nearby anchor, NOT the final destination) — navigate toward it to reach the destination area.
2. **Image Analysis**: Analyze the current image by LEFT/CENTER/RIGHT and NEAR/FAR. Identify likely room/space cues, large nearby objects, smaller farther objects, and where the visible landmark/detected objects sit; use them to infer what space/room each region most likely is.
3. **Waypoint History**: Read WP#1 -> ... -> LAST in order. Use each waypoint's snapped direction/distance to infer which direction continues toward the most likely task-relevant space/object from the explored route.
4. **Arrival Check**: First judge whether you are in the correct room/space, then judge whether the destination object in that room is already within ~1.0m. Only then **STOP immediately**.
5. **Depth Lines**: Which of Left 30 / Front / Right 30 is blocked vs safe?
6. **Distance Estimation**: How far to destination? (e.g., "~3m", "<0.5m")
7. **Action Decision**: After the full reasoning above, prefer the direction that best matches waypoint history + current room/object evidence toward the most likely relevant space/object. If FRONT is blocked, choose the safest side direction that stays closest to the destination.

**STOP Condition** — If the current subtask destination area is already nearby, STOP immediately. Concretely: the correct room/space is reached and the destination object is within ~1.0m. Do not move past it or take extra steps. Do not STOP early if you are still outside the correct room or not yet beside the target object.

**Movement Rule**: Move forward when the destination remains ahead and the path is clear; once the correct room/object target is reached, STOP immediately.

**Safety Priority**: Avoid directions with red distance lines (obstacle <0.5m)

# Output Format (JSON only)

{{
    "reasoning": "Logic: (1) LEFT/CENTER/RIGHT + NEAR/FAR image analysis with room/object cues and detected objects (2) likely space/room for each region (3) waypoint-history alignment (4) destination/landmark distance and arrival check (5) action decision",
    "action_analysis": "One sentence stating the key visual evidence, waypoint-history cue, and why this action is best",
    "action": "MOVE_FORWARD" | "TURN_LEFT" | "TURN_RIGHT" | "STOP",
    "value": 0,
    "progress_summary": "Updated summary: actions taken, current facing direction, locations entered/bypassed"
}}

**Parameter rules**:
- MOVE_FORWARD: "value" = meters (0.25 ~ 1.5)
- TURN_LEFT / TURN_RIGHT: "value" = degrees (30 ~ 90, multiples of 30)
- STOP: "value" = 0

# Examples

**Ex1 - Clear path ahead**
{{
    "reasoning": "CENTER-NEAR is open, CENTER-FAR shows the target area cue, LEFT/RIGHT do not better match the route. Waypoint history still points forward to the relevant room/object. Front depth line is open, so move forward.",
    "action_analysis": "Forward best matches the visible target cue and waypoint history with a clear front path",
    "action": "MOVE_FORWARD",
    "value": 0.75,
    "progress_summary": "Facing the hallway entrance; moved forward 0.5m toward doorway; no obstacles bypassed yet"
}}

**Ex2 - Obstacle detected**
{{
    "reasoning": "CENTER-NEAR is blocked, RIGHT-NEAR is open, and RIGHT still aligns better than LEFT with the waypoint route toward the sofa area. Turn right.",
    "action_analysis": "Front is blocked, and right is the safest direction that still stays aligned with the target route",
    "action": "TURN_RIGHT",
    "value": 30,
    "progress_summary": "Bypassed wall on left; now facing right corridor; moved ~1m total"
}}

**Ex3 - At destination**
{{
    "reasoning": "The current room is already the target room, and the destination object is already within about 0.5m. STOP immediately.",
    "action_analysis": "Destination area is already reached, so stopping now avoids overshooting",
    "action": "STOP",
    "value": 0,
    "progress_summary": "Entered living room from hallway; bypassed table on right; now facing sofa at ~0.3m"
}}

**Critical Rules**:
- **STOP immediately** only when the correct room/space is confirmed and the destination object is within ~1.0m
- reasoning must explicitly cover LEFT/CENTER/RIGHT plus NEAR/FAR image cues, likely room/space, and detected landmark/object evidence before choosing an action
- If the destination is ahead and FRONT is clear, prefer MOVE_FORWARD
- If FRONT is blocked, choose the closest safe side direction toward the destination, not a wider detour
- For off-screen landmarks, **always turn toward the indicated direction first**
- Use waypoint history and current room/object evidence to keep moving toward the most likely relevant space/object
- progress_summary must describe orientation, locations entered/passed, obstacles bypassed
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
    known_landmark_section = ""
    if landmark_map_info:
        known_landmark_section = (
            "# Known Landmark Map (from semantic map, sorted by distance)\n"
            f"{landmark_map_info}\n"
        )

    return ACTION_EXECUTION_PROMPT.format(
        subtask_destination=next_waypoint_destination,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary,
        waypoint_summary=waypoint_summary,
        previous_action_reason=previous_action_reason or "N/A (first step)",
        detected_landmarks=detected_landmarks or "none",
        known_landmark_section=known_landmark_section,
        move_distance=move_distance,
        turn_angle=turn_angle,
    )
