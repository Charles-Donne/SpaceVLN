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

# Visual Observations

You are provided with 3 images:

**IMAGE 1: First-person RGB View** - Current facing direction view
**IMAGE 2: Object Detection View** - Detected objects with bounding boxes (landmark: {detected_landmarks})
**IMAGE 3: Local Semantic Map** - Nearby region top-down view

# Local Map

**Map Orientation**: 
- Top of map = Agent's Front direction
- Map rotates with agent - front is always up
- Agent is at center

**Color Legend**:
- **White**: Unexplored/unknown areas
- **Black**: Obstacles (walls, furniture) - AVOID
- **Green**: Floor areas (safe to navigate) - OK TO MOVE
- **Orange line**: Recent trajectory 
- **Red arrow at center**: Agent position and facing direction (arrow = Front)
- **Blue semi-circle**: Current field of view
  - Opening direction = Front view
  - Objects within blue region are visible in IMAGE 1

# Your Task

Analyze the 3 images to decide the next action for collision avoidance and navigation.

**Decision Process**:
1. **RGB View**: What do you see? Where is the destination?
2. **Detection View**: Are there relevant landmarks detected?
3. **Local Map**: 
   - Check immediate path ahead (black = obstacle)
   - Verify direction to destination
   - Plan collision-free path
4. **Distance Estimation**: How far to destination? (e.g., "~3m", "<0.5m")
5. **Action Decision**: Choose safest action toward destination

**STOP Conditions** (ALL required):
- Moved ≥2 times
- Destination within 0.5m

**Safety Priority**: Avoid obstacles shown as black regions on local map

# Available Actions
- MOVE_FORWARD: Move {move_distance}m forward
- TURN_LEFT: Rotate {turn_angle}° counterclockwise
- TURN_RIGHT: Rotate {turn_angle}° clockwise
- STOP: Declare arrival at subtask destination

# Output Format (JSON only)

{{
    "reasoning": "Logic: (1) Destination location and distance (2) Movement count (3) Action decision",
    "action_analysis": "One-sentence analysis of why this action was chosen",
    "action": "MOVE_FORWARD" | "TURN_LEFT" | "TURN_RIGHT" | "STOP"
}}

# Examples

**Ex1 - Clear path ahead**
{{
    "reasoning": "Local map shows safe green floor ahead. Destination visible. Move forward.",
    "action_analysis": "Clear path ahead on local map, destination visible in detection view",
    "action": "MOVE_FORWARD"
}}

**Ex2 - Obstacle detected**
{{
    "reasoning": "Local map shows black obstacle directly ahead. Must turn to find clear path.",
    "action_analysis": "Obstacle blocking forward path, turning right to find clear route",
    "action": "TURN_RIGHT"
}}

**Ex3 - Approaching destination**
{{
    "reasoning": "Movement: 3, Distance: ~1m. Local map clear. Continue approach.",
    "action_analysis": "Distance ~1m, path clear, continue forward approach",
    "action": "MOVE_FORWARD"
}}

**Ex4 - At destination**
{{
    "reasoning": "Movement: 4 (✓≥2), Distance: <0.5m (✓), Fills view (✓). ALL MET.",
    "action_analysis": "All STOP criteria met: moved >=2 times, distance <0.5m",
    "action": "STOP"
}}

**Critical Rules**:
- Move ≥2 times before STOP
- STOP only when distance ≤0.5m
- When uncertain, MOVE_FORWARD
"""


def get_action_execution_prompt(next_waypoint_destination: str,
                                subtask_instruction: str,
                                progress_summary: str = "",
                                detected_landmarks: str = None,
                                previous_action_reason: str = "",
                                distance_front: str = "Unknown",
                                distance_left_30: str = "Unknown",
                                distance_right_30: str = "Unknown",
                                distance_left_60: str = "Unknown",
                                distance_right_60: str = "Unknown",
                                distance_left_90: str = "Unknown",
                                distance_right_90: str = "Unknown",
                                move_distance: float = 0.25,
                                turn_angle: int = 30) -> str:
    """获取动作执行提示词（精简版）"""
    if not progress_summary:
        progress_summary = "Just started"
        
    return ACTION_EXECUTION_PROMPT.format(
        subtask_destination=next_waypoint_destination,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary,
        detected_landmarks=detected_landmarks or "none",
        move_distance=move_distance,
        turn_angle=turn_angle,
        distance_front=distance_front,
        distance_left_30=distance_left_30,
        distance_left_60=distance_left_60,
        distance_right_30=distance_right_30,
        distance_right_60=distance_right_60,
        distance_left_90=distance_left_90,
        distance_right_90=distance_right_90
    )
