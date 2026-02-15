"""
动作执行提示词模板
==================
用于VLM低层动作决策的提示词模板

动作参数与interactive_navigation保持一致：
- TURN_LEFT/RIGHT: 30°（12步×30°=360°）
- MOVE_FORWARD: 0.25m
"""

ACTION_EXECUTION_PROMPT = """You are executing navigation to reach {next_waypoint_destination}. Analyze view + map to decide: arrived OR move toward destination OR adjust pose to avoid obstacles.

# Current Sub-Task
**Destination**: {next_waypoint_destination}
**Sub-Instruction**: {subtask_instruction}
**Previous Progress**: {progress_summary}
**Last Action Reason**: {previous_action_reason}

# Visual Inputs

**IMAGE 1 - RGB**: First-person view | Destination visible where?
**IMAGE 2 - Detection + Distance**: FRONT {distance_front} | L/R 30° {distance_left_30}/{distance_right_30} | L/R 60° {distance_left_60}/{distance_right_60} | L/R 90° {distance_left_90}/{distance_right_90}
- Distance: <0.5m blocked | >1.0m safe | >1.5m clear
**IMAGE 3 - Local Map**: Red arrow=position | Green circle=0.5m arrival | Black=obstacles | Green=safe | Orange=trajectory

# Decision Process (3 Steps)

**Step 1: Find Destination in View**
- **Where**: Destination visible? Front/Left/Right? Which angle? (Front center/Left 30°/Right 60°/...)
- **Distance**: How far? NEAR(<1m filling view)/MEDIUM(1-2m)/FAR(>2m small)/NOT VISIBLE
- **Example**: "Kitchen table: Front center, 0.4m NEAR" OR "Bedroom door: Left 30°, 3.0m FAR" OR "Target: NOT VISIBLE"

**Step 2: Decide Action Based on Position & Distance**
- **NEAR (<1m, filling view)** → STOP (arrived)
- **NOT VISIBLE** → STOP (lost/confused)
- **Left side (30-150°)** → TURN_LEFT (align toward destination)
- **Right side (210-330°)** → TURN_RIGHT (align toward destination)
- **Front (330-30°, center) + clear path** → MOVE_FORWARD (advance)
- **Front + blocked** → TURN to clearer side

**Step 3: Verify Safety**
- Check obstacle distances in chosen direction
- Adjust action if blocked (<0.5m)

# Available Actions

- **TURN_LEFT/RIGHT**: degrees = 30, 60, 90, 120, 150, 180
- **MOVE_FORWARD**: meters = 0.25, 0.5, 0.75, 1.0 (max 1.0m)
- **STOP**: (when < 0.5m at destination)

# Output (JSON only)

{{
    "reasoning": "<Step 1: Find Destination - Where is [destination]? (Front center/Left 30°/Right 60°/...) Distance? (NEAR<1m/MEDIUM 1-2m/FAR>2m/NOT VISIBLE). Step 2: Decide Action - NEAR→STOP | NOT VISIBLE→STOP | Left side→TURN_LEFT | Right side→TURN_RIGHT | Front+clear→MOVE_FORWARD | Front+blocked→TURN. Step 3: Safety - Check obstacle distances in chosen direction>",
    "action_analysis": "<1-2 sentences: Destination location + distance → action>",
    "action": "STOP" | "MOVE_FORWARD" | "TURN_LEFT" | "TURN_RIGHT",
    "degrees": <30-180> (TURN only),
    "meters": <0.25-1.0> (MOVE only)
}}

# Examples

## Ex1 - Arrived (NEAR, STOP):
{{
    "reasoning": "Step 1: Kitchen table - Front center, 0.3m NEAR (filling entire view, very close). Step 2: NEAR (<1m) → STOP (arrived). Step 3: N/A.",
    "action_analysis": "Kitchen table 0.3m NEAR filling view → arrived, STOP.",
    "action": "STOP"
}}

## Ex2 - Turn toward left opening:
{{
    "reasoning": "Step 1: Living room opening - Left 30°, 2.0m MEDIUM (doorway visible left side). Step 2: Left side (30°) → TURN_LEFT to align. Step 3: L30° >2.0m clear, safe to turn.",
    "action_analysis": "Living room opening at Left 30°, 2.0m → turn left 30° to align.",
    "action": "TURN_LEFT",
    "degrees": 30
}}

## Ex3 - Move forward to distant target:
{{
    "reasoning": "Step 1: Kitchen table - Front center, 3.5m FAR (small, visible ahead). Step 2: Front + FAR → MOVE_FORWARD to approach. Step 3: FRONT 1.5m clear, safe to advance 0.75m.",
    "action_analysis": "Kitchen table 3.5m FAR ahead, path clear → move forward 0.75m.",
    "action": "MOVE_FORWARD",
    "meters": 0.75
}}

## Ex4 - Turn right toward target:
{{
    "reasoning": "Step 1: Bedroom doorway - Right 30°, 1.8m MEDIUM (visible right side). Step 2: Right side (30°) → TURN_RIGHT to align. Step 3: R30° 1.8m clear, safe to turn.",
    "action_analysis": "Bedroom doorway at Right 30°, 1.8m → turn right 30° to align.",
    "action": "TURN_RIGHT",
    "degrees": 30
}}

## Ex5 - Target not visible (STOP):
{{
    "reasoning": "Step 1: Exercise room - NOT VISIBLE (cannot find in any direction, lost). Step 2: NOT VISIBLE → STOP (confused/lost). Step 3: N/A.",
    "action_analysis": "Exercise room NOT VISIBLE in view → lost, STOP for replan.",
    "action": "STOP"
}}

## Ex6 - Front blocked, turn to bypass:
{{
    "reasoning": "Step 1: Hallway ahead - Front center, 2.0m MEDIUM (visible but furniture blocking 0.4m). Step 2: Front but FRONT 0.4m blocked → TURN to clearer side. Right 30° 1.8m clear toward destination → TURN_RIGHT. Step 3: R30° 1.8m clear, safe.",
    "action_analysis": "Hallway ahead but FRONT blocked 0.4m → Right 30° clear 1.8m, turn right.",
    "action": "TURN_RIGHT",
    "degrees": 30
}}

**Critical Rules**:
1. **3-Step Logic**: 1) Find destination (where? distance?) → 2) Decide action (NEAR→STOP, Left→TURN_LEFT, Right→TURN_RIGHT, Front+clear→MOVE, NOT VISIBLE→STOP) → 3) Verify safety
2. **Distance Classification**: NEAR<1m (filling view, STOP) | MEDIUM 1-2m | FAR>2m (small) | NOT VISIBLE (STOP)
3. **Direction Mapping**: Left side (30-150°)→TURN_LEFT | Right side (210-330°)→TURN_RIGHT | Front center (330-30°)→MOVE_FORWARD
4. **NO HALLUCINATION**: Only describe what's actually visible in images
5. **Priority**: Destination location > Obstacle avoidance. Always move toward destination unless blocked"""


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
                                distance_right_90: str = "Unknown") -> str:
    """
    获取动作执行提示词
    
    Args:
        next_waypoint_destination: 下一个waypoint目的地
        subtask_instruction: 子任务指令
        progress_summary: 当前子任务进度摘要
        detected_landmarks: 已检测到的landmark类别字符串（可选，不强制要求）
        previous_action_reason: 上一步动作的action_analysis
        distance_front: 前方(0°)障碍物距离
        distance_left_30: 左前方(30°)障碍物距离
        distance_right_30: 右前方(30°)障碍物距离
        distance_left_60: 左前方(60°)障碍物距离
        distance_right_60: 右前方(60°)障碍物距离
        distance_left_90: 左侧(90°)障碍物距离
        distance_right_90: 右侧(90°)障碍物距离
        
    Returns:
        格式化的提示词字符串
    """
    if not previous_action_reason:
        previous_action_reason = "None"
    
    # 如果progress_summary为空，说明是刚开始
    if not progress_summary:
        progress_summary = "Just started"
        
    return ACTION_EXECUTION_PROMPT.format(
        next_waypoint_destination=next_waypoint_destination,
        subtask_instruction=subtask_instruction,
        progress_summary=progress_summary,
        previous_action_reason=previous_action_reason,
        distance_front=distance_front,
        distance_left_30=distance_left_30,
        distance_right_30=distance_right_30,
        distance_left_60=distance_left_60,
        distance_right_60=distance_right_60,
        distance_left_90=distance_left_90,
        distance_right_90=distance_right_90
    )
