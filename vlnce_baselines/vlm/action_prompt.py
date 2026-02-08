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

# Decision Process

**Step 0: Observe** - RGB/Detection/Map show what? NO HALLUCINATION
**Step 1: Position** - NEAR objects (<1.0m) + map → "I am in [ROOM] near [object]". Compare with destination.
**Step 2: Arrival Check**
- Case A: Destination FILLING VIEW → STOP
- Case B: Destination SMALL/DISTANT → continue
- Case C: Cannot find OR confused → STOP
**Step 3: Navigate** (if Case B) - Analyze 7 directions. FRONT ≥1.0m clear + ahead → MOVE. FRONT <0.5m → TURN clearer side.

# Available Actions

- **TURN_LEFT/RIGHT**: degrees = 30, 60, 90, 120, 150, 180
- **MOVE_FORWARD**: meters = 0.25, 0.5, 0.75, 1.0 (max 1.0m)
- **STOP**: (when < 0.5m at destination)

# Output (JSON only)

{{
    "reasoning": "<0) Observe: RGB/Detection/Map show what? NO HALLUCINATION. 1) Position: NEAR objects → I am in [ROOM] near [object]. Destination: [object]. 2) Arrival: Case A/B/C? 3) Navigate (if B): 7 directions analysis>",
    "action_analysis": "<1-2 sentences>",
    "action": "STOP" | "MOVE_FORWARD" | "TURN_LEFT" | "TURN_RIGHT",
    "degrees": <30-180> (TURN only),
    "meters": <0.25-1.0> (MOVE only)
}}

# Examples

## Ex1 - Arrived:
{{
    "reasoning": "0) RGB: kitchen's table VERY CLOSE filling view, counter 0.9m, cabinets 1.0m. Detection: FRONT 0.3m. Map: inside green circle. 1) Position: NEAR counter 0.9m, cabinets 1.0m → in KITCHEN near counter. Destination: kitchen's table 0.3m ahead. 2) Arrival: Table FILLING VIEW → Case A. STOP. 3) N/A.",
    "action_analysis": "Arrived at kitchen's table (0.3m, filling view). Stop.",
    "action": "STOP"
}}

## Ex2 - Turn toward opening:
{{
    "reasoning": "0) RGB: doorway 0.7m, hallway's wall 0.78m right, living room visible ahead-left. Detection: FRONT 0.70m, L30° >2.0m, R30° 0.78m. Map: at threshold. 1) Position: at DOORWAY THRESHOLD between hallway/living room. Destination: inside living room. 2) Arrival: Living room SMALL/DISTANT → Case B. 3) Navigate: FRONT 0.70m blocked. L30° >2.0m clear toward destination. Turn LEFT 30°.",
    "action_analysis": "L30° shows living room opening (>2.0m). Turn left.",
    "action": "TURN_LEFT",
    "degrees": 30
}}

## Ex3 - Move forward:
{{
    "reasoning": "0) RGB: kitchen's table FAR small, living room's sofa 1.1m, coffee table 1.3m. Detection: FRONT 1.5m clear. Map: in living room. 1) Position: in LIVING ROOM near sofa/coffee table. Destination: kitchen's table ahead. 2) Arrival: Table SMALL/DISTANT → Case B. 3) Navigate: FRONT 1.5m clear toward table. Move forward 0.75m.",
    "action_analysis": "Kitchen's table ahead FAR. FRONT clear 1.5m. Advance.",
    "action": "MOVE_FORWARD",
    "meters": 0.75
}}

## Ex4 - Bypass obstacle:
{{
    "reasoning": "0) RGB: hallway's furniture 0.4m blocking, bedroom's doorway FAR right. Detection: FRONT 0.4m blocked, R30° 1.8m clear, R60° 2.0m. 1) Position: in HALLWAY, blocked by furniture. Destination: bedroom's doorway ahead-right. 2) Arrival: Doorway SMALL/DISTANT → Case B. 3) Navigate: FRONT 0.4m BLOCKED. R30° 1.8m clear toward doorway. Turn RIGHT 30°.",
    "action_analysis": "Furniture blocks FRONT. R30° clear 1.8m. Turn right.",
    "action": "TURN_RIGHT",
    "degrees": 30
}}

## Ex5 - Realign:
{{
    "reasoning": "0) RGB: bedroom's doorway LEFT 1.5m, hallway's wall left, furniture 0.9m behind. Detection: FRONT 2.0m, L30° 1.5m. Map: past obstacle. 1) Position: in HALLWAY past furniture. Destination: bedroom's doorway left. 2) Arrival: Doorway SMALL/DISTANT left → Case B. 3) Navigate: L30° 1.5m toward doorway. Turn LEFT 30°.",
    "action_analysis": "Obstacle bypassed. Doorway at left. Realign.",
    "action": "TURN_LEFT",
    "degrees": 30
}}

## Ex6 - Entered room:
{{
    "reasoning": "0) RGB: exercise room's treadmill 0.8m NEAR, weights 0.9m, mat 1.0m SURROUNDING. Map: in expanded green area. 1) Position: INSIDE EXERCISE ROOM surrounded by equipment. Destination: exercise room. 2) Arrival: Equipment SURROUNDING → Case A. STOP. 3) N/A.",
    "action_analysis": "Inside exercise room, equipment surrounding. Stop.",
    "action": "STOP"
}}

**Critical Rules**:
1. **Room Context**: [room]'s [object] distance (kitchen's chair ≠ living room's chair)
2. **4-Step**: 0) Observe → 1) Position (near what?) → 2) Arrival (A/B/C?) → 3) Navigate
3. **NO HALLUCINATION**: Only describe actual visible content
4. **Arrival**: A) FILLING VIEW → STOP | B) SMALL/DISTANT → continue | C) Cannot find/confused → STOP
5. **Distance**: <0.5m blocked | 1.0-2.0m safe | >2.0m clear"""


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
