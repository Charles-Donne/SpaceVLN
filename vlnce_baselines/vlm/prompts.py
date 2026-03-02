"""
VLM规划提示词模板
================
用于LLM高层规划的提示词模板
"""

# 初始规划提示词 - 在任务开始时生成第一个子任务
INITIAL_PLANNING_PROMPT = """VLN Planning: Analyze environment + Global Task → design next subtask.

**CRITICAL: Be CONCISE. Limit reasoning to 300-400 words total.**

**Role**: NAVIGATION ONLY (TURN/MOVE/STOP). NOT manipulation (doors/objects). End at final waypoint.
Example: "Stop at chair and open doors" → Navigation ends at "chair".

**Task**: {instruction}

# Inputs
**12 Views** (30° FOV, 360°): IMAGE 1=Front 0°, angles increase CCW (30°, 60°, ..., 330°)
- **Obstacle distances** on each: <0.5m=blocked, >0.5m=safe
- **Auto-rotation**: System rotates to your chosen IMAGE → becomes Front (0°)

**2 Maps**: Global (full area) + Local (nearby, agent-centered)

# Map Legend
**Colors**: White=unexplored | Black=obstacles (AVOID) | Green=safe floor | Orange=trajectory (avoid revisit) | Red arrow=you (points Front) | Red dash=Forward direction

**Global**: Full area, shows history
**Local**: Zoomed, Dark green circle=0.5m radius, Blue=90° FOV

**Use**: Global→layout, Local→nearby obstacles

# Your Task

1. **Analyze 12 views + maps**: Identify current position (NEAR objects <1.0m, Local Map green circle) and next waypoint (FAR objects >1.5m)
2. **Select safe direction**: Choose IMAGE with waypoint centered, obstacle distance >0.5m, Global Map shows green path
3. **Plan instruction**: System auto-rotates to your direction, write instruction from Front view after rotation

# Reasoning (6 Parts)

**1) 12-View Quick Scan (CONCISE - 50 words max)** 
**Summarize by regions**: Front(1-2): [key objects+dist] | Left(3-6): [key objects+dist] | Back(7-8): [key objects+dist] | Right(9-12): [key objects+dist]

**Conclusion (1-2 sentences)**:
- Current: [room + NEAR<1m objects]
- Safe directions: [which IMAGEs>0.5m]
- Next target: [destination IMAGE# dist]

**2) Map Quick Check (20 words max)**
Local 0.5m: [what's inside?] Global: [position + safe directions]

**3) Position & Next Step (80 words max)**
Current: [room from NEAR<1m]. Chain: [✓]→[Current]→[unmarked]. Next: [which waypoint? why?]. Arrived?: FAR>1.5m=no | SURROUNDED<1m=yes

**4) Direction (20 words)**: [Destination at IMAGE#? Why? Obstacles? Safe?]
**5) Near-term (30 words)**: [After rotate→subtask details]
**6) Long-term (20 words)**: [Remaining→goal]

# Actions
TURN_LEFT/RIGHT (30-180°) | MOVE_FORWARD (0.25-1.5m) | STOP (<0.5m)

# Output (JSON only)

{{
    "current_waypoint": "<Room - Objects>",
    "waypoint_sequence": "<Current→Next→...→Goal. Mark (✓) passed only>",
    "task_progress": "<Completed✓ current(Current) future unmarked>",
    "next_waypoint_direction": "<IMAGE 1-12>",
    "next_waypoint_destination": "<Next waypoint>",
    "subtask_instruction": "<[room]+[relation]+[object] after auto-rotate>",
    "next_waypoint_landmark": "<landmark>",
    "completion_criteria": "<Detection: NEAR | Map: area | Position: region>",
    "global_task_finish": <true if ALL✓, Else false>,
    "reasoning": "<6 parts TOTAL 300-400 words: 1)Views 50w 2)Map 20w 3)Position+Chain 80w 4)Direction 20w 5)Near 30w 6)Long 20w>"
}}

#Examples (abbreviated):

## Ex1: Exercise room task
**Task**: Turn around walk through exercise room into living room. Wait by Table.
**Obs:** IMAGE 1: Bookshelf. IMAGE 5: Exercise room doorway, gym equipment. IMAGE 10: Toilet, washbasin

{{
    "current_waypoint": "Restroom - toilet, washbasin nearby",
    "waypoint_sequence": "Restroom(Current)→Exercise Room→Living Room→Table(Goal)",
    "task_progress": "Turn around walk through exercise room(Current) into living room. Wait by Table.",
    "next_waypoint_direction": "IMAGE 5 (Left 120°)",
    "next_waypoint_destination": "exercise room",
    "subtask_instruction": "Move forward through doorway to enter exercise room.",
    "next_waypoint_landmark": "exercise equipment",
    "completion_criteria": "Detection: Exercise equipment NEAR | Map: Entered exercise room | Position: Exercise room",
    "global_task_finish": false,
    "reasoning": "1) 12-Views: IMAGE1(Front 0°): bookshelf 2.0m FAR. IMAGE2-4: walls<0.5m (blocked). IMAGE5(Left 120°): exercise doorway 1.5m, equipment visible. IMAGE7: restroom walls. IMAGE9-12: toilet/sink<1m NEAR (current). Obs: walls<0.5m blocked, doorway 1.5m safe. Conclusion: At restroom (fixtures<1m). Available: IMAGE5 doorway. Next: exercise room IMAGE5. 2) Maps: Local-0.5m circle has fixtures. Global-in restroom corner, front doorway green. 3) Position: NEAR=fixtures<1m→Restroom. Chain: Restroom(Current)→Exercise→Living→Table. Progress: exercise(Current). 4) Direction: Need exercise. IMAGE5 centered 1.5m safe. Eliminate: IMAGE2-4/6-8<0.5m, IMAGE9-12(current). Choose IMAGE5. 5) Near: Rotate IMAGE5→move through doorway. 6) Long: Exercise→living room→table→stop."
}}

## Ex2: Corridor task
**Task**: Exit room, turn left, head to kitchen, turn right. Through kitchen, out. Wait at bathroom.
**Obs:** IMAGE 1: Open space. IMAGE 2: Bedroom exit, corridor, pictures. IMAGE 4: Wall

{{
    "current_waypoint": "Bedroom - exit doorway",
    "waypoint_sequence": "Bedroom(Current)→Corridor→Kitchen→Bathroom(Goal)",
    "task_progress": "Exit room(Current), turn left, to kitchen, right. Through kitchen, out. Wait at bathroom.",
    "next_waypoint_direction": "IMAGE 2 (Left 30°)",
    "next_waypoint_destination": "corridor with pictures",
    "subtask_instruction": "Move forward through bedroom exit to corridor.",
    "next_waypoint_landmark": "picture",
    "completion_criteria": "Detection: Pictures NEAR | Map: Bedroom→corridor | Position: Corridor",
    "global_task_finish": false,
    "reasoning": "1) 12-Views: IMAGE1: bedroom space 1.0m. IMAGE2: corridor doorway 1.2m, pictures beyond. IMAGE3-4: walls<0.5m. IMAGE5-8: furniture<0.8m. IMAGE9-12: bed<1m NEAR. Obs: walls<0.5m blocked, corridor 1.2m safe. Conclusion: At bedroom (bed<1m). Available: IMAGE2 safe. Next: corridor IMAGE2. 2) Maps: Local-0.5m has furniture. Global-bedroom near exit, front doorway green. 3) Position: NEAR=bed<1m→Bedroom. Chain: Bedroom(Current)→Corridor→Kitchen→Bathroom. Progress: Exit(Current). 4) Direction: Need corridor. IMAGE2 doorway 1.2m, pictures landmark. Eliminate: IMAGE3-4<0.5m. Choose IMAGE2. 5) Near: Rotate IMAGE2→move to corridor. 6) Long: Corridor→kitchen→bathroom."
}}

**Critical Rules**:
- **Reasoning thoroughness**: Part 1 MUST analyze ALL 12 IMAGEs with angle+direction+content. Part 3 MUST detail position and complete task chain (✓→Current→unmarked)
- **Initial position**: Determine from NEAR<1m + Local Map 0.5m circle. Mark current=(Current), future=unmarked
- **Position awareness**: NEAR<1m (multiple IMAGEs) = current position ≠ destination FAR>1.5m (1-2 views)
- **Markers**: Behind=(✓) | Now=(Current) ONE only | Ahead=unmarked
- **Task chain consistency**: Before current=(✓), current=(Current), after current=unmarked. Chain determines next direction
- **Entrance vs interior**: "Wait at entrance" = doorway, NOT inside room
- **Room-first strategy**: "[room]'s [object]" → navigate to [room] first, then [object]
- **Auto-rotation**: System rotates to your IMAGE → write from Front view after
- **Detail instructions**: Use [room]+[relation]+[object]. "Living room's gray couch" NOT "couch"
- **IMAGE-angle match**: IMAGE1=0°, IMAGE2=30°, ..., IMAGE7=180°, ..., IMAGE12=330°
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """VLN Verification: Verify subtask completion + plan next.

**CRITICAL: Be CONCISE. Limit reasoning to 300-400 words total.**

**Role**: NAVIGATION (TURN/MOVE/STOP). NOT manipulation. End at final waypoint.

**Task**: {instruction}

**Previous Subtask**:
- Destination: {subtask_destination}
- Instruction: {subtask_instruction}
- Criteria: {completion_criteria}

# Inputs
**12 Views** (30° FOV): IMAGE1=Front 0°, angles increase CCW
- **Obstacle distances**: <0.5m=blocked, >1m=safe
- **Waypoint markers**: White circles(ID) + boxes(room) = visited locations
- **Auto-rotation**: System rotates to your IMAGE

**2 Maps**: Global (full + history) + Local (nearby + 0.5m circle)
**Colors**: White=unexplored | Black=obstacles | Green=safe | Orange=trajectory | Red=you | Blue circles=waypoints

**Waypoint History**: {waypoint_summary}

# Reasoning (6 Parts - CONCISE)

**1) 12-View Quick Scan (50 words max)** 
**Summarize by regions**: Front(1-2): [objects+dist] | Left(3-6): [objects+dist] | Back(7-8): [Blue circles? objects+dist] | Right(9-12): [objects+dist]

**Conclusion**:
- Current: [room + NEAR<1m]
- Safe: [which IMAGEs>0.5m]
- Visited: [Blue circles at which IMAGEs]
- Next: [target IMAGE# dist]

**2) Map Quick Check (30 words max)**
Local 0.5m: [what's inside?] Global: [Blue circles where? Trajectory path? Position? Safe directions?]

**3) Position & Next Step (80 words max)**
Current: [room from NEAR<1m]. Chain: [Blue behind=✓]→[Current]→[unmarked]. Next: [which waypoint? why?]. Arrived?: FAR>1.5m=no | SURROUNDED<1m=yes

**4) Direction (20 words)**: [Target at IMAGE#? Why? Blue circles=AVOID. Safe?]
**5) Near-term (30 words)**: [After rotate→subtask]
**6) Long-term (20 words)**: [Remaining→goal]

# Actions
TURN_LEFT/RIGHT (30-180°) | MOVE_FORWARD (0.25-1.5m) | STOP (<0.5m)

# Output (JSON only)

{{
    "current_waypoint": "<Room - Objects>",
    "waypoint_sequence": "<✓→Current→Next→Goal>",
    "task_progress": "<✓ (Current) unmarked>",
    "next_waypoint_direction": "<IMAGE#>",
    "next_waypoint_destination": "<waypoint>",
    "subtask_instruction": "<[room]+[relation]+[object]>",
    "next_waypoint_landmark": "<landmark>",
    "completion_criteria": "<Detection/Map/Position>",
    "global_task_finish": <true/false>,
    "reasoning": "<6 parts MAX 300w: 1)Views 50w 2)Map 30w 3)Chain 80w 4)Dir 20w 5)Near 30w 6)Long 20w>"
}}

# Examples (abbreviated):

## Ex1: Exercise room
**Task**: Turn around walk through exercise room into living room. Wait by Table.
**Previous**: Navigate to exercise room
**Obs:** IMAGE 1: Exercise equipment. IMAGE 7: Restroom behind

{{
    "current_waypoint": "Exercise Room - gym equipment",
    "waypoint_sequence": "Restroom(✓)→Exercise Room(Current)→Living Room→Table(Goal)",
    "task_progress": "Turn around walk through exercise room(Current) into living room. Wait by Table.",
    "next_waypoint_direction": "IMAGE 1 (Front 0°)",
    "next_waypoint_destination": "living room",
    "subtask_instruction": "Move forward through exercise room to living room exit",
    "next_waypoint_landmark": "arched doorway",
    "completion_criteria": "Detection: Doorway NEAR | Map: Through exercise→living room | Position: Living room entrance",
    "global_task_finish": false,
    "reasoning": "1)Views: Front:bookshelf FAR. Left120:exercise doorway. Back:restroom NEAR. Conclusion:At restroom. Safe:IMAGE5. 2)Map:Local has fixtures. Global:exit ahead. 3)Chain:Restroom(Current)→Exercise→Living→Table. Need:exercise. 4)IMAGE5 doorway safe. 5)Rotate→move doorway. 6)Living→table."
}}

## Ex2: Corridor task
**Task**: Exit bedroom, turn left. Walk straight passing gray couch, stop at rug.
**Obs:** IMAGE 1: Rug <0.5m. IMAGE 10: Gray couch beside.

{{
    "current_waypoint": "Bedroom - exit doorway",
    "waypoint_sequence": "Bedroom(Current)→Corridor→Kitchen→Bathroom(Goal)",
    "task_progress": "Exit room(Current), turn left, to kitchen, right. Through kitchen, out. Wait at bathroom.",
    "next_waypoint_direction": "IMAGE 2 (Left 30°)",
    "next_waypoint_destination": "corridor with pictures",
    "subtask_instruction": "Move forward through bedroom exit to corridor.",
    "next_waypoint_landmark": "picture",
    "completion_criteria": "Detection: Pictures NEAR | Map: Bedroom→corridor | Position: Corridor",
    "global_task_finish": false,
    "reasoning": "1)Views: Front:bedroom. Left30:doorway 1.2m safe. Right:bed NEAR. Conclusion:At bedroom. 2)Map:Local has furniture. Global:exit ahead. 3)Chain:Bedroom(Current)→Corridor→Kitchen. Need:corridor. 4)IMAGE2 doorway. 5)Rotate→exit. 6)Kitchen→bathroom."
}}

**Rules**:
- **CONCISE**: 300-400 words TOTAL reasoning
- **Position**: NEAR<1m=current. FAR>1.5m=target
- **Chain**: [✓]→[Current]→[unmarked]
- **No IMAGE7**: Never turn 180° back
- **Auto-rotate**: Write instruction after rotate


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
        action_space=action_space
    )

def get_verification_replanning_prompt(instruction: str,
                                       subtask_destination: str,
                                       subtask_instruction: str,
                                       completion_criteria: str,
                                       action_space: str,
                                       detected_landmarks: str = None,
                                       waypoint_summary: str = None) -> str:
    """
    获取验证和重规划提示词
    
    Args:
        instruction: 完整导航指令
        subtask_destination: 当前子任务目的地
        subtask_instruction: 当前子任务指令
        completion_criteria: 完成条件
        action_space: 动作空间描述
        detected_landmarks: 已检测到的landmark类别字符串
        waypoint_summary: 路径点历史记录字符串
        
    Returns:
        格式化的提示词字符串
    """
    if not detected_landmarks:
        detected_landmarks = "No landmarks detected yet"
    if not waypoint_summary:
        waypoint_summary = "No waypoints recorded yet"
    
    return VERIFICATION_REPLANNING_PROMPT.format(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        completion_criteria=completion_criteria,
        action_space=action_space,
        detected_landmarks=detected_landmarks,
        waypoint_summary=waypoint_summary
    )
