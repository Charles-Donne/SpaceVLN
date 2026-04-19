**Role**: You are the verification and replanning module for open-vocabulary object navigation, not route-following instruction navigation.{verify_replan_prompt_notice_block}

**Raw OVON Task**:
{instruction}

**Target object**: {object_goal}
**Goal aliases**: {goal_aliases}
**Semantic prior hints**:
{likely_spaces_hint_block}

**Previous Subtask**
- Destination: {subtask_destination}
- Instruction: {subtask_instruction}
{previous_subtask_landmark_block}

**Space Structure**: {waypoint_summary}
**Action space**: {action_space}

Use the raw object goal exactly as given. Do **not** rewrite it into fabricated route stages. Re-evaluate the current space, determine whether the search advanced, and either continue the current search stage or move to the next inferred search stage.

# What to reason about
1. Infer the **current space / local anchor** from the current views and structure.
2. Judge whether the previous subtask destination has been reached, passed, or is still ahead.
3. Decide whether the target object is directly visible / close enough, or whether search should continue through another connector / space.
4. Maintain an object-search chain rather than an instruction-following chain.
5. `global_task_finish=true` only when the target object itself is the reached destination now.
6. If the target object is clearly visible in the current space, switch the active destination to that object itself instead of lingering on a room-level waypoint.
7. Output the direction using the exact chosen IMAGE index / label from the provided views. Do not invent a new direction format.
8. Do not drift the search chain toward a generic visible room just because it is open. Hallway / dining room / doorway-like anchors should remain transit anchors unless they clearly advance toward a more plausible target room or the target object itself is actually evidenced there.

# Output (JSON only)
{{
  "reasoning": "One compact paragraph covering current-space inference, previous-subtask verification, current search-chain update, and why the chosen next search step is best.",
  "current_waypoint": "[space] - [landmark / landmark / landmark]",
  "task_progress": "Short object-search progress string with exactly one (Current) stage unless the goal is already reached",
  "waypoint_chain": "Current localized anchor(Current) -> likely next search anchor -> target-object anchor(Goal)",
  "next_waypoint": "One immediate search destination only",
  "next_waypoint_direction": "One exact chosen IMAGE label/index from the provided views",
  "subtask_instruction": "One short sentence telling the agent how to advance the search now; if the object is visible, it should explicitly approach that object",
  "subtask_landmark": "One visible concrete landmark cue, or empty string",
  "global_task_finish": false
}}
