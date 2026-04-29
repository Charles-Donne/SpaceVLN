**NavGBench Complex-Instruction Policy**:
- Applies to both initial planning and replanning for complex/grounded NavGBench tasks.
- You must compress the original `Global Task` into coarse ordered stages before filling any route fields. Use at most 5 coarse stages total, including current and goal.
- `task_progress` and `waypoint_chain` must each contain at most 5 stages total. Do not copy long object-by-object `past/near/then` chains from the original instruction.
- Merge consecutive "past/near/then" landmarks that are in the same observed space, local area, or continuous route segment.
- `task_progress` and `waypoint_chain` must use the compressed stage route, not the full object-by-object instruction text. Preserve task order, but avoid one waypoint per mentioned object.
- Compressed stages are progress checkpoints, not mandatory stops. If current views, map, and space structure already prove a coarse stage has been reached or passed, mark that stage complete and start the next unfinished coarse stage immediately.
- Use compact anchors such as `dining room's high chair/wine area` or `hallway's doorway/stool side`. Choose the current coarse stage from observations/map first, then align it to the compressed route.
- Keep `next_waypoint` and `subtask_instruction` to the nearest unfinished coarse stage only; keep the instruction short and executable.
