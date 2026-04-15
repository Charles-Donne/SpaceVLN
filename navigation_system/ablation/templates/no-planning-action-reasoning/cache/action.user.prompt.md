# Current Subtask
**Destination**: {subtask_destination}
**Instruction**: {subtask_instruction}
**Subtask Progress**: {progress_summary}
**Previous Step Analysis**: {previous_action_reason}
**Controller Notice**: {controller_action_notice}

`Previous Step Analysis` is only the last-step memory for avoiding repeated actions; it is not the current truth. It follows fixed memory modes only:
- `LAST_STEP_AVOID_OBSTACLE | turn=<LEFT/RIGHT> | obstacle=FRONT blocked`
- `LAST_STEP_ALIGN_DESTINATION | turn=<LEFT/RIGHT> | target=<destination/landmark> | target_distance=<...>`
- `LAST_STEP_FORWARD_TO_TARGET | move=<...> | target=<destination/landmark> | target_distance=<...>`
- `LAST_STEP_STOP_AT_TARGET | target=<destination/landmark> | target_distance=<...>`
- `N/A (first step)`
`Controller Notice` is a current-call hard constraint. If it is not `None`, obey it before using the last-step memory.

# Environment Perception
**Obstacle**: {obstacle_perception_summary}
**Landmark**:
{landmark_perception_summary}

- **Yellow bounding box**: candidate current-view landmark detection ({detected_landmarks}); first judge whether it is valid task evidence or noise. If the label/box conflicts with the RGB scene, local geometry, obstacle layout, or task/space context, downweight or ignore it

**Action space**:
{allowed_action_bullets}
