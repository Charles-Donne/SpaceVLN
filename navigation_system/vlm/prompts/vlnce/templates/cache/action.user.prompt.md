# Current Subtask
**Destination**: {subtask_destination}
**Instruction**: {subtask_instruction}
**Subtask Progress**: {progress_summary}

# Environment Perception
**Obstacle**: {obstacle_perception_summary}
**Landmark**:
{landmark_perception_summary}

- **Yellow bounding box**: candidate current-view landmark detection ({detected_landmarks}); judge whether it is valid task evidence or noise. If the label/box conflicts with the RGB scene, local geometry, obstacle layout, or task/space context, downweight or ignore it.

**Action space**:
{allowed_action_bullets}
