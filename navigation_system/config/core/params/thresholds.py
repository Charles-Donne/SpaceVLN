"""Shared threshold families for navigation, prompting, mapping, and evaluation."""

# Obstacle / clearance thresholds shared by prompts, action rendering, and distance formatting.
OBS_BLOCKED_M = 0.5
OBS_RISKY_M = 1.0
OBS_OPEN_M = 2.0

# High-level reasoning threshold for "near / arrived" judgments in thinking + replanning.
ARRIVAL_NEAR_M = 1.0

# Low-level action-side auto-complete thresholds.
AUTOCOMPLETE_OPENING_M = 0.5
AUTOCOMPLETE_SOLID_M = 0.75
AUTOCOMPLETE_TOPK = 2

# Low-level stagnation detection: min(move_distance * ratio, cap).
LOW_LEVEL_STAGNATION_RATIO = 0.2
LOW_LEVEL_STAGNATION_CAP_M = 0.25

# Multi-floor topology thresholds.
FLOOR_SAME_Z_M = 1.0
FLOOR_SWITCH_Z_M = 1.5
FLOOR_SWITCH_STABLE_STEPS = 3

# Evaluation success distance used by task config sync + reports.
EVAL_SUCCESS_DISTANCE_M = 3.0

# Semantic-map fusion thresholds.
SEM_MAP_CAT_THRESH = 5.0
SEM_MAP_EXP_THRESH = 1.0
SEM_MAP_OBS_THRESH = 1.0
SEM_MAP_LANDMARK_THRESH = 0.5
