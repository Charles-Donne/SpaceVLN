"""Semantic category and world-map structure parameters."""

# Base map channels: obstacle, explored, agent/trajectory.
MAP_CHANNELS = 3

MAPPING_CLASSES = [
    "floor", "wall", "door",
    "bed", "sofa", "chair", "table",
    "cabinet", "bookshelf",
    "tv", "sink", "toilet", "plant",
    "painting", "window",
]

NUM_SEMANTIC_CATEGORIES = len(MAPPING_CLASSES)

NAVIGABLE_CLASSES = [
    "floor", "ground", "flooring",
    "walkway", "corridor", "hallway",
    "stair", "stairs", "staircase",
]

# Dynamic per-subtask landmark class list.
LANDMARK_CLASSES = []
