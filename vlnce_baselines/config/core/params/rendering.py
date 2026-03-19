"""Rendering and visualization style parameters."""

# PIL palette in normalized RGB [0, 1].
COLOR_PALETTE = [
    1.00, 1.00, 1.00,
    0.00, 0.00, 0.00,
    0.83, 0.83, 0.83,
    1.00, 0.65, 0.00,
    0.12, 0.47, 0.71,
    0.77, 0.88, 0.65,
]

# Integer palette used by some legacy visualization code.
LEGEND_COLOR_PALETTE = [
    255, 255, 255,
    0, 0, 0,
    211, 211, 211,
    255, 165, 0,
    31, 119, 180,
    196, 225, 165,
]

DETECTION_COLORS = {"landmark": (0, 255, 255)}
DETECTION_THICKNESS = {"landmark": 3}

LANDMARK_MARKER_COLOR = (128, 0, 128)
LANDMARK_MARKER_BORDER = (255, 255, 255)
LANDMARK_MARKER_RADIUS = 6
