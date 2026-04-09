"""Pure direction-format helpers shared by prompts, controllers, and overlays."""


def normalize_relative_bearing(bearing_deg: float) -> float:
    """Normalize relative bearing into [-180, 180]."""
    return ((bearing_deg + 180.0) % 360.0) - 180.0


def snap_relative_bearing(bearing_deg: float) -> int:
    """Snap bearing to the Front/Back and 30-degree bins used in prompts."""
    bearing = normalize_relative_bearing(bearing_deg)
    magnitude = abs(bearing)
    if magnitude <= 15.0:
        return 0
    if magnitude >= 165.0:
        return 180 if bearing >= 0 else -180
    bucket = int((magnitude - 15.0 - 1e-6) // 30.0) + 1
    snapped = min(bucket * 30, 150)
    return snapped if bearing > 0 else -snapped


def format_relative_direction(bearing_deg: float) -> str:
    """Format bearing using the snapped prompt-side direction labels."""
    snapped = snap_relative_bearing(bearing_deg)
    magnitude = abs(snapped)
    if magnitude <= 15.0:
        return "Front 0deg"
    if magnitude >= 165.0:
        return "Back 180deg"
    side = "Right" if snapped > 0 else "Left"
    return f"{side} {magnitude:.0f}deg"


def build_landmark_turn_hint(bearing_deg: float, is_visible: bool = False) -> str:
    """Produce the short action-side hint used for visible/off-screen landmarks."""
    snapped = snap_relative_bearing(bearing_deg)
    if is_visible and snapped == 0:
        return ""
    if snapped == 0:
        return " -> move forward"
    if snapped > 0:
        return f" -> TURN RIGHT {abs(snapped)}deg then move forward"
    return f" -> TURN LEFT {abs(snapped)}deg then move forward"
