from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vlnce_baselines.utils.spatial_formatter import format_relative_direction


@dataclass(frozen=True)
class LandmarkDrawItem:
    bbox: Tuple[int, int, int, int]
    label_text: str
    distance_m: float


@dataclass(frozen=True)
class LandmarkStripSegment:
    text: str
    color: Tuple[int, int, int]


@dataclass(frozen=True)
class LandmarkStripLine:
    distance_m: float
    confidence: float
    priority: int
    sort_key: Tuple[float, float, float]
    segments: Tuple[LandmarkStripSegment, ...]


def draw_action_partition_lines(
    image: np.ndarray,
    hfov_deg: float,
    boundaries_deg: Sequence[float] = (-25.0, 25.0),
    color: Tuple[int, int, int] = (255, 220, 160),
    thickness: int = 2,
) -> None:
    h_img, w_img = image.shape[:2]
    half_fov = float(hfov_deg) / 2.0
    if half_fov <= 1e-6:
        return

    boundary_xs: List[int] = []
    for boundary_deg in boundaries_deg:
        ratio = (float(boundary_deg) + half_fov) / float(hfov_deg)
        x_pos = int(round(ratio * (w_img - 1)))
        x_pos = max(0, min(w_img - 1, x_pos))
        boundary_xs.append(x_pos)
        cv2.line(
            image,
            (x_pos, 0),
            (x_pos, h_img - 1),
            color,
            thickness,
            cv2.LINE_AA,
        )

    section_bounds = [0] + sorted(boundary_xs) + [w_img]
    section_labels = ["Left 30deg", "Front 0deg", "Right 30deg"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.56
    font_thickness = 1
    text_color = (255, 0, 0)
    bg_color = (255, 255, 255)

    for idx, label in enumerate(section_labels):
        left = section_bounds[idx]
        right = section_bounds[idx + 1]
        if right <= left:
            continue
        text_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_w, text_h = text_size
        center_x = (left + right) // 2
        text_x = max(left + 3, min(center_x - text_w // 2, right - text_w - 3))
        text_y = max(text_h + 7, 17)
        cv2.rectangle(
            image,
            (text_x - 2, text_y - text_h - 2),
            (text_x + text_w + 2, text_y + baseline + 1),
            bg_color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )


def draw_landmark_boxes(
    image: np.ndarray,
    draw_items: Sequence[LandmarkDrawItem],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    for item in draw_items:
        x1, y1, x2, y2 = item.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_landmark_labels(
    image: np.ndarray,
    draw_items: Sequence[LandmarkDrawItem],
    color: Tuple[int, int, int],
    font_scale: float = 0.62,
    font_thickness: int = 2,
    pad: int = 5,
    avoid_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> None:
    if not draw_items:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    image_h, image_w = image.shape[:2]
    reserved_boxes = [tuple(int(v) for v in box) for box in (avoid_boxes or [])]
    placed_boxes: List[Tuple[int, int, int, int]] = []

    def _intersects(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> bool:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

    def _is_clear(box: Tuple[int, int, int, int]) -> bool:
        for other in reserved_boxes:
            if _intersects(box, other):
                return False
        for other in placed_boxes:
            if _intersects(box, other):
                return False
        return True

    def _clamp_top_left(x: int, y: int, box_w: int, box_h: int) -> Tuple[int, int]:
        clamped_x = max(0, min(int(x), max(0, image_w - box_w)))
        clamped_y = max(0, min(int(y), max(0, image_h - box_h)))
        return clamped_x, clamped_y

    for item in sorted(draw_items, key=lambda entry: entry.distance_m, reverse=True):
        if not item.label_text:
            continue

        x1, y1, x2, _y2 = item.bbox
        (text_w, text_h), _ = cv2.getTextSize(item.label_text, font, font_scale, font_thickness)
        bg_w = text_w + pad * 2
        bg_h = text_h + pad * 2

        bbox_center_x = (x1 + x2) // 2
        x_candidates = [
            bbox_center_x - bg_w // 2,
            x1,
            x2 - bg_w,
        ]
        y_candidates = [
            y1 - bg_h - 5,
            y1 + 3,
            y1 + bg_h + 8,
            y1 - (2 * bg_h) - 8,
        ]

        candidate_boxes: List[Tuple[int, int, int, int]] = []
        seen_boxes = set()
        for cand_y in y_candidates:
            for cand_x in x_candidates:
                bg_x, bg_y = _clamp_top_left(cand_x, cand_y, bg_w, bg_h)
                box = (bg_x, bg_y, bg_x + bg_w, bg_y + bg_h)
                if box in seen_boxes:
                    continue
                seen_boxes.add(box)
                candidate_boxes.append(box)

        chosen_box: Optional[Tuple[int, int, int, int]] = None
        for box in candidate_boxes:
            if _is_clear(box):
                chosen_box = box
                break

        if chosen_box is None and candidate_boxes:
            base_x1, base_y1, base_x2, base_y2 = candidate_boxes[0]
            step = max(bg_h + 4, 12)
            for shift in range(step, max(image_h, image_w), step):
                for direction in (1, -1):
                    bg_x, bg_y = _clamp_top_left(base_x1, base_y1 + direction * shift, bg_w, bg_h)
                    box = (bg_x, bg_y, bg_x + bg_w, bg_y + bg_h)
                    if _is_clear(box):
                        chosen_box = box
                        break
                if chosen_box is not None:
                    break

        if chosen_box is None:
            chosen_box = candidate_boxes[0]

        bg_x, bg_y, _bg_x2, _bg_y2 = chosen_box
        placed_boxes.append(chosen_box)
        cv2.rectangle(image, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), color, -1)
        text_x = bg_x + (bg_w - text_w) // 2
        text_y = bg_y + pad + text_h
        cv2.putText(
            image,
            item.label_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )


def build_landmark_strip_lines(
    visible_entries_meta: Sequence[Dict[str, object]],
    offscreen_items: Sequence[Dict[str, Any]],
    landmark_dist_map_multi: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    waypoint_entries: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[LandmarkStripLine]:
    def _short_text(text: str, max_len: int) -> str:
        text = str(text or "").strip()
        if len(text) <= max_len:
            return text
        return text[: max(0, max_len - 2)].rstrip() + ".."

    line_entries: List[Tuple[Tuple[float, float, float], LandmarkStripLine]] = []
    landmark_dist_map_multi = landmark_dist_map_multi or {}
    waypoint_entries = waypoint_entries or []
    status_color = (40, 40, 40)
    name_color = (0, 0, 255)
    value_color = (255, 0, 0)

    visible_cls_total: Dict[str, int] = {}
    for entry in visible_entries_meta:
        cls_name = str(entry["name"])
        total_candidates = len(landmark_dist_map_multi.get(cls_name, [])) or 1
        visible_cls_total[cls_name] = max(
            visible_cls_total.get(cls_name, 0),
            total_candidates,
            int(entry.get("class_total", 1) or 1),
            int(entry.get("instance_idx", 0) or 0) + 1,
        )

    for entry in visible_entries_meta:
        cls_name = str(entry["name"])
        distance_m = float(entry["distance_m"])
        angle_deg = float(entry["angle_deg"])
        selection_rank = entry.get("selection_rank")
        display_name = cls_name
        display_id = entry.get("display_id")
        try:
            display_prefix = f"#{int(display_id)} " if display_id is not None else ""
        except (TypeError, ValueError):
            display_prefix = ""
        instance_idx = entry.get("instance_idx")
        suffix = ""
        if not display_prefix and instance_idx is not None and visible_cls_total.get(cls_name, 0) > 1:
            suffix = f" #{int(instance_idx) + 1}"
        sort_key = (
            float(selection_rank) if selection_rank is not None else float(distance_m),
            0.0 if selection_rank is not None else 0.0,
            0.0 if selection_rank is not None else -float(entry.get("confidence", 0.0)),
        )
        line_entries.append((
            sort_key,
            LandmarkStripLine(
                distance_m=float(distance_m),
                confidence=float(entry.get('confidence', 0.0)),
                priority=0,
                sort_key=sort_key,
                segments=(
                    LandmarkStripSegment(f"{display_prefix}{display_name}{suffix}", name_color),
                    LandmarkStripSegment(f" {distance_m:.1f}m", value_color),
                    LandmarkStripSegment(f" {format_relative_direction(angle_deg)}", value_color),
                    LandmarkStripSegment(f" c{float(entry.get('confidence', 0.0)):.2f}", value_color),
                ),
            ),
        ))

    offscreen_cls_total: Dict[str, int] = {}
    for item in offscreen_items:
        cls_name = str(item["name"])
        offscreen_cls_total[cls_name] = max(
            offscreen_cls_total.get(cls_name, 0),
            int(item.get("class_total", 1) or 1),
            int(item.get("instance_idx", 0) or 0) + 1,
        )

    for item in offscreen_items:
        cls_name = str(item["name"])
        inst_idx = int(item.get("instance_idx", 0) or 0)
        distance_m = float(item["distance_m"])
        angle_deg = float(item["angle_deg"])
        confidence = float(item.get("confidence", 0.0))
        selection_rank = item.get("selection_rank")
        display_name = cls_name
        display_id = item.get("display_id")
        try:
            display_prefix = f"#{int(display_id)} " if display_id is not None else ""
        except (TypeError, ValueError):
            display_prefix = ""
        suffix = ""
        if not display_prefix and offscreen_cls_total.get(cls_name, 0) > 1:
            suffix = f" #{inst_idx + 1}"
        sort_key = (
            float(selection_rank) if selection_rank is not None else float(distance_m),
            0.0 if selection_rank is not None else 1.0,
            0.0 if selection_rank is not None else -float(confidence),
        )
        line_entries.append((
            sort_key,
            LandmarkStripLine(
                distance_m=float(distance_m),
                confidence=float(confidence),
                priority=1,
                sort_key=sort_key,
                segments=(
                    LandmarkStripSegment("off ", status_color),
                    LandmarkStripSegment(f"{display_prefix}{display_name}{suffix}", name_color),
                    LandmarkStripSegment(f" {float(distance_m):.1f}m", value_color),
                    LandmarkStripSegment(f" {format_relative_direction(float(angle_deg))}", value_color),
                    LandmarkStripSegment(f" c{confidence:.2f}", value_color),
                ),
            ),
        ))

    for entry in waypoint_entries:
        if bool(entry.get("is_current_area")):
            continue

        waypoint_label = str(
            entry.get("label")
            or entry.get("area_label")
            or f"WP#{int(entry.get('id', 0) or 0)}"
        ).strip() or "Unknown"
        distance_m = float(entry.get("distance_m", 1e9))
        angle_deg = float(entry.get("relative_bearing_deg", entry.get("angle_deg", 0.0)))
        note_parts: List[str] = []
        if bool(entry.get("is_last_visited")) and not bool(entry.get("is_task_initial_position")):
            note_parts.append("LAST POSITION")
        if bool(entry.get("is_task_initial_position")):
            note_parts.append("INITIAL POSITION")
        note = f"  ({' | '.join(note_parts)})" if note_parts else ""
        sort_key = (1e6 + float(distance_m), 2.0, 0.0)
        line_entries.append((
            sort_key,
            LandmarkStripLine(
                distance_m=distance_m,
                confidence=0.0,
                priority=2,
                sort_key=sort_key,
                segments=(
                    LandmarkStripSegment("space waypoint: ", status_color),
                    LandmarkStripSegment(_short_text(waypoint_label, 34), value_color),
                    LandmarkStripSegment(f"  {distance_m:.1f}m", value_color),
                    LandmarkStripSegment(f"  {format_relative_direction(angle_deg)}", value_color),
                    LandmarkStripSegment(note, status_color),
                ),
            ),
        ))

    line_entries.sort(key=lambda item: item[0])
    return [line for _sort_key, line in line_entries]


def _wrap_strip_segments(
    segments: Sequence[LandmarkStripSegment],
    font: int,
    font_scale: float,
    font_thickness: int,
    max_width: int,
) -> List[List[LandmarkStripSegment]]:
    wrapped_lines: List[List[LandmarkStripSegment]] = []
    current_line: List[LandmarkStripSegment] = []
    current_width = 0

    def _push_current() -> None:
        nonlocal current_line, current_width
        if current_line:
            wrapped_lines.append(current_line)
        current_line = []
        current_width = 0

    for segment in segments:
        segment_text = str(segment.text or "")
        if not segment_text:
            continue
        tokens = re.findall(r"\S+\s*|\s+", segment_text) or [segment_text]
        for token in tokens:
            token = str(token)
            if not token:
                continue
            token_w = cv2.getTextSize(token, font, font_scale, font_thickness)[0][0]
            if current_line and current_width + token_w > max_width:
                _push_current()
                token = token.lstrip()
                if not token:
                    continue
                token_w = cv2.getTextSize(token, font, font_scale, font_thickness)[0][0]
            current_line.append(LandmarkStripSegment(token, segment.color))
            current_width += token_w

    _push_current()
    return wrapped_lines or [[LandmarkStripSegment("", (0, 0, 0))]]


def _truncate_strip_segments_to_single_line(
    segments: Sequence[LandmarkStripSegment],
    font: int,
    font_scale: float,
    font_thickness: int,
    max_width: int,
) -> List[LandmarkStripSegment]:
    ellipsis = "..."
    ellipsis_w = cv2.getTextSize(ellipsis, font, font_scale, font_thickness)[0][0]
    truncated: List[LandmarkStripSegment] = []
    current_width = 0
    fallback_color = (0, 0, 0)

    for segment in segments:
        segment_text = str(segment.text or "")
        if not segment_text:
            continue
        fallback_color = segment.color
        segment_w = cv2.getTextSize(segment_text, font, font_scale, font_thickness)[0][0]
        if current_width + segment_w <= max_width:
            truncated.append(LandmarkStripSegment(segment_text, segment.color))
            current_width += segment_w
            continue

        remaining_width = max_width - current_width
        if remaining_width > 0:
            trimmed = segment_text.rstrip()
            while trimmed:
                candidate = trimmed.rstrip() + ellipsis
                if cv2.getTextSize(candidate, font, font_scale, font_thickness)[0][0] <= remaining_width:
                    truncated.append(LandmarkStripSegment(candidate, segment.color))
                    return truncated
                trimmed = trimmed[:-1]
        break

    if not truncated:
        return [LandmarkStripSegment(ellipsis, fallback_color)]

    if current_width + ellipsis_w <= max_width:
        return truncated + [LandmarkStripSegment(ellipsis, truncated[-1].color)]

    leading_segments = truncated[:-1]
    last_segment = truncated[-1]
    leading_width = 0
    for segment in leading_segments:
        if segment.text:
            leading_width += cv2.getTextSize(segment.text, font, font_scale, font_thickness)[0][0]
    remaining_width = max(0, max_width - leading_width)
    trimmed = str(last_segment.text or "").rstrip()
    while trimmed:
        candidate = trimmed.rstrip() + ellipsis
        if cv2.getTextSize(candidate, font, font_scale, font_thickness)[0][0] <= remaining_width:
            return leading_segments + [LandmarkStripSegment(candidate, last_segment.color)]
        trimmed = trimmed[:-1]

    if leading_segments:
        return leading_segments
    return [LandmarkStripSegment(ellipsis, last_segment.color)]


def render_landmark_strip(
    image_width: int,
    item_lines: Sequence[LandmarkStripLine],
    font_scale: float = 0.60,
    font_thickness: int = 1,
    max_lines_per_item: Optional[int] = None,
    compact: bool = False,
) -> Optional[np.ndarray]:
    if not item_lines:
        return None

    font = cv2.FONT_HERSHEY_SIMPLEX
    sample_size = cv2.getTextSize("Ag", font, font_scale, font_thickness)[0]
    line_h = sample_size[1] + (6 if compact else 10)
    strip_pad = 4 if compact else 6
    card_margin_x = 6 if compact else 8
    card_gap_y = 3 if compact else 6
    card_pad_x = 6 if compact else 8
    card_pad_y = 3 if compact else 6
    card_inner_w = max(40, image_width - (card_margin_x * 2) - (card_pad_x * 2) - 2)

    wrapped_per_item: List[List[List[LandmarkStripSegment]]] = []
    total_height = strip_pad
    for line in item_lines:
        if max_lines_per_item == 1:
            wrapped_lines = [[
                segment for segment in _truncate_strip_segments_to_single_line(
                    line.segments,
                    font,
                    font_scale,
                    font_thickness,
                    card_inner_w,
                )
            ]]
        else:
            wrapped_lines = _wrap_strip_segments(
                line.segments,
                font,
                font_scale,
                font_thickness,
                card_inner_w,
            )
            if max_lines_per_item is not None and max_lines_per_item > 0:
                wrapped_lines = wrapped_lines[:max_lines_per_item]
        wrapped_per_item.append(wrapped_lines)
        total_height += (len(wrapped_lines) * line_h) + (card_pad_y * 2) + card_gap_y
    total_height += max(0, strip_pad - card_gap_y)

    strip = np.full((total_height, image_width, 3), 255, dtype=np.uint8)
    cv2.line(strip, (0, 0), (image_width, 0), (180, 180, 180), 1)

    card_y = strip_pad
    for wrapped_lines in wrapped_per_item:
        card_h = (len(wrapped_lines) * line_h) + (card_pad_y * 2)
        x1 = card_margin_x
        y1 = card_y
        x2 = image_width - card_margin_x - 1
        y2 = y1 + card_h
        cv2.rectangle(strip, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(strip, (x1, y1), (x2, y2), (190, 190, 190), 1)

        for line_idx, segments in enumerate(wrapped_lines):
            text_x = x1 + card_pad_x
            text_y = y1 + card_pad_y + sample_size[1] + (line_idx * line_h)
            for segment in segments:
                if not segment.text:
                    continue
                cv2.putText(
                    strip,
                    segment.text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    segment.color,
                    font_thickness,
                    cv2.LINE_AA,
                )
                text_w = cv2.getTextSize(segment.text, font, font_scale, font_thickness)[0][0]
                text_x += text_w

        card_y = y2 + card_gap_y

    return strip
