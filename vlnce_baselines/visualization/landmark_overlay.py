from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vlnce_baselines.common.spatial_formatter import format_relative_direction


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
) -> None:
    if not draw_items:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    image_w = image.shape[1]
    for item in sorted(draw_items, key=lambda entry: entry.distance_m, reverse=True):
        if not item.label_text:
            continue

        x1, y1, x2, _y2 = item.bbox
        (text_w, text_h), _ = cv2.getTextSize(item.label_text, font, font_scale, font_thickness)
        bg_w = text_w + pad * 2
        bg_h = text_h + pad * 2

        bbox_center_x = (x1 + x2) // 2
        bg_x = max(0, min(bbox_center_x - bg_w // 2, image_w - bg_w))
        bg_y = y1 - bg_h - 5
        if bg_y < 0:
            bg_y = y1 + 3

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
        short_name = cls_name if len(cls_name) <= 40 else cls_name[:38] + ".."
        instance_idx = entry.get("instance_idx")
        suffix = ""
        if instance_idx is not None and visible_cls_total.get(cls_name, 0) > 1:
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
                    LandmarkStripSegment("vis ", status_color),
                    LandmarkStripSegment(f"{short_name}{suffix}", name_color),
                    LandmarkStripSegment(f"  {distance_m:.1f}m", value_color),
                    LandmarkStripSegment(f"  {format_relative_direction(angle_deg)}", value_color),
                    LandmarkStripSegment("  confidence: ", status_color),
                    LandmarkStripSegment(f"{float(entry.get('confidence', 0.0)):.3f}", value_color),
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
        short_name = cls_name if len(cls_name) <= 40 else cls_name[:38] + ".."
        suffix = f" #{inst_idx + 1}" if offscreen_cls_total.get(cls_name, 0) > 1 else ""
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
                    LandmarkStripSegment("off vis ", status_color),
                    LandmarkStripSegment(f"{short_name}{suffix}", name_color),
                    LandmarkStripSegment(f"  {float(distance_m):.1f}m", value_color),
                    LandmarkStripSegment(f"  {format_relative_direction(float(angle_deg))}", value_color),
                    LandmarkStripSegment("  confidence: ", status_color),
                    LandmarkStripSegment(f"{confidence:.3f}", value_color),
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
        note = "  (came from here)" if bool(entry.get("is_last_visited")) else ""
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


def render_landmark_strip(
    image_width: int,
    item_lines: Sequence[LandmarkStripLine],
    font_scale: float = 0.60,
    font_thickness: int = 1,
) -> Optional[np.ndarray]:
    if not item_lines:
        return None

    font = cv2.FONT_HERSHEY_SIMPLEX
    sample_size = cv2.getTextSize("Ag", font, font_scale, font_thickness)[0]
    row_h = sample_size[1] + 14
    pad_v = 6
    strip_h = row_h * len(item_lines) + pad_v * 2
    strip = np.full((strip_h, image_width, 3), 255, dtype=np.uint8)
    cv2.line(strip, (0, 0), (image_width, 0), (180, 180, 180), 1)

    for index, line in enumerate(item_lines):
        text_y = pad_v + row_h * index + sample_size[1]
        text_x = 8
        for segment in line.segments:
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

    return strip
