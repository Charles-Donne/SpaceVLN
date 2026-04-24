"""Auto-extracted helper module from MapVisualizer for space-area overlay logic."""

import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _normalize_space_area_type(self, space_type: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(space_type or "").strip().lower())
    normalized = " ".join(normalized.split())
    if not normalized:
        return "unknown"
    for canonical_name, aliases in self.SPACE_TYPE_ALIAS_GROUPS:
        for alias in aliases:
            alias_text = " ".join(str(alias).strip().lower().split())
            if normalized == alias_text or alias_text in normalized:
                return canonical_name
    return normalized


def _space_area_color_preference_order(
    self,
    area_id: int,
    space_type: str,
) -> List[int]:
    palette_len = len(self.SPACE_AREA_COLOR_PALETTE)
    normalized_type = self._normalize_space_area_type(space_type)
    preferred_index = self.SPACE_TYPE_PREFERRED_COLOR_INDEX.get(normalized_type)
    seed = 0
    for index, ch in enumerate(normalized_type):
        seed += (index + 17) * ord(ch)
    if preferred_index is None:
        preferred_index = seed % max(1, palette_len)
    preferred_index = int(preferred_index) % max(1, palette_len)
    variant_shift = (int(area_id) * 3 + seed) % max(1, palette_len)
    start_index = (preferred_index + variant_shift) % max(1, palette_len)
    fallback_order = [start_index]
    for offset in range(1, palette_len):
        fallback_order.append((start_index + offset) % palette_len)
    return fallback_order


def _build_space_area_adjacency(
    self,
    display_layer: np.ndarray,
    area_ids: List[int],
) -> Dict[int, set]:
    adjacency: Dict[int, set] = {int(area_id): set() for area_id in area_ids}
    layer = np.asarray(display_layer, dtype=np.int32)
    if layer.size == 0:
        return adjacency

    for left_view, right_view in (
        (layer[:, :-1], layer[:, 1:]),
        (layer[:-1, :], layer[1:, :]),
    ):
        valid = (
            (left_view > 0)
            & (right_view > 0)
            & (left_view != right_view)
        )
        if not np.any(valid):
            continue
        left_values = left_view[valid].astype(np.int32, copy=False)
        right_values = right_view[valid].astype(np.int32, copy=False)
        pairs = np.stack(
            [
                np.minimum(left_values, right_values),
                np.maximum(left_values, right_values),
            ],
            axis=1,
        )
        for pair in np.unique(pairs, axis=0):
            area_a = int(pair[0])
            area_b = int(pair[1])
            if area_a == area_b:
                continue
            adjacency.setdefault(area_a, set()).add(area_b)
            adjacency.setdefault(area_b, set()).add(area_a)
    return adjacency


def _resolve_space_area_colors(
    self,
    display_layer: np.ndarray,
    space_area_records: List[Dict[str, Any]],
) -> Dict[int, Tuple[int, int, int]]:
    if display_layer is None or display_layer.size == 0 or not space_area_records:
        return {}

    record_by_id: Dict[int, Dict[str, Any]] = {}
    for record in list(space_area_records or []):
        try:
            area_id = int(record.get("id", 0) or 0)
        except (TypeError, ValueError):
            area_id = 0
        if area_id > 0:
            record_by_id[area_id] = dict(record)
    if not record_by_id:
        return {}

    layer_ids, counts = np.unique(np.asarray(display_layer, dtype=np.int32), return_counts=True)
    area_sizes = {
        int(area_id): int(count)
        for area_id, count in zip(layer_ids.tolist(), counts.tolist())
        if int(area_id) > 0 and int(area_id) in record_by_id
    }
    area_ids = sorted(area_sizes.keys())
    if not area_ids:
        return {}

    adjacency = self._build_space_area_adjacency(display_layer, area_ids)
    ordered_area_ids = sorted(
        area_ids,
        key=lambda area_id: (
            -len(adjacency.get(int(area_id), set())),
            -int(area_sizes.get(int(area_id), 0)),
            self._normalize_space_area_type(
                str(record_by_id.get(int(area_id), {}).get("space_type", ""))
            ),
            int(area_id),
        ),
    )

    color_assignments: Dict[int, Tuple[int, int, int]] = {}
    palette = list(self.SPACE_AREA_COLOR_PALETTE)
    for area_id in ordered_area_ids:
        neighbor_colors = {
            color_assignments[neighbor_id]
            for neighbor_id in adjacency.get(int(area_id), set())
            if neighbor_id in color_assignments
        }
        preference_order = self._space_area_color_preference_order(
            area_id=int(area_id),
            space_type=str(record_by_id.get(int(area_id), {}).get("space_type", "")),
        )
        chosen_color = None
        for color_index in preference_order:
            candidate = palette[int(color_index) % len(palette)]
            if candidate not in neighbor_colors:
                chosen_color = candidate
                break
        if chosen_color is None:
            chosen_color = palette[preference_order[0] % len(palette)]
        color_assignments[int(area_id)] = chosen_color
    return color_assignments


def _prepare_space_area_display_layer(
    self,
    space_area_layer: Optional[np.ndarray],
    output_size: int = 480,
    cache_key: Optional[Any] = None,
) -> Optional[np.ndarray]:
    if space_area_layer is None or space_area_layer.size == 0:
        return None
    cache_token = cache_key if cache_key is not None else self._active_render_cache_key
    cache_entry = None
    if cache_token is not None:
        cache_entry = self._render_cache["space_area_layer"].get((cache_token, int(output_size)))
        if cache_entry is not None:
            return cache_entry.copy()
    layer = np.flipud(np.asarray(space_area_layer, dtype=np.int32))
    display_layer = cv2.resize(layer, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    if cache_token is not None:
        self._render_cache["space_area_layer"][(cache_token, int(output_size))] = display_layer.copy()
    return display_layer


def _refine_space_area_mask(self, mask: np.ndarray) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8) * 255
    if mask_uint8.size == 0 or np.count_nonzero(mask_uint8) == 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    closed = self._remove_small_binary_components(closed, 28).astype(np.uint8) * 255
    closed = self._fill_small_binary_holes(closed, 48).astype(np.uint8) * 255
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return closed > 127

    filled = np.zeros_like(mask_uint8)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return filled > 127


def _draw_space_areas_in_place(
    self,
    image: np.ndarray,
    space_area_layer: Optional[np.ndarray],
    space_area_records: Optional[List[Dict[str, Any]]],
    alpha: float = 0.45,
    fill_regions: bool = True,
    show_labels: bool = False,
    use_display_label: bool = True,
    cache_key: Optional[Any] = None,
    display_layer_override: Optional[np.ndarray] = None,
    reserved_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    cache_token = cache_key if cache_key is not None else self._active_render_cache_key
    fill_alpha = float(max(0.0, min(1.0, alpha)))
    if display_layer_override is not None:
        display_layer = np.asarray(display_layer_override, dtype=np.int32)
        if display_layer.shape[:2] != image.shape[:2]:
            display_layer = cv2.resize(
                display_layer,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
    else:
        display_layer = self._prepare_space_area_display_layer(
            space_area_layer,
            output_size=image.shape[1],
            cache_key=cache_token,
        )
    if display_layer is None or not space_area_records:
        return image

    color_assignments: Dict[int, Tuple[int, int, int]] = {}
    color_cache_key = None
    if cache_token is not None:
        layer_ids, counts = np.unique(np.asarray(display_layer, dtype=np.int32), return_counts=True)
        visible_area_signature = tuple(
            (
                int(area_id),
                int(count),
            )
            for area_id, count in zip(layer_ids.tolist(), counts.tolist())
            if int(area_id) > 0
        )
        visible_area_ids = [int(area_id) for area_id, _count in visible_area_signature]
        adjacency = self._build_space_area_adjacency(display_layer, visible_area_ids)
        adjacency_signature = tuple(
            sorted(
                (
                    int(area_id),
                    tuple(sorted(int(neighbor_id) for neighbor_id in list(neighbors or []))),
                )
                for area_id, neighbors in adjacency.items()
                if int(area_id) > 0
            )
        )
        record_signature = tuple(
            sorted(
                (
                    int(record.get("id", 0) or 0),
                    self._normalize_space_area_type(str(record.get("space_type", ""))),
                )
                for record in list(space_area_records or [])
                if int(record.get("id", 0) or 0) > 0
            )
        )
        color_cache_key = (
            cache_token,
            int(image.shape[1]),
            record_signature,
            visible_area_signature,
            adjacency_signature,
        )
        cached_colors = self._render_cache["space_area_color_assignments"].get(color_cache_key)
        if isinstance(cached_colors, dict):
            color_assignments = dict(cached_colors)
    if not color_assignments:
        color_assignments = self._resolve_space_area_colors(display_layer, list(space_area_records or []))
        if color_cache_key is not None:
            self._render_cache["space_area_color_assignments"][color_cache_key] = dict(color_assignments)

    label_candidates: List[Dict[str, Any]] = []
    for record in space_area_records:
        area_id = int(record.get("id", 0) or 0)
        if area_id <= 0:
            continue
        mask, contours, label_anchor = self._get_space_area_render_assets(
            display_layer=display_layer,
            area_id=area_id,
            output_size=int(image.shape[1]),
            cache_token=cache_token,
        )
        if image is not None and image.ndim == 3 and image.shape[:2] == mask.shape:
            drawable_mask = np.any(image != 0, axis=2)
            clipped_mask = np.logical_and(mask, drawable_mask)
            if np.count_nonzero(clipped_mask) != np.count_nonzero(mask):
                mask = clipped_mask
                mask_uint8 = (mask.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                label_anchor = self._compute_space_area_label_anchor(mask_uint8, contours)
        if not np.any(mask):
            continue
        color = tuple(
            color_assignments.get(
                int(area_id),
                self.SPACE_AREA_COLOR_PALETTE[int(area_id) % len(self.SPACE_AREA_COLOR_PALETTE)],
            )
        )
        if fill_regions:
            if fill_alpha >= 1.0:
                image[mask] = color
            else:
                color_arr = np.asarray(color, dtype=np.float32)
                base_pixels = image[mask].astype(np.float32)
                blended = (base_pixels * (1.0 - fill_alpha)) + (color_arr * fill_alpha)
                image[mask] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

        if contours:
            contour_color = tuple(int(channel) for channel in color)
            if fill_regions:
                cv2.drawContours(image, contours, -1, contour_color, 2)
            if show_labels:
                label_key = "display_label" if use_display_label else "label"
                label = self._compact_space_area_overlay_label(
                    str(record.get(label_key, record.get("label", "")) or "")
                )
                if label:
                    label_candidates.append(
                        {
                            "label": label,
                            "color": color,
                            "label_anchor": label_anchor,
                            "mask": mask,
                            "mask_pixels": int(np.count_nonzero(mask)),
                        }
                    )

    if show_labels and label_candidates:
        placed_boxes: List[Tuple[int, int, int, int]] = [
            tuple(box)
            for box in list(reserved_boxes or [])
            if isinstance(box, (list, tuple)) and len(box) == 4
        ]
        for candidate in sorted(
            label_candidates,
            key=lambda item: (-int(item.get("mask_pixels", 0) or 0), str(item.get("label", ""))),
        ):
            label_layout = self._resolve_space_area_label_layout(
                image_shape=image.shape,
                label=str(candidate.get("label", "")),
                label_anchor=candidate.get("label_anchor"),
                placed_boxes=placed_boxes,
                mask=candidate.get("mask"),
            )
            if label_layout is None:
                continue
            self._draw_space_area_label(
                image=image,
                label=str(candidate.get("label", "")),
                color=tuple(candidate.get("color", (255, 0, 0))),
                label_box=label_layout["box"],
                text_origin=label_layout["text_origin"],
                label_anchor=candidate.get("label_anchor"),
                label_center=label_layout["center"],
            )
            placed_boxes.append(label_layout["box"])

    return image


def _overlay_space_areas(
    self,
    image: np.ndarray,
    space_area_layer: Optional[np.ndarray],
    space_area_records: Optional[List[Dict[str, Any]]],
    alpha: float = 0.45,
    fill_regions: bool = True,
    show_labels: bool = False,
    use_display_label: bool = True,
    cache_key: Optional[Any] = None,
    display_layer_override: Optional[np.ndarray] = None,
    reserved_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    output = image.copy()
    return self._draw_space_areas_in_place(
        output,
        space_area_layer,
        space_area_records,
        alpha=alpha,
        fill_regions=fill_regions,
        show_labels=show_labels,
        use_display_label=use_display_label,
        cache_key=cache_key,
        display_layer_override=display_layer_override,
        reserved_boxes=reserved_boxes,
    )


def _get_space_area_render_assets(
    self,
    display_layer: np.ndarray,
    area_id: int,
    output_size: int,
    cache_token: Optional[Any] = None,
) -> Tuple[np.ndarray, List[np.ndarray], Optional[Tuple[int, int]]]:
    cache_key = None
    if cache_token is not None:
        cache_key = (
            cache_token,
            int(output_size),
            int(area_id),
        )

    mask_bucket = self._render_cache["space_area_mask"]
    contours_bucket = self._render_cache["space_area_contours"]
    anchor_bucket = self._render_cache["space_area_label_anchor"]

    has_mask = cache_key is not None and cache_key in mask_bucket
    has_contours = cache_key is not None and cache_key in contours_bucket
    has_anchor = cache_key is not None and cache_key in anchor_bucket

    if has_mask:
        mask = mask_bucket[cache_key]
    else:
        mask = self._refine_space_area_mask(display_layer == int(area_id))
        if cache_key is not None:
            mask_bucket[cache_key] = mask

    mask_uint8 = (mask.astype(np.uint8) * 255)
    if has_contours:
        contours = contours_bucket[cache_key]
    else:
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cache_key is not None:
            contours_bucket[cache_key] = contours

    if has_anchor:
        label_anchor = anchor_bucket[cache_key]
    else:
        label_anchor = self._compute_space_area_label_anchor(mask_uint8, contours)
        if cache_key is not None:
            anchor_bucket[cache_key] = label_anchor

    return mask, contours, label_anchor


def _compute_space_area_label_anchor(
    self,
    mask_uint8: np.ndarray,
    contours: List[np.ndarray],
) -> Optional[Tuple[int, int]]:
    if mask_uint8 is not None and np.count_nonzero(mask_uint8) > 0:
        distance = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(distance)
        if max_val > 0:
            return int(max_loc[0]), int(max_loc[1])

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    if abs(moments["m00"]) < 1e-6:
        return None
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def _compact_space_area_overlay_label(
    self,
    label: str,
) -> str:
    del self
    text = " ".join(str(label or "").split())
    if not text:
        return ""

    links_marker = "[links:"
    if links_marker in text:
        text = text.split(links_marker, 1)[0].strip()

    # Keep the full area type name on the map tag, but expand compact labels
    # like `LivingRoom2` into `Living Room 2` for readability.
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _label_box_intersection_area(
    self,
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> int:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0
    return int((inter_x2 - inter_x1) * (inter_y2 - inter_y1))


def _build_label_box_from_center(
    self,
    image_shape: Tuple[int, ...],
    center_x: int,
    center_y: int,
    box_w: int,
    box_h: int,
) -> Tuple[int, int, int, int]:
    image_h, image_w = image_shape[:2]
    x1 = int(round(center_x - (box_w / 2.0)))
    y1 = int(round(center_y - (box_h / 2.0)))
    x1 = max(0, min(image_w - box_w, x1))
    y1 = max(0, min(image_h - box_h, y1))
    return x1, y1, x1 + box_w, y1 + box_h


def _label_center_from_box(
    self,
    box: Tuple[int, int, int, int],
) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))


def _get_space_area_label_style(
    self,
    image_shape: Tuple[int, ...],
) -> Dict[str, Any]:
    del self
    min_dim = max(1, int(min(image_shape[:2])))
    scale_ratio = max(0.98, min(1.24, float(min_dim) / 440.0))
    font_scale = 0.50 * scale_ratio
    thickness = max(1, int(round(1.35 * scale_ratio)))
    pad_x = max(5, int(round(6 * scale_ratio)))
    pad_top = max(3, int(round(4 * scale_ratio)))
    pad_bottom = max(2, int(round(3 * scale_ratio)))
    min_box_width = max(48, int(round(56 * scale_ratio)))
    connector_threshold = 14.0 * scale_ratio
    offset_radii = tuple(
        max(12, int(round(radius * scale_ratio)))
        for radius in (14, 22, 30, 42, 56, 74, 96, 124)
    )
    return {
        "font_scale": font_scale,
        "thickness": thickness,
        "pad_x": pad_x,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "min_box_width": min_box_width,
        "connector_threshold": connector_threshold,
        "offset_radii": offset_radii,
    }


def _resolve_space_area_label_layout(
    self,
    image_shape: Tuple[int, ...],
    label: str,
    label_anchor: Optional[Tuple[int, int]],
    placed_boxes: List[Tuple[int, int, int, int]],
    mask: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    if label_anchor is None:
        return None

    anchor_x = int(label_anchor[0])
    anchor_y = int(label_anchor[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_style = self._get_space_area_label_style(image_shape)
    font_scale = float(label_style["font_scale"])
    thickness = int(label_style["thickness"])
    (text_w, text_h), baseline = cv2.getTextSize(str(label), font, font_scale, thickness)
    pad_x = int(label_style["pad_x"])
    pad_top = int(label_style["pad_top"])
    pad_bottom = int(label_style["pad_bottom"])
    box_w = max(int(label_style["min_box_width"]), text_w + pad_x * 2)
    box_h = max(1, text_h + baseline + pad_top + pad_bottom)

    candidate_offsets: List[Tuple[int, int]] = [(0, 0)]
    candidate_angles_deg = (270, 90, 180, 0, 315, 225, 45, 135, 300, 240, 120, 60, 330, 210, 150, 30)
    for radius in label_style["offset_radii"]:
        for angle_deg in candidate_angles_deg:
            angle_rad = np.deg2rad(float(angle_deg))
            offset_x = int(round(radius * np.cos(angle_rad)))
            offset_y = int(round(radius * np.sin(angle_rad)))
            candidate_offsets.append((offset_x, offset_y))

    best_layout: Optional[Dict[str, Any]] = None
    best_score: Optional[Tuple[float, float, float, float]] = None
    for offset_x, offset_y in candidate_offsets:
        candidate_center_x = anchor_x + offset_x
        candidate_center_y = anchor_y + offset_y
        box = self._build_label_box_from_center(
            image_shape=image_shape,
            center_x=candidate_center_x,
            center_y=candidate_center_y,
            box_w=box_w,
            box_h=box_h,
        )
        overlap_area = 0
        overlap_count = 0
        for placed_box in placed_boxes:
            intersection = self._label_box_intersection_area(box, placed_box)
            if intersection > 0:
                overlap_count += 1
                overlap_area += intersection

        x1, y1, x2, y2 = box
        mask_coverage_ratio = 0.0
        if mask is not None and y2 > y1 and x2 > x1:
            region = mask[y1:y2, x1:x2]
            if region.size > 0:
                mask_coverage_ratio = float(np.count_nonzero(region)) / float(region.size)

        distance_penalty = float(np.hypot(offset_x, offset_y))
        boundary_penalty = float(
            min(x1, image_shape[1] - x2, y1, image_shape[0] - y2) <= 2
        )
        score = (
            float(overlap_count),
            float(overlap_area),
            distance_penalty - (mask_coverage_ratio * 18.0),
            boundary_penalty,
        )
        if best_score is None or score < best_score:
            text_origin = (int(x1 + pad_x), int(y1 + pad_top + text_h))
            best_layout = {
                "box": box,
                "text_origin": text_origin,
                "center": self._label_center_from_box(box),
            }
            best_score = score
            if overlap_count == 0 and overlap_area == 0 and distance_penalty <= 1.0:
                break

    return best_layout


def _draw_space_area_label(
    self,
    image: np.ndarray,
    label: str,
    color: Tuple[int, int, int],
    label_box: Tuple[int, int, int, int],
    text_origin: Tuple[int, int],
    label_anchor: Optional[Tuple[int, int]] = None,
    label_center: Optional[Tuple[int, int]] = None,
) -> None:
    del color
    x1, y1, x2, y2 = label_box
    draw_x2 = max(x1, x2 - 1)
    draw_y2 = max(y1, y2 - 1)
    label_style = self._get_space_area_label_style(image.shape)
    label_bg_color = tuple(int(v) for v in self.SPACE_AREA_TAG_BG_COLOR)
    label_border_color = tuple(int(v) for v in self.SPACE_AREA_TAG_BORDER_COLOR)
    label_text_color = tuple(int(v) for v in self.SPACE_AREA_TAG_TEXT_COLOR)
    label_bg_alpha = float(getattr(self, "SPACE_AREA_TAG_BG_ALPHA", 0.85))

    if label_anchor is not None and label_center is not None:
        connector_distance = float(np.hypot(label_center[0] - label_anchor[0], label_center[1] - label_anchor[1]))
        if connector_distance >= float(label_style["connector_threshold"]):
            cv2.line(
                image,
                (int(label_anchor[0]), int(label_anchor[1])),
                (int(label_center[0]), int(label_center[1])),
                label_border_color,
                1,
                cv2.LINE_AA,
            )

    roi = image[y1:draw_y2 + 1, x1:draw_x2 + 1]
    if roi.size > 0:
        bg = np.full_like(roi, label_bg_color, dtype=np.uint8)
        blended = cv2.addWeighted(bg, label_bg_alpha, roi, 1.0 - label_bg_alpha, 0.0)
        image[y1:draw_y2 + 1, x1:draw_x2 + 1] = blended
    cv2.rectangle(image, (x1, y1), (draw_x2, draw_y2), label_border_color, 1)
    cv2.putText(
        image,
        str(label),
        (int(text_origin[0]), int(text_origin[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(label_style["font_scale"]),
        label_text_color,
        int(label_style["thickness"]),
        cv2.LINE_AA,
    )
