"""Auto-extracted helper module from MapVisualizer for clearer separation of concerns."""
import os
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vlnce_baselines.utils.spatial_formatter import format_relative_direction
from vlnce_baselines.visualization.landmark_overlay import (
    LandmarkDrawItem,
    build_landmark_strip_lines,
    draw_action_partition_lines,
    draw_landmark_boxes,
    draw_landmark_labels,
    render_landmark_strip,
)
from vlnce_baselines.config.core.constants import (
    detection_colors,
    detection_thickness,
    detection_visible_topk,
    landmark_strip_topk,
    local_map_landmark_topk,
)

def render_detection_bbox(owner, 
                          rgb: np.ndarray,
                          detections,  # sv.Detections object
                          labels: List[str],
                          landmark_classes: Optional[List[str]] = None,
                          mapping_classes: Optional[List[str]] = None,
                          depth_meters: Optional[np.ndarray] = None,
                          hfov: float = 79.0,
                          landmark_dist_map: Optional[Dict[str, Tuple[float, float]]] = None,
                          landmark_dist_map_multi: Optional[Dict[str, List[Tuple[float, float]]]] = None,
                          landmark_masks: Optional[np.ndarray] = None,
                          show_action_partitions: bool = True,
                          append_bottom_strip: bool = True,
                          controller=None,
                          selected_landmark_instances: Optional[Sequence[Dict[str, Any]]] = None,
                          action_landmark_context: Optional[Dict[str, Any]] = None,
                          return_visible_entries: bool = False,
                          action_distance_overlay: Optional[Dict[str, str]] = None) -> np.ndarray:
    """
    直接在RGB上渲染边界框（只标注Landmark类别，显示距离+水平偏角）

    Args:
        rgb: RGB图像 (H, W, 3) BGR格式
        detections: supervision Detections对象
        labels: 标签列表 (例如: ["chair 0.85", "table 0.92"])
        landmark_classes: Landmark类别列表（只标注这些类别）
        mapping_classes: Mapping类别列表（不标注，仅用于建图）
        depth_meters: 深度图；仅在同类多实例时用于把当前检测实例匹配到地图实例
        hfov: 相机水平视场角；仅用于实例匹配，不用于最终距离/角度显示
        landmark_dist_map: {class_name: (dist_m, rel_angle_deg)} 由地图世界坐标预计算
        landmark_dist_map_multi: {class_name: [(dist_m, rel_angle_deg), ...]} 同类多实例地图信息

    Returns:
        detection_vis: 检测可视化图像（只显示Landmark边界框）
    """
    detection_vis = rgb.copy()
    landmark_dist_map = landmark_dist_map or {}
    landmark_dist_map_multi = landmark_dist_map_multi or {}
    if show_action_partitions:
        draw_action_partition_lines(detection_vis, hfov_deg=float(hfov))

    # 统计检测到的landmark
    detected_landmarks = []
    visible_entries_meta = []
    matched_in_view: set = set()  # 当前帧中实际可见的landmark类名
    candidate_entries: List[Dict[str, Any]] = []
    draw_items: List[LandmarkDrawItem] = []
    action_waypoint_entries: List[Dict[str, Any]] = []

    def _build_action_waypoint_entries() -> List[Dict[str, Any]]:
        if controller is None or getattr(controller, "mapper", None) is None:
            return []

        try:
            from vlnce_baselines.vlm.support.thinking_view_renderer import ThinkingViewRenderer

            map_state = controller.mapper.get_map_state()
            waypoint_positions, waypoint_ids, waypoint_descriptions = controller.mapper.get_waypoints()
            waypoint_info = None
            if waypoint_positions and waypoint_ids:
                waypoint_info = (waypoint_positions, waypoint_ids, waypoint_descriptions)

            # Reuse the same waypoint visibility test as the 12-view thinking render.
            waypoint_entries = ThinkingViewRenderer._build_waypoint_view_entries(
                waypoint_info=waypoint_info,
                waypoint_area_labels=map_state.get("waypoint_area_labels", []),
                current_pose=map_state.get("full_pose"),
                resolution_cm=float(getattr(controller.mapper, "resolution", owner.resolution)),
                current_space_area_label=str(map_state.get("current_space_area_label", "Unknown") or "Unknown"),
            )
            waypoint_entries = ThinkingViewRenderer._apply_waypoint_visibility(
                waypoint_entries=waypoint_entries,
                view_angles_deg=[0.0],
                full_map=map_state.get("full_map"),
                current_pose=map_state.get("full_pose"),
                resolution_cm=float(getattr(controller.mapper, "resolution", owner.resolution)),
                crop_offset=map_state.get("crop_offset"),
            )
        except Exception:
            return []

        filtered_entries: List[Dict[str, Any]] = []
        for entry in waypoint_entries:
            if bool(entry.get("is_current_area")):
                continue
            try:
                relative_bearing_deg = float(entry.get("relative_bearing_deg", 999.0))
            except (TypeError, ValueError):
                continue
            if abs(relative_bearing_deg) > 60.0:
                continue
            filtered_entries.append(dict(entry))

        filtered_entries.sort(
            key=lambda item: (
                float(item.get("distance_m", 1e9)),
                int(item.get("id", 0) or 0),
            )
        )
        return filtered_entries

    if action_landmark_context is None:
        if selected_landmark_instances is not None:
            selected_world_landmark_instances = [dict(item) for item in (selected_landmark_instances or [])]
            all_world_landmark_instances: List[Dict[str, Any]] = []
            if controller is not None and getattr(controller, "latest_landmark_instances_world", None):
                all_world_landmark_instances = list(controller.latest_landmark_instances_world or [])
            display_lookup_source = all_world_landmark_instances or selected_world_landmark_instances
            action_landmark_context = {
                "all_instances": all_world_landmark_instances,
                "selected_instances": selected_world_landmark_instances,
                "display_index_lookup": owner._build_landmark_display_index_lookup(display_lookup_source),
                "class_totals": owner._build_landmark_class_totals(display_lookup_source),
            }
        else:
            all_world_landmark_instances: List[Dict[str, Any]] = []
            if controller is not None and getattr(controller, "latest_landmark_instances_world", None):
                all_world_landmark_instances = list(controller.latest_landmark_instances_world or [])
            action_landmark_context = owner._build_action_landmark_context(
                all_world_landmark_instances,
                topk=local_map_landmark_topk,
            )

    all_world_landmark_instances = list(action_landmark_context.get("all_instances", []) or [])
    selected_world_landmark_instances = [
        dict(item) for item in (action_landmark_context.get("selected_instances", []) or [])
    ]
    landmark_display_index_lookup = dict(action_landmark_context.get("display_index_lookup", {}) or {})
    world_class_totals: Dict[str, int] = dict(action_landmark_context.get("class_totals", {}) or {})

    def _float_or_none(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _entry_display_id(entry: Optional[Dict[str, Any]], default: Optional[int] = None) -> Optional[int]:
        if not entry:
            return default
        display_id = entry.get("display_id")
        try:
            if display_id is not None:
                return int(display_id)
        except (TypeError, ValueError):
            pass

        rank_value = entry.get("selection_rank")
        try:
            if rank_value is not None:
                return int(rank_value) + 1
        except (TypeError, ValueError):
            pass

        instance_idx = entry.get("instance_idx")
        try:
            if instance_idx is not None:
                return int(instance_idx) + 1
        except (TypeError, ValueError):
            pass
        return default

    def _normalize_selected_world_entry(
        inst: Dict[str, Any],
        source: str,
        candidate: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized = dict(inst)
        uid = owner._landmark_instance_uid(normalized)
        if uid is not None:
            normalized["instance_uid"] = int(uid)
            if uid in landmark_display_index_lookup:
                normalized["instance_idx"] = int(landmark_display_index_lookup[uid])

        cls_name = str(normalized.get("name", "") or "")
        normalized["source"] = "vis" if source == "vis" else "off"
        normalized["selection_rank"] = int(normalized.get("selection_rank", 0) or 0)
        display_id = _entry_display_id(normalized)
        if display_id is not None:
            normalized["display_id"] = int(display_id)
        normalized["class_total"] = int(world_class_totals.get(cls_name, max(int(normalized.get("instance_idx", 0) or 0) + 1, 1)))
        confidence_value = _float_or_none(normalized.get("confidence"))
        if confidence_value is None and candidate is not None:
            confidence_value = _float_or_none(candidate.get("confidence"))
        normalized["confidence"] = float(confidence_value if confidence_value is not None else 0.0)

        distance_m = _float_or_none(normalized.get("distance_m"))
        angle_deg = _float_or_none(normalized.get("angle_deg"))
        if distance_m is not None:
            normalized["distance_m"] = float(distance_m)
        if angle_deg is not None:
            normalized["angle_deg"] = float(angle_deg)

        if candidate is not None:
            normalized["bbox"] = tuple(candidate.get("bbox", ()))
            normalized["visible_confidence"] = float(candidate.get("confidence", 0.0))
            normalized["is_opening_like"] = bool(
                normalized.get("is_opening_like", False) or candidate.get("is_opening_like", False)
            )
            normalized["used_edge_geometry"] = bool(
                normalized.get("used_edge_geometry", False) or candidate.get("used_edge_geometry", False)
            )
            old_gap = normalized.get("opening_gap_m")
            new_gap = candidate.get("opening_gap_m")
            if old_gap is None:
                normalized["opening_gap_m"] = new_gap
            elif new_gap is None:
                normalized["opening_gap_m"] = old_gap
            else:
                normalized["opening_gap_m"] = max(float(old_gap), float(new_gap))
            normalized["edge_depth_median"] = candidate.get("edge_depth_median", normalized.get("edge_depth_median"))
            normalized["interior_depth_median"] = candidate.get("interior_depth_median", normalized.get("interior_depth_median"))
            normalized["stop_distance_m"] = min(
                float(_float_or_none(normalized.get("stop_distance_m")) or 1.0),
                float(_float_or_none(candidate.get("stop_distance_m")) or 1.0),
            )
        else:
            normalized["stop_distance_m"] = float(_float_or_none(normalized.get("stop_distance_m")) or 1.0)
        return normalized

    def _build_landmark_strip(selected_entries: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        ordered_entries = [dict(item) for item in selected_entries]
        strip = None
        if ordered_entries:
            selected_visible_entries = [
                entry for entry in ordered_entries
                if str(entry.get("source", "off")) == "vis"
            ][:max(1, int(landmark_strip_topk))]
            item_lines = build_landmark_strip_lines(
                selected_visible_entries,
                [],
                landmark_dist_map_multi=landmark_dist_map_multi,
                waypoint_entries=None,
            )
            if item_lines:
                strip = render_landmark_strip(
                    detection_vis.shape[1],
                    item_lines,
                    font_scale=0.54,
                    font_thickness=1,
                    compact=True,
                )
        return strip, ordered_entries

    action_waypoint_entries: List[Dict[str, Any]] = []

    if detections is None or len(detections.xyxy) == 0:
        selected_topk_entries = [
            _normalize_selected_world_entry(inst, source="off")
            for inst in selected_world_landmark_instances
        ]
        strip, selected_topk_entries = _build_landmark_strip(selected_topk_entries)
        if controller is not None:
            controller.latest_visible_landmark_entries = []
            controller.latest_action_landmark_topk_entries = selected_topk_entries
        if append_bottom_strip and strip is not None:
            detection_vis = np.vstack([detection_vis, strip])
        if return_visible_entries:
            return detection_vis, [], set(), strip, []
        return detection_vis, [], set(), strip

    depth_for_match = depth_meters
    if depth_for_match is None and controller is not None:
        depth_for_match = getattr(controller, "latest_depth_meters", None)
    current_pose = None
    if controller is not None and getattr(controller, "mapper", None) is not None:
        current_pose = controller.mapper.get_current_pose()

    for i in range(len(detections.xyxy)):
        bbox = detections.xyxy[i]
        label = labels[i] if i < len(labels) else f"object_{i}"

        # 提取类别名和置信度
        parts = label.split()
        label_name = ' '.join(parts[:-1]) if len(parts) > 1 else (parts[0] if len(parts) > 0 else "unknown")
        confidence = float(parts[-1]) if len(parts) > 1 else 0.0

        # 只标注在landmark_classes中的类别（规范化后精确短语匹配）
        matched_landmark = None
        if landmark_classes:
            lm_name_map = {lm.strip().lower(): lm for lm in landmark_classes}
            label_name_norm = label_name.strip().lower()
            if label_name_norm in lm_name_map:
                matched_landmark = lm_name_map[label_name_norm]
        if matched_landmark is None:
            continue  # 跳过非Landmark类别

        x1, y1, x2, y2 = map(int, bbox)
        label_name = matched_landmark  # 用完整landmark名称显示

        # 仅用 bbox 中心做文字框定位；同类多实例时才做 mask+depth 到地图实例的匹配。
        _, w_img = rgb.shape[:2]
        det_mask = None
        if getattr(detections, "mask", None) is not None and i < len(detections.mask):
            det_mask = detections.mask[i]
        det_rel_xy = None
        det_depth_profile: Dict[str, Any] = {}
        if det_mask is not None and depth_for_match is not None:
            det_rel_xy, det_depth_profile = owner._estimate_mask_rel_xy(
                det_mask,
                depth_for_match,
                hfov,
                landmark_name=label_name,
                return_profile=True,
            )
        world_xy = owner._rel_xy_to_world_xy(det_rel_xy, current_pose)

        candidate_entries.append({
            "name": label_name,
            "confidence": float(confidence),
            "bbox": (x1, y1, x2, y2),
            "det_rel_xy": det_rel_xy,
            "w_img": w_img,
            "raw_index": i,
            "current_pose": current_pose,
            "world_x_m": float(world_xy[0]) if world_xy is not None else None,
            "world_y_m": float(world_xy[1]) if world_xy is not None else None,
            "is_opening_like": bool(det_depth_profile.get("is_opening_like", False)),
            "used_edge_geometry": bool(det_depth_profile.get("used_edge_geometry", False)),
            "opening_gap_m": det_depth_profile.get("opening_gap_m"),
            "edge_depth_median": det_depth_profile.get("edge_depth_median"),
            "interior_depth_median": det_depth_profile.get("interior_depth_median"),
            "stop_distance_m": 0.5 if bool(det_depth_profile.get("is_opening_like", False)) else 1.0,
        })

    deduped_candidates = owner._dedupe_detection_candidates(
        candidate_entries,
        hfov=hfov,
        topk=None if selected_world_landmark_instances else detection_visible_topk,
    )

    selected_topk_entries: List[Dict[str, Any]] = []
    if selected_world_landmark_instances:
        visible_candidates_by_uid: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        for candidate in deduped_candidates:
            matched_inst = owner._match_candidate_to_world_instance(
                candidate,
                selected_world_landmark_instances,
                hfov,
            )
            matched_uid = owner._landmark_instance_uid(matched_inst or {})
            if matched_uid is None:
                continue
            previous = visible_candidates_by_uid.get(matched_uid)
            candidate_key = (
                -float(candidate.get("confidence", 0.0)),
                owner._candidate_distance_m(candidate),
                int(candidate.get("raw_index", 0)),
            )
            if previous is None:
                visible_candidates_by_uid[matched_uid] = (dict(candidate), dict(matched_inst))
                continue
            previous_key = (
                -float(previous[0].get("confidence", 0.0)),
                owner._candidate_distance_m(previous[0]),
                int(previous[0].get("raw_index", 0)),
            )
            if candidate_key < previous_key:
                visible_candidates_by_uid[matched_uid] = (dict(candidate), dict(matched_inst))

        for selected_inst in selected_world_landmark_instances:
            matched_uid = owner._landmark_instance_uid(selected_inst)
            matched_pair = visible_candidates_by_uid.get(matched_uid) if matched_uid is not None else None
            if matched_pair is None:
                selected_topk_entries.append(
                    _normalize_selected_world_entry(selected_inst, source="off")
                )
                continue

            candidate, matched_inst = matched_pair
            label_name = str(matched_inst.get("name", selected_inst.get("name", "")) or "")
            confidence = float(candidate.get("confidence", 0.0))
            x1, y1, x2, y2 = candidate["bbox"]
            det_rel_xy = candidate.get("det_rel_xy")
            matched_uid = owner._landmark_instance_uid(matched_inst)
            display_idx = None
            if matched_uid is not None:
                display_idx = landmark_display_index_lookup.get(matched_uid)
            if display_idx is None:
                try:
                    display_idx = int(matched_inst.get("instance_idx", selected_inst.get("instance_idx", 0)) or 0)
                except (TypeError, ValueError):
                    display_idx = None

            shown_dist_m = _float_or_none(matched_inst.get("distance_m"))
            shown_angle_deg = _float_or_none(matched_inst.get("angle_deg"))
            if shown_dist_m is None:
                shown_dist_m = _float_or_none(selected_inst.get("distance_m"))
            if shown_angle_deg is None:
                shown_angle_deg = _float_or_none(selected_inst.get("angle_deg"))

            same_cls_total = int(world_class_totals.get(label_name, len(landmark_dist_map_multi.get(label_name, [])) or 1))
            display_id = _entry_display_id(selected_inst, default=_entry_display_id(matched_inst))
            inst_prefix = f"#{display_id} " if display_id is not None else ""

            if shown_dist_m is not None and shown_angle_deg is not None:
                row1 = f"{inst_prefix}{shown_dist_m:.1f}m {format_relative_direction(shown_angle_deg)}"
            elif shown_dist_m is not None:
                row1 = f"{inst_prefix}{shown_dist_m:.1f}m"
            else:
                fallback_angle_deg = None
                fallback_dist_str = None
                fallback_dist_m = None
                if det_rel_xy is not None:
                    forward_m, right_m = det_rel_xy
                    fallback_dist_m = float(np.hypot(forward_m, right_m))
                    fallback_angle_deg = float(np.degrees(np.arctan2(right_m, forward_m)))
                    if fallback_dist_m > 0.05:
                        fallback_dist_str = f"{min(fallback_dist_m, 5.0):.1f}m"
                if fallback_angle_deg is None:
                    fallback_angle_deg = owner._candidate_angle_deg(candidate, hfov)
                if fallback_dist_str is None:
                    fallback_dist_str = ">5.0m"
                shown_dist_m = fallback_dist_m if fallback_dist_m is not None else 5.1
                shown_angle_deg = fallback_angle_deg
                row1 = f"{inst_prefix}{fallback_dist_str} {format_relative_direction(fallback_angle_deg)}"

            detected_landmarks.append((label_name, confidence))
            matched_in_view.add(label_name)

            visible_entry = _normalize_selected_world_entry(
                matched_inst,
                source="vis",
                candidate=candidate,
            )
            if display_idx is not None:
                visible_entry["instance_idx"] = int(display_idx)
            if display_id is not None:
                visible_entry["display_id"] = int(display_id)
            visible_entry["distance_m"] = float(shown_dist_m) if shown_dist_m is not None else float(visible_entry.get("distance_m", 1e9))
            visible_entry["angle_deg"] = float(shown_angle_deg) if shown_angle_deg is not None else float(visible_entry.get("angle_deg", 0.0))
            visible_entry["class_total"] = same_cls_total
            visible_entries_meta.append(visible_entry)
            selected_topk_entries.append(visible_entry)
            draw_items.append(
                LandmarkDrawItem(
                    bbox=(x1, y1, x2, y2),
                    label_text=row1,
                    distance_m=float(shown_dist_m) if shown_dist_m is not None else 999.0,
                )
            )
    else:
        used_map_candidates = {}
        selected_entries = deduped_candidates[:max(1, int(detection_visible_topk))]
        for selection_rank, candidate in enumerate(selected_entries):
            label_name = candidate["name"]
            confidence = float(candidate["confidence"])
            x1, y1, x2, y2 = candidate["bbox"]
            det_rel_xy = candidate["det_rel_xy"]

            detected_landmarks.append((label_name, confidence))
            matched_in_view.add(label_name)

            same_cls_total = len(landmark_dist_map_multi.get(label_name, [])) if landmark_dist_map_multi else 1

            map_dist_m = None
            map_angle_deg = None
            map_instance_idx = None
            if landmark_dist_map_multi and label_name in landmark_dist_map_multi:
                used_set = used_map_candidates.setdefault(label_name, set())
                candidates = sorted(landmark_dist_map_multi[label_name], key=lambda x: x[0])
                ranked_candidates = []
                for idx_c, (dist_m_c, angle_deg_c) in enumerate(candidates):
                    if idx_c in used_set:
                        continue
                    angle_rad_c = np.deg2rad(angle_deg_c)
                    cand_rel_xy = (
                        float(dist_m_c * np.cos(angle_rad_c)),
                        float(dist_m_c * np.sin(angle_rad_c)),
                    )
                    if det_rel_xy is not None:
                        match_cost = float(np.hypot(
                            cand_rel_xy[0] - det_rel_xy[0],
                            cand_rel_xy[1] - det_rel_xy[1],
                        ))
                    else:
                        match_cost = float(dist_m_c)
                    ranked_candidates.append((idx_c, dist_m_c, angle_deg_c, match_cost))
                if ranked_candidates:
                    ranked_candidates.sort(key=lambda item: (item[3], item[1]))
                    map_instance_idx, map_dist_m, map_angle_deg, _ = ranked_candidates[0]
                    used_set.add(map_instance_idx)
            elif landmark_dist_map and label_name in landmark_dist_map:
                map_dist_m, map_angle_deg = landmark_dist_map[label_name]
                map_instance_idx = 0

            row1 = ""
            display_id = int(selection_rank) + 1
            inst_prefix = f"#{display_id} "
            shown_dist_m = map_dist_m
            shown_angle_deg = map_angle_deg
            if shown_dist_m is not None and shown_angle_deg is not None:
                row1 = f"{inst_prefix}{shown_dist_m:.1f}m {format_relative_direction(shown_angle_deg)}"
            elif shown_dist_m is not None:
                row1 = f"{inst_prefix}{shown_dist_m:.1f}m"
            else:
                fallback_angle_deg = None
                fallback_dist_str = None
                fallback_dist_m = None
                if det_rel_xy is not None:
                    forward_m, right_m = det_rel_xy
                    fallback_dist_m = float(np.hypot(forward_m, right_m))
                    fallback_angle_deg = float(np.degrees(np.arctan2(right_m, forward_m)))
                    if fallback_dist_m > 0.05:
                        fallback_dist_str = f"{min(fallback_dist_m, 5.0):.1f}m"
                if fallback_angle_deg is None:
                    fallback_angle_deg = owner._candidate_angle_deg(candidate, hfov)
                if fallback_dist_str is None:
                    fallback_dist_str = ">5.0m"
                shown_dist_m = fallback_dist_m if fallback_dist_m is not None else 5.1
                shown_angle_deg = fallback_angle_deg
                row1 = f"{inst_prefix}{fallback_dist_str} {format_relative_direction(fallback_angle_deg)}"

            visible_entry = {
                "name": label_name,
                "confidence": float(confidence),
                "distance_m": float(shown_dist_m),
                "angle_deg": float(shown_angle_deg),
                "instance_idx": map_instance_idx,
                "display_id": display_id,
                "selection_rank": int(selection_rank),
                "source": "vis",
                "class_total": int(max(same_cls_total, (map_instance_idx or 0) + 1)),
                "is_opening_like": bool(candidate.get("is_opening_like", False)),
                "used_edge_geometry": bool(candidate.get("used_edge_geometry", False)),
                "opening_gap_m": candidate.get("opening_gap_m"),
                "edge_depth_median": candidate.get("edge_depth_median"),
                "interior_depth_median": candidate.get("interior_depth_median"),
                "stop_distance_m": float(candidate.get("stop_distance_m", 1.0)),
            }
            visible_entries_meta.append(visible_entry)
            selected_topk_entries.append(dict(visible_entry))
            draw_items.append(
                LandmarkDrawItem(
                    bbox=(x1, y1, x2, y2),
                    label_text=row1,
                    distance_m=float(shown_dist_m) if shown_dist_m is not None else 999.0,
                )
            )

    # 先渲染bbox框
    color = detection_colors["landmark"]
    thickness = detection_thickness["landmark"]
    draw_landmark_boxes(detection_vis, draw_items, color, thickness)
    draw_landmark_labels(
        detection_vis,
        draw_items,
        color,
        avoid_boxes=_build_action_distance_label_boxes(detection_vis.shape, action_distance_overlay),
    )

    strip, selected_topk_entries = _build_landmark_strip(selected_topk_entries)
    if append_bottom_strip and strip is not None:
        detection_vis = np.vstack([detection_vis, strip])

    if controller is not None:
        controller.latest_visible_landmark_entries = visible_entries_meta
        controller.latest_action_landmark_topk_entries = selected_topk_entries

    # 返回检测可视化、检测到的landmark列表、已匹配的类名集合和底部条带
    if return_visible_entries:
        return detection_vis, detected_landmarks, matched_in_view, strip, visible_entries_meta
    return detection_vis, detected_landmarks, matched_in_view, strip

def _build_action_distance_label_boxes(
    image_shape: Tuple[int, int, int],
    distance_dict: Optional[Dict[str, str]],
) -> List[Tuple[int, int, int, int]]:
    if not distance_dict:
        return []

    h, w = image_shape[:2]
    center_x = w // 2
    bottom_y = h - 10
    direction_configs = [
        {'key': 'left_30', 'angle': -120, 'label': 'Left 30'},
        {'key': 'front', 'angle': -90, 'label': 'FRONT'},
        {'key': 'right_30', 'angle': -60, 'label': 'Right 30'},
    ]

    reserved_boxes: List[Tuple[int, int, int, int]] = []
    for config in direction_configs:
        dist_str = distance_dict.get(config['key'], 'Unknown')
        if dist_str == 'Unknown':
            continue

        if 'WARNING' in dist_str or '<0.5' in dist_str:
            line_length = 65 if config['key'] == 'front' else 60
        elif '>2.0' in dist_str or 'open' in dist_str:
            line_length = 140 if config['key'] == 'front' else 120
        else:
            line_length = 105 if config['key'] == 'front' else 90

        angle_rad = np.deg2rad(config['angle'])
        end_x = int(center_x + line_length * np.cos(angle_rad))
        end_y = int(bottom_y + line_length * np.sin(angle_rad))
        font_scale = 0.72 if config['key'] == 'front' else 0.62
        font_thickness = 2
        combined_label = f"{config['label']} {dist_str}"
        label_size = cv2.getTextSize(combined_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        label_offset = 25
        base_x = int(end_x + label_offset * np.cos(angle_rad))
        base_y = int(end_y + label_offset * np.sin(angle_rad))

        if config['key'] == 'front':
            text_x = base_x - label_size[0] // 2
            text_y = base_y + label_size[1] // 2
        elif config['key'] == 'left_30':
            text_x = base_x - label_size[0] - 15
            text_y = base_y + label_size[1] // 2
        else:
            text_x = base_x + 15
            text_y = base_y + label_size[1] // 2

        margin = 8
        x1 = max(0, text_x - 2 - margin)
        y1 = max(0, text_y - label_size[1] - 2 - margin)
        x2 = min(w - 1, text_x + label_size[0] + 2 + margin)
        y2 = min(h - 1, text_y + 2 + margin)
        reserved_boxes.append((x1, y1, x2, y2))

    return reserved_boxes

def save_rgb(owner, step: int, episode_id: int, rgb: np.ndarray, phase: str = "action", controller = None) -> str:
    """
    保存原始RGB帧（添加距离线）

    Args:
        step: 步数
        episode_id: episode ID
        rgb: RGB图像 (H, W, 3) BGR格式
        phase: 阶段标识 ("initial", "action1a", "verify1a" 等)
        controller: VLMNavigationController实例（用于访问_draw_distance_rays_on_first_person_view）

    Returns:
        save_path: 保存路径
    """
    # 如果是action阶段且提供了controller，绘制距离线
    if phase.startswith('action') and controller is not None:
        if hasattr(controller, '_draw_distance_rays_on_first_person_view') and hasattr(controller, 'latest_obstacle_distances'):
            rgb = controller._draw_distance_rays_on_first_person_view(rgb.copy(), controller.latest_obstacle_distances)

    episode_dir = owner._create_episode_directories(episode_id)
    save_path = os.path.join(episode_dir, 'rgb', f'step_{step:04d}_{phase}.png')
    cv2.imwrite(save_path, rgb)
    return save_path

def draw_floor_from_saved_mask(owner, image: np.ndarray, mask_path: str, classes: List[str]) -> np.ndarray:
    """
    使用保存的semantic mask绘制地面分割（直接使用原始检测的floor mask）

    Args:
        image: 图像 (H, W, 3) BGR格式
        mask_path: semantic mask的numpy文件路径
        classes: 类别列表（用于查找floor索引）

    Returns:
        绘制了地面分割的图像
    """
    try:
        if not os.path.exists(mask_path):
            print(f"  ⚠️  Mask file not found: {mask_path}")
            return image

        masks = np.load(mask_path)
        floor_idx = None
        for i, cls in enumerate(classes):
            if cls.lower() == 'floor':
                floor_idx = i
                break

        if floor_idx is None:
            print(f"  ⚠️  'floor' not found in classes: {classes}")
            return image

        if floor_idx >= masks.shape[0]:
            print(f"  ⚠️  floor_idx {floor_idx} >= masks.shape[0] {masks.shape[0]}")
            return image

        floor_mask = masks[floor_idx]

        # 增强可见性：更明显的绿色覆盖
        overlay = image.copy()
        green_color = np.array([0, 255, 0], dtype=np.uint8)  # 纯绿色
        floor_bool = floor_mask > 0.1

        # 如果mask有效像素太少，打印警告
        if np.sum(floor_bool) < 100:
            print(f"  ⚠️  Floor mask has too few pixels: {np.sum(floor_bool)}")
            return image

        # 绘制半透明绿色覆盖
        overlay[floor_bool] = overlay[floor_bool] * 0.6 + green_color * 0.4
        alpha = 0.7  # 增加透明度，让绿色更明显
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Floor mask应用成功（静默）
        return result
    except Exception as e:
        # 静默失败处理
        return image

def draw_distance_on_view(owner, image: np.ndarray, distance_str: str) -> np.ndarray:
    """
    在视图上绘制距离信息（梯形线条 - 用于thinking模式12个方向view）

    Args:
        image: 图像 (H, W, 3) BGR格式
        distance_str: 距离字符串
    """
    h, w = image.shape[:2]
    center_x = w // 2
    bottom_y = h - 5
    side_offset = int(w * 0.25)  # 增大两侧宽度：0.15 → 0.25

    if "WARNING" in distance_str or "<0.5" in distance_str:
        color, line_ratio, top_shrink = (180, 105, 255), 0.15, 0.8  # 淡粉红(HotPink)：只延伸一点点，顶部收缩到0.8
    elif ">2.0" in distance_str or "open" in distance_str:
        color, line_ratio, top_shrink = (0, 255, 0), 0.65, 0.3  # 绿色：降到之前黄色位置，顶部收缩到0.3（最窄）
    else:
        color, line_ratio, top_shrink = (0, 255, 255), 0.4, 0.5  # 黄色：再低一点，顶部收缩到0.5（中等）

    max_length = bottom_y - h // 2
    end_y = bottom_y - int(max_length * line_ratio)

    cv2.line(image, (center_x, bottom_y), (center_x, end_y), color, 3)
    cv2.line(image, (center_x - side_offset, bottom_y), (center_x - int(side_offset * top_shrink), end_y), color, 2)
    cv2.line(image, (center_x + side_offset, bottom_y), (center_x + int(side_offset * top_shrink), end_y), color, 2)

    text_x = center_x + 10
    text_y = (bottom_y + h // 2) // 2
    font_scale, thickness = 0.6, 2
    text_size = cv2.getTextSize(distance_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    cv2.rectangle(image, (text_x - 2, text_y - text_size[1] - 1),
                 (text_x + text_size[0] + 2, text_y + 2), (0, 0, 0), -1)
    cv2.putText(image, distance_str, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return image

def draw_distance_on_action_view(owner, image: np.ndarray, distance_dict: Dict[str, str]) -> np.ndarray:
    """
    在Action模式视图上绘制3个方向的距离信息（Left 30 / Front / Right 30）

    Args:
        image: 图像 (H, W, 3) BGR格式
        distance_dict: 距离字典，key为方向（'front', 'left_30', 'right_30'）
    """
    h, w = image.shape[:2]
    center_x = w // 2
    # 自动检测白底条带：如果图像底部存在白色条带（np.vstack拼接的landmark信息栏），
    # 则只在原始RGB区域内画距离线，避免射线起点落在白色区域内。
    # 检测方法：若最后一行全白（mean>253），向上扫描找到第一个非白行。
    h_rgb = h
    if image[-1].mean() > 253:
        for r in range(h - 1, -1, -1):
            if image[r].mean() < 250:
                h_rgb = r + 1
                break
    bottom_y = h_rgb - 10

    # 3个方向：左30, 前, 右30
    direction_configs = [
        {'key': 'left_30', 'angle': -120, 'label': 'Left 30'},
        {'key': 'front', 'angle': -90, 'label': 'FRONT'},
        {'key': 'right_30', 'angle': -60, 'label': 'Right 30'},
    ]

    for config in direction_configs:
        dist_str = distance_dict.get(config['key'], 'Unknown')
        if dist_str == 'Unknown':
            continue

        # 根据距离确定颜色和长度（FRONT线条更长）
        if "WARNING" in dist_str or "<0.5" in dist_str:
            color = (180, 105, 255)  # 淡粉红(HotPink)
            line_length = 65 if config['key'] == 'front' else 60
        elif ">2.0" in dist_str or "open" in dist_str:
            color = (0, 255, 0)  # 绿色
            line_length = 140 if config['key'] == 'front' else 120
        else:
            color = (0, 255, 255)  # 黄色
            line_length = 105 if config['key'] == 'front' else 90

        # 计算终点
        angle_rad = np.deg2rad(config['angle'])
        end_x = int(center_x + line_length * np.cos(angle_rad))
        end_y = int(bottom_y + line_length * np.sin(angle_rad))

        # 绘制线条（中心线粗一点）
        thickness = 3 if config['key'] == 'front' else 2
        cv2.line(image, (center_x, bottom_y), (end_x, end_y), color, thickness)

        # FRONT用大字号，其他用稍大字号
        font_scale = 0.72 if config['key'] == 'front' else 0.62
        font_thickness = 2 if config['key'] == 'front' else 2

        # 合并标签为单行："Left 90 1.3m" 或 "FRONT 0.70m"
        combined_label = f"{config['label']} {dist_str}"
        label_size = cv2.getTextSize(combined_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

        # 标签位置：从线条终点沿方向延伸
        label_offset = 25
        base_x = int(end_x + label_offset * np.cos(angle_rad))
        base_y = int(end_y + label_offset * np.sin(angle_rad))

        # 根据方向调整标签位置，使其向两侧延伸，远离中心
        if config['key'] == 'front':
            # FRONT标签居中
            text_x = base_x - label_size[0] // 2
            text_y = base_y + label_size[1] // 2
        elif config['key'] == 'left_30':
            # 左侧标签：向左延伸，右对齐（文字在线条左侧）
            side_offset = 15
            text_x = base_x - label_size[0] - side_offset
            text_y = base_y + label_size[1] // 2
        else:  # right_30
            # 右侧标签：向右延伸，左对齐（文字在线条右侧）
            side_offset = 15
            text_x = base_x + side_offset
            text_y = base_y + label_size[1] // 2

        # 绘制黑色背景和文字
        cv2.rectangle(image, (text_x - 2, text_y - label_size[1] - 2),
                     (text_x + label_size[0] + 2, text_y + 2), (0, 0, 0), -1)
        cv2.putText(image, combined_label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    return image

def prepare_action_image_with_enhancements(owner, image_path: str, mask_path: str = None, 
                                           distance_dict: Dict[str, str] = None, classes: List[str] = None,
                                           use_floor: bool = True, use_distance: bool = True) -> str:
    """
    为action模式准备增强图像：添加地面分割（绿色）和3方向距离辅助线

    Args:
        image_path: 原始图像路径
        mask_path: semantic mask路径
        distance_dict: 距离字典 {'front': 'X.XXm', 'left_30': 'X.XXm', ...}
        classes: 类别列表
        use_floor: 是否绘制地面分割
        use_distance: 是否绘制距离辅助线

    Returns:
        增强后的图像路径
    """
    if not os.path.exists(image_path):
        return image_path

    image = cv2.imread(image_path)
    if image is None:
        return image_path

    if use_floor and mask_path and os.path.exists(mask_path) and classes:
        image = owner.draw_floor_from_saved_mask(image, mask_path, classes)

    if use_distance and distance_dict:
        image = owner.draw_distance_on_action_view(image, distance_dict)

    base_path = os.path.splitext(image_path)[0]
    enhanced_path = f"{base_path}_enhanced.png"
    cv2.imwrite(enhanced_path, image)
    return enhanced_path

def save_detection(owner,
                  step: int,
                  episode_id: int,
                  detection_vis: np.ndarray,
                  phase: str = "action") -> str:
    """
    保存检测可视化

    Args:
        step: 步数
        episode_id: episode ID
        detection_vis: 检测可视化图像
        phase: 阶段标识 ("initial", "action1a", "verify1a" 等)

    Returns:
        save_path: 保存路径
    """
    if detection_vis is None:
        return None

    # 简化路径：data/manual_navigation/episode_X/detection/step_XXXX.png
    episode_dir = owner._create_episode_directories(episode_id)
    save_path = os.path.join(episode_dir, 'detection', f'step_{step:04d}_{phase}.png')
    cv2.imwrite(save_path, detection_vis)
    return save_path
