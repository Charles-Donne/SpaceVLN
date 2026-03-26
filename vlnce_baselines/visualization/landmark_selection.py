"""Auto-extracted helper module from MapVisualizer for clearer separation of concerns."""
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vlnce_baselines.config.core.constants import (
    detection_visible_topk,
    landmark_duplicate_angle_diff_deg,
    landmark_duplicate_iou_loose,
    landmark_duplicate_iou_strict,
    landmark_duplicate_rel_dist_m,
    landmark_edge_depth_keywords,
    landmark_edge_depth_min_gap_m,
    landmark_instance_merge_radius_m,
    landmark_instance_topk,
    local_map_landmark_topk,
)
from vlnce_baselines.config.core.params.actions import (
    ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M,
    ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M,
)

MASK_OUTER_RING_EDGE_BUFFER_PX = 2.0
MASK_OUTER_RING_MAX_WIDTH_PX = 8.0
MASK_OUTER_RING_RANDOM_SAMPLE_COUNT = 24


def _build_outer_ring_sampling_mask(
    mask_2d: np.ndarray,
    depth_img: np.ndarray,
    min_depth: float = 0.02,
    edge_buffer_px: float = MASK_OUTER_RING_EDGE_BUFFER_PX,
    outer_ring_max_width_px: float = MASK_OUTER_RING_MAX_WIDTH_PX,
) -> Optional[np.ndarray]:
    """Build a mask-local sampling band that stays in the outer region but away from exact edges."""
    if mask_2d is None or depth_img is None:
        return None

    if mask_2d.shape != depth_img.shape:
        mask_2d = cv2.resize(
            mask_2d.astype(np.float32),
            (depth_img.shape[1], depth_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_bool = np.asarray(mask_2d > 0.5, dtype=bool)
    valid_mask = mask_bool & np.isfinite(depth_img) & (depth_img > float(min_depth))
    if not np.any(valid_mask):
        return None

    distance_to_edge = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)
    max_distance_px = float(distance_to_edge.max()) if np.any(mask_bool) else 0.0
    buffer_px = float(max(1.0, edge_buffer_px))
    band_limit_px = float(
        min(
            max(float(outer_ring_max_width_px), buffer_px + 1.0),
            max(buffer_px + 1.0, max_distance_px * 0.45),
        )
    )

    sample_mask = valid_mask & (distance_to_edge >= buffer_px)
    if max_distance_px > buffer_px + 1.0:
        sample_mask &= (distance_to_edge <= band_limit_px)

    if np.any(sample_mask):
        return sample_mask

    relaxed_mask = valid_mask & (distance_to_edge >= max(1.0, buffer_px * 0.5))
    if np.any(relaxed_mask):
        return relaxed_mask

    return valid_mask


def _sample_random_mask_coords(
    sample_mask: Optional[np.ndarray],
    sample_count: int = MASK_OUTER_RING_RANDOM_SAMPLE_COUNT,
) -> Tuple[np.ndarray, np.ndarray]:
    if sample_mask is None:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)

    ys, xs = np.nonzero(sample_mask)
    if ys.size == 0:
        return ys.astype(np.int32), xs.astype(np.int32)

    target_count = max(1, int(sample_count or MASK_OUTER_RING_RANDOM_SAMPLE_COUNT))
    if ys.size > target_count:
        rng = np.random.default_rng()
        chosen = rng.choice(ys.size, size=target_count, replace=False)
        ys = ys[chosen]
        xs = xs[chosen]
    return ys.astype(np.int32), xs.astype(np.int32)

def _candidate_distance_m(candidate: Dict[str, Any]) -> float:
    if candidate.get("det_rel_xy") is not None:
        rel_xy = candidate["det_rel_xy"]
        return float(np.hypot(rel_xy[0], rel_xy[1]))
    if (
        candidate.get("world_x_m") is not None and
        candidate.get("world_y_m") is not None and
        candidate.get("current_pose") is not None
    ):
        curr_x_m, curr_y_m, _ = candidate["current_pose"]
        return float(np.hypot(
            float(candidate["world_x_m"]) - float(curr_x_m),
            float(candidate["world_y_m"]) - float(curr_y_m),
        ))
    return 1e9

def _candidate_angle_deg(
    self,
    candidate: Dict[str, Any],
    hfov: float,
) -> float:
    rel_xy = candidate.get("det_rel_xy")
    if rel_xy is not None and np.hypot(rel_xy[0], rel_xy[1]) > 1e-6:
        return float(np.degrees(np.arctan2(rel_xy[1], rel_xy[0])))

    x1, _y1, x2, _y2 = candidate["bbox"]
    w_img = max(1, int(candidate.get("w_img", 1)))
    xc = (w_img - 1) / 2.0
    focal = (w_img / 2.0) / np.tan(np.deg2rad(float(hfov)) / 2.0)
    if focal <= 1e-6:
        return 0.0
    center_x = (float(x1) + float(x2)) / 2.0
    return float(np.degrees(np.arctan2(center_x - xc, focal)))

def _should_merge_detection_candidates(
    self,
    candidate: Dict[str, Any],
    kept_candidate: Dict[str, Any],
    hfov: float,
) -> bool:
    if candidate.get("name") != kept_candidate.get("name"):
        return False

    if self._is_duplicate_detection_candidate(candidate, kept_candidate):
        return True

    rel_xy = candidate.get("det_rel_xy")
    kept_rel_xy = kept_candidate.get("det_rel_xy")
    if rel_xy is None or kept_rel_xy is None:
        return False

    spatial_dist = float(np.hypot(rel_xy[0] - kept_rel_xy[0], rel_xy[1] - kept_rel_xy[1]))
    angle_diff = self._angle_diff_deg(
        self._candidate_angle_deg(candidate, hfov),
        self._candidate_angle_deg(kept_candidate, hfov),
    )
    return (
        spatial_dist <= float(landmark_instance_merge_radius_m) and
        angle_diff <= float(landmark_duplicate_angle_diff_deg)
    )

def _merge_detection_candidate_entries(
    self,
    kept_candidate: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(kept_candidate)
    kept_conf = max(float(kept_candidate.get("confidence", 0.0)), 1e-3)
    cand_conf = max(float(candidate.get("confidence", 0.0)), 1e-3)
    total_conf = kept_conf + cand_conf

    kx1, ky1, kx2, ky2 = kept_candidate["bbox"]
    cx1, cy1, cx2, cy2 = candidate["bbox"]
    merged["bbox"] = (
        min(int(kx1), int(cx1)),
        min(int(ky1), int(cy1)),
        max(int(kx2), int(cx2)),
        max(int(ky2), int(cy2)),
    )
    merged["confidence"] = max(float(kept_candidate.get("confidence", 0.0)), float(candidate.get("confidence", 0.0)))
    merged["raw_index"] = min(int(kept_candidate.get("raw_index", 0)), int(candidate.get("raw_index", 0)))

    kept_rel_xy = kept_candidate.get("det_rel_xy")
    cand_rel_xy = candidate.get("det_rel_xy")
    if kept_rel_xy is not None and cand_rel_xy is not None and total_conf > 1e-6:
        merged["det_rel_xy"] = (
            float((kept_rel_xy[0] * kept_conf + cand_rel_xy[0] * cand_conf) / total_conf),
            float((kept_rel_xy[1] * kept_conf + cand_rel_xy[1] * cand_conf) / total_conf),
        )
    elif cand_rel_xy is not None:
        merged["det_rel_xy"] = cand_rel_xy

    if (
        kept_candidate.get("world_x_m") is not None and
        kept_candidate.get("world_y_m") is not None and
        candidate.get("world_x_m") is not None and
        candidate.get("world_y_m") is not None and
        total_conf > 1e-6
    ):
        merged["world_x_m"] = float(
            (float(kept_candidate["world_x_m"]) * kept_conf + float(candidate["world_x_m"]) * cand_conf) / total_conf
        )
        merged["world_y_m"] = float(
            (float(kept_candidate["world_y_m"]) * kept_conf + float(candidate["world_y_m"]) * cand_conf) / total_conf
        )
    elif candidate.get("world_x_m") is not None and candidate.get("world_y_m") is not None:
        merged["world_x_m"] = float(candidate["world_x_m"])
        merged["world_y_m"] = float(candidate["world_y_m"])

    return merged

def _dedupe_detection_candidates(
    self,
    candidate_entries: List[Dict[str, Any]],
    hfov: float,
    topk: Optional[int] = None,
) -> List[Dict[str, Any]]:
    ranked_candidates = sorted(
        candidate_entries,
        key=lambda item: (-float(item.get("confidence", 0.0)), self._candidate_distance_m(item), int(item.get("raw_index", 0))),
    )
    merged_candidates: List[Dict[str, Any]] = []
    for candidate in ranked_candidates:
        merged_idx = None
        for idx, kept_candidate in enumerate(merged_candidates):
            if self._should_merge_detection_candidates(candidate, kept_candidate, hfov):
                merged_idx = idx
                break
        if merged_idx is None:
            merged_candidates.append(dict(candidate))
        else:
            merged_candidates[merged_idx] = self._merge_detection_candidate_entries(
                merged_candidates[merged_idx],
                candidate,
            )

    merged_candidates.sort(
        key=lambda item: (-float(item.get("confidence", 0.0)), self._candidate_distance_m(item), int(item.get("raw_index", 0))),
    )
    if topk is not None and int(topk) > 0:
        return merged_candidates[:max(1, int(topk))]
    return merged_candidates

def _landmark_instance_uid(inst: Dict[str, Any]) -> Optional[int]:
    try:
        value = inst.get("instance_uid")
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None

def _landmark_instance_rel_xy(inst: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        distance_m = float(inst.get("distance_m"))
        angle_deg = float(inst.get("angle_deg"))
    except (TypeError, ValueError):
        return None
    angle_rad = np.deg2rad(angle_deg)
    return (
        float(distance_m * np.cos(angle_rad)),
        float(distance_m * np.sin(angle_rad)),
    )

def _sort_landmark_instances_for_action(
    landmark_instances: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return sorted(
        (dict(inst) for inst in landmark_instances or []),
        key=lambda item: (
            -float(item.get("confidence", 0.0)),
            float(item.get("distance_m", 1e9)),
            str(item.get("name", "")),
            int(item.get("instance_uid", 1e9) or 1e9),
        ),
    )

def _select_action_landmark_instances(
    self,
    landmark_instances: Sequence[Dict[str, Any]],
    topk: int = local_map_landmark_topk,
) -> List[Dict[str, Any]]:
    ranked = self._sort_landmark_instances_for_action(landmark_instances)
    keep_n = max(1, int(topk))
    selected = ranked[:keep_n]
    output: List[Dict[str, Any]] = []
    for rank, inst in enumerate(selected):
        normalized = dict(inst)
        normalized["selection_rank"] = rank
        normalized["display_id"] = rank + 1
        output.append(normalized)
    return output

def _build_landmark_display_index_lookup(
    self,
    landmark_instances: Sequence[Dict[str, Any]],
) -> Dict[int, int]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for inst in landmark_instances or []:
        cls_name = str(inst.get("name", "") or "")
        uid = self._landmark_instance_uid(inst)
        if not cls_name or uid is None:
            continue
        grouped.setdefault(cls_name, []).append(dict(inst))

    lookup: Dict[int, int] = {}
    for _cls_name, bucket in grouped.items():
        ranked = self._sort_landmark_instances_for_action(bucket)
        for display_idx, inst in enumerate(ranked):
            uid = self._landmark_instance_uid(inst)
            if uid is not None:
                lookup[uid] = int(display_idx)
    return lookup

def _build_landmark_class_totals(
    landmark_instances: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for inst in landmark_instances or []:
        cls_name = str(inst.get("name", "") or "")
        if not cls_name:
            continue
        totals[cls_name] = totals.get(cls_name, 0) + 1
    return totals

def _build_action_landmark_context(
    self,
    landmark_instances: Sequence[Dict[str, Any]],
    topk: int = local_map_landmark_topk,
) -> Dict[str, Any]:
    all_instances = list(landmark_instances or [])
    selected_instances = self._select_action_landmark_instances(
        all_instances,
        topk=topk,
    ) if all_instances else []
    display_lookup_source = all_instances or selected_instances
    return {
        "all_instances": all_instances,
        "selected_instances": selected_instances,
        "display_index_lookup": self._build_landmark_display_index_lookup(display_lookup_source),
        "class_totals": self._build_landmark_class_totals(display_lookup_source),
    }

def _match_candidate_to_world_instance(
    self,
    candidate: Dict[str, Any],
    landmark_instances: Sequence[Dict[str, Any]],
    hfov: float,
) -> Optional[Dict[str, Any]]:
    name = str(candidate.get("name", "") or "")
    if not name:
        return None

    det_rel_xy = candidate.get("det_rel_xy")
    cand_world_x = candidate.get("world_x_m")
    cand_world_y = candidate.get("world_y_m")
    ranked: List[Tuple[float, float, float, int, Dict[str, Any]]] = []

    for inst in landmark_instances or []:
        if str(inst.get("name", "") or "") != name:
            continue

        inst_uid = self._landmark_instance_uid(inst)
        if inst_uid is None:
            continue

        inst_rel_xy = self._landmark_instance_rel_xy(inst)
        if det_rel_xy is not None and inst_rel_xy is not None:
            match_cost = float(np.hypot(
                float(inst_rel_xy[0]) - float(det_rel_xy[0]),
                float(inst_rel_xy[1]) - float(det_rel_xy[1]),
            ))
        elif (
            cand_world_x is not None and cand_world_y is not None and
            inst.get("world_x_m") is not None and inst.get("world_y_m") is not None
        ):
            match_cost = float(np.hypot(
                float(inst["world_x_m"]) - float(cand_world_x),
                float(inst["world_y_m"]) - float(cand_world_y),
            ))
        else:
            match_cost = self._candidate_distance_m(candidate)

        ranked.append((
            float(match_cost),
            float(inst.get("distance_m", 1e9)),
            -float(inst.get("confidence", 0.0)),
            int(inst_uid),
            dict(inst),
        ))

    if not ranked:
        return None

    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return ranked[0][4]

def _estimate_mask_rel_xy(owner,
                          mask_2d: np.ndarray,
                          depth_img: np.ndarray,
                          hfov: float,
                          sample_stride: int = 4,
                          landmark_name: Optional[str] = None,
                          return_profile: bool = False):
    """用 mask+depth 估计目标在 agent 坐标系中的前向/右向位置。"""
    profile = owner._analyze_mask_depth_profile(mask_2d, depth_img, landmark_name=landmark_name)
    sample_mask = profile.get("sample_mask")
    if sample_mask is None or not np.any(sample_mask):
        if return_profile:
            return None, profile
        return None

    sample_count = max(
        MASK_OUTER_RING_RANDOM_SAMPLE_COUNT,
        max(1, int(sample_stride or 1)) * 6,
    )
    ys, xs = _sample_random_mask_coords(sample_mask, sample_count=sample_count)
    if ys.size == 0:
        if return_profile:
            return None, profile
        return None
    depth_vals = depth_img[ys, xs].astype(np.float32)
    if depth_vals.size == 0:
        return None

    _, w_img = depth_img.shape[:2]
    xc = (w_img - 1) / 2.0
    focal = (w_img / 2.0) / np.tan(np.deg2rad(float(hfov)) / 2.0)
    if focal <= 1e-6:
        if return_profile:
            return None, profile
        return None

    right_vals = ((xs.astype(np.float32) - xc) * depth_vals) / float(focal)
    forward_vals = depth_vals
    rel_xy = (float(np.median(forward_vals)), float(np.median(right_vals)))
    if return_profile:
        return rel_xy, profile
    return rel_xy

def _analyze_mask_depth_profile(
    self,
    mask_2d: np.ndarray,
    depth_img: np.ndarray,
    landmark_name: Optional[str] = None,
) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "sample_mask": None,
        "is_opening_like": False,
        "used_edge_geometry": False,
        "edge_depth_median": None,
        "interior_depth_median": None,
        "opening_gap_m": None,
        "opening_gap_threshold_m": None,
    }
    if mask_2d is None or depth_img is None:
        return profile

    sample_mask = _build_outer_ring_sampling_mask(
        mask_2d,
        depth_img,
        min_depth=0.02,
    )
    if sample_mask is None or not np.any(sample_mask):
        return profile

    landmark_text = str(landmark_name or "").strip().lower()
    is_opening_like = bool(
        landmark_text and
        any(keyword in landmark_text for keyword in landmark_edge_depth_keywords)
    )
    used_edge_geometry = False
    edge_median = None
    interior_median = None
    opening_gap_m = None
    opening_gap_threshold = None
    ys, xs = _sample_random_mask_coords(sample_mask, sample_count=MASK_OUTER_RING_RANDOM_SAMPLE_COUNT)
    if ys.size > 0:
        sampled_depth = depth_img[ys, xs].astype(np.float32)
        if sampled_depth.size > 0:
            # Keep legacy field names for downstream compatibility.
            edge_median = float(np.median(sampled_depth))

    profile.update({
        "sample_mask": sample_mask,
        "is_opening_like": bool(is_opening_like),
        "used_edge_geometry": bool(used_edge_geometry),
        "edge_depth_median": edge_median,
        "interior_depth_median": interior_median,
        "opening_gap_m": opening_gap_m,
        "opening_gap_threshold_m": opening_gap_threshold,
    })
    return profile

def _project_landmark_instances_from_detections(owner,
                                                detections,
                                                labels: Optional[List[str]],
                                                landmark_classes: Optional[List[str]],
                                                depth_meters: Optional[np.ndarray],
                                                current_pose: Optional[Tuple[float, float, float]],
                                                hfov: float,
                                                topk: Optional[int] = None) -> List[Dict[str, Any]]:
    """将每个 landmark 检测实例直接投影为世界坐标实例列表。"""
    if (detections is None or getattr(detections, 'xyxy', None) is None or
            len(detections.xyxy) == 0 or not labels or not landmark_classes or
            depth_meters is None or current_pose is None):
        return []

    canonical = {name.strip().lower(): name for name in landmark_classes}

    per_class: Dict[str, List[Dict[str, Any]]] = {}
    for i in range(len(detections.xyxy)):
        label = labels[i] if i < len(labels) else f"object_{i}"
        parts = label.split()
        label_name = ' '.join(parts[:-1]) if len(parts) > 1 else (parts[0] if parts else "unknown")
        confidence = float(parts[-1]) if len(parts) > 1 else 0.0
        matched_landmark = canonical.get(label_name.strip().lower())
        if matched_landmark is None:
            continue

        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        det_mask = None
        if getattr(detections, 'mask', None) is not None and i < len(detections.mask):
            det_mask = detections.mask[i]
        rel_xy, depth_profile = owner._estimate_mask_rel_xy(
            det_mask,
            depth_meters,
            hfov,
            landmark_name=matched_landmark,
            return_profile=True,
        )
        if rel_xy is None:
            continue

        world_xy = owner._rel_xy_to_world_xy(rel_xy, current_pose)
        if world_xy is None:
            continue
        world_x_m, world_y_m = world_xy

        world_row_px = int(round(world_y_m * 100.0 / owner.resolution))
        world_col_px = int(round(world_x_m * 100.0 / owner.resolution))
        dist_m = float(np.hypot(rel_xy[0], rel_xy[1]))
        rel_bearing = float(np.degrees(np.arctan2(rel_xy[1], rel_xy[0]))) if dist_m > 1e-6 else 0.0

        per_class.setdefault(matched_landmark, []).append({
            "name": matched_landmark,
            "confidence": float(confidence),
            "distance_m": dist_m,
            "angle_deg": float(rel_bearing),
            "world_row_px": world_row_px,
            "world_col_px": world_col_px,
            "world_x_m": float(world_x_m),
            "world_y_m": float(world_y_m),
            "bbox": (x1, y1, x2, y2),
            "det_rel_xy": (float(rel_xy[0]), float(rel_xy[1])),
            "is_opening_like": bool(depth_profile.get("is_opening_like", False)),
            "used_edge_geometry": bool(depth_profile.get("used_edge_geometry", False)),
            "opening_gap_m": depth_profile.get("opening_gap_m"),
            "edge_depth_median": depth_profile.get("edge_depth_median"),
            "interior_depth_median": depth_profile.get("interior_depth_median"),
            "stop_distance_m": (
                float(ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M)
                if bool(depth_profile.get("is_opening_like", False))
                else float(ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M)
            ),
            "observation_count": 1,
            "weight_sum": max(float(confidence), 1e-3),
        })

    projected_instances: List[Dict[str, Any]] = []
    for cls_name, candidates in per_class.items():
        selected = owner._dedupe_detection_candidates(candidates, hfov=hfov, topk=topk)

        for inst_idx, item in enumerate(selected):
            item = dict(item)
            item.pop("bbox", None)
            item.pop("det_rel_xy", None)
            item["instance_idx"] = inst_idx
            projected_instances.append(item)

    return projected_instances

def _merge_landmark_instances_world(owner,
                                    existing_instances: Optional[List[Dict[str, Any]]],
                                    new_instances: Optional[List[Dict[str, Any]]],
                                    current_pose: Optional[Tuple[float, float, float]],
                                    topk: Optional[int] = landmark_instance_topk,
                                    merge_radius_m: float = landmark_instance_merge_radius_m
                                    ) -> List[Dict[str, Any]]:
    """在同一子任务内累计 landmark 实例，并按世界坐标去重融合。"""
    merged_by_class: Dict[str, List[Dict[str, Any]]] = {}
    next_instance_uid = (
        max(
            [owner._landmark_instance_uid(inst) or 0 for inst in (existing_instances or []) + (new_instances or [])],
            default=0,
        ) + 1
    )

    def _ensure_instance_uid(inst: Dict[str, Any]) -> int:
        nonlocal next_instance_uid
        current_uid = owner._landmark_instance_uid(inst)
        if current_uid is not None:
            inst["instance_uid"] = int(current_uid)
            return int(current_uid)
        inst["instance_uid"] = int(next_instance_uid)
        next_instance_uid += 1
        return int(inst["instance_uid"])

    def _inst_weight(inst: Dict[str, Any]) -> float:
        stored = inst.get("weight_sum")
        if stored is not None:
            try:
                return max(float(stored), 1e-3)
            except (TypeError, ValueError):
                pass
        try:
            return max(float(inst.get("confidence", 0.0)), 1e-3)
        except (TypeError, ValueError):
            return 1e-3

    def _curr_metrics(inst: Dict[str, Any]) -> Tuple[float, float]:
        if current_pose is None or "world_x_m" not in inst or "world_y_m" not in inst:
            return (
                float(inst.get("distance_m", 1e9)),
                float(inst.get("angle_deg", 0.0)),
            )
        curr_x, curr_y, curr_ori = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])
        dx_m = float(inst["world_x_m"]) - curr_x
        dy_m = float(inst["world_y_m"]) - curr_y
        dist_m = float(np.hypot(dx_m, dy_m))
        abs_angle = np.degrees(np.arctan2(dy_m, dx_m)) if dist_m > 1e-6 else curr_ori
        rel_bearing = curr_ori - abs_angle
        rel_bearing = ((rel_bearing + 180.0) % 360.0) - 180.0
        return dist_m, float(rel_bearing)

    def _merge_one(inst: Dict[str, Any]) -> None:
        cls_name = inst.get("name")
        if not cls_name:
            return
        inst = dict(inst)
        _ensure_instance_uid(inst)
        cls_bucket = merged_by_class.setdefault(cls_name, [])
        best_idx = None
        best_dist = None
        if "world_x_m" in inst and "world_y_m" in inst:
            for idx, old in enumerate(cls_bucket):
                if "world_x_m" not in old or "world_y_m" not in old:
                    continue
                dist = float(np.hypot(
                    float(old["world_x_m"]) - float(inst["world_x_m"]),
                    float(old["world_y_m"]) - float(inst["world_y_m"]),
                ))
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx

        if best_idx is not None and best_dist is not None and best_dist <= merge_radius_m:
            old = cls_bucket[best_idx]
            refreshed = dict(old)
            _ensure_instance_uid(refreshed)
            old_weight = _inst_weight(old)
            new_weight = _inst_weight(inst)
            total_weight = old_weight + new_weight

            refreshed.update(inst)
            refreshed["instance_uid"] = owner._landmark_instance_uid(old) or owner._landmark_instance_uid(inst)
            refreshed["confidence"] = max(
                float(old.get("confidence", 0.0)),
                float(inst.get("confidence", 0.0)),
            )
            refreshed["is_opening_like"] = bool(
                old.get("is_opening_like", False) or inst.get("is_opening_like", False)
            )
            refreshed["used_edge_geometry"] = bool(
                old.get("used_edge_geometry", False) or inst.get("used_edge_geometry", False)
            )
            old_gap = old.get("opening_gap_m")
            new_gap = inst.get("opening_gap_m")
            if old_gap is None:
                refreshed["opening_gap_m"] = new_gap
            elif new_gap is None:
                refreshed["opening_gap_m"] = old_gap
            else:
                refreshed["opening_gap_m"] = max(float(old_gap), float(new_gap))
            refreshed["stop_distance_m"] = min(
                float(old.get("stop_distance_m", ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M)),
                float(inst.get("stop_distance_m", ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M)),
            )
            refreshed["observation_count"] = int(old.get("observation_count", 1)) + int(inst.get("observation_count", 1))
            refreshed["weight_sum"] = total_weight

            if (
                "world_x_m" in old and "world_y_m" in old and
                "world_x_m" in inst and "world_y_m" in inst and
                total_weight > 1e-6
            ):
                world_x_m = (
                    float(old["world_x_m"]) * old_weight +
                    float(inst["world_x_m"]) * new_weight
                ) / total_weight
                world_y_m = (
                    float(old["world_y_m"]) * old_weight +
                    float(inst["world_y_m"]) * new_weight
                ) / total_weight
                refreshed["world_x_m"] = float(world_x_m)
                refreshed["world_y_m"] = float(world_y_m)
                refreshed["world_row_px"] = int(round(world_y_m * 100.0 / owner.resolution))
                refreshed["world_col_px"] = int(round(world_x_m * 100.0 / owner.resolution))
            cls_bucket[best_idx] = refreshed
        else:
            normalized = dict(inst)
            _ensure_instance_uid(normalized)
            normalized["observation_count"] = int(normalized.get("observation_count", 1))
            normalized["weight_sum"] = _inst_weight(normalized)
            cls_bucket.append(normalized)

    for inst in existing_instances or []:
        _merge_one(dict(inst))
    for inst in new_instances or []:
        _merge_one(dict(inst))

    merged_instances: List[Dict[str, Any]] = []
    for cls_name, bucket in merged_by_class.items():
        ranked = sorted(
            bucket,
            key=lambda item: (-float(item.get("confidence", 0.0)), _curr_metrics(item)[0]),
        )
        if topk is None or int(topk) <= 0:
            kept = ranked
        else:
            kept = ranked[:max(1, int(topk))]
        kept = sorted(kept, key=lambda item: _curr_metrics(item)[0])
        for inst_idx, item in enumerate(kept):
            dist_m, angle_deg = _curr_metrics(item)
            normalized = dict(item)
            _ensure_instance_uid(normalized)
            normalized["distance_m"] = float(dist_m)
            normalized["angle_deg"] = float(angle_deg)
            normalized["instance_idx"] = inst_idx
            merged_instances.append(normalized)

    return merged_instances

def _world_instance_to_rotated_landmark(owner,
                                        inst: Dict[str, Any],
                                        full_map: np.ndarray,
                                        current_pose: Optional[Tuple[float, float, float]],
                                        crop_offset: Optional[Tuple[int, int]]) -> Optional[Tuple[float, float, str, float, float]]:
    """将世界坐标实例转换到当前旋转后 full_map 像素坐标。"""
    if current_pose is None or crop_offset is None or full_map is None:
        return None

    projector = owner._build_map_projector(full_map, current_pose, crop_offset)
    if projector is None:
        return None

    world_row_px = int(inst["world_row_px"])
    world_col_px = int(inst["world_col_px"])

    rotated = projector.world_to_rotated_pixel(world_row_px, world_col_px)
    if rotated is None:
        return None
    rotated_row, rotated_col = rotated

    if "world_x_m" in inst and "world_y_m" in inst:
        dx_m = float(inst["world_x_m"]) - float(current_pose[0])
        dy_m = float(inst["world_y_m"]) - float(current_pose[1])
        dist_m = float(np.hypot(dx_m, dy_m))
        abs_angle = np.degrees(np.arctan2(dy_m, dx_m)) if dist_m > 1e-6 else float(current_pose[2])
        rel_bearing = float(current_pose[2]) - abs_angle
        rel_bearing = ((rel_bearing + 180.0) % 360.0) - 180.0
    else:
        dist_m = float(inst.get("distance_m", 0.0))
        rel_bearing = float(inst.get("angle_deg", 0.0))

    return (
        float(rotated_col),
        float(rotated_row),
        inst["name"],
        dist_m,
        rel_bearing,
    )

def _build_landmarks_from_instances(owner,
                                    landmark_instances: Optional[List[Dict[str, Any]]],
                                    full_map: np.ndarray,
                                    current_pose: Optional[Tuple[float, float, float]],
                                    crop_offset: Optional[Tuple[int, int]]) -> List[Tuple[float, float, str, float, float]]:
    """把显式实例列表转换成当前渲染使用的 landmark 点。"""
    if not landmark_instances:
        return []

    landmarks: List[Tuple[float, float, str, float, float]] = []
    for inst in landmark_instances:
        converted = owner._world_instance_to_rotated_landmark(inst, full_map, current_pose, crop_offset)
        if converted is not None:
            landmarks.append(converted)
    return landmarks

def _build_local_landmarks_from_instances(owner,
                                          landmark_instances: Optional[List[Dict[str, Any]]],
                                          full_map: np.ndarray,
                                          current_pose: Optional[Tuple[float, float, float]],
                                          crop_offset: Optional[Tuple[int, int]],
                                          topk: int = local_map_landmark_topk
                                          ) -> List[Dict[str, Any]]:
    """Keep only the selected local-map landmarks and preserve their display numbering."""
    if not landmark_instances:
        return []

    projector = owner._build_map_projector(full_map, current_pose, crop_offset)
    if projector is None:
        return []

    ranked_candidates: List[Tuple[Tuple[float, float, float], Dict[str, Any]]] = []
    for inst in landmark_instances:
        converted = owner._world_instance_to_rotated_landmark(inst, full_map, current_pose, crop_offset)
        if converted is None:
            continue
        marker_x, marker_y, cls_name, dist_m, angle_deg = converted
        local_display = projector.rotated_to_local_display(marker_y, marker_x)
        if local_display is None:
            continue

        selection_rank = inst.get("selection_rank")
        try:
            selection_key = float(selection_rank) if selection_rank is not None else 1e9
        except (TypeError, ValueError):
            selection_key = 1e9
        ranked_candidates.append((
            (
                selection_key,
                float(dist_m),
                -float(inst.get("confidence", 0.0)),
            ),
            {
                "marker_x": float(marker_x),
                "marker_y": float(marker_y),
                "name": str(cls_name),
                "distance_m": float(dist_m),
                "angle_deg": float(angle_deg),
                "display_id": int(inst.get("display_id", int(selection_key) + 1 if selection_key < 1e9 else 0) or 0),
                "selection_rank": int(selection_rank) if selection_rank is not None else None,
                "confidence": float(inst.get("confidence", 0.0)),
            },
        ))

    ranked_candidates.sort(key=lambda item: item[0])
    keep_n = max(1, int(topk))
    return [item[1] for item in ranked_candidates[:keep_n]]
