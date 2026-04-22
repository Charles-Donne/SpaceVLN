"""Centralized landmark memory pool for action/thinking runtime."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


DetectedLandmark = Tuple[str, Any]


def _clone_entry(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    return dict(entry)


def _clone_entries(entries: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [_clone_entry(entry) for entry in list(entries or []) if isinstance(entry, dict)]


def _clone_detected_landmarks(items: Optional[Sequence[Any]]) -> List[DetectedLandmark]:
    cloned: List[DetectedLandmark] = []
    for item in list(items or []):
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            cloned.append((str(item[0]), item[1]))
    return cloned


def _clone_dist_map(
    dist_map: Optional[Dict[str, Tuple[Any, Any]]],
) -> Dict[str, Tuple[Any, Any]]:
    cloned: Dict[str, Tuple[Any, Any]] = {}
    for key, value in dict(dist_map or {}).items():
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            cloned[str(key)] = (value[0], value[1])
    return cloned


def _clone_dist_map_multi(
    dist_map_multi: Optional[Dict[str, Sequence[Tuple[Any, Any]]]],
) -> Dict[str, List[Tuple[Any, Any]]]:
    cloned: Dict[str, List[Tuple[Any, Any]]] = {}
    for key, values in dict(dist_map_multi or {}).items():
        rows: List[Tuple[Any, Any]] = []
        for value in list(values or []):
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                rows.append((value[0], value[1]))
        cloned[str(key)] = rows
    return cloned


def _merge_world_instance_confidence(
    existing: Optional[Dict[str, Any]],
    incoming: Dict[str, Any],
) -> Dict[str, Any]:
    if not existing:
        return _clone_entry(incoming)

    merged = dict(existing)
    merged.update(incoming)
    try:
        existing_conf = float(existing.get("confidence", 0.0))
    except (TypeError, ValueError):
        existing_conf = 0.0
    try:
        incoming_conf = float(incoming.get("confidence", 0.0))
    except (TypeError, ValueError):
        incoming_conf = 0.0
    merged["confidence"] = max(existing_conf, incoming_conf)
    return merged


@dataclass
class LandmarkMemoryPool:
    """Single owner for landmark instance memory plus prompt-facing derived views."""

    world_instances: List[Dict[str, Any]] = field(default_factory=list)
    latest_visible_entries: List[Dict[str, Any]] = field(default_factory=list)
    latest_prompt_entries: List[Dict[str, Any]] = field(default_factory=list)
    latest_dist_map: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    latest_dist_map_multi: Dict[str, List[Tuple[Any, Any]]] = field(default_factory=dict)
    detected_by_step: Dict[int, List[DetectedLandmark]] = field(default_factory=dict)
    visible_entries_by_step: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    prompt_entries_by_step: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)

    def reset_episode(self) -> None:
        self.reset_subtask()

    def reset_subtask(self) -> None:
        self.world_instances.clear()
        self.clear_latest_detection_cache()
        self.detected_by_step.clear()
        self.visible_entries_by_step.clear()
        self.prompt_entries_by_step.clear()

    def clear_latest_detection_cache(self) -> None:
        self.latest_dist_map.clear()
        self.latest_dist_map_multi.clear()
        self.clear_latest_prompt_view()

    def clear_latest_prompt_view(self) -> None:
        self.latest_visible_entries.clear()
        self.latest_prompt_entries.clear()

    def set_world_instances(self, instances: Optional[Sequence[Dict[str, Any]]]) -> None:
        merged_by_uid: Dict[int, Dict[str, Any]] = {}
        passthrough: List[Dict[str, Any]] = []
        for raw in list(instances or []):
            if not isinstance(raw, dict):
                continue
            entry = dict(raw)
            uid = entry.get("instance_uid")
            try:
                uid_int = int(uid) if uid is not None else None
            except (TypeError, ValueError):
                uid_int = None
            if uid_int is None:
                passthrough.append(entry)
                continue
            merged_by_uid[uid_int] = _merge_world_instance_confidence(
                merged_by_uid.get(uid_int),
                entry,
            )
        self.world_instances = list(merged_by_uid.values()) + passthrough

    def set_latest_distance_maps(
        self,
        dist_map: Optional[Dict[str, Tuple[Any, Any]]] = None,
        dist_map_multi: Optional[Dict[str, Sequence[Tuple[Any, Any]]]] = None,
    ) -> None:
        self.latest_dist_map = _clone_dist_map(dist_map)
        self.latest_dist_map_multi = _clone_dist_map_multi(dist_map_multi)

    def set_latest_prompt_entries(
        self,
        visible_entries: Optional[Sequence[Dict[str, Any]]] = None,
        prompt_entries: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        self.latest_visible_entries = _clone_entries(visible_entries)
        self.latest_prompt_entries = _clone_entries(prompt_entries)

    def record_latest_view(
        self,
        *,
        visible_entries: Optional[Sequence[Dict[str, Any]]] = None,
        prompt_entries: Optional[Sequence[Dict[str, Any]]] = None,
        dist_map: Optional[Dict[str, Tuple[Any, Any]]] = None,
        dist_map_multi: Optional[Dict[str, Sequence[Tuple[Any, Any]]]] = None,
    ) -> None:
        self.set_latest_prompt_entries(
            visible_entries=visible_entries,
            prompt_entries=prompt_entries,
        )
        self.set_latest_distance_maps(
            dist_map=dist_map,
            dist_map_multi=dist_map_multi,
        )

    def record_step(
        self,
        step_idx: int,
        detected_landmarks: Optional[Sequence[Any]],
        visible_entries: Optional[Sequence[Dict[str, Any]]] = None,
        prompt_entries: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        step_key = int(step_idx)
        self.detected_by_step[step_key] = _clone_detected_landmarks(detected_landmarks)
        self.visible_entries_by_step[step_key] = _clone_entries(
            visible_entries if visible_entries is not None else self.latest_visible_entries
        )
        self.prompt_entries_by_step[step_key] = _clone_entries(
            prompt_entries if prompt_entries is not None else self.latest_prompt_entries
        )

    def has_step(self, step_idx: int) -> bool:
        return int(step_idx) in self.detected_by_step

    def get_world_instances(self) -> List[Dict[str, Any]]:
        return _clone_entries(self.world_instances)

    def get_latest_visible_entries(self) -> List[Dict[str, Any]]:
        return _clone_entries(self.latest_visible_entries)

    def get_latest_prompt_entries(self) -> List[Dict[str, Any]]:
        return _clone_entries(self.latest_prompt_entries)

    def get_latest_dist_map(self) -> Dict[str, Tuple[Any, Any]]:
        return _clone_dist_map(self.latest_dist_map)

    def get_latest_dist_map_multi(self) -> Dict[str, List[Tuple[Any, Any]]]:
        return _clone_dist_map_multi(self.latest_dist_map_multi)

    def get_step_detected(self, step_idx: int) -> List[DetectedLandmark]:
        return _clone_detected_landmarks(self.detected_by_step.get(int(step_idx), []))

    def get_step_visible_entries(self, step_idx: int) -> List[Dict[str, Any]]:
        return _clone_entries(self.visible_entries_by_step.get(int(step_idx), []))

    def get_step_prompt_entries(self, step_idx: int) -> List[Dict[str, Any]]:
        return _clone_entries(self.prompt_entries_by_step.get(int(step_idx), []))

    def set_detected_by_step(
        self,
        history: Optional[Dict[int, Sequence[Any]]],
    ) -> None:
        self.detected_by_step = {
            int(step_idx): _clone_detected_landmarks(entries)
            for step_idx, entries in dict(history or {}).items()
        }

    def set_visible_entries_by_step(
        self,
        history: Optional[Dict[int, Sequence[Dict[str, Any]]]],
    ) -> None:
        self.visible_entries_by_step = {
            int(step_idx): _clone_entries(entries)
            for step_idx, entries in dict(history or {}).items()
        }

    def set_prompt_entries_by_step(
        self,
        history: Optional[Dict[int, Sequence[Dict[str, Any]]]],
    ) -> None:
        self.prompt_entries_by_step = {
            int(step_idx): _clone_entries(entries)
            for step_idx, entries in dict(history or {}).items()
        }
