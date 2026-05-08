"""Episode-level path and result helpers for SpaceVLN runs."""
import os
import json
import math
from typing import Dict, List, Optional
from datetime import datetime

from navigation_system.runtime.storage.naming import (
    build_subtask_name,
    build_subtask_name_from_token,
)


DETAIL_DIR_NAME = "detail"
LOG_DIR_NAME = "log"
EPISODE_BUCKET_SIZE = 100


def get_episode_bucket_name(episode_id: int, bucket_size: int = EPISODE_BUCKET_SIZE) -> str:
    episode_id = int(episode_id)
    bucket_size = max(1, int(bucket_size))
    if episode_id <= 0:
        start = 0
        end = bucket_size
    else:
        start = ((episode_id - 1) // bucket_size) * bucket_size + 1
        end = start + bucket_size - 1
    return f"{start}-{end}"


def get_episode_bucket_dir(root_dir: str, episode_id: int, bucket_size: int = EPISODE_BUCKET_SIZE) -> str:
    return os.path.join(root_dir, get_episode_bucket_name(episode_id, bucket_size=bucket_size))


def get_episode_detail_root(dump_dir: str) -> str:
    return os.path.join(dump_dir, DETAIL_DIR_NAME)


def _entry_dir_name(entry_id: int, entry_kind: str = "episode") -> str:
    prefix = str(entry_kind or "episode").strip() or "episode"
    return f"{prefix}_{int(entry_id)}"


def get_episode_detail_dir(
    dump_dir: str,
    episode_id: int,
    *,
    entry_kind: str = "episode",
) -> str:
    return os.path.join(
        get_episode_bucket_dir(get_episode_detail_root(dump_dir), episode_id),
        _entry_dir_name(episode_id, entry_kind=entry_kind),
    )


def get_episode_detail_path_candidates(
    dump_dir: str,
    episode_id: int,
    *,
    entry_kind: str = "episode",
) -> List[str]:
    return [get_episode_detail_dir(dump_dir, episode_id, entry_kind=entry_kind)]


def get_log_root(dump_dir: str) -> str:
    return os.path.join(dump_dir, LOG_DIR_NAME)


def get_episode_log_path(
    dump_dir: str,
    episode_id: int,
    *,
    entry_kind: str = "episode",
) -> str:
    return os.path.join(
        get_episode_bucket_dir(get_log_root(dump_dir), episode_id),
        f"{_entry_dir_name(episode_id, entry_kind=entry_kind)}.json",
    )


def get_episode_log_path_candidates(
    dump_dir: str,
    episode_id: int,
    *,
    entry_kind: str = "episode",
) -> List[str]:
    return [get_episode_log_path(dump_dir, episode_id, entry_kind=entry_kind)]


def iter_all_episode_log_paths(dump_dir: str) -> List[str]:
    log_root = get_log_root(dump_dir)
    if not os.path.exists(log_root):
        return []

    matched_paths: List[str] = []
    for current_root, dirnames, filenames in os.walk(log_root):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.endswith(".json") and (
                filename.startswith("episode_") or filename.startswith("sample_")
            ):
                matched_paths.append(os.path.join(current_root, filename))
    return matched_paths


class SaveManager:
    """Lightweight episode output helper."""

    COMMON_DETAIL_RESULT_FIELDS = (
        "episode_id",
        "instruction",
        "total_steps",
        "subtask_count",
        "episode_duration_s",
        "local_non_api_duration_s",
        "failed_api_total_duration_s",
        "failed_retry_wait_duration_s",
        "failed_wasted_duration_s",
        "success",
        "spl",
        "distance_to_goal",
        "path_length",
    )
    COMMON_DETAIL_RESULT_TAIL_FIELDS = (
        "thinking_api_summary",
        "action_api_summary",
        "timestamp",
        "sr",
        "ne",
    )
    R2RCE_DETAIL_RESULT_FIELDS = (
        "episode_id",
        "sample_index",
        "instruction",
        "total_steps",
        "subtask_count",
        "episode_duration_s",
        "local_non_api_duration_s",
        "failed_api_total_duration_s",
        "failed_retry_wait_duration_s",
        "failed_wasted_duration_s",
        "success",
        "spl",
        "distance_to_goal",
        "ndtw",
        "path_length",
        "oracle_success",
        "oracle_navigation_error",
        "oracle_spl",
        "thinking_api_summary",
        "action_api_summary",
        "timestamp",
        "sr",
        "osr",
        "ne",
    )
    NAVGBENCH_DETAIL_RESULT_FIELDS = (
        "episode_id",
        "sample_index",
        "navgbench_id",
        "instruction",
        "total_steps",
        "subtask_count",
        "episode_duration_s",
        "local_non_api_duration_s",
        "failed_api_total_duration_s",
        "failed_retry_wait_duration_s",
        "failed_wasted_duration_s",
        "success",
        "spl",
        "distance_to_goal",
        "path_length",
        "oracle_success",
        *COMMON_DETAIL_RESULT_TAIL_FIELDS,
        "osr",
    )
    OVON_DETAIL_RESULT_FIELDS = (
        "episode_id",
        "sample_index",
        "instruction",
        "total_steps",
        "subtask_count",
        "episode_duration_s",
        "local_non_api_duration_s",
        "failed_api_total_duration_s",
        "failed_retry_wait_duration_s",
        "failed_wasted_duration_s",
        "success",
        "spl",
        "soft_spl",
        "distance_to_goal",
        "path_length",
        *COMMON_DETAIL_RESULT_TAIL_FIELDS,
    )
    DETAIL_RESULT_FIELD_SETS = {
        "r2rce": R2RCE_DETAIL_RESULT_FIELDS,
        "navgbench": NAVGBENCH_DETAIL_RESULT_FIELDS,
        "ovon": OVON_DETAIL_RESULT_FIELDS,
    }
    LOG_DEFAULT_FIELDS = (
        ("episode_id", None),
        ("instruction", ""),
        ("total_steps", 0),
        ("subtask_count", 0),
        ("episode_duration_s", 0.0),
        ("local_non_api_duration_s", 0.0),
        ("failed_api_total_duration_s", 0.0),
        ("failed_retry_wait_duration_s", 0.0),
        ("failed_wasted_duration_s", 0.0),
        ("ne", -1),
        ("sr", 0),
        ("spl", 0.0),
        ("thinking_api_summary", {}),
        ("action_api_summary", {}),
        ("path_length", 0.0),
        ("timestamp", None),
    )
    LOG_OPTIONAL_FIELD_SETS = {
        "r2rce": (
            "sample_index",
            "osr",
            "ndtw",
            "oracle_success",
            "oracle_navigation_error",
            "oracle_spl",
        ),
        "navgbench": (
            "sample_index",
            "navgbench_id",
            "osr",
            "oracle_success",
        ),
        "ovon": (
            "sample_index",
            "soft_spl",
        ),
    }
    LOG_COMMON_OPTIONAL_FIELDS = (
        "reason",
        "error",
        "gif_path",
        "topdown_path",
    )
    REQUIRED_RESULT_FIELDS = (
        "episode_id",
        "instruction",
        "total_steps",
        "subtask_count",
        "episode_duration_s",
        "failed_api_total_duration_s",
        "failed_retry_wait_duration_s",
        "failed_wasted_duration_s",
        "ne",
        "sr",
        "spl",
        "path_length",
        "timestamp",
    )
    API_SUMMARY_FIELDS = (
        "count",
        "failure_count",
        "avg_duration_s",
        "total_duration_s",
        "failed_total_duration_s",
    )
    
    def __init__(
        self,
        dump_dir: str,
        episode_id: int,
        *,
        storage_entry_id: Optional[int] = None,
        entry_kind: str = "episode",
        save_waypoint_memory: bool = False,
    ):
        """
        Initialize the episode save manager.

        Args:
            dump_dir: Root results directory.
            episode_id: Episode id.
        """
        self.dump_dir = dump_dir
        self.episode_id = episode_id
        self.storage_entry_id = int(storage_entry_id) if storage_entry_id is not None else int(episode_id)
        self.entry_kind = str(entry_kind or "episode").strip() or "episode"
        self.save_waypoint_memory_enabled = bool(save_waypoint_memory)
        self.detail_dir = get_episode_detail_root(dump_dir)
        self.episode_dir = get_episode_detail_dir(
            dump_dir,
            self.storage_entry_id,
            entry_kind=self.entry_kind,
        )
        self.records_dir = os.path.join(self.episode_dir, "records")
        os.makedirs(self.detail_dir, exist_ok=True)
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.records_dir, exist_ok=True)

    @staticmethod
    def _load_json_if_exists(path: str) -> Optional[Dict]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _safe_float(value, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _has_required_keys(cls, payload: Optional[Dict], required_keys) -> bool:
        return isinstance(payload, dict) and all(key in payload for key in tuple(required_keys))

    @classmethod
    def _has_complete_api_summary(cls, payload: Optional[Dict], key: str) -> bool:
        if not isinstance(payload, dict):
            return False
        summary = payload.get(key)
        if cls._has_required_keys(summary, cls.API_SUMMARY_FIELDS):
            return True

        prefix = key.replace("_api_summary", "")
        flat_summary = {
            "count": payload.get(f"{prefix}_api_count"),
            "failure_count": payload.get(f"{prefix}_api_failed_count"),
            "avg_duration_s": payload.get(f"{prefix}_api_avg_duration_s"),
            "total_duration_s": payload.get(f"{prefix}_api_total_duration_s"),
            "failed_total_duration_s": payload.get(f"{prefix}_api_failed_total_duration_s"),
        }
        return cls._has_required_keys(flat_summary, cls.API_SUMMARY_FIELDS)

    @classmethod
    def is_complete_result(cls, result: Optional[Dict]) -> bool:
        if not cls._has_required_keys(result, cls.REQUIRED_RESULT_FIELDS):
            return False
        if not all(
            cls._has_complete_api_summary(result, key)
            for key in ("thinking_api_summary", "action_api_summary")
        ):
            return False

        sr = cls._safe_int(
            (result or {}).get("sr", (result or {}).get("success", 0)),
            0,
        )
        total_steps = cls._safe_int((result or {}).get("total_steps", 0), 0)
        spl = cls._safe_float((result or {}).get("spl", 0.0), 0.0)

        if sr == 1 and total_steps > 0:
            # Historical OVON sample logs often store `path_length=0.0` even
            # when SR/SPL are already valid and complete. Treat positive SPL as
            # the completeness gate for successful results so skip-sr1 works on
            # the saved sample summary logs. Keep SR=1 + SPL=0 as incomplete.
            if spl <= 0.0:
                return False

        return True

    @classmethod
    def result_has_complete_sr1(cls, result: Optional[Dict]) -> bool:
        if not cls.is_complete_result(result):
            return False
        try:
            return cls._safe_int(result.get("sr", 0), 0) == 1
        except Exception:
            return False

    def _result_rank_key(self, result: Optional[Dict]) -> tuple:
        if not result:
            return (
                -1,
                float("-inf"),
                float("-inf"),
                float("-inf"),
                float("-inf"),
                float("-inf"),
                float("-inf"),
                float("-inf"),
                float("-inf"),
            )

        success = self._safe_int(result.get('sr', 0), 0)
        spl = self._safe_float(result.get('spl', 0.0), 0.0)
        episode_duration_s = self._safe_float(
            result.get('episode_duration_s', float('inf')),
            float('inf'),
        )
        if episode_duration_s < 0:
            episode_duration_s = float('inf')
        ndtw = self._safe_float(result.get('ndtw', 0.0), 0.0)
        oracle_success = self._safe_int(result.get('osr', 0), 0)
        oracle_spl = self._safe_float(result.get('oracle_spl', 0.0), 0.0)

        dtg = self._safe_float(result.get('ne', float('inf')), float('inf'))
        if dtg < 0:
            dtg = float('inf')
        path_length = self._safe_float(result.get('path_length', float('inf')), float('inf'))
        if path_length < 0:
            path_length = float('inf')
        total_steps = self._safe_float(result.get('total_steps', float('inf')), float('inf'))
        if total_steps < 0:
            total_steps = float('inf')

        # Rank priority:
        # SR > OSR > SPL > nDTW > lower NE > shorter total time
        # > higher oracle SPL > shorter path > fewer steps
        return (
            success,
            oracle_success,
            spl,
            ndtw,
            -dtg,
            -episode_duration_s,
            oracle_spl,
            -path_length,
            -total_steps,
        )

    def _is_better_result(self, new_result: Dict, old_result: Optional[Dict]) -> bool:
        if not old_result:
            return True
        return self._result_rank_key(new_result) > self._result_rank_key(old_result)
    
    def thinking_subtask_dir(self, subtask_count: int, create: bool = False) -> str:
        path = os.path.join(self.episode_dir, "thinking", build_subtask_name(subtask_count))
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    def action_subtask_dir(self, subtask_id: str, create: bool = False) -> str:
        path = os.path.join(
            self.episode_dir,
            "action",
            build_subtask_name_from_token(subtask_id),
        )
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    def action_step_dir(self, subtask_id: str, step: int, create: bool = False) -> str:
        path = os.path.join(self.action_subtask_dir(subtask_id, create=create), f"step_{int(step)}")
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    def action_info_path(self, subtask_id: str) -> str:
        return os.path.join(self.action_subtask_dir(subtask_id), "info.json")
    
    def save_waypoint_memory(self, waypoint_memory: Dict,
                            instruction: str, current_step: int):
        """Save waypoint / area memory snapshots under `records/`."""
        if not self.save_waypoint_memory_enabled:
            return None
        payload = {
            "instruction": instruction,
            "current_step": int(current_step),
            "waypoint_memory": waypoint_memory,
            "timestamp": datetime.now().isoformat(),
        }
        save_path = os.path.join(self.records_dir, "waypoint_memory.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def _result_benchmark(cls, result: Dict) -> str:
        marker = str(
            (result or {}).get("_benchmark")
            or (result or {}).get("benchmark")
            or ""
        ).strip().lower()
        marker_aliases = {
            "r2r": "r2rce",
            "r2r-ce": "r2rce",
            "r2rce": "r2rce",
            "vlnce": "r2rce",
            "vln-ce": "r2rce",
            "navgbench": "navgbench",
            "gnbench": "navgbench",
            "ovon": "ovon",
            "objectnav": "ovon",
            "object_navigation": "ovon",
        }
        if marker in marker_aliases:
            return marker_aliases[marker]
        if (result or {}).get("navgbench_id") is not None:
            return "navgbench"
        if "soft_spl" in (result or {}):
            return "ovon"
        return "r2rce"

    @staticmethod
    def _copy_present_fields(result: Dict, fields) -> Dict:
        payload = {}
        for key in tuple(fields):
            if key in result:
                payload[key] = result.get(key)
        return payload

    @classmethod
    def _build_detail_result(cls, result: Dict) -> Dict:
        """Keep records/result.json compact and benchmark-specific."""
        benchmark = cls._result_benchmark(result)
        fields = cls.DETAIL_RESULT_FIELD_SETS.get(
            benchmark,
            cls.R2RCE_DETAIL_RESULT_FIELDS,
        )
        return cls._copy_present_fields(result, fields)

    @classmethod
    def _build_log_result(cls, result: Dict) -> Dict:
        """Build the per-entry best summary used by reports and skip-sr1."""
        log_result = {}
        for key, default in cls.LOG_DEFAULT_FIELDS:
            if key == "timestamp":
                default = datetime.now().isoformat()
            if key == "failed_wasted_duration_s":
                default = result.get("failed_api_total_duration_s", 0.0)
            log_result[key] = result.get(key, default)

        benchmark = cls._result_benchmark(result)
        for key in cls.LOG_OPTIONAL_FIELD_SETS.get(benchmark, ()):
            if key in result:
                log_result[key] = result.get(key)
        for key in cls.LOG_COMMON_OPTIONAL_FIELDS:
            if key in result:
                log_result[key] = str(result.get(key, "") or "")
        return log_result
    
    def save_result(self, result: Dict):
        """
        Save the current run result and maintain the per-episode best summary:
        1. `detail/<bucket>/episode_xxx/records/result.json` for this run
        2. `log/<bucket>/episode_xxx.json` for the current best episode result
        """
        result_path = os.path.join(self.records_dir, "result.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(self._build_detail_result(result), f, indent=2, ensure_ascii=False)

        log_dir = os.path.dirname(
            get_episode_log_path(
                self.dump_dir,
                self.storage_entry_id,
                entry_kind=self.entry_kind,
            )
        )
        os.makedirs(log_dir, exist_ok=True)
        log_path = get_episode_log_path(
            self.dump_dir,
            self.storage_entry_id,
            entry_kind=self.entry_kind,
        )

        log_result = self._build_log_result(result)

        existing_best_log = self._load_json_if_exists(log_path)
        compare_baseline = existing_best_log if self.is_complete_result(existing_best_log) else None
        new_best_candidate_is_complete = self.is_complete_result(log_result)
        should_update_best = (
            new_best_candidate_is_complete
            and (
                compare_baseline is None
                or self._is_better_result(log_result, compare_baseline)
            )
        )

        if should_update_best:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_result, f, indent=2, ensure_ascii=False)

        if not new_best_candidate_is_complete:
            status = "kept(new incomplete ignored)"
        else:
            status = "updated" if should_update_best else "kept"

        return log_path
    
