"""Episode-level path and result helpers for SpaceVLN runs."""
import os
import json
import math
from typing import Dict, List, Optional
from datetime import datetime


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


def get_episode_detail_dir(dump_dir: str, episode_id: int) -> str:
    return os.path.join(
        get_episode_bucket_dir(get_episode_detail_root(dump_dir), episode_id),
        f"episode_{int(episode_id)}",
    )


def get_episode_detail_legacy_dir(dump_dir: str, episode_id: int) -> str:
    return os.path.join(get_episode_detail_root(dump_dir), f"episode_{int(episode_id)}")


def get_episode_detail_path_candidates(dump_dir: str, episode_id: int) -> List[str]:
    candidates = [
        get_episode_detail_dir(dump_dir, episode_id),
        get_episode_detail_legacy_dir(dump_dir, episode_id),
    ]
    deduped: List[str] = []
    for path in candidates:
        if path not in deduped:
            deduped.append(path)
    return deduped


def get_log_root(dump_dir: str) -> str:
    return os.path.join(dump_dir, LOG_DIR_NAME)


def get_episode_log_path(dump_dir: str, episode_id: int) -> str:
    return os.path.join(
        get_episode_bucket_dir(get_log_root(dump_dir), episode_id),
        f"episode_{int(episode_id)}.json",
    )


def get_episode_log_legacy_path(dump_dir: str, episode_id: int) -> str:
    return os.path.join(get_log_root(dump_dir), f"episode_{int(episode_id)}.json")


def get_episode_log_path_candidates(dump_dir: str, episode_id: int) -> List[str]:
    candidates = [
        get_episode_log_path(dump_dir, episode_id),
        get_episode_log_legacy_path(dump_dir, episode_id),
    ]
    deduped: List[str] = []
    for path in candidates:
        if path not in deduped:
            deduped.append(path)
    return deduped


def iter_all_episode_log_paths(dump_dir: str) -> List[str]:
    log_root = get_log_root(dump_dir)
    if not os.path.exists(log_root):
        return []

    matched_paths: List[str] = []
    for current_root, dirnames, filenames in os.walk(log_root):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.startswith("episode_") and filename.endswith(".json"):
                matched_paths.append(os.path.join(current_root, filename))
    return matched_paths


class SaveManager:
    """Lightweight episode output helper."""

    COMMON_RESULT_FIELDS = (
        "episode_id",
        "instruction",
        "total_steps",
        "subtask_count",
        "episode_duration_s",
        "episode_duration_including_failed_s",
        "episode_duration_excluding_failed_s",
        "failed_api_total_duration_s",
        "failed_retry_wait_duration_s",
        "failed_wasted_duration_s",
        "ne",
        "osr",
        "sr",
        "spl",
        "ndtw",
        "path_length",
        "oracle_navigation_error",
        "oracle_spl",
        "timestamp",
    )
    LOG_RESULT_FIELDS = (
        "thinking_api_count",
        "thinking_api_failed_count",
        "thinking_api_avg_duration_s",
        "thinking_api_total_duration_s",
        "thinking_api_failed_total_duration_s",
        "action_api_count",
        "action_api_failed_count",
        "action_api_avg_duration_s",
        "action_api_total_duration_s",
        "action_api_failed_total_duration_s",
        "api_total_duration_s",
        "api_failed_total_duration_s",
    )
    FULL_RESULT_FIELDS = (
        "success",
        "distance_to_goal",
        "oracle_success",
        "detected_objects",
        "subtask_history",
        "thinking_api_calls",
        "action_api_calls",
        "thinking_api_summary",
        "action_api_summary",
        "api_summary",
    )
    API_SUMMARY_FIELDS = (
        "count",
        "failure_count",
        "avg_duration_s",
        "total_duration_s",
        "failed_total_duration_s",
    )
    
    def __init__(self, dump_dir: str, episode_id: int):
        """
        初始化保存管理器
        
        Args:
            dump_dir: 数据保存根目录
            episode_id: Episode ID
        """
        self.dump_dir = dump_dir
        self.episode_id = episode_id
        self.detail_dir = get_episode_detail_root(dump_dir)
        self.episode_dir = get_episode_detail_dir(dump_dir, episode_id)
        self.records_dir = os.path.join(self.episode_dir, "records")  # 统一的记录目录
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
        summary = payload.get(key) if isinstance(payload, dict) else None
        return cls._has_required_keys(summary, cls.API_SUMMARY_FIELDS)

    @classmethod
    def result_has_sr1(cls, result: Optional[Dict]) -> bool:
        if not isinstance(result, dict) or not result:
            return False
        try:
            return cls._safe_int(result.get("sr", 0), 0) == 1
        except Exception:
            return False

    @classmethod
    def is_complete_log_result(cls, result: Optional[Dict]) -> bool:
        return (
            cls._has_required_keys(result, cls.COMMON_RESULT_FIELDS)
            and cls._has_required_keys(result, cls.LOG_RESULT_FIELDS)
        )

    @classmethod
    def is_complete_full_result(cls, result: Optional[Dict]) -> bool:
        if not (
            cls._has_required_keys(result, cls.COMMON_RESULT_FIELDS)
            and cls._has_required_keys(result, cls.FULL_RESULT_FIELDS)
        ):
            return False
        return all(
            cls._has_complete_api_summary(result, key)
            for key in ("thinking_api_summary", "action_api_summary", "api_summary")
        )

    @classmethod
    def is_complete_saved_result(cls, result: Optional[Dict]) -> bool:
        return cls.is_complete_log_result(result) or cls.is_complete_full_result(result)

    @classmethod
    def should_replace_incomplete_non_sr1_best(
        cls,
        new_result: Optional[Dict],
        new_log_result: Optional[Dict],
        existing_best_result: Optional[Dict],
        existing_best_log: Optional[Dict],
    ) -> bool:
        if not (
            cls.is_complete_saved_result(new_result)
            and cls.is_complete_saved_result(new_log_result)
        ):
            return False

        existing_candidates = [
            candidate
            for candidate in (existing_best_result, existing_best_log)
            if candidate
        ]
        if not existing_candidates:
            return False
        if any(cls.result_has_sr1(candidate) for candidate in existing_candidates):
            return False
        return any(not cls.is_complete_saved_result(candidate) for candidate in existing_candidates)

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

        # 排序规则：SR > OSR > SPL > nDTW > NE更小 > 总时间更短 > oracle SPL > path更短 > steps更少
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
        path = os.path.join(self.episode_dir, "thinking", f"subtask_{int(subtask_count)}")
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    def action_subtask_dir(self, subtask_id: str, create: bool = False) -> str:
        path = os.path.join(self.episode_dir, "action", f"subtask_{subtask_id}")
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
        """保存路径点与房间区域记忆到 records/。"""
        payload = {
            "instruction": instruction,
            "current_step": int(current_step),
            "waypoint_memory": waypoint_memory,
            "timestamp": datetime.now().isoformat(),
        }
        save_path = os.path.join(self.records_dir, "waypoint_memory.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    
    def save_result(self, result: Dict):
        """
        保存最终结果，并维护按episode的最佳结果:
        1. detail/<bucket>/episode_xxx/records/result_latest.json (本次运行结果)
        2. detail/<bucket>/episode_xxx/records/result.json (该episode当前最佳结果)
        3. log/<bucket>/episode_xxx.json (该episode当前最佳结果，供结果报告程序使用)
        """
        latest_result_path = os.path.join(self.records_dir, "result_latest.json")
        with open(latest_result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 保存到records/目录（最佳结果）
        result_path = os.path.join(self.records_dir, "result.json")
        existing_best_result = self._load_json_if_exists(result_path)

        log_dir = os.path.dirname(get_episode_log_path(self.dump_dir, self.episode_id))
        os.makedirs(log_dir, exist_ok=True)
        log_path = get_episode_log_path(self.dump_dir, self.episode_id)

        log_result = {
            'episode_id': result['episode_id'],
            'instruction': result.get('instruction', ''),
            'total_steps': result.get('total_steps', 0),
            'subtask_count': result.get('subtask_count', 0),
            'episode_duration_s': result.get('episode_duration_s', 0.0),
            'episode_duration_including_failed_s': result.get(
                'episode_duration_including_failed_s',
                result.get('episode_duration_s', 0.0),
            ),
            'episode_duration_excluding_failed_s': result.get(
                'episode_duration_excluding_failed_s',
                result.get('episode_duration_s', 0.0),
            ),
            'failed_api_total_duration_s': result.get('failed_api_total_duration_s', 0.0),
            'failed_retry_wait_duration_s': result.get('failed_retry_wait_duration_s', 0.0),
            'failed_wasted_duration_s': result.get('failed_wasted_duration_s', result.get('failed_api_total_duration_s', 0.0)),

            'ne': result.get('ne', -1),
            'osr': result.get('osr', 0),
            'sr': result.get('sr', 0),
            'spl': result.get('spl', 0.0),
            'ndtw': result.get('ndtw', 0.0),

            'thinking_api_count': (result.get('thinking_api_summary') or {}).get('count', 0),
            'thinking_api_failed_count': (result.get('thinking_api_summary') or {}).get('failure_count', 0),
            'thinking_api_avg_duration_s': (result.get('thinking_api_summary') or {}).get('avg_duration_s', 0.0),
            'thinking_api_total_duration_s': (result.get('thinking_api_summary') or {}).get('total_duration_s', 0.0),
            'thinking_api_failed_total_duration_s': (result.get('thinking_api_summary') or {}).get('failed_total_duration_s', 0.0),
            'action_api_count': (result.get('action_api_summary') or {}).get('count', 0),
            'action_api_failed_count': (result.get('action_api_summary') or {}).get('failure_count', 0),
            'action_api_avg_duration_s': (result.get('action_api_summary') or {}).get('avg_duration_s', 0.0),
            'action_api_total_duration_s': (result.get('action_api_summary') or {}).get('total_duration_s', 0.0),
            'action_api_failed_total_duration_s': (result.get('action_api_summary') or {}).get('failed_total_duration_s', 0.0),
            'api_total_duration_s': (result.get('api_summary') or {}).get('total_duration_s', 0.0),
            'api_failed_total_duration_s': (result.get('api_summary') or {}).get('failed_total_duration_s', 0.0),
            'path_length': result.get('path_length', 0.0),
            'oracle_navigation_error': result.get('oracle_navigation_error', float('inf')),
            'oracle_spl': result.get('oracle_spl', 0.0),
            'timestamp': result.get('timestamp', datetime.now().isoformat()),
        }

        existing_best_log = self._load_json_if_exists(log_path)
        compare_baseline = None
        for candidate in (existing_best_result, existing_best_log):
            if not candidate:
                continue
            if compare_baseline is None or self._is_better_result(candidate, compare_baseline):
                compare_baseline = candidate
        existing_best_has_sr1 = any(
            self.result_has_sr1(candidate)
            for candidate in (existing_best_result, existing_best_log)
            if candidate
        )
        should_update_best = (
            (not existing_best_has_sr1) and (
                compare_baseline is None
                or self.should_replace_incomplete_non_sr1_best(
                    new_result=result,
                    new_log_result=log_result,
                    existing_best_result=existing_best_result,
                    existing_best_log=existing_best_log,
                )
                or self._is_better_result(log_result, compare_baseline)
            )
        )

        if should_update_best:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_result, f, indent=2, ensure_ascii=False)

        if existing_best_has_sr1 and compare_baseline is not None:
            status = "kept(sr=1 locked)"
        else:
            status = "updated" if should_update_best else "kept"

        print(
            f"[Save] latest={latest_result_path} | best={status} | log={log_path}"
        )

        return log_path
    
