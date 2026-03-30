"""Episode-level path and result helpers for SpaceVLN runs."""
import os
import json
import math
from typing import Dict, Optional
from datetime import datetime


class SaveManager:
    """Lightweight episode output helper."""
    
    def __init__(self, dump_dir: str, episode_id: int):
        """
        初始化保存管理器
        
        Args:
            dump_dir: 数据保存根目录
            episode_id: Episode ID
        """
        self.dump_dir = dump_dir
        self.episode_id = episode_id
        self.episode_dir = os.path.join(dump_dir, f"episode_{episode_id}")
        self.records_dir = os.path.join(self.episode_dir, "records")  # 统一的记录目录
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
            )

        success = self._safe_int(result.get('sr', 0), 0)
        spl = self._safe_float(result.get('spl', 0.0), 0.0)
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

        # 排序规则：SR > OSR > SPL > nDTW > NE更小 > oracle SPL > path更短 > steps更少
        return (
            success,
            oracle_success,
            spl,
            ndtw,
            -dtg,
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
        1. episode_xxx/records/result_latest.json (本次运行结果)
        2. episode_xxx/records/result.json (该episode当前最佳结果)
        3. log/episode_xxx.json (该episode当前最佳结果，供结果报告程序使用)
        """
        latest_result_path = os.path.join(self.records_dir, "result_latest.json")
        with open(latest_result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 保存到records/目录（最佳结果）
        result_path = os.path.join(self.records_dir, "result.json")
        existing_best_result = self._load_json_if_exists(result_path)

        log_dir = os.path.join(self.dump_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"episode_{self.episode_id}.json")

        log_result = {
            'episode_id': result['episode_id'],
            'instruction': result.get('instruction', ''),
            'total_steps': result.get('total_steps', 0),
            'subtask_count': result.get('subtask_count', 0),
            'episode_duration_s': result.get('episode_duration_s', 0.0),

            'ne': result.get('ne', -1),
            'osr': result.get('osr', 0),
            'sr': result.get('sr', 0),
            'spl': result.get('spl', 0.0),
            'ndtw': result.get('ndtw', 0.0),

            'thinking_api_count': (result.get('thinking_api_summary') or {}).get('count', 0),
            'thinking_api_avg_duration_s': (result.get('thinking_api_summary') or {}).get('avg_duration_s', 0.0),
            'thinking_api_total_duration_s': (result.get('thinking_api_summary') or {}).get('total_duration_s', 0.0),
            'action_api_count': (result.get('action_api_summary') or {}).get('count', 0),
            'action_api_avg_duration_s': (result.get('action_api_summary') or {}).get('avg_duration_s', 0.0),
            'action_api_total_duration_s': (result.get('action_api_summary') or {}).get('total_duration_s', 0.0),
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
        should_update_best = (
            compare_baseline is None
            or self._is_better_result(log_result, compare_baseline)
        )

        if should_update_best:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_result, f, indent=2, ensure_ascii=False)
        
        status = "updated" if should_update_best else "kept"
        print(
            f"[Save] latest={latest_result_path} | best={status} | log={log_path}"
        )
        
        return log_path
    
