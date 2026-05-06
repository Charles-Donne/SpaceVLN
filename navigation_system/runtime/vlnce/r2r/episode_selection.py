"""Dataset-driven episode selection and resume filtering helpers."""

import gzip
import json
import os
import random
import time
from typing import Any, Dict, List, Set

from navigation_system.runtime.storage.artifacts import (
    SaveManager,
    get_episode_detail_path_candidates,
    get_episode_log_path_candidates,
)
from navigation_system.runtime.episode_io import load_json_if_exists


MIN_EPISODE_ID = 1
MAX_EPISODE_ID = 1800
_DATASET_EPISODE_ID_CACHE: Dict[str, List[int]] = {}


def _get_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_dataset_path(data_path: str) -> str:
    if not data_path:
        return ""
    if os.path.isabs(data_path) and os.path.exists(data_path):
        return data_path

    candidates = [
        os.path.abspath(data_path),
        os.path.abspath(os.path.join(os.getcwd(), data_path)),
        os.path.abspath(os.path.join(_get_repo_root(), data_path)),
    ]
    deduped: List[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    for candidate in deduped:
        if os.path.exists(candidate):
            return candidate
    return deduped[0] if deduped else ""


def _dataset_roles(config) -> List[str]:
    roles = list(getattr(config.TASK_CONFIG.DATASET, "ROLES", []) or [])
    if not roles or "*" in roles:
        return ["guide", "follower"]
    return [str(role) for role in roles]


def _dataset_languages(config) -> List[str]:
    languages = list(getattr(config.TASK_CONFIG.DATASET, "LANGUAGES", []) or [])
    return [str(language) for language in languages]


def _iter_dataset_paths(data_path: str, split: str, roles: List[str]) -> List[str]:
    paths: List[str] = []
    if "{role}" in data_path:
        for role in roles:
            paths.append(data_path.format(split=split, role=role))
    else:
        paths.append(data_path.format(split=split))
    return paths


def _load_dataset_episode_ids(config) -> List[int]:
    data_path = str(getattr(config.TASK_CONFIG.DATASET, "DATA_PATH", "") or "").strip()
    split = str(getattr(config.TASK_CONFIG.DATASET, "SPLIT", "") or "").strip()
    roles = _dataset_roles(config)
    languages = _dataset_languages(config)
    language_filter_enabled = bool(languages) and "*" not in languages
    allowed_languages = set(languages)
    raw_paths = _iter_dataset_paths(data_path, split, roles)
    resolved_paths = [_resolve_dataset_path(path) for path in raw_paths]
    resolved_paths = [path for path in resolved_paths if path]
    cache_key = "|".join(resolved_paths)
    if not cache_key:
        return []
    if cache_key in _DATASET_EPISODE_ID_CACHE:
        return list(_DATASET_EPISODE_ID_CACHE[cache_key])

    episode_ids: Set[int] = set()
    for resolved_path in resolved_paths:
        if not os.path.exists(resolved_path):
            continue
        try:
            opener = gzip.open if resolved_path.endswith(".gz") else open
            with opener(resolved_path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
            episodes = payload.get("episodes", []) if isinstance(payload, dict) else payload
            if not isinstance(episodes, list):
                episodes = []
            for item in episodes:
                if not isinstance(item, dict):
                    continue
                if language_filter_enabled:
                    instruction = item.get("instruction")
                    episode_language = ""
                    if isinstance(instruction, dict):
                        episode_language = str(instruction.get("language") or "")
                    if episode_language not in allowed_languages:
                        continue
                try:
                    episode_ids.add(int(item.get("episode_id")))
                except Exception:
                    continue
        except Exception:
            continue

    sorted_episode_ids = sorted(episode_ids)
    _DATASET_EPISODE_ID_CACHE[cache_key] = list(sorted_episode_ids)
    return sorted_episode_ids


def get_available_episode_ids(config) -> List[int]:
    episode_ids = _load_dataset_episode_ids(config)
    if episode_ids:
        return episode_ids
    return list(range(MIN_EPISODE_ID, MAX_EPISODE_ID + 1))


def resolve_episode_ids(args, config) -> List[int]:
    available_episode_ids = get_available_episode_ids(config)
    available_episode_set = set(available_episode_ids)
    min_episode_id = int(available_episode_ids[0]) if available_episode_ids else MIN_EPISODE_ID
    max_episode_id = int(available_episode_ids[-1]) if available_episode_ids else MAX_EPISODE_ID

    if args.episode_ids:
        episode_ids = [int(x.strip()) for x in args.episode_ids.split(",")]
        invalid_ids = [eid for eid in episode_ids if eid not in available_episode_set]
        if invalid_ids:
            print(
                f"\n❌ Error: episode ids are outside the valid range "
                f"[{min_episode_id}, {max_episode_id}] or missing from the current dataset: {invalid_ids}"
            )
            return []
        return episode_ids

    if getattr(args, "ordered", False):
        start_index = max(1, int(getattr(args, "start_index", 1) or 1))
        start_offset = start_index - 1
        if start_offset >= len(available_episode_ids):
            print(
                f"\n❌ Error: ordered start index {start_index} exceeds "
                f"the available dataset count {len(available_episode_ids)}"
            )
            return []
        num_to_select = min(args.num_episodes, len(available_episode_ids) - start_offset)
        if num_to_select == 0:
            print("\n❌ Error: requested episode count is zero")
            return []
        return list(available_episode_ids[start_offset : start_offset + num_to_select])

    if args.random:
        random_seed = int(time.time() * 1000) % (2 ** 32)
        random.seed(random_seed)
        num_to_sample = min(args.num_episodes, len(available_episode_ids))
        if num_to_sample == 0:
            print("\n❌ Error: requested episode count is zero")
            return []
        return random.sample(list(available_episode_ids), num_to_sample)

    start_id = args.episode_id
    end_id = args.episode_id + args.num_episodes - 1

    if start_id < min_episode_id:
        print(f"\n❌ Error: start episode id {start_id} is smaller than the dataset minimum {min_episode_id}")
        print(f"   Suggested value: --episode-id {min_episode_id}")
        return []

    if start_id > max_episode_id:
        print(f"\n❌ Error: start episode id {start_id} exceeds the dataset maximum {max_episode_id}")
        return []

    clipped_end_id = min(end_id, max_episode_id)
    if clipped_end_id < end_id:
        print(
            f"\n⚠️  Requested range {start_id}-{end_id} exceeds the dataset end; "
            f"clipped to {start_id}-{clipped_end_id}"
        )

    episode_ids = [
        episode_id
        for episode_id in available_episode_ids
        if start_id <= episode_id <= clipped_end_id
    ]
    if not episode_ids:
        print("\n❌ Error: no runnable episodes were found in the requested range")
        return []
    return episode_ids


def _episode_has_existing_sr1(results_dir: str, episode_id: int) -> bool:
    candidate_paths = []
    candidate_paths.extend(get_episode_log_path_candidates(results_dir, episode_id))
    for detail_dir in get_episode_detail_path_candidates(results_dir, episode_id):
        candidate_paths.append(os.path.join(detail_dir, "records", "result.json"))

    deduped_paths: List[str] = []
    for path in candidate_paths:
        if path not in deduped_paths:
            deduped_paths.append(path)

    for path in deduped_paths:
        loaded = load_json_if_exists(path)
        if SaveManager.result_has_complete_sr1(loaded):
            return True
    return False


def filter_episode_ids(args, config, episode_ids: List[int]) -> List[int]:
    if not args.skip_sr1:
        return episode_ids

    results_dir = os.path.abspath(args.results_dir or config.PATHS.RESULTS_DIR or "")
    if not results_dir:
        return episode_ids

    kept_episode_ids: List[int] = []
    for episode_id in episode_ids:
        if not _episode_has_existing_sr1(results_dir, episode_id):
            kept_episode_ids.append(int(episode_id))
    return kept_episode_ids


__all__ = [
    "MAX_EPISODE_ID",
    "MIN_EPISODE_ID",
    "filter_episode_ids",
    "get_available_episode_ids",
    "resolve_episode_ids",
]
