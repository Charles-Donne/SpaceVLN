"""Shared API-config loading and results-path helpers."""

import os
import re
from typing import Any, Dict

import yaml

from navigation_system.config.core.params.api import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
)
from navigation_system.runtime.storage.results_layout import (
    build_default_results_dir_from_api_config,
    build_default_results_family_root,
    build_default_results_root,
    build_model_results_dir_name,
    get_active_provider_and_models,
    resolve_api_config_path,
    resolve_results_dir_path,
    resolve_results_root_path,
)


class APIConfig:
    """Unified provider config with role-based model selection."""

    def __init__(self, config_path: str, role: str = None):
        resolved_path = resolve_api_config_path(config_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"配置文件不存在: {config_path}. "
                f"请在 `navigation_system/config/vlm/` 下创建统一 yaml 配置。"
            )

        self.path = resolved_path
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self.config = raw
        self._role = role or "llm"

        provider = str(raw.get("provider") or "").strip()
        if not provider:
            raise ValueError("统一 API 配置缺少 provider 字段")
        if provider not in raw:
            raise ValueError(f"配置文件中找不到 provider '{provider}' 的配置块")

        provider_config = raw[provider]
        role_name = self._role
        self._api_key = self._resolve_config_string(provider_config.get("api_key", ""))
        self._base_url = self._resolve_config_string(provider_config.get("base_url", "")).rstrip("/")
        self._model = self._resolve_config_string(
            provider_config.get(f"{role_name}_model") or provider_config.get("model", "")
        )
        self._temperature = raw.get("temperature", DEFAULT_TEMPERATURE)
        self._max_tokens = raw.get(f"{role_name}_max_tokens", DEFAULT_MAX_TOKENS)
        self._timeout = raw.get(f"{role_name}_timeout", DEFAULT_TIMEOUT_S)
        self._provider_name = provider
        self._wire_api = self._normalize_wire_api(
            provider_config.get("wire_api") or raw.get("wire_api")
        )
        self._reasoning_effort = str(
            provider_config.get(f"{role_name}_reasoning_effort")
            or provider_config.get("reasoning_effort")
            or raw.get(f"{role_name}_reasoning_effort")
            or raw.get("reasoning_effort")
            or ""
        ).strip().lower()

        missing = []
        if not provider_config.get("api_key") or not self._api_key:
            missing.append("api_key")
        if not provider_config.get("base_url") or not self._base_url:
            missing.append("base_url")
        if not self._model:
            missing.append(f"{role_name}_model")
        if missing:
            env_hint = ""
            raw_api_key = str(provider_config.get("api_key", "") or "").strip()
            if raw_api_key.startswith("env:") and not self._api_key:
                env_hint = f"；请先 export {raw_api_key[4:].strip()}=..."
            raise ValueError(f"[{provider}] 配置块缺少必要字段或解析为空: {', '.join(missing)}{env_hint}")

    @staticmethod
    def _resolve_config_string(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        if text.startswith("env:"):
            env_name = text[4:].strip()
            return os.getenv(env_name, "")
        env_match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", text)
        if env_match:
            return os.getenv(env_match.group(1), "")
        return text

    @staticmethod
    def _normalize_wire_api(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"", "chat", "chat_completions", "chat-completions", "chat/completions"}:
            return "chat_completions"
        if text in {"responses", "response"}:
            return "responses"
        return text

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def timeout(self) -> int:
        return self._timeout

    @property
    def provider(self) -> str:
        return self._provider_name

    @property
    def wire_api(self) -> str:
        return self._wire_api

    @property
    def reasoning_effort(self) -> str:
        return self._reasoning_effort

    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


__all__ = [
    "APIConfig",
    "build_default_results_family_root",
    "build_default_results_root",
    "build_default_results_dir_from_api_config",
    "build_model_results_dir_name",
    "get_active_provider_and_models",
    "resolve_results_dir_path",
    "resolve_results_root_path",
    "resolve_api_config_path",
]
