"""Shared API-config loading and results-path helpers."""

import os
import re
from typing import Any, Dict, Tuple

import yaml

from navigation_system.config.core.params.api import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
)


def resolve_api_config_path(config_path: str) -> str:
    """Normalize the configured unified API config path."""
    return str(config_path or "").strip()


def _slugify_results_component(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/").replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    return text or "unknown"


def get_active_provider_and_models(config_path: str) -> Tuple[str, str, str]:
    resolved_path = resolve_api_config_path(config_path)
    if not os.path.exists(resolved_path):
        return "", "", ""

    with open(resolved_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    provider = str(raw.get("provider", "") or "")
    provider_config = raw.get(provider) or {}
    llm_model = str(provider_config.get("llm_model") or provider_config.get("model") or "")
    vlm_model = str(provider_config.get("vlm_model") or provider_config.get("model") or "")
    return provider, llm_model, vlm_model


def build_model_results_dir_name(config_path: str) -> str:
    _, llm_model, vlm_model = get_active_provider_and_models(config_path)
    llm_slug = _slugify_results_component(llm_model)
    vlm_slug = _slugify_results_component(vlm_model)
    if llm_slug == vlm_slug:
        return llm_slug
    return f"{llm_slug}__{vlm_slug}"


def build_default_results_dir_from_api_config(
    config_path: str,
    repo_root: str = None,
) -> str:
    if repo_root:
        resolved_repo_root = os.path.abspath(repo_root)
    else:
        resolved_repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
    return os.path.abspath(
        os.path.join(
            resolved_repo_root,
            "..",
            "data",
            "result",
            "vlnce",
            build_model_results_dir_name(config_path),
        )
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

        missing = [field for field in ("api_key", "base_url") if not provider_config.get(field)]
        if not self._model:
            missing.append(f"{role_name}_model")
        if missing:
            raise ValueError(f"[{provider}] 配置块缺少必要字段: {', '.join(missing)}")

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
    "build_default_results_dir_from_api_config",
    "build_model_results_dir_name",
    "get_active_provider_and_models",
    "resolve_api_config_path",
]
