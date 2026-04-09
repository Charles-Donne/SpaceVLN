"""DashScope explicit-context-cache helpers for Qwen planning/action runtimes."""

from dataclasses import asdict, dataclass
import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
import yaml

from navigation_system.vlm.prompts.common import ExplicitCachePromptBundle
from navigation_system.vlm.api.config import (
    build_default_results_dir_from_api_config,
    resolve_api_config_path,
)


SUPPORTED_QWEN_CONTEXT_CACHE_PREFIXES = (
    "qwen3.5-plus",
    "qwen3.5-flash",
    "qwen3.6-plus",
)


def supports_qwen_explicit_context_cache(model_name: str) -> bool:
    normalized = str(model_name or "").strip().lower()
    return any(normalized.startswith(prefix) for prefix in SUPPORTED_QWEN_CONTEXT_CACHE_PREFIXES)


@dataclass(frozen=True)
class QwenContextCacheSettings:
    enabled: bool = True
    cache_type: str = "ephemeral"
    print_usage: bool = True
    save_usage_json: bool = True
    results_dir_suffix: str = "_cache"

    @classmethod
    def from_config_path(cls, config_path: str) -> "QwenContextCacheSettings":
        resolved_path = resolve_api_config_path(config_path)
        if not resolved_path or not os.path.exists(resolved_path):
            return cls()
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        block = dict(raw.get("qwen_context_cache") or {})
        return cls(
            enabled=bool(block.get("enabled", True)),
            cache_type=str(block.get("cache_type") or "ephemeral").strip().lower() or "ephemeral",
            print_usage=bool(block.get("print_usage", True)),
            save_usage_json=bool(block.get("save_usage_json", True)),
            results_dir_suffix=str(block.get("results_dir_suffix") or "_cache").strip()
            or "_cache",
        )


@dataclass(frozen=True)
class QwenCacheUsageSummary:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    cache_creation_input_tokens: int
    uncached_prompt_tokens: int
    effective_input_cost_multiplier: float
    input_cost_savings_ratio: float
    cache_hit_ratio: float
    latency_s: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["effective_input_cost_multiplier"] = round(
            float(self.effective_input_cost_multiplier), 4
        )
        payload["input_cost_savings_ratio"] = round(float(self.input_cost_savings_ratio), 4)
        payload["cache_hit_ratio"] = round(float(self.cache_hit_ratio), 4)
        payload["latency_s"] = round(float(self.latency_s), 4)
        return payload


def build_default_qwen_context_cache_results_dir(
    config_path: str,
    repo_root: Optional[str] = None,
) -> str:
    base_dir = build_default_results_dir_from_api_config(config_path, repo_root=repo_root)
    settings = QwenContextCacheSettings.from_config_path(config_path)
    suffix = str(settings.results_dir_suffix or "").strip()
    return f"{base_dir}{suffix}" if suffix else base_dir


class QwenContextCacheMixin:
    """Mixin for BaseAPIClient subclasses that want DashScope explicit cache support."""

    context_cache_settings: QwenContextCacheSettings
    last_cache_usage: Optional[QwenCacheUsageSummary]

    def _init_qwen_context_cache(self, config_path: str) -> None:
        self.context_cache_settings = QwenContextCacheSettings.from_config_path(config_path)
        self.last_cache_usage = None
        model_name = str(getattr(self.config, "model", "") or "")
        provider = str(getattr(self.config, "provider", "") or "").lower()
        base_url = str(getattr(self.config, "base_url", "") or "").lower()
        if not self.context_cache_settings.enabled:
            return
        if provider != "dashscope" and "dashscope" not in base_url:
            raise ValueError("Qwen explicit-context-cache runtime requires DashScope provider/base_url")
        if not supports_qwen_explicit_context_cache(model_name):
            raise ValueError(
                f"Model does not support this cached runtime: {model_name}. "
                "Use qwen3.5-plus / qwen3.5-flash (or newer supported Qwen cache models)."
            )

    @staticmethod
    def _extract_chat_completion_text(result: Dict[str, Any]) -> str:
        choices = list(result.get("choices") or [])
        if not choices:
            return ""
        message = dict((choices[0] or {}).get("message") or {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _extract_prompt_token_details(usage: Dict[str, Any]) -> Dict[str, Any]:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            return details
        details = usage.get("input_tokens_details")
        if isinstance(details, dict):
            return details
        return {}

    @classmethod
    def _has_explicit_cache_counters(cls, usage: Dict[str, Any]) -> bool:
        details = cls._extract_prompt_token_details(usage)
        cache_creation = details.get("cache_creation")
        return any(
            (
                "cached_tokens" in usage,
                "cached_tokens" in details,
                "cache_creation_input_tokens" in usage,
                "cache_creation_input_tokens" in details,
                isinstance(cache_creation, dict) and bool(cache_creation),
            )
        )

    @classmethod
    def _extract_cache_usage_summary(
        cls,
        result: Dict[str, Any],
        *,
        latency_s: float,
    ) -> Optional[QwenCacheUsageSummary]:
        usage = dict(result.get("usage") or {})
        if not usage:
            return None

        prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion_tokens = int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
        total_tokens = int(
            usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
        )
        details = cls._extract_prompt_token_details(usage)
        cache_creation = dict(details.get("cache_creation") or {})
        cached_tokens = int(
            details.get("cached_tokens", usage.get("cached_tokens", 0)) or 0
        )
        cache_creation_input_tokens = int(
            details.get("cache_creation_input_tokens")
            or cache_creation.get("cache_creation_input_tokens")
            or cache_creation.get("ephemeral_5m_input_tokens")
            or 0
        )
        if (
            prompt_tokens <= 0
            and completion_tokens <= 0
            and cached_tokens <= 0
            and cache_creation_input_tokens <= 0
        ):
            return None

        uncached_prompt_tokens = max(
            0,
            prompt_tokens - max(0, cached_tokens) - max(0, cache_creation_input_tokens),
        )
        baseline_units = max(1.0, float(prompt_tokens))
        effective_units = (
            float(uncached_prompt_tokens)
            + float(cached_tokens) * 0.1
            + float(cache_creation_input_tokens) * 1.25
        )
        effective_multiplier = effective_units / baseline_units
        input_cost_savings_ratio = 1.0 - effective_multiplier
        cache_hit_ratio = float(cached_tokens) / baseline_units
        return QwenCacheUsageSummary(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            uncached_prompt_tokens=uncached_prompt_tokens,
            effective_input_cost_multiplier=effective_multiplier,
            input_cost_savings_ratio=input_cost_savings_ratio,
            cache_hit_ratio=cache_hit_ratio,
            latency_s=max(0.0, float(latency_s)),
        )

    @staticmethod
    def _format_cache_usage_summary(summary: QwenCacheUsageSummary) -> str:
        savings_pct = float(summary.input_cost_savings_ratio) * 100.0
        hit_pct = float(summary.cache_hit_ratio) * 100.0
        sign = "+" if savings_pct >= 0.0 else ""
        return (
            "[CtxCache] "
            f"prompt={summary.prompt_tokens} "
            f"| cached={summary.cached_tokens} "
            f"| create={summary.cache_creation_input_tokens} "
            f"| uncached={summary.uncached_prompt_tokens} "
            f"| hit={hit_pct:.1f}% "
            f"| input_cost_x={summary.effective_input_cost_multiplier:.3f} "
            f"| savings={sign}{savings_pct:.1f}%"
        )

    def _save_context_cache_usage_artifact(
        self,
        *,
        save_dir: Optional[str],
        system_prompt: str,
        user_prompt: str,
        full_prompt: Optional[str],
        result: Dict[str, Any],
        summary: Optional[QwenCacheUsageSummary],
    ) -> None:
        if not (save_dir and self.save_request_artifacts):
            return
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "system_prompt.md"), "w", encoding="utf-8") as f:
            f.write(system_prompt)
        with open(os.path.join(save_dir, "user_prompt.md"), "w", encoding="utf-8") as f:
            f.write(user_prompt)
        if not self.context_cache_settings.save_usage_json:
            return
        payload: Dict[str, Any] = {
            "model": str(getattr(self.config, "model", "") or ""),
            "provider": str(getattr(self.config, "provider", "") or ""),
            "usage": dict(result.get("usage") or {}),
            "prompt_token_details": self._extract_prompt_token_details(dict(result.get("usage") or {})),
            "cache_control_requested": bool(
                getattr(self.context_cache_settings, "enabled", True)
            ),
            "cache_type_requested": str(
                getattr(self.context_cache_settings, "cache_type", "") or ""
            ),
            "provider_reported_explicit_cache_counters": self._has_explicit_cache_counters(
                dict(result.get("usage") or {})
            ),
        }
        if summary is not None:
            payload["context_cache_status"] = "reported"
            payload["context_cache"] = summary.to_dict()
        else:
            payload["context_cache_status"] = "not_reported"
            payload["context_cache"] = None
        with open(os.path.join(save_dir, "cache_usage.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _build_content_preview(
        content: List[Dict[str, Any]],
        artifact_records: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Render a compact request preview without inlining base64 image bytes."""
        artifact_records = list(artifact_records or [])
        image_records = [
            record for record in artifact_records
            if isinstance(record, dict) and record.get("kind") == "image"
        ]
        image_idx = 0
        preview: List[Dict[str, Any]] = []
        for item in list(content or []):
            if not isinstance(item, dict):
                preview.append({"type": "unknown", "value": str(item)})
                continue

            item_type = str(item.get("type") or "")
            if item_type == "text":
                preview.append({
                    "type": "text",
                    "text": str(item.get("text") or ""),
                })
                continue

            if item_type == "image_url":
                image_record = image_records[image_idx] if image_idx < len(image_records) else {}
                image_idx += 1
                preview.append({
                    "type": "image_url",
                    "image_url": {
                        "url": "<omitted:data-url>",
                        "artifact_filename": image_record.get("artifact_filename"),
                        "artifact_path": image_record.get("artifact_path"),
                        "mime_type": image_record.get("mime_type"),
                        "compressed": image_record.get("compressed"),
                        "size_bytes": image_record.get("size_bytes"),
                    },
                })
                continue

            preview.append(item)
        return preview

    def _save_request_payload_preview(
        self,
        *,
        save_dir: Optional[str],
        payload: Dict[str, Any],
        user_content: List[Dict[str, Any]],
        artifact_records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Save the final message structure that is sent to the provider."""
        if not (save_dir and self.save_request_artifacts):
            return
        preview_payload = dict(payload)
        preview_payload["messages"] = [
            {
                "role": "system",
                "content": list(((payload.get("messages") or [{}])[0]).get("content") or []),
            },
            {
                "role": "user",
                "content": self._build_content_preview(
                    user_content,
                    artifact_records=artifact_records,
                ),
            },
        ]
        self._save_json_artifact(save_dir, "request_payload.preview.json", preview_payload)

    def _build_qwen_chat_completion_url(self) -> str:
        base_url = str(getattr(self.config, "base_url", "") or "").rstrip("/")
        if not base_url:
            raise ValueError("DashScope base_url is required for explicit-context-cache runtime")
        return f"{base_url}/chat/completions"

    def call_api_with_explicit_context_cache(
        self,
        *,
        prompt_bundle: Optional[ExplicitCachePromptBundle] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        full_prompt: Optional[str] = None,
        image_paths: List[Any],
        save_dir: Optional[str] = None,
        no_compress_indices: Optional[set] = None,
    ) -> Optional[Dict[str, Any]]:
        if not getattr(self.context_cache_settings, "enabled", True):
            raise RuntimeError("Explicit context cache is disabled in qwen_context_cache settings")

        t_start = time.time()
        try:
            if prompt_bundle is not None:
                system_prompt = prompt_bundle.system_prompt
                user_prompt = prompt_bundle.user_prompt
                full_prompt = prompt_bundle.full_prompt
            if isinstance(system_prompt, bytes):
                system_prompt = system_prompt.decode("utf-8")
            system_prompt = str(system_prompt or "")
            if isinstance(user_prompt, bytes):
                user_prompt = user_prompt.decode("utf-8")
            user_prompt = str(user_prompt or "")
            if isinstance(full_prompt, bytes):
                full_prompt = full_prompt.decode("utf-8")
            if full_prompt is None:
                full_prompt = "\n\n".join(
                    [part.strip() for part in (system_prompt, user_prompt) if str(part or "").strip()]
                )

            artifact_records: List[Dict[str, Any]] = []
            user_content = self.build_message_content(
                user_prompt,
                image_paths,
                save_dir=save_dir,
                no_compress_indices=no_compress_indices,
                prompt_artifact_filename=None,
                artifact_records=artifact_records,
            )
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {
                                "type": str(self.context_cache_settings.cache_type or "ephemeral")
                            },
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]
            payload: Dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            response_format = self._build_response_format()
            if response_format is not None:
                payload["response_format"] = response_format
            payload = self._apply_reasoning_disabled_defaults(payload)
            headers = self.config.get_headers()
            response = requests.post(
                self._build_qwen_chat_completion_url(),
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )

            structured_hint_rejected = (
                response.status_code in {400, 422}
                and "response_format" in payload
            )
            if structured_hint_rejected:
                print("[WARN] Structured-output hint was rejected; retry without it")
                payload = dict(payload)
                payload.pop("response_format", None)
                response = requests.post(
                    self._build_qwen_chat_completion_url(),
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout,
                )

            latency_s = time.time() - t_start
            if response.status_code != 200:
                print(f"✗ API error: {response.status_code} ({latency_s:.1f}s)")
                try:
                    print(f"✗ Error detail: {response.json()}")
                except Exception:
                    print(f"✗ Response: {response.text[:500]}")
                return None

            result = response.json()
            content = self._extract_chat_completion_text(result)
            usage = dict(result.get("usage") or {})
            has_explicit_cache_counters = self._has_explicit_cache_counters(usage)
            prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            completion_tokens = int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            )
            total_tokens = int(
                usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
            )
            tokens_per_sec = completion_tokens / latency_s if latency_s > 0 and completion_tokens > 0 else 0.0
            model_short = str(self.config.model).split("/")[-1][:40]
            print(
                f"{model_short} | {latency_s:.1f}s | "
                f"{prompt_tokens}->{completion_tokens} tok | {tokens_per_sec:.0f} tok/s | total={total_tokens}"
            )

            raw_cache_usage = self._extract_cache_usage_summary(
                result,
                latency_s=latency_s,
            )
            self.last_cache_usage = raw_cache_usage if has_explicit_cache_counters else None
            if (
                self.last_cache_usage is not None
                and has_explicit_cache_counters
                and self.context_cache_settings.print_usage
            ):
                print(self._format_cache_usage_summary(self.last_cache_usage))
            elif self.context_cache_settings.print_usage:
                raw_details = self._extract_prompt_token_details(usage)
                detail_keys = sorted(str(key) for key in raw_details.keys())
                print(
                    "[CtxCache] provider returned no explicit cache counters "
                    f"(details_keys={detail_keys or ['<none>']})"
                )

            self._save_context_cache_usage_artifact(
                save_dir=save_dir,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                full_prompt=full_prompt,
                result=result,
                summary=self.last_cache_usage,
            )

            if not content or len(content.strip()) < 10:
                print(f"✗ Empty or too short API response: {content}")
                return None

            finish_reason = dict((result.get("choices") or [{}])[0] or {}).get(
                "finish_reason",
                "unknown",
            )
            if finish_reason == "length":
                print(f"[WARN] Response truncated (max_tokens={self.config.max_tokens})")

            parsed = self.parse_json_response(content)
            if parsed is None:
                print(f"✗ JSON parse failed | Raw (first 300): {content[:300]}")
            return parsed

        except requests.exceptions.Timeout:
            elapsed = time.time() - t_start
            print(f"✗ API timeout after {elapsed:.1f}s (limit={self.config.timeout}s)")
            return None
        except json.JSONDecodeError as exc:
            elapsed = time.time() - t_start
            print(f"✗ JSON decode error ({elapsed:.1f}s): {exc}")
            return None
        except Exception as exc:
            elapsed = time.time() - t_start
            print(f"✗ Explicit-cache API call failed ({elapsed:.1f}s): {exc}")
            return None
