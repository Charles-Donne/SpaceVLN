"""Generic API client base shared by planner and action runtimes."""

import base64
import io
import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from navigation_system.config.core.params.api import (
    DEFAULT_IMAGE_COMPRESSION_ENABLED,
    DEFAULT_IMAGE_COMPRESSION_MAX_SIZE,
    DEFAULT_IMAGE_COMPRESSION_QUALITY,
)
from navigation_system.vlm.api.config import APIConfig


class BaseAPIClient(ABC):
    """API客户端基类"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        # 压缩配置（默认启用）
        self.compress_images = DEFAULT_IMAGE_COMPRESSION_ENABLED
        self.compression_max_size = DEFAULT_IMAGE_COMPRESSION_MAX_SIZE
        self.compression_quality = DEFAULT_IMAGE_COMPRESSION_QUALITY
        self.save_request_artifacts = False

    def _apply_reasoning_disabled_defaults(self, payload: Dict) -> Dict:
        """Best-effort disable provider-side thinking/reasoning by default."""
        base_url = str(getattr(self.config, 'base_url', '') or '').lower()
        provider = str(getattr(self.config, 'provider', '') or '').lower()
        wire_api = str(getattr(self.config, 'wire_api', '') or '').lower()
        reasoning_effort = str(getattr(self.config, 'reasoning_effort', '') or '').lower()

        if 'dashscope' in base_url or provider == 'dashscope':
            payload['enable_thinking'] = False

        if 'openrouter' in base_url or provider == 'openrouter':
            payload['reasoning'] = {
                'effort': reasoning_effort or 'none',
                'exclude': True,
            }

        if provider == 'openai' or ('openai' in base_url and wire_api == 'responses'):
            payload['reasoning'] = {
                'effort': reasoning_effort or 'none',
            }

        return payload

    def _supports_json_object_response_format(self) -> bool:
        """Whether the current provider likely supports OpenAI-compatible JSON mode."""
        base_url = str(getattr(self.config, 'base_url', '') or '').lower()
        provider = str(getattr(self.config, 'provider', '') or '').lower()
        return (
            'dashscope' in base_url
            or 'openrouter' in base_url
            or provider in {'dashscope', 'openrouter', 'openai'}
        )

    def _build_response_format(self) -> Optional[Dict[str, Any]]:
        """Best-effort structured output request for providers that support it."""
        if not self._supports_json_object_response_format():
            return None
        return {"type": "json_object"}
    
    def set_compression_config(self, enabled: bool = True, max_size: int = 384, quality: int = 80):
        """
        设置图片压缩配置
        
        Args:
            enabled: 是否启用压缩
            max_size: 最大边长（推荐：384=高节省，512=平衡，768=高质量）
            quality: JPEG质量（推荐：75=高节省，80=平衡，85=高质量）
        """
        self.compress_images = enabled
        self.compression_max_size = max_size
        self.compression_quality = quality

    def set_request_artifact_saving(self, enabled: bool = False):
        """Control whether prompt/image request artifacts are written to disk."""
        self.save_request_artifacts = bool(enabled)

    def _save_json_artifact(self, save_dir: Optional[str], filename: str, payload: Any) -> None:
        if not (save_dir and self.save_request_artifacts and filename):
            return
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _mime_from_path(path: str) -> str:
        suffix = os.path.splitext(str(path or ""))[1].lower()
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".webp":
            return "image/webp"
        return "image/png"

    @staticmethod
    def _replace_suffix(filename: str, suffix: str) -> str:
        stem, _ext = os.path.splitext(str(filename or "img"))
        normalized_suffix = str(suffix or "").strip() or ".png"
        if not normalized_suffix.startswith("."):
            normalized_suffix = f".{normalized_suffix}"
        return f"{stem}{normalized_suffix}"

    def _derive_artifact_filename(
        self,
        image_input: Any,
        idx: int,
        compress: bool,
    ) -> str:
        if isinstance(image_input, dict):
            artifact_name = str(image_input.get("artifact_name") or "").strip()
            if artifact_name:
                return artifact_name
            base_name = str(image_input.get("name") or f"img_{idx:02d}").strip() or f"img_{idx:02d}"
            return self._replace_suffix(base_name, ".jpg" if compress else ".png")

        img_path = str(image_input)
        parent_dir = os.path.basename(os.path.dirname(img_path))
        if parent_dir == 'directions':
            basename = os.path.basename(img_path)
            angle_match = re.search(r'(\d{3})\.(?:png|jpg|jpeg)$', basename, re.IGNORECASE)
            if angle_match:
                return f"direction_{angle_match.group(1)}{'.jpg' if compress else '.png'}"
            return f"img_{idx:02d}{'.jpg' if compress else '.png'}"
        if parent_dir == 'global_map':
            return f"global_map{'.jpg' if compress else '.png'}"
        if parent_dir == 'local_map':
            return f"local_map{'.jpg' if compress else '.png'}"
        if parent_dir == 'detection':
            return f"detection{'.jpg' if compress else '.png'}"
        if parent_dir == 'rgb':
            return f"rgb{'.jpg' if compress else '.png'}"

        basename = os.path.basename(img_path) or f"img_{idx:02d}"
        return self._replace_suffix(basename, ".jpg" if compress else os.path.splitext(basename)[1] or ".png")

    @staticmethod
    def _encode_image_array_bytes(
        image_array: np.ndarray,
        compress: bool = True,
        max_size: int = 384,
        quality: int = 80,
        color_space: str = "bgr",
    ) -> Tuple[bytes, str]:
        from PIL import Image

        array = np.asarray(image_array)
        if array.ndim == 2:
            pil_image = Image.fromarray(array.astype(np.uint8), mode="L")
        else:
            if array.shape[-1] == 4:
                if str(color_space).lower() == "bgra":
                    array = array[..., [2, 1, 0, 3]]
                pil_image = Image.fromarray(array.astype(np.uint8), mode="RGBA")
            else:
                if str(color_space).lower() == "bgr":
                    array = array[..., ::-1]
                pil_image = Image.fromarray(array.astype(np.uint8), mode="RGB")

        if compress:
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
                pil_image = pil_image.resize(new_size, Image.LANCZOS)
            buffer = io.BytesIO()
            pil_image.convert('RGB').save(buffer, 'JPEG', quality=quality, optimize=True)
            return buffer.getvalue(), "image/jpeg"

        buffer = io.BytesIO()
        if pil_image.mode not in ("RGB", "RGBA", "L"):
            pil_image = pil_image.convert("RGB")
        pil_image.save(buffer, 'PNG', optimize=True)
        return buffer.getvalue(), "image/png"

    def _load_image_input_bytes(
        self,
        image_input: Any,
        compress: bool = True,
        max_size: int = 384,
        quality: int = 80,
    ) -> Tuple[bytes, str]:
        if isinstance(image_input, dict):
            if image_input.get("image_array") is not None:
                return self._encode_image_array_bytes(
                    image_array=image_input.get("image_array"),
                    compress=compress,
                    max_size=max_size,
                    quality=quality,
                    color_space=str(image_input.get("color_space") or "bgr"),
                )
            if image_input.get("image_bytes") is not None:
                image_bytes = bytes(image_input.get("image_bytes") or b"")
                if not compress:
                    mime_type = str(image_input.get("mime_type") or "image/png")
                    return image_bytes, mime_type
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                buffer = io.BytesIO()
                img.convert('RGB').save(buffer, 'JPEG', quality=quality, optimize=True)
                return buffer.getvalue(), "image/jpeg"
            if image_input.get("path") is not None:
                image_input = image_input.get("path")
            else:
                raise ValueError("Unsupported image input dict: missing image_array/image_bytes/path")

        image_path = str(image_input)
        if not compress:
            with open(image_path, "rb") as f:
                return f.read(), self._mime_from_path(image_path)

        from PIL import Image

        img = Image.open(image_path)
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        img.convert('RGB').save(buffer, 'JPEG', quality=quality, optimize=True)
        return buffer.getvalue(), "image/jpeg"
    
    @staticmethod
    def clean_json_response(text: str) -> str:
        """清理响应文本"""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        return text.strip()

    @staticmethod
    def repair_truncated_json_object(text: str) -> Optional[str]:
        """Best-effort repair for a truncated JSON object that still starts with `{`."""
        cleaned = BaseAPIClient.clean_json_response(text)
        start = cleaned.find('{')
        if start < 0:
            return None

        candidate = cleaned[start:].rstrip()
        if not candidate:
            return None

        in_string = False
        escape = False
        brace_depth = 0
        bracket_depth = 0

        for ch in candidate:
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth = max(0, brace_depth - 1)
            elif ch == '[':
                bracket_depth += 1
            elif ch == ']':
                bracket_depth = max(0, bracket_depth - 1)

        repaired = re.sub(r',\s*$', '', candidate)
        if in_string:
            repaired += '"'
        if bracket_depth > 0:
            repaired += ']' * bracket_depth
        if brace_depth > 0:
            repaired += '}' * brace_depth
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        return repaired

    @staticmethod
    def extract_linewise_scalar_fields(text: str) -> Dict[str, Any]:
        """Recover top-level scalar fields from malformed JSON-like output."""
        cleaned = BaseAPIClient.clean_json_response(text)
        pattern = re.compile(
            r'(?m)^\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*'
            r'("(?:(?:\\.|[^"\\])*)"|true|false|null|-?\d+(?:\.\d+)?)\s*,?\s*$'
        )
        recovered: Dict[str, Any] = {}
        for key, raw_value in pattern.findall(cleaned):
            try:
                recovered[key] = json.loads(raw_value)
            except json.JSONDecodeError:
                continue
        return recovered
    
    @staticmethod
    def extract_json_object(text: str) -> Optional[str]:
        """提取JSON对象"""
        match = re.search(r'\{[\s\S]*\}', text)
        return match.group(0) if match else None

    @staticmethod
    def extract_json_objects(text: str) -> List[Dict]:
        """Extract every decodable top-level JSON object from a noisy response."""
        decoder = json.JSONDecoder()
        results: List[Dict] = []
        text = str(text or "")
        for start_idx, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                parsed, end_idx = decoder.raw_decode(text[start_idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                results.append(parsed)
        return results
    
    def parse_json_response(self, response_text: str) -> Optional[Dict]:
        """解析JSON响应"""
        cleaned = self.clean_json_response(response_text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            json_candidates = self.extract_json_objects(cleaned)
            if json_candidates:
                return json_candidates[-1]

            json_str = self.extract_json_object(cleaned)
            if json_str:
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"✗ JSON parse error: {e}")

            repaired = self.repair_truncated_json_object(cleaned)
            if repaired:
                try:
                    parsed = json.loads(repaired)
                    if isinstance(parsed, dict):
                        print("[WARN] Recovered truncated JSON response by auto-closing the object")
                        return parsed
                except json.JSONDecodeError:
                    pass

            recovered_fields = self.extract_linewise_scalar_fields(cleaned)
            if recovered_fields:
                print(
                    "[WARN] Recovered scalar JSON fields from malformed response: "
                    f"{', '.join(sorted(recovered_fields.keys()))}"
                )
                return recovered_fields

            if json_str:
                self._save_failed_response(response_text, "Malformed JSON object")
            else:
                print(f"✗ No JSON found in response")
                self._save_failed_response(response_text, "No JSON object found")
            return None
    
    def _save_failed_response(self, response_text: str, error_msg: str):
        """保存解析失败的VLM原始输出"""
        import os
        from datetime import datetime

        # 保存到仓库内已忽略的 tmp 目录，避免误提交到 GitHub。
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        failed_dir = os.path.join(project_root, "tmp", "failed_vlm_responses")
        os.makedirs(failed_dir, exist_ok=True)
        
        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(failed_dir, f"failed_{timestamp}.txt")
        
        # 保存VLM原始输出
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response_text)  # 直接写入原始文本
            print(f"  Raw output saved: {os.path.abspath(filename)}")
        except Exception as e:
            print(f"  [WARN] Save failed: {e}")
    
    def build_message_content(
        self,
        text: str,
        image_paths: List[Any],
        save_dir: str = None,
        no_compress_indices: set = None,
        prompt_artifact_filename: Optional[str] = "prompt.md",
        artifact_records: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict]:
        """构建消息内容，可选保存压缩后的图片（即模型实际看到的版本）
        
        Args:
            text: prompt文本
            image_paths: 图片路径列表
            save_dir: 如果指定，将压缩后的图片和prompt保存到此目录
            no_compress_indices: 不压缩的图片索引集合（如 {4} 表示第5张图不压缩）
            prompt_artifact_filename: prompt 落盘文件名，传 None 可跳过文本副本
        """
        content = [{"type": "text", "text": text}]
        should_save_artifacts = bool(save_dir and self.save_request_artifacts)
        
        if should_save_artifacts and prompt_artifact_filename:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, prompt_artifact_filename), 'w', encoding='utf-8') as f:
                f.write(text)
        if artifact_records is not None:
            artifact_records.append({
                "kind": "text",
                "content_type": "text",
                "artifact_filename": prompt_artifact_filename,
                "artifact_path": (
                    os.path.join(save_dir, prompt_artifact_filename)
                    if should_save_artifacts and prompt_artifact_filename
                    else None
                ),
                "text_length": len(str(text or "")),
            })
        
        for idx, image_input in enumerate(image_paths):
            skip_compress = no_compress_indices is not None and idx in no_compress_indices
            compress = False if skip_compress else self.compress_images
            image_bytes, mime_type = self._load_image_input_bytes(
                image_input,
                compress=compress,
                max_size=self.compression_max_size,
                quality=self.compression_quality,
            )
            img_base64 = base64.b64encode(image_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
            })
            
            if should_save_artifacts:
                img_filename = self._derive_artifact_filename(image_input, idx=idx, compress=compress)
                save_path = os.path.join(save_dir, img_filename)
                with open(save_path, 'wb') as f:
                    f.write(image_bytes)
            else:
                img_filename = self._derive_artifact_filename(image_input, idx=idx, compress=compress)
                save_path = None
            if artifact_records is not None:
                artifact_records.append({
                    "kind": "image",
                    "content_type": "image_url",
                    "index": int(idx),
                    "artifact_filename": img_filename,
                    "artifact_path": save_path,
                    "mime_type": mime_type,
                    "compressed": bool(compress),
                    "size_bytes": len(image_bytes),
                })
        
        return content

    def build_responses_input_content(
        self,
        text: str,
        image_paths: List[Any],
        save_dir: str = None,
        no_compress_indices: set = None,
        prompt_artifact_filename: Optional[str] = "prompt.md",
        artifact_records: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict]:
        """Build Responses API content blocks with optional request artifact saving."""
        content = [{"type": "input_text", "text": text}]
        should_save_artifacts = bool(save_dir and self.save_request_artifacts)

        if should_save_artifacts and prompt_artifact_filename:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, prompt_artifact_filename), 'w', encoding='utf-8') as f:
                f.write(text)
        if artifact_records is not None:
            artifact_records.append({
                "kind": "text",
                "content_type": "input_text",
                "artifact_filename": prompt_artifact_filename,
                "artifact_path": (
                    os.path.join(save_dir, prompt_artifact_filename)
                    if should_save_artifacts and prompt_artifact_filename
                    else None
                ),
                "text_length": len(str(text or "")),
            })

        for idx, image_input in enumerate(image_paths):
            skip_compress = no_compress_indices is not None and idx in no_compress_indices
            compress = False if skip_compress else self.compress_images
            image_bytes, mime_type = self._load_image_input_bytes(
                image_input,
                compress=compress,
                max_size=self.compression_max_size,
                quality=self.compression_quality,
            )
            img_base64 = base64.b64encode(image_bytes).decode("utf-8")
            content.append({
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{img_base64}",
            })

            if should_save_artifacts:
                img_filename = self._derive_artifact_filename(image_input, idx=idx, compress=compress)
                save_path = os.path.join(save_dir, img_filename)
                with open(save_path, 'wb') as f:
                    f.write(image_bytes)
            else:
                img_filename = self._derive_artifact_filename(image_input, idx=idx, compress=compress)
                save_path = None
            if artifact_records is not None:
                artifact_records.append({
                    "kind": "image",
                    "content_type": "input_image",
                    "index": int(idx),
                    "artifact_filename": img_filename,
                    "artifact_path": save_path,
                    "mime_type": mime_type,
                    "compressed": bool(compress),
                    "size_bytes": len(image_bytes),
                })

        return content

    def _candidate_endpoint_urls(self, endpoint_suffix: str) -> List[str]:
        base_url = str(getattr(self.config, 'base_url', '') or '').rstrip('/')
        if not base_url:
            return [endpoint_suffix]

        urls = [f"{base_url}{endpoint_suffix}"]
        if not re.search(r"/v\d+$", base_url):
            urls.append(f"{base_url}/v1{endpoint_suffix}")

        deduped: List[str] = []
        for url in urls:
            if url not in deduped:
                deduped.append(url)
        return deduped

    def _post_json_with_base_url_fallback(
        self,
        endpoint_suffix: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> requests.Response:
        candidate_urls = self._candidate_endpoint_urls(endpoint_suffix)
        response = None
        for index, url in enumerate(candidate_urls):
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            if response.status_code != 404:
                return response
            if index < len(candidate_urls) - 1:
                print(f"[WARN] Endpoint not found: {url} | retrying next candidate")

        assert response is not None
        return response

    @staticmethod
    def _extract_responses_api_text(result: Dict[str, Any]) -> str:
        direct_text = result.get("output_text")
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text

        texts: List[str] = []
        for item in result.get("output", []) or []:
            if not isinstance(item, dict):
                continue

            if item.get("type") in {"output_text", "text"}:
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text)
                continue

            if item.get("type") != "message":
                continue

            for content_item in item.get("content", []) or []:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") not in {"output_text", "text"}:
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text)

        return "\n".join(texts).strip()
    
    def call_api(self, prompt: Any, image_paths: List[Any], save_dir: str = None,
                 no_compress_indices: set = None) -> Optional[Dict]:
        """调用API（带计时和速度统计）
        
        Args:
            prompt: prompt文本；特殊运行时可在子类中扩展为其他 prompt 容器
            image_paths: 图片路径列表
            save_dir: 如果指定，在发送时同步保存压缩图片+prompt到此目录
            no_compress_indices: 不压缩的图片索引集合（如 {4} 表示第5张图不压缩）
        """
        t_start = time.time()
        try:
            # 确保 prompt 是 UTF-8 编码的字符串
            if isinstance(prompt, bytes):
                prompt = prompt.decode('utf-8')
            
            wire_api = str(getattr(self.config, 'wire_api', '') or 'chat_completions').lower()
            is_responses_api = wire_api == 'responses'
            is_openrouter = 'openrouter' in self.config.base_url.lower()

            if is_responses_api:
                payload = {
                    "model": self.config.model,
                    "input": [{
                        "role": "user",
                        "content": self.build_responses_input_content(
                            prompt,
                            image_paths,
                            save_dir=save_dir,
                            no_compress_indices=no_compress_indices,
                        ),
                    }],
                    "max_output_tokens": self.config.max_tokens,
                }
                if self._supports_json_object_response_format():
                    payload["text"] = {"format": {"type": "json_object"}}
                endpoint_suffix = "/responses"
            else:
                payload = {
                    "model": self.config.model,
                    "messages": [{
                        "role": "user",
                        "content": self.build_message_content(prompt, image_paths, save_dir=save_dir,
                                                              no_compress_indices=no_compress_indices)
                    }],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
                response_format = self._build_response_format()
                if response_format is not None:
                    payload["response_format"] = response_format
                endpoint_suffix = "/chat/completions"

            payload = self._apply_reasoning_disabled_defaults(payload)

            # 构建headers（支持OpenRouter优化）
            headers = self.config.get_headers()
            if is_openrouter:
                # OpenRouter: 固定优先走阿里云 Tongyi 后端（Qwen 系列在此最快，~90-100 TPS）
                # 若阿里云不可用则 fallback 到其他 provider
                headers["X-Title"] = "SpaceVLN"
                payload["provider"] = {
                    "order": ["Alibaba"],         # 固定优先走阿里云（最高吞吐）
                    "allow_fallbacks": True       # 阿里云不可用时允许回退
                }

            response = self._post_json_with_base_url_fallback(
                endpoint_suffix=endpoint_suffix,
                headers=headers,
                payload=payload,
            )

            structured_hint_rejected = (
                response.status_code in {400, 422}
                and (
                    ("response_format" in payload)
                    or ("text" in payload and isinstance(payload.get("text"), dict))
                )
            )
            if structured_hint_rejected:
                print("[WARN] Structured-output hint was rejected; retry without it")
                payload = dict(payload)
                payload.pop("response_format", None)
                payload.pop("text", None)
                response = self._post_json_with_base_url_fallback(
                    endpoint_suffix=endpoint_suffix,
                    headers=headers,
                    payload=payload,
                )
            
            t_response = time.time()
            latency = t_response - t_start
            
            if response.status_code != 200:
                print(f"✗ API error: {response.status_code} ({latency:.1f}s)")
                # 诊断信息：记录请求参数和响应
                try:
                    error_detail = response.json()
                    print(f"✗ Error detail: {error_detail}")
                except:
                    print(f"✗ Response: {response.text[:500]}")
                
                # 调试：记录payload大小和内容样本
                payload_size = len(json.dumps(payload))
                print(f"  [DEBUG] Payload size: {payload_size} bytes")
                if is_responses_api:
                    content_blocks = (((payload.get('input') or [{}])[0]).get('content') or [])
                else:
                    content_blocks = (((payload.get('messages') or [{}])[0]).get('content') or [])
                if len(content_blocks) > 0:
                    first_item = content_blocks[0]
                    if first_item.get('type') == 'text':
                        print(f"  [DEBUG] Prompt length: {len(first_item.get('text', ''))} chars")
                    elif first_item.get('type') == 'input_text':
                        print(f"  [DEBUG] Prompt length: {len(first_item.get('text', ''))} chars")
                    elif first_item.get('type') == 'image_url':
                        print(f"  [DEBUG] First item is image_url")
                    elif first_item.get('type') == 'input_image':
                        print(f"  [DEBUG] First item is input_image")
                
                return None
            
            result = response.json()
            if is_responses_api:
                content = self._extract_responses_api_text(result)
                usage = result.get('usage', {})
                prompt_tokens = usage.get('input_tokens', usage.get('prompt_tokens', 0))
                completion_tokens = usage.get('output_tokens', usage.get('completion_tokens', 0))
                total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
            else:
                content = result['choices'][0]['message']['content']
                usage = result.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
            
            # 计算速度
            tokens_per_sec = completion_tokens / latency if latency > 0 and completion_tokens > 0 else 0
            
            # 打印速度统计
            model_short = self.config.model.split('/')[-1][:30]
            speed_info = f"{model_short} | {latency:.1f}s | {prompt_tokens}->{completion_tokens} tok | {tokens_per_sec:.0f} tok/s"
            
            # OpenRouter额外信息
            if is_openrouter:
                provider = result.get('provider', '')
                if provider:
                    speed_info += f" | via {provider}"
            
            print(speed_info)
            
            # 检查响应
            if not content or len(content.strip()) < 10:
                print(f"✗ Empty or too short API response: {content}")
                return None
            
            # 检查截断
            if is_responses_api:
                status = str(result.get('status', '') or '').lower()
                incomplete_details = result.get('incomplete_details') or {}
                if status == 'incomplete':
                    reason = incomplete_details.get('reason', 'unknown')
                    print(f"[WARN] Response incomplete (max_output_tokens={self.config.max_tokens}, reason={reason})")
            else:
                finish_reason = result['choices'][0].get('finish_reason', 'unknown')
                if finish_reason == 'length':
                    print(f"[WARN] Response truncated (max_tokens={self.config.max_tokens})")
            
            parsed = self.parse_json_response(content)
            
            if parsed is None:
                print(f"✗ JSON parse failed | Raw (first 300): {content[:300]}")
                
            return parsed
            
        except requests.exceptions.Timeout:
            elapsed = time.time() - t_start
            print(f"✗ API timeout after {elapsed:.1f}s (limit={self.config.timeout}s)")
            return None
        except json.JSONDecodeError as e:
            elapsed = time.time() - t_start
            print(f"✗ JSON decode error ({elapsed:.1f}s): {e}")
            print(f"✗ Response text: {response.text[:300]}")
            return None
        except Exception as e:
            elapsed = time.time() - t_start
            print(f"✗ API call failed ({elapsed:.1f}s): {e}")
            return None
    
    def validate_fields(self, response: Dict, required_fields: List[str]) -> bool:
        """验证响应字段"""
        missing = [f for f in required_fields if f not in response]
        if missing:
            print(f"✗ Response missing fields: {', '.join(missing)}")
            print(f"✗ Received fields: {list(response.keys())}")
            return False
        return True
    
    @abstractmethod
    def validate_response(self, response: Dict) -> bool:
        """验证响应（子类实现）"""
        pass
