"""
API配置和客户端基类
==================
统一管理API配置和调用逻辑
"""
import os
import yaml
import base64
import json
import re
import time
import io
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from vlnce_baselines.config.core.params.api import (
    DEFAULT_IMAGE_COMPRESSION_ENABLED,
    DEFAULT_IMAGE_COMPRESSION_MAX_SIZE,
    DEFAULT_IMAGE_COMPRESSION_QUALITY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
)


def resolve_api_config_path(config_path: str) -> str:
    """Resolve canonical and legacy API config locations."""
    path = str(config_path or "").strip()
    if not path:
        return path
    if os.path.exists(path):
        return path

    fallback_map = {
        "vlnce_baselines/config/api/vlm_api_config.yaml": "vlnce_baselines/config/api_config.yaml",
        "vlnce_baselines/config/llm_config.yaml": "vlnce_baselines/config/api/vlm_api_config.yaml",
        "vlnce_baselines/config/vlm_config.yaml": "vlnce_baselines/config/api/vlm_api_config.yaml",
        "vlnce_baselines/config/api_config.yaml": "vlnce_baselines/config/api/vlm_api_config.yaml",
        "vlnce_baselines/vlm/api_config.yaml": "vlnce_baselines/config/api/vlm_api_config.yaml",
        "vlnce_baselines/vlm/llm_config.yaml": "vlnce_baselines/config/api/vlm_api_config.yaml",
        "vlnce_baselines/vlm/vlm_config.yaml": "vlnce_baselines/config/api/vlm_api_config.yaml",
    }
    fallback = fallback_map.get(path)
    if fallback and os.path.exists(fallback):
        return fallback
    return path


class APIConfig:
    """统一API配置类
    
    支持两种格式：
      统一格式（推荐）: 包含 provider 字段，通过 role='llm'/'vlm' 选择对应模型和参数
      Legacy 格式: 直接包含 api_key / base_url / model（向后兼容）
    """
    
    def __init__(self, config_path: str, role: str = None):
        """
        Args:
            config_path: 配置文件路径
            role: 'llm'（高层规划）或 'vlm'（低层执行）
                  统一格式中用于选择 {role}_model / {role}_max_tokens / {role}_timeout
                  Legacy 格式中忽略此参数
        """
        resolved_path = resolve_api_config_path(config_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"配置文件不存在: {config_path}. "
                f"请在 `vlnce_baselines/config/` 下创建对应 yaml，或显式传入旧路径。"
            )
        
        self.path = resolved_path

        with open(resolved_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        
        self.config = raw
        self._role = role or 'llm'
        
        if 'provider' in raw:
            # ── 统一格式 ──────────────────────────────────────────────
            provider = raw['provider']
            if provider not in raw:
                raise ValueError(f"配置文件中找不到 provider '{provider}' 的配置块")
            pc = raw[provider]          # provider config block
            r  = self._role             # 'llm' or 'vlm'
            
            self._api_key      = pc.get('api_key', '')
            self._base_url     = pc.get('base_url', '')
            self._model        = pc.get(f'{r}_model') or pc.get('model', '')
            self._temperature  = raw.get('temperature', DEFAULT_TEMPERATURE)
            self._max_tokens   = raw.get(f'{r}_max_tokens', DEFAULT_MAX_TOKENS)
            self._timeout      = raw.get(f'{r}_timeout', DEFAULT_TIMEOUT_S)
            self._provider_name = provider
            
            missing = [f for f in ['api_key', 'base_url'] if not pc.get(f)]
            if not self._model:
                missing.append(f'{r}_model')
            if missing:
                raise ValueError(f"[{provider}] 配置块缺少必要字段: {', '.join(missing)}")
        else:
            # ── Legacy 格式（向后兼容）──────────────────────────────────
            required = ['api_key', 'base_url', 'model']
            missing = [f for f in required if not raw.get(f)]
            if missing:
                raise ValueError(f"配置文件缺少必要字段: {', '.join(missing)}")
            
            self._api_key      = raw['api_key']
            self._base_url     = raw['base_url']
            self._model        = raw['model']
            self._temperature  = raw.get('temperature', DEFAULT_TEMPERATURE)
            self._max_tokens   = raw.get('max_tokens', DEFAULT_MAX_TOKENS)
            self._timeout      = raw.get('timeout', DEFAULT_TIMEOUT_S)
            self._provider_name = None
    
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
        """当前使用的服务商名称（统一格式返回 'dashscope'/'openrouter'，legacy 返回 ''）"""
        return self._provider_name or ''
    
    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }


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

        if 'dashscope' in base_url or provider == 'dashscope':
            payload['enable_thinking'] = False

        if 'openrouter' in base_url or provider == 'openrouter':
            payload['reasoning'] = {
                'effort': 'none',
                'exclude': True,
            }

        return payload
    
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
    
    @staticmethod
    def compress_image(image_path: str, max_size: int = 384, quality: int = 80) -> str:
        """
        压缩图片并返回临时路径
        
        Args:
            image_path: 原始图片路径
            max_size: 最大边长（默认384px，原始可能是1024px）
            quality: JPEG质量（80可节省50%大小而保持清晰）
        
        Returns:
            压缩后的临时文件路径
        """
        from PIL import Image
        import tempfile
        
        try:
            img = Image.open(image_path)
            
            # 等比例缩放
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # 保存为临时文件
            tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            img.convert('RGB').save(tmp.name, 'JPEG', quality=quality, optimize=True)
            tmp.close()
            
            # 输出压缩统计
            original_size = os.path.getsize(image_path)
            compressed_size = os.path.getsize(tmp.name)
            ratio = (1 - compressed_size / original_size) * 100
            # print(f"  🗜️  {os.path.basename(image_path)}: {original_size/1024:.0f}KB → {compressed_size/1024:.0f}KB (-{ratio:.0f}%)")
            
            return tmp.name
        except Exception as e:
            print(f"[WARN] Image compression failed: {e}, using original")
            return image_path

    @staticmethod
    def _load_image_bytes(image_path: str, compress: bool = True, max_size: int = 384, quality: int = 80) -> bytes:
        """Load image bytes, compressing in memory when requested."""
        if not compress:
            with open(image_path, "rb") as f:
                return f.read()

        from PIL import Image

        img = Image.open(image_path)
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        img.convert('RGB').save(buffer, 'JPEG', quality=quality, optimize=True)
        return buffer.getvalue()
    
    def encode_image_base64(self, image_path: str, compress: bool = None) -> str:
        """
        编码图像为base64
        
        Args:
            image_path: 图片路径
            compress: 是否压缩（None=使用self.compress_images，True=强制压缩，False=不压缩）
        
        Returns:
            base64编码字符串
        """
        import os
        
        # 确定是否压缩
        should_compress = compress if compress is not None else self.compress_images
        
        image_bytes = self._load_image_bytes(
            image_path,
            compress=should_compress,
            max_size=self.compression_max_size,
            quality=self.compression_quality,
        )
        if not should_compress and len(image_bytes) > 5 * 1024 * 1024:
            print(f"[WARN] Large image: {os.path.basename(image_path)} ({len(image_bytes) / 1024 / 1024:.2f}MB)")
        return base64.b64encode(image_bytes).decode("utf-8")
    
    @staticmethod
    def clean_json_response(text: str) -> str:
        """清理响应文本"""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        return text.strip()
    
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
        length = len(text)
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
                    self._save_failed_response(response_text, str(e))
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
    
    def build_message_content(self, text: str, image_paths: List[str], save_dir: str = None,
                              no_compress_indices: set = None) -> List[Dict]:
        """构建消息内容，可选保存压缩后的图片（即模型实际看到的版本）
        
        Args:
            text: prompt文本
            image_paths: 图片路径列表
            save_dir: 如果指定，将压缩后的图片和prompt保存到此目录
            no_compress_indices: 不压缩的图片索引集合（如 {4} 表示第5张图不压缩）
        """
        content = [{"type": "text", "text": text}]
        should_save_artifacts = bool(save_dir and self.save_request_artifacts)
        
        # 如果需要保存，先创建目录
        if should_save_artifacts:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(text)
        
        for idx, img_path in enumerate(image_paths):
            skip_compress = no_compress_indices is not None and idx in no_compress_indices
            compress = False if skip_compress else self.compress_images
            image_bytes = self._load_image_bytes(
                img_path,
                compress=compress,
                max_size=self.compression_max_size,
                quality=self.compression_quality,
            )
            img_base64 = base64.b64encode(image_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
            
            if should_save_artifacts:
                parent_dir = os.path.basename(os.path.dirname(img_path))
                if parent_dir == 'directions':
                    basename = os.path.basename(img_path)
                    angle_match = re.search(r'(\d{3})\.png$', basename)
                    if angle_match:
                        img_filename = f"direction_{angle_match.group(1)}.png"
                    else:
                        img_filename = f"img_{idx:02d}.png"
                elif parent_dir == 'global_map':
                    img_filename = "global_map.png"
                elif parent_dir == 'local_map':
                    img_filename = "local_map.png"
                elif parent_dir == 'detection':
                    img_filename = "detection.png"
                elif parent_dir == 'rgb':
                    img_filename = "rgb.png"
                else:
                    img_filename = f"img_{idx:02d}.png"
                save_path = os.path.join(save_dir, img_filename)
                with open(save_path, 'wb') as f:
                    f.write(image_bytes)
        
        return content
    
    def call_api(self, prompt: str, image_paths: List[str], save_dir: str = None,
                 no_compress_indices: set = None) -> Optional[Dict]:
        """调用API（带计时和速度统计）
        
        Args:
            prompt: prompt文本
            image_paths: 图片路径列表
            save_dir: 如果指定，在发送时同步保存压缩图片+prompt到此目录
            no_compress_indices: 不压缩的图片索引集合（如 {4} 表示第5张图不压缩）
        """
        t_start = time.time()
        try:
            # 确保 prompt 是 UTF-8 编码的字符串
            if isinstance(prompt, bytes):
                prompt = prompt.decode('utf-8')
            
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
            payload = self._apply_reasoning_disabled_defaults(payload)
            
            # 构建headers（支持OpenRouter优化）
            headers = self.config.get_headers()
            is_openrouter = 'openrouter' in self.config.base_url.lower()
            if is_openrouter:
                # OpenRouter: 固定优先走阿里云 Tongyi 后端（Qwen 系列在此最快，~90-100 TPS）
                # 若阿里云不可用则 fallback 到其他 provider
                headers["X-Title"] = "SpaceVLN"
                payload["provider"] = {
                    "order": ["Alibaba"],         # 固定优先走阿里云（最高吞吐）
                    "allow_fallbacks": True       # 阿里云不可用时允许回退
                }
            
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
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
                if len(payload['messages'][0]['content']) > 0:
                    first_item = payload['messages'][0]['content'][0]
                    if first_item.get('type') == 'text':
                        print(f"  [DEBUG] Prompt length: {len(first_item.get('text', ''))} chars")
                    elif first_item.get('type') == 'image_url':
                        print(f"  [DEBUG] First item is image_url")
                
                return None
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # 提取token用量
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
