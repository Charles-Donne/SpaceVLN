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
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class APIConfig:
    """统一API配置类"""
    
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 验证必要字段
        required = ['api_key', 'base_url', 'model']
        missing = [f for f in required if f not in self.config or not self.config[f]]
        if missing:
            raise ValueError(f"配置文件缺少必要字段: {', '.join(missing)}")
    
    @property
    def api_key(self) -> str:
        return self.config['api_key']
    
    @property
    def base_url(self) -> str:
        return self.config['base_url']
    
    @property
    def model(self) -> str:
        return self.config['model']
    
    @property
    def temperature(self) -> float:
        return self.config.get('temperature', 0.1)
    
    @property
    def max_tokens(self) -> int:
        return self.config.get('max_tokens', 2000)
    
    @property
    def timeout(self) -> int:
        return self.config.get('timeout', 60)
    
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
        self.compress_images = True
        self.compression_max_size = 384  # 最大边长（像素）
        self.compression_quality = 80    # JPEG质量
    
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
        
        # 处理图片
        if should_compress:
            compressed_path = self.compress_image(
                image_path, 
                max_size=self.compression_max_size,
                quality=self.compression_quality
            )
            with open(compressed_path, "rb") as f:
                result = base64.b64encode(f.read()).decode("utf-8")
            
            # 清理临时文件（如果不是原始文件）
            if compressed_path != image_path:
                try:
                    os.remove(compressed_path)
                except:
                    pass
            return result
        else:
            # 不压缩，直接编码
            file_size = os.path.getsize(image_path)
            if file_size > 5 * 1024 * 1024:  # 5MB
                print(f"[WARN] Large image: {os.path.basename(image_path)} ({file_size / 1024 / 1024:.2f}MB)")
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    
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
    
    def parse_json_response(self, response_text: str) -> Optional[Dict]:
        """解析JSON响应"""
        try:
            cleaned = self.clean_json_response(response_text)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            json_str = self.extract_json_object(response_text)
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
        
        # 保存到当前工作目录下的failed_vlm_responses文件夹
        failed_dir = "failed_vlm_responses"
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
    
    def build_message_content(self, text: str, image_paths: List[str], save_dir: str = None) -> List[Dict]:
        """构建消息内容，可选保存压缩后的图片（即模型实际看到的版本）
        
        Args:
            text: prompt文本
            image_paths: 图片路径列表
            save_dir: 如果指定，将压缩后的图片和prompt保存到此目录
        """
        content = [{"type": "text", "text": text}]
        
        # 如果需要保存，先创建目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            # 保存prompt
            with open(os.path.join(save_dir, "prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(text)
        
        for idx, img_path in enumerate(image_paths):
            img_base64 = self.encode_image_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
            
            # 保存压缩后的图片（模型实际看到的版本）
            # 从路径提取语义名称: directions/xxx -> direction_000.png, global_map/xxx -> global_map.png
            if save_dir:
                parent_dir = os.path.basename(os.path.dirname(img_path))
                if parent_dir == 'directions':
                    # directions/initial_direction_030.png -> direction_030.png
                    basename = os.path.basename(img_path)
                    # 提取角度部分: initial_direction_030.png -> 030
                    import re
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
                    f.write(base64.b64decode(img_base64))
        
        return content
    
    def call_api(self, prompt: str, image_paths: List[str], save_dir: str = None) -> Optional[Dict]:
        """调用API（带计时和速度统计）
        
        Args:
            prompt: prompt文本
            image_paths: 图片路径列表
            save_dir: 如果指定，在发送时同步保存压缩图片+prompt到此目录
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
                    "content": self.build_message_content(prompt, image_paths, save_dir=save_dir)
                }],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            # 构建headers（支持OpenRouter优化）
            headers = self.config.get_headers()
            is_openrouter = 'openrouter' in self.config.base_url.lower()
            if is_openrouter:
                # OpenRouter: 选择最快的provider
                headers["X-Title"] = "MapReAct-VLN"
                payload["provider"] = {
                    "sort": "throughput",        # 按吞吐量排序，选最快provider
                    "allow_fallbacks": True       # 允许回退到其他provider
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
