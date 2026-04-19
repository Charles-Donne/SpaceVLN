"""
导航可视化模块
=============
负责保存RGB+俯视图拼接可视化、生成GIF动画

参考Sub-VLM-VLN的实现细节，确保俯视图正确显示
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Optional
from habitat.utils.visualizations import maps

from navigation_system.runtime.storage.naming import build_subtask_name_from_token

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("⚠️  imageio not installed, GIF generation disabled")


class NavigationVisualizer:
    """
    导航可视化器
    
    负责:
    - RGB + 俯视图拼接可视化
    - 文本信息叠加
    - GIF动画生成
    """
    
    
    def __init__(
        self,
        output_dir: str,
        save_step_images: bool = False,
        keep_frames_for_gif: bool = True,
    ):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.visualization_dir = None
        self.video_frames = []
        self.last_top_down_map = None
        self.save_step_images = bool(save_step_images)
        self.keep_frames_for_gif = bool(keep_frames_for_gif)
        if self.save_step_images or self.keep_frames_for_gif:
            os.makedirs(output_dir, exist_ok=True)
    
    def setup_maps_dir(self, episode_dir: str):
        """
        设置可视化目录（RGB+俯视图拼接）
        
        Args:
            episode_dir: Episode输出根目录
        """
        self.visualization_dir = os.path.join(episode_dir, "visualization")
        if self.save_step_images or self.keep_frames_for_gif:
            os.makedirs(self.visualization_dir, exist_ok=True)
        self.video_frames = []
        self.last_top_down_map = None

    @staticmethod
    def _safe_overlay_text(text: Optional[str], fallback: str) -> str:
        """Keep overlay text ASCII-safe for OpenCV rendering."""
        try:
            safe_text = str(text or "").encode('ascii', 'ignore').decode('ascii')
        except Exception:
            safe_text = ""
        safe_text = safe_text.strip()
        return safe_text if safe_text else fallback
    
    def save_step_visualization(self,
                                observations: Dict,
                                info: Dict,
                                step: int,
                                instruction: str,
                                current_subtask: str = None,
                                distance: float = 0.0,
                                action: str = "",
                                subtask_id: str = None) -> Optional[str]:
        """
        保存单步可视化：左边第一人称视角 + 右边俯视图 + 文本信息
        
        参考Sub-VLM-VLN的实现，确保俯视图正确显示
        
        Args:
            observations: 环境观测字典（需包含"rgb"键）
            info: 环境指标字典（需包含"top_down_map_vlnce"键）
            step: 当前步数
            instruction: 全局导航指令
            current_subtask: 当前子任务指令（可选）
            distance: 到目标距离
            action: 当前执行的动作名称
            subtask_id: 子任务标识（会统一落盘成 "subtask1" 这种形式）
            
        Returns:
            保存的图像路径，失败返回None
        """
        if not self.visualization_dir or "rgb" not in observations:
            return None
        
        # 获取第一人称RGB
        rgb = observations["rgb"]
        
        top_down_map = self._extract_top_down_map(info or {}, rgb, step)
        self.last_top_down_map = top_down_map.copy() if top_down_map is not None else None
        
        # 拼接：左边RGB + 右边俯视图
        combined = np.concatenate((rgb, top_down_map), axis=1)
        
        # 添加文本信息
        combined = self._add_text_overlay(
            combined, 
            instruction, 
            current_subtask, 
            step, 
            distance,
            action
        )
        
        # 保存（RGB格式需要转换为BGR）
        # 文件名格式：step_0001_subtask1.png（统一使用PNG格式）
        if subtask_id:
            filename = f"step_{step:04d}_{build_subtask_name_from_token(subtask_id)}.png"
        else:
            filename = f"step_{step:04d}.png"
        filepath = None
        if self.save_step_images:
            filepath = os.path.join(self.visualization_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        if self.keep_frames_for_gif:
            self.video_frames.append(combined)
        
        return filepath

    def _extract_top_down_map(self, info: Dict, rgb: np.ndarray, step: int) -> np.ndarray:
        """Resolve the best available top-down / trajectory map for the current step."""
        for key in ("top_down_map_vlnce", "top_down_map"):
            if key not in info or info[key] is None:
                continue
            try:
                return maps.colorize_draw_agent_and_fit_to_height(info[key], rgb.shape[0])
            except Exception as exc:
                print(f"⚠️  [Step {step}] {key} 渲染失败: {exc}")

        fallback = info.get("global_map_input")
        fallback_meta = fallback if isinstance(fallback, dict) else {}
        if isinstance(fallback, dict):
            fallback = fallback.get("image_array")
        if isinstance(fallback, np.ndarray) and fallback.size > 0:
            image = fallback
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3 and image.shape[-1] == 3:
                color_space = str(fallback_meta.get("color_space", "bgr")).lower()
                if color_space == "bgr":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return cv2.resize(
                image,
                (rgb.shape[0], rgb.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        if step == 1:
            print("⚠️  info中没有 top_down_map/top_down_map_vlnce，且没有 global_map fallback")
        return np.zeros_like(rgb)

    def save_final_top_down_map(self, output_path: str = None) -> Optional[str]:
        """Persist the final top-down/trajectory map as a standalone image."""
        if self.last_top_down_map is None or not self.visualization_dir:
            return None
        if output_path is None:
            output_path = os.path.join(self.visualization_dir, "topdown_trajectory_final.png")
        try:
            image = self.last_top_down_map
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return output_path if os.path.exists(output_path) else None
        except Exception as exc:
            print(f"⚠️  Failed to save final top-down map: {exc}")
            return None
    
    def _add_text_overlay(self,
                          image: np.ndarray,
                          instruction: str,
                          current_subtask: Optional[str],
                          step: int,
                          distance: float,
                          action: str = "") -> np.ndarray:
        """
        在图像底部添加文本信息（完全按照Sub-VLM-VLN的实现）
        
        Args:
            image: 拼接后的RGB图像
            instruction: 全局指令
            current_subtask: 当前子任务
            step: 步数
            distance: 距离
            action: 动作名称
            
        Returns:
            添加文本后的图像
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        # 创建文本区域（深灰色背景）
        text_height = 120
        text_area = np.zeros((text_height, w, 3), dtype=np.uint8)
        text_area.fill(40)  # 深灰色背景
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)  # 白色
        
        y_offset = 25
        
        # 第1行：步数、距离、动作（青色高亮）
        action_safe = self._safe_overlay_text(action, "[Action]")
        metrics_text = f"Step: {step} | Distance: {distance:.2f}m | Action: {action_safe}"
        cv2.putText(text_area, metrics_text, (10, y_offset), font, font_scale, (0, 255, 255), thickness)
        y_offset += 30
        
        # 第2-3行：全局指令（白色，最多2行）
        # 处理中文字符：先编码为ASCII可表示的形式
        instruction_safe = self._safe_overlay_text(instruction, "[Instruction]")
        
        instruction_lines = self._wrap_text(instruction_safe, w - 20, font, font_scale)
        for line in instruction_lines[:2]:
            cv2.putText(text_area, line, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 25
        
        # 第4行：当前子任务（绿色，最多1行）
        if current_subtask:
            y_offset += 5
            subtask_safe = self._safe_overlay_text(current_subtask, "[Subtask]")
            
            subtask_text = f"Subtask: {subtask_safe}"
            subtask_lines = self._wrap_text(subtask_text, w - 20, font, font_scale)
            for line in subtask_lines[:1]:
                cv2.putText(text_area, line, (10, y_offset), font, font_scale, (0, 255, 0), thickness)
        
        # 拼接文本区域到图像底部（使用vstack而不是concatenate）
        result = np.vstack([img, text_area])
        return result
    
    def _wrap_text(self, text: str, max_width: int, font, font_scale: float) -> List[str]:
        """
        文本自动换行（完全按照Sub-VLM-VLN的实现）
        
        Args:
            text: 要换行的文本
            max_width: 最大宽度（像素）
            font: OpenCV字体
            font_scale: 字体缩放
            
        Returns:
            换行后的文本列表
        """
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, 1)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def save_gif(self, output_path: str = None, fps: int = 2) -> Optional[str]:
        """
        将所有帧保存为GIF动画
        
        Args:
            output_path: 输出路径（可选，默认在visualization目录下）
            fps: 帧率
            
        Returns:
            GIF路径，失败返回None
        """
        if not self.keep_frames_for_gif or not self.video_frames:
            return None
        
        if not HAS_IMAGEIO:
            print("⚠️  imageio not installed, cannot create GIF")
            return None
        
        if output_path is None and self.visualization_dir:
            output_path = os.path.join(self.visualization_dir, "navigation.gif")
        
        if not output_path:
            return None
        
        try:
            # 转换帧为uint8格式
            frames_rgb = []
            for frame in self.video_frames:
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                frames_rgb.append(frame)
            
            # 计算每帧持续时间
            duration = 1.0 / fps
            
            # 保存GIF
            imageio.mimsave(output_path, frames_rgb, duration=duration, loop=0)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            else:
                print("✗ GIF file creation failed")
                return None
                
        except Exception as e:
            print(f"✗ Error saving GIF: {e}")
            return None

    def cleanup_step_images(self) -> int:
        """Delete saved step PNG frames after GIF generation to save disk space."""
        if not self.visualization_dir or not os.path.isdir(self.visualization_dir):
            return 0

        removed_count = 0
        for filename in sorted(os.listdir(self.visualization_dir)):
            if not (filename.startswith("step_") and filename.endswith(".png")):
                continue
            path = os.path.join(self.visualization_dir, filename)
            try:
                os.remove(path)
                removed_count += 1
            except FileNotFoundError:
                continue
            except Exception as exc:
                print(f"⚠️  Failed to remove visualization frame {path}: {exc}")
        return removed_count
    
    def clear_frames(self):
        """清空视频帧列表"""
        self.video_frames = []
        self.last_top_down_map = None
