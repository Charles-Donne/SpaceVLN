"""
Navigation visualization utilities.

This module saves RGB + top-down composites and optionally assembles GIFs.
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
    Save per-step navigation visualizations and optional episode GIFs.
    """
    
    
    def __init__(
        self,
        output_dir: str,
        save_step_images: bool = False,
        keep_frames_for_gif: bool = True,
    ):
        """Initialize the visualization helper."""
        self.output_dir = output_dir
        self.visualization_dir = None
        self.video_frames = []
        self.last_top_down_map = None
        self.save_step_images = bool(save_step_images)
        self.keep_frames_for_gif = bool(keep_frames_for_gif)
        if self.save_step_images or self.keep_frames_for_gif:
            os.makedirs(output_dir, exist_ok=True)
    
    def setup_maps_dir(self, episode_dir: str):
        """Prepare the episode visualization directory."""
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
        """Save one RGB + top-down composite frame with text overlays."""
        if not self.visualization_dir or "rgb" not in observations:
            return None
        
        rgb = observations["rgb"]
        
        top_down_map = self._extract_top_down_map(info or {}, rgb, step)
        self.last_top_down_map = top_down_map.copy() if top_down_map is not None else None
        
        combined = np.concatenate((rgb, top_down_map), axis=1)
        combined = self._add_text_overlay(
            combined, 
            instruction, 
            current_subtask, 
            step, 
            distance,
            action
        )
        
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
                print(f"⚠️  [Step {step}] Failed to render `{key}`: {exc}")

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
            print("⚠️  Missing `top_down_map`, `top_down_map_vlnce`, and `global_map_input` fallback in info")
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
        """Append the standard text panel below the composite frame."""
        img = image.copy()
        h, w = img.shape[:2]
        
        text_height = 120
        text_area = np.zeros((text_height, w, 3), dtype=np.uint8)
        text_area.fill(40)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        
        y_offset = 25
        
        action_safe = self._safe_overlay_text(action, "[Action]")
        metrics_text = f"Step: {step} | Distance: {distance:.2f}m | Action: {action_safe}"
        cv2.putText(text_area, metrics_text, (10, y_offset), font, font_scale, (0, 255, 255), thickness)
        y_offset += 30
        
        instruction_safe = self._safe_overlay_text(instruction, "[Instruction]")
        
        instruction_lines = self._wrap_text(instruction_safe, w - 20, font, font_scale)
        for line in instruction_lines[:2]:
            cv2.putText(text_area, line, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 25
        
        if current_subtask:
            y_offset += 5
            subtask_safe = self._safe_overlay_text(current_subtask, "[Subtask]")
            
            subtask_text = f"Subtask: {subtask_safe}"
            subtask_lines = self._wrap_text(subtask_text, w - 20, font, font_scale)
            for line in subtask_lines[:1]:
                cv2.putText(text_area, line, (10, y_offset), font, font_scale, (0, 255, 0), thickness)
        
        result = np.vstack([img, text_area])
        return result
    
    def _wrap_text(self, text: str, max_width: int, font, font_scale: float) -> List[str]:
        """Wrap overlay text to fit within the target pixel width."""
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
        Save all buffered frames as a GIF animation.

        Args:
            output_path: Optional output path under `visualization/`.
            fps: Frames per second.

        Returns:
            GIF path, or `None` on failure.
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
            # Convert frames to uint8
            frames_rgb = []
            for frame in self.video_frames:
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                frames_rgb.append(frame)
            
            # Compute per-frame duration
            duration = 1.0 / fps
            
            # Save the GIF
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
        """Clear buffered video frames."""
        self.video_frames = []
        self.last_top_down_map = None
