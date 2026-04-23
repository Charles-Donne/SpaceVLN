"""
全景图生成模块
=============
负责从多张图像拼接生成全景图，并添加方向标注

功能：
- 使用OpenCV Stitcher拼接3张图像生成90°全景图
- 跨平台TrueType字体支持（macOS/Linux/Windows）
- 自动添加方向和角度标注
"""

import os
import cv2
import numpy as np
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont


class PanoramaGenerator:
    """全景图生成器"""
    
    def __init__(self):
        """初始化全景图生成器"""
        self.font = None
        self.font_size = 40
        self._load_font()
    
    def _load_font(self):
        """加载跨平台TrueType字体"""
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux (Debian/Ubuntu)
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",  # Linux (CentOS/RHEL)
            "C:/Windows/Fonts/arial.ttf",  # Windows
        ]
        
        for font_path in font_paths:
            try:
                self.font = ImageFont.truetype(font_path, self.font_size)
                return
            except:
                continue
        
        print("  [WARNING] TrueType font not found, will use OpenCV fallback")
    
    def create_panorama(self, images: List[np.ndarray], direction_name: str = "") -> np.ndarray:
        """
        从3张图像拼接生成90°全景图并添加方向标注
        
        Args:
            images: 3张图像列表（按顺序：左-中-右）
            direction_name: 方向名称（如"Front", "Left 90°", "Back 180°", "Right 90°"）
            
        Returns:
            90°全景图（带方向标注）
        """
        # 使用OpenCV Stitcher拼接
        try:
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            status, panorama = stitcher.stitch(images)
            
            if status != cv2.Stitcher_OK:
                # 拼接失败，返回水平拼接的降级版本
                print(f"  ⚠️  Stitcher failed (status={status}), using fallback horizontal concat")
                panorama = np.hstack(images)
        except cv2.error as e:
            # OpenCV错误（如特征点不足），使用降级方案
            print(f"  ⚠️  Stitcher error ({str(e)[:50]}...), using fallback horizontal concat")
            panorama = np.hstack(images)
        
        # 添加方向标注
        if direction_name:
            panorama = self._add_direction_labels(panorama, direction_name)
        
        return panorama
    
    def _add_direction_labels(self, panorama: np.ndarray, direction_name: str) -> np.ndarray:
        """
        在全景图上添加方向标注（白边红字）
        
        Args:
            panorama: 全景图
            direction_name: 方向名称
            
        Returns:
            带标注的全景图
        """
        h, w = panorama.shape[:2]
        
        # 获取标签映射
        direction_key = direction_name.split()[0]
        label_map = {
            "Front": {"left": "Left 30°", "center": "Front 0°", "right": "Right 30°"},
            "Left": {"left": "Left 120°", "center": "Left 90°", "right": "Left 60°"},
            "Back": {"left": "Right 150°", "center": "Back 180°", "right": "Left 150°"},
            "Right": {"left": "Right 60°", "center": "Right 90°", "right": "Right 120°"},
        }
        labels = label_map.get(direction_key, {"left": "Left", "center": "Center", "right": "Right"})
        
        # 标注位置
        positions = [
            (int(w * 0.15), 30),  # 左侧
            (int(w * 0.50), 30),  # 中间
            (int(w * 0.85), 30),  # 右侧
        ]
        
        # 使用PIL绘制（支持Unicode度数符号）
        if self.font is not None:
            panorama = self._draw_labels_with_pil(panorama, labels, positions)
        else:
            # 回退到OpenCV（不支持Unicode，使用"deg"代替°）
            panorama = self._draw_labels_with_opencv(panorama, labels, positions, direction_key)
        
        return panorama
    
    def _draw_labels_with_pil(self, panorama: np.ndarray, labels: Dict[str, str], 
                              positions: List[tuple]) -> np.ndarray:
        """使用PIL绘制标签（支持Unicode）"""
        # 转换为PIL格式
        pil_img = Image.fromarray(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        for pos, key in zip(positions, ["left", "center", "right"]):
            text = labels[key]
            
            # 计算文字边界框以居中
            bbox = draw.textbbox((0, 0), text, font=self.font)
            text_width = bbox[2] - bbox[0]
            text_x = pos[0] - text_width // 2
            text_y = pos[1]
            
            # 绘制白色边框（描边效果）
            for offset_x in range(-2, 3):
                for offset_y in range(-2, 3):
                    if offset_x != 0 or offset_y != 0:
                        draw.text((text_x + offset_x, text_y + offset_y), 
                                text, font=self.font, fill=(255, 255, 255))
            
            # 绘制红色文字
            draw.text((text_x, text_y), text, font=self.font, fill=(255, 0, 0))
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def _draw_labels_with_opencv(self, panorama: np.ndarray, labels: Dict[str, str],
                                 positions: List[tuple], direction_key: str) -> np.ndarray:
        """使用OpenCV绘制标签（不支持Unicode，使用deg代替°）"""
        # 替换度数符号
        label_map_deg = {
            "Front": {"left": "Left 30deg", "center": "Front 0deg", "right": "Right 30deg"},
            "Left": {"left": "Left 120deg", "center": "Left 90deg", "right": "Left 60deg"},
            "Back": {"left": "Right 150deg", "center": "Back 180deg", "right": "Left 150deg"},
            "Right": {"left": "Right 60deg", "center": "Right 90deg", "right": "Right 120deg"},
        }
        labels_deg = label_map_deg.get(direction_key, labels)
        
        for pos, key in zip(positions, ["left", "center", "right"]):
            text = labels_deg[key]
            font_thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, font_thickness)
            text_x = pos[0] - text_width // 2
            text_y = pos[1]
            
            # 白色边框
            for offset_x in range(-2, 3):
                for offset_y in range(-2, 3):
                    if offset_x != 0 or offset_y != 0:
                        cv2.putText(panorama, text, (text_x + offset_x, text_y + offset_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), font_thickness)
            # 红色文字
            cv2.putText(panorama, text, (text_x, text_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), font_thickness)
        
        return panorama
