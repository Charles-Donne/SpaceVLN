"""
路径点管理模块
==============
管理导航过程中的空间记忆和路径点标注
"""
import numpy as np
import cv2
from typing import List, Dict, Optional


class WaypointManager:
    """路径点管理器"""
    
    def __init__(self):
        """初始化路径点管理器"""
        self.waypoint_memory: List[Dict] = []
        self.waypoint_counter: int = 0
    
    def add_waypoint(self, waypoint_description: str) -> int:
        """
        添加路径点到空间记忆
        
        Args:
            waypoint_description: 路径点描述（格式: "<Area Type> - <Key Landmarks>"）
            
        Returns:
            waypoint_id: 新添加的路径点ID
        """
        self.waypoint_counter += 1
        waypoint_id = self.waypoint_counter
        
        waypoint = {
            "id": waypoint_id,
            "waypoint": waypoint_description
        }
        
        self.waypoint_memory.append(waypoint)
        
        print(f"  📍 Waypoint #{waypoint_id} 已记录: {waypoint_description}")
        
        return waypoint_id
    
    def get_summary(self) -> str:
        """
        获取路径点摘要（用于LLM提示词）
        
        Returns:
            路径点摘要字符串
        """
        if not self.waypoint_memory:
            return ""
        
        summary_lines = []
        for wp in self.waypoint_memory:
            summary_lines.append(f"#{wp['id']}: {wp['waypoint']}")
        
        return "\n".join(summary_lines)
    
    def visualize_on_map(self, map_image: np.ndarray, 
                         trajectory_points: List[np.ndarray]) -> np.ndarray:
        """
        在地图上可视化路径点
        
        Args:
            map_image: 输入地图图像
            trajectory_points: 轨迹点列表（来自mapper）
            
        Returns:
            标注后的地图图像
        """
        if not self.waypoint_memory or not trajectory_points:
            return map_image
        
        annotated = map_image.copy()
        
        # 为每个waypoint标注（使用对应的trajectory点）
        for i, waypoint in enumerate(self.waypoint_memory):
            if i < len(trajectory_points):
                pos = trajectory_points[i]
                
                # 绘制深红色圆圈
                cv2.circle(annotated, tuple(pos), 15, (0, 0, 139), 3)
                
                # 绘制白色编号
                wp_id = waypoint['id']
                text = f"{wp_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                
                # 计算文本大小以居中
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = pos[0] - text_size[0] // 2
                text_y = pos[1] + text_size[1] // 2
                
                # 绘制白色文本
                cv2.putText(annotated, text, (text_x, text_y), 
                          font, font_scale, (255, 255, 255), thickness)
        
        return annotated
    
    def reset(self):
        """重置waypoint管理器"""
        self.waypoint_memory = []
        self.waypoint_counter = 0
    
    def get_memory(self) -> List[Dict]:
        """获取完整的waypoint记忆"""
        return self.waypoint_memory
    
    def get_count(self) -> int:
        """获取waypoint总数"""
        return len(self.waypoint_memory)
