"""
动作解析器
==========
将VLM输出的复合动作（如90度旋转、0.5米前进）分解为环境可执行的基本动作序列

基本动作单位：
- TURN_LEFT: 30° (单次)
- TURN_RIGHT: 30° (单次)
- MOVE_FORWARD: 0.25m (单次)
- STOP: 到达目标

复合动作示例：
- TURN_LEFT 90° → [TURN_LEFT, TURN_LEFT, TURN_LEFT]
- MOVE_FORWARD 0.5m → [MOVE_FORWARD, MOVE_FORWARD]
- TURN_RIGHT 180° → [TURN_RIGHT] * 6
"""

from typing import List, Dict, Any, Optional
import json


class ActionParser:
    """动作解析器 - 将VLM复合动作转换为基本动作序列"""
    
    # 基本动作常量
    BASE_TURN_ANGLE = 30  # 度
    BASE_MOVE_DISTANCE = 0.25  # 米
    
    # 动作类型
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    MOVE_FORWARD = "MOVE_FORWARD"
    STOP = "STOP"
    
    # 有效动作范围
    VALID_TURN_DEGREES = [30]
    VALID_MOVE_METERS = [0.25, 0.5, 0.75, 1.0, 1.25]
    
    def __init__(self):
        """初始化动作解析器"""
        pass
    
    def parse_action(self, vlm_response: Dict[str, Any]) -> Optional[List[str]]:
        """
        解析VLM响应，生成基本动作序列
        
        Args:
            vlm_response: VLM输出的动作字典
                - action: "TURN_LEFT" | "TURN_RIGHT" | "MOVE_FORWARD" | "STOP"
                - degrees: 旋转角度 (仅TURN_LEFT/RIGHT需要)
                - meters: 前进距离 (仅MOVE_FORWARD需要)
        
        Returns:
            基本动作序列列表，例如 ["TURN_LEFT", "TURN_LEFT", "MOVE_FORWARD"]
            如果解析失败返回 None
        """
        if not vlm_response or 'action' not in vlm_response:
            print("⚠️  VLM响应缺少action字段")
            return None
        
        action = vlm_response['action'].upper()
        
        # 处理STOP动作
        if action == self.STOP:
            return [self.STOP]
        
        # 处理旋转动作
        if action in [self.TURN_LEFT, self.TURN_RIGHT]:
            degrees = vlm_response.get('degrees')
            if degrees is None:
                print(f"⚠️  {action} 缺少degrees参数")
                return None
            
            return self._parse_rotation(action, degrees)
        
        # 处理前进动作
        if action == self.MOVE_FORWARD:
            meters = vlm_response.get('meters')
            if meters is None:
                print(f"⚠️  MOVE_FORWARD 缺少meters参数")
                return None
            
            return self._parse_movement(meters)
        
        print(f"⚠️  未知动作类型: {action}")
        return None
    
    def _parse_rotation(self, action: str, degrees: float) -> Optional[List[str]]:
        """
        解析旋转动作
        
        Args:
            action: "TURN_LEFT" 或 "TURN_RIGHT"
            degrees: 目标旋转角度
        
        Returns:
            基本旋转动作序列，例如 ["TURN_LEFT", "TURN_LEFT", "TURN_LEFT"] (90度)
        """
        # 验证角度有效性
        if degrees not in self.VALID_TURN_DEGREES:
            print(f"⚠️  无效旋转角度: {degrees}°，必须是 {self.VALID_TURN_DEGREES} 之一")
            return None
        
        # 计算需要的基本动作次数
        num_steps = int(degrees / self.BASE_TURN_ANGLE)
        
        if num_steps <= 0:
            print(f"⚠️  旋转角度过小: {degrees}°")
            return None
        
        # 生成动作序列
        return [action] * num_steps
    
    def _parse_movement(self, meters: float) -> Optional[List[str]]:
        """
        解析前进动作
        
        Args:
            meters: 目标前进距离（米）
        
        Returns:
            基本前进动作序列，例如 ["MOVE_FORWARD", "MOVE_FORWARD"] (0.5米)
        """
        # 验证距离有效性
        if meters not in self.VALID_MOVE_METERS:
            print(f"⚠️  无效前进距离: {meters}m，必须是 {self.VALID_MOVE_METERS} 之一")
            return None
        
        # 计算需要的基本动作次数
        num_steps = int(round(meters / self.BASE_MOVE_DISTANCE))
        
        if num_steps <= 0:
            print(f"⚠️  前进距离过小: {meters}m")
            return None
        
        # 生成动作序列
        return [self.MOVE_FORWARD] * num_steps
    
    def get_action_description(self, action_sequence: List[str]) -> str:
        """
        获取动作序列的人类可读描述
        
        Args:
            action_sequence: 基本动作序列
        
        Returns:
            描述字符串，例如 "Turn left 90° (3 steps)" 或 "Move forward 0.5m (2 steps)"
        """
        if not action_sequence:
            return "No action"
        
        if action_sequence[0] == self.STOP:
            return "STOP"
        
        action = action_sequence[0]
        count = len(action_sequence)
        
        if action in [self.TURN_LEFT, self.TURN_RIGHT]:
            degrees = count * self.BASE_TURN_ANGLE
            direction = "left" if action == self.TURN_LEFT else "right"
            return f"Turn {direction} {degrees}° ({count} steps)"
        
        if action == self.MOVE_FORWARD:
            meters = count * self.BASE_MOVE_DISTANCE
            return f"Move forward {meters}m ({count} steps)"
        
        return f"Unknown action: {action}"
    
    def validate_response(self, vlm_response: Dict[str, Any]) -> bool:
        """
        验证VLM响应的完整性和合法性
        
        Args:
            vlm_response: VLM输出字典
        
        Returns:
            是否合法
        """
        if not vlm_response or 'action' not in vlm_response:
            return False
        
        action = vlm_response['action'].upper()
        
        # STOP动作不需要额外参数
        if action == self.STOP:
            return True
        
        # 旋转动作需要degrees参数
        if action in [self.TURN_LEFT, self.TURN_RIGHT]:
            degrees = vlm_response.get('degrees')
            if degrees is None or degrees not in self.VALID_TURN_DEGREES:
                return False
            return True
        
        # 前进动作需要meters参数
        if action == self.MOVE_FORWARD:
            meters = vlm_response.get('meters')
            if meters is None or meters not in self.VALID_MOVE_METERS:
                return False
            return True
        
        return False
    
    def get_total_rotation(self, action_history: List[Dict[str, Any]]) -> int:
        """
        计算累计旋转角度（正数=左转，负数=右转）
        
        Args:
            action_history: VLM动作历史列表
        
        Returns:
            累计旋转角度（度）
        """
        total = 0
        for action_dict in action_history:
            action = action_dict.get('action', '').upper()
            degrees = action_dict.get('degrees', 0)
            
            if action == self.TURN_LEFT:
                total += degrees
            elif action == self.TURN_RIGHT:
                total -= degrees
        
        return total
    
    def get_total_distance(self, action_history: List[Dict[str, Any]]) -> float:
        """
        计算累计前进距离
        
        Args:
            action_history: VLM动作历史列表
        
        Returns:
            累计前进距离（米）
        """
        total = 0.0
        for action_dict in action_history:
            action = action_dict.get('action', '').upper()
            meters = action_dict.get('meters', 0.0)
            
            if action == self.MOVE_FORWARD:
                total += meters
        
        return total


# 测试代码
if __name__ == "__main__":
    parser = ActionParser()
    
    # 测试旋转
    print("=== 测试旋转动作 ===")
    response1 = {"action": "TURN_LEFT", "degrees": 90}
    seq1 = parser.parse_action(response1)
    print(f"输入: {response1}")
    print(f"输出: {seq1}")
    print(f"描述: {parser.get_action_description(seq1)}")
    print()
    
    # 测试前进
    print("=== 测试前进动作 ===")
    response2 = {"action": "MOVE_FORWARD", "meters": 0.75}
    seq2 = parser.parse_action(response2)
    print(f"输入: {response2}")
    print(f"输出: {seq2}")
    print(f"描述: {parser.get_action_description(seq2)}")
    print()
    
    # 测试STOP
    print("=== 测试STOP动作 ===")
    response3 = {"action": "STOP"}
    seq3 = parser.parse_action(response3)
    print(f"输入: {response3}")
    print(f"输出: {seq3}")
    print(f"描述: {parser.get_action_description(seq3)}")
    print()
    
    # 测试无效输入
    print("=== 测试无效输入 ===")
    response4 = {"action": "TURN_LEFT", "degrees": 45}  # 无效角度
    seq4 = parser.parse_action(response4)
    print(f"输入: {response4}")
    print(f"输出: {seq4}")
    print()
    
    # 测试累计统计
    print("=== 测试累计统计 ===")
    history = [
        {"action": "TURN_LEFT", "degrees": 90},
        {"action": "MOVE_FORWARD", "meters": 0.5},
        {"action": "TURN_RIGHT", "degrees": 60},
        {"action": "MOVE_FORWARD", "meters": 0.25}
    ]
    print(f"动作历史: {history}")
    print(f"累计旋转: {parser.get_total_rotation(history)}° (正=左，负=右)")
    print(f"累计距离: {parser.get_total_distance(history)}m")
