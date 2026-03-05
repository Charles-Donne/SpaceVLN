"""
数据保存管理模块
================
统一管理VLM导航系统中所有数据的保存逻辑

目录结构:
data/vlm_navigation/episode_XXX/
├── rgb/, detection/, global_map/, local_map/  # 每步的观察和地图
├── panoramas/                                  # 环视全景图
├── thinking/subtask_Na/                        # LLM规划详细输入输出（带尝试标识）
├── action/                                     # VLM动作（按subtask组织）
│   └── subtask_Na/                             # 带尝试标识
│       ├── info.json                           # 子任务信息
│       └── step_X/                             # 每步的VLM输入输出
└── records/                                    # 所有摘要记录
    ├── thinking_summary.json                   # LLM调用摘要（按子任务分组）
    │   └── {subtask_id: {phase: record}}
    ├── action_summary.json                     # VLM调用摘要（按子任务分组）
    │   └── {subtask_id: {step_X: record}}
    ├── waypoint_memory.json                    # 路径点记录
    └── result.json                             # 最终结果

子任务标识格式: 1a, 1b, 1c... (同一子任务的多次尝试)
summary结构示例:
{
  "1a": {
    "initial_planning": {...},
    "verify_1a": {...}
  },
  "1b": {
    "verify_1b": {...}
  },
  "2a": {
    "verify_2a": {...}
  }
}
"""
import os
import json
import shutil
from typing import Dict, List, Optional
from datetime import datetime


class SaveManager:
    """数据保存管理器"""
    
    def __init__(self, dump_dir: str, episode_id: int):
        """
        初始化保存管理器
        
        Args:
            dump_dir: 数据保存根目录
            episode_id: Episode ID
        """
        self.dump_dir = dump_dir
        self.episode_id = episode_id
        self.episode_dir = os.path.join(dump_dir, f"episode_{episode_id}")
        self.records_dir = os.path.join(self.episode_dir, "records")  # 统一的记录目录
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.records_dir, exist_ok=True)
    
    def save_thinking_input(self, thinking_record: Dict) -> str:
        """
        保存LLM思考输入（图片 + prompt）
        在调用API之前调用，确保输入数据被保存
        
        Returns:
            thinking_dir: 保存目录路径
        """
        # 使用subtask_count作为目录名，不再使用attempt字母
        subtask_count = thinking_record.get('subtask_count', 1)
        
        thinking_dir = os.path.join(self.episode_dir, "thinking", f"subtask_{subtask_count}")
        os.makedirs(thinking_dir, exist_ok=True)
        
        # 保存输入图片
        if 'input_images' in thinking_record:
            images_dir = os.path.join(thinking_dir, "input_images")
            os.makedirs(images_dir, exist_ok=True)
            for img_name, img_path in thinking_record['input_images'].items():
                if img_path and os.path.exists(img_path):
                    shutil.copy(img_path, os.path.join(images_dir, img_name))
        
        # 保存prompt
        if 'prompt' in thinking_record:
            with open(os.path.join(thinking_dir, "prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(thinking_record['prompt'])
        
        return thinking_dir
    
    def save_thinking_response(self, thinking_record: Dict, thinking_dir: str = None):
        """
        保存LLM思考输出（response）
        在API调用成功后调用
        
        Args:
            thinking_record: 思考记录（必须包含response）
            thinking_dir: 保存目录（如果None则重新计算）
        """
        if not thinking_dir:
            subtask_count = thinking_record.get('subtask_count', 1)
            thinking_dir = os.path.join(self.episode_dir, "thinking", f"subtask_{subtask_count}")
        
        # 保存response
        if 'response' in thinking_record:
            with open(os.path.join(thinking_dir, "response.json"), 'w', encoding='utf-8') as f:
                json.dump(thinking_record['response'], f, ensure_ascii=False, indent=2)
    
    def save_thinking(self, thinking_record: Dict):
        """
        保存LLM思考输出（完整版本，一次性保存输入+输出）
        兼容旧代码，新代码应使用 save_thinking_input + save_thinking_response
        
        结构: thinking/subtask_Xa/
          - subtask_1a/ - 初始规划
          - subtask_1b/ - 验证未完成，继续尝试（保存verify_1a的结果到1b）
          - subtask_2a/ - 验证完成，新子任务（保存verify_1a的结果到2a）
          
        每个目录包含:
            - input_images/ (输入图片)
            - prompt.txt (prompt)
            - response.json (响应)
        
        同时更新records/thinking_summary.json汇总文件
        """
        thinking_dir = self.save_thinking_input(thinking_record)
        if 'response' in thinking_record:
            self.save_thinking_response(thinking_record, thinking_dir)
    
    def save_action_input(self, action_record: Dict, subtask_info: Optional[Dict] = None) -> str:
        """
        保存VLM动作输入（图片 + prompt）
        在调用API之前调用，确保输入数据被保存
        
        Returns:
            action_dir: 保存目录路径
        """
        # 使用subtask_id（如 "1a", "1b"）作为目录名
        subtask_id = action_record.get('subtask_id', f"{action_record.get('subtask_count', 1)}a")
        step = action_record.get('step', 0)
        
        # 子任务目录
        subtask_dir = os.path.join(self.episode_dir, "action", f"subtask_{subtask_id}")
        os.makedirs(subtask_dir, exist_ok=True)
        
        # 保存子任务信息（首次创建时）
        if subtask_info:
            info_file = os.path.join(subtask_dir, "info.json")
            if not os.path.exists(info_file):  # 只在第一次保存
                with open(info_file, 'w', encoding='utf-8') as f:
                    json.dump(subtask_info, f, ensure_ascii=False, indent=2)
        
        # 步骤目录
        action_dir = os.path.join(subtask_dir, f"step_{step}")
        os.makedirs(action_dir, exist_ok=True)
        
        # 保存输入图片
        if 'input_images' in action_record:
            images_dir = os.path.join(action_dir, "input_images")
            os.makedirs(images_dir, exist_ok=True)
            for img_name, img_path in action_record['input_images'].items():
                if img_path and os.path.exists(img_path):
                    shutil.copy(img_path, os.path.join(images_dir, img_name))
        
        # 保存prompt
        if 'prompt' in action_record:
            with open(os.path.join(action_dir, "prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(action_record['prompt'])
        
        return action_dir
    
    def save_action_response(self, action_record: Dict):
        """
        保存VLM动作响应和prompt
        在API返回后调用
        """
        # 使用subtask_id（如 "1a", "1b"）作为目录名
        subtask_id = action_record.get('subtask_id', f"{action_record.get('subtask_count', 1)}a")
        step = action_record.get('step', 0)
        
        # 步骤目录（应该已经存在）
        subtask_dir = os.path.join(self.episode_dir, "action", f"subtask_{subtask_id}")
        action_dir = os.path.join(subtask_dir, f"step_{step}")
        
        # 保存prompt（如果有）
        if 'prompt' in action_record and action_record['prompt']:
            with open(os.path.join(action_dir, "prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(action_record['prompt'])
        
        # 保存response
        with open(os.path.join(action_dir, "response.json"), 'w', encoding='utf-8') as f:
            json.dump(action_record.get('response', {}), f, ensure_ascii=False, indent=2)
    
    def save_action(self, action_record: Dict, subtask_info: Optional[Dict] = None):
        """
        保存VLM动作输出（兼容旧代码，内部调用save_action_input和save_action_response）
        
        结构: action/subtask_Na/ (a/b/c标识尝试次数)
            - info.json (子任务信息：destination、instruction等)
            - step_X/
                - input_images/ (输入图片)
                - response.json (响应)
        
        同时更新records/action_summary.json汇总文件
        """
        self.save_action_input(action_record, subtask_info)
        self.save_action_response(action_record)
    
    def save_waypoint_memory(self, waypoint_memory: List[Dict], 
                            instruction: str, current_step: int):
        """保存路径点记忆到records/ - DISABLED for performance"""
        pass  # 不再保存waypoint_memory，减少IO开销
    
    def save_result(self, result: Dict):
        """
        保存最终结果到两个位置:
        1. episode_xxx/records/result.json (详细结果)
        2. log/episode_xxx.json (用于analyze_results.py分析)
        """
        # 保存到records/目录（详细结果）
        result_path = os.path.join(self.records_dir, "result.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # 保存到log/目录（用于结果分析）
        log_dir = os.path.join(self.dump_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        
        log_path = os.path.join(log_dir, f"episode_{self.episode_id}.json")
        
        # 提取关键指标用于分析
        log_result = {
            'episode_id': result['episode_id'],
            'instruction': result.get('instruction', ''),
            'total_steps': result.get('total_steps', 0),
            
            # 核心导航指标（用于analyze_results.py）
            'success': result.get('success', 0),
            'spl': result.get('spl', 0.0),
            'distance_to_goal': result.get('distance_to_goal', -1),
            'path_length': result.get('path_length', 0.0),
            
            # Oracle指标
            'oracle_success': result.get('oracle_success', 0),
            'oracle_navigation_error': result.get('oracle_navigation_error', float('inf')),
            'oracle_spl': result.get('oracle_spl', 0.0),
        }
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Results saved:")
        print(f"  {result_path}")
        print(f"  {log_path}")
        print(f"  Steps: {result.get('total_steps', 0)} | Subtasks: {result.get('subtask_count', 0)}")
        print(f"  Success: {result.get('success', 0)} | SPL: {result.get('spl', 0.0):.4f}")
        print(f"  Distance to Goal: {result.get('distance_to_goal', -1):.3f}m")
        print(f"{'='*60}")
        
        return log_path
    
    def _update_summary_file(self, filename: str, record: Dict, exclude_keys: List[str] = None):
        """
        DEPRECATED - Summary files removed for performance
        保留此方法以保持向后兼容
        """
        pass  # 不再生成summary文件，减少IO开销
