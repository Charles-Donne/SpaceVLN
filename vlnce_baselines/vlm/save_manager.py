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
    
    def save_thinking(self, thinking_record: Dict):
        """
        保存LLM思考输出
        
        结构: thinking/subtask_Xa/
          - subtask_1a/ - 初始规划
          - subtask_1b/ - 验证未完成，继续尝试
          - subtask_2a/ - 验证完成，新子任务
          
        每个目录包含:
            - input_images/ (输入图片)
            - prompt.txt (prompt)
            - response.json (响应)
        
        同时更新records/thinking_summary.json汇总文件
        """
        # 使用subtask_id（如 "1a", "2b"）构建目录名
        subtask_id = thinking_record.get('subtask_id', f"{thinking_record.get('subtask_count', 1)}a")
        thinking_dir = os.path.join(self.episode_dir, "thinking", f"subtask_{subtask_id}")
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
        
        # 保存response
        with open(os.path.join(thinking_dir, "response.json"), 'w', encoding='utf-8') as f:
            json.dump(thinking_record['response'], f, ensure_ascii=False, indent=2)
        
        # 更新汇总文件到records/
        self._update_summary_file("thinking_summary.json", thinking_record, 
                                 exclude_keys=['input_images'])
    
    def save_action(self, action_record: Dict, subtask_info: Optional[Dict] = None):
        """
        保存VLM动作输出
        
        结构: action/subtask_Na/ (a/b/c标识尝试次数)
            - info.json (子任务信息：destination、instruction等)
            - step_X/
                - input_images/ (输入图片)
                - response.json (响应)
        
        同时更新records/action_summary.json汇总文件
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
        
        # 保存response
        with open(os.path.join(action_dir, "response.json"), 'w', encoding='utf-8') as f:
            json.dump(action_record.get('response', {}), f, ensure_ascii=False, indent=2)
        
        # 更新汇总文件到records/
        self._update_summary_file("action_summary.json", action_record, 
                                 exclude_keys=['input_images'])
    
    def save_waypoint_memory(self, waypoint_memory: List[Dict], 
                            instruction: str, current_step: int):
        """保存路径点记忆到records/"""
        data = {
            "episode_id": self.episode_id,
            "instruction": instruction,
            "waypoints": waypoint_memory,
            "total_waypoints": len(waypoint_memory),
            "last_updated_step": current_step
        }
        
        filepath = os.path.join(self.records_dir, 'waypoint_memory.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_result(self, result: Dict):
        """保存最终导航结果到records/"""
        filepath = os.path.join(self.records_dir, 'result.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"📁 结果已保存: {filepath}")
        print(f"   Steps: {result.get('total_steps', 0)} | Subtasks: {result.get('subtask_count', 0)}")
        print(f"   Detected Classes: {len(result.get('detected_classes', []))}")
        print(f"   Thinking(LLM): {result.get('thinking_count', 0)} | Action(VLM): {result.get('action_count', 0)}")
        if result.get('metrics'):
            metrics = result['metrics']
            print(f"   Success: {metrics.get('success', False)} | SPL: {metrics.get('spl', 0.0):.4f}")
        print(f"{'='*60}")
        
        return filepath
    
    def _update_summary_file(self, filename: str, record: Dict, exclude_keys: List[str] = None):
        """
        更新records/目录下的汇总JSON文件
        
        结构:
        - thinking_summary.json: 按子任务分组 {subtask_id: {phase: record}}
        - action_summary.json: 按子任务分组 {subtask_id: {step: record}}
        """
        filepath = os.path.join(self.records_dir, filename)
        
        # 过滤不需要的键
        if exclude_keys:
            summary_record = {k: v for k, v in record.items() if k not in exclude_keys}
        else:
            summary_record = record
        
        # 读取现有记录
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
        
        # 根据文件类型组织结构
        if filename == "thinking_summary.json":
            # thinking: {subtask_id: {phase: record}}
            subtask_id = summary_record.get('subtask_id', f"{summary_record.get('subtask_count', 1)}a")
            phase = summary_record.get('phase', 'unknown')
            
            if subtask_id not in data:
                data[subtask_id] = {}
            data[subtask_id][phase] = summary_record
            
        elif filename == "action_summary.json":
            # action: {subtask_id: {step_X: record}}
            subtask_id = summary_record.get('subtask_id', f"{summary_record.get('subtask_count', 1)}a")
            step = summary_record.get('step', 0)
            
            if subtask_id not in data:
                data[subtask_id] = {}
            data[subtask_id][f"step_{step}"] = summary_record
        else:
            # 其他文件保持列表格式
            if not isinstance(data, list):
                data = []
            data.append(summary_record)
        
        # 保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
