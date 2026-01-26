#!/usr/bin/env python3
"""
MapReAct-VLN 结果分析脚本
=========================
分析评估结果，计算汇总统计指标
参考Sub-VLM-VLN的analyze_results.py实现

使用方法:
    python analyze_results.py --path results/experiment_name
"""
import os
import argparse
import json
import math
from typing import List, Dict, Any
from datetime import datetime


def check_inf_nan(value):
    """检查并修正无效值（inf/nan）"""
    if isinstance(value, (int, float)):
        if math.isinf(value) or math.isnan(value):
            return 0
    return value


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """
    从log目录加载所有episode结果
    
    Args:
        results_dir: 结果目录路径
        
    Returns:
        结果字典列表
    """
    log_dir = os.path.join(results_dir, 'log')
    if not os.path.exists(log_dir):
        print(f"❌ Log目录不存在: {log_dir}")
        return []
    
    results = []
    json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
    
    print(f"📁 找到 {len(json_files)} 个结果文件")
    
    for json_file in sorted(json_files):
        filepath = os.path.join(log_dir, json_file)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # 数据验证和清洗
                cleaned_data = {
                    'episode_id': data.get('episode_id', 'unknown'),
                    'success': int(check_inf_nan(data.get('success', 0))),
                    'spl': float(check_inf_nan(data.get('spl', 0.0))),
                    'distance_to_goal': float(check_inf_nan(data.get('distance_to_goal', 0.0))),
                    'path_length': float(check_inf_nan(data.get('path_length', 0.0))),
                    'oracle_success': int(check_inf_nan(data.get('oracle_success', 0))),
                    'oracle_navigation_error': float(check_inf_nan(data.get('oracle_navigation_error', float('inf')))),
                    'oracle_spl': float(check_inf_nan(data.get('oracle_spl', 0.0))),
                    'total_steps': data.get('total_steps', 0),
                    'subtask_count': data.get('subtask_count', 0),
                }
                results.append(cleaned_data)
                
        except json.JSONDecodeError as e:
            print(f"⚠️  解析失败: {json_file} - {e}")
        except Exception as e:
            print(f"⚠️  读取失败: {json_file} - {e}")
    
    return results


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    计算汇总统计指标
    
    Args:
        results: 结果列表
        
    Returns:
        统计指标字典
    """
    if not results:
        return {}
    
    n = len(results)
    
    stats = {
        'num_episodes': n,
        'success_rate': sum(r['success'] for r in results) / n,
        'oracle_success_rate': sum(r['oracle_success'] for r in results) / n,
        'avg_spl': sum(r['spl'] for r in results) / n,
        'avg_oracle_spl': sum(r['oracle_spl'] for r in results) / n,
        'avg_distance_to_goal': sum(r['distance_to_goal'] for r in results) / n,
        'avg_oracle_navigation_error': sum(
            r['oracle_navigation_error'] for r in results 
            if r['oracle_navigation_error'] != float('inf')
        ) / len([r for r in results if r['oracle_navigation_error'] != float('inf')]) if any(
            r['oracle_navigation_error'] != float('inf') for r in results
        ) else 0.0,
        'avg_path_length': sum(r['path_length'] for r in results) / n,
        'avg_steps': sum(r['total_steps'] for r in results) / n,
        'avg_subtasks': sum(r['subtask_count'] for r in results) / n,
    }
    
    return stats


def print_statistics(stats: Dict[str, float], results: List[Dict[str, Any]]):
    """
    打印统计结果（参考Sub-VLM-VLN格式）
    
    Args:
        stats: 统计指标字典
        results: 原始结果列表
    """
    n = stats['num_episodes']
    success_count = int(stats['success_rate'] * n)
    oracle_success_count = int(stats['oracle_success_rate'] * n)
    
    print("\n" + "="*80)
    print("📊 MapReAct-VLN 评估结果汇总")
    print("="*80)
    
    print(f"\n🎯 核心指标:")
    print(f"  Success rate:       {success_count}/{n} ({stats['success_rate']:.3f})")
    print(f"  Oracle success rate: {oracle_success_count}/{n} ({stats['oracle_success_rate']:.3f})")
    print(f"  SPL:                {stats['avg_spl']:.3f}")
    print(f"  Oracle SPL:         {stats['avg_oracle_spl']:.3f}")
    
    print(f"\n📏 距离指标:")
    print(f"  Distance to goal:           {stats['avg_distance_to_goal']:.3f}m")
    print(f"  Oracle navigation error:    {stats['avg_oracle_navigation_error']:.3f}m")
    print(f"  Path length:                {stats['avg_path_length']:.3f}m")
    
    print(f"\n⚙️  执行统计:")
    print(f"  Average steps:     {stats['avg_steps']:.1f}")
    print(f"  Average subtasks:  {stats['avg_subtasks']:.1f}")
    
    print("\n" + "="*80)


def save_summary(results_dir: str, stats: Dict[str, float], results: List[Dict[str, Any]]):
    """
    保存汇总报告
    
    Args:
        results_dir: 结果目录
        stats: 统计指标
        results: 原始结果列表
    """
    summary_file = os.path.join(results_dir, "summary.txt")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(summary_file, 'w') as f:
        f.write("#"*80 + "\n")
        f.write(f"MapReAct-VLN 评估汇总报告 | {now_str}\n")
        f.write("#"*80 + "\n\n")
        
        n = stats['num_episodes']
        success_count = int(stats['success_rate'] * n)
        oracle_success_count = int(stats['oracle_success_rate'] * n)
        
        f.write(f"评估Episode数: {n}\n\n")
        
        f.write("测试的Episode列表:\n")
        for i, result in enumerate(results, 1):
            f.write(f"  {i}. Episode {result['episode_id']}\n")
        f.write("\n")
        
        f.write("评估指标汇总:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'success_rate':<30s}: {stats['success_rate']:.4f} ({success_count}/{n})\n")
        f.write(f"{'oracle_success_rate':<30s}: {stats['oracle_success_rate']:.4f} ({oracle_success_count}/{n})\n")
        f.write(f"{'spl':<30s}: {stats['avg_spl']:.4f}\n")
        f.write(f"{'oracle_spl':<30s}: {stats['avg_oracle_spl']:.4f}\n")
        f.write(f"{'distance_to_goal':<30s}: {stats['avg_distance_to_goal']:.4f}\n")
        f.write(f"{'oracle_navigation_error':<30s}: {stats['avg_oracle_navigation_error']:.4f}\n")
        f.write(f"{'path_length':<30s}: {stats['avg_path_length']:.4f}\n")
        f.write(f"{'avg_steps':<30s}: {stats['avg_steps']:.2f}\n")
        f.write(f"{'avg_subtasks':<30s}: {stats['avg_subtasks']:.2f}\n")
        f.write("\n")
        
        # 详细结果
        f.write("\n详细结果:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Episode':<12} {'Success':<8} {'SPL':<8} {'DTG':<10} {'Path':<10} {'Steps':<8}\n")
        f.write("-"*80 + "\n")
        for r in results:
            f.write(f"{str(r['episode_id']):<12} "
                   f"{r['success']:<8} "
                   f"{r['spl']:<8.3f} "
                   f"{r['distance_to_goal']:<10.3f} "
                   f"{r['path_length']:<10.3f} "
                   f"{r['total_steps']:<8}\n")
    
    print(f"\n💾 汇总报告已保存: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="分析MapReAct-VLN评估结果")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="结果目录路径（包含log/子目录）"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存汇总报告到summary.txt"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"❌ 结果目录不存在: {args.path}")
        return
    
    # 加载结果
    results = load_results(args.path)
    
    if not results:
        print("❌ 未找到有效的结果文件")
        return
    
    # 计算统计
    stats = calculate_statistics(results)
    
    # 打印结果
    print_statistics(stats, results)
    
    # 保存汇总（如果指定）
    if args.save:
        save_summary(args.path, stats, results)
    
    print("\n✅ 分析完成")


if __name__ == "__main__":
    main()
