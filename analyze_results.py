#!/usr/bin/env python3
"""
SpaceVLN 结果分析脚本
=========================
分析评估结果，计算汇总统计指标，并检查指标计算逻辑

使用方法:
    python analyze_results.py --path data/vlm_navigation --save
    
指标说明:
1. Distance to Goal: 智能体最后位置与目标点的测地线距离(geodesic distance)
   - 来源: Habitat DistanceToGoal measure
   - 计算: sim.geodesic_distance(agent_position, goal_positions)
   
2. Success: 是否在SUCCESS_DISTANCE(3米)内成功到达
   - 来源: Habitat Success measure  
   - 计算: distance_to_goal < SUCCESS_DISTANCE
   
3. SPL (Success weighted by Path Length):
   - 公式: SPL = success * (shortest_path_length / max(actual_path_length, shortest_path_length))
   - 如果成功但绕路，SPL会降低
   - 如果失败，SPL = 0
   
4. Oracle Success: 整个轨迹中是否曾经进入过3米内
   - 计算: min(所有step的distance_to_goal) < SUCCESS_DISTANCE
   
5. Oracle Navigation Error: 轨迹中与目标的最小距离
   - 计算: min(所有step的distance_to_goal)
"""
import os
import argparse
import json
import math
import csv
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
        results_dir: 结果根目录（包含log/子目录）
    
    Returns:
        results: Episode结果列表
    """
    log_dir = os.path.join(results_dir, "log")
    
    if not os.path.exists(log_dir):
        print(f"❌ Log目录不存在: {log_dir}")
        return []
    
    results = []
    for filename in sorted(os.listdir(log_dir)):
        if filename.startswith("episode_") and filename.endswith(".json"):
            filepath = os.path.join(log_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"⚠️  读取文件失败 {filename}: {e}")
    
    return results


def compute_metrics(results: List[Dict]) -> Dict[str, Any]:
    """
    计算汇总统计指标
    
    Args:
        results: Episode结果列表
    
    Returns:
        metrics: 汇总指标字典
    """
    if not results:
        return {}
    
    n = len(results)
    
    # 提取所有指标（带数据验证）
    success_list = [check_inf_nan(r.get('success', 0)) for r in results]
    spl_list = [check_inf_nan(r.get('spl', 0.0)) for r in results]
    dtg_list = [check_inf_nan(r.get('distance_to_goal', -1)) for r in results]
    path_length_list = [check_inf_nan(r.get('path_length', 0.0)) for r in results]
    steps_list = [check_inf_nan(r.get('total_steps', 0)) for r in results]
    
    oracle_success_list = [check_inf_nan(r.get('oracle_success', 0)) for r in results]
    oracle_error_list = [check_inf_nan(r.get('oracle_navigation_error', float('inf'))) for r in results]
    oracle_spl_list = [check_inf_nan(r.get('oracle_spl', 0.0)) for r in results]
    
    subtask_list = [check_inf_nan(r.get('subtask_count', 0)) for r in results]
    
    # 过滤有效值（距离 >= 0）
    valid_dtg = [d for d in dtg_list if d >= 0]
    valid_oracle_error = [e for e in oracle_error_list if e != float('inf') and e >= 0]
    
    # 计算成功率
    success_count = sum(success_list)
    success_rate = success_count / n if n > 0 else 0.0
    
    oracle_success_count = sum(oracle_success_list)
    oracle_success_rate = oracle_success_count / n if n > 0 else 0.0
    
    # 计算平均SPL
    avg_spl = sum(spl_list) / n if n > 0 else 0.0
    avg_oracle_spl = sum(oracle_spl_list) / n if n > 0 else 0.0
    
    # 计算平均距离
    avg_dtg = sum(valid_dtg) / len(valid_dtg) if valid_dtg else -1
    avg_oracle_error = sum(valid_oracle_error) / len(valid_oracle_error) if valid_oracle_error else -1
    
    # 计算平均路径长度和步数
    avg_path_length = sum(path_length_list) / n if n > 0 else 0.0
    avg_steps = sum(steps_list) / n if n > 0 else 0.0
    avg_subtasks = sum(subtask_list) / n if n > 0 else 0.0
    
    metrics = {
        'total_episodes': n,
        'success_count': success_count,
        'success_rate': success_rate,
        'oracle_success_count': oracle_success_count,
        'oracle_success_rate': oracle_success_rate,
        'avg_spl': avg_spl,
        'avg_oracle_spl': avg_oracle_spl,
        'avg_distance_to_goal': avg_dtg,
        'avg_oracle_navigation_error': avg_oracle_error,
        'avg_path_length': avg_path_length,
        'avg_steps': avg_steps,
        'avg_subtasks': avg_subtasks,
        'detailed_results': results  # 保留详细数据用于调试
    }
    
    return metrics


def print_summary(metrics: Dict[str, Any]):
    """打印汇总报告"""
    n = metrics['total_episodes']
    
    print("\n" + "="*80)
    print("📊 SpaceVLN 评估结果汇总")
    print("="*80)
    
    print(f"\n🎯 核心指标:")
    print(f"  Success rate:       {metrics['success_count']}/{n} ({metrics['success_rate']:.3f})")
    print(f"  Oracle success rate: {metrics['oracle_success_count']}/{n} ({metrics['oracle_success_rate']:.3f})")
    print(f"  SPL:                {metrics['avg_spl']:.3f}")
    print(f"  Oracle SPL:         {metrics['avg_oracle_spl']:.3f}")
    
    print(f"\n📏 距离指标:")
    print(f"  Distance to goal:           {metrics['avg_distance_to_goal']:.3f}m")
    print(f"  Oracle navigation error:    {metrics['avg_oracle_navigation_error']:.3f}m")
    print(f"  Path length:                {metrics['avg_path_length']:.3f}m")
    
    print(f"\n⚙️  执行统计:")
    print(f"  Average steps:     {metrics['avg_steps']:.1f}")
    print(f"  Average subtasks:  {metrics['avg_subtasks']:.1f}")
    
    print(f"\n{'='*80}")


def print_debug_info(metrics: Dict[str, Any]):
    """打印调试信息，检查指标计算"""
    print(f"\n🔍 指标计算调试信息:")
    print(f"{'='*80}")
    
    results = metrics['detailed_results']
    print(f"\nEpisode详情:")
    print(f"{'ID':<6} {'Success':<8} {'DTG(m)':<10} {'SPL':<8} {'Path(m)':<10} {'Steps':<6}")
    print(f"{'-'*60}")
    
    for r in results:
        ep_id = r.get('episode_id', '?')
        success = r.get('success', 0)
        dtg = r.get('distance_to_goal', -1)
        spl = r.get('spl', 0.0)
        path = r.get('path_length', 0.0)
        steps = r.get('total_steps', 0)
        
        print(f"{ep_id:<6} {success:<8} {dtg:<10.3f} {spl:<8.4f} {path:<10.3f} {steps:<6}")
    
    # 检查指标异常
    print(f"\n⚠️  异常检测:")
    for r in results:
        ep_id = r.get('episode_id', '?')
        success = r.get('success', 0)
        dtg = r.get('distance_to_goal', -1)
        spl = r.get('spl', 0.0)
        oracle_success = r.get('oracle_success', 0)
        oracle_error = r.get('oracle_navigation_error', float('inf'))
        
        # 检查1: Success=1但DTG>3米 
        if success == 1 and dtg > 3.0:
            print(f"  ❌ Episode {ep_id}: Success=1 但 DTG={dtg:.3f}m > 3m (不应该成功)")
        
        # 检查2: Success=0但DTG<3米
        if success == 0 and 0 <= dtg < 3.0:
            print(f"  ⚠️  Episode {ep_id}: Success=0 但 DTG={dtg:.3f}m < 3m (应该成功)")
        
        # 检查3: SPL计算异常
        if success == 0 and spl > 0:
            print(f"  ❌ Episode {ep_id}: Success=0 但 SPL={spl:.4f} > 0 (不应该有SPL)")
        
        # 检查4: Oracle异常
        if oracle_success == 1 and oracle_error > 3.0:
            print(f"  ❌ Episode {ep_id}: Oracle Success=1 但 Oracle Error={oracle_error:.3f}m > 3m")
        
        # 检查5: 最终距离异常（DTG应该是停止时的距离）
        if dtg < 0:
            print(f"  ⚠️  Episode {ep_id}: DTG={dtg} < 0 (距离数据无效)")
    
    print(f"{'='*80}")


def save_summary(metrics: Dict[str, Any], output_path: str):
    """保存汇总报告到文件"""
    n = metrics['total_episodes']
    
    content = f"""
================================================================================
📊 SpaceVLN 评估结果汇总
================================================================================

🎯 核心指标:
  Success rate:       {metrics['success_count']}/{n} ({metrics['success_rate']:.3f})
  Oracle success rate: {metrics['oracle_success_count']}/{n} ({metrics['oracle_success_rate']:.3f})
  SPL:                {metrics['avg_spl']:.3f}
  Oracle SPL:         {metrics['avg_oracle_spl']:.3f}

📏 距离指标:
  Distance to goal:           {metrics['avg_distance_to_goal']:.3f}m
  Oracle navigation error:    {metrics['avg_oracle_navigation_error']:.3f}m
  Path length:                {metrics['avg_path_length']:.3f}m

⚙️  执行统计:
  Average steps:     {metrics['avg_steps']:.1f}
  Average subtasks:  {metrics['avg_subtasks']:.1f}

================================================================================

💾 汇总报告已保存: {output_path}

✅ 分析完成
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return content


def _format_metric_value(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)


def save_episode_tables(results: List[Dict[str, Any]], metrics: Dict[str, Any], results_dir: str) -> Dict[str, str]:
    """保存逐episode结果表格（CSV + Markdown）。"""
    csv_path = os.path.join(results_dir, "episode_results.csv")
    md_path = os.path.join(results_dir, "episode_results.md")

    sorted_results = sorted(results, key=lambda item: int(item.get("episode_id", -1)))
    headers = [
        "episode_id",
        "success",
        "spl",
        "distance_to_goal",
        "path_length",
        "total_steps",
        "subtask_count",
        "oracle_success",
        "oracle_navigation_error",
        "oracle_spl",
        "success_rate",
        "instruction",
        "timestamp",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for item in sorted_results:
            writer.writerow({
                "episode_id": item.get("episode_id", ""),
                "success": item.get("success", 0),
                "spl": _format_metric_value(check_inf_nan(item.get("spl", 0.0)), 4),
                "distance_to_goal": _format_metric_value(check_inf_nan(item.get("distance_to_goal", -1.0)), 3),
                "path_length": _format_metric_value(check_inf_nan(item.get("path_length", 0.0)), 3),
                "total_steps": item.get("total_steps", 0),
                "subtask_count": item.get("subtask_count", 0),
                "oracle_success": item.get("oracle_success", 0),
                "oracle_navigation_error": _format_metric_value(check_inf_nan(item.get("oracle_navigation_error", -1.0)), 3),
                "oracle_spl": _format_metric_value(check_inf_nan(item.get("oracle_spl", 0.0)), 4),
                "success_rate": "",
                "instruction": item.get("instruction", ""),
                "timestamp": item.get("timestamp", ""),
            })
        writer.writerow({
            "episode_id": "TOTAL",
            "success": f"{metrics['success_count']}/{metrics['total_episodes']}",
            "spl": _format_metric_value(metrics["avg_spl"], 4),
            "distance_to_goal": _format_metric_value(metrics["avg_distance_to_goal"], 3),
            "path_length": _format_metric_value(metrics["avg_path_length"], 3),
            "total_steps": _format_metric_value(metrics["avg_steps"], 1),
            "subtask_count": _format_metric_value(metrics["avg_subtasks"], 1),
            "oracle_success": f"{metrics['oracle_success_count']}/{metrics['total_episodes']}",
            "oracle_navigation_error": _format_metric_value(metrics["avg_oracle_navigation_error"], 3),
            "oracle_spl": _format_metric_value(metrics["avg_oracle_spl"], 4),
            "success_rate": _format_metric_value(metrics["success_rate"], 4),
            "instruction": "best result kept per episode",
            "timestamp": datetime.now().isoformat(),
        })

    md_lines = [
        "# Episode Results",
        "",
        "| Episode | Success | SPL | DTG(m) | Path(m) | Steps | Subtasks | Oracle | OracleErr(m) | Oracle SPL |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in sorted_results:
        md_lines.append(
            "| {episode} | {success} | {spl} | {dtg} | {path} | {steps} | {subtasks} | {oracle_success} | {oracle_err} | {oracle_spl} |".format(
                episode=item.get("episode_id", ""),
                success=item.get("success", 0),
                spl=_format_metric_value(check_inf_nan(item.get("spl", 0.0)), 4),
                dtg=_format_metric_value(check_inf_nan(item.get("distance_to_goal", -1.0)), 3),
                path=_format_metric_value(check_inf_nan(item.get("path_length", 0.0)), 3),
                steps=item.get("total_steps", 0),
                subtasks=item.get("subtask_count", 0),
                oracle_success=item.get("oracle_success", 0),
                oracle_err=_format_metric_value(check_inf_nan(item.get("oracle_navigation_error", -1.0)), 3),
                oracle_spl=_format_metric_value(check_inf_nan(item.get("oracle_spl", 0.0)), 4),
            )
        )
    md_lines.extend([
        "",
        "## Summary",
        "",
        "| Total Episodes | Success Rate | Avg SPL | Avg DTG(m) | Avg Path(m) | Avg Steps | Avg Subtasks | Oracle Success Rate | Avg Oracle SPL |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        "| {total} | {success_count}/{total} ({success_rate}) | {avg_spl} | {avg_dtg} | {avg_path} | {avg_steps} | {avg_subtasks} | {oracle_count}/{total} ({oracle_rate}) | {avg_oracle_spl} |".format(
            total=metrics["total_episodes"],
            success_count=metrics["success_count"],
            success_rate=_format_metric_value(metrics["success_rate"], 4),
            avg_spl=_format_metric_value(metrics["avg_spl"], 4),
            avg_dtg=_format_metric_value(metrics["avg_distance_to_goal"], 3),
            avg_path=_format_metric_value(metrics["avg_path_length"], 3),
            avg_steps=_format_metric_value(metrics["avg_steps"], 1),
            avg_subtasks=_format_metric_value(metrics["avg_subtasks"], 1),
            oracle_count=metrics["oracle_success_count"],
            oracle_rate=_format_metric_value(metrics["oracle_success_rate"], 4),
            avg_oracle_spl=_format_metric_value(metrics["avg_oracle_spl"], 4),
        ),
        "",
        "> Note: repeated evaluation of the same episode keeps only the better result in `log/episode_XXX.json`.",
    ])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return {"csv": csv_path, "md": md_path}


def main():
    parser = argparse.ArgumentParser(description="分析VLN评估结果")
    parser.add_argument("--path", type=str, required=True, help="结果目录路径")
    parser.add_argument("--save", action="store_true", help="保存汇总报告到summary.txt")
    parser.add_argument("--debug", action="store_true", help="显示调试信息")
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"❌ 目录不存在: {args.path}")
        return
    
    # 加载结果
    print(f"📂 加载结果: {args.path}")
    results = load_results(args.path)
    
    if not results:
        print(f"⚠️  未找到任何episode结果")
        return
    
    print(f"✅ 加载了 {len(results)} 个episode")
    
    # 计算指标
    metrics = compute_metrics(results)
    
    # 打印汇总
    print_summary(metrics)
    
    # 打印调试信息
    if args.debug:
        print_debug_info(metrics)
    
    # 保存报告
    if args.save:
        summary_path = os.path.join(args.path, "summary.txt")
        content = save_summary(metrics, summary_path)
        table_paths = save_episode_tables(results, metrics, args.path)
        print(content)
        print(f"📋 Episode表格已保存: {table_paths['csv']}")
        print(f"📋 Markdown表格已保存: {table_paths['md']}")


if __name__ == "__main__":
    main()
