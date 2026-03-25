#!/usr/bin/env python3
"""
SpaceVLN 结果分析脚本
=========================
分析评估结果，计算汇总统计指标，并检查指标计算逻辑

使用方法:
    python analyze_results.py --path data/vlm_navigation --save
    
指标说明:
1. NE: Navigation Error，最终距离目标的测地线距离，越低越好
2. OSR: Oracle Success Rate，轨迹中是否曾进入目标阈值范围
3. SR: Success Rate，最终停止位置是否成功
4. SPL: Success weighted by Path Length
5. nDTW: 轨迹与GT路径的一致性，越高越好
"""
import os
import argparse
import json
import math
import csv
from typing import List, Dict, Any


def check_inf_nan(value):
    """检查并修正无效值（inf/nan）"""
    if isinstance(value, (int, float)):
        if math.isinf(value) or math.isnan(value):
            return 0
    return value


def get_metric(result: Dict[str, Any], *keys: str, default: Any):
    """按优先级读取指标，兼容新旧字段名。"""
    for key in keys:
        if key in result and result[key] is not None:
            return result[key]
    return default


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
    
    ne_list = [check_inf_nan(get_metric(r, 'ne', 'distance_to_goal', default=-1.0)) for r in results]
    osr_list = [check_inf_nan(get_metric(r, 'osr', 'oracle_success', default=0)) for r in results]
    sr_list = [check_inf_nan(get_metric(r, 'sr', 'success', default=0)) for r in results]
    spl_list = [check_inf_nan(get_metric(r, 'spl', default=0.0)) for r in results]
    ndtw_list = [check_inf_nan(get_metric(r, 'ndtw', 'nDTW', default=0.0)) for r in results]

    valid_ne = [value for value in ne_list if value >= 0]
    sr_count = sum(sr_list)
    osr_count = sum(osr_list)
    avg_spl = sum(spl_list) / n if n > 0 else 0.0

    metrics = {
        'total_episodes': n,
        'avg_ne': sum(valid_ne) / len(valid_ne) if valid_ne else -1.0,
        'osr_count': osr_count,
        'avg_osr': osr_count / n if n > 0 else 0.0,
        'sr_count': sr_count,
        'avg_sr': sr_count / n if n > 0 else 0.0,
        'avg_spl': avg_spl,
        'avg_ndtw': sum(ndtw_list) / n if n > 0 else 0.0,
        'detailed_results': results  # 保留详细数据用于调试
    }
    
    return metrics


def print_summary(metrics: Dict[str, Any]):
    """打印汇总报告"""
    n = metrics['total_episodes']
    
    print("\n" + "="*80)
    print("📊 SpaceVLN 评估结果汇总")
    print("="*80)

    print(f"\n🎯 统一指标:")
    print(f"  NE:    {metrics['avg_ne']:.3f}m")
    print(f"  OSR:   {metrics['osr_count']}/{n} ({metrics['avg_osr']:.3f})")
    print(f"  SR:    {metrics['sr_count']}/{n} ({metrics['avg_sr']:.3f})")
    print(f"  SPL:   {metrics['avg_spl']:.3f}")
    print(f"  nDTW:  {metrics['avg_ndtw']:.3f}")
    
    print(f"\n{'='*80}")


def print_debug_info(metrics: Dict[str, Any]):
    """打印调试信息，检查指标计算"""
    print(f"\n🔍 指标计算调试信息:")
    print(f"{'='*80}")
    
    results = metrics['detailed_results']
    print(f"\nEpisode详情:")
    print(f"{'ID':<6} {'NE(m)':<10} {'OSR':<6} {'SR':<6} {'SPL':<8} {'nDTW':<8}")
    print(f"{'-'*56}")
    
    for r in results:
        ep_id = r.get('episode_id', '?')
        ne = check_inf_nan(get_metric(r, 'ne', 'distance_to_goal', default=-1.0))
        osr = check_inf_nan(get_metric(r, 'osr', 'oracle_success', default=0))
        sr = check_inf_nan(get_metric(r, 'sr', 'success', default=0))
        spl = check_inf_nan(get_metric(r, 'spl', default=0.0))
        ndtw = check_inf_nan(get_metric(r, 'ndtw', 'nDTW', default=0.0))
        
        print(f"{ep_id:<6} {ne:<10.3f} {osr:<6} {sr:<6} {spl:<8.4f} {ndtw:<8.4f}")
    
    # 检查指标异常
    print(f"\n⚠️  异常检测:")
    for r in results:
        ep_id = r.get('episode_id', '?')
        ne = check_inf_nan(get_metric(r, 'ne', 'distance_to_goal', default=-1.0))
        osr = check_inf_nan(get_metric(r, 'osr', 'oracle_success', default=0))
        sr = check_inf_nan(get_metric(r, 'sr', 'success', default=0))
        spl = check_inf_nan(get_metric(r, 'spl', default=0.0))
        ndtw = check_inf_nan(get_metric(r, 'ndtw', 'nDTW', default=0.0))

        if sr == 1 and ne > 3.0:
            print(f"  ❌ Episode {ep_id}: SR=1 但 NE={ne:.3f}m > 3m")

        if sr == 0 and 0 <= ne < 3.0:
            print(f"  ⚠️  Episode {ep_id}: SR=0 但 NE={ne:.3f}m < 3m")

        if sr == 0 and spl > 0:
            print(f"  ❌ Episode {ep_id}: SR=0 但 SPL={spl:.4f} > 0")

        if osr < sr:
            print(f"  ❌ Episode {ep_id}: OSR={osr} < SR={sr}，这在定义上不成立")

        if not 0.0 <= ndtw <= 1.0:
            print(f"  ⚠️  Episode {ep_id}: nDTW={ndtw:.4f} 超出[0,1]范围")

        if ne < 0:
            print(f"  ⚠️  Episode {ep_id}: NE={ne} < 0 (距离数据无效)")
    
    print(f"{'='*80}")


def save_summary(metrics: Dict[str, Any], output_path: str):
    """保存汇总报告到文件"""
    n = metrics['total_episodes']
    
    content = f"""
================================================================================
📊 SpaceVLN 评估结果汇总
================================================================================

🎯 统一指标:
  NE:    {metrics['avg_ne']:.3f}m
  OSR:   {metrics['osr_count']}/{n} ({metrics['avg_osr']:.3f})
  SR:    {metrics['sr_count']}/{n} ({metrics['avg_sr']:.3f})
  SPL:   {metrics['avg_spl']:.3f}
  nDTW:  {metrics['avg_ndtw']:.3f}

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
        "NE",
        "OSR",
        "SR",
        "SPL",
        "nDTW",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for item in sorted_results:
            writer.writerow({
                "episode_id": item.get("episode_id", ""),
                "NE": _format_metric_value(check_inf_nan(get_metric(item, "ne", "distance_to_goal", default=-1.0)), 3),
                "OSR": str(int(check_inf_nan(get_metric(item, "osr", "oracle_success", default=0)))),
                "SR": str(int(check_inf_nan(get_metric(item, "sr", "success", default=0)))),
                "SPL": _format_metric_value(check_inf_nan(get_metric(item, "spl", default=0.0)), 4),
                "nDTW": _format_metric_value(check_inf_nan(get_metric(item, "ndtw", "nDTW", default=0.0)), 4),
            })
        writer.writerow({
            "episode_id": "AVERAGE",
            "NE": _format_metric_value(metrics["avg_ne"], 3),
            "OSR": _format_metric_value(metrics["avg_osr"], 4),
            "SR": _format_metric_value(metrics["avg_sr"], 4),
            "SPL": _format_metric_value(metrics["avg_spl"], 4),
            "nDTW": _format_metric_value(metrics["avg_ndtw"], 4),
        })

    md_lines = [
        "# Episode Results",
        "",
        "| Episode | NE(m) | OSR | SR | SPL | nDTW |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in sorted_results:
        md_lines.append(
            "| {episode} | {ne} | {osr} | {sr} | {spl} | {ndtw} |".format(
                episode=item.get("episode_id", ""),
                ne=_format_metric_value(check_inf_nan(get_metric(item, "ne", "distance_to_goal", default=-1.0)), 3),
                osr=str(int(check_inf_nan(get_metric(item, "osr", "oracle_success", default=0)))),
                sr=str(int(check_inf_nan(get_metric(item, "sr", "success", default=0)))),
                spl=_format_metric_value(check_inf_nan(get_metric(item, "spl", default=0.0)), 4),
                ndtw=_format_metric_value(check_inf_nan(get_metric(item, "ndtw", "nDTW", default=0.0)), 4),
            )
        )
    md_lines.extend([
        "",
        "## Average",
        "",
        "| Episodes | NE(m) | OSR | SR | SPL | nDTW |",
        "| --- | --- | --- | --- | --- | --- |",
        "| {total} | {avg_ne} | {avg_osr} | {avg_sr} | {avg_spl} | {avg_ndtw} |".format(
            total=metrics["total_episodes"],
            avg_ne=_format_metric_value(metrics["avg_ne"], 3),
            avg_osr=_format_metric_value(metrics["avg_osr"], 4),
            avg_sr=_format_metric_value(metrics["avg_sr"], 4),
            avg_spl=_format_metric_value(metrics["avg_spl"], 4),
            avg_ndtw=_format_metric_value(metrics["avg_ndtw"], 4),
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
