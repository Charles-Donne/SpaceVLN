"""
VLN Navigation Results Analyzer
================================
分析导航结果并计算关键指标：SR, SPL, Oracle Success, Distance to Goal, Path Length
"""
import os
import argparse
import json
import math
from typing import Dict, List
from pathlib import Path


def check_inf_nan(value):
    """处理无穷大和NaN值"""
    if math.isinf(value) or math.isnan(value):
        return 0
    return value


class ResultsAnalyzer:
    """结果分析器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.log_dir = self.results_dir / 'log'
        
        if not self.log_dir.exists():
            raise ValueError(f"Log目录不存在: {self.log_dir}")
    
    def load_episode_results(self) -> List[Dict]:
        """加载所有episode结果"""
        results = []
        json_files = list(self.log_dir.glob('*.json'))
        
        if not json_files:
            print(f"⚠️  在 {self.log_dir} 中未找到结果JSON文件")
            return results
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"⚠️  加载 {json_file.name} 失败: {e}")
        
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """计算所有指标"""
        if not results:
            return {}
        
        metrics = {
            'success': 0,
            'spl': 0.0,
            'distance_to_goal': 0.0,
            'path_length': 0.0,
            'oracle_success': 0,
            'oracle_navigation_error': 0.0,
            'total_episodes': len(results)
        }
        
        for data in results:
            # Success Rate (SR)
            metrics['success'] += check_inf_nan(int(data.get('success', 0)))
            
            # SPL
            metrics['spl'] += check_inf_nan(data.get('spl', 0.0))
            
            # Navigation Error (NE) - Distance to Goal
            metrics['distance_to_goal'] += check_inf_nan(data.get('distance_to_goal', 0.0))
            
            # Path Length
            metrics['path_length'] += check_inf_nan(data.get('path_length', 0.0))
            
            # Oracle Success
            metrics['oracle_success'] += check_inf_nan(int(data.get('oracle_success', 0)))
            
            # Oracle Navigation Error (ONE)
            metrics['oracle_navigation_error'] += check_inf_nan(
                data.get('oracle_navigation_error', float('inf'))
            )
        
        return metrics
    
    def print_results(self, metrics: Dict):
        """打印结果"""
        n = metrics['total_episodes']
        
        print("\n" + "="*60)
        print("📊 VLN Navigation Results Summary")
        print("="*60)
        print(f"\n📁 Results Directory: {self.results_dir}")
        print(f"📝 Total Episodes: {n}")
        print("\n" + "-"*60)
        
        # Success Rate (SR)
        sr = metrics['success'] / n if n > 0 else 0
        print(f"✅ Success Rate (SR): {metrics['success']}/{n} ({sr:.3f})")
        
        # SPL
        avg_spl = metrics['spl'] / n if n > 0 else 0
        print(f"🎯 SPL (Success weighted by Path Length): {avg_spl:.3f}")
        
        # Oracle Success Rate
        osr = metrics['oracle_success'] / n if n > 0 else 0
        print(f"🔮 Oracle Success Rate: {metrics['oracle_success']}/{n} ({osr:.3f})")
        
        # Navigation Error (Average Distance to Goal)
        avg_dtg = metrics['distance_to_goal'] / n if n > 0 else 0
        print(f"📍 Average Navigation Error (NE): {avg_dtg:.3f}m")
        
        # Oracle Navigation Error
        avg_one = metrics['oracle_navigation_error'] / n if n > 0 else 0
        print(f"📍 Average Oracle Navigation Error (ONE): {avg_one:.3f}m")
        
        # Average Path Length
        avg_path = metrics['path_length'] / n if n > 0 else 0
        print(f"📏 Average Path Length: {avg_path:.3f}m")
        
        print("="*60 + "\n")
    
    def save_summary(self, metrics: Dict, output_file: str = "summary.json"):
        """保存汇总结果"""
        n = metrics['total_episodes']
        
        summary = {
            'total_episodes': n,
            'success_rate': metrics['success'] / n if n > 0 else 0,
            'spl': metrics['spl'] / n if n > 0 else 0,
            'oracle_success_rate': metrics['oracle_success'] / n if n > 0 else 0,
            'avg_navigation_error': metrics['distance_to_goal'] / n if n > 0 else 0,
            'avg_oracle_navigation_error': metrics['oracle_navigation_error'] / n if n > 0 else 0,
            'avg_path_length': metrics['path_length'] / n if n > 0 else 0,
            'raw_metrics': metrics
        }
        
        output_path = self.results_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Summary saved to: {output_path}")
    
    def analyze(self, save_summary: bool = True):
        """执行完整分析"""
        print("\n🔍 Loading results...")
        results = self.load_episode_results()
        
        if not results:
            print("❌ No results found!")
            return
        
        print(f"✅ Loaded {len(results)} episode results")
        
        print("\n📊 Calculating metrics...")
        metrics = self.calculate_metrics(results)
        
        self.print_results(metrics)
        
        if save_summary:
            self.save_summary(metrics)


def main():
    parser = argparse.ArgumentParser(
        description="分析VLN导航结果并计算SR、SPL、NE等指标"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="结果目录路径（包含log子目录）"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存汇总文件"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = ResultsAnalyzer(args.path)
        analyzer.analyze(save_summary=not args.no_save)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
