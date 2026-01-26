"""
VLM Navigation Runner
=====================
VLM自动导航系统入口

使用LLM进行高层规划 + VLM进行低层动作执行
基于interactive_navigation架构，集成语义建图和可视化
"""
import os
import argparse
from vlnce_baselines.config.default import get_config
from vlnce_baselines.vlm_navigation_controller import VLMNavigationController


def main():
    parser = argparse.ArgumentParser(description="VLM自动导航系统")
    
    # 基础配置（与interactive_navigation一致）
    parser.add_argument("--exp-config", type=str, required=True, help="Habitat配置文件")
    parser.add_argument("--episode-id", type=int, default=0, help="起始Episode ID")
    parser.add_argument("--episode-ids", type=str, default=None, help="指定episode ID列表，逗号分隔（如 '832,701,231'）")
    parser.add_argument("--num-episodes", type=int, default=1, help="运行Episode数量（连续或随机）")
    parser.add_argument("--random", action="store_true", help="随机选择episodes而非连续运行")
    parser.add_argument("--results-dir", type=str, default=None, help="结果保存目录")
    parser.add_argument("--max-steps", type=int, default=500, help="最大总步数")
    
    # VLM配置
    parser.add_argument("--llm-config", type=str, 
                       default="vlnce_baselines/vlm/llm_config.yaml",
                       help="LLM配置文件路径")
    parser.add_argument("--vlm-config", type=str,
                       default="vlnce_baselines/vlm/vlm_config.yaml", 
                       help="VLM配置文件路径")
    
    # 导航参数
    parser.add_argument("--max-subtask-steps", type=int, default=5,
                       help="每个子任务最大步数（达到后强制触发验证，默认5步）")
    
    # 运行模式
    parser.add_argument("--auto", action="store_true",
                       help="全自动运行（无需确认）")
    
    args = parser.parse_args()
    
    # 加载配置
    config = get_config(args.exp_config, [])
    
    from vlnce_baselines.config_system import ConfigHelper
    
    # 确定要运行的episode列表
    if args.episode_ids:
        # 使用指定的episode ID列表
        episode_ids = [int(x.strip()) for x in args.episode_ids.split(',')]
        print(f"\n📝 指定运行 {len(episode_ids)} 个episodes")
        print(f"📊 Episodes: {episode_ids}")
    elif args.random:
        import random
        import habitat
        # 加载数据集获取总episode数
        temp_config = config.clone()
        temp_config.defrost()
        temp_config.TASK_CONFIG.DATASET.SPLIT = config.TASK_CONFIG.DATASET.SPLIT
        temp_config.freeze()
        dataset = habitat.datasets.make_dataset(temp_config.TASK_CONFIG.DATASET.TYPE)
        total_episodes = len(dataset.episodes)
        episode_ids = random.sample(range(total_episodes), min(args.num_episodes, total_episodes))
        print(f"\n🎲 随机选择 {len(episode_ids)} 个episodes (共{total_episodes}个可用)")
        print(f"📊 Episodes: {episode_ids}")
    else:
        episode_ids = list(range(args.episode_id, args.episode_id + args.num_episodes))
        print(f"\n📋 连续运行 episodes {args.episode_id} 到 {args.episode_id + args.num_episodes - 1}")
        print(f"📊 Episodes: {episode_ids}")
    
    # 统计结果
    results_summary = []
    
    # 循环运行每个episode
    for idx, episode_id in enumerate(episode_ids, 1):
        print(f"\n{'='*80}")
        print(f"🔄 [{idx}/{len(episode_ids)}] 开始Episode {episode_id}")
        print(f"{'='*80}")
        
        controller = None
        try:
            # 重新配置episode
            episode_config = config.clone()
            episode_config.defrost()
            episode_config = ConfigHelper.setup_episode_config(episode_config, [episode_id], num_environments=1)
            if args.results_dir:
                episode_config = ConfigHelper.setup_results_dir(episode_config, args.results_dir)
            episode_config = ConfigHelper.setup_navigation_config(episode_config)
            episode_config.freeze()
            
            # 初始化控制器
            controller = VLMNavigationController(
                episode_config,
                llm_config_path=args.llm_config,
                vlm_config_path=args.vlm_config
            )
            
            # 重置Episode
            controller.reset_episode(episode_id=episode_id)
            
            print(f"\n📝 指令: {controller.current_instruction}")
            print(f"⚙️  配置: Episode {episode_id} | 最大步数 {args.max_steps}")
            print(f"🔧 VLM: LLM={args.llm_config} | VLM={args.vlm_config}")
            
            # 运行VLM导航
            result = controller.run_vlm_navigation(
                max_steps=args.max_steps,
                max_subtask_steps=args.max_subtask_steps
            )
            
            # 结束Episode
            controller.finish_episode(
                success=result['success'],
                stop_action=True
            )
            
            results_summary.append({
                'episode_id': episode_id,
                'success': result['success'],
                'steps': result.get('steps', 0),
                'error': None
            })
            
            print(f"\n{'='*80}")
            print(f"{'✅' if result['success'] else '❌'} Episode {episode_id} 完成 | 成功: {result['success']} | 步数: {result.get('steps', 0)}")
            print(f"{'='*80}")
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"\n❌ Episode {episode_id} 运行失败: {error_msg}")
            print(f"\n完整错误堆栈:")
            traceback.print_exc()
            
            results_summary.append({
                'episode_id': episode_id,
                'success': False,
                'steps': 0,
                'error': error_msg
            })
        finally:
            # 清理控制器
            if controller is not None:
                try:
                    controller.envs.close()
                except Exception as cleanup_error:
                    print(f"⚠️  清理环境时出错: {cleanup_error}")
    
    # 打印总结
    print(f"\n\n{'='*80}")
    print("📊 批量运行总结")
    print(f"{'='*80}")
    
    success_count = sum(1 for r in results_summary if r['success'])
    total_count = len(results_summary)
    
    print(f"\n✅ 成功: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    print(f"❌ 失败: {total_count - success_count}/{total_count}")
    
    print("\n详细结果:")
    for r in results_summary:
        status = '✅' if r['success'] else '❌'
        error_msg = f" (错误: {r['error']})" if r['error'] else ""
        print(f"  {status} Episode {r['episode_id']}: 步数={r['steps']}{error_msg}")
    
    print(f"\n{'='*80}")
    
    # 使用analyze_results脚本生成完整统计
    print("\n📊 生成详细评估报告...")
    import subprocess
    analyze_script = os.path.join(os.path.dirname(__file__), "analyze_results.py")
    if os.path.exists(analyze_script) and args.results_dir:
        try:
            result = subprocess.run(
                ["python", analyze_script, "--path", args.results_dir, "--save"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"⚠️  分析脚本执行失败: {result.stderr}")
        except Exception as e:
            print(f"⚠️  无法运行分析脚本: {e}")
    
    print("\n" + "="*60)
    print("🏁 批量评估完成")
    print("="*60)
    print(f"✅ 成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    print(f"📊 平均步数: {sum(r['steps'] for r in results_summary)/total_count:.1f}")
    print(f"📁 结果目录: {args.results_dir or config.RESULTS_DIR}")
    print(f"📄 详细报告: {os.path.join(args.results_dir or config.RESULTS_DIR, 'summary.txt')}")
    print("="*60)


if __name__ == "__main__":
    main()
