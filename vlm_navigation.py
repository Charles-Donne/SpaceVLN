"""
VLM Navigation Runner
=====================
VLM自动导航系统入口

使用LLM进行高层规划 + VLM进行低层动作执行
基于interactive_navigation架构，集成语义建图和可视化
"""
import argparse
from vlnce_baselines.config.default import get_config
from vlnce_baselines.vlm_navigation_controller import VLMNavigationController


def main():
    parser = argparse.ArgumentParser(description="VLM自动导航系统")
    
    # 基础配置（与interactive_navigation一致）
    parser.add_argument("--exp-config", type=str, required=True, help="Habitat配置文件")
    parser.add_argument("--episode-id", type=int, default=0, help="起始Episode ID")
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
    parser.add_argument("--max-subtask-steps", type=int, default=10,
                       help="每个子任务最大步数（达到后触发验证）")
    
    # 运行模式
    parser.add_argument("--auto", action="store_true",
                       help="全自动运行（无需确认）")
    
    args = parser.parse_args()
    
    # 加载配置
    config = get_config(args.exp_config, [])
    
    from vlnce_baselines.config_system import ConfigHelper
    
    # 确定要运行的episode列表
    if args.random:
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
        try:
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
            print(f"\n❌ Episode {episode_id} 运行失败: {e}")
            results_summary.append({
                'episode_id': episode_id,
                'success': False,
                'steps': 0,
                'error': str(e)
            })
        finally:
            # 清理控制器
            try:
                controller.envs.close()
            except:
                pass
    
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
    controller.close()
    
    # 打印结果
    print("\n" + "="*60)
    print("🏁 导航结果")
    print("="*60)
    print(f"✅ 成功: {result.get('success', False)}")
    print(f"📊 总步数: {result.get('total_steps', 0)}")
    print(f"📋 子任务数: {result.get('subtask_count', 0)}")
    print(f"🔍 检测类别: {len(result.get('detected_classes', []))}")
    if result.get('reason'):
        print(f"❌ 失败原因: {result['reason']}")
    print(f"📁 结果目录: {config.RESULTS_DIR}/episode_{args.episode_id}/")
    print("="*60)


if __name__ == "__main__":
    main()
