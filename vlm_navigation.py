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
    
    # VLM配置
    parser.add_argument("--config", type=str,
                       default="vlnce_baselines/vlm/api_config.yaml",
                       help="统一API配置文件路径（推荐）：同时设置LLM和VLM服务商/模型")
    parser.add_argument("--llm-config", type=str, 
                       default=None,
                       help="LLM配置文件路径（仅当不使用 --config 时生效）")
    parser.add_argument("--vlm-config", type=str,
                       default=None,
                       help="VLM配置文件路径（仅当不使用 --config 时生效）")
    
    # 导航参数
    parser.add_argument("--max-subtask-steps", type=int, default=5,
                       help="每个子任务最大步数（达到后强制触发验证，默认5步）")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Episode最大总步数（覆盖配置文件，默认使用配置文件值）")
    
    # 运行模式
    parser.add_argument("--auto", action="store_true",
                       help="全自动运行（无需确认）")
    
    args = parser.parse_args()
    
    # 加载配置
    config = get_config(args.exp_config, [])
    
    # 如果指定了 --max-steps，覆盖配置文件的值
    if args.max_steps is not None:
        config.defrost()
        config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_steps
        config.freeze()
        print(f"\n⚙️  覆盖最大步数: {args.max_steps} (命令行参数)")
    
    from vlnce_baselines.config_system import ConfigHelper
    
    # Episode ID范围验证配置
    MIN_EPISODE_ID = 1  # 最小episode ID（通常从0或1开始，这里设为1）
    MAX_EPISODE_ID = 1800  # 最大episode ID（根据数据集设置）
    
    # 确定要运行的episode列表
    if args.episode_ids:
        # 使用指定的episode ID列表
        episode_ids = [int(x.strip()) for x in args.episode_ids.split(',')]
        # 验证episode ID范围
        invalid_ids = [eid for eid in episode_ids if eid < MIN_EPISODE_ID or eid > MAX_EPISODE_ID]
        if invalid_ids:
            print(f"\n❌ 错误: 以下episode ID超出有效范围 [{MIN_EPISODE_ID}, {MAX_EPISODE_ID}]: {invalid_ids}")
            return
        print(f"\n📝 指定运行 {len(episode_ids)} 个episodes")
        print(f"📊 Episodes: {episode_ids}")
    elif args.random:
        import random
        import time
        # 随机模式：不加载数据集，直接从有效范围随机选择
        # 注意：如果选择的episode ID不存在，运行时会跳过
        
        # 使用当前时间戳作为随机种子，确保每次运行结果不同
        random_seed = int(time.time() * 1000) % (2**32)  # 使用毫秒级时间戳
        random.seed(random_seed)
        print(f"\n🎲 随机选择模式（从有效范围 [{MIN_EPISODE_ID}, {MAX_EPISODE_ID}] 中选择）")
        print(f"   🎯 随机种子: {random_seed}")
        print(f"   ⚠️  不验证episode是否存在，不存在的会自动跳过")
        
        # 使用配置的有效范围
        valid_range = range(MIN_EPISODE_ID, MAX_EPISODE_ID + 1)
        num_to_sample = min(args.num_episodes, len(valid_range))
        
        if num_to_sample == 0:
            print(f"\n❌ 错误: 请求的episode数量为0")
            return
        
        episode_ids = random.sample(list(valid_range), num_to_sample)
        print(f"📊 随机选择了 {len(episode_ids)} 个episodes: {episode_ids}")
    else:
        # 连续运行模式：验证范围
        start_id = args.episode_id
        end_id = args.episode_id + args.num_episodes - 1
        
        if start_id < MIN_EPISODE_ID:
            print(f"\n❌ 错误: 起始episode ID {start_id} 小于最小值 {MIN_EPISODE_ID}")
            print(f"   建议使用: --episode-id {MIN_EPISODE_ID}")
            return
        
        if end_id > MAX_EPISODE_ID:
            print(f"\n❌ 错误: 结束episode ID {end_id} 超过最大值 {MAX_EPISODE_ID}")
            max_num = MAX_EPISODE_ID - start_id + 1
            print(f"   建议使用: --num-episodes {max_num} (最多可运行到episode {MAX_EPISODE_ID})")
            return
        
        episode_ids = list(range(start_id, end_id + 1))
        print(f"\n📋 连续运行 episodes {start_id} 到 {end_id}")
        print(f"📊 Episodes: {episode_ids}")
    
    # 最终验证：确保有episodes要运行
    if not episode_ids or len(episode_ids) == 0:
        print(f"\n❌ 错误: 没有可运行的episodes")
        return
    
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
            
            # 初始化控制器（优先使用统一配置文件，回退到分开的 llm/vlm config）
            import os as _os
            _use_unified = _os.path.exists(args.config)
            if _use_unified:
                controller = VLMNavigationController(
                    episode_config,
                    config_path=args.config
                )
            else:
                # 回退到 legacy 双文件模式
                _llm = args.llm_config or "vlnce_baselines/vlm/llm_config.yaml"
                _vlm = args.vlm_config or "vlnce_baselines/vlm/vlm_config.yaml"
                controller = VLMNavigationController(
                    episode_config,
                    llm_config_path=_llm,
                    vlm_config_path=_vlm
                )
            
            # 重置Episode
            controller.reset_episode(episode_id=episode_id)
            
            # 从配置读取最大步数
            max_steps = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
            
            print(f"\n📝 指令: {controller.current_instruction}")
            print(f"⚙️  配置: Episode {episode_id} | 最大步数 {max_steps} (从 Habitat 配置)")
            if _use_unified:
                print(f"🔧 API config: {args.config}")
            else:
                print(f"🔧 VLM: LLM={_llm} | VLM={_vlm}")
            
            # 运行VLM导航
            result = controller.run_vlm_navigation(
                max_subtask_steps=args.max_subtask_steps
            )
            
            # 注意：run_vlm_navigation()内部已经调用了finish_episode()
            # 不需要再次调用，否则会导致"Episode over, call reset"错误
            
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
    
    if total_count > 0:
        success_rate = success_count / total_count * 100
        print(f"\n✅ 成功: {success_count}/{total_count} ({success_rate:.1f}%)")
        print(f"❌ 失败: {total_count - success_count}/{total_count}")
    else:
        print(f"\n⚠️  没有运行任何episodes")
    
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
