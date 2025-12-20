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
    parser.add_argument("--episode-id", type=int, default=0, help="Episode ID")
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
    parser.add_argument("--max-subtask-steps", type=int, default=50,
                       help="每个子任务最大步数")
    parser.add_argument("--verify-interval", type=int, default=10,
                       help="子任务验证间隔步数")
    
    # 运行模式
    parser.add_argument("--auto", action="store_true",
                       help="全自动运行（无需确认）")
    
    args = parser.parse_args()
    
    # 加载配置
    config = get_config(args.exp_config, [])
    
    from vlnce_baselines.config_system import ConfigHelper
    config = ConfigHelper.setup_episode_config(config, [args.episode_id], num_environments=1)
    if args.results_dir:
        config = ConfigHelper.setup_results_dir(config, args.results_dir)
    config = ConfigHelper.setup_navigation_config(config)
    ConfigHelper.print_config_summary(config)
    
    # 初始化控制器
    controller = VLMNavigationController(
        config,
        llm_config_path=args.llm_config,
        vlm_config_path=args.vlm_config
    )
    
    # 重置Episode
    controller.reset_episode(episode_id=args.episode_id)
    
    print("\n" + "="*60)
    print("🤖 VLM自动导航系统")
    print("="*60)
    print(f"📝 指令: {controller.current_instruction}")
    print(f"⚙️  配置: Episode {args.episode_id} | 最大步数 {args.max_steps}")
    print(f"🔧 VLM: LLM={args.llm_config} | VLM={args.vlm_config}")
    print("="*60)
    
    # 运行VLM导航（移除手动确认，直接开始）
    
    # 运行VLM导航
    result = controller.run_vlm_navigation(
        max_steps=args.max_steps,
        max_subtask_steps=args.max_subtask_steps,
        verify_interval=args.verify_interval
    )
    
    # 结束Episode
    controller.finish_episode(
        success=result['success'],
        stop_action=True
    )
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
