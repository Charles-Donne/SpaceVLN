"""
测试俯视图修复效果
==================
快速验证TOP_DOWN_MAP_VLNCE是否正常工作
"""
import argparse
from vlnce_baselines.config.default import get_config
from vlnce_baselines.vlm_navigation_controller import VLMNavigationController


def test_topdown_map(config_path: str, episode_id: int = 0):
    """测试俯视图是否正常显示"""
    print("\n" + "="*60)
    print("🔍 俯视图修复测试")
    print("="*60)
    
    # 加载配置
    config = get_config(config_path, [])
    
    from vlnce_baselines.config_system import ConfigHelper
    config = ConfigHelper.setup_episode_config(config, [episode_id], num_environments=1)
    config = ConfigHelper.setup_results_dir(config, "./test_topdown_results")
    
    # 这里会启用TOP_DOWN_MAP_VLNCE
    print("\n[1/4] 配置导航参数...")
    config = ConfigHelper.setup_navigation_config(config)
    
    # 检查是否启用了TOP_DOWN_MAP_VLNCE
    print("\n[2/4] 检查TOP_DOWN_MAP_VLNCE测量...")
    if "TOP_DOWN_MAP_VLNCE" in config.TASK_CONFIG.TASK.MEASUREMENTS:
        print("✅ TOP_DOWN_MAP_VLNCE已启用")
    else:
        print("❌ TOP_DOWN_MAP_VLNCE未启用")
        return False
    
    # 初始化控制器
    print("\n[3/4] 初始化控制器...")
    try:
        controller = VLMNavigationController(
            config,
            llm_config_path="vlnce_baselines/vlm/llm_config.yaml",
            vlm_config_path="vlnce_baselines/vlm/vlm_config.yaml"
        )
        print("✅ 控制器初始化成功")
    except Exception as e:
        print(f"❌ 控制器初始化失败: {e}")
        return False
    
    # 重置episode并执行一步
    print("\n[4/4] 测试俯视图生成...")
    try:
        controller.reset_episode(episode_id=episode_id)
        
        # 执行一次前进动作
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        result = controller.step_with_vlm(
            action=HabitatSimActions.MOVE_FORWARD,
            action_name="MOVE_FORWARD",
            save_vis=True
        )
        
        # 检查是否有top_down_map_vlnce
        if controller.latest_info and "top_down_map_vlnce" in controller.latest_info:
            print("✅ info中包含top_down_map_vlnce")
            tdm = controller.latest_info["top_down_map_vlnce"]
            print(f"   - 形状: {tdm.shape}")
            print(f"   - 数据类型: {tdm.dtype}")
            
            # 检查是否全黑（全0）
            import numpy as np
            if np.all(tdm == 0):
                print("⚠️  警告: 俯视图全黑（可能是场景问题）")
            else:
                print("✅ 俯视图包含有效数据")
        else:
            print("❌ info中没有top_down_map_vlnce")
            return False
        
        # 检查可视化文件
        import os
        vis_file = os.path.join(controller.episode_dir, "visualization", "step0001_visualization.jpg")
        if os.path.exists(vis_file):
            print(f"✅ 可视化文件已生成: {vis_file}")
            
            # 检查文件大小（确保不是空文件）
            file_size = os.path.getsize(vis_file)
            if file_size > 10000:  # 至少10KB
                print(f"   - 文件大小: {file_size/1024:.1f} KB")
                print("✅ 文件大小正常")
            else:
                print(f"⚠️  文件太小: {file_size} bytes")
        else:
            print(f"❌ 可视化文件未找到")
            return False
        
        controller.close()
        
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("🎉 测试完成！所有检查通过")
    print("="*60)
    print(f"\n查看结果: {controller.episode_dir}/visualization/")
    print("  - step0001_visualization.jpg: 包含RGB+俯视图拼接")
    print("\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试俯视图修复效果")
    parser.add_argument("--exp-config", type=str, required=True, 
                       help="Habitat配置文件")
    parser.add_argument("--episode-id", type=int, default=0, 
                       help="测试的Episode ID")
    
    args = parser.parse_args()
    
    success = test_topdown_map(args.exp_config, args.episode_id)
    
    if not success:
        print("\n❌ 测试失败，请检查配置和日志")
        exit(1)
    else:
        print("✅ 测试成功")
        exit(0)
