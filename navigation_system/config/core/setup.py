"""Focused runtime config mutation helpers for SpaceVLN navigation."""

from typing import List, Optional
from habitat import Config
from navigation_system.config.runtime.sync import sync_runtime_panels


class ConfigHelper:
    """Helpers that keep derived runtime config in sync after mutations."""
    
    @staticmethod
    def setup_navigation_config(
        config: Config,
        torch_gpu_id: Optional[int] = None,
        num_environments: Optional[int] = None
    ) -> Config:
        """
        配置导航相关参数
        
        Args:
            config: Habitat配置对象
            torch_gpu_id: GPU设备ID（可选，从config读取）
            num_environments: 环境数量（可选，从config读取）
        
        Returns:
            config: 配置后的Config对象
        """
        config.defrost()
        
        # 从config读取默认值
        if torch_gpu_id is None:
            torch_gpu_id = config.TORCH_GPU_ID
        if num_environments is None:
            num_environments = config.NUM_ENVIRONMENTS

        config.TORCH_GPU_ID = int(torch_gpu_id)
        config.NUM_ENVIRONMENTS = int(num_environments)
        if hasattr(config, "RUNTIME"):
            config.RUNTIME.TORCH_GPU_ID = int(torch_gpu_id)
            config.RUNTIME.NUM_ENVIRONMENTS = int(num_environments)
        
        # 同步运行时派生字段到结构化面板
        config.SPACE.SENSOR.DEVICE_ID = int(torch_gpu_id)
        config.SPACE.SENSOR.NUM_ENVIRONMENTS = int(num_environments)

        # ===== 启用必要的Habitat测量指标 =====
        required_measurements = [
            "TOP_DOWN_MAP_VLNCE",         # 俯视图可视化必需
            "DISTANCE_TO_GOAL",           # 距离目标点的距离
            "SUCCESS",                    # 是否成功（3米内）
            "SPL",                        # Success weighted by Path Length
            "ORACLE_NAVIGATION_ERROR",    # 轨迹中与目标的最小距离
            "ORACLE_SUCCESS",             # 轨迹中是否曾到达目标
            "ORACLE_SPL"                  # 基于oracle_success的SPL
        ]
        
        for measurement in required_measurements:
            if measurement not in config.TASK_CONFIG.TASK.MEASUREMENTS:
                config.TASK_CONFIG.TASK.MEASUREMENTS.append(measurement)

        sync_runtime_panels(config)
        config.freeze()
        return config
    
    @staticmethod
    def setup_episode_config(
        config: Config,
        episode_ids: List[int],
        num_environments: int = 1
    ) -> Config:
        """
        配置Episode相关参数
        
        Args:
            config: Habitat配置对象
            episode_ids: 要运行的episode ID列表
            num_environments: 环境数量
        
        Returns:
            config: 配置后的Config对象
        """
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episode_ids
        config.NUM_ENVIRONMENTS = num_environments
        if hasattr(config, "RUNTIME"):
            config.RUNTIME.NUM_ENVIRONMENTS = int(num_environments)
        config.SPACE.SENSOR.NUM_ENVIRONMENTS = int(num_environments)
        sync_runtime_panels(config)
        config.freeze()
        return config
    
    @staticmethod
    def setup_results_dir(
        config: Config,
        results_dir: str
    ) -> Config:
        """
        设置结果保存目录
        
        Args:
            config: Habitat配置对象
            results_dir: 结果保存路径
        
        Returns:
            config: 配置后的Config对象
        """
        config.defrost()
        config.RESULTS_DIR = results_dir
        if hasattr(config, "PATHS"):
            config.PATHS.RESULTS_DIR = str(results_dir)
        sync_runtime_panels(config)
        config.freeze()
        return config
