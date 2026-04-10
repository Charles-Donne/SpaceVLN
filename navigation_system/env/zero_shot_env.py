from typing import Any, Dict, Tuple, Union

import habitat
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_extensions.pose_utils import get_sim_location


@baseline_registry.register_env(name="VLNCEZeroShotEnv")
class VLNCEZeroShotEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Union[Dataset, None] = None) -> None:
        super().__init__(config.TASK_CONFIG, dataset)
        self.sensor_pose_sensor = self.habitat_env.task.sensor_suite.get('sensor_pose')
    
    def reset(self) -> Observations:
        self.sensor_pose_sensor.episode_start = False
        self.last_sim_location = get_sim_location(self.habitat_env.sim)
        self.sensor_pose = [0., 0., 0.] # initialize last sensor pose as [0,0,0]
        obs = super().reset()
        return obs

    def get_reward(self, observations: Observations) -> float:
        return 0.0
        
    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()
    
    def get_metrics(self) -> Dict[Any, Any]:
        """获取当前episode的评估指标"""
        return self.habitat_env.get_metrics()
    
    def get_done(self, observations):
        return self._env.episode_over
    
    def get_reward_range(self):
        return (0.0, 0.0)
    
    def get_agent_pose(self) -> Tuple[float, float, float]:
        """获取agent当前pose (x, y, orientation)
        
        Returns:
            tuple: (x, y, o) where x, y are coordinates and o is orientation in radians
        """
        return get_sim_location(self.habitat_env.sim)
