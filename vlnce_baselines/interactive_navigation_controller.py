"""
Interactive Navigation Controller
实时键盘控制导航系统：建图、检测、可视化
"""
import numpy as np
import cv2
import torch
from typing import Dict, Any
from torchvision import transforms
from habitat import Config
from habitat.core.simulator import Observations
from habitat_baselines.common.environments import get_env_class

from vlnce_baselines.detection import GroundedSAM
from vlnce_baselines.mapping import Semantic_Mapping, SemanticMapper, SemanticProcessor
from vlnce_baselines.visualization import MapVisualizer
from vlnce_baselines.config_system import ConfigHelper, create_category_config
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import get_device


class InteractiveNavigationController:
    """实时键盘控制导航器"""
    
    def __init__(self, config: Config):
        # print("[Init] 配置MAP参数...")
        self.config = ConfigHelper.setup_navigation_config(config)
        self.device = get_device(self.config.TORCH_GPU_ID)
        torch.cuda.set_device(self.device)
        
        self.map_args = self.config.MAP
        self.resolution = self.config.MAP.MAP_RESOLUTION
        self.width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        self.height = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
        self.map_shape = (self.config.MAP.MAP_SIZE_CM // self.resolution,
                         self.config.MAP.MAP_SIZE_CM // self.resolution)
        
        # print("[Init] 初始化Habitat环境...")
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,
        )
        # print(f"[Init] 环境初始化完成，episodes: {self.envs.number_of_episodes}")
        
        # print("[Init] 初始化GroundedSAM...")
        self.segment_module = GroundedSAM(self.config, self.device)
        
        # print("[Init] 初始化Semantic Mapping...")
        mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        mapping_module.eval()
        
        # print("[Init] 初始化Semantic Mapper...")
        self.mapper = SemanticMapper(mapping_module, self.map_shape, self.resolution)
        
        # print("[Init] 初始化Map Visualizer...")
        self.visualizer = MapVisualizer(
            self.config.RESULTS_DIR, 
            self.resolution, 
            self.map_shape,
            enable_global_map_crop=self.config.MAP.ENABLE_GLOBAL_MAP_CROP,
            enable_adaptive_zoom=self.config.MAP.ENABLE_ADAPTIVE_ZOOM
        )
        
        self.category_config = create_category_config()
        self.mapping_classes = self.category_config.mapping_classes
        self.landmark_classes = self.category_config.landmark_classes
        self.detection_classes = self.category_config.detection_classes
        self.classes = []
        
        from vlnce_baselines.config_system.constants import landmark_min_area_threshold, landmark_min_total_pixels
        self.landmark_min_area_threshold = landmark_min_area_threshold
        self.landmark_min_total_pixels = landmark_min_total_pixels
        
        self.current_episode_id = None
        self.current_step = 0
    
    @property
    def detected_classes(self):
        """便捷访问detected_classes（代理到category_config）"""
        return self.category_config._detected_classes
    
    def reset_episode(self, episode_id: int = None):
        print(f"\n{'='*60}\nEpisode {episode_id if episode_id else 0}\n{'='*60}")
        
        self.envs.reset()
        self.current_step = 0
        self.current_episode_id = episode_id if episode_id is not None else 0
        
        self.category_config.reset_detected()
        self.classes = self.category_config.detection_classes
        self.mapper.reset()
        self.mapper.init_map_and_pose(num_detected_classes=0)
        
        current_episodes = self.envs.current_episodes()
        self.current_instruction = current_episodes[0].instruction.instruction_text
        
        print(f"Instruction: {self.current_instruction[:100]}{'...' if len(self.current_instruction) > 100 else ''}")
    
    def look_around(self) -> None:
        """360度环视建图(12步×30°)，步数0-11"""
# print("🔄 360°环视...", end="", flush=True)
        
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        
        for step in range(12):
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            outputs = self.envs.step(actions)
            obs, _, dones, _ = [list(x) for x in zip(*outputs)]
            
            if dones[0]:
                print(" [WARN] Episode ended early")
                self.current_step = step + 1
                return
            
            prev_class_count = len(self.detected_classes)
            batch_obs = self._batch_obs(obs, save_object_detection=True, step=step)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            
            map_state = self.mapper.update_map(
                batch_obs, poses, step,
                list(self.detected_classes), self.current_episode_id
            )
            
            new_classes = len(self.detected_classes) - prev_class_count
            # 不再打印每步的进度，只在最后汇总
        
        self.current_step = 12
        landmarks_found = [cls for cls in self.detected_classes if cls in landmark_classes]
        # print(f" ✅ {len(self.detected_classes)}类")
    
    def step(self, action: int, save_vis: bool = True, phase: str = "action") -> Dict[str, Any]:
        """执行一步动作，更新地图并保存可视化"""
        # ⚠️ 关键修复：在使用current_step之前先累加，避免覆盖环视最后一步
        self.current_step += 1
        
        print(f"[{self.current_step}]{self._action_name(action)}", end=" ")
        
        outputs = self.envs.step([action])
        obs, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        
        # 保存done标志和info（用于finish_episode检查）
        self.latest_done = dones[0]
        self.latest_info = infos[0]
        
        if dones[0]:
            print(" → Episode结束")
            return {
                'obs': obs[0],
                'reward': rewards[0],
                'done': dones[0],
                'info': infos[0],
                'detected_classes': list(self.detected_classes)
            }
        
        prev_class_count = len(self.detected_classes)
        batch_obs = self._batch_obs(obs, save_object_detection=True)
        poses = torch.from_numpy(
            np.array([item['sensor_pose'] for item in obs])
        ).float().to(self.device)
        
        map_state = self.mapper.update_map(
            batch_obs, poses, self.current_step,
            list(self.detected_classes), self.current_episode_id
        )
        
        # print(f"[Controller.step] 从mapper接收轨迹: 全局={len(map_state.get('global_trajectory_points', []))}, 子任务={len(map_state.get('subtask_trajectory_points', []))}")
        
        new_classes = len(self.detected_classes) - prev_class_count
# print(f" +{new_classes}类" if new_classes > 0 else "")
        
        if save_vis:
            # action执行时不传waypoint信息，不计算角度（只在环视后计算）
            rgb_bgr = cv2.cvtColor(obs[0]['rgb'], cv2.COLOR_RGB2BGR)
            _, detected_landmarks_step, _ = self.visualizer.save_step_visualization(
                step=self.current_step,
                episode_id=self.current_episode_id,
                rgb=rgb_bgr,
                full_map=map_state['full_map'],
                trajectory_points=map_state.get('subtask_trajectory_points', []),  # 从map_state获取（local map用子任务轨迹）
                detected_classes=list(self.detected_classes),
                current_pose=map_state['full_pose'],
                floor=map_state['floor'],
                hfov=self.config.MAP.HFOV,
                detections=self.latest_detections_full if hasattr(self, 'latest_detections_full') else None,
                labels=self.latest_labels_full if hasattr(self, 'latest_labels_full') else None,
                landmark_classes=self.landmark_classes,
                mapping_classes=self.mapping_classes,
                landmark_config={
                    'min_total_pixels': self.landmark_min_total_pixels,
                    'min_area_threshold': self.landmark_min_area_threshold
                },
                waypoint_positions=map_state.get('waypoint_positions', []),  # 从map_state获取（已旋转）
                waypoint_ids=map_state.get('waypoint_ids', []),  # 从map_state获取
                phase=phase,
                global_trajectory_points=map_state.get('global_trajectory_points', []),  # 从map_state获取（global map用全局轨迹）
                crop_offset=map_state.get('crop_offset')  # 从map_state获取
            )
            
            # 保存当前step检测到的landmarks（用于action决策）
            if detected_landmarks_step:
                if not hasattr(self, 'current_step_landmarks'):
                    self.current_step_landmarks = {}
                self.current_step_landmarks[self.current_step] = detected_landmarks_step
        
        return {
            'obs': obs[0],
            'reward': rewards[0],
            'done': dones[0],
            'info': infos[0],
            'detected_classes': list(self.detected_classes)
        }
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        map_state = self.mapper.get_map_state()
        
        return {
            'step': self.current_step,
            'episode_id': self.current_episode_id,
            'full_map': map_state['full_map'],
            # 'trajectory_points': map_state['trajectory_points'],  # 已废弃，轨迹在 Channel 2
            'floor': map_state['floor'],
            'detected_classes': list(self.detected_classes),
            'current_pose': map_state['full_pose']
        }
    
    def finish_episode(self, success: bool = False, stop_action: bool = False) -> dict:
        """
        Episode结束总结
        
        重要：调用STOP动作以正确触发Habitat的Success判定
        Success需要同时满足:
        1. distance_to_goal < SUCCESS_DISTANCE (3米)
        2. is_stop_called = True (必须调用STOP动作)
        
        Returns:
            final_metrics: 调用STOP后的最终评估指标
        """
        print(f"\n{'='*60}")
        print(f"EPISODE FINISH")
        print(f"{'='*60}")
        print(f"Episode: {self.current_episode_id}")
        print(f"Steps: {self.current_step} | Classes: {len(self.detected_classes)}")
        status = "STOP" if stop_action else "MAX_STEPS"
        print(f"Reason: {status}")
        
        # 🔑 关键修复：调用STOP动作以触发Habitat的Success判定
        final_metrics = {}
        
        # 检查episode是否已经结束（避免在已done的episode上调用step）
        episode_already_done = False
        if hasattr(self, 'latest_info') and self.latest_info:
            episode_already_done = self.latest_info.get('done', False)
        
        # 额外检查：如果latest_done标志存在且为True，也认为episode已结束
        if hasattr(self, 'latest_done') and self.latest_done:
            episode_already_done = True
        
        if stop_action and not episode_already_done:
            # print("\n🛑 执行STOP动作以完成Episode...")
            try:
                # 调用STOP动作 (action_id = 0)
                outputs = self.envs.step([0])
                # 🔑 关键修复：与step()方法保持一致的解包方式
                observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
                
                # 获取最终指标
                if infos and len(infos) > 0:
                    final_metrics = infos[0]
                    dtg = final_metrics.get('distance_to_goal', -1)
                    success_flag = final_metrics.get('success', 0)
                    print(f"DTG: {dtg:.3f}m | Success: {success_flag} | SPL: {final_metrics.get('spl', 0.0):.4f}")
                    
                    # 数据验证
                    if success_flag == 1 and dtg > 3.0:
                        print(f"   [WARN] Anomaly: Success=1 but DTG={dtg:.3f}m > 3m")
                    elif success_flag == 0 and 0 <= dtg < 3.0:
                        print(f"   [WARN] DTG={dtg:.3f}m < 3m but Success=0")
            except AssertionError as e:
                # Episode已经结束，无法调用STOP
                print(f"\n[WARN] Episode already ended, cannot STOP: {e}")
                print("   Using last step metrics")
                if hasattr(self, 'latest_info') and self.latest_info:
                    final_metrics = self.latest_info.copy()
            except Exception as e:
                print(f"   [ERR] STOP failed: {e}")
                final_metrics = {}
        elif stop_action and episode_already_done:
            print("\n[WARN] Episode already done, skip STOP")
            print("   Using cached metrics")
            # 使用最后一次的info作为最终指标
            if hasattr(self, 'latest_info') and self.latest_info:
                final_metrics = self.latest_info.copy()
        else:
            print("MAX_STEPS reached (Success=0)")
            # 获取当前指标（不调用STOP）
            if self.latest_info:
                final_metrics = self.latest_info.copy()
        
        print(f"{'='*60}\n")
        return final_metrics
    
    def _concat_obs(self, obs: Observations) -> np.ndarray:
        """合并RGB和Depth"""
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        return state
    
    def _get_sem_pred(self, rgb: np.ndarray, save_object_detection: bool = False, step: int = None) -> np.ndarray:
        """
        语义分割：GroundedSAM检测 + Winner-Takes-All
        
        检测逻辑：
        - detection_classes = mapping_classes(15个固定) + landmark_classes(动态)
        - mapping_classes检测结果 → 进入15通道语义地图
        - landmark_classes检测结果 → 保留但不进入地图，用于可视化标注
        
        Returns:
            semantic_masks: [H, W, 15] 固定15个通道的语义地图
        """
        # 使用 detection_classes = mapping + landmark 进行检测
        masks_all, labels_all, annotated_images, current_detections = \
            self.segment_module.segment(rgb, classes=self.classes)
        self.mapper.mapping_module.rgb_vis = annotated_images
        
        self.latest_detections_full = current_detections
        self.latest_labels_full = labels_all.copy()
        self.latest_masks_full = masks_all.copy()  # 保存原始masks用于地面分割
        self.latest_rgb_original = rgb.copy()
        
        # 预定义的基础类别（固定15个）
        predefined_classes = self.mapping_classes
        
        # 分类处理检测结果
        valid_masks = []        # 用于建图的mapping类别
        valid_labels = []
        valid_confidences = []
        
        for i, label in enumerate(labels_all):
            parts = label.split()
            label_name = ' '.join(parts[:-1]) if len(parts) > 1 else parts[0]
            confidence = float(parts[-1]) if len(parts) > 1 else 0.5
            
            # 只有mapping_classes的检测进入语义地图
            if label_name in predefined_classes:
                valid_masks.append(masks_all[i])
                valid_labels.append(label_name)
                valid_confidences.append(confidence)
            
            # 所有检测到的类别都记录（包括landmark）
            self.detected_classes.add(label_name)
        
        if len(valid_masks) == 0:
            # 没有检测到有效的mapping类别，返回全0的15通道
            return np.zeros((self.height, self.width, len(predefined_classes)), dtype=np.float32)
        
        # Winner-Takes-All处理（只处理mapping类别）
        valid_masks = np.array(valid_masks)
        masks_processed = self._process_masks_with_labels(valid_masks, valid_labels, valid_confidences)
        
        # 按照预定义类别顺序组织mask（固定15通道）
        global_masks = np.zeros((len(predefined_classes), self.height, self.width), dtype=np.float32)
        
        for i, cls_name in enumerate(valid_labels):
            if cls_name in predefined_classes:
                global_idx = predefined_classes.index(cls_name)
                if i < masks_processed.shape[0]:
                    global_masks[global_idx] = masks_processed[i]
        
        return global_masks.transpose(1, 2, 0)  # [H, W, 15]
    
    def _process_masks_with_labels(self, masks: np.ndarray, labels: list, confidences: list = None) -> np.ndarray:
        """Winner-Takes-All掩码处理"""
        return SemanticProcessor.apply_winner_takes_all(
            masks, labels, confidences, self.height, self.width
        )
    
    def _preprocess_depth(self, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """预处理深度图"""
        depth = depth[:, :, 0] * 1
        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()
        mask2 = depth > 0.99
        depth[mask2] = 0.
        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_depth * 100.0 + depth * max_depth * 100.0
        return depth
    
    def _preprocess_state(self, state: np.ndarray, save_object_detection: bool = False, step: int = None) -> np.ndarray:
        """预处理状态：RGB+Depth+Semantic"""
        state = state.transpose(1, 2, 0)
        rgb = state[:, :, :3].astype(np.uint8)
        rgb = rgb[:,:,::-1]
        depth = state[:, :, 3:4]
        
        min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        env_frame_width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        
        sem_seg_pred = self._get_sem_pred(rgb, save_object_detection, step)
        depth = self._preprocess_depth(depth, min_depth, max_depth)
        
        ds = env_frame_width // self.map_args.FRAME_WIDTH
        if ds != 1:
            trans = transforms.Resize((self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH))
            rgb_tensor = torch.from_numpy(rgb.astype(np.uint8)).permute(2,0,1)
            rgb = np.asarray(trans(rgb_tensor).permute(1,2,0))
            depth = depth[ds//2::ds, ds//2::ds]
            sem_seg_pred = sem_seg_pred[ds//2::ds, ds//2::ds]
        
        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)
        return state
    
    def _preprocess_obs(self, obs: np.ndarray, save_object_detection: bool = False, step: int = None) -> np.ndarray:
        """预处理观察"""
        concated_obs = self._concat_obs(obs)
        return self._preprocess_state(concated_obs, save_object_detection, step)
    
    def _batch_obs(self, n_obs: list, save_object_detection: bool = False, step: int = None) -> torch.Tensor:
        """批处理观察"""
        n_states = [self._preprocess_obs(obs, save_object_detection, step) for obs in n_obs]
        max_channels = max([len(state) for state in n_states])
        batch = np.stack([np.pad(state,
                [(0, max_channels - state.shape[0]),
                 (0, 0),
                 (0, 0)],
                mode='constant')
         for state in n_states], axis=0)
        return torch.from_numpy(batch).float().to(self.device)
    
    def toggle_trajectory(self):
        status = self.mapper.toggle_trajectory()
        # print(f"[轨迹] {status}")
    
    def clear_trajectory(self):
        self.mapper.clear_trajectory()
        # print("[轨迹] 已清空")
    
    def get_keyboard_action(self) -> int:
        """获取键盘输入：w=前进 a=左转 d=右转 t=切换轨迹 c=清空轨迹"""
        a = input("action: ")
        if a == 'w':
            return 1
        elif a == 'a':
            return 2
        elif a == 'd':
            return 3
        elif a == 't':
            self.toggle_trajectory()
            return self.get_keyboard_action()
        elif a == 'c':
            self.clear_trajectory()
            return self.get_keyboard_action()
        else:
            return 0
    
    @staticmethod
    def _action_name(action: int) -> str:
        names = {0: 'STOP', 1: 'FORWARD', 2: 'LEFT', 3: 'RIGHT'}
        return names.get(action, f'UNKNOWN({action})')
    
    def close(self):
        # print("\n[Close] 关闭环境...")
        self.envs.close()
        # print("[Close] 完成！")

