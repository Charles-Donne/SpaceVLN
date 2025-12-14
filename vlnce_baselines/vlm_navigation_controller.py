"""
VLM Navigation Controller
=========================
基于VLM的自动导航控制器

继承InteractiveNavigationController的核心功能：
- 语义建图（GroundedSAM + Semantic Mapping）
- 可视化（MapVisualizer）
- 12步×30°环视建图

新增VLM功能：
- LLM高层规划（生成子任务）
- VLM低层动作执行（基于RGB+地图决策）
- 4方向观察收集（前/右/后/左）
- RGB+俯视图拼接可视化（使用环境提供的top_down_map_vlnce）
- 结果保存供后续测评
"""
import os
import cv2
import json
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from habitat import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from vlnce_baselines.interactive_navigation_controller import InteractiveNavigationController
from vlnce_baselines.vlm import LLMPlanner, ActionExecutor, ObservationCollector
from vlnce_baselines.vlm.navigation_config import (
    DIRECTION_STEPS, DIRECTION_NAMES, PANORAMA_CONFIG, ACTION_MAPPING
)


class VLMNavigationController(InteractiveNavigationController):
    """
    VLM导航控制器
    
    继承自InteractiveNavigationController，添加VLM规划和执行功能
    
    工作流程：
    1. 初始环视建图（12步×30°）→ 收集4方向图像
    2. LLM规划 → 生成初始子任务
    3. VLM执行 → 循环执行动作直到子任务完成
    4. 验证环视建图（12步×30°）→ 更新地图和4方向图像
    5. 验证重规划 → 检查完成状态，生成下一子任务
    6. 重复3-5直到导航完成
    
    注意：每次验证重规划前都会执行360°环视，以更新语义地图和当前位置的4方向观察
    """
    
    def __init__(self, config: Config,
                 llm_config_path: str = "vlnce_baselines/vlm/llm_config.yaml",
                 vlm_config_path: str = "vlnce_baselines/vlm/vlm_config.yaml"):
        """
        初始化VLM导航控制器
        
        Args:
            config: Habitat配置
            llm_config_path: LLM配置文件路径
            vlm_config_path: VLM配置文件路径
        """
        # 调用父类初始化（初始化环境、检测、建图、可视化）
        super().__init__(config)
        
        # 初始化VLM模块
        print("\n[Init] 初始化VLM模块...")
        
        # 获取动作参数
        self.turn_angle = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE  # 30°
        self.move_distance = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE  # 0.25m
        
        # 动作空间描述
        self.action_space = f"MOVE_FORWARD ({self.move_distance}m), TURN_LEFT ({self.turn_angle}°), TURN_RIGHT ({self.turn_angle}°), STOP"
        
        # 初始化LLM规划器
        try:
            self.planner = LLMPlanner(llm_config_path, self.action_space)
        except Exception as e:
            print(f"⚠️  LLM Planner初始化失败: {e}")
            self.planner = None
        
        # 初始化VLM执行器
        try:
            self.action_executor = ActionExecutor(vlm_config_path, self.turn_angle, self.move_distance)
        except Exception as e:
            print(f"⚠️  Action Executor初始化失败: {e}")
            self.action_executor = None
        
        # VLM状态
        self.current_subtask = None
        self.subtask_count = 0
        self.progress_summary = ""
        self.subtask_history = []
        self.current_subtask_file = None  # 当前子任务文件路径
        
        # 空间记忆系统（Waypoint Memory）
        self.waypoint_memory = []  # 路径点列表 [{id, position, area_name, description, step, detected_objects}]
        self.waypoint_counter = 0   # 路径点计数器
        
        # 观察缓存
        self.latest_obs = None  # 缓存最新的观察
        self.latest_info = None  # 缓存最新的info（包含top_down_map_vlnce）
        
        # 观察缓存（环视时收集的4方向图像）
        self.direction_images = {}  # {direction_name: image_path}
        self.latest_map_image = None
        
        # ObservationCollector（用于RGB+俯视图拼接和GIF生成）
        self.obs_collector = None
        
        print("[Init] VLM模块初始化完成\n")
    
    def reset_episode(self, episode_id: int = None):
        """重置Episode，包括VLM状态"""
        # 调用父类重置
        super().reset_episode(episode_id)
        
        # 直接使用父类reset后的观察，不需要额外执行STOP动作
        # 父类已经通过envs.reset()获取了初始观察并存储在self.latest_obs中
        
        # 重置VLM状态
        self.current_subtask = None
        self.subtask_count = 0
        self.progress_summary = ""
        self.subtask_history = []
        self.current_subtask_file = None
        self.direction_images = {}
        self.latest_map_image = None
        
        # 重置空间记忆
        self.waypoint_memory = []
        self.waypoint_counter = 0
        
        # 创建VLM专用目录
        self.vlm_dir = os.path.join(
            self.config.RESULTS_DIR, 
            f'episode_{self.current_episode_id}',
            'vlm'
        )
        os.makedirs(self.vlm_dir, exist_ok=True)
        os.makedirs(os.path.join(self.vlm_dir, 'observations'), exist_ok=True)
        os.makedirs(os.path.join(self.vlm_dir, 'subtasks'), exist_ok=True)
        
        # 初始化ObservationCollector（用于RGB+俯视图拼接）
        self.obs_collector = ObservationCollector(os.path.join(self.vlm_dir, 'observations'))
        self.obs_collector.setup_maps_dir(self.vlm_dir)
        
        # 初始化输出记录列表
        self.thinking_outputs = []  # 记录LLM(thinking)的所有输出
        self.action_outputs = []    # 记录VLM(action)的所有输出
    
    def look_around_and_collect(self, phase: str = "initial") -> Tuple[List[str], List[str]]:
        """
        360°环视建图 + 生成4方向全景图
        
        执行12次×30°逆时针旋转（TURN_LEFT），每次转完后拍照并更新地图：
        - step 1: 第1次左转30°后拍照
        - step 2: 第2次左转60°后拍照
        - ...
        - step 12: 第12次左转360°后拍照（回到正前方）
        
        合成4个方向的90°视角全景图：
        - 前方：step-11(330°) + step-12(360°=0°) + step-1(30°) = 前方90°
        - 左侧：step-2(60°) + step-3(90°) + step-4(120°) = 左侧90°
        - 后方：step-5(150°) + step-6(180°) + step-7(210°) = 后方90°
        - 右侧：step-8(240°) + step-9(270°) + step-10(300°) = 右侧90°
        
        所有图像和地图统一保存到 vlm/observations/ 目录
        使用柱面投影拼接生成连贯的全景图
        环视过程不影响current_step和trajectory（环视后恢复）
        
        Args:
            phase: 阶段名称（用于文件命名，如 "initial", "verify_1"）
        
        Returns:
            (image_paths, direction_names) - 4个全景图路径和方向名称
        """
        print(f"\n[环视建图] {phase}...")
        
        # 存储12张环视图像用于合成全景图（step 1-12）
        lookaround_images = []
        total_new_classes = 0
        
        # 保存当前current_step和轨迹，环视结束后恢复
        saved_current_step = self.current_step
        saved_trajectory = self.mapper.trajectory_points.copy() if hasattr(self.mapper, 'trajectory_points') else []
        
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        
        # 直接开始12次旋转（1-12），不保存初始观察
        # step 12会回到正前方（360°）
        for look_step in range(1, 13):  # 1, 2, 3, ..., 12
            print(f"  [{look_step}/12] 第{look_step}次左转 (30°×{look_step}={look_step*30}°)", end="", flush=True)
            
            # 执行旋转
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            outputs = self.envs.step(actions)
            obs, _, dones, _ = [list(x) for x in zip(*outputs)]
            
            if dones[0]:
                print(" - Episode提前结束")
                break
            
            # 更新检测和建图（不保存step文件）
            prev_class_count = len(self.detected_classes)
            batch_obs = self._batch_obs(obs, save_object_detection=False)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            
            map_state = self.mapper.update_map(
                batch_obs, poses, look_step,
                list(self.detected_classes), self.current_episode_id
            )
            
            new_classes = len(self.detected_classes) - prev_class_count
            total_new_classes += new_classes
            
            if new_classes > 0:
                print(f" +{new_classes}类")
            else:
                print()
            
            # 保存所有12张环视图像（用于后续合成全景图）
            rgb_bgr = cv2.cvtColor(obs[0]['rgb'], cv2.COLOR_RGB2BGR)
            lookaround_images.append(rgb_bgr.copy())
        
        # 环视建图完成，恢复current_step和轨迹
        self.current_step = saved_current_step
        self.mapper.trajectory_points = saved_trajectory
        
        # 缓存最后的观察（step 12，回到正前方）
        self.latest_obs = obs[0]
        
        # 缓存最后的观察（最后一次旋转后）
        self.latest_obs = obs[0]
        
        print(f"  扫描完成: +{total_new_classes}类 | 总计{len(self.detected_classes)}类")
        
        # 合成4个方向的全景图（使用真正的全景拼接）
        panorama_paths = []
        panorama_names = []
        
        # 使用固定的水平视场角30°（每次TURN_LEFT旋转30°）
        hfov = 30.0  # 每张图的水平视场角（度）
        
        for config in PANORAMA_CONFIG:
            direction_name = config["name"]
            steps = config["steps"]
            
            # 获取3张图像（注意：steps是1-based索引，需要转换为0-based）
            images_to_stitch = [lookaround_images[s-1] for s in steps]
            
            # 使用柱面投影拼接全景图
            panorama = self._stitch_panorama(images_to_stitch, hfov)
            
            # 保存全景图到 vlm/observations/
            panorama_filename = f"{phase}_panorama_{direction_name.split()[0].lower()}.jpg"
            panorama_path = os.path.join(self.vlm_dir, 'observations', panorama_filename)
            cv2.imwrite(panorama_path, panorama)
            
            panorama_paths.append(panorama_path)
            panorama_names.append(direction_name)
            self.direction_images[direction_name] = panorama_path
        
        # 保存全局地图和局部地图到 vlm/observations/
        episode_dir = os.path.join(self.config.RESULTS_DIR, f'episode_{self.current_episode_id}')
        
        # 使用 step-12 的地图（第12次左转后，完成360°扫描，地图最完整）
        global_map_src = os.path.join(episode_dir, 'global_map', f'step-12.png')
        local_map_src = os.path.join(episode_dir, 'local_map', f'step-12.png')
        
        # 复制到 vlm/observations/ 目录
        global_map_dst = os.path.join(self.vlm_dir, 'observations', f'{phase}_global_map.png')
        local_map_dst = os.path.join(self.vlm_dir, 'observations', f'{phase}_local_map.png')
        
        if os.path.exists(global_map_src):
            import shutil
            shutil.copy(global_map_src, global_map_dst)
            self.latest_map_image = global_map_dst
        else:
            print(f"  ⚠️  Global Map not found: {global_map_src}")
            self.latest_map_image = None
        
        if os.path.exists(local_map_src):
            import shutil
            shutil.copy(local_map_src, local_map_dst)
        else:
            print(f"  ⚠️  Local Map not found: {local_map_src}")
        
        print(f"  4方向全景图已保存 | Step={self.current_step}")
        print("="*60 + "\n")
        
        return panorama_paths, panorama_names
    
    def _get_current_map_path(self) -> str:
        """
        获取当前语义地图路径（使用global_map/目录中的图像，避免重复保存）
        
        Returns:
            global_map目录中上一步保存的地图路径
        """
        # 返回上一步保存的地图（当前步的地图要等step()执行后才会保存）
        episode_dir = os.path.join(
            self.config.RESULTS_DIR, 
            f'episode_{self.current_episode_id}'
        )
        last_step = self.current_step - 1
        map_path = os.path.join(episode_dir, 'global_map', f'step-{last_step}.png')
        self.latest_map_image = map_path
        return map_path

    def get_observations_and_maps(self, phase: str) -> Tuple[List[str], List[str], str, str]:
        """
        从 vlm/observations/ 目录获取4方向全景图和地图
        
        Args:
            phase: 阶段名称（如 "initial", "verify_1"）
            
        Returns:
            (panorama_paths, direction_names, global_map_path, local_map_path)
        """
        panorama_paths = []
        direction_names = []
        
        # 获取4个全景图
        for config in PANORAMA_CONFIG:
            direction_name = config["name"]
            panorama_filename = f"{phase}_panorama_{direction_name.split()[0].lower()}.jpg"
            panorama_path = os.path.join(self.vlm_dir, 'observations', panorama_filename)
            
            if os.path.exists(panorama_path):
                panorama_paths.append(panorama_path)
                direction_names.append(direction_name)
            else:
                print(f"  ⚠️  {direction_name} 未找到: {panorama_filename}")
        
        # 获取地图
        global_map_path = os.path.join(self.vlm_dir, 'observations', f'{phase}_global_map.png')
        local_map_path = os.path.join(self.vlm_dir, 'observations', f'{phase}_local_map.png')
        
        if not os.path.exists(global_map_path):
            print(f"  ⚠️  Global Map 未找到")
            global_map_path = None
        
        if not os.path.exists(local_map_path):
            print(f"  ⚠️  Local Map 未找到")
            local_map_path = None
        
        return panorama_paths, direction_names, global_map_path, local_map_path
    
    def generate_initial_subtask(self) -> Optional[Dict]:
        """
        生成初始子任务
        
        使用环视收集的4方向全景图 + 全局地图 + 局部地图调用LLM生成子任务
        """
        if not self.planner:
            print("✗ LLM Planner未初始化")
            return None
        
        print(f"\n[LLM规划] 生成初始子任务...")
        
        # 从 vlm/observations/ 获取全景图和地图
        image_paths, direction_names, global_map, local_map = self.get_observations_and_maps("initial")
        
        # 验证地图文件存在
        if not global_map or not os.path.exists(global_map):
            print(f"✗ Global map not found: {global_map}")
            return None
        
        # 创建带waypoint标注的地图副本（不覆盖原始地图）
        global_map_for_llm = global_map
        if len(self.waypoint_memory) > 0:
            global_map_img = cv2.imread(global_map)
            global_map_img = self.visualize_waypoints_on_map(global_map_img)
            # 保存为单独的可视化版本
            global_map_for_llm = os.path.join(self.vlm_dir, 'observations', 'initial_global_map_with_waypoints.png')
            cv2.imwrite(global_map_for_llm, global_map_img)
        
        # 调用LLM生成初始子任务（第一次规划，无detected_landmarks和waypoint_summary）
        response = self.planner.generate_initial_subtask(
            instruction=self.current_instruction,
            observation_images=image_paths,
            direction_names=direction_names,
            global_map_image=global_map_for_llm,  # 使用带waypoint标注的版本
            local_map_image=local_map
        )
        
        if not response:
            print("✗ LLM未返回有效响应")
            return None
        
        # 记录thinking输出
        thinking_record = {
            "step": self.current_step,
            "phase": "initial_planning",
            "subtask_count": self.subtask_count + 1,
            "prompt_type": "initial",
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self.thinking_outputs.append(thinking_record)
        self._save_thinking_output(thinking_record)
        
        # 保存子任务
        self.current_subtask = response
        self.subtask_count += 1
        self.progress_summary = ""
        
        # 记录当前位置信息（用于后续验证参考）
        self.current_position_info = {
            'waypoint': response.get('waypoint', 'Unknown'),
            'observation': response.get('current_observation', ''),
            'step': self.current_step
        }
        
        # 创建路径点记录（空间记忆）
        waypoint_desc = response.get('waypoint', 'Unknown location')
        # 不传position参数，让add_waypoint()从mapper.curr_loc获取正确的地图像素坐标
        self.add_waypoint(waypoint_desc)
        
        # 记录并动态更新目标landmark
        subtask_landmark = response.get('subtask_landmark', None)
        if subtask_landmark:
            # 验证：只要在mapping_classes中（能被GroundedSAM检测）就可以作为landmark
            if subtask_landmark in self.mapping_classes:
                # 动态更新landmark_classes（只标注当前子任务的目标）
                self.landmark_classes = [subtask_landmark]
                self.target_landmark = subtask_landmark
                print(f"  🎯 目标Landmark已设定: {self.target_landmark}")
                print(f"  📍 已动态更新landmark_classes: {self.landmark_classes}")
            else:
                print(f"  ⚠️  警告: '{subtask_landmark}' 不在mapping_classes中，GroundedSAM无法检测")
                print(f"  💡 可检测类别: {', '.join(self.mapping_classes)}")
                self.target_landmark = None
                self.landmark_classes = []  # 重置为空
        else:
            print(f"  ℹ️  未指定subtask_landmark，不标注landmark")
            self.target_landmark = None
            self.landmark_classes = []  # 重置为空
        
        self._save_subtask(response, "initial")
        
        # 打印子任务信息
        self._print_subtask_info(response, is_initial=True)
        
        return response
    
    def verify_and_replan(self) -> Tuple[bool, Optional[Dict]]:
        """
        验证当前子任务并重新规划
        
        流程：
        1. 执行360°环视建图（更新语义地图）
        2. 生成当前位置的4方向全景图
        3. 调用LLM验证子任务完成状态
        4. 如未完成，生成新子任务
        
        Returns:
            (is_completed, new_subtask)
        """
        if not self.planner or not self.current_subtask:
            return False, None
        
        # 重新执行环视建图并生成全景图
        phase = f"verify_{self.subtask_count}"
        print(f"\n[验证] 重新环视以验证子任务完成状态...")
        image_paths, direction_names = self.look_around_and_collect(phase)
        
        if not image_paths:
            print("✗ 环视建图失败")
            return False, None
        # 从 vlm/observations/ 获取地图（已在 look_around_and_collect 中保存）
        _, _, global_map, local_map = self.get_observations_and_maps(phase)
        
        # 验证地图文件存在
        if not global_map or not os.path.exists(global_map):
            print(f"✗ Global map not found: {global_map}")
            return False, None
        
        # 创建带waypoint标注的地图副本（不覆盖原始地图）
        global_map_for_llm = global_map
        if len(self.waypoint_memory) > 0:
            global_map_img = cv2.imread(global_map)
            global_map_img = self.visualize_waypoints_on_map(global_map_img)
            # 保存为单独的可视化版本
            global_map_for_llm = os.path.join(episode_dir, 'global_map', f'step-{last_saved_step}_with_waypoints.png')
            cv2.imwrite(global_map_for_llm, global_map_img)
        
        # 获取已检测到的landmark类别
        detected_landmarks = list(self.detected_classes) if hasattr(self, 'detected_classes') else []
        
        # 获取路径点历史记录
        waypoint_summary = self.get_waypoint_summary()
        
        # 调用LLM验证（全局地图必需，局部地图可选，传递实际检测到的类别）
        response, is_completed = self.planner.verify_and_replan(
            instruction=self.current_instruction,
            current_subtask=self.current_subtask,
            observation_images=image_paths,
            direction_names=direction_names,
            global_map_image=global_map_for_llm,  # 使用带waypoint标注的版本
            local_map_image=local_map if os.path.exists(local_map) else None,
            detected_landmarks=detected_landmarks,
            waypoint_summary=waypoint_summary
        )
        
        print(f"  🏷️  Detected landmarks: {detected_landmarks if detected_landmarks else 'None'}")
        
        if not response:
            print("✗ LLM验证未返回有效响应")
            return False, None
        
        # 记录thinking输出
        thinking_record = {
            "step": self.current_step,
            "phase": f"verify_subtask_{self.subtask_count}",
            "subtask_count": self.subtask_count,
            "prompt_type": "verification",
            "is_completed": is_completed,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self.thinking_outputs.append(thinking_record)
        self._save_thinking_output(thinking_record)
        
        if is_completed:
            print(f"\n[子任务完成] #{self.subtask_count}")
            
            # 检查是否是最终子任务
            if response.get('is_final_subtask', False):
                print("到达最终目的地")
                return True, response
            
            # 清空上一个子任务的轨迹（每个子任务独立显示）
            print("  清空上一子任务轨迹")
            self.mapper.clear_trajectory()
            
            # 更新到新子任务
            self.subtask_count += 1
            self.current_subtask = response
            self.progress_summary = ""
            
            # 更新当前位置信息（用于后续参考）
            self.current_position_info = {
                'waypoint': response.get('waypoint', 'Unknown'),
                'observation': response.get('current_observation', ''),
                'step': self.current_step
            }
            
            # 创建路径点记录（空间记忆）
            waypoint_desc = response.get('waypoint', 'Unknown location')
            # 不传position参数，让add_waypoint()从mapper.curr_loc获取正确的地图像素坐标
            self.add_waypoint(waypoint_desc)
            
            # 动态更新目标landmark（自动替换上一个子任务的landmark）
            subtask_landmark = response.get('subtask_landmark', None)
            if subtask_landmark:
                # 验证：只要在mapping_classes中（能被GroundedSAM检测）就可以作为landmark
                if subtask_landmark in self.mapping_classes:
                    # 动态更新landmark_classes（只标注当前子任务的目标）
                    # 注意：这里更新会自动替换掉上一个子任务的landmark标注
                    self.landmark_classes = [subtask_landmark]
                    self.target_landmark = subtask_landmark
                    print(f"  🎯 新目标Landmark: {self.target_landmark}")
                    print(f"  📍 已更新landmark_classes: {self.landmark_classes} (替换上一子任务)")
                else:
                    print(f"  ⚠️  警告: '{subtask_landmark}' 不在mapping_classes中，GroundedSAM无法检测")
                    self.target_landmark = None
                    self.landmark_classes = []  # 重置为空
            else:
                print(f"  ℹ️  未指定新landmark，不标注landmark")
                self.target_landmark = None
                self.landmark_classes = []  # 重置为空
            
            self._save_subtask(response, f"subtask_{self.subtask_count}")
            self._print_subtask_info(response)
        else:
            print(f"\n[子任务继续] #{self.subtask_count} 未完成")
            
            # 即使未完成也更新位置观察（用于记录轨迹）
            if 'current_observation' in response:
                self.current_position_info = {
                    'waypoint': response.get('waypoint', getattr(self, 'current_position_info', {}).get('waypoint', 'Unknown')),
                    'observation': response.get('current_observation', ''),
                    'step': self.current_step
                }
            self.current_subtask = response
            self._save_subtask(response, f"subtask_{self.subtask_count}_refined")
        
        return is_completed, response
    
    def execute_action_with_vlm(self) -> Tuple[Optional[int], Optional[str], bool]:
        """
        使用VLM决策并执行动作
        
        Returns:
            (action_id, action_name, should_stop)
        """
        if not self.action_executor or not self.current_subtask:
            return None, None, True
        
        # 获取当前观察：使用缓存的观察或通过旋转获取
        if self.latest_obs is not None:
            obs = self.latest_obs
        else:
            # 如果没有缓存，执行一次右转再左转回来获取观察
            actions = [{"action": HabitatSimActions.TURN_RIGHT}]
            outputs = self.envs.step(actions)
            obs, _, dones, _ = [list(x) for x in zip(*outputs)]
            if dones[0]:
                print("⚠️ Episode结束")
                return None, None, True
            
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            outputs = self.envs.step(actions)
            obs, _, dones, _ = [list(x) for x in zip(*outputs)]
            if dones[0]:
                print("⚠️ Episode结束")
                return None, None, True
            obs = obs[0]
        
        # 使用上一步保存的图像（当前步图像要等step()执行后才会保存）
        episode_dir = os.path.join(
            self.config.RESULTS_DIR, 
            f'episode_{self.current_episode_id}'
        )
        last_step = self.current_step - 1  # 上一步已保存的文件
        fp_image = os.path.join(episode_dir, 'rgb', f'step-{last_step}.png')
        
        # 如果rgb/中的图像还不存在，用当前观察创建临时文件
        if not os.path.exists(fp_image):
            rgb_bgr = cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR)
            temp_image = os.path.join(
                self.vlm_dir, 'observations',
                f'step{last_step}_first_person.jpg'
            )
            cv2.imwrite(temp_image, rgb_bgr)
            fp_image = temp_image
        
        # 获取当前地图路径和检测图像
        self._get_current_map_path()
        
        # 获取detection图像路径（如果存在）
        detection_image = os.path.join(episode_dir, 'detection', f'step-{last_step}.png')
        if not os.path.exists(detection_image):
            detection_image = None
        
        # 获取局部地图路径
        local_map = os.path.join(episode_dir, 'local_map', f'step-{last_step}.png')
        if not os.path.exists(local_map):
            local_map = None
        
        # 获取已检测的landmark类别
        detected_landmarks = ', '.join(self.detected_classes) if hasattr(self, 'detected_classes') and self.detected_classes else None
        
        # 调用VLM决策
        action_id, action_name, updated_progress, response = self.action_executor.decide_action(
            subtask_destination=self.current_subtask.get('subtask_destination', ''),
            subtask_instruction=self.current_subtask.get('subtask_instruction', ''),
            first_person_image=fp_image,
            action_mapping=ACTION_MAPPING,
            progress_summary=self.progress_summary,
            detection_image=detection_image,
            local_map_image=local_map,
            detected_landmarks=detected_landmarks
        )
        
        if action_id is None:
            print("✗ VLM决策失败")
            return None, None, True
        
        # 记录action输出
        action_record = {
            "step": self.current_step,
            "subtask_count": self.subtask_count,
            "action_name": action_name,
            "action_id": action_id,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self.action_outputs.append(action_record)
        self._save_action_output(action_record)
        
        # 更新进度
        self.progress_summary = updated_progress
        
        # 检查是否停止
        should_stop = (action_name == "STOP")
        
        return action_id, action_name, should_stop
    
    def step_with_vlm(self, action: int, action_name: str = "", save_vis: bool = True) -> Dict[str, Any]:
        """
        执行VLM决策的动作（调用父类step方法）并缓存观察
        
        Args:
            action: 动作ID
            action_name: 动作名称（用于可视化）
            save_vis: 是否保存可视化
            
        Returns:
            步骤结果字典
        """
        result = self.step(action, save_vis)
        # 缓存最新观察和info用于下次VLM决策和可视化
        self.latest_obs = result.get('obs', None)
        self.latest_info = result.get('info', None)
        
        # 保存RGB+俯视图拼接可视化（使用环境提供的top_down_map_vlnce）
        if save_vis and self.obs_collector and self.latest_obs is not None:
            subtask_text = None
            if self.current_subtask:
                subtask_text = self.current_subtask.get('subtask_instruction', '')
            
            distance = 0.0
            if self.latest_info:
                distance = self.latest_info.get('distance_to_goal', 0.0)
            
            self.obs_collector.save_step_visualization(
                observations=self.latest_obs,
                info=self.latest_info or {},
                step=self.current_step,
                instruction=self.current_instruction,
                current_subtask=subtask_text,
                distance=distance,
                action=action_name
            )
        
        return result
    
    def run_vlm_navigation(self, max_steps: int = 500, 
                          max_subtask_steps: int = 50,
                          verify_interval: int = 10) -> Dict[str, Any]:
        """
        运行完整的VLM导航流程
        
        Args:
            max_steps: 最大总步数
            max_subtask_steps: 每个子任务最大步数
            verify_interval: 验证间隔步数
            
        Returns:
            导航结果字典
        """
        print("\n" + "="*60)
        print("启动VLM自动导航")
        print("="*60)
        print(f"指令: {self.current_instruction}")
        print(f"最大步数: {max_steps} | 子任务步数: {max_subtask_steps} | 验证间隔: {verify_interval}")
        print("="*60 + "\n")
        
        # 1. 环视建图 + 收集观察
        self.look_around_and_collect()
        
        # 2. 生成初始子任务
        subtask = self.generate_initial_subtask()
        if not subtask:
            print("✗ 初始子任务生成失败")
            return {
                'success': False,
                'total_steps': self.current_step,
                'subtask_count': 0,
                'detected_classes': list(self.detected_classes) if hasattr(self, 'detected_classes') else [],
                'gif_path': None,
                'result_file': None,
                'reason': 'initial_subtask_failed'
            }
        
        # 3. 主导航循环
        total_steps = self.current_step
        subtask_steps = 0
        navigation_complete = False
        
        while total_steps < max_steps:
            # VLM决策动作
            action_id, action_name, should_stop = self.execute_action_with_vlm()
            
            if action_id is None:
                print("VLM决策失败，尝试手动输入")
                action_id = self.get_keyboard_action()
                action_name = self._action_name(action_id)
                should_stop = (action_id == 0)
            
            # 如果VLM决定停止 → 验证子任务
            if should_stop:
                is_completed, new_subtask = self.verify_and_replan()
                
                if is_completed and new_subtask and new_subtask.get('is_final_subtask', False):
                    print("\n[导航完成]")
                    navigation_complete = True
                    break
                
                subtask_steps = 0
                continue
            
            # 执行动作（传入action_name用于可视化）
            result = self.step_with_vlm(action_id, action_name=action_name, save_vis=True)
            total_steps = self.current_step
            subtask_steps += 1
            
            print(f"[Step {total_steps}] {action_name} | 子任务步数: {subtask_steps}")
            
            if result['done']:
                print("\nEpisode自动完成")
                navigation_complete = True
                break
            
            # 定期验证
            if subtask_steps >= verify_interval:
                is_completed, _ = self.verify_and_replan()
                if is_completed:
                    subtask_steps = 0
            
            # 子任务超时
            if subtask_steps >= max_subtask_steps:
                print(f"\n[警告] 子任务超时 ({max_subtask_steps}步)，重新规划")
                _, _ = self.verify_and_replan()
                subtask_steps = 0
        
        # 4. 保存GIF动画
        gif_path = None
        if self.obs_collector:
            gif_path = self.obs_collector.save_gif(fps=2)
        
        # 5. 保存结果（供后续测评）
        final_result = self._save_navigation_result(navigation_complete, total_steps)
        
        return {
            'success': navigation_complete,
            'total_steps': total_steps,
            'subtask_count': self.subtask_count,
            'detected_classes': list(self.detected_classes),
            'gif_path': gif_path,
            'result_file': final_result
        }
    
    def _save_subtask(self, subtask: Dict, name: str):
        """保存子任务到文件"""
        filepath = os.path.join(self.vlm_dir, 'subtasks', f'{name}.json')
        
        data = {
            'episode_id': self.current_episode_id,
            'instruction': self.current_instruction,
            'subtask_id': self.subtask_count,
            'step': self.current_step,
            'timestamp': datetime.now().isoformat(),
            'subtask': subtask
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.subtask_history.append(data)
        self.current_subtask_file = filepath
    
    def _save_thinking_output(self, thinking_record: Dict):
        """保存单次thinking(LLM)输出到文件"""
        thinking_file = os.path.join(self.vlm_dir, 'thinking_outputs.jsonl')
        
        # 追加模式保存为JSONL格式（每行一个JSON）
        with open(thinking_file, 'a', encoding='utf-8') as f:
            json.dump(thinking_record, f, ensure_ascii=False)
            f.write('\n')
    
    def _save_action_output(self, action_record: Dict):
        """保存单次action(VLM)输出到文件"""
        action_file = os.path.join(self.vlm_dir, 'action_outputs.jsonl')
        
        # 追加模式保存为JSONL格式（每行一个JSON）
        with open(action_file, 'a', encoding='utf-8') as f:
            json.dump(action_record, f, ensure_ascii=False)
            f.write('\n')
    
    def _save_navigation_result(self, success: bool, total_steps: int) -> str:
        """
        保存导航结果（供后续测评）
        
        Args:
            success: 是否成功
            total_steps: 总步数
            
        Returns:
            结果文件路径
        """
        # 尝试获取评测指标（如果有的话）
        metrics = {}
        if self.latest_info:
            metrics = {
                'distance_to_goal': self.latest_info.get('distance_to_goal', -1),
                'success': self.latest_info.get('success', success),
                'spl': self.latest_info.get('spl', 0.0),
                'path_length': self.latest_info.get('path_length', 0.0),
                'oracle_success': self.latest_info.get('oracle_success', False)
            }
        
        result = {
            'episode_id': self.current_episode_id,
            'instruction': self.current_instruction,
            'success': success,
            'total_steps': total_steps,
            'subtask_count': self.subtask_count,
            'detected_classes': list(self.detected_classes),
            'subtask_history': self.subtask_history,
            'thinking_count': len(self.thinking_outputs),
            'action_count': len(self.action_outputs),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存到vlm目录
        filepath = os.path.join(self.vlm_dir, 'result.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"📁 结果已保存: {filepath}")
        print(f"   Steps: {total_steps} | Subtasks: {self.subtask_count}")
        print(f"   Thinking(LLM) Calls: {len(self.thinking_outputs)} | Action(VLM) Calls: {len(self.action_outputs)}")
        print(f"   Outputs: vlm/thinking_outputs.jsonl, vlm/action_outputs.jsonl")
        if metrics:
            print(f"   Success: {metrics.get('success', success)} | SPL: {metrics.get('spl', 0.0):.4f}")
        print(f"{'='*60}")
        
        return filepath
    
    def record_action(self, action_name: str, action_id: int, vlm_response: Dict = None):
        """
        记录动作到当前子任务文件（与llm_vlm_control兼容的格式）
        
        Args:
            action_name: 动作名称
            action_id: 动作ID
            vlm_response: VLM响应字典（可选）
        """
        if not self.current_subtask_file or not os.path.exists(self.current_subtask_file):
            return
        
        action_data = {
            "step": self.current_step,
            "action_name": action_name,
            "action_id": action_id,
        }
        
        if self.latest_info:
            action_data["distance_to_goal"] = self.latest_info.get("distance_to_goal", -1)
        
        if vlm_response:
            action_data["vlm_response"] = {
                k: vlm_response.get(k, "") 
                for k in ['observation', 'reasoning', 'action', 'progress_summary']
            }
        
        # 读取并更新文件
        try:
            with open(self.current_subtask_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "actions" not in data:
                data["actions"] = []
            data["actions"].append(action_data)
            
            with open(self.current_subtask_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 记录动作失败: {e}")
    
    def _print_subtask_info(self, response: Dict, is_initial: bool = False):
        """打印子任务信息"""
        title = "初始子任务" if is_initial else f"子任务 #{self.subtask_count}"
        print(f"\n===== {title} =====")
        print(f"全局指令: {self.current_instruction}")
        print(f"Waypoint: {response.get('waypoint', 'N/A')}")
        print(f"环境观察: {response.get('current_observation', 'N/A')}")
        print(f"目的地: {response.get('subtask_destination', 'N/A')}")
        print(f"目标Landmark: {response.get('subtask_destination_landmark', 'N/A')}")
        print(f"子任务指令: {response.get('subtask_instruction', 'N/A')}")
        print(f"规划提示: {response.get('planning_hints', 'N/A')}")
        print(f"完成条件: {response.get('completion_criteria', 'N/A')}")
        print(f"是否最终: {response.get('is_final_subtask', False)}")
        print(f"{'='*50}\n")
    def add_waypoint(self, waypoint_description: str, position: np.ndarray = None) -> int:
        """
        添加路径点到空间记忆
        
        Args:
            waypoint_description: 路径点描述（格式: "<Area Type> - <Key Landmarks>"）
            position: 当前位置地图像素坐标（如果为None则从mapper获取）
            
        Returns:
            waypoint_id: 新添加的路径点ID
        """
        self.waypoint_counter += 1
        waypoint_id = self.waypoint_counter
        
        # 创建路径点记录 - 只保存编号和描述
        waypoint = {
            "id": waypoint_id,
            "waypoint": waypoint_description
        }
        
        self.waypoint_memory.append(waypoint)
        
        print(f"  📍 Waypoint #{waypoint_id} 已记录: {waypoint_description}")
        
        # 保存到文件
        self._save_waypoint_memory()
        
        return waypoint_id
    
    def get_waypoint_summary(self) -> str:
        """
        获取路径点摘要（用于LLM提示词）
        
        只在验证重规划时使用，初始规划时为空
        
        Returns:
            路径点摘要字符串
        """
        if not self.waypoint_memory:
            return ""
        
        summary_lines = []
        for wp in self.waypoint_memory:
            # 格式: "1. Bedroom(Current) - near bed and door"
            summary_lines.append(f"{wp['id']}. {wp['waypoint']}")
        
        return "\n".join(summary_lines)
    
    def _save_waypoint_memory(self):
        """保存路径点记忆到JSON文件"""
        waypoint_file = os.path.join(self.vlm_dir, 'waypoint_memory.json')
        
        data = {
            "episode_id": self.current_episode_id,
            "instruction": self.current_instruction,
            "waypoints": self.waypoint_memory,
            "total_waypoints": len(self.waypoint_memory),
            "last_updated_step": self.current_step
        }
        
        try:
            with open(waypoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  保存waypoint记忆失败: {e}")
    
    def visualize_waypoints_on_map(self, map_image: np.ndarray) -> np.ndarray:
        """
        在地图上可视化路径点，使用深红色圆圈和白色编号
        坐标从mapper的trajectory_points获取（过去的观察位置）
        
        Args:
            map_image: 输入地图图像（BGR格式）
            
        Returns:
            标注了路径点的地图图像
        """
        if not self.waypoint_memory:
            return map_image
        
        # 复制地图避免修改原图
        annotated_map = map_image.copy()
        
        # 从mapper获取轨迹点列表
        if not hasattr(self.mapper, 'trajectory_points') or len(self.mapper.trajectory_points) == 0:
            return annotated_map
        
        trajectory_points = self.mapper.trajectory_points
        
        for wp in self.waypoint_memory:
            try:
                wp_id = wp["id"]
                
                # 使用对应的轨迹点作为waypoint位置
                # waypoint是在特定步骤记录的，使用该步骤的轨迹点
                # 由于waypoint数量远少于轨迹点，使用简单策略：均匀分布
                if wp_id <= len(trajectory_points):
                    # 使用记录waypoint时的轨迹点索引
                    # 简单策略：假设waypoint按顺序记录，使用对应比例的轨迹点
                    idx = min(wp_id - 1, len(trajectory_points) - 1)
                    idx = int(idx * len(trajectory_points) / len(self.waypoint_memory)) if len(self.waypoint_memory) > 1 else 0
                    x, y = trajectory_points[idx]
                else:
                    # 如果超出范围，使用最后一个轨迹点
                    x, y = trajectory_points[-1]
                
                # 检查坐标是否在地图范围内
                h, w = map_image.shape[:2]
                if not (0 <= x < w and 0 <= y < h):
                    continue
                
                # 绘制圆形标记（深红色）
                cv2.circle(annotated_map, (x, y), 15, (0, 0, 139), -1)  # 深红色填充圆
                cv2.circle(annotated_map, (x, y), 15, (255, 255, 255), 2)  # 白色边框
                
                # 绘制ID数字（白色）
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(wp_id)
                text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
                text_x = x - text_size[0] // 2
                text_y = y + text_size[1] // 2
                cv2.putText(annotated_map, text, (text_x, text_y), 
                           font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
            except Exception as e:
                print(f"⚠️  标注waypoint {wp['id']} 失败: {e}")
                continue
        
        return annotated_map
    
    def _stitch_panorama(self, images: List[np.ndarray], hfov: float) -> np.ndarray:
        """
        使用柱面投影拼接全景图
        
        Args:
            images: 3张图像列表（BGR格式）
            hfov: 单张图像的水平视场角（度）
            
        Returns:
            拼接后的全景图
        """
        if len(images) != 3:
            # 如果不是3张图，简单拼接
            return np.hstack(images)
        
        h, w = images[0].shape[:2]
        
        # 柱面投影参数
        # 3张图×hfov = 90°总视场
        total_fov = hfov * 3
        focal_length = w / (2 * np.tan(np.radians(hfov / 2)))
        
        # 计算柱面全景图尺寸
        cylinder_width = int(2 * focal_length * np.tan(np.radians(total_fov / 2)))
        cylinder_height = h
        
        # 创建全景画布
        panorama = np.zeros((cylinder_height, cylinder_width, 3), dtype=np.uint8)
        
        # 对每张图进行柱面投影并拼接
        for idx, img in enumerate(images):
            # 计算当前图像的角度偏移（中间图为0，左右分别为±hfov）
            angle_offset = (idx - 1) * hfov  # -hfov, 0, +hfov
            
            # 柱面投影
            projected = self._cylindrical_projection(img, focal_length, angle_offset)
            
            # 计算在全景图中的位置
            start_x = int((cylinder_width / 2) + focal_length * np.tan(np.radians(angle_offset - hfov/2)))
            end_x = int((cylinder_width / 2) + focal_length * np.tan(np.radians(angle_offset + hfov/2)))
            
            # 确保索引在范围内
            start_x = max(0, start_x)
            end_x = min(cylinder_width, end_x)
            proj_w = projected.shape[1]
            
            # 拼接到全景图
            if end_x - start_x > 0:
                # 调整projected的宽度以匹配目标区域
                scale = (end_x - start_x) / proj_w
                resized = cv2.resize(projected, (end_x - start_x, cylinder_height))
                panorama[:, start_x:end_x] = resized
        
        return panorama
    
    def _cylindrical_projection(self, img: np.ndarray, focal_length: float, angle_offset: float = 0) -> np.ndarray:
        """
        将平面图像投影到柱面
        
        Args:
            img: 输入图像
            focal_length: 焦距
            angle_offset: 角度偏移（度）
            
        Returns:
            柱面投影后的图像
        """
        h, w = img.shape[:2]
        
        # 创建输出图像
        output = np.zeros_like(img)
        
        # 图像中心
        cx, cy = w / 2, h / 2
        
        # 对每个像素进行反向投影
        for y in range(h):
            for x in range(w):
                # 柱面坐标到球面坐标
                theta = (x - cx) / focal_length
                h_coord = (y - cy) / focal_length
                
                # 球面坐标到平面坐标（加上角度偏移）
                theta_offset = theta + np.radians(angle_offset)
                x_src = focal_length * np.tan(theta_offset) + cx
                y_src = h_coord * focal_length / np.cos(theta) + cy
                
                # 双线性插值
                if 0 <= x_src < w-1 and 0 <= y_src < h-1:
                    x0, y0 = int(x_src), int(y_src)
                    x1, y1 = x0 + 1, y0 + 1
                    
                    dx, dy = x_src - x0, y_src - y0
                    
                    output[y, x] = (1-dx)*(1-dy)*img[y0, x0] + dx*(1-dy)*img[y0, x1] + \
                                   (1-dx)*dy*img[y1, x0] + dx*dy*img[y1, x1]
        
        return output
