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
from vlnce_baselines.vlm import (
    LLMPlanner, ActionExecutor, SaveManager, NavigationVisualizer
)
from vlnce_baselines.visualization import PanoramaGenerator
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
        self.subtask_attempt = 0  # 当前子任务的尝试次数（a, b, c...）
        self.progress_summary = ""
        self.subtask_history = []
        self.current_subtask_file = None
        
        # 初始化管理器
        self.save_manager = None  # 在reset_episode时初始化
        # waypoint_manager已废弃，直接使用mapper.add_waypoint()
        
        # 观察缓存
        self.latest_obs = None  # 缓存最新的观察
        self.latest_info = None  # 缓存最新的info（包含top_down_map_vlnce）
        
        # 观察缓存（环视时收集的4方向图像）
        self.direction_images = {}  # {direction_name: image_path}
        self.latest_map_image = None
        
        # NavigationVisualizer（用于RGB+俯视图拼接和GIF生成）
        self.nav_visualizer = None
        
        # PanoramaGenerator（用于全景图拼接和标注）
        self.panorama_generator = PanoramaGenerator()
        
        print("[Init] VLM模块初始化完成\n")
    
    def reset_episode(self, episode_id: int = None):
        """重置Episode，包括VLM状态"""
        # 清理之前episode的输出目录
        if episode_id is not None:
            import shutil
            old_episode_dir = os.path.join(self.config.RESULTS_DIR, f'episode_{episode_id}')
            if os.path.exists(old_episode_dir):
                print(f"[Reset] 清理旧数据: {old_episode_dir}")
                shutil.rmtree(old_episode_dir)
        
        # 调用父类重置
        super().reset_episode(episode_id)
        
        # 初始化SaveManager（使用RESULTS_DIR作为输出根目录）
        self.save_manager = SaveManager(self.config.RESULTS_DIR, self.current_episode_id)
        
        # 重置VLM状态
        self.current_subtask = None
        self.subtask_count = 0
        self.subtask_attempt = 0  # 重置尝试计数
        self.progress_summary = ""
        self.subtask_history = []
        self.current_subtask_file = None
        self.direction_images = {}
        self.latest_map_image = None
        
        # waypoint已集成到mapper中，mapper.reset()会自动清空
        
        print(f"[Reset] Episode {self.current_episode_id} 重置完成")
        
        # 初始化NavigationVisualizer（用于RGB+俯视图拼接和GIF生成）
        visualization_dir = os.path.join(self.episode_dir, 'visualization')
        self.nav_visualizer = NavigationVisualizer(visualization_dir)
        self.nav_visualizer.setup_maps_dir(self.episode_dir)
        
        # 初始化输出记录列表
        self.thinking_outputs = []  # 记录LLM(thinking)的所有输出
        self.action_outputs = []    # 记录VLM(action)的所有输出
    
    @property
    def episode_dir(self) -> str:
        """获取当前episode的输出目录（动态属性，自动根据current_episode_id生成）"""
        return os.path.join(self.config.RESULTS_DIR, f'episode_{self.current_episode_id}')
    
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
        
        # ===== 关键修复：环视前清空旧landmark和轨迹 =====
        # 这样环视期间渲染的地图不会显示上一个子任务的landmark标注
        print("[状态清理] 清空轨迹和landmark（准备环视建图）")
        self.mapper.clear_trajectory()
        self.landmark_classes = []
        if hasattr(self, 'current_step_landmarks'):
            self.current_step_landmarks.clear()
        
        # 存储12张环视图像用于合成全景图（step 1-12）
        lookaround_images = []
        total_new_classes = 0
        
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        
        # 直接开始12次旋转，每一步保存rgb、detection、maps
        # 使用累加的self.current_step，避免覆盖之前的数据
        for i in range(1, 13):  # 12次旋转
            self.current_step += 1  # 累加总步数
            look_step = self.current_step
            print(f"  [{i}/12] 第{i}次左转 (30°×{i}={i*30}°)", end="", flush=True)
            
            # 执行旋转
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            
            if dones[0]:
                print(" - Episode提前结束")
                break
            
            # 更新检测和建图（每一步都保存）
            prev_class_count = len(self.detected_classes)
            batch_obs = self._batch_obs(obs, save_object_detection=True)  # 保存检测结果
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            
            map_state = self.mapper.update_map(
                batch_obs, poses, look_step,
                list(self.detected_classes), self.current_episode_id
            )
            
            new_classes = len(self.detected_classes) - prev_class_count
            total_new_classes += new_classes
            
            # 调用visualizer保存所有数据（RGB、检测、全局地图、局部地图）
            # 自动从mapper获取waypoint并渲染（忽略descriptions，可视化不需要）
            wp_positions, wp_ids, _ = self.mapper.get_waypoints()
            rgb_bgr = cv2.cvtColor(obs[0]['rgb'], cv2.COLOR_RGB2BGR)
            paths, detected_landmarks_step = self.visualizer.save_step_visualization(
                step=look_step,
                episode_id=self.current_episode_id,
                rgb=rgb_bgr,
                full_map=map_state['full_map'],
                trajectory_points=map_state['trajectory_points'],
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
                waypoint_positions=wp_positions,
                waypoint_ids=wp_ids,
                phase=phase
            )
            
            # 累积当前step检测到的landmarks
            if detected_landmarks_step:
                if not hasattr(self, 'current_step_landmarks'):
                    self.current_step_landmarks = {}
                self.current_step_landmarks[look_step] = detected_landmarks_step
            
            # 保存导航可视化（RGB+俯视图拼接）
            if self.nav_visualizer:
                subtask_text = self.current_subtask.get('subtask_instruction', '') if self.current_subtask else f"[环视建图 {phase}]"
                distance = 0.0
                if infos and len(infos) > 0:
                    distance = infos[0].get('distance_to_goal', 0.0)
                
                # 环视阶段的subtask_id为phase（如initial, verify_1a）
                self.nav_visualizer.save_step_visualization(
                    observations=obs[0],
                    info=infos[0] if infos and len(infos) > 0 else {},
                    step=look_step,
                    instruction=self.current_instruction,
                    current_subtask=subtask_text,
                    distance=distance,
                    action=f"TURN_LEFT (360°环视 {i}/12)",
                    subtask_id=phase
                )
            
            if new_classes > 0:
                print(f" +{new_classes}类")
            else:
                print()
            
            # 保存所有12张环视图像（用于后续合成全景图）
            lookaround_images.append(rgb_bgr.copy())
        
        # 环视建图完成
        # 注意：不恢复轨迹，轨迹会自然显示在地图上
        # 如需清空轨迹，应在verify_and_replan中的子任务完成时调用mapper.clear_trajectory()
        
        # 缓存最后的观察（step 12，回到正前方）
        self.latest_obs = obs[0]
        
        # 缓存最后的观察（最后一次旋转后）
        self.latest_obs = obs[0]
        
        print(f"  扫描完成: +{total_new_classes}类 | 总计{len(self.detected_classes)}类")
        
        # 直接为每个方向拼接3张图片生成90°全景图
        panorama_paths = []
        panorama_names = []
        panorama_dir = os.path.join(self.config.RESULTS_DIR, f"episode_{self.current_episode_id}", "panoramas")
        os.makedirs(panorama_dir, exist_ok=True)
        
        for config in PANORAMA_CONFIG:
            direction_name = config["name"]
            steps = config["steps"]  # 3张图片的索引（如[1,12,11]）
            
            # 获取该方向的3张图片（steps是1-based，需要转为0-based索引）
            direction_images = [lookaround_images[step-1] for step in steps]
            
            # 使用PanoramaGenerator拼接3张图片并添加方向标注
            panorama = self.panorama_generator.create_panorama(direction_images, direction_name)
            
            # 保存全景图（统一使用PNG格式）
            panorama_filename = f"{phase}_panorama_{direction_name.split()[0].lower()}.png"
            panorama_path = os.path.join(panorama_dir, panorama_filename)
            cv2.imwrite(panorama_path, panorama)
            
            panorama_paths.append(panorama_path)
            panorama_names.append(direction_name)
            self.direction_images[direction_name] = panorama_path
        
        # 保存全局地图和局部地图到 vlm/observations/
        # 使用当前step的地图（环视完成后的最新地图）
        self.latest_global_map = os.path.join(self.episode_dir, 'global_map', f'step_{self.current_step:04d}_{phase}.png')
        self.latest_local_map = os.path.join(self.episode_dir, 'local_map', f'step_{self.current_step:04d}_{phase}.png')
        
        if not os.path.exists(self.latest_global_map):
            print(f"  ⚠️  Global Map not found: {self.latest_global_map}")
            self.latest_global_map = None
        
        if not os.path.exists(self.latest_local_map):
            print(f"  ⚠️  Local Map not found: {self.latest_local_map}")
            self.latest_local_map = None
        
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
        last_step = self.current_step - 1
        map_path = os.path.join(self.episode_dir, 'global_map', f'step_{last_step:04d}.png')
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
        
        # 直接从episode的panoramas/目录读取
        panorama_dir = os.path.join(self.episode_dir, 'panoramas')
        
        # 获取4个全景图
        for config in PANORAMA_CONFIG:
            direction_name = config["name"]
            panorama_filename = f"{phase}_panorama_{direction_name.split()[0].lower()}.png"
            panorama_path = os.path.join(panorama_dir, panorama_filename)
            
            if os.path.exists(panorama_path):
                panorama_paths.append(panorama_path)
                direction_names.append(direction_name)
            else:
                print(f"  ⚠️  {direction_name} 未找到: {panorama_filename}")
        
        # 获取地图（使用当前step的地图，每次环视后current_step已更新）
        # current_step是最后一次环视后的step，地图文件名需要加上phase后缀
        global_map_path = os.path.join(self.episode_dir, 'global_map', f'step_{self.current_step:04d}_{phase}.png')
        local_map_path = os.path.join(self.episode_dir, 'local_map', f'step_{self.current_step:04d}_{phase}.png')
        
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
        
        # 地图已包含waypoint标记（在visualizer.save_step_visualization中渲染）
        global_map_for_llm = global_map
        
        # 调用LLM生成初始子任务
        response, prompt = self.planner.generate_initial_subtask(
            instruction=self.current_instruction,
            observation_images=image_paths,
            direction_names=direction_names,
            global_map_image=global_map_for_llm,  # 使用带waypoint标注的版本
            local_map_image=local_map
        )
        
        if not response:
            print("✗ LLM未返回有效响应")
            return None
        
        # 记录thinking输出（包含输入图片和prompt）
        # 此时current_step=12（环视完成），规划在step 12之后完成
        # 初始规划的subtask_id为 "1a" (第1个子任务，第a次尝试)
        thinking_record = {
            "step": self.current_step,  # 12
            "phase": "initial_planning",
            "subtask_count": 1,  # 初始化总是第1个子任务
            "subtask_attempt": 0,  # 初始规划总是a
            "subtask_id": "1a",  # 初始化总是1a
            "prompt_type": "initial",
            "prompt": prompt,  # 保存prompt
            "response": response,
            "timestamp": datetime.now().isoformat(),
            # 保存输入图片路径（与PANORAMA_CONFIG顺序一致：Front, Left, Back, Right）
            "input_images": {
                "global_map.png": global_map_for_llm,
                "local_map.png": local_map,
                "IMAGE 1 - Front (0°).png": image_paths[0] if len(image_paths) > 0 else None,
                "IMAGE 2 - Left (90°).png": image_paths[1] if len(image_paths) > 1 else None,
                "IMAGE 3 - Back (180°).png": image_paths[2] if len(image_paths) > 2 else None,
                "IMAGE 4 - Right (270°).png": image_paths[3] if len(image_paths) > 3 else None,
            }
        }
        self.thinking_outputs.append(thinking_record)
        self.save_manager.save_thinking(thinking_record)
        
        # 保存子任务并初始化计数
        self.current_subtask = response
        self.subtask_count = 1  # 初始化为第1个子任务
        self.subtask_attempt = 0  # 第a次尝试
        self.progress_summary = ""
        
        # 记录当前位置信息（用于后续验证参考）
        self.current_position_info = {
            'waypoint': response.get('waypoint', 'Unknown'),
            'observation': response.get('current_observation', ''),
            'step': self.current_step
        }
        
        # 在mapper中添加waypoint（自动计算地图坐标）
        waypoint_desc = response.get('waypoint', 'Unknown location')
        waypoint_id = self.mapper.add_waypoint(waypoint_desc)
        
        # 保存waypoint摘要（用于后续LLM提示词）
        waypoint_summary = self._get_waypoint_summary()
        self.save_manager.save_waypoint_memory(
            waypoint_summary,
            self.current_instruction,
            self.current_step
        )
        
        # 记录并动态更新目标landmark（从instruction中提取所有相关landmark）
        subtask_instruction = response.get('subtask_instruction', '')
        subtask_landmark = response.get('subtask_landmark', None)
        
        # 从instruction中提取所有mapping_classes中的类别作为landmark
        landmarks_in_instruction = []
        for cls in self.mapping_classes:
            if cls.lower() in subtask_instruction.lower():
                landmarks_in_instruction.append(cls)
        
        # 如果有明确的subtask_landmark，优先使用；否则使用提取的所有landmarks
        if subtask_landmark and subtask_landmark in self.mapping_classes:
            self.landmark_classes = [subtask_landmark]
            self.target_landmark = subtask_landmark
            print(f"  🎯 Target Landmark: {self.target_landmark}")
        elif landmarks_in_instruction:
            self.landmark_classes = landmarks_in_instruction
            self.target_landmark = landmarks_in_instruction[0]  # 主要目标
            print(f"  🎯 Target Landmarks: {', '.join(self.landmark_classes)}")
        else:
            self.target_landmark = None
            self.landmark_classes = []
            print(f"  ℹ️  No target landmark")
        
        # ⚠️ 重要：self.classes始终保持为所有mapping_classes，用于完整的语义建图
        # landmark_classes只用于可视化标注和导航决策
        
        # 打印子任务信息
        self._print_subtask_info(response, is_initial=True)
        
        return response
    
    def verify_and_replan(self) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        验证当前子任务并重新规划
        
        流程：
        1. 执行360°环视建图（更新语义地图）- 占用12个step
        2. 生成当前位置的4方向全景图
        3. 调用LLM验证子任务完成状态
        4. 如未完成，生成新子任务
        
        注意：重新扫描会占用新的12个step，验证完成后下一个action继续累加
        
        Returns:
            (is_completed, new_subtask, prompt)
        """
        if not self.planner or not self.current_subtask:
            return False, None, None
        
        # 重新执行环视建图并生成全景图（占用12个step）
        # 注意：如果子任务已完成，会在后面清空轨迹；如果未完成，轨迹继续累积
        # 使用attempt字母标识（a=0, b=1, c=2...）
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        phase = f"verify_{self.subtask_count}{attempt_letter}"
        print(f"\n[验证] 子任务#{self.subtask_count}{attempt_letter} - 重新环视（step {self.current_step + 1}-{self.current_step + 12}）...")
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
        
        # 地图已包含waypoint标记（在visualizer.save_step_visualization中渲染）
        global_map_for_llm = global_map
        
        # 获取已检测到的landmark类别 - 汇总环视12步中检测到的所有landmarks
        detected_landmarks = []
        if hasattr(self, 'current_step_landmarks') and self.current_step_landmarks:
            # 汇总12步环视中所有检测到的landmarks
            all_landmarks = set()
            for step_idx, landmarks_list in self.current_step_landmarks.items():
                for name, conf in landmarks_list:
                    all_landmarks.add(name)
            detected_landmarks = sorted(list(all_landmarks))
        else:
            # 退化使用全局detected_classes
            detected_landmarks = sorted(list(self.detected_classes)) if hasattr(self, 'detected_classes') else []
        
        # 获取waypoint摘要
        waypoint_summary = self._get_waypoint_summary()
        
        # 调用LLM验证（全局地图必需，局部地图可选，传递实际检测到的类别）
        response, is_completed, prompt = self.planner.verify_and_replan(
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
            return False, None, None
        
        # 记录thinking输出（包含输入图片和prompt）
        # 此时current_step已经是验证扫描完成后的step（+12）
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        subtask_id = f"{self.subtask_count}{attempt_letter}"  # 当前验证的子任务，如 "1a"
        
        # 计算下一个subtask_id
        if is_completed and not response.get('is_final_subtask', False):
            next_subtask_count = self.subtask_count + 1
            next_attempt = 0
        else:
            next_subtask_count = self.subtask_count
            next_attempt = self.subtask_attempt + 1 if not is_completed else 0
        next_attempt_letter = chr(ord('a') + next_attempt)
        
        thinking_record = {
            "step": self.current_step,  # 验证扫描完成后的step
            "phase": f"verify_{subtask_id}",  # verify_1a, verify_2b, etc.
            "subtask_count": self.subtask_count,
            "subtask_attempt": self.subtask_attempt,
            "subtask_id": subtask_id,  # 当前验证的子任务，如 "1a"
            "next_subtask_id": f"{next_subtask_count}{next_attempt_letter}" if not response.get('is_final_subtask', False) else "final",
            "prompt_type": "verification",
            "is_completed": is_completed,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "detected_landmarks": detected_landmarks,  # 记录传递给LLM的landmarks
            # 保存输入图片路径（与PANORAMA_CONFIG顺序一致：Front, Left, Back, Right）
            "input_images": {
                "global_map.png": global_map_for_llm,
                "local_map.png": local_map if os.path.exists(local_map) else None,
                "IMAGE 1 - Front (0°).png": image_paths[0] if len(image_paths) > 0 else None,
                "IMAGE 2 - Left (90°).png": image_paths[1] if len(image_paths) > 1 else None,
                "IMAGE 3 - Back (180°).png": image_paths[2] if len(image_paths) > 2 else None,
                "IMAGE 4 - Right (270°).png": image_paths[3] if len(image_paths) > 3 else None,
            },
            "prompt": prompt,
        }
        self.thinking_outputs.append(thinking_record)
        self.save_manager.save_thinking(thinking_record)
        
        # 注意：轨迹和landmark已经在look_around_and_collect开始时清空
        # 这里只需要重置progress_summary即可
        self.progress_summary = ""  # 重置进度摘要
        
        if is_completed:
            attempt_letter = chr(ord('a') + self.subtask_attempt)
            print(f"\n[子任务完成] #{self.subtask_count}{attempt_letter}")
            
            # 检查是否是最终子任务
            if response.get('is_final_subtask', False):
                print("到达最终目的地")
                return True, response, prompt
            
            # 更新到新子任务：递增计数，重置尝试
            self.subtask_count += 1
            self.subtask_attempt = 0  # 新子任务从a开始
            self.current_subtask = response
            # progress_summary已在上面清空
            
            # 更新当前位置信息（用于后续参考）
            self.current_position_info = {
                'waypoint': response.get('waypoint', 'Unknown'),
                'observation': response.get('current_observation', ''),
                'step': self.current_step
            }
            
            # 创建路径点记录（空间记忆）
            waypoint_desc = response.get('waypoint', 'Unknown location')
            waypoint_id = self.mapper.add_waypoint(waypoint_desc)
            
            # 动态更新目标landmark（从instruction中提取所有相关landmark）
            subtask_instruction = response.get('subtask_instruction', '')
            subtask_landmark = response.get('subtask_landmark', None)
            
            # 从instruction中提取所有mapping_classes中的类别作为landmark
            landmarks_in_instruction = []
            for cls in self.mapping_classes:
                if cls.lower() in subtask_instruction.lower():
                    landmarks_in_instruction.append(cls)
            
            # 如果有明确的subtask_landmark，优先使用；否则使用提取的所有landmarks
            if subtask_landmark and subtask_landmark in self.mapping_classes:
                self.landmark_classes = [subtask_landmark]
                self.target_landmark = subtask_landmark
                print(f"  🎯 New Target Landmark: {self.target_landmark}")
            elif landmarks_in_instruction:
                self.landmark_classes = landmarks_in_instruction
                self.target_landmark = landmarks_in_instruction[0]  # 主要目标
                print(f"  🎯 New Target Landmarks: {', '.join(self.landmark_classes)}")
            else:
                self.target_landmark = None
                self.landmark_classes = []
                print(f"  ℹ️  No target landmark")
            
            # ⚠️ 重要：self.classes始终保持为所有mapping_classes，用于完整的语义建图
            
            self._print_subtask_info(response)
        else:
            attempt_letter = chr(ord('a') + self.subtask_attempt)
            print(f"\n[子任务未完成] #{self.subtask_count}{attempt_letter} - 重新规划")
            
            # 未完成时保持subtask_count不变，递增attempt
            self.subtask_attempt += 1  # 下次验证用b, c, d...
            self.current_subtask = response
            # progress_summary和landmark_classes已在上面清空
            
            # 更新位置观察（用于记录轨迹）
            if 'current_observation' in response:
                self.current_position_info = {
                    'waypoint': response.get('waypoint', getattr(self, 'current_position_info', {}).get('waypoint', 'Unknown')),
                    'observation': response.get('current_observation', ''),
                    'step': self.current_step
                }
            
            # 从新的子任务指令中提取landmark（与完成时逻辑相同）
            subtask_instruction = response.get('subtask_instruction', '')
            subtask_landmark = response.get('subtask_landmark', None)
            
            landmarks_in_instruction = []
            for cls in self.mapping_classes:
                if cls.lower() in subtask_instruction.lower():
                    landmarks_in_instruction.append(cls)
            
            if subtask_landmark and subtask_landmark in self.mapping_classes:
                self.landmark_classes = [subtask_landmark]
                self.target_landmark = subtask_landmark
                print(f"  🎯 Updated Target Landmark: {self.target_landmark}")
            elif landmarks_in_instruction:
                self.landmark_classes = landmarks_in_instruction
                self.target_landmark = landmarks_in_instruction[0]
                print(f"  🎯 Updated Target Landmarks: {', '.join(self.landmark_classes)}")
            else:
                self.target_landmark = None
                self.landmark_classes = []
                print(f"  ℹ️  No target landmark")
            
            # ⚠️ 重要：self.classes始终保持为所有mapping_classes，用于完整的语义建图
            
            # 输出LLM验证结果和调整后的子任务
            self._print_subtask_info(response, is_initial=False)
        
        return is_completed, response, prompt
    
    def execute_action_with_vlm(self) -> Tuple[Optional[int], Optional[str], bool, int]:
        """
        使用VLM决策并执行动作
        
        Returns:
            (action_id, action_name, should_stop, repeat_count)
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
        
        # 获取最新保存的观察信息
        # 上一步已保存的文件（如果current_step=13，则读取step_0012的地图）
        last_step = self.current_step  # execute_action在step执行前调用，所以用current_step
        
        # 生成当前子任务的phase标识
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        action_phase = f"action{self.subtask_count}{attempt_letter}"
        
        # 智能查找可用的图像：优先使用action phase，回退到verify/initial
        # 可能的phase顺序: action1a -> verify_1a -> initial (注意verify带下划线)
        possible_phases = [action_phase]
        if self.subtask_attempt > 0:
            # 如果是1b, 1c等，可能需要回退到verify_1a
            verify_phase = f"verify_{self.subtask_count}{chr(ord('a') + self.subtask_attempt - 1)}"
            possible_phases.append(verify_phase)
        elif self.subtask_count > 1:
            # 如果是2a, 3a等，回退到上一个verify
            prev_verify_phase = f"verify_{self.subtask_count - 1}a"
            possible_phases.append(prev_verify_phase)
        else:
            # 如果是1a，回退到initial
            possible_phases.append("initial")
        
        # 查找RGB图像
        fp_image = None
        for phase in possible_phases:
            candidate = os.path.join(self.episode_dir, 'rgb', f'step_{last_step:04d}_{phase}.png')
            if os.path.exists(candidate):
                fp_image = candidate
                break
        
        # 如果都不存在，用当前观察创建临时文件
        if not fp_image:
            rgb_bgr = cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR)
            temp_image = os.path.join(self.episode_dir, f'temp_fp_step{last_step}.png')
            cv2.imwrite(temp_image, rgb_bgr)
            fp_image = temp_image
        
        # 获取当前地图路径和检测图像
        self._get_current_map_path()
        
        # 查找detection图像（使用相同的回退逻辑）
        detection_image = None
        detection_step = None  # 记录找到的detection图像对应的step
        for phase in possible_phases:
            candidate = os.path.join(self.episode_dir, 'detection', f'step_{last_step:04d}_{phase}.png')
            if os.path.exists(candidate):
                detection_image = candidate
                detection_step = last_step
                break
        if not detection_image:
            print(f"  ⚠️  Detection image not found for step {last_step} (tried phases: {possible_phases})")
        
        # 查找局部地图（使用相同的回退逻辑）
        local_map = None
        for phase in possible_phases:
            candidate = os.path.join(self.episode_dir, 'local_map', f'step_{last_step:04d}_{phase}.png')
            if os.path.exists(candidate):
                local_map = candidate
                break
        if not local_map:
            print(f"  ⚠️  Local map not found for step {last_step} (tried phases: {possible_phases})")
        
        # 获取detection图像对应的landmark类别
        # 使用找到的detection图像对应的step
        detected_landmarks = None
        if detection_step is not None and hasattr(self, 'current_step_landmarks') and detection_step in self.current_step_landmarks:
            # 当前step检测到的landmarks: [(name, confidence), ...]
            step_landmarks = self.current_step_landmarks[detection_step]
            if step_landmarks:
                # 格式化为 "name1 (conf1), name2 (conf2)"
                detected_landmarks = ', '.join([f"{name} ({conf:.2f})" for name, conf in step_landmarks])
        
        # 退化策略：如果没有检测结果，报告"未检测到"
        if not detected_landmarks:
            if hasattr(self, 'target_landmark') and self.target_landmark:
                detected_landmarks = f"No {self.target_landmark} detected in current view"
            else:
                detected_landmarks = "No landmarks detected"
        
        # 调用VLM决策
        result = self.action_executor.decide_action(
            subtask_destination=self.current_subtask.get('subtask_destination', ''),
            subtask_instruction=self.current_subtask.get('subtask_instruction', ''),
            first_person_image=fp_image,
            action_mapping=ACTION_MAPPING,
            progress_summary=self.progress_summary,
            detection_image=detection_image,
            local_map_image=local_map,
            detected_landmarks=detected_landmarks
        )
        
        if len(result) == 7:
            action_id, action_name, updated_progress, response, degrees, meters, prompt = result
        elif len(result) == 6:
            action_id, action_name, updated_progress, response, degrees, meters = result
            prompt = None
        else:
            # 兼容旧版本返回（没有degrees/meters）
            action_id, action_name, updated_progress, response = result
            degrees, meters = 0, 0
            prompt = None
        
        if action_id is None:
            print("✗ VLM决策失败")
            return None, None, True, 1
        
        # 记录action输出（包含输入图片和prompt）
        # step记录即将执行的action的步数（例如：当前step=12，下一个action将在step 13执行）
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        action_record = {
            "step": self.current_step + 1,  # 即将执行的action的step
            "subtask_count": self.subtask_count,
            "subtask_attempt": self.subtask_attempt,
            "subtask_id": f"{self.subtask_count}{attempt_letter}",  # 如 "1a"
            "action_name": action_name,
            "action_id": action_id,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "detected_landmarks": detected_landmarks,  # 记录传递给VLM的landmarks
            # 保存输入图片路径（action模块只使用3张图：RGB + Detection + Local Map）
            "input_images": {
                "rgb.png": fp_image,
                "detection.png": detection_image,
                "local_map.png": local_map,
            },
            "prompt": prompt,
        }
        self.action_outputs.append(action_record)
        
        # 保存action记录，同时保存子任务信息
        subtask_info = {
            "subtask_id": self.subtask_count,
            "subtask_destination": self.current_subtask.get('subtask_destination', ''),
            "subtask_instruction": self.current_subtask.get('subtask_instruction', ''),
            "start_step": self.current_step,  # 子任务开始时的step（决策前）
            "timestamp": datetime.now().isoformat()
        }
        self.save_manager.save_action(action_record, subtask_info)
        
        # 更新进度
        self.progress_summary = updated_progress
        
        # 检查是否停止
        should_stop = (action_name == "STOP")
        
        # 计算需要重复执行的次数
        repeat_count = 1
        if action_name == 'TURN_LEFT' or action_name == 'TURN_RIGHT':
            # 每次转30度，计算需要转几次
            if degrees > 0:
                repeat_count = max(1, round(degrees / self.action_executor.turn_angle))
        elif action_name == 'MOVE_FORWARD':
            # 每次移动0.25m，计算需要移动几次
            if meters > 0:
                repeat_count = max(1, round(meters / self.action_executor.move_distance))
        
        return action_id, action_name, should_stop, repeat_count
    
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
        # 生成phase标识: action1a, action2b等
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        phase = f"action{self.subtask_count}{attempt_letter}"
        
        result = self.step(action, save_vis, phase)
        # 缓存最新观察和info用于下次VLM决策和可视化
        self.latest_obs = result.get('obs', None)
        self.latest_info = result.get('info', None)
        
        # 保存RGB+俯视图拼接可视化
        if save_vis and self.nav_visualizer and self.latest_obs is not None:
            subtask_text = None
            if self.current_subtask:
                subtask_text = self.current_subtask.get('subtask_instruction', '')
            
            distance = 0.0
            if self.latest_info:
                distance = self.latest_info.get('distance_to_goal', 0.0)
            
            attempt_letter = chr(ord('a') + self.subtask_attempt)
            subtask_id = f"{self.subtask_count}{attempt_letter}"
            
            self.nav_visualizer.save_step_visualization(
                observations=self.latest_obs,
                info=self.latest_info or {},
                step=self.current_step,
                instruction=self.current_instruction,
                current_subtask=subtask_text,
                distance=distance,
                action=action_name,
                subtask_id=subtask_id
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
        
        # 1. 环视建图 + 收集观察（占用step 1-12）
        self.look_around_and_collect()
        
        # 2. 生成初始子任务（在step 12完成，下一个action从step 13开始）
        subtask = self.generate_initial_subtask()
        if not subtask:
            print("✗ 初始子任务生成失败")
            return {
                'success': False,
                'total_steps': self.current_step,  # 12
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
            action_id, action_name, should_stop, repeat_count = self.execute_action_with_vlm()
            
            if action_id is None:
                print("VLM决策失败，尝试手动输入")
                action_id = self.get_keyboard_action()
                action_name = self._action_name(action_id)
                should_stop = (action_id == 0)
                repeat_count = 1
            
            # 如果VLM决定停止 → 验证子任务
            if should_stop:
                is_completed, new_subtask, _ = self.verify_and_replan()
                
                if is_completed and new_subtask and new_subtask.get('is_final_subtask', False):
                    print("\n[导航完成]")
                    navigation_complete = True
                    break
                
                subtask_steps = 0
                continue
            
            # 执行动作（可能需要重复多次）
            for i in range(repeat_count):
                result = self.step_with_vlm(action_id, action_name=action_name, save_vis=True)
                total_steps = self.current_step
                subtask_steps += 1
                
                if i == 0 and repeat_count > 1:
                    print(f"[Step {total_steps}] {action_name} (1/{repeat_count}) | 子任务步数: {subtask_steps}")
                elif repeat_count > 1:
                    print(f"[Step {total_steps}] {action_name} ({i+1}/{repeat_count}) | 子任务步数: {subtask_steps}")
                else:
                    print(f"[Step {total_steps}] {action_name} | 子任务步数: {subtask_steps}")
                
                if result['done']:
                    print("\nEpisode自动完成")
                    navigation_complete = True
                    break
            
            if navigation_complete:
                break
            
            if result['done']:
                print("\nEpisode自动完成")
                navigation_complete = True
                break
            
            # 定期验证
            if subtask_steps >= verify_interval:
                is_completed, _, _ = self.verify_and_replan()
                if is_completed:
                    subtask_steps = 0
            
            # 子任务超时
            if subtask_steps >= max_subtask_steps:
                print(f"\n[警告] 子任务超时 ({max_subtask_steps}步)，重新规划")
                _, _, _ = self.verify_and_replan()
                subtask_steps = 0
        
        # 4. 生成GIF动画
        gif_path = None
        if self.nav_visualizer:
            gif_path = self.nav_visualizer.save_gif(fps=2)
            print(f"\n🎬 GIF动画: {gif_path if gif_path else '未生成'}")
        
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
    
    def _save_thinking_output(self, thinking_record: Dict):
        """保存LLM思考输出（调用save_manager）"""
        self.save_manager.save_thinking(thinking_record)
    
    def _save_action_output(self, action_record: Dict):
        """保存VLM动作输出（调用save_manager）"""
        self.save_manager.save_action(action_record)
    
    def _save_navigation_result(self, success: bool, total_steps: int) -> str:
        """保存导航结果"""
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
        
        return self.save_manager.save_result(result)
        
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
        """打印子任务信息（JSON格式）"""
        import json
        
        # 根据响应类型确定标题
        attempt_letter = chr(ord('a') + self.subtask_attempt)
        if is_initial:
            title = f"Initial Subtask #{self.subtask_count}{attempt_letter}"
        elif 'is_completed' in response:
            # 验证响应
            if response.get('is_completed', False):
                title = f"Subtask #{self.subtask_count}{attempt_letter} - Completed ✓"
            else:
                title = f"Subtask #{self.subtask_count}{attempt_letter} - Continue (Not Completed)"
        else:
            title = f"Subtask #{self.subtask_count}{attempt_letter}"
        
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"Global Instruction: {self.current_instruction}")
        print(f"{'-'*60}")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print(f"{'='*60}\n")
    
    # ========== Waypoint辅助方法 ==========
    
    def _get_waypoint_summary(self) -> str:
        """获取waypoint摘要（用于LLM提示词）"""
        wp_pos, wp_ids, wp_descs = self.mapper.get_waypoints()
        if len(wp_ids) == 0:
            return ""
        
        # 根据waypoint ID和描述生成摘要
        summary_lines = []
        for wp_id, wp_desc in zip(wp_ids, wp_descs):
            summary_lines.append(f"#{wp_id}: {wp_desc}")
        
        return "\n".join(summary_lines)
    
    # ========== 原有方法 ==========

