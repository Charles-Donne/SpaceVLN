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
from vlnce_baselines.vlm import LLMPlanner, ActionExecutor, SaveManager, WaypointManager, NavigationVisualizer
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
        self.current_subtask_file = None
        
        # 初始化管理器
        self.save_manager = None  # 在reset_episode时初始化
        self.waypoint_manager = WaypointManager()
        
        # 观察缓存
        self.latest_obs = None  # 缓存最新的观察
        self.latest_info = None  # 缓存最新的info（包含top_down_map_vlnce）
        
        # 观察缓存（环视时收集的4方向图像）
        self.direction_images = {}  # {direction_name: image_path}
        self.latest_map_image = None
        
        # NavigationVisualizer（用于RGB+俯视图拼接和GIF生成）
        self.nav_visualizer = None
        
        print("[Init] VLM模块初始化完成\n")
    
    def reset_episode(self, episode_id: int = None):
        """重置Episode，包括VLM状态"""
        # 调用父类重置
        super().reset_episode(episode_id)
        
        # 初始化SaveManager（使用RESULTS_DIR作为输出根目录）
        self.save_manager = SaveManager(self.config.RESULTS_DIR, self.current_episode_id)
        
        # 重置VLM状态
        self.current_subtask = None
        self.subtask_count = 0
        self.progress_summary = ""
        self.subtask_history = []
        self.current_subtask_file = None
        self.direction_images = {}
        self.latest_map_image = None
        
        # 重置waypoint管理器
        self.waypoint_manager.reset()
        
        print(f"[Reset] Episode {self.current_episode_id} 重置完成")
        
        # 初始化NavigationVisualizer（用于RGB+俯视图拼接和GIF生成）
        episode_dir = os.path.join(self.config.RESULTS_DIR, f'episode_{self.current_episode_id}')
        visualization_dir = os.path.join(episode_dir, 'visualization')
        self.nav_visualizer = NavigationVisualizer(visualization_dir)
        self.nav_visualizer.setup_maps_dir(episode_dir)
        
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
        
        # 保存轨迹，环视结束后恢复（但不恢复current_step，环视步骤计入总步数）
        saved_trajectory = self.mapper.trajectory_points.copy() if hasattr(self.mapper, 'trajectory_points') else []
        
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
            obs, _, dones, _ = [list(x) for x in zip(*outputs)]
            
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
            rgb_bgr = cv2.cvtColor(obs[0]['rgb'], cv2.COLOR_RGB2BGR)
            paths, landmarks = self.visualizer.save_step_visualization(
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
                }
            )
            
            if new_classes > 0:
                print(f" +{new_classes}类")
            else:
                print()
            
            # 保存所有12张环视图像（用于后续合成全景图）
            rgb_bgr = cv2.cvtColor(obs[0]['rgb'], cv2.COLOR_RGB2BGR)
            lookaround_images.append(rgb_bgr.copy())
        
        # 环视建图完成，恢复轨迹（current_step保持为12，后续导航从13开始）
        self.mapper.trajectory_points = saved_trajectory
        self.current_step = 12  # 环视完成后停在step-12
        
        # 缓存最后的观察（step 12，回到正前方）
        self.latest_obs = obs[0]
        
        # 缓存最后的观察（最后一次旋转后）
        self.latest_obs = obs[0]
        
        print(f"  扫描完成: +{total_new_classes}类 | 总计{len(self.detected_classes)}类")
        
        # 使用固定的水平视场角30°（每次TURN_LEFT旋转30°）
        hfov = 30.0  # 每张图的水平视场角（度）
        
        # 步骤1: 将12张图像投影到完整的360°柱面全景图
        full_panorama = self._create_full_panorama(lookaround_images, hfov)
        
        # 步骤2: 从完整全景图中裁剪出4个方向的90°视图
        panorama_paths = []
        panorama_names = []
        panorama_dir = os.path.join(self.config.RESULTS_DIR, f"episode_{self.current_episode_id}", "panoramas")
        os.makedirs(panorama_dir, exist_ok=True)
        
        for config in PANORAMA_CONFIG:
            direction_name = config["name"]
            center_angle = config.get("center_angle", 0)  # 方向中心角度
            
            # 从完整全景图中裁剪90°视图
            panorama = self._crop_panorama_view(full_panorama, center_angle, 90.0, hfov)
            
            # 保存全景图
            panorama_filename = f"{phase}_panorama_{direction_name.split()[0].lower()}.jpg"
            panorama_path = os.path.join(panorama_dir, panorama_filename)
            cv2.imwrite(panorama_path, panorama)
            
            panorama_paths.append(panorama_path)
            panorama_names.append(direction_name)
            self.direction_images[direction_name] = panorama_path
        
        # 保存全局地图和局部地图到 vlm/observations/
        # 直接使用episode目录下的地图（step-12是完成360°扫描后最完整的地图）
        episode_dir = os.path.join(self.config.RESULTS_DIR, f'episode_{self.current_episode_id}')
        
        self.latest_global_map = os.path.join(episode_dir, 'global_map', f'step-12.png')
        self.latest_local_map = os.path.join(episode_dir, 'local_map', f'step-12.png')
        
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
        
        # 直接从episode的panoramas/目录读取
        episode_dir = os.path.join(self.config.RESULTS_DIR, f'episode_{self.current_episode_id}')
        panorama_dir = os.path.join(episode_dir, 'panoramas')
        
        # 获取4个全景图
        for config in PANORAMA_CONFIG:
            direction_name = config["name"]
            panorama_filename = f"{phase}_panorama_{direction_name.split()[0].lower()}.jpg"
            panorama_path = os.path.join(panorama_dir, panorama_filename)
            
            if os.path.exists(panorama_path):
                panorama_paths.append(panorama_path)
                direction_names.append(direction_name)
            else:
                print(f"  ⚠️  {direction_name} 未找到: {panorama_filename}")
        
        # 获取地图（直接使用episode目录下的step-12地图）
        global_map_path = os.path.join(episode_dir, 'global_map', 'step-12.png')
        local_map_path = os.path.join(episode_dir, 'local_map', 'step-12.png')
        
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
        
        # 创建带waypoint标注的地图副本
        global_map_for_llm = global_map
        if self.waypoint_manager.get_count() > 0:
            global_map_img = cv2.imread(global_map)
            global_map_img = self.waypoint_manager.visualize_on_map(
                global_map_img,
                self.mapper.trajectory_points if hasattr(self.mapper, 'trajectory_points') else []
            )
            # 保存带waypoint的地图到global_map目录
            episode_dir = os.path.join(self.config.RESULTS_DIR, f"episode_{self.current_episode_id}")
            global_map_for_llm = os.path.join(episode_dir, 'global_map', 'initial_with_waypoints.png')
            cv2.imwrite(global_map_for_llm, global_map_img)
        
        # 调用LLM生成初始子任务
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
        
        # 记录thinking输出（包含输入图片和prompt）
        thinking_record = {
            "step": self.current_step,
            "phase": "initial_planning",
            "subtask_count": self.subtask_count + 1,
            "prompt_type": "initial",
            "response": response,
            "timestamp": datetime.now().isoformat(),
            # 保存输入图片路径
            "input_images": {
                "global_map.png": global_map_for_llm,
                "local_map.png": local_map,
                "front.jpg": image_paths[0] if len(image_paths) > 0 else None,
                "left.jpg": image_paths[1] if len(image_paths) > 1 else None,
                "back.jpg": image_paths[2] if len(image_paths) > 2 else None,
                "right.jpg": image_paths[3] if len(image_paths) > 3 else None,
            },
            # 保存prompt（后续从planner获取）
            "prompt": f"Instruction: {self.current_instruction}\nDirection Names: {direction_names}"
        }
        self.thinking_outputs.append(thinking_record)
        self.save_manager.save_thinking(thinking_record)
        
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
        self.waypoint_manager.add_waypoint(waypoint_desc)
        self.save_manager.save_waypoint_memory(
            self.waypoint_manager.get_memory(),
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
            print(f"  🎯 Landmark: {self.target_landmark}")
        elif landmarks_in_instruction:
            self.landmark_classes = landmarks_in_instruction
            self.target_landmark = landmarks_in_instruction[0]  # 主要目标
            print(f"  🎯 Landmarks from instruction: {', '.join(self.landmark_classes)}")
        else:
            self.target_landmark = None
            self.landmark_classes = []
            print(f"  ℹ️  No landmarks to mark")
        
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
        
        # 创建带waypoint标注的地图副本
        global_map_for_llm = global_map
        if self.waypoint_manager.get_count() > 0:
            global_map_img = cv2.imread(global_map)
            global_map_img = self.waypoint_manager.visualize_on_map(
                global_map_img,
                self.mapper.trajectory_points if hasattr(self.mapper, 'trajectory_points') else []
            )
            # 保存带waypoint的地图
            global_map_for_llm = os.path.join(episode_dir, 'global_map', f'verify_{last_saved_step}_with_waypoints.png')
            cv2.imwrite(global_map_for_llm, global_map_img)
        
        # 获取已检测到的landmark类别
        detected_landmarks = list(self.detected_classes) if hasattr(self, 'detected_classes') else []
        
        # 获取路径点历史记录
        waypoint_summary = self.waypoint_manager.get_summary()
        
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
        
        # 记录thinking输出（包含输入图片和prompt）
        thinking_record = {
            "step": self.current_step,
            "phase": f"verify_subtask_{self.subtask_count}",
            "subtask_count": self.subtask_count + (1 if is_completed and not response.get('is_final_subtask', False) else 0),
            "prompt_type": "verification",
            "is_completed": is_completed,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            # 保存输入图片路径
            "input_images": {
                "global_map.png": global_map_for_llm,
                "local_map.png": local_map if os.path.exists(local_map) else None,
                "front.jpg": image_paths[0] if len(image_paths) > 0 else None,
                "left.jpg": image_paths[1] if len(image_paths) > 1 else None,
                "back.jpg": image_paths[2] if len(image_paths) > 2 else None,
                "right.jpg": image_paths[3] if len(image_paths) > 3 else None,
            },
            # 保存prompt关键信息
            "prompt": f"Instruction: {self.current_instruction}\nCurrent Subtask: {self.current_subtask.get('subtask_instruction', '')}\nDetected Landmarks: {detected_landmarks}\nWaypoint Summary: {waypoint_summary}"
        }
        self.thinking_outputs.append(thinking_record)
        self.save_manager.save_thinking(thinking_record)
        
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
                print(f"  🎯 New Landmark: {self.target_landmark}")
            elif landmarks_in_instruction:
                self.landmark_classes = landmarks_in_instruction
                self.target_landmark = landmarks_in_instruction[0]  # 主要目标
                print(f"  🎯 New Landmarks from instruction: {', '.join(self.landmark_classes)}")
            else:
                self.target_landmark = None
                self.landmark_classes = []
                print(f"  ℹ️  No landmarks to mark")
            
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
                episode_dir,
                f'temp_fp_step{last_step}.png'
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
        
        # 记录action输出（包含输入图片和prompt）
        action_record = {
            "step": self.current_step,
            "subtask_count": self.subtask_count,
            "action_name": action_name,
            "action_id": action_id,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            # 保存输入图片路径
            "input_images": {
                "rgb.jpg": fp_image,
                "detection.jpg": detection_image,
                "local_map.png": local_map,
                "global_map.png": self.latest_global_map if self.latest_global_map and os.path.exists(self.latest_global_map) else None,
            },
            # 保存prompt关键信息
            "prompt": f"Subtask: {self.current_subtask.get('subtask_instruction', '')}\nProgress: {self.progress_summary}\nDetected: {detected_landmarks}"
        }
        self.action_outputs.append(action_record)
        
        # 保存action记录，同时保存子任务信息
        subtask_info = {
            "subtask_id": self.subtask_count,
            "subtask_destination": self.current_subtask.get('subtask_destination', ''),
            "subtask_instruction": self.current_subtask.get('subtask_instruction', ''),
            "start_step": self.current_step,  # 记录子任务开始的步数
            "timestamp": datetime.now().isoformat()
        }
        self.save_manager.save_action(action_record, subtask_info)
        
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
        
        # 保存RGB+俯视图拼接可视化
        if save_vis and self.nav_visualizer and self.latest_obs is not None:
            subtask_text = None
            if self.current_subtask:
                subtask_text = self.current_subtask.get('subtask_instruction', '')
            
            distance = 0.0
            if self.latest_info:
                distance = self.latest_info.get('distance_to_goal', 0.0)
            
            self.nav_visualizer.save_step_visualization(
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
        title = "Initial Subtask" if is_initial else f"Subtask #{self.subtask_count}"
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"Global Instruction: {self.current_instruction}")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print(f"{'='*50}\n")
    # ========== 向后兼容的方法（调用manager） ==========
    
    def add_waypoint(self, waypoint_description: str, position: np.ndarray = None) -> int:
        """添加waypoint（调用waypoint_manager）"""
        waypoint_id = self.waypoint_manager.add_waypoint(waypoint_description)
        self.save_manager.save_waypoint_memory(
            self.waypoint_manager.get_memory(),
            self.current_instruction,
            self.current_step
        )
        return waypoint_id
    
    def get_waypoint_summary(self) -> str:
        """获取waypoint摘要（调用waypoint_manager）"""
        return self.waypoint_manager.get_summary()
    
    def visualize_waypoints_on_map(self, map_image: np.ndarray) -> np.ndarray:
        """在地图上可视化waypoint（调用waypoint_manager）"""
        return self.waypoint_manager.visualize_on_map(
            map_image,
            self.mapper.trajectory_points if hasattr(self.mapper, 'trajectory_points') else []
        )
    
    def _create_full_panorama(self, images: List[np.ndarray], hfov: float) -> np.ndarray:
        """
        使用OpenCV Stitcher拼接12张图像生成360°全景图
        
        Args:
            images: 12张环视图像
            hfov: 单张图像水平视场角（度，未使用但保留接口兼容性）
            
        Returns:
            360°全景图
        """
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, panorama = stitcher.stitch(images)
        return panorama
    
    
    def _crop_panorama_view(self, full_panorama: np.ndarray, center_angle: float, 
                            fov: float, hfov_per_image: float) -> np.ndarray:
        """
        从360°全景图中裁剪指定方向的视图
        
        Args:
            full_panorama: 360°完整全景图
            center_angle: 视图中心角度（度，0°=前方，90°=左侧）
            fov: 视图的水平视场角（度，通常90°）
            hfov_per_image: 单张原始图像的视场角（用于计算焦距）
            
        Returns:
            裁剪后的视图
        """
        h, pano_w = full_panorama.shape[:2]
        
        # 使用与_create_full_panorama相同的方法计算焦距
        w_original = int(pano_w / (2 * np.pi) * 2 * np.tan(np.radians(hfov_per_image / 2)))
        f_pixels = w_original / (2.0 * np.tan(np.radians(hfov_per_image / 2.0)))
        
        # 计算裁剪的角度范围
        center_rad = np.radians(center_angle)
        half_fov_rad = np.radians(fov / 2.0)
        
        theta_start = center_rad - half_fov_rad
        theta_end = center_rad + half_fov_rad
        
        # 映射到全景图像素坐标
        x_start = int((theta_start % (2 * np.pi)) * f_pixels)
        x_end = int((theta_end % (2 * np.pi)) * f_pixels)
        
        # 处理跨越0°/360°边界的情况
        if x_end < x_start:
            # 环绕拼接
            part1 = full_panorama[:, x_start:]
            part2 = full_panorama[:, :x_end]
            view = np.hstack([part1, part2])
        else:
            view = full_panorama[:, x_start:x_end]
        
        return view
    
    def _stitch_panorama(self, images: List[np.ndarray], hfov: float) -> np.ndarray:
        """
        [已废弃] 柱面投影拼接3张图像生成90°全景图
        现在使用 _create_full_panorama + _crop_panorama_view 替代
        保留此方法以防向后兼容
        
        Args:
            images: 3张连续的图像列表（左-中-右）
            hfov: 每张图的水平视场角（度）
            
        Returns:
            拼接后的全景图
        """
        # 简单水平拼接作为降级方案
        return np.hstack(images)

