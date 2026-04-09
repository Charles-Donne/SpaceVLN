"""Parallel controller that swaps in the Qwen explicit-context-cache planner/action chain."""

from habitat import Config

from navigation_system.controllers.vlm_navigation_controller import VLMNavigationController
from navigation_system.vlm.execution.executor_qwen_cache import QwenContextCacheActionExecutor
from navigation_system.vlm.planning.planner_qwen_cache import QwenContextCachePlanner


class QwenContextCacheNavigationController(VLMNavigationController):
    """Navigation controller variant that keeps the original runtime flow but swaps model adapters."""

    def __init__(
        self,
        config: Config,
        config_path: str = "navigation_system/config/api/vlm_api_config_qwen_cache.yaml",
    ):
        super().__init__(config, config_path=config_path)

        try:
            self.planner = QwenContextCachePlanner(config_path, self.action_space)
            self.planner.set_request_artifact_saving(
                self.runtime_options.save_api_request_artifacts
            )
        except Exception as exc:
            print(f"[WARN] Cached LLM Planner init failed: {exc}")
            self.planner = None

        try:
            self.action_executor = QwenContextCacheActionExecutor(
                config_path,
                self.turn_angle,
                self.move_distance,
            )
            self.action_executor.set_request_artifact_saving(
                self.runtime_options.save_api_request_artifacts
            )
        except Exception as exc:
            print(f"[WARN] Cached Action Executor init failed: {exc}")
            self.action_executor = None
