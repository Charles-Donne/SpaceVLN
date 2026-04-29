"""VLN-CE environment adapter entrypoints."""

from __future__ import annotations

from typing import Any, Optional, Sequence


def build_vlnce_vector_env(
    config: Any,
    *,
    auto_reset_done: bool = False,
    episodes_allowed: Optional[Sequence[str]] = None,
) -> Any:
    """Build the Habitat-backed VLN-CE VectorEnv behind SpaceVLN's env contract."""
    from habitat_baselines.common.environments import get_env_class

    from navigation_system.env import construct_envs, ensure_env_registered

    ensure_env_registered()
    env_class = get_env_class(config.ENV_NAME)
    if env_class is None:
        raise RuntimeError(
            f"Habitat environment '{config.ENV_NAME}' is not registered. "
            "Please check navigation_system.env.vlnce.r2r.zero_shot_env."
        )
    return construct_envs(
        config,
        env_class,
        auto_reset_done=auto_reset_done,
        episodes_allowed=list(episodes_allowed or []),
    )
