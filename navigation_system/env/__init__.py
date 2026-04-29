"""Environment construction and explicit Habitat environment registration."""

from navigation_system.env.common import NavigationVectorEnv

_ENV_REGISTRATION_DONE = False


def construct_envs(*args, **kwargs):
    from navigation_system.env.factory import construct_envs as _construct_envs

    return _construct_envs(*args, **kwargs)


def ensure_env_registered() -> None:
    """Register custom Habitat envs exactly once before `get_env_class` is used."""
    global _ENV_REGISTRATION_DONE
    if _ENV_REGISTRATION_DONE:
        return

    import habitat_extensions.habitat_simulator as _habitat_simulator  # noqa: F401
    import habitat_extensions.measures as _measures  # noqa: F401
    import habitat_extensions.sensors as _sensors  # noqa: F401
    import habitat_extensions.task as _task  # noqa: F401
    import navigation_system.env.vlnce.r2r.zero_shot_env as _zero_shot_env  # noqa: F401

    _ENV_REGISTRATION_DONE = True


__all__ = [
    "NavigationVectorEnv",
    "construct_envs",
    "ensure_env_registered",
]
