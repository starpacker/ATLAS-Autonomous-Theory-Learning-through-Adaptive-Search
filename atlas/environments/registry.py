"""Environment registry: lookup by ID."""
from __future__ import annotations

from atlas.environments.base import BaseEnvironment

_REGISTRY: dict[str, type[BaseEnvironment]] = {}


def register(cls: type[BaseEnvironment]) -> type[BaseEnvironment]:
    instance = cls()
    _REGISTRY[instance.env_id] = cls
    return cls


def get_environment(env_id: str, seed: int | None = None) -> BaseEnvironment:
    """Retrieve an environment instance by ID.

    Parameters
    ----------
    seed:
        Optional RNG seed passed to the environment constructor.  Environments
        that support stochastic output (e.g. ENV-04 low-intensity mode) use
        this to control randomness.  Ignored by environments whose ``__init__``
        does not accept a *seed* parameter.
    """
    if env_id not in _REGISTRY:
        raise KeyError(f"Unknown environment: {env_id}. Available: {list(_REGISTRY.keys())}")
    cls = _REGISTRY[env_id]
    try:
        return cls(seed=seed)
    except TypeError:
        # Environment __init__ doesn't accept seed — construct without it
        return cls()


def get_all_environments() -> list[BaseEnvironment]:
    return [cls() for cls in _REGISTRY.values()]
