"""Environment registry: lookup by ID."""
from __future__ import annotations

from atlas.environments.base import BaseEnvironment

_REGISTRY: dict[str, type[BaseEnvironment]] = {}


def register(cls: type[BaseEnvironment]) -> type[BaseEnvironment]:
    instance = cls()
    _REGISTRY[instance.env_id] = cls
    return cls


def get_environment(env_id: str) -> BaseEnvironment:
    if env_id not in _REGISTRY:
        raise KeyError(f"Unknown environment: {env_id}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[env_id]()


def get_all_environments() -> list[BaseEnvironment]:
    return [cls() for cls in _REGISTRY.values()]
