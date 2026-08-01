"""内置 setup adapters；外部 adapter 可通过 ``register_adapter`` 注册。"""
from __future__ import annotations

from .base import AdapterContext, AdapterError, AgentAdapter, Launcher
from .codex import CodexAdapter
from .json_agents import ClaudeCodeAdapter, CursorAdapter


_ADAPTERS: dict[str, AgentAdapter] = {}


def register_adapter(adapter: AgentAdapter) -> None:
    """注册 adapter；名称冲突时显式失败，避免静默覆盖。"""

    if adapter.name in _ADAPTERS:
        raise ValueError(f"setup adapter 已注册: {adapter.name}")
    _ADAPTERS[adapter.name] = adapter


def get_adapters() -> tuple[AgentAdapter, ...]:
    return tuple(_ADAPTERS.values())


for _adapter in (ClaudeCodeAdapter(), CursorAdapter(), CodexAdapter()):
    register_adapter(_adapter)


__all__ = [
    "AdapterContext",
    "AdapterError",
    "AgentAdapter",
    "Launcher",
    "get_adapters",
    "register_adapter",
]
