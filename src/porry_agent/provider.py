"""Provider 接口、离线 mock 实现与组装工厂。"""

from __future__ import annotations

from typing import Protocol

from porry_agent.config import Settings
from porry_agent.messages import Message
from porry_agent.models import ModelSpec, resolve_model
from porry_agent.openai_provider import OpenAIProvider, resolve_connection
from porry_agent.tools import ToolSpec


class Provider(Protocol):
    def complete(self, messages: list[Message]) -> Message: ...


class MockProvider:
    """离线确定性回复：回声最近一条用户消息。"""

    def complete(self, messages: list[Message]) -> Message:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), "")
        return Message(role="assistant", content=f"mock: {last_user}")


def build_provider(settings: Settings, models: dict[str, ModelSpec],
                   registry: dict[str, ToolSpec]) -> Provider:
    """按「模型覆盖 > 全局兜底」选 mock 或 OpenAI 兼容适配器。"""
    spec = resolve_model(settings.model, models)
    protocol = spec.protocol or settings.provider
    if protocol == "mock":
        return MockProvider()
    base_url, api_key = resolve_connection(
        settings.model, spec,
        fallback_base_url=settings.base_url,
        fallback_api_key=(settings.api_key.get_secret_value()
                          if settings.api_key else None))
    return OpenAIProvider(model=settings.model, base_url=base_url,
                          api_key=api_key, max_output_tokens=spec.max_output_tokens,
                          tool_specs=list(registry.values()))