"""Provider 接口、离线 mock 实现、厂商档案与组装工厂。"""

from __future__ import annotations

import os
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from porry_agent.config import Settings
from porry_agent.messages import Message
from porry_agent.openai_provider import OpenAIProvider, ProviderError
from porry_agent.tools import ToolSpec


class Provider(Protocol):
    def complete(self, messages: list[Message]) -> Message: ...


class MockProvider:
    """离线确定性回复：回声最近一条用户消息。"""

    def complete(self, messages: list[Message]) -> Message:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        return Message(role="assistant", content=f"mock: {last_user}")


class Vendor(BaseModel):
    """厂商连接档案：端点 + 密钥环境变量名，密钥值只从 env 读。"""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    api_key_env: str


# 新增厂商：这里加一条记录 + 设好它的密钥环境变量；其模型即只是名字
VENDORS: dict[str, Vendor] = {
    "openai": Vendor(
        base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY"
    ),
    "deepseek": Vendor(
        base_url="https://api.deepseek.com", api_key_env="DEEPSEEK_API_KEY"
    ),
    "dashscope": Vendor(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
    ),
    "openlux": Vendor(
        base_url="https://api.openlux.ai/v1", api_key_env="OPENLUX_API_KEY"
    ),
}


def build_provider(settings: Settings, tools: dict[str, ToolSpec]) -> Provider:
    """按「厂商:模型名」选择组装 mock 或 OpenAI 兼容适配器；模型名原样发给厂商。"""
    if settings.model == "mock":
        return MockProvider()
    vendor_name, sep, model_name = settings.model.partition(":")
    if not sep:
        model_name = vendor_name
        if not settings.vendor:
            raise ProviderError(
                f"模型 '{settings.model}' 未指定厂商：用「厂商:模型名」格式，"
                f"或设置 PORRY_AGENT_VENDOR。"
            )
        vendor_name = settings.vendor
    if not model_name:
        raise ProviderError(f"模型名为空：'{settings.model}'。")
    if vendor_name not in VENDORS:
        known = ", ".join(sorted(VENDORS))
        raise ProviderError(f"未知厂商 '{vendor_name}'。已知厂商：{known}。")
    vendor = VENDORS[vendor_name]
    key = os.environ.get(vendor.api_key_env)
    if not key or not key.strip():
        raise ProviderError(
            f"厂商 '{vendor_name}' 缺少密钥：设置环境变量 {vendor.api_key_env}，"
            f"设置后重新启动程序生效。"
        )
    return OpenAIProvider(
        model=model_name,
        base_url=vendor.base_url,
        api_key=key,
        tool_specs=list(tools.values()),
    )
