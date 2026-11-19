"""OpenAI 兼容适配器：连接解析、请求/响应转换、错误分类与重试。"""

from __future__ import annotations

import json
import os
import time
from typing import Mapping, Optional
from urllib.parse import urlparse

import httpx

from porry_agent.messages import Message, ToolCall
from porry_agent.models import ModelSpec
from porry_agent.tools import ToolSpec

REQUEST_TIMEOUT_SECONDS = 60.0
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0


class ProviderError(Exception):
    """模型请求或连接配置错误。"""


def resolve_connection(
    model_name: str,
    spec: ModelSpec,
    *,
    fallback_base_url: str,
    fallback_api_key: Optional[str],
    env: Optional[Mapping[str, str]] = None,
) -> tuple[str, str]:
    """按「模型覆盖 > 全局兜底」解析 base_url 与密钥；密钥只从 env 读。"""
    env = os.environ if env is None else env

    base_url = spec.base_url or fallback_base_url
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProviderError(
            f"模型 '{model_name}' 的 base_url 非法：{base_url!r}，"
            f"需要 http(s):// 开头的完整地址。")

    if spec.api_key_env:
        key = env.get(spec.api_key_env)
        if not key or not key.strip():
            raise ProviderError(
                f"模型 '{model_name}' 需要密钥：设置环境变量 {spec.api_key_env}。"
                f"环境变量只对设置后启动的进程生效：可在当前终端直接设置"
                f"（macOS/Linux：export {spec.api_key_env}=值；"
                f"PowerShell：$env:{spec.api_key_env}='值'），"
                f"或写入配置文件/系统设置后新开终端。")
    else:
        key = fallback_api_key
        if not key or not key.strip():
            raise ProviderError(
                f"模型 '{model_name}' 需要密钥：在 MODELS_FILE 给它配 api_key_env，"
                f"或设置 PORRY_AGENT_API_KEY。")
    return base_url, key


def _tool_to_wire(spec: ToolSpec) -> dict:
    return {"type": "function",
            "function": {"name": spec.name, "description": spec.description,
                         "parameters": spec.parameters}}


def _error_detail(response: httpx.Response) -> str:
    try:
        message = response.json()["error"]["message"]
        if isinstance(message, str) and message:
            return message[:200]
    except (KeyError, TypeError, ValueError):
        pass
    return response.text[:200]


def _to_wire(m: Message) -> dict:
    if m.role == "tool":
        return {"role": "tool", "content": m.content, "tool_call_id": m.tool_call_id}
    wire: dict = {"role": m.role, "content": m.content}
    if m.tool_calls:
        wire["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": c.arguments}}
            for c in m.tool_calls]
    return wire


def _parse_reply(response: httpx.Response) -> Message:
    try:
        result = response.json()["choices"][0]
        if result.get("finish_reason") == "length":
            raise ProviderError(
                "回复被 max_tokens 截断（finish_reason=length），"
                "请调大该模型的 max_output_tokens。")
        choice = result["message"]
        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            arguments = tc["function"].get("arguments") or ""
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append(ToolCall(
                id=tc["id"], name=tc["function"]["name"], arguments=arguments))
        return Message(role="assistant", content=choice.get("content") or "",
                       tool_calls=tool_calls)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ProviderError(f"响应不是预期的 OpenAI 兼容格式：{exc}") from None


class OpenAIProvider:
    """对接任意 OpenAI 兼容端点；换厂商只换 base_url 和模型名。"""

    def __init__(self, *, model: str, base_url: str, api_key: str,
                 max_output_tokens: Optional[int] = None,
                 tool_specs: Optional[list[ToolSpec]] = None,
                 transport: Optional[httpx.BaseTransport] = None) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._tools = [_tool_to_wire(s) for s in tool_specs or []]
        try:
            self._client = httpx.Client(
                base_url=base_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=REQUEST_TIMEOUT_SECONDS,
                transport=transport,
                follow_redirects=True)
        except httpx.InvalidURL as exc:
            raise ProviderError(
                f"HTTP 客户端初始化失败：{exc}。"
                f"常见原因是代理环境变量含非法条目（如 NO_PROXY 里的 [::1]），"
                f"请清空 NO_PROXY 与 no_proxy 后重试。") from None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAIProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def complete(self, messages: list[Message]) -> Message:
        """发一轮请求拿下一条回复；瞬时错误退避重试，永久错误与截断抛 ProviderError。"""
        payload: dict = {
            "model": self._model,
            "messages": [_to_wire(m) for m in messages],
        }
        if self._max_output_tokens is not None:
            payload["max_tokens"] = self._max_output_tokens
        if self._tools:
            payload["tools"] = self._tools

        last_error = ""
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                last_error = f"网络错误：{exc}"
            else:
                if response.status_code == 200:
                    return _parse_reply(response)
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}：{_error_detail(response)}"
                else:
                    raise ProviderError(
                        f"请求失败 HTTP {response.status_code}：{_error_detail(response)}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
        raise ProviderError(f"重试 {MAX_RETRIES} 次后仍失败，最后错误：{last_error}")