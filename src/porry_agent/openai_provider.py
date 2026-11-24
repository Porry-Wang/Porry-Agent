"""OpenAI 兼容适配器：openai SDK 承担协议与重试，错误翻译为 ProviderError。"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx2
from openai import APIConnectionError, APIStatusError, OpenAI
from openai.types.chat import ChatCompletion

from porry_agent.messages import Message, ToolCall
from porry_agent.tools import ToolSpec

REQUEST_TIMEOUT_SECONDS = 60.0
MAX_RETRIES = 2

_TRUNCATED_MESSAGE = (
    "回复被截断（finish_reason=length）：输出达到 max_tokens 上限，"
    "可让模型分步输出或缩小单次任务。"
)


class ProviderError(Exception):
    """模型请求或连接配置错误。"""


def _tool_to_wire(spec: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _to_wire(m: Message) -> dict:
    if m.role == "tool":
        return {"role": "tool", "content": m.content, "tool_call_id": m.tool_call_id}
    wire: dict = {"role": m.role, "content": m.content}
    if m.tool_calls:
        wire["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in m.tool_calls
        ]
    return wire


def _error_detail(response: httpx2.Response) -> str:
    try:
        message = response.json()["error"]["message"]
        if isinstance(message, str) and message:
            return message[:200]
    except (KeyError, TypeError, ValueError):
        pass
    return response.text[:200]


def _status_message(exc: APIStatusError) -> str:
    status = exc.response.status_code
    detail = _error_detail(exc.response)
    if status == 429 or status >= 500:
        return f"重试 {MAX_RETRIES} 次后仍失败，最后错误：HTTP {status}：{detail}"
    return f"请求失败 HTTP {status}：{detail}"


def _normalized_arguments(arguments: object) -> str:
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    return arguments if isinstance(arguments, str) else ""


def _parse_reply(completion: ChatCompletion) -> Message:
    if not completion.choices:
        raise ProviderError("响应不是预期的 OpenAI 兼容格式：缺少 choices。")
    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise ProviderError(_TRUNCATED_MESSAGE)
    message = choice.message
    tool_calls = [
        ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=_normalized_arguments(tc.function.arguments),
        )
        for tc in message.tool_calls or []
    ]
    return Message(
        role="assistant", content=message.content or "", tool_calls=tool_calls
    )


class OpenAIProvider:
    """对接任意 OpenAI 兼容端点；换厂商只换 base_url 和模型名。"""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        max_output_tokens: int | None = None,
        tool_specs: list[ToolSpec] | None = None,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._tools = [_tool_to_wire(s) for s in tool_specs or []]
        self._http_client = httpx2.Client(
            timeout=REQUEST_TIMEOUT_SECONDS,
            transport=transport,
            follow_redirects=True,
        )
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=MAX_RETRIES,
            timeout=REQUEST_TIMEOUT_SECONDS,
            http_client=self._http_client,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAIProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _request_kwargs(self, messages: list[Message]) -> dict:
        kwargs: dict = {
            "model": self._model,
            "messages": [_to_wire(m) for m in messages],
        }
        if self._max_output_tokens is not None:
            kwargs["max_tokens"] = self._max_output_tokens
        if self._tools:
            kwargs["tools"] = self._tools
        return kwargs

    def complete(self, messages: list[Message]) -> Message:
        """发一轮请求拿下一条回复；瞬时错误由 SDK 退避重试，永久错误与截断抛 ProviderError。"""
        try:
            completion = self._client.chat.completions.create(
                **self._request_kwargs(messages)
            )
        except APIStatusError as exc:
            raise ProviderError(_status_message(exc)) from None
        except APIConnectionError as exc:
            raise ProviderError(
                f"网络错误（已重试 {MAX_RETRIES} 次）：{exc.__cause__ or exc}"
            ) from None
        return _parse_reply(completion)

    def stream_complete(
        self, messages: list[Message], *, on_text: Callable[[str], None]
    ) -> Message:
        """流式拿一条回复：content 碎片实时交给 on_text，拼完返回完整 Message。

        建立期错误由 SDK 重试；迭代期错误一律抛 ProviderError，不重试；迭代正常结束即完成。
        """
        kwargs = self._request_kwargs(messages)
        kwargs["stream"] = True
        try:
            stream = self._client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            raise ProviderError(_status_message(exc)) from None
        except APIConnectionError as exc:
            raise ProviderError(
                f"网络错误（已重试 {MAX_RETRIES} 次）：{exc.__cause__ or exc}"
            ) from None

        content_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta is None:
                    continue
                if delta.content:
                    content_parts.append(delta.content)
                    on_text(delta.content)
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(
                        tc.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function is None:
                        continue
                    if tc.function.name:
                        slot["name"] = tc.function.name
                    slot["arguments"] += _normalized_arguments(tc.function.arguments)
        except APIConnectionError as exc:
            raise ProviderError(
                f"流式回复中断：{exc.__cause__ or exc}。"
                f"已输出内容无法续传，请重发本轮。"
            ) from None
        except json.JSONDecodeError:
            raise ProviderError("流式数据不是合法 JSON，无法解析。") from None

        if finish_reason == "length":
            raise ProviderError(_TRUNCATED_MESSAGE)
        tool_calls = [
            ToolCall(id=slot["id"], name=slot["name"], arguments=slot["arguments"])
            for _, slot in sorted(calls.items())
        ]
        return Message(
            role="assistant", content="".join(content_parts), tool_calls=tool_calls
        )