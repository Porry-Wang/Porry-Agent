from porry_agent.messages import Message
from porry_agent.provider import MockProvider


def test_mock_echoes_last_user():
    msgs = [
        Message(role="user", content="first"),
        Message(role="assistant", content="hi"),
        Message(role="user", content="second"),
    ]
    assert MockProvider().complete(msgs).content == "mock: second"


def test_mock_without_user_message():
    msgs = [Message(role="system", content="s")]
    assert MockProvider().complete(msgs).content == "mock: "


def test_message_defaults():
    m = Message(role="user", content="x")
    assert m.tool_calls == []
    assert m.tool_call_id is None