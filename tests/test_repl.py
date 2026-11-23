from porry_agent.messages import Message, ToolCall
from porry_agent.openai_provider import ProviderError
from porry_agent.repl import run_repl


class RecordingProvider:
    """按剧本回复并记录每轮收到的 messages；剧本项可为异常。"""

    def __init__(self, replies: list[Message | Exception]) -> None:
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> Message:
        self.calls.append(list(messages))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def stream_complete(self, messages, *, on_text):
        reply = self.complete(messages)
        on_text(reply.content)
        return reply


def _feed(monkeypatch, inputs: list[str]) -> None:
    it = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(it))


def _reply(text: str) -> Message:
    return Message(role="assistant", content=text)


def test_two_turns_accumulate_history(monkeypatch, capsys):
    provider = RecordingProvider([_reply("答一"), _reply("答二")])
    _feed(monkeypatch, ["第一问", "第二问", "exit"])
    assert run_repl(provider, lambda call: "ok", max_turns=5, system="s") == 0
    assert len(provider.calls) == 2
    second = provider.calls[1]
    assert second[0].role == "system"
    assert [m.content for m in second] == ["s", "第一问", "答一", "第二问"]
    out = capsys.readouterr().out
    assert "答一" in out and "答二" in out
    assert out.count("[done] turns=1 stop=completed") == 2


def test_exit_and_quit_without_running(monkeypatch):
    for command in ("exit", "quit", "EXIT"):
        provider = RecordingProvider([])
        _feed(monkeypatch, [command])
        assert run_repl(provider, lambda call: "ok", max_turns=5) == 0
        assert provider.calls == []


def test_empty_input_skipped(monkeypatch):
    provider = RecordingProvider([_reply("答")])
    _feed(monkeypatch, ["", "   ", "问", "exit"])
    assert run_repl(provider, lambda call: "ok", max_turns=5) == 0
    assert len(provider.calls) == 1


def test_eof_exits(monkeypatch):
    def raise_eof(_prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", raise_eof)
    provider = RecordingProvider([])
    assert run_repl(provider, lambda call: "ok", max_turns=5) == 0


def test_bad_turn_continues_session(monkeypatch, capsys):
    provider = RecordingProvider([ProviderError("模拟故障"), _reply("恢复")])
    _feed(monkeypatch, ["坏轮", "好轮", "exit"])
    assert run_repl(provider, lambda call: "ok", max_turns=5) == 0
    assert len(provider.calls) == 2
    err = capsys.readouterr().err
    assert "本轮失败" in err and "模拟故障" in err


def test_keyboard_interrupt_exits(monkeypatch):
    def raise_interrupt(_prompt=""):
        raise KeyboardInterrupt
    monkeypatch.setattr("builtins.input", raise_interrupt)
    provider = RecordingProvider([])
    assert run_repl(provider, lambda call: "ok", max_turns=5) == 0


class StreamingProvider:
    """流式剧本：每轮一个事件列表，str 吐碎片，Exception 当场抛。"""

    def __init__(self, turns: list[list]) -> None:
        self._turns = list(turns)

    def stream_complete(self, messages, *, on_text):
        events = self._turns.pop(0)
        for event in events:
            if isinstance(event, Exception):
                raise event
            on_text(event)
        return Message(
            role="assistant",
            content="".join(e for e in events if isinstance(e, str)),
        )


class ToolThenStreamProvider:
    """第一轮流一句话再请求工具；第二轮流收尾。"""

    def __init__(self) -> None:
        self.turn = 0

    def stream_complete(self, messages, *, on_text):
        self.turn += 1
        if self.turn == 1:
            on_text("我先读文件。")
            return Message(
                role="assistant",
                content="我先读文件。",
                tool_calls=[ToolCall(
                    id="c1", name="read_file", arguments='{"path": "a.txt"}')],
            )
        on_text("读完了")
        return Message(role="assistant", content="读完了")


def test_stream_prints_fragments_once(monkeypatch, capsys):
    provider = StreamingProvider([["你", "好"]])
    _feed(monkeypatch, ["问", "exit"])
    assert run_repl(provider, lambda call: "ok", max_turns=5) == 0
    out = capsys.readouterr().out
    assert out.count("你好") == 1
    assert "你好\n[done] turns=1 stop=completed" in out


def test_stream_failure_midway_continues(monkeypatch, capsys):
    provider = StreamingProvider([
        ["半截", ProviderError("断了")],
        ["恢复"],
    ])
    _feed(monkeypatch, ["坏轮", "好轮", "exit"])
    assert run_repl(provider, lambda call: "ok", max_turns=5) == 0
    captured = capsys.readouterr()
    assert "半截\n" in captured.out
    assert "本轮失败" in captured.err and "断了" in captured.err
    assert "恢复" in captured.out


def test_stream_across_tool_turns(monkeypatch, capsys):
    provider = ToolThenStreamProvider()
    _feed(monkeypatch, ["读文件", "exit"])
    assert run_repl(provider, lambda call: "文件内容", max_turns=5) == 0
    out = capsys.readouterr().out
    assert "我先读文件。" in out
    assert out.count("读完了") == 1
    assert "[done] turns=2 stop=completed" in out