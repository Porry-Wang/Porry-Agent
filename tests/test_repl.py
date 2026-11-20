from porry_agent.messages import Message
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