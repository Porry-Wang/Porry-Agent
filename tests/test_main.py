from porry_agent.__main__ import main
from porry_agent.provider import MockProvider


def test_default_mock_warns(capsys):
    assert main(["你好"]) == 0
    out = capsys.readouterr()
    assert "模型：mock｜默认厂商：未设置｜类型：离线 mock" in out.out
    assert "[警告] 未指定模型" in out.err
    assert "mock: 你好" in out.out
    assert "[done] turns=1 stop=completed" in out.out


def test_explicit_mock_env_no_warning(monkeypatch, capsys):
    monkeypatch.setenv("PORRY_AGENT_MODEL", "mock")
    assert main(["你好"]) == 0
    out = capsys.readouterr()
    assert "模型：mock" in out.out
    assert "[警告]" not in out.err


def test_explicit_mock_cli_no_warning(capsys):
    assert main(["--model", "mock", "你好"]) == 0
    out = capsys.readouterr()
    assert "模型：mock" in out.out
    assert "[警告]" not in out.err


def test_env_file_model_and_vendor(tmp_path, monkeypatch, capsys):
    (tmp_path / ".env").write_text(
        "PORRY_AGENT_MODEL=qwen3.8-flash\nPORRY_AGENT_VENDOR=openlux\n",
        encoding="utf-8")
    monkeypatch.setattr("porry_agent.__main__.build_provider",
                        lambda settings, registry: MockProvider())
    assert main(["你好"]) == 0
    out = capsys.readouterr()
    assert "模型：qwen3.8-flash" in out.out
    assert "默认厂商：openlux" in out.out
    assert "[警告]" not in out.err


def test_env_file_explicit_mock_no_warning(tmp_path, capsys):
    (tmp_path / ".env").write_text("PORRY_AGENT_MODEL=mock\n", encoding="utf-8")
    assert main(["你好"]) == 0
    out = capsys.readouterr()
    assert "模型：mock" in out.out
    assert "[警告]" not in out.err


def test_prefix_selection_startup_line(monkeypatch, capsys):
    monkeypatch.setattr("porry_agent.__main__.build_provider",
                        lambda settings, registry: MockProvider())
    assert main(["--model", "dashscope:qwen3.8-flash", "你好"]) == 0
    out = capsys.readouterr()
    assert "模型：dashscope:qwen3.8-flash" in out.out


def test_unknown_vendor_fails_with_context(capsys):
    assert main(["--model", "nope:m", "你好"]) == 1
    err = capsys.readouterr().err
    assert "启动失败" in err
    assert "未知厂商" in err
    assert "默认厂商：未设置" in err
    assert "工作目录：" in err


def test_bare_name_without_vendor_fails(capsys):
    assert main(["--model", "qwen3.8-flash", "你好"]) == 1
    assert "PORRY_AGENT_VENDOR" in capsys.readouterr().err


def test_empty_cli_model_rejected(capsys):
    assert main(["--model", "", "你好"]) == 1
    assert "启动失败" in capsys.readouterr().err


def test_no_task_enters_repl(monkeypatch, capsys):
    inputs = iter(["你好", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    assert main([]) == 0
    out = capsys.readouterr()
    assert "mock: 你好" in out.out
    assert "[done] turns=1 stop=completed" in out.out


def test_task_arg_one_shot_does_not_read_stdin(monkeypatch, capsys):
    def fail_input(_prompt=""):
        raise AssertionError("一次性模式不应读 stdin")
    monkeypatch.setattr("builtins.input", fail_input)
    assert main(["你好"]) == 0
    assert "mock: 你好" in capsys.readouterr().out