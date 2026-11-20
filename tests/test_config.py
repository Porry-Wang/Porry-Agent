import pytest

from porry_agent.config import ConfigError, load_settings


def test_defaults():
    s = load_settings()
    assert s.model == "mock"
    assert s.max_turns == 50
    assert s.log_level == "INFO"


def test_env_override(monkeypatch):
    monkeypatch.setenv("PORRY_AGENT_MAX_TURNS", "3")
    assert load_settings().max_turns == 3


def test_env_file_loaded(tmp_path):
    (tmp_path / ".env").write_text("PORRY_AGENT_MODEL=deepseek-chat\n", encoding="utf-8")
    assert load_settings().model == "deepseek-chat"


def test_env_file_found_from_subdirectory(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("PORRY_AGENT_MODEL=deepseek-chat\n", encoding="utf-8")
    sub = tmp_path / "src" / "pkg"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    assert load_settings().model == "deepseek-chat"


def test_invalid_value_raises():
    with pytest.raises(ConfigError, match="命令行参数 max_turns"):
        load_settings(max_turns=0)


def test_env_invalid_value_mentions_env_var(monkeypatch):
    monkeypatch.setenv("PORRY_AGENT_MAX_TURNS", "0")
    with pytest.raises(ConfigError, match="PORRY_AGENT_MAX_TURNS"):
        load_settings()


def test_unknown_init_kwarg_raises():
    with pytest.raises(ConfigError):
        load_settings(bogus=1)


def test_env_file_unknown_key_raises(tmp_path):
    (tmp_path / ".env").write_text("PORRY_AGENT_BOGUS=1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings()