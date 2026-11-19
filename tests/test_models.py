import pytest

from porry_agent.models import (
    ModelRegistryError,
    load_models,
    resolve_model,
)


def test_builtin_table():
    models = load_models()
    assert {"mock", "gpt-4o-mini", "deepseek-flash", "qwen3.8-flash"} <= set(models)
    assert "deepseek-chat" not in models
    assert models["mock"].context_window == 200_000


def test_resolve_unknown_model():
    with pytest.raises(ModelRegistryError, match="未知模型 'nope'"):
        resolve_model("nope", load_models())


def test_toml_adds_model(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text(
        '[qwen-plus]\ncontext_window = 131072\nmax_output_tokens = 8192\n'
        'protocol = "openai"\nbase_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"\n'
        'api_key_env = "DASHSCOPE_API_KEY"\n',
        encoding="utf-8",
    )
    spec = load_models(p)["qwen-plus"]
    assert spec.context_window == 131072
    assert spec.api_key_env == "DASHSCOPE_API_KEY"


def test_toml_connection_only(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text(
        '[my-relay]\nprotocol = "openai"\nbase_url = "https://api.openlux.ai/v1"\n'
        'api_key_env = "OPENLUX_API_KEY"\n',
        encoding="utf-8",
    )
    spec = load_models(p)["my-relay"]
    assert spec.context_window is None
    assert spec.max_output_tokens is None
    assert spec.base_url == "https://api.openlux.ai/v1"


def test_toml_overrides_builtin(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text("[mock]\ncontext_window = 1000\nmax_output_tokens = 100\n", encoding="utf-8")
    assert load_models(p)["mock"].context_window == 1000


def test_toml_invalid_value(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text("[qwen-plus]\ncontext_window = -1\nmax_output_tokens = 8000\n", encoding="utf-8")
    with pytest.raises(ModelRegistryError, match="qwen-plus"):
        load_models(p)


def test_toml_unknown_key(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text("[qwen-plus]\ncontext_window = 1000\nmax_output_tokens = 100\nbogus = 1\n", encoding="utf-8")
    with pytest.raises(ModelRegistryError, match="bogus"):
        load_models(p)


def test_toml_missing_file(tmp_path):
    with pytest.raises(ModelRegistryError, match="不存在"):
        load_models(tmp_path / "nope.toml")


def test_toml_bad_syntax(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text("[broken\n", encoding="utf-8")
    with pytest.raises(ModelRegistryError, match="TOML"):
        load_models(p)