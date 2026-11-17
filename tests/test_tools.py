import json

import pytest

from porry_agent.messages import ToolCall
from porry_agent.tools import (
    ToolSpec,
    build_default_registry,
    make_executor,
    render_system_prompt,
)


@pytest.fixture()
def workspace(tmp_path):
    (tmp_path / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def executor(workspace):
    return make_executor(build_default_registry(workspace))


def call(executor, name, **args):
    return executor(ToolCall(id="t1", name=name, arguments=json.dumps(args)))


def test_read_file(executor):
    assert call(executor, "read_file", path="a.txt") == "hello\nworld\n"


def test_read_missing(executor):
    assert "not_found" in call(executor, "read_file", path="nope.txt")


def test_dotdot_escape(executor):
    assert "outside_workspace" in call(executor, "read_file", path="../outside.txt")


def test_symlink_escape(executor, workspace, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("s", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside)
    assert "outside_workspace" in call(executor, "read_file", path="link.txt")


def test_edit_file(executor, workspace):
    call(executor, "read_file", path="a.txt")
    out = call(executor, "edit_file", path="a.txt", old_string="world", new_string="there")
    assert "已编辑" in out
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "hello\nthere\n"


def test_edit_requires_read_first(executor):
    assert "stale_read" in call(executor, "edit_file", path="a.txt",
                                 old_string="world", new_string="x")


def test_edit_no_match(executor):
    call(executor, "read_file", path="a.txt")
    assert "no_match" in call(executor, "edit_file", path="a.txt",
                              old_string="zzz", new_string="x")


def test_edit_multiple_matches(executor, workspace):
    (workspace / "dup.txt").write_text("x\nx\n", encoding="utf-8")
    call(executor, "read_file", path="dup.txt")
    out = call(executor, "edit_file", path="dup.txt", old_string="x", new_string="y")
    assert "multiple_matches" in out
    assert "1, 2" in out


def test_unknown_tool(executor):
    assert "unknown_tool" in call(executor, "bogus")


def test_bad_arguments(executor):
    out = executor(ToolCall(id="t1", name="read_file", arguments="{not json"))
    assert "invalid_args" in out


def test_handler_exception_becomes_observation(workspace):
    registry = build_default_registry(workspace)

    def boom(args):
        raise RuntimeError("炸了")

    registry["boom"] = ToolSpec(name="boom", description="", parameters={}, handler=boom)
    out = make_executor(registry)(ToolCall(id="t1", name="boom", arguments="{}"))
    assert "tool_error" in out
    assert "炸了" in out


def test_system_prompt_lists_tools(workspace):
    prompt = render_system_prompt(build_default_registry(workspace))
    assert "read_file" in prompt
    assert "edit_file" in prompt