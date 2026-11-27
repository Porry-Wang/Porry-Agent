import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("PORRY_AGENT_") or "proxy" in key.lower():
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture()
def assert_mode():
    """
    证明“文件权限没有写丢”
    权限断言：POSIX 校验完整权限位；Windows 的 chmod 只切换只读位。
    """

    def check(path, expected):
        mode = path.stat().st_mode & 0o777
        if os.name == "nt":
            # Windows 精简比较
            assert bool(mode & 0o222) == bool(expected & 0o222)
        else:
            # POSIX 精确比较
            assert mode == expected

    return check