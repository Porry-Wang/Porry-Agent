import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("PORRY_AGENT_") or "proxy" in key.lower():
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)