import logging

import pytest

from porry_agent.logging_setup import setup_logging


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    httpx2 = logging.getLogger("httpx2")
    handlers, level, h_level = root.handlers[:], root.level, httpx2.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    httpx2.setLevel(h_level)


def test_httpx2_pinned_to_warning_at_info(restore_logging):
    setup_logging("INFO")
    assert logging.getLogger().level == logging.INFO
    assert logging.getLogger("httpx2").level == logging.WARNING


def test_httpx2_follows_debug(restore_logging):
    setup_logging("DEBUG")
    assert logging.getLogger("httpx2").level == logging.DEBUG