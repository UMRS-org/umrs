"""One rule set for numeric and on/off ``UMRS_*`` knobs (review U1)."""

from __future__ import annotations

import logging

import pytest
from umrs.shared import env_knobs

NAME = "UMRS_TEST_KNOB"


@pytest.fixture(autouse=True)
def _unset(monkeypatch):
    monkeypatch.delenv(NAME, raising=False)


@pytest.mark.parametrize(("raw", "expected"), [(None, 7), ("", 7), (" 3 ", 3), ("1", 1), ("10", 10)])
def test_env_int_reads_valid_values(monkeypatch, raw, expected):
    if raw is not None:
        monkeypatch.setenv(NAME, raw)
    assert env_knobs.env_int(NAME, 7, lo=1, hi=10) == expected


@pytest.mark.parametrize("raw", ["0", "11", "-2", "abc", "2.5"])
def test_env_int_invalid_or_out_of_bounds_warns_and_keeps_default(monkeypatch, caplog, raw):
    monkeypatch.setenv(NAME, raw)
    with caplog.at_level(logging.WARNING, logger=env_knobs.logger.name):
        assert env_knobs.env_int(NAME, 7, lo=1, hi=10) == 7
    (record,) = [r for r in caplog.records if NAME in r.getMessage()]
    assert "1..10" in record.getMessage()


def test_env_int_without_upper_bound(monkeypatch):
    monkeypatch.setenv(NAME, "100000")
    assert env_knobs.env_int(NAME, 0, lo=0) == 100000


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, True), ("", True), ("1", True), ("yes", True), ("0", False), ("OFF", False),
     ("False", False), ("no", False)],
)
def test_env_flag(monkeypatch, raw, expected):
    if raw is not None:
        monkeypatch.setenv(NAME, raw)
    assert env_knobs.env_flag(NAME) is expected


def test_env_flag_default_off(monkeypatch):
    assert env_knobs.env_flag(NAME, default=False) is False
    monkeypatch.setenv(NAME, "on")
    assert env_knobs.env_flag(NAME, default=False) is True
