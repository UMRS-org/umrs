"""The embedding sandbox lane fails closed on unknown configuration."""

from __future__ import annotations

import re

import pytest
from umrs.knowledge import embed_sandbox as es

OFF_VALUES = ["", "off", "OFF", "none", "0", "false", "no", "in_process", "inprocess"]
UNKNOWN_VALUES = ["podmann", "docker", "on", "true", "1"]


def test_podman_opts_in(monkeypatch):
    monkeypatch.setenv(es.SANDBOX_ENV, " Podman ")
    assert es.sandbox_opted_in() is True


@pytest.mark.parametrize("raw", OFF_VALUES)
def test_the_off_words_keep_the_in_process_lane(monkeypatch, raw):
    monkeypatch.setenv(es.SANDBOX_ENV, raw)
    assert es.sandbox_opted_in() is False
    assert es.sandbox_config_from_env("/m") is None
    assert es.worker_sandbox_config() is None


def test_unset_keeps_the_in_process_lane(monkeypatch):
    monkeypatch.delenv(es.SANDBOX_ENV, raising=False)
    assert es.sandbox_opted_in() is False
    assert es.sandbox_config_from_env("/m") is None
    assert es.worker_sandbox_config() is None


@pytest.mark.parametrize("raw", UNKNOWN_VALUES)
def test_an_unknown_value_is_refused_by_name(monkeypatch, raw):
    monkeypatch.setenv(es.SANDBOX_ENV, raw)
    message = re.escape(f"{es.SANDBOX_ENV}={raw!r}")
    with pytest.raises(es.EmbeddingSandboxError, match=message):
        es.sandbox_config_from_env("/m")
    with pytest.raises(es.EmbeddingSandboxError, match=message):
        es.worker_sandbox_config()
    with pytest.raises(es.EmbeddingSandboxError, match=message):
        es.sandbox_opted_in()
