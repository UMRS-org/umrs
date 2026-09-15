"""Unit tests for deployment-scoped model visibility filtering."""

import importlib

import pytest

import umrs.api.model_visibility as mv

FRONTIER = "GLM-5.1,GLM-5.2,Kimi-K2.6,DeepSeek-V4-Pro,DeepSeek-V4-Pro-128K,DeepSeek-V4-Pro-TEP"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("UMRS_MODEL_ALLOWLIST", raising=False)
    monkeypatch.delenv("UMRS_MODEL_DENYLIST", raising=False)
    importlib.reload(mv)
    yield


def test_no_filter_shows_everything():
    assert mv.is_model_visible("Qwen3-32B")
    assert mv.is_model_visible("GLM-5.1")
    assert mv.is_model_visible(None)


def test_allowlist_restricts_to_listed_models(monkeypatch):
    monkeypatch.setenv("UMRS_MODEL_ALLOWLIST", FRONTIER)
    assert mv.is_model_visible("GLM-5.1")
    assert mv.is_model_visible("deepseek-v4-pro")  # case-insensitive
    assert not mv.is_model_visible("Qwen3-32B")
    assert not mv.is_model_visible("sam3.1")


def test_denylist_hides_listed_models(monkeypatch):
    monkeypatch.setenv("UMRS_MODEL_DENYLIST", FRONTIER)
    assert mv.is_model_visible("Qwen3-32B")
    assert not mv.is_model_visible("GLM-5.1")
    assert not mv.is_model_visible("DeepSeek-V4-Pro-TEP")


def test_allowlist_takes_precedence_over_denylist(monkeypatch):
    monkeypatch.setenv("UMRS_MODEL_ALLOWLIST", "GLM-5.1")
    monkeypatch.setenv("UMRS_MODEL_DENYLIST", "GLM-5.1")
    assert mv.is_model_visible("GLM-5.1")
    assert not mv.is_model_visible("Kimi-K2.6")


def test_blank_entries_ignored(monkeypatch):
    monkeypatch.setenv("UMRS_MODEL_ALLOWLIST", " GLM-5.1 , , ")
    assert mv.is_model_visible("GLM-5.1")
    assert not mv.is_model_visible("Kimi-K2.6")
