"""Bounded boot-time model loading (issue #635).

A stalled Hugging Face transfer hung ``load_sentence_transformers_model`` for
9 minutes during boot provisioning, past the compose health window. These
tests drive the loader with fake ``SentenceTransformer`` constructors that
hang, fail, then succeed — no network, no torch model.
"""

from __future__ import annotations

import threading
import time

import pytest
from umrs.inference import sentence_transformers as st_mod
from umrs.knowledge import text_embedder
from umrs.shared import bounded_load
from umrs.shared.bounded_load import LoadPolicy, ModelLoadError, load_bounded

FAST = {
    bounded_load.TIMEOUT_ENV: "0.2",
    bounded_load.ATTEMPTS_ENV: "3",
    bounded_load.BACKOFF_ENV: "0",
    bounded_load.COOLDOWN_ENV: "600",
}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name, value in FAST.items():
        monkeypatch.setenv(name, value)
    bounded_load.reset_cooldowns()
    st_mod._model_cache.clear()
    release = threading.Event()
    yield release
    release.set()  # let any abandoned fake loader thread finish
    bounded_load.reset_cooldowns()
    st_mod._model_cache.clear()


class _ScriptedLoader:
    """Fake SentenceTransformer: plays one scripted outcome per attempt."""

    def __init__(self, script, release):
        self.script = list(script)
        self.release = release
        self.calls = []

    def __call__(self, source, local_files_only=False):
        self.calls.append((source, local_files_only))
        if local_files_only:
            raise OSError("local cache empty")
        step = self.script.pop(0)
        if step == "hang":
            self.release.wait(30)
            return "late"
        if step == "fail":
            raise ConnectionError("hf transfer reset")
        return f"model:{source}"


def test_a_hanging_download_is_abandoned_and_retried(monkeypatch, _clean):
    fake = _ScriptedLoader(["hang", "fail", "ok"], _clean)
    monkeypatch.setattr(st_mod, "SentenceTransformer", fake)
    started = time.monotonic()
    model = st_mod.load_sentence_transformers_model("intfloat/e5-base-v2")
    assert model == "model:intfloat/e5-base-v2"
    assert time.monotonic() - started < 5  # old code: blocked until the hang ended
    assert [c for c in fake.calls if not c[1]] == [("intfloat/e5-base-v2", False)] * 3


def test_attempts_are_bounded_then_the_model_cools_down(monkeypatch, _clean):
    fake = _ScriptedLoader(["hang", "hang", "hang"], _clean)
    monkeypatch.setattr(st_mod, "SentenceTransformer", fake)
    started = time.monotonic()
    with pytest.raises(ModelLoadError, match="not loaded after 3 attempts"):
        st_mod.load_sentence_transformers_model("intfloat/e5-base-v2")
    assert time.monotonic() - started < 5
    # A second modality naming the same model fails at once (one bound per boot).
    calls_before = len(fake.calls)
    with pytest.raises(ModelLoadError, match="cooldown"):
        st_mod.load_sentence_transformers_model("intfloat/e5-base-v2")
    assert len(fake.calls) == calls_before


def test_backoff_doubles_between_attempts():
    pauses = []

    def _always_fails():
        raise ConnectionError("down")

    policy = LoadPolicy(timeout_s=1, attempts=3, backoff_s=5, cooldown_s=0)
    with pytest.raises(ModelLoadError):
        load_bounded("k", _always_fails, policy, sleep=pauses.append)
    assert pauses == [5, 10]
    assert policy.worst_case_s() == 3 * 1 + 5 + 10


def test_the_bundled_copy_is_loaded_without_the_network(monkeypatch, tmp_path, _clean):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr(text_embedder, "BUNDLED_TEXT_EMBEDDER_DIR", str(tmp_path))
    seen = []

    def _offline_only(source, local_files_only=False):
        seen.append((source, local_files_only))
        if not local_files_only:
            raise AssertionError("network download attempted")
        return f"model:{source}"

    monkeypatch.setattr(st_mod, "SentenceTransformer", _offline_only)
    assert st_mod.load_sentence_transformers_model("intfloat/e5-base-v2") == f"model:{tmp_path}"
    assert seen == [(str(tmp_path), True)]


def test_env_policy_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv(bounded_load.TIMEOUT_ENV, "soon")
    monkeypatch.setenv(bounded_load.ATTEMPTS_ENV, "0")
    policy = LoadPolicy.from_env()
    assert policy.timeout_s == LoadPolicy().timeout_s
    assert policy.attempts == 1
