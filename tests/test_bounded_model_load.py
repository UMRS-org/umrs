"""Bounded boot-time model loading (issue #635).

A stalled Hugging Face transfer hung ``load_sentence_transformers_model`` for
9 minutes during boot provisioning, past the compose health window. These
tests drive the loaders with fake constructors that hang, fail, are slow, then
succeed — no network, no torch model, no real bundle directory.
"""

from __future__ import annotations

import threading
import time

import pytest
from umrs.inference import loading
from umrs.inference import sam3_segmentation as sam_mod
from umrs.inference import sentence_transformers as st_mod
from umrs.knowledge import text_embedder
from umrs.shared import bounded_load
from umrs.shared.bounded_load import LoadPolicy, ModelLoadError, load_bounded

FAST = {
    bounded_load.TIMEOUT_ENV: "1",  # the env floor
    bounded_load.ATTEMPTS_ENV: "3",
    bounded_load.BACKOFF_ENV: "0",
    bounded_load.COOLDOWN_ENV: "600",
}
E5 = "intfloat/e5-base-v2"


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for name, value in FAST.items():
        monkeypatch.setenv(name, value)
    # Hermetic: a host that staged the real bundle must not change the path.
    monkeypatch.setattr(text_embedder, "BUNDLED_TEXT_EMBEDDER_DIR", str(tmp_path / "no-bundle"))
    bounded_load.reset_cooldowns()
    st_mod._model_cache.clear()
    sam_mod._sam3_cache.clear()
    release = threading.Event()
    yield release
    release.set()  # let any abandoned fake loader thread finish
    bounded_load.reset_cooldowns()
    st_mod._model_cache.clear()
    sam_mod._sam3_cache.clear()


class _ScriptedLoader:
    """Fake hub loader: plays one scripted outcome per network call."""

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
        if step == "slow":
            time.sleep(1.5)  # > one 1 s attempt, < the 3-attempt bound
            return f"slow:{source}"
        if step == "fail":
            raise ConnectionError("hf transfer reset")
        return f"model:{source}"

    def network_calls(self):
        return [c for c in self.calls if not c[1]]


def test_a_failed_download_is_retried_fresh(monkeypatch, _clean):
    fake = _ScriptedLoader(["fail", "fail", "ok"], _clean)
    monkeypatch.setattr(st_mod, "SentenceTransformer", fake)
    assert st_mod.load_sentence_transformers_model(E5) == f"model:{E5}"
    assert fake.network_calls() == [(E5, False)] * 3


def test_a_stalled_download_is_bounded_and_never_duplicated(monkeypatch, _clean):
    fake = _ScriptedLoader(["hang", "ok", "ok"], _clean)
    monkeypatch.setattr(st_mod, "SentenceTransformer", fake)
    started = time.monotonic()
    with pytest.raises(ModelLoadError, match="not loaded after 3 attempts"):
        st_mod.load_sentence_transformers_model(E5)
    assert time.monotonic() - started < 6  # pre-#635 code: blocked until the hang ended
    # One wedged load holds the blob lock; attempts 2 and 3 wait on it rather
    # than start loads that would queue behind that lock.
    assert len(fake.network_calls()) == 1
    # A second modality naming the same model fails at once (one bound per boot).
    with pytest.raises(ModelLoadError, match="cooldown"):
        st_mod.load_sentence_transformers_model(E5)
    assert len(fake.network_calls()) == 1


def test_a_slow_load_is_joined_not_restarted(monkeypatch, _clean):
    fake = _ScriptedLoader(["slow", "ok", "ok"], _clean)
    monkeypatch.setattr(st_mod, "SentenceTransformer", fake)
    # The first load outlives attempt 1; attempt 2 or 3 collects its result
    # instead of loading a second copy alongside it.
    assert st_mod.load_sentence_transformers_model(E5) == f"slow:{E5}"
    assert len(fake.network_calls()) == 1


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
    assert st_mod.load_sentence_transformers_model(E5) == f"model:{tmp_path}"
    assert seen == [(str(tmp_path), True)]


def test_the_openai_spec_tokenizer_load_is_bounded(monkeypatch, _clean):
    import transformers

    fake = _ScriptedLoader(["hang"], _clean)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake)
    started = time.monotonic()
    with pytest.raises(ModelLoadError, match="tokenizer:org/remote-model"):
        loading._load_hub_tokenizer("org/remote-model")
    assert time.monotonic() - started < 6


def test_the_transformers_sam_load_is_bounded(monkeypatch, _clean):
    fake = _ScriptedLoader(["hang"], _clean)
    monkeypatch.setattr(sam_mod, "_load_hub_sam", lambda name, _device: fake(name))
    started = time.monotonic()
    with pytest.raises(ModelLoadError, match="sam:facebook/sam3"):
        sam_mod.load_sam3_model("facebook/sam3")
    assert time.monotonic() - started < 6
    assert "facebook/sam3" not in sam_mod._sam3_cache


def test_env_policy_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv(bounded_load.TIMEOUT_ENV, "soon")
    monkeypatch.setenv(bounded_load.ATTEMPTS_ENV, "0")
    policy = LoadPolicy.from_env()
    assert policy.timeout_s == LoadPolicy().timeout_s
    assert policy.attempts == 1
