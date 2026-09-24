"""D-12 move leg: KNOWLEDGE_EMBED compute in the ``embedding-job`` sandbox
(ADR-061) — isolation flags, JSON contract, named failures, site opt-in."""

from __future__ import annotations

import io
import json
import subprocess
from types import SimpleNamespace

import pytest
from umrs.knowledge import embed_sandbox as es
from umrs.knowledge import embed_worker as ew


class _FakeEmbedder:
    client = SimpleNamespace(tokenizer=None)

    def embed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


CONFIG = es.SandboxConfig(image="localhost/umrs-server:abc", model_host_dir="/srv/models/e5")


def test_argv_carries_the_profile_isolation():
    argv = es.build_argv(CONFIG)
    isolation = {
        "--network=none",
        "--read-only",
        "--security-opt=no-new-privileges",
        "--cap-drop=all",
        "--rm",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
    }
    assert isolation <= set(argv)
    assert argv[argv.index("-v") + 1] == "/srv/models/e5:/model:ro"
    assert argv[-4:] == ["localhost/umrs-server:abc", "python", "-m", "umrs.knowledge.embed_worker"]
    assert not [a for a in argv if "AWS_" in a or "DATABASE" in a]  # no credentials cross


def test_config_is_a_site_opt_in(monkeypatch):
    monkeypatch.delenv(es.SANDBOX_ENV, raising=False)
    assert es.sandbox_config_from_env("/m") is None
    monkeypatch.setenv(es.SANDBOX_ENV, "podman")
    monkeypatch.delenv(es.IMAGE_ENV, raising=False)
    monkeypatch.delenv("UMRS_STEP_IMAGE", raising=False)
    with pytest.raises(es.EmbeddingSandboxError, match="no image"):
        es.sandbox_config_from_env("/m")
    monkeypatch.setenv(es.IMAGE_ENV, "img")
    monkeypatch.setenv(es.MODEL_HOST_DIR_ENV, "/host/m")
    cfg = es.sandbox_config_from_env("/container/m")
    assert (cfg.image, cfg.model_host_dir) == ("img", "/host/m")


def _runner(stdout, code=0, capture=None):
    def run(argv, *, input_text, timeout):
        if capture is not None:
            capture.update(argv=argv, request=json.loads(input_text), timeout=timeout)
        return SimpleNamespace(stdout=stdout, stderr="boom-trace", returncode=code)

    return run


def test_round_trip_and_request_shape():
    seen = {}
    reply = json.dumps({"results": [{"chunks": [{"text": "a"}], "vectors": [[0.1]]}]})
    emb = es.SandboxChunkEmbedder(
        CONFIG, prefix_model_name="/m/e5", max_tokens=64, runner=_runner(reply, capture=seen)
    )
    assert emb.chunk_and_embed(["hello"]) == [{"chunks": [{"text": "a"}], "vectors": [[0.1]]}]
    assert seen["request"] == {
        "model_dir": "/model",
        "prefix_model_name": "/m/e5",
        "max_tokens": 64,
        "texts": ["hello"],
    }


@pytest.mark.parametrize(
    ("stdout", "code", "match"),
    [
        ("not json", 1, "unreadable reply"),
        (json.dumps({"error": "OSError: no model"}), 1, "OSError: no model"),
        (json.dumps({"results": []}), 0, "0 results for 1 texts"),
    ],
)
def test_failures_are_named_never_masked(stdout, code, match):
    emb = es.SandboxChunkEmbedder(
        CONFIG, prefix_model_name="", max_tokens=8, runner=_runner(stdout, code)
    )
    with pytest.raises(es.EmbeddingSandboxError, match=match):
        emb.chunk_and_embed(["x"])


def test_timeout_is_named():
    def slow(argv, *, input_text, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    emb = es.SandboxChunkEmbedder(CONFIG, prefix_model_name="", max_tokens=8, runner=slow)
    with pytest.raises(es.EmbeddingSandboxError, match="timed out"):
        emb.chunk_and_embed(["x"])


def test_worker_contract_over_stdin_stdout(monkeypatch):
    monkeypatch.setattr(ew, "_load_embedder", lambda _d: _FakeEmbedder())
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps({"model_dir": "/model", "texts": ["abc def", ""], "max_tokens": 16})
        ),
    )
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    assert ew.main() == 0
    results = json.loads(out.getvalue())["results"]
    assert len(results) == 2
    assert results[0]["chunks"] and len(results[0]["vectors"]) == len(results[0]["chunks"])
    assert results[1] == {"chunks": [], "vectors": []}  # empty text embeds nothing


def test_worker_reports_errors(monkeypatch):
    def broken(_d):
        raise OSError("model dir missing")

    monkeypatch.setattr(ew, "_load_embedder", broken)
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps({"model_dir": "/model", "texts": ["a"]}))
    )
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    assert ew.main() == 1
    assert "OSError: model dir missing" in json.loads(out.getvalue())["error"]


def test_embed_job_picks_the_sandbox_lane_without_loading_the_model(monkeypatch):
    from umrs.jobs import knowledge_adapters as ka

    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", "/srv/models/e5")
    monkeypatch.setenv(es.SANDBOX_ENV, "podman")
    monkeypatch.setenv(es.IMAGE_ENV, "img")
    monkeypatch.setattr(ka, "_get_node_embedder", lambda: pytest.fail("server loaded the model"))
    emb, reason = ka._chunk_embedder_for_site()
    assert (emb.lane, reason) == ("sandbox", "")


def test_embed_job_default_lane_is_in_process(monkeypatch):
    from umrs.jobs import knowledge_adapters as ka

    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", "/srv/models/e5")
    monkeypatch.delenv(es.SANDBOX_ENV, raising=False)
    monkeypatch.setattr(ka, "_get_node_embedder", lambda: _FakeEmbedder())
    emb, _ = ka._chunk_embedder_for_site()
    assert emb.lane == "in_process"


@pytest.mark.asyncio
async def test_failed_batch_fails_its_artifacts_by_name(monkeypatch):
    from umrs.jobs import knowledge_adapters as ka
    from umrs.jobs.knowledge_adapters import KnowledgeEmbedResult

    monkeypatch.setattr(ka, "_fetch_artifact_bytes", lambda *_a, **_k: b"some text")

    class _Broken:
        lane = "sandbox"

        def chunk_and_embed(self, _texts):
            raise es.EmbeddingSandboxError("sandbox exit=137")

    result = KnowledgeEmbedResult()
    sources = [{"artifact_id": "a1", "source_ref": "k1"}, {"artifact_id": "a2", "source_ref": "k2"}]
    await ka._embed_sources(object(), sources, "p", "", _Broken(), (None, "b"), result)
    assert [f["artifact_id"] for f in result.failed_items] == ["a1", "a2"]
    assert all("exit=137" in f["error"] for f in result.failed_items)
