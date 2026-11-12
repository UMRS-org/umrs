"""Docling and GLiNER as default-on platform models (ADR-116 amendment, 2026-11-13).

Covers: the defaults and off values, bundled-means-no-mount in the ingest
sandbox, the readiness fallbacks, the understand runtime order (docmodels
service, then local, then text), the per-document budget, the tagging
deadline, the window clock, the ``umrs-docmodels`` server and its client.

Hermetic: fake models and loaders, the server through Starlette's
TestClient, the client through httpx.MockTransport; no network, no docling
or gliner import, no real weights.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from umrs.knowledge import (
    doc_parse,
    docmodels_client,
    docmodels_server,
    embed_sandbox,
    ingest_models,
    llm_extract,
    pretag,
)

_ENV = (
    "UMRS_KNOWLEDGE_DOC_PARSER",
    "UMRS_KNOWLEDGE_DOCLING_ARTIFACTS",
    "UMRS_KNOWLEDGE_DOCLING_ARTIFACTS_HOST_DIR",
    "UMRS_KNOWLEDGE_GLINER_MODEL",
    "UMRS_KNOWLEDGE_GLINER_MODEL_HOST_DIR",
    "UMRS_KNOWLEDGE_MODELS_HOST_DIR",
    "UMRS_KNOWLEDGE_DOCMODELS_URL",
    "UMRS_KNOWLEDGE_DOCMODELS_TIMEOUT_S",
    "UMRS_KNOWLEDGE_DOCLING_MAX_PAGES",
    "UMRS_KNOWLEDGE_DOCLING_MAX_BYTES",
    "UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK",
    "UMRS_KNOWLEDGE_DOCMODELS_TOKEN",
    "UMRS_DOCMODELS_TOKEN",
    "UMRS_EMBEDDING_SANDBOX",
    "UMRS_EMBEDDING_SANDBOX_IMAGE",
    "UMRS_EMBEDDING_SANDBOX_TIMEOUT",
)
_URL = "http://umrs-docmodels:8091"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def _staged_docling(path):
    for name in ingest_models.DOCLING_SUBDIRS:
        (path / name).mkdir(parents=True, exist_ok=True)
        (path / name / "model.bin").write_bytes(b"w")
    return path


def _staged_gliner(path):
    (path / "backbone").mkdir(parents=True, exist_ok=True)
    for name in ("gliner_config.json", "model.safetensors", "tokenizer_config.json"):
        (path / name).write_text("{}")
    (path / "backbone" / "config.json").write_text("{}")
    return path


def _pdf(pages: int) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _sandbox(monkeypatch) -> list[dict[str, Any]]:
    """Opt into the podman lane and record each worker launch."""
    monkeypatch.setenv("UMRS_EMBEDDING_SANDBOX", "podman")
    monkeypatch.setenv("UMRS_EMBEDDING_SANDBOX_IMAGE", "umrs-server:test")
    seen: list[dict[str, Any]] = []

    def runner(argv: list[str], *, input_text: str, timeout: float) -> Any:
        request = json.loads(input_text)
        seen.append({"argv": argv, "request": request, "timeout": timeout})
        reply = (
            {"text": "# T", "version": "2.9"}
            if request["op"] == "parse"
            else {"candidates": [[] for _ in request["texts"]]}
        )
        return SimpleNamespace(returncode=0, stdout=json.dumps(reply), stderr="")

    monkeypatch.setattr(embed_sandbox, "_default_runner", runner)
    return seen


# ── defaults: one source, on, bundled ───────────────────────────────────────


def test_doc_parse_and_pretag_read_the_defaults_from_ingest_models():
    """Three sources of truth before (doc_parse, pretag, ingest_models); the
    parser was off and GLiNER unset by default."""
    assert doc_parse.configured_parser() == "docling"
    assert doc_parse.artifacts_dir() == ingest_models.BUNDLED_DOCLING_DIR
    assert pretag.configured_model_dir() == ingest_models.BUNDLED_GLINER_DIR
    assert doc_parse.wants_docling("paper.pdf", "document")
    assert not hasattr(doc_parse, "DEFAULT_ARTIFACTS")


def test_off_values_switch_both_off(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOC_PARSER", "pypdf")
    monkeypatch.setenv("UMRS_KNOWLEDGE_GLINER_MODEL", "off")
    assert not doc_parse.wants_docling("paper.pdf", "document")
    notes: list[str] = []
    assert pretag.window_tagger(SimpleNamespace(gliner_labels=lambda: ["m"]), notes) is None
    assert pretag.pretag_windows(["text"], ["m"], notes) is None
    assert notes == []


# ── readiness: staged, not merely present ───────────────────────────────────


def test_an_unstaged_default_falls_back_by_name():
    """The bundled directories are absent on a test host (as in an image built
    with UMRS_BUNDLE_INGEST_MODELS=0): fall back, never fail."""
    parsed = doc_parse.parse_for_extraction(b"%PDF-1.7", "a.pdf", "document", fallback_text="t")
    assert (parsed.text, parsed.parser) == ("t", "pypdf")
    assert parsed.notes == ("doc_parser_fallback:artifacts_missing",)
    notes: list[str] = []
    tagger = pretag.window_tagger(SimpleNamespace(gliner_labels=lambda: ["m"]), notes)
    assert tagger is not None
    assert tagger(["a"]) == [[]] and tagger(["b"]) == [[]]
    assert notes == ["pretag_unavailable:model_dir_missing"]  # once, not per call


def test_a_partial_directory_is_not_staged(monkeypatch, tmp_path):
    """os.path.isdir passed an empty or half-copied directory to Docling."""
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCLING_ARTIFACTS", str(tmp_path))
    monkeypatch.setenv("UMRS_KNOWLEDGE_GLINER_MODEL", str(tmp_path))
    monkeypatch.setattr(doc_parse, "docling_convert", lambda *_a: ("never", "x"))
    parsed = doc_parse.parse_for_extraction(b"%PDF-1.7", "a.pdf", "document", fallback_text="t")
    assert parsed.notes == ("doc_parser_fallback:artifacts_missing",)
    notes: list[str] = []
    assert pretag.pretag_windows(["text"], ["m"], notes) is None
    assert notes == ["pretag_unavailable:model_dir_missing"]


# ── sandbox: a bundled model is not mounted ─────────────────────────────────


def test_a_bundled_path_needs_no_mount():
    assert doc_parse.sandbox_host_path("/opt/umrs/models/docling", "UNSET_X") == ""
    assert doc_parse.sandbox_mounts("/opt/umrs/models/docling", "UNSET_X", "/m") == (
        [],
        "/opt/umrs/models/docling",
    )
    assert doc_parse.sandbox_mounts("/srv/d", "UNSET_X", "/m") == ([("/srv/d", "/m")], "/m")


def test_a_bundled_docling_runs_in_the_sandbox_without_a_mount(monkeypatch):
    """The sandbox runs the same image: mounting /opt/umrs/models/docling from
    the host would name a host path that does not exist."""
    seen = _sandbox(monkeypatch)
    monkeypatch.setattr(ingest_models, "is_docling_staged", lambda _d: True)
    parsed = doc_parse.parse_for_extraction(b"%PDF-1.7", "a.pdf", "document", fallback_text="t")
    assert (parsed.text, parsed.parser) == ("# T", "docling@2.9")
    argv = seen[0]["argv"]
    assert "-v" not in argv
    assert seen[0]["request"]["artifacts_path"] == "/opt/umrs/models/docling"
    assert seen[0]["request"]["max_pages"] == 80


def test_a_bundled_gliner_runs_in_the_sandbox_without_a_mount(monkeypatch):
    seen = _sandbox(monkeypatch)
    monkeypatch.setattr(ingest_models, "is_gliner_staged", lambda _d: True)
    notes: list[str] = []
    assert pretag.pretag_windows(["text"], ["m"], notes) == [[]]
    assert "-v" not in seen[0]["argv"]
    assert seen[0]["request"]["model_dir"] == "/opt/umrs/models/gliner_medium-v2.1"
    assert notes == []


# ── runtime order: service, then local, then text ───────────────────────────


def _fake_service_parse(monkeypatch, outcome: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def parse(data: bytes, filename: str, timeout: float, **kw: Any) -> Any:
        calls.append({"filename": filename, "timeout": timeout, **kw})
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(docmodels_client, "parse", parse)
    return calls


def _local_docling(monkeypatch, tmp_path) -> list[str]:
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCLING_ARTIFACTS", str(_staged_docling(tmp_path)))
    local: list[str] = []
    monkeypatch.setattr(
        doc_parse, "docling_convert", lambda _d, name, *_a: local.append(name) or ("local", "2.1")
    )
    return local


def test_the_service_parses_first(monkeypatch, tmp_path):
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", _URL)
    local = _local_docling(monkeypatch, tmp_path)
    reply = docmodels_client.ParseReply(text="served", parser="docling@2.150.0", pages=3)
    calls = _fake_service_parse(monkeypatch, reply)
    parsed = doc_parse.parse_for_extraction(
        b"%PDF-1.7", "dir/a.pdf", "document", fallback_text="t", timeout_s=30
    )
    assert (parsed.text, parsed.parser, parsed.notes) == ("served", "docling@2.150.0", ())
    assert calls == [
        {
            "filename": "a.pdf",
            "timeout": 30,
            "max_pages": 80,
            "artifacts": str(tmp_path),
            "base_url": _URL,
        }
    ]
    assert local == []


def test_an_unavailable_service_falls_back_in_process_only_when_opted_in(monkeypatch, tmp_path):
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", _URL)
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK", "inprocess")
    local = _local_docling(monkeypatch, tmp_path)
    _fake_service_parse(monkeypatch, docmodels_client.DocmodelsUnavailable("connect_failed"))
    parsed = doc_parse.parse_for_extraction(b"%PDF-1.7", "a.pdf", "document", fallback_text="t")
    assert (parsed.text, parsed.parser) == ("local", "docling@2.1")
    assert parsed.notes == ("docmodels_unavailable:connect_failed",)
    assert local == ["a.pdf"]


def test_without_service_or_local_models_the_text_path_is_named(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", _URL)
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK", "inprocess")
    _fake_service_parse(monkeypatch, docmodels_client.DocmodelsUnavailable("busy"))
    parsed = doc_parse.parse_for_extraction(b"%PDF-1.7", "a.pdf", "document", fallback_text="t")
    assert (parsed.text, parsed.parser) == ("t", "pypdf")
    assert parsed.notes == ("docmodels_unavailable:busy", "doc_parser_fallback:artifacts_missing")


def _no_local_docling(monkeypatch, tmp_path) -> list[str]:
    """Staged weights (as the bundled ones are in every front image) and a
    record of every local Docling run, in the sandbox or in this process."""
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCLING_ARTIFACTS", str(_staged_docling(tmp_path)))
    local: list[str] = []
    monkeypatch.setattr(
        doc_parse, "_convert_bounded", lambda _d, name, *_a: local.append(name) or ("x", "1")
    )
    monkeypatch.setattr(
        doc_parse, "_docling_in_sandbox", lambda _c, _d, name, _a: local.append(name) or ("x", "1")
    )
    return local


def test_a_busy_service_never_loads_docling_in_the_server_by_default(monkeypatch, tmp_path):
    """Any service failure but too_many_pages ran Docling in the front server
    (host network, credentials, no memory limit) and kept it loaded."""
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", _URL)
    local = _no_local_docling(monkeypatch, tmp_path)
    for reason in ("busy", "loading", "connect_failed"):
        _fake_service_parse(monkeypatch, docmodels_client.DocmodelsUnavailable(reason))
        parsed = doc_parse.parse_for_extraction(b"%PDF-1.7", "a.pdf", "document", fallback_text="t")
        assert parsed.parser == "pypdf"
        assert parsed.notes == (
            f"docmodels_unavailable:{reason}",
            "doc_parser_fallback:no_local_fallback",
        )
    assert local == []


def test_a_busy_service_falls_back_to_the_sandbox_lane(monkeypatch, tmp_path):
    """The default allows the one-shot sandbox, which frees its memory."""
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", _URL)
    seen = _sandbox(monkeypatch)
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCLING_ARTIFACTS", str(_staged_docling(tmp_path)))
    _fake_service_parse(monkeypatch, docmodels_client.DocmodelsUnavailable("busy"))
    parsed = doc_parse.parse_for_extraction(b"%PDF-1.7", "a.pdf", "document", fallback_text="t")
    assert (parsed.text, parsed.parser) == ("# T", "docling@2.9")
    assert parsed.notes == ("docmodels_unavailable:busy",)
    assert [launch["request"]["op"] for launch in seen] == ["parse"]


@pytest.mark.parametrize(
    "reason", ["too_large", "parse_failed", "transport_error", "timeout", "bad_request"]
)
def test_a_document_the_service_refused_is_never_parsed_locally(monkeypatch, tmp_path, reason):
    """The service is the isolation boundary: a document it refused, failed on
    or died on was parsed again in the front server."""
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", _URL)
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK", "inprocess")
    local = _no_local_docling(monkeypatch, tmp_path)
    _fake_service_parse(monkeypatch, docmodels_client.DocmodelsUnavailable(reason))
    parsed = doc_parse.parse_for_extraction(b"%PDF-1.7", "a.pdf", "document", fallback_text="t")
    assert (parsed.text, parsed.parser) == ("t", "pypdf")
    assert parsed.notes == (f"doc_parser_fallback:{reason}",)
    assert local == []


def test_the_local_path_has_the_same_byte_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCLING_MAX_BYTES", "8")
    local = _local_docling(monkeypatch, tmp_path)
    parsed = doc_parse.parse_for_extraction(
        b"%PDF-1.7 0123456789", "a.pdf", "document", fallback_text="t"
    )
    assert parsed.notes == ("doc_parser_fallback:too_large",)
    assert local == []


def test_too_many_pages_at_the_service_skips_the_local_parse(monkeypatch, tmp_path):
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", _URL)
    local = _local_docling(monkeypatch, tmp_path)
    _fake_service_parse(monkeypatch, docmodels_client.DocmodelsUnavailable("too_many_pages"))
    parsed = doc_parse.parse_for_extraction(b"PK\x03\x04", "a.docx", "document", fallback_text="t")
    assert parsed.notes == ("doc_parser_fallback:too_many_pages",)
    assert local == []


def test_a_long_pdf_is_not_given_to_docling(monkeypatch, tmp_path):
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCLING_MAX_PAGES", "2")
    local = _local_docling(monkeypatch, tmp_path)
    parsed = doc_parse.parse_for_extraction(_pdf(3), "a.pdf", "document", fallback_text="t")
    assert parsed.notes == ("doc_parser_fallback:too_many_pages",)
    assert local == []
    short = doc_parse.parse_for_extraction(_pdf(2), "a.pdf", "document", fallback_text="t")
    assert short.text == "local"


def test_the_service_and_local_parse_share_one_budget(monkeypatch, tmp_path):
    """A service that times out leaves the local path only what is left."""
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(doc_parse, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", _URL)
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK", "inprocess")
    local = _local_docling(monkeypatch, tmp_path)

    def slow_service(*_a: Any, **_kw: Any) -> Any:
        clock.now += 30.0
        raise docmodels_client.DocmodelsUnavailable("busy")

    monkeypatch.setattr(docmodels_client, "parse", slow_service)
    parsed = doc_parse.parse_for_extraction(
        b"%PDF-1.7", "a.pdf", "document", fallback_text="t", timeout_s=30
    )
    assert parsed.notes == ("docmodels_unavailable:busy", "doc_parser_fallback:budget_exhausted")
    assert local == []


def _fake_service_tag(monkeypatch, outcome: Any) -> list[float]:
    timeouts: list[float] = []

    def tag(texts: list[str], labels: list[str], timeout: float, **_kw: Any) -> Any:
        timeouts.append(timeout)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(docmodels_client, "tag", tag)
    return timeouts


class _Model:
    def __init__(self, release: threading.Event | None = None) -> None:
        self.release = release

    def predict_entities(self, _text: str, labels: list[str], **_kw: Any) -> list:
        if self.release is not None:
            self.release.wait(5)
        return [{"text": "Zirconia", "label": labels[0], "score": 0.9}]


def test_the_service_tags_first_then_the_local_model(monkeypatch, tmp_path):
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", _URL)
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK", "inprocess")
    model_dir = str(_staged_gliner(tmp_path))
    monkeypatch.setenv("UMRS_KNOWLEDGE_GLINER_MODEL", model_dir)
    monkeypatch.setattr(pretag, "_MODELS", {model_dir: _Model()})
    served = [[{"text": "served", "label": "m", "score": 1.0}]]
    timeouts = _fake_service_tag(monkeypatch, served)
    notes: list[str] = []
    assert pretag.pretag_windows(["w"], ["m"], notes, 12.0) == served
    assert timeouts == [12.0] and notes == []
    _fake_service_tag(monkeypatch, docmodels_client.DocmodelsUnavailable("loading"))
    local = pretag.pretag_windows(["w"], ["m"], notes, 12.0)
    assert local == [[{"text": "Zirconia", "label": "m", "score": 0.9}]]
    assert notes == ["docmodels_unavailable:loading"]


def test_without_service_or_local_model_tagging_is_skipped_by_name(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", _URL)
    _fake_service_tag(monkeypatch, docmodels_client.DocmodelsUnavailable("connect_failed"))
    notes: list[str] = []
    assert pretag.pretag_windows(["w"], ["m"], notes) is None
    assert notes == [
        "docmodels_unavailable:connect_failed",
        "pretag_unavailable:no_local_fallback",
    ]
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK", "inprocess")
    notes = []
    assert pretag.pretag_windows(["w"], ["m"], notes) is None
    assert notes == [
        "docmodels_unavailable:connect_failed",
        "pretag_unavailable:model_dir_missing",
    ]


def _no_local_gliner(monkeypatch, tmp_path) -> list[str]:
    monkeypatch.setenv("UMRS_KNOWLEDGE_GLINER_MODEL", str(_staged_gliner(tmp_path)))
    loads: list[str] = []
    monkeypatch.setattr(pretag, "load_model", lambda d: loads.append(d) or _Model())
    monkeypatch.setattr(pretag, "_tag_in_sandbox", lambda *_a: loads.append("sandbox") or [[]])
    return loads


def test_a_busy_service_never_loads_gliner_in_the_server_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", _URL)
    loads = _no_local_gliner(monkeypatch, tmp_path)
    _fake_service_tag(monkeypatch, docmodels_client.DocmodelsUnavailable("busy"))
    notes: list[str] = []
    assert pretag.pretag_windows(["w"], ["m"], notes) is None
    assert notes == ["docmodels_unavailable:busy", "pretag_unavailable:no_local_fallback"]
    assert loads == []


@pytest.mark.parametrize("reason", ["tag_failed", "too_large", "transport_error"])
def test_texts_the_service_refused_are_never_tagged_locally(monkeypatch, tmp_path, reason):
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", _URL)
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK", "inprocess")
    loads = _no_local_gliner(monkeypatch, tmp_path)
    _fake_service_tag(monkeypatch, docmodels_client.DocmodelsUnavailable(reason))
    notes: list[str] = []
    assert pretag.pretag_windows(["w"], ["m"], notes) is None
    assert notes == [f"pretag_unavailable:{reason}"]
    assert loads == []


# ── budgets: tagging is bounded, the window clock starts after the hints ────


def test_in_process_tagging_is_bounded(monkeypatch, tmp_path):
    """Tagging had no timeout at all in process (runtime.md §4)."""
    model_dir = str(_staged_gliner(tmp_path))
    monkeypatch.setenv("UMRS_KNOWLEDGE_GLINER_MODEL", model_dir)
    release = threading.Event()
    monkeypatch.setattr(pretag, "_MODELS", {model_dir: _Model(release)})
    notes: list[str] = []
    started = time.monotonic()
    try:
        assert pretag.pretag_windows(["w"], ["m"], notes, timeout_s=0.2) is None
    finally:
        release.set()
    assert time.monotonic() - started < 2.0
    assert notes == ["pretag_unavailable:timeout"]


def test_sandbox_tagging_gets_the_budget_as_its_timeout(monkeypatch, tmp_path):
    """The sandbox used its full 600 s whatever was left of the job."""
    seen = _sandbox(monkeypatch)
    monkeypatch.setenv("UMRS_KNOWLEDGE_GLINER_MODEL", str(_staged_gliner(tmp_path)))
    notes: list[str] = []
    assert pretag.pretag_windows(["w"], ["m"], notes, timeout_s=7.0) == [[]]
    assert 0 < seen[0]["timeout"] <= 7.0


def test_the_window_tagger_is_bounded_by_the_job_deadline(monkeypatch, tmp_path):
    monkeypatch.setenv("UMRS_KNOWLEDGE_GLINER_MODEL", str(_staged_gliner(tmp_path)))
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_TIMEOUT_S", "120")
    budgets: list[float] = []
    monkeypatch.setattr(
        pretag, "pretag_windows", lambda _t, _l, _n, timeout_s=None: budgets.append(timeout_s)
    )
    deadline = time.monotonic() - 119.0  # one second left of deadline + T
    tagger = pretag.window_tagger(SimpleNamespace(gliner_labels=lambda: ["m"]), [], deadline)
    assert tagger is not None
    assert tagger(["w"]) == [[]]
    assert 0 < budgets[0] <= 1.0


def test_the_understand_job_passes_its_deadline_to_the_tagger(monkeypatch):
    from umrs.jobs import knowledge_adapters as ka

    seen: list[Any] = []
    monkeypatch.setattr(
        pretag, "window_tagger", lambda _v, _n, deadline=None: seen.append(deadline)
    )
    monkeypatch.setattr(llm_extract, "extract_document", lambda *_a: {"nodes": [], "edges": []})
    ka._windowed_extract("text", ("http://llm", "m"), [], "full", {"deadline": 123.0})
    assert seen == [123.0]


def test_the_parse_budget_is_outside_the_job_deadline(monkeypatch):
    """The parse got ``deadline - now`` (negative past the deadline); it now
    gets its own cap, until deadline + cap."""
    from umrs.jobs import knowledge_adapters as ka

    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_TIMEOUT_S", "120")
    late = ka._ExtractInputs(deadline=time.monotonic() - 20.0).bound()["timeout_s"]
    assert 99.0 < late <= 100.0
    early = ka._ExtractInputs(deadline=time.monotonic() + 1000.0).bound()["timeout_s"]
    assert early == 120.0
    assert ka._ExtractInputs().bound() == {}


def test_the_window_clock_starts_after_the_hints(monkeypatch):
    """A slow tagger ate the per-document window budget before any window ran."""
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(llm_extract, "time", SimpleNamespace(monotonic=lambda: clock.now))
    seen: list[float] = []

    def run_bounded(jobs: list[Any], _concurrency: int, timeout: float) -> tuple[list, int]:
        seen.append(timeout)
        return [], 0

    def slow_tagger(texts: list[str]) -> list[list[dict[str, Any]]]:
        clock.now += 100.0
        return [[] for _ in texts]

    monkeypatch.setattr(llm_extract, "run_bounded", run_bounded)
    request = llm_extract.ExtractRequest(
        chat=lambda *_a, **_kw: "{}",
        model="m",
        tagger=slow_tagger,
        settings=llm_extract.ExtractSettings(budget_s=50.0),
    )
    notes: list[str] = []
    assert llm_extract.extract_document("One sentence. Another sentence.", request, notes) is None
    assert seen == [50.0]


# ── the service ─────────────────────────────────────────────────────────────


def _fake_loaders(**overrides: Any) -> docmodels_server.Loaders:
    def convert(data: bytes, filename: str, artifacts: str, cap: int) -> tuple[str, str, int]:
        return f"# {filename} {len(data)}", "2.150.0", 3

    def tag(model: Any, texts: list[str], labels: list[str], threshold: float) -> list:
        return [[{"text": t, "label": labels[0], "score": threshold}] for t in texts]

    base: dict[str, Any] = {
        "warm_docling": lambda _d: object(),
        "load_gliner": lambda _d: "gliner-model",
        "convert": convert,
        "tag": tag,
        "docling_version": lambda: "2.150.0",
    }
    base.update(overrides)
    return docmodels_server.Loaders(**base)


def _app(tmp_path, *, loaders=None, limits=None, load=True, gliner=True, **kwargs):
    state = docmodels_server.ModelState(
        docling_dir=str(_staged_docling(tmp_path / "docling")),
        gliner_dir=str(_staged_gliner(tmp_path / "gliner")) if gliner else "",
    )
    loaders = loaders or _fake_loaders()
    kwargs.setdefault("token", "")
    app = docmodels_server.create_app(
        state=state,
        loaders=loaders,
        limits=limits or docmodels_server.Limits(),
        warm=False,
        **kwargs,
    )
    if load:
        docmodels_server.load_models(state, loaders)
    return app, state


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_readyz_is_503_while_loading_then_200(tmp_path):
    release = threading.Event()

    def slow_warm(_d: str) -> object:
        release.wait(5)
        return object()

    state = docmodels_server.ModelState(
        docling_dir=str(_staged_docling(tmp_path / "d")), gliner_dir=""
    )
    app = docmodels_server.create_app(
        state=state, loaders=_fake_loaders(warm_docling=slow_warm), warm=True
    )
    with TestClient(app) as client:  # the lifespan starts the warm load
        loading = client.get("/readyz")
        assert loading.status_code == 503 and loading.json()["error"] == "loading"
        assert client.post("/parse", json={"filename": "a.pdf", "data_b64": ""}).json() == {
            "error": "loading",
            "detail": "models are still loading",
        }
        release.set()
        assert state.done.wait(5)
        ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json() == {"ready": True, "docling": True, "gliner": False}


def test_readyz_is_503_when_no_model_loaded(tmp_path):
    def broken(_d: str) -> Any:
        raise OSError("no weights")

    state = docmodels_server.ModelState(
        docling_dir=str(tmp_path / "absent"), gliner_dir=str(tmp_path)
    )
    app = docmodels_server.create_app(
        state=state, loaders=_fake_loaders(load_gliner=broken), warm=False
    )
    docmodels_server.load_models(state, app.state.docmodels.loaders)
    reply = TestClient(app).get("/readyz")
    assert reply.status_code == 503 and reply.json()["error"] == "no_models"
    assert state.errors == {"docling": "not_staged", "gliner": "not_staged"}


def test_info_names_versions_dirs_and_limits(tmp_path):
    app, state = _app(tmp_path)
    body = TestClient(app).get("/info").json()
    assert body == {
        "service": "umrs-docmodels",
        "ready": True,
        "loading": False,
        "stuck": False,
        "docling": {
            "version": "2.150.0",
            "artifacts": state.docling_dir,
            "staged": True,
            "loaded": True,
        },
        "gliner": {
            "repo": "urchade/gliner_medium-v2.1",
            "revision": ingest_models.GLINER_REVISION,
            "dir": state.gliner_dir,
            "staged": True,
            "loaded": True,
        },
        "limits": {
            "max_bytes": 64 * 1024 * 1024,
            "max_pages": 80,
            "concurrency": 1,
            "max_run_s": 300.0,
        },
    }


def test_info_reports_what_loaded_not_only_what_is_staged(tmp_path):
    """/info said staged:true while loading and after a failed load, so
    connect-info reported the service ready while every parse fell back."""

    def broken(_d: str) -> Any:
        raise ImportError("onnxruntime")

    app, state = _app(tmp_path, loaders=_fake_loaders(warm_docling=broken), load=False)
    client = TestClient(app)
    loading = client.get("/info").json()
    assert (loading["ready"], loading["loading"], loading["docling"]["staged"]) == (
        False,
        True,
        True,
    )
    docmodels_server.load_models(state, app.state.docmodels.loaders)
    failed = client.get("/info").json()
    assert failed["docling"] == {**failed["docling"], "staged": True, "loaded": False}
    assert failed["gliner"]["loaded"] is True and failed["ready"] is True


def test_parse_and_tag(tmp_path):
    app, _state = _app(tmp_path)
    client = TestClient(app)
    parsed = client.post("/parse", json={"filename": "x/a.pdf", "data_b64": _b64(b"%PDF-1")})
    assert parsed.status_code == 200
    assert parsed.json() == {"text": "# a.pdf 6", "parser": "docling@2.150.0", "pages": 3}
    tagged = client.post("/tag", json={"texts": ["w1", "w2"], "labels": ["m"], "threshold": 0.4})
    assert tagged.json() == {
        "candidates": [
            [{"text": "w1", "label": "m", "score": 0.4}],
            [{"text": "w2", "label": "m", "score": 0.4}],
        ]
    }


@pytest.mark.parametrize(
    "body",
    [
        {"filename": "a.pdf"},
        {"filename": "a.pdf", "data_b64": "not base64!"},
        ["not", "an", "object"],
    ],
)
def test_a_bad_parse_request_is_400(tmp_path, body):
    app, _state = _app(tmp_path)
    reply = TestClient(app).post("/parse", json=body)
    assert reply.status_code == 400 and reply.json()["error"] == "bad_request"


def test_a_body_over_the_cap_is_413(tmp_path):
    app, _state = _app(tmp_path, limits=docmodels_server.Limits(max_bytes=100))
    reply = TestClient(app).post("/parse", json={"filename": "a.pdf", "data_b64": _b64(b"x" * 200)})
    assert reply.status_code == 413 and reply.json()["error"] == "too_large"


def test_a_pdf_over_the_page_cap_is_too_many_pages(tmp_path):
    app, _state = _app(tmp_path, limits=docmodels_server.Limits(max_pages=2))
    client = TestClient(app)
    reply = client.post("/parse", json={"filename": "a.pdf", "data_b64": _b64(_pdf(3))})
    assert reply.status_code == 422 and reply.json()["error"] == "too_many_pages"
    lowered = {"filename": "a.pdf", "data_b64": _b64(_pdf(2)), "max_pages": 1}
    assert client.post("/parse", json=lowered).json()["error"] == "too_many_pages"


def test_a_saturated_service_is_busy(tmp_path):
    limits = docmodels_server.Limits(concurrency=1, queue_wait_s=0.05)
    app, _state = _app(tmp_path, limits=limits)
    slots = app.state.docmodels.slots["tag"]
    assert slots.acquire(timeout=1)
    try:
        reply = TestClient(app).post("/tag", json={"texts": ["w"], "labels": ["m"]})
    finally:
        slots.release()
    assert reply.status_code == 503 and reply.json()["error"] == "busy"


def test_a_model_that_did_not_load_is_named(tmp_path):
    app, _state = _app(tmp_path, gliner=False)
    reply = TestClient(app).post("/tag", json={"texts": ["w"], "labels": ["m"]})
    assert reply.status_code == 503
    assert reply.json() == {"error": "gliner_unavailable", "detail": "off"}


def test_a_failing_parse_is_a_typed_500(tmp_path):
    def broken(*_a: Any) -> Any:
        raise RuntimeError("layout model crashed")

    app, _state = _app(tmp_path, loaders=_fake_loaders(convert=broken))
    reply = TestClient(app).post("/parse", json={"filename": "a.pdf", "data_b64": _b64(b"%PDF")})
    assert reply.status_code == 500
    assert reply.json() == {"error": "parse_failed", "detail": "RuntimeError"}


def test_the_service_forces_the_hub_offline():
    env = {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
    docmodels_server.enforce_offline_env(env)
    assert env == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": "/tmp/hf"}


def test_main_serves_on_8091_with_a_graceful_shutdown(monkeypatch):
    import uvicorn

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME"):
        monkeypatch.setenv(name, "0")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw, app=app))
    assert docmodels_server.main() == 0
    assert (seen["host"], seen["port"]) == ("0.0.0.0", 8091)
    assert seen["timeout_graceful_shutdown"] == docmodels_server.GRACEFUL_S
    assert seen["app"].state.docmodels.warm is True


# ── the client ──────────────────────────────────────────────────────────────


def _mock_client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_the_client_retries_a_refused_connection_once():
    calls: list[str] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(docmodels_client.DocmodelsUnavailable) as err:
        docmodels_client.tag(["w"], ["m"], 1.0, base_url=_URL, client=_mock_client(refuse))
    assert err.value.reason == "connect_failed" and calls == ["/tag", "/tag"]

    def flaky(request: httpx.Request) -> httpx.Response:
        calls.append("again")
        if len(calls) == 3:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"candidates": [[]]})

    again = docmodels_client.tag(["w"], ["m"], 1.0, base_url=_URL, client=_mock_client(flaky))
    assert again == [[]]


def test_a_timeout_is_not_retried():
    calls: list[int] = []

    def slow(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(docmodels_client.DocmodelsUnavailable) as err:
        docmodels_client.parse(b"%PDF", "a.pdf", 1.0, base_url=_URL, client=_mock_client(slow))
    assert err.value.reason == "timeout" and calls == [1]


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx.Response(503, json={"error": "busy"}), "busy"),
        (httpx.Response(413, json={"error": "too_large"}), "too_large"),
        (httpx.Response(422, json={"error": "too_many_pages"}), "too_many_pages"),
        (httpx.Response(500, text="<html>oops</html>"), "http_500"),
        (httpx.Response(200, json={"text": 3}), "bad_reply"),
        (httpx.Response(200, text="not json"), "bad_reply"),
    ],
)
def test_client_error_mapping(response, reason):
    client = _mock_client(lambda _request: response)
    with pytest.raises(docmodels_client.DocmodelsUnavailable) as err:
        docmodels_client.parse(b"%PDF", "a.pdf", 1.0, base_url=_URL, client=client)
    assert err.value.reason == reason


def test_the_client_needs_a_url():
    with pytest.raises(docmodels_client.DocmodelsUnavailable) as err:
        docmodels_client.parse(b"%PDF", "a.pdf", 1.0)
    assert err.value.reason == "not_configured"


def test_the_understand_path_parses_and_tags_through_the_real_server(monkeypatch, tmp_path):
    """doc_parse and pretag -> docmodels_client -> docmodels_server (fake models)."""
    app, state = _app(tmp_path / "svc")
    monkeypatch.setattr(docmodels_client, "_new_client", lambda _timeout: TestClient(app))
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", "http://testserver")
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCLING_ARTIFACTS", state.docling_dir)
    monkeypatch.setenv("UMRS_KNOWLEDGE_GLINER_MODEL", state.gliner_dir)
    parsed = doc_parse.parse_for_extraction(b"%PDF-1.7", "a.pdf", "document", fallback_text="t")
    assert (parsed.text, parsed.parser, parsed.notes) == ("# a.pdf 8", "docling@2.150.0", ())
    notes: list[str] = []
    tagged = pretag.pretag_windows(["w"], ["m"], notes)
    assert tagged == [[{"text": "w", "label": "m", "score": 0.5}]] and notes == []
    assert docmodels_client.info(2.0)["service"] == "umrs-docmodels"


# ── review fixes (2026-11-13): deadlines, admission, token, model identity ──


def _held_convert() -> tuple[threading.Event, list[str], Any]:
    """A convert whose first call blocks until released."""
    hold = threading.Event()
    calls: list[str] = []

    def convert(data: bytes, filename: str, _d: str, _cap: int) -> tuple[str, str, int]:
        calls.append(filename)
        if len(calls) == 1:
            hold.wait(10)
        return f"# {filename}", "2.150.0", 1

    return hold, calls, convert


def _wait_for(predicate: Any, seconds: float = 5.0) -> None:
    stop = time.monotonic() + seconds
    while not predicate() and time.monotonic() < stop:
        time.sleep(0.01)
    assert predicate()


_DOCX = {"filename": "a.docx", "data_b64": base64.b64encode(b"PK\x03\x04x").decode()}


def _hold_one_parse(client: TestClient, service: Any, budget: str = "60") -> threading.Thread:
    worker = threading.Thread(
        target=lambda: client.post("/parse", json=_DOCX, headers={"X-Docmodels-Budget": budget})
    )
    worker.start()
    _wait_for(lambda: bool(service.running))
    return worker


def _assert_stuck(client: TestClient) -> None:
    for path in ("/readyz", "/livez"):
        reply = client.get(path)
        assert reply.status_code == 503 and reply.json()["error"] == "stuck", path


def test_a_request_past_its_deadline_frees_its_slot_and_reports_stuck(tmp_path):
    """No server-side deadline: one slow document held the only slot for as
    long as it ran, every front got busy, and /readyz stayed 200."""
    hold, _calls, convert = _held_convert()
    stuck: list[str] = []
    limits = docmodels_server.Limits(queue_wait_s=0.05)
    app, _state = _app(
        tmp_path, loaders=_fake_loaders(convert=convert), limits=limits, on_stuck=stuck.append
    )
    service = app.state.docmodels
    client = TestClient(app)
    worker = _hold_one_parse(client, service, budget="1")
    try:
        assert client.post("/parse", json=_DOCX).json()["error"] == "busy"
        assert not service.check_stuck()  # within its deadline
        assert client.get("/readyz").status_code == 200
        assert service.check_stuck(now=time.monotonic() + 100)
        assert stuck == ["parse"]
        _assert_stuck(client)
        assert client.post("/parse", json=_DOCX).status_code == 200  # the slot is free
    finally:
        hold.set()
        worker.join(5)
    assert not service.running


def test_the_watchdog_ends_a_stuck_service(monkeypatch, tmp_path):
    """The lifespan's watchdog finds the overrun by itself."""
    monkeypatch.setattr(docmodels_server, "WATCH_INTERVAL_S", 0.02)
    monkeypatch.setattr(docmodels_server, "GRACE_S", 0.0)
    hold, _calls, convert = _held_convert()
    stuck = threading.Event()
    app, _state = _app(
        tmp_path, loaders=_fake_loaders(convert=convert), on_stuck=lambda _op: stuck.set()
    )
    with TestClient(app) as client:
        worker = _hold_one_parse(client, app.state.docmodels, budget="0.1")
        try:
            assert stuck.wait(5)
            assert client.get("/livez").status_code == 503
        finally:
            hold.set()
            worker.join(5)


def test_the_default_stuck_action_exits_for_a_restart(monkeypatch):
    codes: list[int] = []
    monkeypatch.setattr(docmodels_server.os, "_exit", codes.append)
    docmodels_server.exit_for_restart("parse")
    assert codes == [docmodels_server.STUCK_EXIT_CODE]


def test_parse_and_tag_have_their_own_slots(tmp_path):
    """One long parse made every /tag answer busy."""
    hold, _calls, convert = _held_convert()
    limits = docmodels_server.Limits(queue_wait_s=0.05)
    app, _state = _app(tmp_path, loaders=_fake_loaders(convert=convert), limits=limits)
    client = TestClient(app)
    worker = _hold_one_parse(client, app.state.docmodels)
    try:
        assert client.post("/tag", json={"texts": ["w"], "labels": ["m"]}).status_code == 200
    finally:
        hold.set()
        worker.join(5)


def test_the_slot_wait_is_bounded_by_the_callers_budget(tmp_path):
    hold, _calls, convert = _held_convert()
    limits = docmodels_server.Limits(queue_wait_s=30.0)
    app, _state = _app(tmp_path, loaders=_fake_loaders(convert=convert), limits=limits)
    client = TestClient(app)
    worker = _hold_one_parse(client, app.state.docmodels)
    try:
        started = time.monotonic()
        reply = client.post("/parse", json=_DOCX, headers={"X-Docmodels-Budget": "0.2"})
        assert reply.json()["error"] == "busy" and time.monotonic() - started < 3.0
    finally:
        hold.set()
        worker.join(5)


def test_requests_past_the_admission_cap_are_busy_at_once(tmp_path):
    """Every waiting request held its whole body in memory, uncounted."""
    hold, _calls, convert = _held_convert()
    limits = docmodels_server.Limits(queue_wait_s=30.0, max_requests=1)
    app, _state = _app(tmp_path, loaders=_fake_loaders(convert=convert), limits=limits)
    client = TestClient(app)
    worker = _hold_one_parse(client, app.state.docmodels)
    try:
        started = time.monotonic()
        reply = client.post("/parse", json=_DOCX)
        assert reply.json()["error"] == "busy" and time.monotonic() - started < 3.0
    finally:
        hold.set()
        worker.join(5)


def test_the_page_count_does_not_block_the_event_loop(monkeypatch, tmp_path):
    """json.loads, base64 and pypdf ran on the event loop, before any slot."""
    entered, release = threading.Event(), threading.Event()

    def slow_count(_data: bytes, _filename: str) -> int:
        entered.set()
        release.wait(10)
        return 1

    monkeypatch.setattr(doc_parse, "pdf_page_count", slow_count)
    app, _state = _app(tmp_path)
    client = TestClient(app)
    body = {"filename": "a.pdf", "data_b64": _b64(b"%PDF-1")}
    worker = threading.Thread(target=lambda: client.post("/parse", json=body))
    worker.start()
    try:
        assert entered.wait(5)
        started = time.monotonic()
        assert client.get("/readyz").status_code == 200
        assert time.monotonic() - started < 2.0
    finally:
        release.set()
        worker.join(5)


def test_a_document_over_the_shared_byte_cap_is_413(tmp_path):
    """The body cap is the document cap after base64, so a 48-64 MiB PDF is
    no longer refused by the service and parsed in the front instead."""
    limits = docmodels_server.Limits(max_bytes=1000)
    assert limits.body_bytes > 1000 * 4 // 3
    app, _state = _app(tmp_path, limits=limits)
    client = TestClient(app)
    fits = client.post("/parse", json={"filename": "a.docx", "data_b64": _b64(b"P" * 1000)})
    assert fits.status_code == 200
    over = client.post("/parse", json={"filename": "a.docx", "data_b64": _b64(b"P" * 1001)})
    assert over.status_code == 413 and over.json()["error"] == "too_large"


def test_the_limits_read_the_settings_the_servers_read(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCLING_MAX_PAGES", "12")
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCLING_MAX_BYTES", "4096")
    limits = docmodels_server.Limits.from_env()
    assert (limits.max_pages, limits.max_bytes) == (12, 4096)


def test_a_token_guards_parse_tag_and_info(tmp_path):
    """The loopback port took any local caller's documents unauthenticated."""
    app, _state = _app(tmp_path, token="s3cret-token")
    client = TestClient(app)
    tag = {"texts": ["w"], "labels": ["m"]}
    for method, path, body in (("post", "/tag", tag), ("get", "/info", None)):
        call = getattr(client, method)
        kwargs = {"json": body} if body is not None else {}
        denied = call(path, **kwargs)
        assert denied.status_code == 401 and denied.json()["error"] == "unauthorized"
        wrong = call(path, headers={"Authorization": "Bearer nope"}, **kwargs)
        assert wrong.status_code == 401
        ok = call(path, headers={"Authorization": "Bearer s3cret-token"}, **kwargs)
        assert ok.status_code == 200
    assert client.get("/readyz").status_code == 200 and client.get("/livez").status_code == 200


def test_the_proof_answers_a_nonce_with_the_token_hmac(tmp_path):
    app, _state = _app(tmp_path, token="s3cret-token")
    nonce = "n" * 32
    reply = TestClient(app).get("/proof", headers={"X-Docmodels-Nonce": nonce})
    assert reply.json() == {"proof": docmodels_client.proof("s3cret-token", nonce)}
    open_app, _state = _app(tmp_path / "open")
    no_token = TestClient(open_app).get("/proof", headers={"X-Docmodels-Nonce": nonce})
    assert no_token.status_code == 404


def test_the_client_sends_nothing_to_a_listener_without_the_token(monkeypatch):
    """A process squatting on the port while the service was down got every
    document and could answer with any text."""
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_TOKEN", "s3cret-token")
    seen: list[tuple[str, str]] = []

    def squatter(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("authorization", "")))
        if request.url.path == "/proof":
            return httpx.Response(200, json={"proof": "0" * 64})
        return httpx.Response(200, json={"text": "injected", "parser": "docling@x"})

    with pytest.raises(docmodels_client.DocmodelsUnavailable) as err:
        docmodels_client.parse(b"%PDF", "a.pdf", 1.0, base_url=_URL, client=_mock_client(squatter))
    assert err.value.reason == "untrusted"
    assert seen == [("/proof", "")]


def test_the_client_proves_and_authorizes_against_the_real_server(monkeypatch, tmp_path):
    app, state = _app(tmp_path / "svc", token="s3cret-token")
    monkeypatch.setattr(docmodels_client, "_new_client", lambda _timeout: TestClient(app))
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_TOKEN", "s3cret-token")
    reply = docmodels_client.parse(
        b"%PDF-1.7", "a.pdf", 5.0, artifacts=state.docling_dir, base_url="http://testserver"
    )
    assert reply.text == "# a.pdf 8"
    assert docmodels_client.info(2.0, base_url="http://testserver")["ready"] is True
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_TOKEN", "other")
    with pytest.raises(docmodels_client.DocmodelsUnavailable) as err:
        docmodels_client.info(2.0, base_url="http://testserver")
    assert err.value.reason == "untrusted"


def test_a_front_set_to_another_model_is_not_served_the_shared_one(monkeypatch, tmp_path):
    """A per-front GLINER_MODEL / DOCLING_ARTIFACTS was silently ignored while
    the service answered with its own model."""
    app, state = _app(tmp_path / "svc")
    client = TestClient(app)
    tag = {"texts": ["w"], "labels": ["m"], "model_dir": "/models/other"}
    assert client.post("/tag", json=tag).status_code == 409
    same = {**tag, "model_dir": state.gliner_dir + "/"}
    assert client.post("/tag", json=same).status_code == 200
    monkeypatch.setattr(docmodels_client, "_new_client", lambda _timeout: client)
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_URL", "http://testserver")
    monkeypatch.setenv("UMRS_KNOWLEDGE_GLINER_MODEL", str(_staged_gliner(tmp_path / "own")))
    notes: list[str] = []
    assert pretag.pretag_windows(["w"], ["m"], notes) is None
    assert notes == [
        "docmodels_unavailable:model_mismatch",
        "pretag_unavailable:no_local_fallback",
    ]


class _FakeConverter:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.initialized: list[Any] = []
        self.converted: list[dict[str, Any]] = []

    def initialize_pipeline(self, fmt: Any) -> None:
        if self.fail:
            raise OSError("corrupt layout weights")
        self.initialized.append(fmt)

    def convert(self, _stream: Any, **kwargs: Any) -> Any:
        self.converted.append(kwargs)
        document = SimpleNamespace(export_to_markdown=lambda: "# md", num_pages=lambda: 2)
        return SimpleNamespace(document=document)


def _fake_docling(monkeypatch, converter: _FakeConverter) -> None:
    monkeypatch.setattr(doc_parse, "_CONVERTERS", {})
    monkeypatch.setattr(doc_parse, "_new_converter", lambda _path: converter)
    monkeypatch.setattr(doc_parse, "_pdf_format", lambda: "PDF")
    monkeypatch.setattr(doc_parse, "_document_stream", lambda name, _data: name)
    monkeypatch.setattr(doc_parse, "set_offline_env", lambda: None)


def test_the_warm_load_loads_the_pdf_pipeline(monkeypatch):
    """DocumentConverter() loads no model: /readyz said docling ready before
    the layout model was loaded, and broken weights failed every parse."""
    converter = _FakeConverter()
    _fake_docling(monkeypatch, converter)
    assert doc_parse.warm_converter("/d") is converter
    assert converter.initialized == ["PDF"]


def test_a_warm_load_failure_is_load_failed(monkeypatch, tmp_path):
    _fake_docling(monkeypatch, _FakeConverter(fail=True))
    state = docmodels_server.ModelState(
        docling_dir=str(_staged_docling(tmp_path / "d")), gliner_dir=""
    )
    loaders = _fake_loaders(warm_docling=doc_parse.warm_converter)
    docmodels_server.load_models(state, loaders)
    assert state.docling is False
    assert state.errors["docling"].startswith("load_failed")


def test_the_page_cap_applies_to_pdfs_only(monkeypatch):
    """Docling counts PPTX slides as pages: a deck over 80 slides was refused
    (twice) after parsing fine at the base commit."""
    converter = _FakeConverter()
    _fake_docling(monkeypatch, converter)
    doc_parse.docling_parse(b"PK", "deck.pptx", "/d", 80)
    doc_parse.docling_parse(b"%PDF", "paper.pdf", "/d", 80)
    assert converter.converted == [{}, {"max_num_pages": 80}]


def test_tagging_that_crosses_the_job_deadline_still_reads_one_window(monkeypatch):
    """Tagging may run past the job deadline (deadline + T); the windows then
    got a negative budget and the document no graph at all."""
    monkeypatch.setattr(llm_extract, "LITE_FLOOR_S", 0.3)
    reply = json.dumps({"nodes": [{"name": "Zirconia", "type": "Material"}], "edges": []})

    def slow_tagger(texts: list[str]) -> list[list[dict[str, Any]]]:
        time.sleep(0.8)
        return [[] for _ in texts]

    def chat(*_a: Any, **_kw: Any) -> str:
        time.sleep(0.2)
        return reply

    request = llm_extract.ExtractRequest(
        chat=chat, model="m", tagger=slow_tagger, deadline=time.monotonic() + 0.6
    )
    notes: list[str] = []
    payload = llm_extract.extract_document("Zirconia is a ceramic. " * 400, request, notes)
    assert payload is not None and payload["nodes"]
    assert "extract_job_budget_exhausted" in notes
