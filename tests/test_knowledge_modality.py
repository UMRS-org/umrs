"""Each artifact reaches the model for its modality, or fails by name.

Maintainer decision 2026-10-09: text embedding works out of the box (the
umrs-server image bundles e5-base-v2), and an upload whose modality has no
embedder — an image, with only a text model — is flagged as a failure. Before,
every upload was filed as a text "document": an image's bytes were decoded as
UTF-8 and embedded as noise, and the understand stage fed them to the parser
and the LLM. A PDF went the same way, although a text extractor exists.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from umrs.jobs import knowledge_adapters as ka

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 64
_PDF = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\n"


# ── ingest: the kind follows the modality ───────────────────────────────────


@pytest.mark.parametrize(
    ("name", "content", "kind"),
    [
        ("photo.png", _PNG, "image"),
        ("scan.tiff", b"II*\x00" + b"\x00" * 32, "image"),
        ("clip.mp4", b"\x00\x00\x00\x18ftypmp42", "video"),
        ("tone.wav", b"RIFF$\x00\x00\x00WAVEfmt ", "audio"),
        ("blob.bin", b"\x00\x01\x02\xff" * 20, "binary"),
        ("unnamed", _PNG, "image"),
        ("note.md", b"# Note\nGraphene.", "document"),
        ("paper.pdf", _PDF, "document"),
        ("fit.py", b"def fit():\n    pass\n", "code"),
        # Legacy-encoded text is text, not a binary blob.
        ("legacy.txt", "café crème".encode("latin-1"), "document"),
    ],
)
def test_artifact_kind_follows_the_modality(name, content, kind):
    assert ka._artifact_kind(name, content) == kind


def test_an_uploaded_image_is_registered_as_an_image(monkeypatch):
    monkeypatch.setattr(ka, "_bridge_a2l_ref", lambda *_a, **_k: None)

    class _S3:
        exceptions = SimpleNamespace(ClientError=KeyError)

        def __init__(self):
            self.objects = {("r4-data", "knowledge-uploads/p/u/photo.png"): _PNG}

        def get_object(self, Bucket, Key):  # noqa: N803 — boto3 signature
            import io

            return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

        def head_object(self, **_kw):
            raise KeyError("404")

        def put_object(self, Bucket, Key, Body, **_kw):  # noqa: N803
            self.objects[(Bucket, Key)] = Body

    result = ka.KnowledgeIngestResult()
    ids: list = []
    descs: list = []
    ka._ingest_one_source(
        SimpleNamespace(job_id="j", spec={}),
        SimpleNamespace(attempt_id="a"),
        _S3(),
        "kb",
        "p",
        "s3_uri",
        "s3://r4-data/knowledge-uploads/p/u/photo.png",
        [],
        result,
        ids,
        descs,
    )
    assert descs[0]["kind"] == "image"


# ── the text an artifact offers ─────────────────────────────────────────────


def test_non_text_modalities_offer_no_text():
    assert ka._text_content({"kind": "image"}, _PNG) == (None, "image")
    assert ka._text_content({"kind": "document"}, _PNG) == (None, "image")  # older rows


def test_a_pdf_offers_its_extracted_text(monkeypatch):
    from umrs.knowledge import extract

    monkeypatch.setattr(extract, "resolve_extraction_content", lambda _n, _c: b"Graphene abstract")
    assert ka._text_content({"kind": "document"}, _PDF) == (b"Graphene abstract", "")


def test_a_pdf_without_extractable_text_says_so(monkeypatch):
    from umrs.knowledge import extract

    monkeypatch.setattr(extract, "resolve_extraction_content", lambda _n, c: c)
    text, reason = ka._text_content({"kind": "document"}, _PDF)
    assert text is None and "no extractable text" in reason


def test_text_passes_through():
    note = b"# Note\nGraphene."
    assert ka._text_content({"kind": "document"}, note) == (note, "")


# ── understand ──────────────────────────────────────────────────────────────


def _understand(monkeypatch, src, content):
    import asyncio

    async def no_ir(_store, _aid):
        return None

    parsed = []
    monkeypatch.setattr(ka, "_existing_current_ir", no_ir)
    monkeypatch.setattr(ka, "_fetch_artifact_bytes", lambda *_a: content)
    monkeypatch.setattr(
        ka, "_parse_for_understand", lambda _aid, c, _kind: parsed.append(c) or {"ok": True}
    )
    result = ka.KnowledgeUnderstandResult()
    out = asyncio.run(
        ka._understand_input(
            SimpleNamespace(job_id="j"), SimpleNamespace(attempt_id="a"), None, None, "b", src, result
        )
    )
    return out, result, parsed


def test_understand_refuses_an_image_by_name(monkeypatch):
    out, result, parsed = _understand(monkeypatch, {"artifact_id": "art-1", "kind": "image"}, _PNG)
    assert out is None and parsed == []
    assert "no understanding for image content" in result.failed_items[0]["error"]


def test_understand_reads_a_pdf_through_its_text(monkeypatch):
    from umrs.knowledge import extract

    monkeypatch.setattr(extract, "resolve_extraction_content", lambda _n, _c: b"Graphene abstract")
    out, _result, parsed = _understand(
        monkeypatch, {"artifact_id": "art-1", "kind": "document"}, _PDF
    )
    assert parsed == [b"Graphene abstract"] and out[2] == b"Graphene abstract"


# ── embed ───────────────────────────────────────────────────────────────────


def test_embed_fails_an_image_for_want_of_an_image_model(monkeypatch):
    monkeypatch.setattr(ka, "_fetch_artifact_bytes", lambda *_a: _PNG)
    result = ka.KnowledgeEmbedResult()
    assert ka._artifact_text({"artifact_id": "art-1", "kind": "image"}, None, "b", result) is None
    failure = result.failed_items[0]
    assert failure["artifact_id"] == "art-1"
    assert "no embedding model for image content" in failure["error"]


@pytest.mark.asyncio
async def test_an_image_only_embed_job_fails():
    ref = "knowledge_embed|project=p|lane=in_process|embedded=0|skipped=0|failed=1"
    attempt = SimpleNamespace(backend_ref=ref)
    result = await ka.KnowledgeEmbedStatusChecker().check_status(attempt, SimpleNamespace())
    assert result.status == "failed"


@pytest.mark.asyncio
async def test_a_mixed_batch_succeeds_but_names_what_was_not_embedded():
    ref = "knowledge_embed|project=p|lane=in_process|embedded=4|skipped=0|failed=1"
    attempt = SimpleNamespace(backend_ref=ref)
    result = await ka.KnowledgeEmbedStatusChecker().check_status(attempt, SimpleNamespace())
    assert result.status == "succeeded"
    assert "1 artifact(s) not embedded" in result.status_detail


@pytest.mark.asyncio
async def test_no_text_embedder_fails_the_job():
    """Maintainer decision 2026-10-09: lacking the model is a failure, not a no-op."""
    ref = "knowledge_embed|mode=degraded|reason=no_embedder|project=p"
    attempt = SimpleNamespace(backend_ref=ref)
    result = await ka.KnowledgeEmbedStatusChecker().check_status(attempt, SimpleNamespace())
    assert result.status == "failed"
    assert "no text embedding model" in result.exit_reason


def test_the_bundled_embedder_is_the_default(monkeypatch):
    from umrs.knowledge import text_embedder

    monkeypatch.delenv("UMRS_KNOWLEDGE_EMBED_MODEL", raising=False)
    assert text_embedder.configured_text_embedder() == "/opt/umrs/models/e5-base-v2"
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", "/models/other")
    assert text_embedder.configured_text_embedder() == "/models/other"


# ── the reason reaches the job record ───────────────────────────────────────


def _attempt_with_result(ref, failed_items):
    import json

    return SimpleNamespace(
        backend_ref=ref, metrics={"result": json.dumps({"failed_items": failed_items})}
    )


@pytest.mark.asyncio
async def test_the_failed_job_says_why():
    reason = "no embedding model for image content: this site embeds text only (/opt/x)"
    attempt = _attempt_with_result(
        "knowledge_embed|project=p|lane=in_process|embedded=0|skipped=0|failed=1",
        [{"artifact_id": "art-1", "error": reason}],
    )
    result = await ka.KnowledgeEmbedStatusChecker().check_status(attempt, SimpleNamespace())
    assert result.exit_reason == f"embed: all 1 artifact(s) failed, 0 vectors written — {reason}"


@pytest.mark.asyncio
async def test_understand_failure_says_why():
    attempt = _attempt_with_result(
        "knowledge_understand|project=p|ir_ids=0|already=0|fail=1",
        [{"artifact_id": "a", "error": "no understanding for image content: only text"}],
    )
    result = await ka.KnowledgeUnderstandStatusChecker().check_status(attempt, SimpleNamespace())
    assert result.exit_reason.endswith("— no understanding for image content: only text")


def test_the_scheduler_saves_what_a_backend_wrote():
    """The result never left memory: the stored attempt metrics were {}."""
    from unittest.mock import MagicMock

    from umrs.jobs.scheduler import _persist_backend_metrics

    store = MagicMock()
    _persist_backend_metrics(store, "att-1", SimpleNamespace(metrics={"result": "{}"}))
    store.update_attempt_metrics.assert_called_once_with("att-1", {"result": "{}"})
    store.reset_mock()
    _persist_backend_metrics(store, "att-1", SimpleNamespace(metrics={}))
    store.update_attempt_metrics.assert_not_called()


# ── an IR whose graph was never written is repaired, not skipped ────────────


def _understand_with(monkeypatch, *, ir_exists, file_node_exists):
    import asyncio

    async def ir(_store, _aid):
        return "ir-1" if ir_exists else None

    class _Pool:
        async def fetchval(self, _sql, _node_id):
            return 1 if file_node_exists else None

    monkeypatch.setattr(ka, "_existing_current_ir", ir)
    monkeypatch.setattr(ka, "_fetch_artifact_bytes", lambda *_a: b"# Note\nZirconia.")
    monkeypatch.setattr(ka, "_parse_for_understand", lambda *_a: {"ok": True})
    result = ka.KnowledgeUnderstandResult()
    out = asyncio.run(
        ka._understand_input(
            SimpleNamespace(job_id="j", spec={}, project_id="p"),
            SimpleNamespace(attempt_id="a"),
            SimpleNamespace(_pool=_Pool()),
            None,
            "b",
            {"artifact_id": "art-1", "kind": "document"},
            result,
        )
    )
    return out, result


def test_an_ir_with_its_graph_is_skipped(monkeypatch):
    out, result = _understand_with(monkeypatch, ir_exists=True, file_node_exists=True)
    assert out is None and result.already_ir == 1


def test_an_ir_without_its_graph_is_rebuilt(monkeypatch):
    """Postgres restarted between the IR insert and the node writes (live 2026-10-09):
    every retry said "already has IR", the nodes never came, and the chained embed
    failed forever on the missing file node (knowledge_edges_target_id_fkey)."""
    out, result = _understand_with(monkeypatch, ir_exists=True, file_node_exists=False)
    assert out is not None and result.already_ir == 0
