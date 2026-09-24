"""ADR-032 raw-materials amendment (2026-09-24): opt-in raw objects.

Export: licence gate, D-46 ``raw`` redaction, dedup, caps, unreadable and
hash-mismatched objects skipped by reason. Import: every object re-hashed
BEFORE any write (one bad object refuses the bundle), objects stored in the
destination and their artifacts re-bound; declined when the importer opts
out; storage failure is a named degraded result. API gate: owner/admin,
licence, sensitive acknowledgement, fail-closed on an unreachable registry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from umrs.api._raw_export_gate import authorize_raw_export
from umrs.knowledge import bundle_raw
from umrs.knowledge.bundle_export import KnowledgeExportEngine
from umrs.knowledge.bundle_import import BundleFormatError, KnowledgeImportEngine

_PDF = b"%PDF-1.7 fake paper bytes"
_CODE = b"print('hello')\n"


def _h(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _row(art: str, raw: bytes, kind: str = "document", **over) -> dict:
    return {
        "artifact_id": art,
        "content_hash": _h(raw),
        "kind": kind,
        "source_ref": f"k/{art}",
        "size_bytes": len(raw),
        "mime_type": "application/pdf",
        **over,
    }


_CAPS = bundle_raw.RawCaps(object_max=1024, total_max=4096)


# ── export: planning ──────────────────────────────────────────────────────────


def test_no_licence_carries_nothing():
    rows = [_row("a", _PDF)]
    out, skipped = bundle_raw.plan_raw(rows, licence=" ", redaction_rules=[], caps=_CAPS)
    assert (out, skipped) == ([], {"no_licence": 1})


def test_plan_skips_by_reason():
    rows = [
        _row("a", _PDF),
        _row("b", _PDF),  # same bytes → one object
        _row("c", _CODE, kind="code"),
        _row("d", b"x", content_hash="synthetic16hexab"),
        _row("e", b"big", size_bytes=10_000),
    ]
    rules = [{"target": "raw", "action": "drop", "kinds": ["code"]}]
    out, skipped = bundle_raw.plan_raw(rows, licence="CC-BY-4.0", redaction_rules=rules, caps=_CAPS)
    assert [r["artifact_id"] for r in out] == ["a"]
    assert skipped == {"duplicate": 1, "redacted": 1, "no_bytes": 1, "over_object_cap": 1}


def test_raw_redaction_without_kinds_drops_everything():
    assert bundle_raw.raw_redaction([{"target": "raw", "action": "drop"}]) == (True, frozenset())
    assert bundle_raw.raw_redaction([{"target": "raw", "action": "hash"}]) == (False, frozenset())


def test_env_caps_only_tighten(monkeypatch):
    monkeypatch.setenv("UMRS_KB_RAW_OBJECT_MAX_BYTES", "10")
    monkeypatch.setenv("UMRS_KB_RAW_TOTAL_MAX_BYTES", str(1 << 50))
    caps = bundle_raw.RawCaps.from_env()
    assert caps.object_max == 10
    assert caps.total_max == bundle_raw.RawCaps().total_max


# ── export: writing ───────────────────────────────────────────────────────────


def test_write_raw_verifies_bytes_and_caps(tmp_path):
    blobs = {"k/a": _PDF, "k/b": b"tampered", "k/c": None}
    rows = [_row("a", _PDF), _row("b", _CODE), _row("c", b"gone")]
    skipped: dict[str, int] = {}
    written = bundle_raw.write_raw(tmp_path, rows, blobs.get, _CAPS, skipped)
    assert [r["artifact_id"] for r in written] == ["a"]
    assert skipped == {"hash_mismatch": 1, "unreadable": 1}
    assert (tmp_path / "raw" / _h(_PDF)).read_bytes() == _PDF
    table = [json.loads(x) for x in (tmp_path / "raw_objects.jsonl").read_text().splitlines()]
    assert table == written


def test_total_cap_stops_carriage(tmp_path):
    caps = bundle_raw.RawCaps(object_max=1024, total_max=len(_PDF) + 1)
    rows = [_row("a", _PDF), _row("b", _CODE)]
    skipped: dict[str, int] = {}
    blobs = {"k/a": _PDF, "k/b": _CODE}
    written = bundle_raw.write_raw(tmp_path, rows, blobs.get, caps, skipped)
    assert len(written) == 1
    assert skipped == {"over_total_cap": 1}


def test_engine_raw_block(tmp_path):
    engine = KnowledgeExportEngine(
        store=None, output_dir=tmp_path, allow_unsigned=True, artifact_reader={"k/a": _PDF}.get
    )
    assert engine._dump_raw(None, {}, {}) == {"included": False}

    class _Writer:
        path = tmp_path

    engine = KnowledgeExportEngine(
        store=None,
        output_dir=tmp_path,
        allow_unsigned=True,
        artifact_reader={"k/a": _PDF}.get,
        include_raw=True,
        raw_acknowledgement={"by": "alice", "data_class": "sensitive"},
    )
    engine._artifact_rows = [_row("a", _PDF)]
    counts: dict = {}
    block = engine._dump_raw(_Writer(), {"licence": "CC-BY-4.0"}, counts)
    assert block["count"] == 1 and block["bytes"] == len(_PDF)
    assert block["acknowledgement"]["by"] == "alice"
    assert counts["raw_objects"] == (1, "raw_objects.jsonl")


# ── import ────────────────────────────────────────────────────────────────────


def _bundle(root: Path, objects: dict[str, bytes]) -> Path:
    bundle = root / "b"
    bundle.mkdir()
    rows = [_row(a, raw) for a, raw in objects.items()]
    (bundle / "artifacts.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    bundle_raw.write_raw(
        bundle, rows, {r["source_ref"]: objects[r["artifact_id"]] for r in rows}.get, _CAPS, {}
    )
    manifest = {"format": "r4-kb", "version": "1.2", "project_id": "src", "tables": {}}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return bundle


class _Store:
    def __init__(self, bound=True):
        self.artifacts: list[dict] = []
        self.rebound: list[tuple] = []
        self._bound = bound

    async def insert_artifact(self, **kwargs):
        self.artifacts.append(kwargs)
        return True

    async def rebind_artifact_source(self, project_id, artifact_id, source_ref):
        self.rebound.append((project_id, artifact_id, source_ref))
        return self._bound


def _writer(stored):
    def write(project_id, content):
        stored.append((project_id, content))
        return f"projects/{project_id}/knowledge/artifacts/{_h(content)}"

    return write


async def _import(bundle, store, writer, **kw):
    engine = KnowledgeImportEngine(store=store, raw_writer=writer)
    return await engine.import_project(
        bundle, target_project_id="dst", verify=False, allow_unsigned=True, **kw
    )


async def test_import_stores_and_rebinds(tmp_path):
    stored: list = []
    store = _Store()
    result = await _import(_bundle(tmp_path, {"a": _PDF, "b": _CODE}), store, _writer(stored))
    assert (result.raw_restored, result.raw_unbound, result.raw_declined) == (2, 0, 0)
    assert {c for _p, c in stored} == {_PDF, _CODE}
    assert ("dst", "a", f"projects/dst/knowledge/artifacts/{_h(_PDF)}") in store.rebound


async def test_import_counts_unbound_when_the_row_deduped(tmp_path):
    result = await _import(_bundle(tmp_path, {"a": _PDF}), _Store(bound=False), _writer([]))
    assert (result.raw_restored, result.raw_unbound) == (0, 1)


async def test_importer_can_decline_raw(tmp_path):
    stored: list = []
    result = await _import(
        _bundle(tmp_path, {"a": _PDF}), _Store(), _writer(stored), accept_raw=False
    )
    assert result.raw_declined == 1 and stored == []


async def test_a_tampered_object_refuses_before_any_write(tmp_path):
    bundle = _bundle(tmp_path, {"a": _PDF})
    (bundle / "raw" / _h(_PDF)).write_bytes(b"swapped")
    store = _Store()
    with pytest.raises(BundleFormatError, match="raw objects refused"):
        await _import(bundle, store, _writer([]))
    assert store.artifacts == []


async def test_an_unlisted_object_refuses(tmp_path):
    bundle = _bundle(tmp_path, {"a": _PDF})
    (bundle / "raw" / ("f" * 64)).write_bytes(b"smuggled")
    with pytest.raises(BundleFormatError, match="unlisted"):
        await _import(bundle, _Store(), _writer([]))


async def test_storage_failure_is_degraded_not_fatal(tmp_path):
    def broken(_project_id, _content):
        raise RuntimeError("s3 down")

    result = await _import(_bundle(tmp_path, {"a": _PDF}), _Store(), broken)
    assert result.degraded and "raw objects not stored" in result.degraded_reason
    assert result.artifact_count == 1


def test_objects_without_table_are_refused(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "x").write_bytes(b"x")
    with pytest.raises(bundle_raw.RawBundleError):
        bundle_raw.verified_raw_rows(tmp_path)


# ── API gate ─────────────────────────────────────────────────────────────────


class _Pool:
    def __init__(self, row):
        self._row = row

    def getconn(self):
        row = self._row

        class _Cur:
            def execute(self, _sql, _params):
                pass

            def fetchone(self):
                return row

        class _Conn:
            def cursor(self):
                return _Cur()

        return _Conn()

    def putconn(self, _conn):
        pass


def _gate(row, *, user="alice", admin=False, ack=False, pool=True):
    return authorize_raw_export(
        _Pool(row) if pool else None,
        project_id="p",
        user_id=user,
        is_admin=admin,
        acknowledge_sensitive=ack,
    )


def _code(exc: pytest.ExceptionInfo) -> str:
    """The typed code rides the error-code header (status body unchanged)."""
    return " ".join((exc.value.headers or {}).values())


def test_owner_of_a_non_sensitive_project_with_licence_passes():
    assert _gate(("alice", "non_sensitive", "CC-BY-4.0")) == {}


def test_non_owner_is_refused_and_admin_passes():
    with pytest.raises(HTTPException) as exc:
        _gate(("bob", "non_sensitive", "MIT"))
    assert "raw_forbidden" in _code(exc)
    assert _gate(("bob", "non_sensitive", "MIT"), admin=True) == {}


def test_licence_is_required():
    with pytest.raises(HTTPException) as exc:
        _gate(("alice", "non_sensitive", ""))
    assert "raw_licence_required" in _code(exc)


@pytest.mark.parametrize("data_class", ["sensitive", ""])
def test_sensitive_or_unset_needs_acknowledgement(data_class):
    with pytest.raises(HTTPException) as exc:
        _gate(("alice", data_class, "MIT"))
    assert "raw_unacknowledged" in _code(exc)
    record = _gate(("alice", data_class, "MIT"), ack=True)
    assert record["by"] == "alice" and record["data_class"] == "sensitive"


def test_unreachable_registry_fails_closed():
    with pytest.raises(HTTPException) as exc:
        _gate(None, pool=False)
    assert "unavailable" in _code(exc)
