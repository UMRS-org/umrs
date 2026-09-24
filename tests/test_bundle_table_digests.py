"""Bundle integrity covers table CONTENTS (2026-09-24).

Before, the HMAC signature covered only manifest.json (table names + counts),
so a signed bundle whose artifacts/nodes/edges rows were edited still
verified. The signed manifest now pins each table file's sha256.
"""

from __future__ import annotations

import hashlib
import hmac
import json

from umrs.knowledge.bundle_export import (
    stamp_table_digests,
    tables_pinned,
    verify_bundle_integrity,
)

_KEY = "k"


def _signed_bundle(tmp_path, *, pin=True, file_name="nodes.jsonl"):
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "nodes.jsonl").write_text('{"node_id": "n1", "label": "true"}\n')
    manifest = {
        "format": "r4-kb",
        "version": "1.2",
        "project_id": "p",
        "tables": {"nodes": {"count": 1, "file": file_name}},
    }
    if pin:
        stamp_table_digests(manifest, bundle)
    body = json.dumps(manifest, sort_keys=True).encode()
    manifest["hash"] = hashlib.sha256(body).hexdigest()
    manifest["signature"] = hmac.new(_KEY.encode(), body, hashlib.sha256).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "signature.sig").write_text(manifest["signature"])
    return bundle


def test_signed_bundle_with_pinned_tables_verifies(tmp_path):
    bundle = _signed_bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert tables_pinned(manifest)
    assert verify_bundle_integrity(bundle, _KEY) is True


def test_an_edited_table_row_fails_verification(tmp_path):
    bundle = _signed_bundle(tmp_path)
    (bundle / "nodes.jsonl").write_text('{"node_id": "n1", "label": "forged"}\n')
    assert verify_bundle_integrity(bundle, _KEY) is False


def test_a_missing_table_file_fails_verification(tmp_path):
    bundle = _signed_bundle(tmp_path)
    (bundle / "nodes.jsonl").unlink()
    assert verify_bundle_integrity(bundle, _KEY) is False


def test_pre_digest_bundles_still_verify_but_are_not_pinned(tmp_path):
    bundle = _signed_bundle(tmp_path, pin=False)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert not tables_pinned(manifest)
    assert verify_bundle_integrity(bundle, _KEY) is True


def test_a_pinned_entry_naming_a_path_fails(tmp_path):
    bundle = _signed_bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["tables"]["nodes"]["file"] = "../nodes.jsonl"
    manifest.pop("hash")
    manifest.pop("signature")
    body = json.dumps(manifest, sort_keys=True).encode()
    manifest["hash"] = hashlib.sha256(body).hexdigest()
    manifest["signature"] = hmac.new(_KEY.encode(), body, hashlib.sha256).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "signature.sig").write_text(manifest["signature"])
    assert verify_bundle_integrity(bundle, _KEY) is False
