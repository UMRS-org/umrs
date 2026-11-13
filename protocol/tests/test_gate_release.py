"""The HPC gate's bytes are a release (ADR-112; U2.0 spec §6.2).

``hpc_gate.py`` runs under sshd on HPC login nodes, and its sha256 is stored
at enrolment (``site_accounts.gate_sha256``). Any byte change, a reformat
included, is a gate release: it needs a reviewed row in ``golden/gate.json``
and normally a ``GATE_VERSION`` bump, which forces re-enrolment. These tests
make an unreviewed change fail.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from umrs_protocol import hpc_gate

GATE_JSON = json.loads((Path(__file__).parent / "golden" / "gate.json").read_text(encoding="utf-8"))
#: gate_version -> the sha256 digests released under it.
KNOWN_GATE_RELEASES: dict[str, set[str]] = {GATE_JSON["gate_version"]: {GATE_JSON["sha256"]}}
#: gate_version -> the git blob ids of those releases.
KNOWN_GATE_BLOBS: dict[str, set[str]] = {GATE_JSON["gate_version"]: {GATE_JSON["blob"]}}
CHECKOUT_GATE = Path(__file__).resolve().parents[1] / "src" / "umrs_protocol" / "hpc_gate.py"


def _git_blob(data: bytes) -> str:
    """``git hash-object`` of ``data``, computed without git."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def test_gate_version_matches_the_pin() -> None:
    assert GATE_JSON["gate_version"] == hpc_gate.GATE_VERSION


def test_checkout_gate_is_a_known_release() -> None:
    data = CHECKOUT_GATE.read_bytes()
    assert hashlib.sha256(data).hexdigest() in KNOWN_GATE_RELEASES[hpc_gate.GATE_VERSION]
    assert _git_blob(data) in KNOWN_GATE_BLOBS[hpc_gate.GATE_VERSION]


def test_imported_gate_is_a_known_release() -> None:
    """The module that runs is the released file (not a stale install)."""
    data = Path(hpc_gate.__file__).read_bytes()
    assert hashlib.sha256(data).hexdigest() in KNOWN_GATE_RELEASES[hpc_gate.GATE_VERSION]
    assert _git_blob(data) in KNOWN_GATE_BLOBS[hpc_gate.GATE_VERSION]


def test_gate_sha256_hashes_its_own_file() -> None:
    data = Path(hpc_gate.__file__).read_bytes()
    assert hpc_gate.gate_sha256() == hashlib.sha256(data).hexdigest()


def test_git_blob_matches_git_hash_object() -> None:
    """The helper agrees with git on a known vector (empty blob)."""
    assert _git_blob(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
