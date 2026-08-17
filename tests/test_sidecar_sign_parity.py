"""Guard: sidecar's inlined signer stays byte-identical with the contract.

The auth-sidecar image has a narrow ansible build context and cannot ship
r4-contracts, so it carries an inlined twin of handshake_hmac.sign_claims
with a KEEP IN SYNC comment. Any canonical-format drift breaks the
gateway↔UMRS handshake silently (401-for-everyone) — this test makes drift
loud instead.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from r4_contracts.handshake_hmac import sign_claims, sign_workflow_author_claim


def _load_sidecar():
    sidecar_path = Path(__file__).resolve().parents[2] / "gateway" / "sidecar" / "main.py"
    assert sidecar_path.is_file(), f"expected sidecar at {sidecar_path}"
    spec = importlib.util.spec_from_file_location("_sidecar_main", sidecar_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_sidecar_signer_matches_contract_canonical_format():
    sidecar = _load_sidecar()
    cases = [
        ("s3crt", "alice", "proj-a", False),
        ("s3crt", "alice", "proj-a", True),
        ("other", "b_ob", "proj-b", False),
        ("x" * 64, "u-y", "proj-c", True),
    ]
    for secret, user_id, project_id, is_admin in cases:
        expected = sign_claims(secret, user_id, project_id, is_admin)
        got = sidecar._sign_identity_hmac(secret, user_id, project_id, is_admin)
        assert expected == got, f"canonical drift for ({user_id}, {project_id}, {is_admin})"


def test_sidecar_workflow_author_signer_matches_contract():
    """ADR-027 wf-claim signer parity (own basis, same twin arrangement)."""
    sidecar = _load_sidecar()
    cases = [
        ("s3crt", "alice", "proj-a", False),
        ("s3crt", "alice", "proj-a", True),
        ("other", "b_ob", "proj-b", False),
        ("x" * 64, "u-y", "proj-c", True),
    ]
    for secret, user_id, project_id, author in cases:
        expected = sign_workflow_author_claim(secret, user_id, project_id, author)
        got = sidecar._sign_workflow_author_hmac(secret, user_id, project_id, author)
        assert expected == got, f"wf canonical drift for ({user_id}, {project_id}, {author})"
