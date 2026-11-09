"""Shared builders for the wave C server tests: a valid grant body (C1), a
grant context, and a site account store with an active, linked account."""

from __future__ import annotations

import base64
import secrets
import time
from typing import Any

from umrs.hpc_accounts.grants import GrantContext, SignedGrant
from umrs.hpc_accounts.store import InMemorySiteAccountStore

from r4_contracts import hpc_gate

SITE = "bdrt-front"
SIG = "-----BEGIN SSH SIGNATURE-----\nU1NIU0lH\n-----END SSH SIGNATURE-----"
SRC_SHA, STEPS_SHA = "1" * 64, "2" * 64
IMAGE = {"ref": "umrs-step:1", "image_id": "a" * 64, "archive_sha256": "b" * 64}
VERIFIED = {
    "site_login": "alice1",
    "verified_login": "alice1",
    "verified_assocs": [{"cluster": "c", "account": "grp-a", "partition": "", "qos": ""}],
    "gate_version": "3",
    "gate_sha256": "ab" * 32,
    "signer_fingerprints": ["SHA256:sig"],
    "uv_required": False,
}


def grant_body(**over: Any) -> dict[str, Any]:
    now = int(time.time())
    body: dict[str, Any] = {
        "v": 1,
        "type": "grant",
        "grant_id": secrets.token_hex(16),
        "site": SITE,
        "login": "alice1",
        "principal": "alice",
        "accounts": ["grp-a"],
        "projects": ["p1"],
        "workflows": [
            {"name": "wf", "version": "1.0.0", "src_manifest_sha256": SRC_SHA, "steps_sha256": STEPS_SHA, "params": []}
        ],
        "images": [dict(IMAGE)],
        "param_schema": {"type": "object", "additionalProperties": False, "required": [], "properties": {}},
        "inputs": {"mode": "any", "listed": []},
        "caps": {"max_jobs": 10, "max_gpu_hours": 2, "max_core_hours": 0, "max_concurrent": 2},
        "not_before": now,
        "expires": now + 86400,
        "issued": now,
        "delegable": False,
        "nonce": secrets.token_hex(16),
    }
    body.update(over)
    return hpc_gate.validate_grant(body)


def encoded(body: dict[str, Any]) -> str:
    return base64.b64encode(hpc_gate.canonical_bytes(body)).decode("ascii")


def signed(body: dict[str, Any]) -> SignedGrant:
    return SignedGrant(payload=hpc_gate.canonical_bytes(body), body=body, signature=SIG)


def account_store(venue: str = SITE, tables: Any = None) -> InMemorySiteAccountStore:
    store = InMemorySiteAccountStore(venue, tables) if tables is not None else InMemorySiteAccountStore(venue)
    store.issue_key("alice", "tk-000000000001", "SHA256:fp")
    store.record_verified("alice", "tk-000000000001", VERIFIED)
    store.upsert_link("alice", "p1", "grp-a")
    return store


def context(store: Any, **over: Any) -> GrantContext:
    fields: dict[str, Any] = {
        "venue": SITE,
        "principal": "alice",
        "account": dict(store.get_account("alice") or {}),
        "links": list(store.list_links("alice")),
        "memberships": frozenset({"p1", "p2"}),
        "digests_for": lambda name, version: (SRC_SHA, STEPS_SHA) if (name, version) == ("wf", "1.0.0") else None,
        "images": frozenset({(IMAGE["ref"], IMAGE["image_id"], IMAGE["archive_sha256"])}),
    }
    fields.update(over)
    return GrantContext(**fields)
