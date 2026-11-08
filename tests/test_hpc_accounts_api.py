"""``/umrs/sites/me/hpc`` (ADR-112 D-3/D-7; plan §4 B1.5a wire contract).

The verified caller registers their own login at this site: a transport key
(public half only in any answer), a verify that needs the gate's own probe
refusal, the claimed login and this server's gate release, and project links
that need an active account, live membership and an allocation the site
listed. Registration is refused where the site does not allow user accounts.
Transport keys are real ``ssh-keygen`` keys under ``tmp_path``; the gate is a
scripted runner.
"""

from __future__ import annotations

import json
import shutil
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import hpc_accounts as api
from umrs.hpc_accounts.store import ACCOUNT_VIEW, LINK_COLUMNS, InMemorySiteAccountStore
from umrs.orchestrator.site_config import (
    HpcAccessConfig,
    ServiceAccountAccess,
    SiteConfig,
    UserAccountAccess,
)

pytestmark = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="needs ssh-keygen")

SITE = "bdrt-front"
SHA = "ab" * 32
ALICE = {"X-User-ID": "alice", "X-Verified-Project-ID": "p1"}
BOB = {"X-User-ID": "bob", "X-Verified-Project-ID": "p1"}
ASSOCS = [
    {"cluster": "c1", "account": "grp-a", "partition": "gpu", "qos": "normal"},
    {"cluster": "c1", "account": "grp-b", "partition": "", "qos": ""},
]
PROBE_REFUSED = (4, b"", b'{"error": "gate_probe_refusal", "detail": "the UMRS gate answered"}')


def _hello(**over: Any) -> tuple[int, bytes, bytes]:
    data = {
        "gate_version": "1",
        "gate_sha256": SHA,
        "site": SITE,
        "login": "alice1",
        "uid": 1001,
        "python": "3.9.18",
        "openssh": "OpenSSH_8.7p1",
        "signers": [{"fingerprint": "SHA256:sig", "type": "sk-ssh-ed25519@openssh.com"}],
        "assocs": ASSOCS,
        "log_head": {"n": 0, "h": "0" * 64},
    }
    data.update(over)
    return 0, json.dumps(data).encode(), b""


def _site(modes: list[str], user_account: UserAccountAccess | None = None) -> SiteConfig:
    site = SiteConfig(name=SITE)
    site.extra = {"ssh_target": "login-0", "ssh_port": 22, "ssh_user": "op"}
    site.scheduler.hpc_access = HpcAccessConfig(
        operator="Centre X",
        modes_allowed=modes,
        default_mode="service_account",
        service_account=ServiceAccountAccess(login="op", personal=True),
        user_account=user_account,
    )
    return site


def _user_site(**over: Any) -> SiteConfig:
    ua = UserAccountAccess(
        methods=["gated_key"], signing_domain="platform.example.org", from_address="10.1.2.3"
    )
    for key, value in over.items():
        setattr(ua, key, value)
    return _site(["service_account", "user_account"], ua)


class _Gate:
    """Scripted gate: verb -> (code, stdout, stderr); records each argv."""

    def __init__(self) -> None:
        self.answers: dict[str, tuple[int, bytes, bytes]] = {
            "probe": PROBE_REFUSED,
            "hello": _hello(),
        }
        self.calls: list[list[str]] = []

    async def __call__(self, argv: list[str], _stdin: bytes, _timeout: float):
        self.calls.append(argv)
        return self.answers[argv[-1]]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    monkeypatch.setenv("UMRS_TRANSPORT_KEY_DIR", str(tmp_path / "keys"))
    store = InMemorySiteAccountStore(SITE)
    gate = _Gate()
    members = {"alice": ["p1", "p2"], "bob": ["p1"]}
    state = {"site": _user_site()}
    monkeypatch.setattr(api, "_load_site", lambda: state["site"])
    monkeypatch.setattr(api, "_make_store", lambda _req, venue: _scoped(store, venue))
    monkeypatch.setattr(api, "_gate_manifest", lambda: ("1", SHA))
    monkeypatch.setattr(api, "_gate_runner", gate)
    monkeypatch.setattr(api, "_live_memberships", lambda _req, user: members.get(user))
    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    return client, store, gate, state, tmp_path / "keys"


def _scoped(store: InMemorySiteAccountStore, venue: str) -> InMemorySiteAccountStore:
    return InMemorySiteAccountStore(venue, store.tables)


def _code(response) -> str:
    return response.headers.get("x-platform-error-code", "")


def _register(client, headers=ALICE, login="alice1"):
    key = client.post("/umrs/sites/me/hpc/transport-key", json={"rotate": False}, headers=headers)
    assert key.status_code == 200, key.text
    verify = client.post(
        "/umrs/sites/me/hpc/verify",
        json={
            "login": login,
            "signers": ["SHA256:sig"],
            "uv_required": False,
            "allow_non_sk": False,
        },
        headers=headers,
    )
    return key.json(), verify


# ── mode ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/umrs/sites/me/hpc/transport-key", {"rotate": False}),
        ("post", "/umrs/sites/me/hpc/verify", {"login": "alice1"}),
        ("post", "/umrs/sites/me/hpc/links", {"project_id": "p1", "slurm_account": "grp-a"}),
    ],
)
def test_registration_is_refused_where_user_accounts_are_not_allowed(env, method, path, body):
    client, store, gate, state, keys = env
    state["site"] = _site(["service_account"], UserAccountAccess(methods=["gated_key"]))
    response = getattr(client, method)(path, json=body, headers=ALICE)
    assert (response.status_code, _code(response)) == (403, "hpcaccess.mode_undeclared")
    assert store.list_accounts() == [] and gate.calls == [] and not keys.exists()


@pytest.mark.parametrize("name", ["Bdrt", "bdrt_front", "a" * 41])
def test_registration_is_refused_under_a_name_the_gate_cannot_carry(env, name):
    """The venue becomes the gate's --site and the key comment; r4 and the
    gate accept only [a-z0-9-]{1,40} there (2026-11-08 integration)."""
    client, store, gate, state, keys = env
    site = _user_site()
    site.name = name
    state["site"] = site
    response = client.post(
        "/umrs/sites/me/hpc/transport-key", json={"rotate": False}, headers=ALICE
    )
    assert (response.status_code, _code(response)) == (403, "hpcaccess.mode_undeclared")
    assert "cannot name a gate" in response.text
    assert gate.calls == [] and not keys.exists()


def test_a_server_without_a_site_refuses_every_route(env):
    client, _store, _gate, state, _keys = env
    state["site"] = None
    response = client.get("/umrs/sites/me/hpc", headers=ALICE)
    assert _code(response) == "hpcaccess.mode_undeclared"


def test_reading_and_revoking_stay_open_after_the_mode_is_withdrawn(env):
    client, _store, _gate, state, keys = env
    _register(client)
    state["site"] = _site(["service_account"])
    assert client.get("/umrs/sites/me/hpc", headers=ALICE).json()["account"]["status"] == "active"
    assert client.delete("/umrs/sites/me/hpc", headers=ALICE).json() == {"status": "revoked"}
    assert list((keys / SITE / "alice").iterdir()) == []


def test_every_route_needs_a_verified_caller(env):
    client = env[0]
    assert client.get("/umrs/sites/me/hpc").status_code == 401


# ── transport key ────────────────────────────────────────────────────────


def test_transport_key_answers_the_contract_shape_with_the_public_half_only(env):
    client, store, _gate, _state, keys = env
    response = client.post(
        "/umrs/sites/me/hpc/transport-key", json={"rotate": False}, headers=ALICE
    )
    body = response.json()
    assert set(body) == {"key_id", "public_key", "site", "domain", "from", "gate"}
    assert (body["site"], body["domain"], body["from"]) == (
        SITE,
        "platform.example.org",
        "10.1.2.3",
    )
    assert body["gate"] == {"version": "1", "sha256": SHA}
    kind, _b64, comment = body["public_key"].split(" ")
    assert (kind, comment) == ("ssh-ed25519", f"umrs-transport-{SITE}-{body['key_id']}")
    assert "PRIVATE KEY" not in response.text
    private = keys / SITE / "alice" / body["key_id"]
    assert private.read_text().startswith("-----BEGIN OPENSSH PRIVATE KEY-----")
    account = store.get_account("alice")
    assert (account["status"], account["transport_key_id"]) == ("pending", body["key_id"])


def test_transport_key_is_idempotent_without_rotate_and_new_with_it(env):
    client, store, _gate, _state, keys = env
    url = "/umrs/sites/me/hpc/transport-key"
    first = client.post(url, json={"rotate": False}, headers=ALICE).json()
    assert client.post(url, headers=ALICE).json()["key_id"] == first["key_id"]
    rotated = client.post(url, json={"rotate": True}, headers=ALICE).json()
    assert rotated["key_id"] != first["key_id"]
    assert store.get_account("alice")["transport_key_id"] == rotated["key_id"]
    # the old key stays until the gate answers on the new one
    assert {p.name for p in (keys / SITE / "alice").iterdir() if "." not in p.name} == {
        first["key_id"],
        rotated["key_id"],
    }


def test_a_rotation_is_verified_on_the_new_key_and_drops_the_old(env):
    client, store, gate, _state, keys = env
    first, _ = _register(client)
    rotated = client.post(
        "/umrs/sites/me/hpc/transport-key", json={"rotate": True}, headers=ALICE
    ).json()
    assert store.get_account("alice")["status"] == "pending"
    verify = client.post("/umrs/sites/me/hpc/verify", json={"login": "alice1"}, headers=ALICE)
    assert verify.status_code == 200
    assert _key_arg(gate.calls[-1]).endswith(rotated["key_id"])
    assert {p.name for p in (keys / SITE / "alice").iterdir()} == {
        rotated["key_id"],
        rotated["key_id"] + ".pub",
    }
    assert first["key_id"] != rotated["key_id"]


def test_two_people_get_two_keys(env):
    client = env[0]
    a = client.post("/umrs/sites/me/hpc/transport-key", headers=ALICE).json()
    b = client.post("/umrs/sites/me/hpc/transport-key", headers=BOB).json()
    assert a["key_id"] != b["key_id"] and a["public_key"] != b["public_key"]


# ── verify ───────────────────────────────────────────────────────────────


def test_verify_stores_the_sites_report_and_activates(env):
    client = env[0]
    _key, verify = _register(client)
    assert verify.status_code == 200, verify.text
    account = verify.json()["account"]
    assert tuple(account) == ACCOUNT_VIEW
    got = tuple(account[k] for k in ("status", "gate_probe", "site_login", "verified_login"))
    assert got == ("active", "refused", "alice1", "alice1")
    got = tuple(account[k] for k in ("gate_sha256", "gate_version", "verified_assocs"))
    assert got == (SHA, "1", ASSOCS)
    assert account["signer_fingerprints"] == ["SHA256:sig"]


def test_verify_probes_then_says_hello_over_the_persons_key(env):
    client, _store, gate, _state, keys = env
    key, verify = _register(client)
    key_path = str(keys / SITE / "alice" / key["key_id"])
    assert [argv[-1] for argv in gate.calls] == ["probe", "hello"]
    shapes = {
        (tuple(argv[:5]), "IdentitiesOnly=yes" in argv, "ControlPath=none" in argv, argv[-2])
        for argv in gate.calls
    }
    assert shapes == {(("ssh", "-F", "none", "-i", key_path), True, True, "alice1@login-0")}
    assert "PRIVATE KEY" not in verify.text


def test_verify_needs_a_transport_key_first(env):
    client = env[0]
    response = client.post("/umrs/sites/me/hpc/verify", json={"login": "alice1"}, headers=ALICE)
    assert (response.status_code, _code(response)) == (403, "hpcaccess.not_enrolled")


def test_verify_without_the_key_file_is_not_enrolled(env):
    client, _store, gate, _state, keys = env
    key = client.post("/umrs/sites/me/hpc/transport-key", headers=ALICE).json()
    (keys / SITE / "alice" / key["key_id"]).unlink()
    response = client.post("/umrs/sites/me/hpc/verify", json={"login": "alice1"}, headers=ALICE)
    assert _code(response) == "hpcaccess.not_enrolled" and gate.calls == []


@pytest.mark.parametrize(
    ("probe", "recorded"),
    [
        ((0, b'{"anything": 1}', b""), "bypassed"),  # a site ForceCommand answered
        ((4, b"", b'{"error": "unknown_verb", "detail": "x"}'), "bypassed"),
        ((255, b"", b"Permission denied (publickey)"), "unknown"),
    ],
)
def test_verify_requires_the_gates_own_probe_refusal(env, probe, recorded):
    client, store, gate, _state, _keys = env
    gate.answers["probe"] = probe
    _key, verify = _register(client)
    assert (verify.status_code, _code(verify)) == (403, "hpcaccess.gate_refused")
    account = store.get_account("alice")
    assert (account["status"], account["gate_probe"]) == ("pending", recorded)
    assert [argv[-1] for argv in gate.calls] == ["probe"]  # hello is never asked


def test_verify_requires_the_claimed_login(env):
    client, store, gate, _state, _keys = env
    gate.answers["hello"] = _hello(login="someone-else")
    _key, verify = _register(client)
    assert _code(verify) == "hpcaccess.gate_refused"
    assert json.loads(verify.text)["detail"].endswith("not 'alice1'")
    assert store.get_account("alice")["status"] == "pending"


def test_a_refused_hello_is_gate_refused(env):
    client, _store, gate, _state, _keys = env
    gate.answers["hello"] = (4, b"", b'{"error": "log_unwritable", "detail": "disk full"}')
    _key, verify = _register(client)
    assert _code(verify) == "hpcaccess.gate_refused"


@pytest.mark.parametrize(
    ("hello", "min_version"),
    [
        (_hello(gate_sha256="cd" * 32), 1),
        (_hello(gate_version="1"), 2),
        (_hello(gate_version="x"), 1),
    ],
)
def test_verify_requires_this_servers_gate_release(env, hello, min_version):
    client, store, gate, state, _keys = env
    gate.answers["hello"] = hello
    state["site"] = _user_site(gate_min_version=min_version)
    _key, verify = _register(client)
    assert (verify.status_code, _code(verify)) == (409, "hpcaccess.gate_version")
    assert store.get_account("alice")["status"] == "pending"


@pytest.mark.parametrize(
    ("uv_required", "allow_non_sk", "ok"),
    [(True, False, True), (False, False, False), (True, True, False), (None, None, False)],
)
def test_a_site_that_requires_a_pin_accepts_only_a_gate_that_does(
    env, uv_required, allow_non_sk, ok
):
    """The policy is the gate's own report (hello, from gate.json); the
    request's claim is not evidence (B1 review, 2026-11-08)."""
    client, store, gate, state, _keys = env
    state["site"] = _user_site(uv_required=True)
    gate.answers["hello"] = _hello(uv_required=uv_required, allow_non_sk=allow_non_sk)
    client.post("/umrs/sites/me/hpc/transport-key", headers=ALICE)
    body = {"login": "alice1", "uv_required": True, "allow_non_sk": False}  # a claim only
    verify = client.post("/umrs/sites/me/hpc/verify", json=body, headers=ALICE)
    assert (verify.status_code == 200) is ok, verify.text
    if ok:
        assert verify.json()["account"]["uv_required"] is True
    else:
        assert _code(verify) == "hpcaccess.gate_refused"
        assert store.get_account("alice")["status"] == "pending"


def test_a_login_that_could_be_an_ssh_option_is_refused(env):
    client, _store, gate, _state, _keys = env
    client.post("/umrs/sites/me/hpc/transport-key", headers=ALICE)
    response = client.post(
        "/umrs/sites/me/hpc/verify", json={"login": "-oProxyCommand=x"}, headers=ALICE
    )
    assert response.status_code == 422 and gate.calls == []


def test_assocs_from_the_site_are_trimmed_to_the_four_keys(env):
    client, _store, gate, _state, _keys = env
    gate.answers["hello"] = _hello(assocs=[{"account": "grp-a", "extra": "x"}, "junk"])
    _key, verify = _register(client)
    assert verify.json()["account"]["verified_assocs"] == [
        {"cluster": "", "account": "grp-a", "partition": "", "qos": ""}
    ]


# ── read ─────────────────────────────────────────────────────────────────


def test_get_shows_only_the_callers_account_and_links(env):
    client = env[0]
    assert client.get("/umrs/sites/me/hpc", headers=ALICE).json() == {"account": None, "links": []}
    _register(client)
    client.post(
        "/umrs/sites/me/hpc/links",
        json={"project_id": "p1", "slurm_account": "grp-a"},
        headers=ALICE,
    )
    mine = client.get("/umrs/sites/me/hpc", headers=ALICE).json()
    assert mine["account"]["username"] == "alice" and len(mine["links"]) == 1
    assert client.get("/umrs/sites/me/hpc", headers=BOB).json() == {"account": None, "links": []}


# ── links ────────────────────────────────────────────────────────────────


def _link(client, project="p1", account="grp-a", headers=ALICE):
    return client.post(
        "/umrs/sites/me/hpc/links",
        json={"project_id": project, "slurm_account": account},
        headers=headers,
    )


def test_a_member_links_an_allocation_the_site_listed(env):
    client = env[0]
    _register(client)
    response = _link(client)
    assert response.status_code == 200, response.text
    link = response.json()["link"]
    assert tuple(link) == LINK_COLUMNS
    assert (
        link["venue"],
        link["project_id"],
        link["username"],
        link["slurm_account"],
        link["status"],
    ) == (
        SITE,
        "p1",
        "alice",
        "grp-a",
        "active",
    )


def test_a_link_needs_an_active_account(env):
    client = env[0]
    client.post("/umrs/sites/me/hpc/transport-key", headers=ALICE)  # pending, not verified
    assert _code(_link(client)) == "hpcaccess.not_enrolled"
    assert _code(_link(client, headers=BOB)) == "hpcaccess.not_enrolled"


def test_a_link_needs_live_membership(env):
    client = env[0]
    _register(client)
    response = _link(client, project="p-other")
    assert (response.status_code, _code(response)) == (403, "projects.forbidden")


def test_unreadable_memberships_fail_closed(env, monkeypatch):
    client = env[0]
    _register(client)
    monkeypatch.setattr(api, "_live_memberships", lambda _req, _user: None)
    assert _code(_link(client)) == "projects.unavailable"


def test_a_link_needs_an_allocation_the_site_listed(env):
    client = env[0]
    _register(client)
    response = _link(client, account="grp-someone-elses")
    assert (response.status_code, _code(response)) == (403, "hpcaccess.allocation_not_at_site")


def test_unlink_removes_only_the_callers_link(env):
    client, store, _gate, _state, _keys = env
    _register(client)
    _link(client)
    assert client.delete("/umrs/sites/me/hpc/links/p1", headers=BOB).status_code == 404
    response = client.delete("/umrs/sites/me/hpc/links/p1", headers=ALICE)
    assert response.json() == {"status": "deleted"}
    assert store.list_links("alice") == []
    assert (
        _code(client.delete("/umrs/sites/me/hpc/links/p1", headers=ALICE)) == "projects.not_found"
    )


# ── revoke ───────────────────────────────────────────────────────────────


def test_revoke_destroys_the_key_and_suspends_links(env):
    client, store, _gate, _state, keys = env
    key, _ = _register(client)
    _link(client)
    bob_key = client.post("/umrs/sites/me/hpc/transport-key", headers=BOB).json()
    response = client.delete("/umrs/sites/me/hpc", headers=ALICE)
    assert response.json() == {"status": "revoked"}
    assert not (keys / SITE / "alice" / key["key_id"]).exists()
    assert (keys / SITE / "bob" / bob_key["key_id"]).exists()
    assert store.get_account("alice")["status"] == "revoked"
    assert [x["status"] for x in store.list_links("alice")] == ["suspended"]
    again = client.post("/umrs/sites/me/hpc/verify", json={"login": "alice1"}, headers=ALICE)
    assert _code(again) == "hpcaccess.not_enrolled"


def test_revoke_without_an_account_is_not_enrolled(env):
    client = env[0]
    assert _code(client.delete("/umrs/sites/me/hpc", headers=ALICE)) == "hpcaccess.not_enrolled"


def test_a_new_key_after_revoke_starts_a_new_registration(env):
    client, store, _gate, _state, _keys = env
    first, _ = _register(client)
    client.delete("/umrs/sites/me/hpc", headers=ALICE)
    again = client.post("/umrs/sites/me/hpc/transport-key", headers=ALICE).json()
    assert again["key_id"] != first["key_id"]
    assert store.get_account("alice")["status"] == "pending"


def test_no_answer_ever_carries_a_private_key(env):
    client = env[0]
    texts = [client.post("/umrs/sites/me/hpc/transport-key", headers=ALICE).text]
    texts.append(_register(client)[1].text)
    texts.append(_link(client).text)
    texts.append(client.get("/umrs/sites/me/hpc", headers=ALICE).text)
    texts.append(client.delete("/umrs/sites/me/hpc", headers=ALICE).text)
    assert not [t for t in texts if "PRIVATE KEY" in t or "BEGIN OPENSSH" in t]


def test_the_store_needs_the_shared_pool(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    monkeypatch.setattr(api, "_load_site", _user_site)
    app = FastAPI()
    app.include_router(api.router)
    response = TestClient(app).get("/umrs/sites/me/hpc", headers=ALICE)
    assert (response.status_code, _code(response)) == (503, "projects.unavailable")


def test_the_gate_manifest_is_the_shipped_gate_file():
    hpc_gate = pytest.importorskip("r4_contracts.hpc_gate")
    assert api._gate_manifest() == (hpc_gate.GATE_VERSION, hpc_gate.gate_sha256())


def test_a_reverify_suspends_links_to_allocations_the_site_dropped(env):
    client, store, gate, _state, _keys = env
    _register(client)
    _link(client, project="p1", account="grp-a")
    _link(client, project="p2", account="grp-b")
    gate.answers["hello"] = _hello(assocs=[ASSOCS[1]])  # grp-a is gone
    _key, verify = _register(client)
    assert verify.status_code == 200
    statuses = {x["project_id"]: x["status"] for x in store.list_links("alice")}
    assert statuses == {"p1": "suspended", "p2": "active"}


def test_the_probe_result_is_recorded_even_when_hello_fails(env):
    client, store, gate, _state, _keys = env
    gate.answers["hello"] = _hello(login="someone-else")
    _register(client)
    assert store.get_account("alice")["gate_probe"] == "refused"


def test_an_identity_that_cannot_name_a_key_path_is_not_enrolled(env):
    client, store, _gate, _state, keys = env
    headers = {"X-User-ID": "../evil", "X-Verified-Project-ID": "p1"}
    response = client.post("/umrs/sites/me/hpc/transport-key", headers=headers)
    assert (response.status_code, _code(response)) == (403, "hpcaccess.not_enrolled")
    assert (store.list_accounts(), keys.exists()) == ([], False)


# ── B1 review fixes (2026-11-08) ─────────────────────────────────────────


def _key_arg(argv: list[str]) -> str:
    return argv[argv.index("-i") + 1]


def _verify(client, headers=ALICE, login="alice1"):
    return client.post("/umrs/sites/me/hpc/verify", json={"login": login}, headers=headers)


@pytest.mark.parametrize(
    ("probe", "hello"),
    [
        ((0, b'{"ok": true}', b""), _hello()),  # a ForceCommand answers the probe
        ((255, b"", b"Permission denied (publickey)"), _hello()),
        (PROBE_REFUSED, _hello(login="someone-else")),
        (PROBE_REFUSED, _hello(gate_sha256="cd" * 32)),
        (PROBE_REFUSED, (4, b"", b'{"error": "log_unwritable", "detail": "x"}')),
    ],
)
def test_a_failed_reverify_never_leaves_the_account_active(env, probe, hello):
    client, store, gate, _state, _keys = env
    _register(client)
    assert store.get_account("alice")["status"] == "active"
    gate.answers.update(probe=probe, hello=hello)
    assert _verify(client).status_code in (403, 409)
    assert store.get_account("alice")["status"] == "suspended"
    link = client.post(
        "/umrs/sites/me/hpc/links",
        json={"project_id": "p1", "slurm_account": "grp-a"},
        headers=ALICE,
    )
    assert _code(link) == "hpcaccess.not_enrolled"
    gate.answers.update(probe=PROBE_REFUSED, hello=_hello())
    assert _verify(client).json()["account"]["status"] == "active"


def test_the_probe_counts_only_on_the_gates_refusal_exit(env):
    client, store, gate, _state, _keys = env
    gate.answers["probe"] = (1, b"", b'{"error": "gate_probe_refusal", "detail": "wrapper"}')
    _key, verify = _register(client)
    assert _code(verify) == "hpcaccess.gate_refused"
    assert store.get_account("alice")["gate_probe"] == "bypassed"


def test_a_gate_for_another_site_is_refused(env):
    client, store, gate, _state, _keys = env
    gate.answers["hello"] = _hello(site="other-site")
    _key, verify = _register(client)
    assert _code(verify) == "hpcaccess.gate_refused"
    assert "configured for site 'other-site'" in json.loads(verify.text)["detail"]
    assert store.get_account("alice")["status"] == "pending"


@pytest.mark.parametrize(
    ("openssh", "signer", "ok"),
    [
        ("OpenSSH_7.4p1", "ssh-ed25519", False),
        ("OpenSSH_8.0p1", "ssh-ed25519", False),
        ("OpenSSH_8.1p1", "ssh-ed25519", True),
        ("OpenSSH_8.1p1", "sk-ssh-ed25519@openssh.com", False),
        ("OpenSSH_8.2p1", "sk-ssh-ed25519@openssh.com", True),
        ("", "ssh-ed25519", False),
    ],
)
def test_the_sites_openssh_must_meet_the_floor(env, openssh, signer, ok):
    client, store, gate, _state, _keys = env
    signers = [{"fingerprint": "SHA256:sig", "type": signer}]
    gate.answers["hello"] = _hello(openssh=openssh, signers=signers)
    _key, verify = _register(client)
    assert (verify.status_code == 200) is ok, verify.text
    if not ok:
        assert (verify.status_code, _code(verify)) == (409, "hpcaccess.gate_version")
        assert store.get_account("alice")["status"] == "pending"


def test_a_second_verify_at_once_is_refused_busy(env):
    client, _store, gate, _state, _keys = env
    client.post("/umrs/sites/me/hpc/transport-key", headers=ALICE)
    lock = api._VERIFY_LOCKS.setdefault((SITE, "alice"), __import__("threading").Lock())
    lock.acquire()
    try:
        busy = _verify(client)
    finally:
        lock.release()
    assert _code(busy) == "hpcaccess.gate_refused"
    assert "already running" in json.loads(busy.text)["detail"] and gate.calls == []
    assert _verify(client).status_code == 200


def test_repeated_rotations_keep_at_most_two_keys(env):
    client, store, _gate, _state, keys = env
    _register(client)
    for _ in range(5):
        client.post("/umrs/sites/me/hpc/transport-key", json={"rotate": True}, headers=ALICE)
    private = [p for p in (keys / SITE / "alice").iterdir() if "." not in p.name]
    assert len(private) == 2
    assert store.get_account("alice")["transport_key_id"] in {p.name for p in private}


def test_a_rotation_during_verify_keeps_its_new_key(env, monkeypatch):
    """verify drops the keys its key replaced, never the account's current key."""
    client, store, gate, _state, keys = env
    _register(client)
    rotated: dict = {}
    real = store.record_verified

    def rotate_then_record(user, key_id, verified):
        url = "/umrs/sites/me/hpc/transport-key"
        rotated.update(client.post(url, json={"rotate": True}, headers=ALICE).json())
        return real(user, key_id, verified)

    monkeypatch.setattr(store, "record_verified", rotate_then_record)
    monkeypatch.setattr(api, "_make_store", lambda _req, _venue: store)
    _verify(client)
    assert (keys / SITE / "alice" / rotated["key_id"]).exists()
    assert store.get_account("alice")["transport_key_id"] == rotated["key_id"]


def test_a_keygen_that_fails_is_not_enrolled_not_a_500(env, monkeypatch):
    import subprocess

    client, store, _gate, _state, _keys = env

    def broken(*_a, **_k):
        raise subprocess.CalledProcessError(1, ["ssh-keygen"])

    real = api.transport_keys.generate
    monkeypatch.setattr(
        api.transport_keys, "generate", lambda v, u: real(v, u, runner=broken)
    )
    response = client.post("/umrs/sites/me/hpc/transport-key", headers=ALICE)
    assert (response.status_code, _code(response)) == (403, "hpcaccess.not_enrolled")
    assert store.list_accounts() == []


def test_a_key_without_its_public_half_is_not_enrolled_not_a_500(env):
    client, _store, gate, _state, keys = env
    key = client.post("/umrs/sites/me/hpc/transport-key", headers=ALICE).json()
    (keys / SITE / "alice" / (key["key_id"] + ".pub")).unlink()
    for response in (
        client.post("/umrs/sites/me/hpc/transport-key", headers=ALICE),
        _verify(client),
    ):
        assert (response.status_code, _code(response)) == (403, "hpcaccess.not_enrolled")
    assert gate.calls == []


def test_a_site_name_longer_than_the_venue_column_is_refused(env):
    client, store, gate, state, keys = env
    site = _user_site()
    site.name = "a" * 33
    state["site"] = site
    response = client.post("/umrs/sites/me/hpc/transport-key", headers=ALICE)
    assert (response.status_code, _code(response)) == (403, "hpcaccess.mode_undeclared")
    assert store.list_accounts() == [] and gate.calls == [] and not keys.exists()


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/umrs/sites/me/hpc/transport-key", {"rotate": True}),
        ("post", "/umrs/sites/me/hpc/verify", {"login": "alice1"}),
        ("post", "/umrs/sites/me/hpc/links", {"project_id": "p1", "slurm_account": "grp-a"}),
        ("delete", "/umrs/sites/me/hpc/links/p1", None),
        ("delete", "/umrs/sites/me/hpc", None),
    ],
)
def test_an_agent_marked_request_never_changes_the_account(env, method, path, body):
    client, store, gate, _state, keys = env
    _register(client)
    before = (store.get_account("alice"), sorted(p.name for p in keys.rglob("tk-*")))
    calls = len(gate.calls)
    headers = {**ALICE, "x-r4-actor": "agent"}
    kwargs = {"json": body} if body is not None else {}
    response = getattr(client, method)(path, headers=headers, **kwargs)
    assert (response.status_code, _code(response)) == (403, "hpcaccess.person_required")
    assert (store.get_account("alice"), sorted(p.name for p in keys.rglob("tk-*"))) == before
    assert len(gate.calls) == calls
    assert client.get("/umrs/sites/me/hpc", headers=headers).status_code == 200


@pytest.mark.parametrize(
    "marker", [{"x-r4-actor": "companion"}, {"x-r4-scope": "e30.c2ln"}]
)
def test_any_agent_or_work_level_marker_is_refused(env, marker):
    client, store, gate, _state, _keys = env
    key = client.post("/umrs/sites/me/hpc/transport-key", headers={**ALICE, **marker})
    assert _code(key) == "hpcaccess.person_required"
    assert store.list_accounts() == [] and gate.calls == []
    person = client.post("/umrs/sites/me/hpc/transport-key", headers={**ALICE, "x-r4-actor": "user"})
    assert person.status_code == 200
