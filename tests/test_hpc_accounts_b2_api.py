"""``/umrs/sites/me/hpc`` wave B2 additions (ADR-112; plan §5 B2.0 §2, §10.3).

- ``GET /site-profile`` publishes the gate site profile r4 installs (read
  only; its bytes are the gate's canonical encoding and the sha their digest);
- ``POST /verify`` refuses ``gate_version`` (``site_profile_missing``) for a
  gate without its profile and says whether the installed one is current;
- ``GET /gate`` is a live ``hello`` with no state change;
- ``POST /control`` relays a signed ``revoke`` / ``rotate`` to the person's
  gate, the person's own act.
"""

from __future__ import annotations

import base64
import hashlib
import json

import pytest
from test_hpc_accounts_api import ALICE, _code, _hello, _register, _user_site
from test_hpc_accounts_api import env as _accounts_env  # noqa: F401 — fixture

from r4_contracts import hpc_gate

PROFILE_OVERRIDES = {"image_archive_dir": "/shared/umrs/archives"}


@pytest.fixture
def b2(request):
    """The B1 route tests' app (``env``): a scripted gate, real keys."""
    return request.getfixturevalue("_accounts_env")


def _profiled_site():
    site = _user_site(profile=dict(PROFILE_OVERRIDES))
    site.step_executor.partition = "gpu"
    return site


def test_the_site_profile_is_published_as_the_gates_canonical_bytes(b2):
    client, _store, gate, state, _keys = b2
    state["site"] = _profiled_site()
    r = client.get("/umrs/sites/me/hpc/site-profile", headers=ALICE)
    assert r.status_code == 200, r.text
    data = base64.b64decode(r.json()["bytes_b64"])
    assert r.json()["sha256"] == hashlib.sha256(data).hexdigest()
    profile = hpc_gate.validate_site_profile(json.loads(data))
    assert hpc_gate.site_profile_bytes(profile) == data
    assert (profile["site"], profile["domain"]) == ("bdrt-front", "platform.example.org")
    assert profile["image_archive_dir"] == "/shared/umrs/archives"
    assert list(profile["partitions"]) == ["gpu"]
    assert gate.calls == []  # read only: no gate call


def test_the_default_slurm_path_passes_the_gates_own_check_on_this_host():
    """Fixer finding 2026-11-08: ``/bin`` is a link on usrmerge hosts (RHEL
    8/9, Fedora, current Debian and Ubuntu), and the gate refuses a linked
    ``slurm.path`` entry; the published default must pass the gate as is."""
    import os

    from umrs.hpc_accounts.site_profile import DEFAULT_SLURM_PATH, build_site_profile

    entries = DEFAULT_SLURM_PATH.split(":")
    assert not [entry for entry in entries if os.path.islink(entry)]
    profile = build_site_profile(_profiled_site())
    assert profile["slurm"]["path"] == DEFAULT_SLURM_PATH
    hpc_gate.check_slurm_path(profile)


def test_a_site_without_an_archive_directory_publishes_no_profile(b2):
    client, _store, _gate, _state, _keys = b2
    r = client.get("/umrs/sites/me/hpc/site-profile", headers=ALICE)
    assert (r.status_code, _code(r)) == (409, "hpcaccess.gate_version")


def test_the_site_profile_needs_user_accounts_allowed(b2):
    from test_hpc_accounts_api import _site

    client, _store, _gate, state, _keys = b2
    state["site"] = _site(["service_account"])
    r = client.get("/umrs/sites/me/hpc/site-profile", headers=ALICE)
    assert _code(r) == "hpcaccess.mode_undeclared"


def test_verify_refuses_a_gate_without_its_site_profile(b2):
    client, store, gate, _state, _keys = b2
    gate.answers["hello"] = _hello(site_profile_sha256="")
    _key, verify = _register(client)
    assert (verify.status_code, _code(verify)) == (409, "hpcaccess.gate_version")
    assert "site_profile_missing" in verify.text
    assert store.get_account("alice")["status"] != "active"


def test_verify_says_whether_the_installed_profile_is_current(b2):
    client, _store, gate, state, _keys = b2
    state["site"] = _profiled_site()
    _key, verify = _register(client)
    assert verify.json()["site_profile_current"] is False  # "ef"*32 is not it
    published = client.get("/umrs/sites/me/hpc/site-profile", headers=ALICE).json()["sha256"]
    gate.answers["hello"] = _hello(site_profile_sha256=published)
    _key, verify = _register(client)
    assert verify.json()["site_profile_current"] is True
    assert verify.json()["account"]["status"] == "active"


def test_get_gate_is_a_live_hello_with_no_state_change(b2):
    client, store, gate, _state, _keys = b2
    _register(client)
    before = store.get_account("alice")
    r = client.get("/umrs/sites/me/hpc/gate", headers=ALICE)
    assert r.status_code == 200, r.text
    data = r.json()
    assert (data["gate_sha256"], data["site_profile_sha256"]) == ("ab" * 32, "ef" * 32)
    assert data["current"] == {"gate": True, "site_profile": False}
    assert gate.calls[-1][-1] == "hello"
    assert store.get_account("alice") == before


def test_get_gate_needs_an_active_account(b2):
    client, _store, _gate, _state, _keys = b2
    r = client.get("/umrs/sites/me/hpc/gate", headers=ALICE)
    assert _code(r) == "hpcaccess.not_enrolled"


ENVELOPE = {"op": "revoke", "payload": "eyJ2IjoxfQ==", "signature": "-----BEGIN SSH SIGNATURE-----"}


def test_control_relays_the_signed_envelope_to_that_verb(b2):
    client, _store, gate, _state, _keys = b2
    _register(client)
    seen: list[bytes] = []
    original = gate.__call__

    async def recording(argv, stdin, timeout):
        seen.append(stdin)
        if argv[-1] == "revoke":
            return 0, b'{"removed": 1}', b""
        return await original(argv, stdin, timeout)

    gate.__class__ = type("Recording", (type(gate),), {"__call__": staticmethod(recording)})
    r = client.post("/umrs/sites/me/hpc/control", json=ENVELOPE, headers=ALICE)
    assert r.status_code == 200, r.text
    assert r.json() == {"result": {"removed": 1}}
    assert json.loads(seen[-1]) == {"payload": ENVELOPE["payload"], "signature": ENVELOPE["signature"]}


def test_control_surfaces_the_gates_refusal_by_name(b2):
    client, _store, gate, _state, _keys = b2
    _register(client)
    gate.answers["rotate"] = (4, b"", b'{"error": "replayed", "detail": "nonce used"}')
    r = client.post("/umrs/sites/me/hpc/control", json={**ENVELOPE, "op": "rotate"}, headers=ALICE)
    assert (r.status_code, _code(r)) == (403, "hpcaccess.gate_refused")
    assert "replayed" in r.text


@pytest.mark.parametrize("header", [{"x-r4-actor": "agent"}, {"x-r4-scope": "tok"}])
def test_control_is_the_persons_own_act(b2, header):
    client, _store, gate, _state, _keys = b2
    _register(client)
    calls = len(gate.calls)
    r = client.post("/umrs/sites/me/hpc/control", json=ENVELOPE, headers={**ALICE, **header})
    assert _code(r) == "hpcaccess.person_required" and len(gate.calls) == calls


def test_control_takes_only_revoke_or_rotate(b2):
    client, _store, _gate, _state, _keys = b2
    r = client.post("/umrs/sites/me/hpc/control", json={**ENVELOPE, "op": "submit"}, headers=ALICE)
    assert r.status_code == 422
