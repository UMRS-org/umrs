"""Server pieces of the B2.0 wire contract (ADR-112; plan §5 B2.0).

- the gate client's verb lines and per-verb timeouts (§5);
- ``expected_step_config`` (the gate's shared code) is today's
  ``_assemble_step_config`` minus the staged lane's platform keys, for the
  repository's ``workflow.json`` files, with and without params (§1.3);
- the site keys B2 adds (``hpc_access.user_account.profile``,
  ``step_executor.image_digests``) and the published profile (§2);
- migration 0136 and the run record's intent fields (§10.4);
- the runner binds a run with an intent to the person's gate (§10.5);
- the server's strict intent decoding (§10.3).
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest
from umrs.hpc_accounts import gate_client
from umrs.hpc_accounts.intents import IntentInvalid, decode_signed_intent
from umrs.orchestrator.site_config import SiteConfig, _parse_user_account

from r4_contracts import hpc_gate

UMRS = Path(__file__).resolve().parents[1]
SIGNATURE = "-----BEGIN SSH SIGNATURE-----\nU1NIU0lH\n-----END SSH SIGNATURE-----"


# ── gate client (§5) ─────────────────────────────────────────────────────────


def _site() -> SiteConfig:
    return SiteConfig(name="site-x", extra={"ssh_target": "login-0", "ssh_port": 22})


def test_a_verb_line_is_one_remote_word_joined_from_checked_words():
    argv = gate_client.gate_argv(_site(), "alice1", "/k", "mkwd", {}, ("a" * 64, "s1"))
    assert argv[-1] == f"mkwd {'a' * 64} s1"
    assert gate_client.gate_line("sweep", ("--older-than", "3600")) == "sweep --older-than 3600"
    assert gate_client.gate_argv(_site(), "alice1", "/k", "stage-in", {}, ("wd",))[-1] == "stage-in wd"


@pytest.mark.parametrize("bad", ["a b", "$(id)", "-o", "a;b", "", "x" * 200])
def test_a_verb_argument_never_becomes_shell_or_an_option(bad):
    with pytest.raises(ValueError):
        gate_client.gate_argv(_site(), "alice1", "/k", "status", {}, (bad,))


def test_each_verb_waits_for_the_gates_own_deadline_plus_slack():
    assert gate_client.verb_timeout("hello") == gate_client.DEFAULT_TIMEOUT_S
    assert gate_client.verb_timeout("submit") == 630
    assert gate_client.verb_timeout("stage-in") == gate_client.verb_timeout("collect-out") == 7230
    assert gate_client.verb_timeout("rmwd") == gate_client.verb_timeout("sweep") == 1830
    assert gate_client.verb_timeout("status") == 90


def test_the_client_deadlines_are_the_gates():
    assert gate_client.GATE_DEADLINES == hpc_gate.DEADLINES


# ── expected_step_config is the runner's assembly minus the platform keys ────


def _workflow_files() -> list[Path]:
    return sorted([*UMRS.glob("demos/**/workflow.json"), *UMRS.glob("src/**/workflow.json")])


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: str(p.relative_to(UMRS)))
@pytest.mark.parametrize("params", [{}, {"project_id": "p", "threshold": 0.5, "api_key": "k"}])
def test_expected_step_config_pins_the_runners_assembly(path, params):
    from umrs.server.remote_slurm_step_executor import _PLATFORM_CONFIG_KEYS
    from umrs.server.workflow_runner import _assemble_step_config

    steps = json.loads(path.read_text()).get("steps") or []
    for (name, _after, _chained), raw in zip(hpc_gate.step_graph(steps), steps, strict=True):
        runner = _assemble_step_config({**raw, "name": name}, params)
        for key in _PLATFORM_CONFIG_KEYS:
            runner.pop(key, None)
        assert hpc_gate.expected_step_config(raw, name, params) == runner


def test_the_repository_carries_workflow_files_to_pin():
    assert len(_workflow_files()) >= 5


# ── site keys (§2) ───────────────────────────────────────────────────────────


def test_the_user_account_profile_block_takes_only_profile_keys():
    parsed = _parse_user_account({"signing_domain": "x.org", "profile": {"image_archive_dir": "/a"}})
    assert parsed.profile == {"image_archive_dir": "/a"}
    with pytest.raises(ValueError, match="unknown key"):
        _parse_user_account({"profile": {"script": "/bin/sh"}})


def test_the_schema_knows_the_b2_site_keys():
    schema = json.loads((UMRS / "src/umrs/orchestrator/site-config.schema.json").read_text())
    text = json.dumps(schema)
    assert '"image_digests"' in text and '"profile"' in text


def test_the_profile_override_wins_one_level_deep():
    from umrs.hpc_accounts.site_profile import _override

    base = {"slurm": {"path": "/usr/bin:/bin", "queue_seconds": 1}, "watch_seconds": 15}
    merged = _override(base, {"slurm": {"path": "/opt/slurm/bin"}, "watch_seconds": 30})
    assert merged == {"slurm": {"path": "/opt/slurm/bin", "queue_seconds": 1}, "watch_seconds": 30}


# ── migration 0136 and the run record (§10.4) ────────────────────────────────


def test_0136_adds_the_intent_columns_and_one_run_per_intent():
    from umrs import migrations

    versions = [v for v, _ in migrations.get_migration_files()]
    assert versions.count("0136") == 1 and versions.index("0136") == versions.index("0135") + 1
    sql = dict(migrations.get_migration_files())["0136"].read_text()
    for column in ("intent_b64 TEXT", "intent_signature TEXT", "intent_id CHAR(64)"):
        assert f"ADD COLUMN IF NOT EXISTS {column}" in sql
    assert "CREATE UNIQUE INDEX IF NOT EXISTS" in sql and "WHERE intent_id IS NOT NULL" in sql


def test_the_run_status_shows_the_intent_id_only():
    from umrs.server.workflow_execution_registry import WorkflowExecutionRegistry

    record = WorkflowExecutionRegistry().create(
        "wf", "alice", execution_id="wf-0123456789ab",
        intent={"b64": "cGF5bG9hZA==", "signature": SIGNATURE, "id": "c" * 64},
    )
    assert (record.intent_b64, record.intent_signature) == ("cGF5bG9hZA==", SIGNATURE)
    view = record.to_dict()
    assert view["intent_id"] == "c" * 64
    assert "cGF5bG9hZA==" not in json.dumps(view) and "SSH SIGNATURE" not in json.dumps(view)


def test_the_durable_record_persists_the_three_columns_and_round_trips_the_id():
    from umrs.workflow import pg_registry

    assert "intent_b64, intent_signature, intent_id" in pg_registry._COLUMNS
    record = pg_registry._record_from_payload(
        {"execution_id": "wf-0123456789ab", "intent_id": "c" * 64}
    )
    assert record.intent_id == "c" * 64


class _OneRowPool:
    """A pool whose cursor answers every SELECT with *row* (no database)."""

    def __init__(self, row: dict) -> None:
        self.row = row

    def getconn(self):
        from types import SimpleNamespace

        cursor = SimpleNamespace(
            execute=lambda *_a: None, fetchone=lambda: self.row, fetchall=lambda: [self.row]
        )
        return _Conn(cursor)

    def putconn(self, _conn) -> None:
        return None


class _Conn:
    def __init__(self, cursor) -> None:
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def cursor(self, **_kw):
        return self._cursor


@pytest.mark.parametrize("path", ["get", "list"])
def test_a_record_loaded_from_postgres_carries_its_intent_envelope(path):
    """Fixer finding 2026-11-08: ``to_dict()`` leaves the payload and the
    signature out of ``record_json``; a record loaded on another front (or
    after a restart) gets them from their own columns, else its gated steps
    would hand the gate an empty envelope ('intent unreadable')."""
    pytest.importorskip("psycopg2")
    from umrs.workflow.pg_registry import PgWorkflowExecutionRegistry

    row = {
        "execution_id": "wf-0123456789ab",
        "record_json": {"execution_id": "wf-0123456789ab", "user_id": "alice", "intent_id": "c" * 64},
        "intent_b64": "cGF5bG9hZA==",
        "intent_signature": SIGNATURE,
        "intent_id": "c" * 64,
    }
    registry = PgWorkflowExecutionRegistry(_OneRowPool(row))
    record = (
        registry.get("wf-0123456789ab") if path == "get" else registry.list(user_id="alice")[0]
    )
    assert (record.intent_b64, record.intent_signature, record.intent_id) == (
        "cGF5bG9hZA==", SIGNATURE, "c" * 64,
    )  # fmt: skip


def test_the_durable_spec_carries_the_intent_to_every_dispatch():
    from umrs.hpc_accounts.intents import intent_from_spec

    spec = {"intent": {"b64": "x", "signature": SIGNATURE, "id": "c" * 64}}
    assert intent_from_spec(spec) == spec["intent"]
    assert intent_from_spec({}) is None and intent_from_spec({"intent": {"b64": "x"}}) is None


# ── the runner binds a gated run (§10.5) ─────────────────────────────────────


def test_a_run_with_an_intent_binds_the_persons_gate(monkeypatch):
    from types import SimpleNamespace

    from umrs.server import workflow_runner as wr
    from umrs.server.workflow_execution_registry import ExecutionRecord
    from umrs.shared import hpc_user_work_access as access

    monkeypatch.setattr(access, "current_site", lambda: SimpleNamespace(name="site-x"))
    record = ExecutionRecord(
        execution_id="wf-0123456789ab", workflow_name="wf", user_id="alice",
        intent_b64="e30=", intent_signature=SIGNATURE, intent_id="c" * 64,
    )
    gated = wr._gated_binding(record)["gated"]
    assert (gated.venue, gated.username, gated.intent_id) == ("site-x", "alice", "c" * 64)
    assert gated.transport == "gated:site-x:alice"
    plain = ExecutionRecord(execution_id="wf-1", workflow_name="wf", user_id="alice")
    assert wr._gated_binding(plain) == {}


# ── strict decoding (§10.3) ──────────────────────────────────────────────────


def _payload(**over) -> str:
    body = {"v": 1, **over}
    return base64.b64encode(json.dumps(body).encode()).decode()


@pytest.mark.parametrize(
    ("intent", "signature", "reason"),
    [
        ("not base64!", SIGNATURE, "bad_envelope"),
        (_payload(), "not armored", "bad_envelope"),
        (_payload(), SIGNATURE, "bad_schema"),
        (base64.b64encode(b'{"v":1,"v":1}').decode(), SIGNATURE, "bad_schema"),
    ],
)
def test_decoding_refuses_by_name(intent, signature, reason):
    with pytest.raises(IntentInvalid) as info:
        decode_signed_intent(intent, signature)
    assert info.value.reason == reason


# ── the proposal's source manifest (§3) ──────────────────────────────────────


def test_a_source_path_that_is_not_utf8_cannot_be_signed(tmp_path):
    from umrs.server.intent_proposal import src_manifest

    (tmp_path / "m.py").write_text("class S: pass\n")
    assert [f["path"] for f in src_manifest(str(tmp_path), {"s1": {"entrypoint": "m:S"}})["files"]] == ["m.py"]
    with open(os.path.join(os.fsencode(tmp_path), b"bad-\xff.py"), "wb") as handle:
        handle.write(b"x")
    with pytest.raises(IntentInvalid) as info:
        src_manifest(str(tmp_path), {"s1": {"entrypoint": "m:S"}})
    assert info.value.reason == "bad_src_path"
