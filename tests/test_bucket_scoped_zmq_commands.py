"""ZMQ commands that name a bucket act only on buckets the caller may use.

Once client_id is bound to the CURVE session (2026-10-04), the bucket inside
a command was the next claim: PROCESS_OBJECTS events naming another user's
bucket ran that bucket's model (user_script / user_workflow included) on keys
the sender picked, EXPORT_EMBEDDINGS exported any bucket's vectors and keys,
and WAIT_OUTPUT watched any bucket. All three now use ``bucket_owned_by`` —
the rule the files API and the HPC lane share.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from umrs.core.TaskCmdStatus import TaskStatus
from umrs.server import processing
from umrs.server.step_staging import register_bucket_registry
from umrs.shared.storage import user_buckets as ub

ALICE_BUCKET = "alice.xp.images"
BOB_BUCKET = "bob.xp.images"


class _Registry:
    """BucketsCfg-shaped: provisioned buckets and their owners."""

    def __init__(self, owners: dict[str, str]):
        self.entries = {
            name: SimpleNamespace(s3_username=owner, data_base=None, table_name="t", db_name="d")
            for name, owner in owners.items()
        }

    def get(self, name, filter=None):  # noqa: A002, ARG002 — BucketsCfg's signature
        return self.entries.get(name)

    def find_by_output_path(self, output):
        name = output[: -len("-output")] if output.endswith("-output") else output
        return (name, self.entries.get(name)) if name in self.entries else (None, None)

    def update_num_objects(self, _name):
        pass

    def values(self):
        return list(self.entries.values())


class _NoPersonalBuckets:
    def owner_of(self, _bucket):
        return None

    def can_access(self, _bucket, _user_id):
        return False


@pytest.fixture()
def buckets():
    registry = _Registry({ALICE_BUCKET: "alice", BOB_BUCKET: "bob"})
    register_bucket_registry(registry)
    ub.set_user_bucket_store(_NoPersonalBuckets())
    yield registry
    register_bucket_registry(None)
    ub.set_user_bucket_store(None)


# ── PROCESS_OBJECTS ─────────────────────────────────────────────────────────


@pytest.mark.usefixtures("buckets")
def test_events_naming_another_users_bucket_refuse_the_whole_request():
    statuses: dict[str, list] = {"alice": [], "bob": []}
    callbacks = {cid: statuses[cid].append for cid in statuses}
    entries = [
        ("alice", [f"s3://{ALICE_BUCKET}/a.png", f"s3://{BOB_BUCKET}/secret.png"]),
        ("bob", [f"s3://{BOB_BUCKET}/b.png"]),
    ]
    keys, members, refused = processing._build_bucket_membership(entries)
    assert refused == {"alice"}
    assert keys == {BOB_BUCKET: ["b.png"]}  # none of alice's keys, own bucket included
    assert members[BOB_BUCKET] == {"bob"}

    errored = set()
    for cid in refused:
        callbacks[cid](TaskStatus.ERROR)
        errored.add(cid)
    assert statuses["alice"] == [TaskStatus.ERROR]
    assert statuses["bob"] == []


@pytest.mark.usefixtures("buckets")
def test_own_bucket_events_pass():
    keys, members, refused = processing._build_bucket_membership(
        [("alice", [f"s3://{ALICE_BUCKET}/a.png", f"s3://{ALICE_BUCKET}/b.png"])]
    )
    assert refused == set()
    assert keys == {ALICE_BUCKET: ["a.png", "b.png"]}
    assert members[ALICE_BUCKET] == {"alice"}


def test_prepare_model_groups_errors_the_refused_client_only(buckets, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        processing,
        "_resolve_model_groups",
        lambda keys, _members, *_a: (seen.setdefault("keys", keys) and {}, set()),
    )
    statuses = {"alice": [], "bob": []}
    groups = processing.prepare_model_groups(
        None,
        [("alice", [f"s3://{BOB_BUCKET}/x.png"]), ("bob", [f"s3://{BOB_BUCKET}/y.png"])],
        buckets,
        "cpu",
        "process_objects",
        client_callbacks={cid: statuses[cid].append for cid in statuses},
    )
    assert groups == []
    assert seen["keys"] == {BOB_BUCKET: ["y.png"]}
    assert statuses == {"alice": [TaskStatus.ERROR], "bob": []}


# ── EXPORT_EMBEDDINGS ───────────────────────────────────────────────────────


def test_export_of_another_users_bucket_reads_as_missing(buckets):
    from umrs.server.sql import export_embeddings

    sent = []
    control_mq = SimpleNamespace(send_string=lambda text, _sender: sent.append(text))
    threads = MagicMock()  # the export's task bookkeeping
    export_embeddings(threads, "alice", f"{BOB_BUCKET};csv;minimal", buckets, control_mq, b"s", "k")
    reply = json.loads(sent[-1])
    assert reply == {"status": "error", "message": f"No bucket entry found for path: {BOB_BUCKET}"}


# ── WAIT_OUTPUT ─────────────────────────────────────────────────────────────


def test_wait_output_on_another_users_bucket_reads_as_unknown(buckets):
    from umrs.server.commands.wait_output import WaitOutputHandler

    sent, statuses = [], []
    ctx = SimpleNamespace(
        buckets_cfg=buckets,
        control_mq=SimpleNamespace(send_string=lambda text, _sender: sent.append(text)),
    )
    handler = WaitOutputHandler()
    handler._make_task = lambda _ctx, _cmd, _payload: (1, statuses.append)
    handler.execute(("alice", f"{BOB_BUCKET}-output", 5, b"s"), ctx)
    assert sent == ["NO_SUCH_BUCKET"]
    assert statuses == [TaskStatus.NO_SUCH_BUCKET]


# ── STATUS_ALL / LIST_WORKFLOWS scoped to the session; bootstrap retired ───


class _Grants:
    def __init__(self, admin=False):
        self.is_admin = admin


class _Entry:
    def __init__(self, user_id, admin=False):
        import time

        self.user_id = user_id
        self.grants = _Grants(admin)
        self.session_id = f"s-{user_id}"
        self.expires_at = time.time() + 600


def _status_mq(entries):
    import collections

    from umrs.core.MessageQueue import MessageQueue

    mq = MessageQueue.__new__(MessageQueue)
    mq._session_entries = collections.OrderedDict()
    mq._identity_provider_cache = (SimpleNamespace(lookup_session=entries.get),)
    mq._auth_sessions = {}
    mq.status_store = {"alice": {"t1": "done"}, "bob": {"t9": "running"}}
    mq.task_manager = None
    mq.use_status_all_cache = False
    mq.sent = []
    mq.send_string = lambda text, _sender: mq.sent.append(text)
    return mq


def test_status_all_shows_a_user_only_their_own_entry():
    mq = _status_mq({"s-alice": _Entry("alice")})
    mq._dispatch_session_id = "s-alice"
    mq._parse_status_all(["STATUS_ALL"], b"r")
    assert sorted(json.loads(mq.sent[-1])["clients"]) == ["alice"]


def test_status_all_shows_an_admin_every_client():
    mq = _status_mq({"s-maint": _Entry("maint", admin=True)})
    mq._dispatch_session_id = "s-maint"
    mq._parse_status_all(["STATUS_ALL"], b"r")
    assert sorted(json.loads(mq.sent[-1])["clients"]) == ["alice", "bob"]


def test_status_all_without_a_session_is_refused():
    mq = _status_mq({})
    mq._dispatch_session_id = None
    mq._parse_status_all(["STATUS_ALL"], b"r")
    assert [json.loads(m) for m in mq.sent] == [
        {"status": "error", "message": "no authenticated ZMQ session"}
    ]


def test_list_workflows_payload_is_forced_to_the_caller_for_non_admins():
    from umrs.server.commands.workflow_commands import _scoped_list_payload

    scoped = json.loads(
        _scoped_list_payload(
            json.dumps(
                {
                    "user_id": "bob",
                    "include_shared_for": "bob",
                    "verified_project": "p",
                    "name": "x",
                }
            ),
            "alice",
            is_admin=False,
        )
    )
    assert scoped == {
        "user_id": "alice",
        "include_shared_for": "alice",
        "include_system": True,
        "name": "x",
    }
    assert json.loads(_scoped_list_payload("", "alice", is_admin=False)) == {
        "user_id": "alice",
        "include_shared_for": "alice",
        "include_system": True,
    }
    # A non-admin's ``all_users`` claim is stripped, never honoured.
    assert json.loads(_scoped_list_payload('{"all_users": true}', "alice", is_admin=False)) == {
        "user_id": "alice",
        "include_shared_for": "alice",
        "include_system": True,
    }


def test_list_workflows_payload_is_scoped_for_admins_by_default():
    from umrs.server.commands.workflow_commands import _scoped_list_payload

    scoped = json.loads(_scoped_list_payload('{"user_id": "bob"}', "maint", is_admin=True))
    assert scoped == {
        "user_id": "maint",
        "include_shared_for": "maint",
        "include_system": True,
    }
    assert json.loads(_scoped_list_payload("", "maint", is_admin=True)) == {
        "user_id": "maint",
        "include_shared_for": "maint",
        "include_system": True,
    }


def test_list_workflows_all_users_is_the_explicit_admin_audit_view():
    from umrs.server.commands.workflow_commands import _scoped_list_payload

    assert json.loads(_scoped_list_payload('{"all_users": true}', "maint", is_admin=True)) == {}
    # Admin filters survive the audit view; only the marker key is removed.
    assert json.loads(
        _scoped_list_payload(
            '{"all_users": true, "user_id": "bob", "name": "x"}', "maint", is_admin=True
        )
    ) == {"user_id": "bob", "name": "x"}
    # A non-true value never opens the audit view.
    assert json.loads(_scoped_list_payload('{"all_users": "true"}', "maint", is_admin=True)) == {
        "user_id": "maint",
        "include_shared_for": "maint",
        "include_system": True,
    }


def test_list_workflows_handler_scopes_through_the_session_role(monkeypatch):
    from umrs.server.commands import workflow_commands as wc

    seen = {}
    monkeypatch.setattr(
        wc,
        "handle_workflow_command",
        lambda _cmd, _cid, payload, **_k: seen.setdefault("p", payload) and "{}",
    )
    ctx = SimpleNamespace(
        control_mq=SimpleNamespace(
            get_auth_context=lambda _cid: SimpleNamespace(is_admin=False),
            send_string=lambda *_a: None,
        )
    )
    handler = wc.ListWorkflowsHandler()
    handler._make_task = lambda *_a: (1, lambda _s: None)
    handler.execute(("alice", "{}", b"r"), ctx)
    assert json.loads(seen["p"]) == {
        "user_id": "alice",
        "include_shared_for": "alice",
        "include_system": True,
    }


def test_list_workflows_handler_scopes_admin_sessions_too(monkeypatch):
    from umrs.server.commands import workflow_commands as wc

    seen = {}
    monkeypatch.setattr(
        wc,
        "handle_workflow_command",
        lambda _cmd, _cid, payload, **_k: seen.setdefault("p", payload) and "{}",
    )
    auth = SimpleNamespace(is_admin=True)
    ctx = SimpleNamespace(
        control_mq=SimpleNamespace(
            get_auth_context=lambda _cid: auth,
            send_string=lambda *_a: None,
        )
    )
    handler = wc.ListWorkflowsHandler()
    handler._make_task = lambda *_a: (1, lambda _s: None)
    handler.execute(("maint", "{}", b"r"), ctx)
    assert json.loads(seen["p"]) == {
        "user_id": "maint",
        "include_shared_for": "maint",
        "include_system": True,
    }
    seen.clear()
    handler.execute(("maint", '{"all_users": true}', b"r"), ctx)
    assert json.loads(seen["p"]) == {}


def test_get_bootstrap_context_is_retired_on_zmq():
    from umrs.core.MessageQueue import MessageQueue
    from umrs.core.TaskCmdStatus import TaskCmd

    assert not hasattr(TaskCmd, "GET_BOOTSTRAP_CONTEXT")
    assert "GET_BOOTSTRAP_CONTEXT" not in MessageQueue._CONTROL_PARSERS


# ── review fixes (2026-10-04) ───────────────────────────────────────────────


def test_each_bucket_is_checked_once_per_client(buckets, monkeypatch):
    calls = []
    real = processing._client_may_use
    monkeypatch.setattr(
        processing, "_client_may_use", lambda b, c: calls.append((b, c)) or real(b, c)
    )
    events = [f"s3://{ALICE_BUCKET}/{i}.png" for i in range(500)]
    processing._build_bucket_membership([("alice", events)])
    assert calls == [(ALICE_BUCKET, "alice")]


def test_a_malformed_event_refuses_only_its_sender(buckets):
    keys, _members, refused = processing._build_bucket_membership(
        [("alice", ["s3://no-key-here"]), ("bob", [f"s3://{BOB_BUCKET}/b.png"])]
    )
    assert refused == {"alice"}
    assert keys == {BOB_BUCKET: ["b.png"]}


def test_a_users_list_keeps_the_platform_system_workflows(monkeypatch):
    from umrs.server import workflow_registry_handler as wrh
    from umrs.server.commands.workflow_commands import _scoped_list_payload

    catalog = {
        "workflows": [
            {"name": "cnn-track", "version": "1", "user_id": "system", "scope": "system"},
            {"name": "mine", "version": "1", "user_id": "alice", "scope": "user"},
            {"name": "bobs", "version": "1", "user_id": "bob", "scope": "user"},
        ]
    }
    monkeypatch.setattr(wrh, "_load_catalog", lambda: catalog)
    out = json.loads(
        wrh._handle_list_workflows("alice", _scoped_list_payload("{}", "alice", False))
    )
    assert sorted(w["name"] for w in out["workflows"]) == ["cnn-track", "mine"]


def test_purge_needs_the_exact_owned_shape(buckets):
    from umrs.server.step_staging import bucket_purgeable_by

    assert bucket_purgeable_by(ALICE_BUCKET, "alice")
    assert bucket_purgeable_by(f"{ALICE_BUCKET}-output", "alice")
    assert bucket_purgeable_by("alice.old.leftover", "alice")  # unregistered, still hers
    assert not bucket_purgeable_by("first.last.xp.img", "first")  # user first.last's
    assert not bucket_purgeable_by(BOB_BUCKET, "alice")
    buckets.entries["alice.xp.taken"] = SimpleNamespace(s3_username="mallory")
    assert not bucket_purgeable_by("alice.xp.taken", "alice")  # registered to another


def _submit_store():
    store = MagicMock()
    store.create_job.return_value = "job-1"
    from umrs.jobs.models import JobType

    store.get_job.return_value = SimpleNamespace(
        job_id="job-1",
        user_id="alice",
        job_type=JobType.DATA_INGESTION,
        to_dict=lambda: {"job_id": "job-1"},
    )
    store.get_attempts_for_job.return_value = []
    return store


def test_data_ingestion_of_another_users_bucket_is_refused(buckets):
    from umrs.server.job_tracking_handler import DurableJobCommandBackend

    backend = DurableJobCommandBackend(_submit_store())
    payload = json.dumps(
        {
            "project_id": "p",
            "job_type": "data_ingestion",
            "spec": {"bucket": BOB_BUCKET, "s3_key": "k", "project_id": "p"},
        }
    )
    refused = json.loads(backend.submit("alice", payload))
    assert refused == {
        "status": "error",
        "message": "data_ingestion spec.bucket is not a bucket you may use",
    }
    assert json.loads(backend.submit("maint", payload, is_admin=True))["status"] == "ok"


def test_index_build_needs_a_table_of_the_callers_buckets(buckets):
    from umrs.jobs.models import JobType
    from umrs.server.job_tracking_handler import spec_resource_refusal

    buckets.entries[ALICE_BUCKET].db_name = "alice_xp"
    buckets.entries[ALICE_BUCKET].table_name = "images"
    own = {"database": "alice_xp", "table": "images"}
    other = {"database": "bob_xp", "table": "images"}
    assert spec_resource_refusal(JobType.INDEX_BUILD, own, "alice", False) is None
    assert spec_resource_refusal(JobType.INDEX_BUILD, other, "alice", False)
    assert spec_resource_refusal(JobType.INDEX_BUILD, other, "maint", True) is None


def test_the_http_submit_applies_the_same_rule_to_fan_out_children(buckets):
    from fastapi import HTTPException
    from umrs.api.jobs import _enforce_spec_resources
    from umrs.jobs.models import JobType

    child = {
        "job_type": "data_ingestion",
        "spec": {"bucket": BOB_BUCKET, "s3_key": "k", "project_id": "p"},
    }
    with pytest.raises(HTTPException):
        _enforce_spec_resources(JobType.CLEANUP, {}, "alice", False, [child])
    _enforce_spec_resources(JobType.CLEANUP, {}, "maint", True, [child])


def test_status_all_counts_an_admin_token_the_session_presented():
    from umrs.shared.auth import AuthContext, UserRole

    mq = _status_mq({"s-ops": _Entry("ops")})
    mq._auth_sessions = {
        "session:s-ops": AuthContext(user_id="ops", username="ops", role=UserRole.ADMIN)
    }
    mq._dispatch_session_id = "s-ops"
    mq._parse_status_all(["STATUS_ALL"], b"r")
    assert sorted(json.loads(mq.sent[-1])["clients"]) == ["alice", "bob"]


def test_purge_runs_end_to_end_for_own_buckets_and_refuses_others(buckets, tmp_path, monkeypatch):
    from umrs.server import maintenance
    from umrs.shared import json_config
    from umrs.shared.storage import backends

    admin_cfg = tmp_path / "admin.s3cfg"
    admin_cfg.write_text("[default]\n")
    monkeypatch.setattr(json_config, "load_server_cfg_file", lambda _f: (str(admin_cfg), [], {}))
    removed = []
    monkeypatch.setattr(
        backends,
        "get_backend",
        lambda: SimpleNamespace(remove_bucket=lambda b, _c: removed.append(b)),
    )
    monkeypatch.setattr(maintenance, "_remove_purged_experiences_from_user_config", lambda *_a: [])
    ok = json.loads(
        maintenance.handle_purge_buckets(
            "PURGE_BUCKETS", "alice", json.dumps({"buckets": [ALICE_BUCKET]}), configure_file="c"
        )
    )
    assert ok["status"] == "ok" and removed == [ALICE_BUCKET]
    denied = json.loads(
        maintenance.handle_purge_buckets(
            "PURGE_BUCKETS",
            "first",
            json.dumps({"buckets": ["first.last.xp.img"]}),
            configure_file="c",
        )
    )
    assert denied["status"] == "error" and removed == [ALICE_BUCKET]
