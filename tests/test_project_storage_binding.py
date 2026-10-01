"""ADR-077 pins: per-project external storage binding (gap row D-17).

Contracts locked:

- locators: S3 = ``http(s)://AK:SK@host:port[/bucket][?region=&verify=&backend=]``,
  Postgres = ``postgresql://user:pw@host:port/db[?sslmode=]``; libpq keys that
  name server files or re-target the connection are refused.
- target policy: loopback / link-local / unspecified / multicast refused
  (IPv4-mapped too); private addresses allowed; the operator allowlist
  narrows targets and is what lets an owner (not only an admin) bind.
- secrets are sealed at rest (AAD = project + field) and never returned.
- the resolver serves unbound projects without a per-project read, keeps
  the last good answer when a refresh fails, and fails closed only for a
  project it never resolved.
- the files API, the workflow runner and workflow artifact persistence use
  the bound store; an unreadable binding refuses instead of falling back.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.shared.storage import project_storage as ps
from umrs.shared.storage.project_storage import (
    ProjectS3Config,
    ProjectStorageResolver,
    ProjectStorageStore,
    ProjectStorageUnavailable,
    StorageBinding,
    StorageBindingError,
    binding_refusal,
    check_target_host,
    parse_postgres_uri,
    parse_s3_url,
    seal,
    unseal,
)

S3_URL = "http://AKIDEXAMPLE:s3cr3t-key@10.129.0.31:28900"
PG_URI = "postgresql://opossum:pg-pass@10.129.0.31:25432/opossum?sslmode=require"


@pytest.fixture(autouse=True)
def _sealing_key(monkeypatch):
    monkeypatch.delenv(ps.KEY_ENV, raising=False)
    monkeypatch.delenv(ps.ALLOWED_ENV, raising=False)
    monkeypatch.setenv("JWT_SECRET", "unit-test-jwt-secret")
    ps.set_project_storage_resolver(None)
    yield
    ps.set_project_storage_resolver(None)


# -- fakes ---------------------------------------------------------------------


class _Cursor:
    def __init__(self, conn):
        self._conn = conn
        self._rows: list = []
        self.rowcount = 0

    def execute(self, sql, params=None):
        self._rows, self.rowcount = self._conn.handle(" ".join(sql.split()), params)

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows


class _Conn:
    """psycopg-shaped fake over the projects + project_storage_bindings tables."""

    def __init__(self, projects=None):
        self.projects = dict(projects or {})
        self.bindings: dict[str, tuple] = {}
        self.reads = 0
        self.fail = None
        self.rollbacks = 0

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        pass

    def rollback(self):
        self.rollbacks += 1

    def handle(self, sql, params):
        if self.fail is not None:
            raise self.fail
        for matches, run in self._routes():
            if matches(sql):
                return run(params)
        raise AssertionError(f"unexpected SQL: {sql}")

    def _routes(self):
        return (
            (lambda q: q.startswith("SELECT project_id FROM project_storage_bindings"), self._ids),
            (lambda q: q.startswith("DELETE FROM project_storage_bindings"), self._delete),
            (lambda q: q.startswith("INSERT INTO project_storage_bindings"), self._upsert),
            (lambda q: "FROM project_storage_bindings WHERE project_id" in q, self._select),
            (lambda q: "FROM projects WHERE project_id" in q, self._project),
        )

    def _ids(self, _params):
        return [(pid,) for pid in self.bindings], 0

    def _delete(self, params):
        return [], int(self.bindings.pop(params[0], None) is not None)

    def _upsert(self, params):
        pid, s3, pg, who = params
        self.bindings[pid] = (pid, s3, pg, who, "2026-10-01T00:00:00+00:00")
        return [], 1

    def _select(self, params):
        self.reads += 1
        row = self.bindings.get(params[0])
        return ([row] if row else []), 0

    def _project(self, params):
        row = self.projects.get(params[0])
        return ([row] if row else []), 0


def _project_row(pid: str, owner: str) -> tuple:
    return (pid, "", owner, "active", None, None, None, "sensitive", "", "[]")


class _Pool:
    def __init__(self, conn):
        self.conn = conn
        self.borrowed = 0

    def getconn(self):
        self.borrowed += 1
        return self.conn

    def putconn(self, _conn):
        self.borrowed -= 1


# -- locators ------------------------------------------------------------------


class TestS3Url:
    def test_parses_endpoint_credentials_and_defaults(self) -> None:
        loc = parse_s3_url(S3_URL)
        assert loc.endpoint_url == "http://10.129.0.31:28900"
        assert (loc.access_key, loc.secret_key) == ("AKIDEXAMPLE", "s3cr3t-key")
        assert (loc.region, loc.bucket, loc.backend, loc.verify) == (
            "us-east-1",
            "",
            "seaweedfs",
            True,
        )
        assert not loc.use_ssl

    def test_bucket_query_and_encoded_credentials(self) -> None:
        loc = parse_s3_url(
            "https://AK%2F1:sk%40x@store.example.org/proj-data?region=ap-1&verify=false&backend=minio"
        )
        assert loc.endpoint_url == "https://store.example.org:443"
        assert (loc.access_key, loc.secret_key) == ("AK/1", "sk@x")
        assert (loc.bucket, loc.region, loc.verify, loc.backend) == (
            "proj-data",
            "ap-1",
            False,
            "minio",
        )
        assert loc.use_ssl

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("", "locator_empty"),
            ("http://10.0.0.1:9000", "credentials_required"),
            ("ftp://a:b@10.0.0.1", "scheme_unsupported"),
            ("http://a:b@10.0.0.1/Bad_Bucket", "bucket_invalid"),
            ("http://a:b@10.0.0.1/a/b", "bucket_invalid"),
            ("http://a:b@10.0.0.1?token=x", "query_key_unsupported"),
            ("http://a:b@10.0.0.1?backend=ceph", "backend_unsupported"),
            ("http://a:b@10.0.0.1:99999", "port_invalid"),
            ("http://a:b@h1,h2:9000", "host_invalid"),
        ],
    )
    def test_refusals(self, text: str, reason: str) -> None:
        with pytest.raises(StorageBindingError) as exc:
            parse_s3_url(text)
        assert exc.value.reason == reason

    def test_redacted_never_carries_the_secret(self) -> None:
        shown = json.dumps(parse_s3_url(S3_URL).redacted())
        assert "s3cr3t-key" not in shown
        assert "AKIDEXAMPLE" not in shown
        assert "AKID" in shown


class TestPostgresUri:
    def test_parses(self) -> None:
        loc = parse_postgres_uri(PG_URI)
        assert (loc.host, loc.port, loc.dbname, loc.user, loc.sslmode) == (
            "10.129.0.31",
            25432,
            "opossum",
            "opossum",
            "require",
        )
        assert "pg-pass" not in json.dumps(loc.redacted())

    def test_postgres_scheme_alias_and_default_port(self) -> None:
        loc = parse_postgres_uri("postgres://u:p@db.example.org/data")
        assert (loc.port, loc.dbname) == (5432, "data")

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("postgresql://u:p@10.0.0.1:5432/", "dbname_required"),
            ("postgresql://u@10.0.0.1/db", "credentials_required"),
            ("postgresql://u:p@10.0.0.1/db?sslrootcert=/etc/passwd", "query_key_unsupported"),
            ("postgresql://u:p@10.0.0.1/db?host=/var/run", "query_key_unsupported"),
            ("postgresql://u:p@10.0.0.1/db?sslmode=always", "sslmode_invalid"),
            ("postgresql://u:p@h1:5432,h2:5432/db", "host_invalid"),
            ("mysql://u:p@10.0.0.1/db", "scheme_unsupported"),
        ],
    )
    def test_refusals(self, text: str, reason: str) -> None:
        with pytest.raises(StorageBindingError) as exc:
            parse_postgres_uri(text)
        assert exc.value.reason == reason


# -- target policy ---------------------------------------------------------------


class TestTargetPolicy:
    @pytest.mark.parametrize(
        "host", ["127.0.0.1", "localhost", "169.254.1.2", "0.0.0.0", "::1", "::ffff:127.0.0.1"]
    )
    def test_infrastructure_addresses_refused(self, host: str) -> None:
        with pytest.raises(StorageBindingError) as exc:
            check_target_host(host, allowed=())
        assert exc.value.reason == "address_forbidden"

    def test_private_address_allowed_without_allowlist(self) -> None:
        check_target_host("10.129.0.31", allowed=())

    def test_allowlist_cidr_and_name(self) -> None:
        check_target_host("10.129.0.31", allowed=("10.129.0.0/18",))
        with pytest.raises(StorageBindingError) as exc:
            check_target_host("10.134.0.8", allowed=("10.129.0.0/18",))
        assert exc.value.reason == "host_not_allowlisted"

    def test_unresolvable_name(self) -> None:
        with pytest.raises(StorageBindingError) as exc:
            check_target_host("no-such-host.invalid", allowed=())
        assert exc.value.reason == "host_unresolvable"

    @pytest.mark.parametrize(
        ("is_admin", "is_owner", "allowed", "expected"),
        [
            (True, False, (), ""),
            (False, True, (), "admin_required"),
            (False, True, ("10.0.0.0/8",), ""),
            (False, False, ("10.0.0.0/8",), "forbidden"),
        ],
    )
    def test_who_may_bind(self, is_admin, is_owner, allowed, expected) -> None:
        assert binding_refusal(is_admin=is_admin, is_owner=is_owner, allowed=allowed) == expected


# -- sealing ------------------------------------------------------------------------


class TestSealing:
    def test_roundtrip_and_no_plaintext(self) -> None:
        token = seal(S3_URL, project_id="opossum", field="s3")
        assert "s3cr3t" not in token and token.startswith("v1:")
        assert unseal(token, project_id="opossum", field="s3") == S3_URL

    def test_ciphertext_bound_to_project_and_field(self) -> None:
        token = seal(S3_URL, project_id="opossum", field="s3")
        for pid, field in (("other", "s3"), ("opossum", "postgres")):
            with pytest.raises(StorageBindingError) as exc:
                unseal(token, project_id=pid, field=field)
            assert exc.value.reason == "sealed_value_invalid"

    def test_key_change_is_named(self, monkeypatch) -> None:
        token = seal(S3_URL, project_id="opossum", field="s3")
        monkeypatch.setenv("JWT_SECRET", "rotated")
        with pytest.raises(StorageBindingError) as exc:
            unseal(token, project_id="opossum", field="s3")
        assert exc.value.reason == "sealing_key_changed"

    def test_explicit_key_wins_and_missing_key_refuses(self, monkeypatch) -> None:
        jwt_token = seal("x", project_id="p", field="s3")
        monkeypatch.setenv(ps.KEY_ENV, "dedicated-key")
        assert seal("x", project_id="p", field="s3").split(":")[1] != jwt_token.split(":")[1]
        monkeypatch.delenv(ps.KEY_ENV)
        monkeypatch.delenv("JWT_SECRET")
        with pytest.raises(StorageBindingError) as exc:
            seal("x", project_id="p", field="s3")
        assert exc.value.reason == "sealing_key_unavailable"


# -- store ---------------------------------------------------------------------------


class TestStore:
    def test_none_keeps_the_other_store(self) -> None:
        store = ProjectStorageStore(_Conn())
        first = store.put("opossum", s3_url=S3_URL, postgres_uri=None, updated_by="mnagaso")
        assert first.postgres is None
        both = store.put("opossum", s3_url=None, postgres_uri=PG_URI, updated_by="mnagaso")
        assert both.s3.endpoint_url == "http://10.129.0.31:28900"  # kept
        assert both.postgres.dbname == "opossum"

    def test_secrets_are_sealed_at_rest(self) -> None:
        conn = _Conn()
        ProjectStorageStore(conn).put(
            "opossum", s3_url=S3_URL, postgres_uri=PG_URI, updated_by="mnagaso"
        )
        raw = json.dumps(conn.bindings["opossum"])
        assert "s3cr3t-key" not in raw
        assert "pg-pass" not in raw

    def test_blank_unbinds_and_empty_binding_is_deleted(self) -> None:
        conn = _Conn()
        store = ProjectStorageStore(conn)
        store.put("opossum", s3_url=S3_URL, postgres_uri=PG_URI, updated_by="mnagaso")
        pg_only = store.put("opossum", s3_url="", postgres_uri=None, updated_by="mnagaso")
        assert pg_only.s3 is None
        assert pg_only.postgres is not None
        assert store.put("opossum", s3_url=None, postgres_uri="", updated_by="x") is None
        assert "opossum" not in conn.bindings

    def test_clear_reports_existence(self) -> None:
        store = ProjectStorageStore(_Conn())
        assert store.clear("opossum") is False
        store.put("opossum", s3_url=S3_URL, postgres_uri=None, updated_by="u")
        assert store.clear("opossum") is True


# -- resolver --------------------------------------------------------------------------


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _bound_conn() -> _Conn:
    conn = _Conn()
    ProjectStorageStore(conn).put("opossum", s3_url=S3_URL, postgres_uri=None, updated_by="u")
    conn.reads = 0  # count only the resolver's own reads
    return conn


class TestResolver:
    def test_unbound_project_needs_no_per_project_read(self) -> None:
        conn = _bound_conn()
        pool = _Pool(conn)
        resolver = ProjectStorageResolver(pool_getter=lambda: pool, ttl_sec=30)
        assert resolver.binding("default") is None
        assert conn.reads == 0
        assert pool.borrowed == 0 and conn.rollbacks >= 1  # returned, not idle-in-tx

    def test_bound_project_cached_for_ttl(self) -> None:
        conn = _bound_conn()
        clock = _Clock()
        resolver = ProjectStorageResolver(pool_getter=lambda: _Pool(conn), ttl_sec=30, clock=clock)
        assert resolver.s3_locator("opossum").endpoint_url == "http://10.129.0.31:28900"
        resolver.binding("opossum")
        assert conn.reads == 1
        clock.now += 31
        resolver.binding("opossum")
        assert conn.reads == 2

    def test_failed_refresh_serves_last_answer(self) -> None:
        conn = _bound_conn()
        clock = _Clock()
        resolver = ProjectStorageResolver(pool_getter=lambda: _Pool(conn), ttl_sec=30, clock=clock)
        assert resolver.binding("opossum") is not None
        clock.now += 31
        conn.fail = RuntimeError("pool exhausted")
        assert resolver.binding("opossum") is not None
        assert resolver.binding("default") is None

    def test_never_resolved_fails_closed(self) -> None:
        conn = _Conn()
        conn.fail = RuntimeError("pool exhausted")
        resolver = ProjectStorageResolver(pool_getter=lambda: _Pool(conn), ttl_sec=30)
        with pytest.raises(ProjectStorageUnavailable):
            resolver.binding("opossum")

    def test_missing_table_means_no_bindings(self) -> None:
        conn = _Conn()
        err = RuntimeError("relation does not exist")
        err.pgcode = "42P01"
        conn.fail = err
        resolver = ProjectStorageResolver(pool_getter=lambda: _Pool(conn), ttl_sec=30)
        assert resolver.binding("opossum") is None

    def test_no_pool_means_no_bindings(self) -> None:
        assert ProjectStorageResolver(pool_getter=lambda: None).binding("opossum") is None

    def test_invalidate_drops_the_id_set(self) -> None:
        conn = _Conn()
        resolver = ProjectStorageResolver(pool_getter=lambda: _Pool(conn), ttl_sec=300)
        assert resolver.binding("opossum") is None
        ProjectStorageStore(conn).put("opossum", s3_url=S3_URL, postgres_uri=None, updated_by="u")
        assert resolver.binding("opossum") is None  # cached id set
        resolver.invalidate("opossum")
        assert resolver.binding("opossum") is not None


class TestProjectS3Config:
    def test_bound_store_and_presign_pin(self, monkeypatch) -> None:
        monkeypatch.setenv("S3_BUCKET", "r4-data")
        cfg = ProjectS3Config(parse_s3_url(S3_URL))
        assert (cfg.endpoint_url, cfg.bucket, cfg.presign_endpoint) == (
            "http://10.129.0.31:28900",
            "r4-data",
            "http://10.129.0.31:28900",
        )
        assert cfg.get_client().meta.endpoint_url == "http://10.129.0.31:28900"
        named = ProjectS3Config(parse_s3_url(S3_URL + "/opossum-data"))
        assert named.bucket == "opossum-data"


# -- routes ----------------------------------------------------------------------------


def _client(conn: _Conn) -> TestClient:
    from umrs.api.projects import router

    app = FastAPI()
    app.include_router(router)
    pool = MagicMock()
    pool.getconn.return_value = conn
    app.state.umrs_db_pool = pool
    return TestClient(app)


def _headers(user: str, admin: bool = False) -> dict[str, str]:
    headers = {"X-User-ID": user, "X-Verified-Project-ID": "default"}
    if admin:
        headers["X-Verified-Is-Admin"] = "true"
    return headers


def _assert_no_secrets(text: str) -> None:
    assert "s3cr3t-key" not in text
    assert "pg-pass" not in text


def _route_conn() -> _Conn:
    return _Conn(projects={"opossum": _project_row("opossum", "mnagaso")})


class TestRoutes:
    def test_get_unbound_then_owner_only(self) -> None:
        client = _client(_route_conn())
        r = client.get("/umrs/projects/opossum/storage", headers=_headers("mnagaso"))
        assert r.status_code == 200
        assert r.json()["storage"]["bound"] is False
        other = client.get("/umrs/projects/opossum/storage", headers=_headers("eve"))
        assert other.status_code == 403
        ghost = client.get("/umrs/projects/ghost/storage", headers=_headers("mnagaso"))
        assert ghost.status_code == 404

    def test_admin_binds_without_echoing_secrets(self) -> None:
        conn = _route_conn()
        client = _client(conn)
        r = client.put(
            "/umrs/projects/opossum/storage",
            json={"s3_url": S3_URL, "postgres_uri": PG_URI, "check": False},
            headers=_headers("mnagaso", admin=True),
        )
        assert r.status_code == 200, r.text
        _assert_no_secrets(r.text)
        storage = r.json()["storage"]
        assert storage["bound"] is True
        assert storage["s3"]["endpoint_url"] == "http://10.129.0.31:28900"
        assert storage["postgres"]["dbname"] == "opossum"
        assert storage["updated_by"] == "mnagaso"
        shown = client.get("/umrs/projects/opossum/storage", headers=_headers("mnagaso"))
        _assert_no_secrets(shown.text)
        assert shown.json()["storage"]["bound"] is True

    def test_owner_needs_operator_allowlist(self, monkeypatch) -> None:
        client = _client(_route_conn())
        body = {"s3_url": S3_URL, "check": False}
        r = client.put("/umrs/projects/opossum/storage", json=body, headers=_headers("mnagaso"))
        assert r.status_code == 403
        assert "UMRS_PROJECT_STORAGE_ALLOWED" in r.text
        monkeypatch.setenv(ps.ALLOWED_ENV, "10.129.0.0/18")
        ok = client.put("/umrs/projects/opossum/storage", json=body, headers=_headers("mnagaso"))
        assert ok.status_code == 200, ok.text
        stranger = client.put("/umrs/projects/opossum/storage", json=body, headers=_headers("eve"))
        assert stranger.status_code == 403

    def test_loopback_target_refused(self) -> None:
        client = _client(_route_conn())
        r = client.put(
            "/umrs/projects/opossum/storage",
            json={"postgres_uri": "postgresql://u:p@127.0.0.1:5432/umrs_db", "check": False},
            headers=_headers("mnagaso", admin=True),
        )
        assert r.status_code == 400
        assert "address_forbidden" in r.text

    def test_failing_probe_saves_nothing(self, monkeypatch) -> None:
        conn = _route_conn()
        client = _client(conn)
        monkeypatch.setattr(
            ps,
            "probe_binding",
            lambda *_args, **_kwargs: {
                "ok": False,
                "s3": {"ok": False, "error": "EndpointConnectionError", "detail": "refused"},
                "postgres": None,
            },
        )
        r = client.put(
            "/umrs/projects/opossum/storage",
            json={"s3_url": S3_URL},
            headers=_headers("mnagaso", admin=True),
        )
        assert r.status_code == 409
        assert "nothing saved" in r.text
        assert conn.bindings == {}

    def test_probe_then_save_and_check_and_clear(self, monkeypatch) -> None:
        conn = _route_conn()
        client = _client(conn)
        seen = []

        def fake_probe(binding, default_bucket):
            seen.append((binding.s3.endpoint_url, default_bucket))
            return {"ok": True, "s3": {"ok": True}, "postgres": None, "project_id": "opossum"}

        monkeypatch.setattr(ps, "probe_binding", fake_probe)
        monkeypatch.setenv("S3_BUCKET", "r4-data")
        from umrs.shared.storage import s3_config

        monkeypatch.setattr(s3_config, "_s3_config", None)
        put = client.put(
            "/umrs/projects/opossum/storage",
            json={"s3_url": S3_URL},
            headers=_headers("mnagaso", admin=True),
        )
        assert put.status_code == 200, put.text
        assert put.json()["probe"]["ok"] is True
        check = client.post("/umrs/projects/opossum/storage/check", headers=_headers("mnagaso"))
        assert check.status_code == 200 and check.json()["probe"]["ok"] is True
        assert seen == [("http://10.129.0.31:28900", "r4-data")] * 2
        gone = client.delete("/umrs/projects/opossum/storage", headers=_headers("mnagaso"))
        assert gone.json()["cleared"] is True and conn.bindings == {}
        missing = client.post("/umrs/projects/opossum/storage/check", headers=_headers("mnagaso"))
        assert missing.status_code == 404


# -- routed consumers ---------------------------------------------------------------------


class _FakeResolver(ProjectStorageResolver):
    def __init__(self, bindings=None, fail=False):
        super().__init__(pool_getter=lambda: None)
        self._bindings = bindings or {}
        self._fail = fail

    def binding(self, project_id, *, pool=None):  # noqa: ARG002 — resolver signature
        if self._fail:
            raise ProjectStorageUnavailable("registry unreadable")
        return self._bindings.get(project_id)


def _bound_resolver() -> _FakeResolver:
    return _FakeResolver({"opossum": StorageBinding("opossum", s3=parse_s3_url(S3_URL))})


class TestFilesDependency:
    def test_bound_project_gets_its_store(self) -> None:
        from umrs.api.files import get_project_s3_config

        ps.set_project_storage_resolver(_bound_resolver())
        default = object()
        bound = get_project_s3_config(project_id="opossum", default=default)
        assert isinstance(bound, ProjectS3Config)
        assert bound.endpoint_url == "http://10.129.0.31:28900"
        assert get_project_s3_config(project_id="default", default=default) is default

    def test_unreadable_binding_refuses(self) -> None:
        from fastapi import HTTPException
        from umrs.api.files import get_project_s3_config

        ps.set_project_storage_resolver(_FakeResolver(fail=True))
        with pytest.raises(HTTPException) as exc:
            get_project_s3_config(project_id="opossum", default=object())
        assert exc.value.status_code == 503


class TestWorkflowRunner:
    def _record(self, bucket: str):
        record = MagicMock()
        record.params = {"bucket": bucket}
        return record

    def test_bound_project_run_uses_bound_store(self) -> None:
        import threading

        from umrs.server.workflow_runner import _resolve_s3_client

        ps.set_project_storage_resolver(_bound_resolver())
        client, bucket = _resolve_s3_client(
            self._record("r4-data"), threading.Lock(), project_id="opossum"
        )
        assert client.meta.endpoint_url == "http://10.129.0.31:28900"
        assert bucket == "r4-data"

    def test_in_memory_runs_ignore_the_binding(self) -> None:
        import threading

        from umrs.server.workflow_runner import _InMemoryS3Client, _resolve_s3_client

        ps.set_project_storage_resolver(_bound_resolver())
        client, bucket = _resolve_s3_client(
            self._record("wf-runner"), threading.Lock(), project_id="opossum"
        )
        assert isinstance(client, _InMemoryS3Client) and bucket == "wf-runner"

    def test_unreadable_binding_fails_the_run(self, monkeypatch) -> None:
        import threading

        from umrs.server import workflow_runner

        ps.set_project_storage_resolver(_FakeResolver(fail=True))
        monkeypatch.setattr(workflow_runner, "_emit_workflow_metric", lambda *_a, **_k: None)
        monkeypatch.setattr(workflow_runner, "_build_levels", lambda _spec: ([], []))
        record = self._record("r4-data")
        out = workflow_runner._prepare_run(record, [], threading.Lock(), "opossum", None)
        assert out is None
        kwargs = record.update.call_args.kwargs
        assert kwargs["status"] == "failed"
        assert "project storage binding unreadable" in kwargs["error"]


class TestWorkflowArtifacts:
    def _job(self, project_id: str):
        return SimpleNamespace(project_id=project_id, spec={}, job_id="j-1")

    def test_bound_project_persists_to_its_store(self, monkeypatch) -> None:
        from umrs.jobs.workflow_run import _resolve_s3

        monkeypatch.setenv("S3_BUCKET", "r4-data")
        ps.set_project_storage_resolver(_bound_resolver())
        client, bucket = _resolve_s3(self._job("opossum"), None)
        assert client.meta.endpoint_url == "http://10.129.0.31:28900"
        assert bucket == "r4-data"

    def test_unreadable_binding_skips_persistence(self) -> None:
        from umrs.jobs.workflow_run import _resolve_s3

        ps.set_project_storage_resolver(_FakeResolver(fail=True))
        client, _bucket = _resolve_s3(self._job("opossum"), None)
        assert client is None
