"""Security Watch rules pack and SQL detectors (ADR-088 W1, slice S-D61a).

Acceptance (decisions-2026-11-24 §5, S-D61a): the module keeps its import
boundary; the default pack loads with no site file; a malformed site rule
is refused by file and id; an auth-failure burst fires. The detector tests
run on a real Postgres through the read-only ``umrs_audit_reader`` grant
(docs/admin/security/audit-reader-role.md) and skip without ``DATABASE_URL``
(or the CI ``POSTGRES_*`` shape).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any

import pytest
from umrs.audit.security_events import AUDIT_FAMILIES
from umrs.security_watch import (
    RulePackError,
    WriterRoleRefused,
    build_query,
    load_rule_pack,
    run_pack,
    run_rule,
    site_rules_path,
)

PKG = Path(__file__).resolve().parents[1] / "src" / "umrs" / "security_watch"
FORBIDDEN = ("umrs.api", "umrs.server")


# ── import boundary (ADR-088 D-1) ─────────────────────────────────────────


def _imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _forbidden(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in FORBIDDEN)


def test_no_module_imports_api_or_server_even_lazily() -> None:
    sources = sorted(PKG.rglob("*.py"))
    assert sources, "umrs.security_watch is missing"
    bad = {p.name: sorted(n for n in _imported_modules(p) if _forbidden(n)) for p in sources}
    assert not any(bad.values()), bad


def test_importing_the_watch_loads_no_upper_layer() -> None:
    code = (
        "import sys, umrs.security_watch as w; w.load_rule_pack(); "
        "print([m for m in sys.modules if m.startswith(('umrs.api', 'umrs.server'))])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_import_linter_fences_the_watch() -> None:
    config = (PKG.parents[2] / ".importlinter").read_text(encoding="utf-8")
    foundation = config.split("[importlinter:contract:foundation]", 1)[1].split(
        "[importlinter:", 1
    )[0]
    assert "    umrs.security_watch\n" in foundation


# ── the default pack ──────────────────────────────────────────────────────


def test_default_pack_loads_with_no_site_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UMRS_SECURITY_WATCH_RULES", raising=False)
    pack = load_rule_pack(site_rules_path())
    assert pack.rules_version == "2026.11.27-1"
    assert pack.get("auth-failure-burst-actor") is not None
    assert {rule.origin for rule in pack.rules} == {"default"}
    assert len({rule.id for rule in pack.rules}) == len(pack.rules) >= 5


def test_default_pack_watches_only_events_the_stream_carries() -> None:
    audit_events = {
        e for r in load_rule_pack().rules if r.source == "audit_events" for e in r.events
    }
    assert audit_events <= set(AUDIT_FAMILIES)
    assert not any(event.startswith("security.") for event in audit_events)
    assert "step.refused" not in audit_events  # not emitted yet (ADR-088 W0 deferred)


def test_default_pack_is_package_data() -> None:
    pyproject = (PKG.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"umrs.security_watch" = ["rules/*.yaml"]' in pyproject


# ── site overlay ──────────────────────────────────────────────────────────


def _site(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "site-rules.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_site_overlay_changes_a_rule_by_id_and_adds_one(tmp_path: Path) -> None:
    path = _site(
        tmp_path,
        """
        rules_version: "s1"
        rules:
          - {id: auth-failure-burst-actor, threshold: 50}
          - {id: scanner-finding, enabled: false}
          - id: zmq-revoke-burst
            severity: low
            source: audit_events
            detector: burst
            events: [zmq.session_revoked]
            group_by: details.cause
            window_seconds: 600
            threshold: 5
        """,
    )
    pack = load_rule_pack(path)
    assert pack.rules_version == "2026.11.27-1+site.s1"
    auth = pack.get("auth-failure-burst-actor")
    # threshold from the site, the rest of the default stays
    assert (auth.threshold, auth.origin, auth.window_seconds) == (50, "site", 300)  # type: ignore[union-attr]
    assert "scanner-finding" not in {rule.id for rule in pack.enabled()}
    assert getattr(pack.get("zmq-revoke-burst"), "group_by", None) == "details.cause"


def test_site_rules_path_reads_the_env() -> None:
    assert site_rules_path({}) is None
    assert site_rules_path({"UMRS_SECURITY_WATCH_RULES": "/etc/w.yaml"}) == Path("/etc/w.yaml")


_BAD_RULES = {
    "unknown key": ("auth-failure-burst-actor", "sql: 'DROP TABLE x'"),
    "bad severity": ("auth-failure-burst-actor", "severity: urgent"),
    "group_by injection": (
        "auth-failure-burst-actor",
        'group_by: "actor); DELETE FROM audit_events; --"',
    ),
    "match with threshold": ("access-system-grant", "threshold: 3"),
    "threshold too low": ("egress-deny-spike", "threshold: 1"),
    "window too short": ("egress-deny-spike", "window_seconds: 5"),
    "unknown event": (
        "new-rule",
        "severity: low, source: audit_events, detector: match, "
        "events: [step.refused], window_seconds: 600",
    ),
    "watch's own event": (
        "new-rule",
        "severity: low, source: audit_events, detector: match, "
        "events: [security.incident], window_seconds: 600",
    ),
    "where off schema": ("raw-knowledge-export", "where: {prompt: x}"),
    "burst without threshold": (
        "new-rule",
        "severity: low, source: gateway_audit_log, "
        "detector: burst, events: [auth.failed], window_seconds: 600",
    ),
}


@pytest.mark.parametrize("case", sorted(_BAD_RULES))
def test_malformed_site_rule_is_refused_by_file_and_id(tmp_path: Path, case: str) -> None:
    rule_id, fields = _BAD_RULES[case]
    path = _site(tmp_path, f'rules_version: "s1"\nrules:\n  - {{id: {rule_id}, {fields}}}\n')
    with pytest.raises(RulePackError) as err:
        load_rule_pack(path)
    assert err.value.source == str(path)
    assert err.value.rule_id == rule_id
    assert str(path) in str(err.value) and rule_id in str(err.value)


@pytest.mark.parametrize(
    ("body", "rule_id"),
    [
        ('rules_version: "s1"\nrules:\n  - {severity: low}\n', "#0"),
        ('rules_version: "s1"\nrules:\n  - {id: a-b}\n  - {id: a-b}\n', "a-b"),
        ("rules: [\n", "-"),
        ("rules: []\n", "-"),
        ('rules_version: "s1"\nextra: 1\n', "-"),
    ],
)
def test_malformed_site_file_is_refused_by_file(tmp_path: Path, body: str, rule_id: str) -> None:
    path = _site(tmp_path, body)
    with pytest.raises(RulePackError) as err:
        load_rule_pack(path)
    assert (err.value.source, err.value.rule_id) == (str(path), rule_id)


def test_missing_or_oversized_site_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RulePackError, match="cannot read"):
        load_rule_pack(tmp_path / "absent.yaml")
    big = _site(tmp_path, "#" * (256 * 1024 + 1))
    with pytest.raises(RulePackError, match="larger than"):
        load_rule_pack(big)


# ── query building (no rule text in SQL) ─────────────────────────────────


def test_query_binds_every_rule_value(tmp_path: Path) -> None:
    path = _site(
        tmp_path,
        'rules_version: "s1"\nrules:\n  - {id: raw-knowledge-export, group_by: details.job_id}\n',
    )
    rule = load_rule_pack(path).get("raw-knowledge-export")
    assert rule is not None
    sql, params = build_query(rule, now=10_000.0)
    for value in ("job_id", "include_raw", "export.requested", "true"):
        assert value not in sql
    assert params == ["job_id", ["export.requested"], 10_000.0 - 900, "include_raw", "true", 1]


def test_autocommit_connection_is_refused() -> None:
    class _Conn:
        autocommit = True

    rule = load_rule_pack().get("auth-failure-burst-actor")
    assert rule is not None
    with pytest.raises(ValueError, match="transactional"):
        run_rule(_Conn(), rule, now=1.0)


# ── detectors on Postgres, through the read-only role ─────────────────────

_GATEWAY_DDL = """
CREATE TABLE IF NOT EXISTS gateway_audit_log (
    id BIGSERIAL PRIMARY KEY, timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    client_ip TEXT NOT NULL, method TEXT NOT NULL, path TEXT NOT NULL,
    user_id TEXT, action TEXT NOT NULL, resource_id TEXT,
    bytes_transferred BIGINT, response_status INTEGER, duration_ms INTEGER)
"""
_READER_DDL = """
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'umrs_audit_reader') THEN
    CREATE ROLE umrs_audit_reader NOLOGIN;
  END IF;
END $$;
GRANT USAGE ON SCHEMA public TO umrs_audit_reader;
GRANT SELECT ON audit_events TO umrs_audit_reader;
GRANT SELECT ON gateway_audit_log TO umrs_audit_reader;
"""
# Far-future timestamps keep these rows out of every other suite's windows
# and keep other rows out of this suite's windows on a shared CI database.
BASE_TS = 4_100_000_000.0


def _dsn() -> str:
    from umrs.api._asyncpg_pool import _resolve_knowledge_dsn

    dsn, _local = _resolve_knowledge_dsn()
    if not dsn:
        pytest.skip("needs Postgres (CI shape: POSTGRES_* or DATABASE_URL)")
    return dsn


def _prepare(conn: Any) -> None:
    from umrs.api.audit_store import ensure_audit_schema

    ensure_audit_schema(conn)
    with conn.cursor() as cur:
        cur.execute(_GATEWAY_DDL)
        cur.execute(_READER_DDL)
    conn.commit()


def _purge(conn: Any, tag: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM audit_events WHERE actor LIKE %s", (f"{tag}%",))
        cur.execute("DELETE FROM gateway_audit_log WHERE client_ip LIKE %s", (f"{tag}%",))
    conn.commit()


@pytest.fixture
def pg() -> Any:
    psycopg2 = pytest.importorskip("psycopg2")
    try:
        owner = psycopg2.connect(_dsn())
    except psycopg2.Error as exc:
        pytest.skip(f"Postgres unreachable: {exc}")
    try:
        _prepare(owner)
    except psycopg2.Error as exc:
        owner.close()
        pytest.skip(f"cannot create the audit reader grant here: {exc}")
    tag = f"sdw-{uuid.uuid4().hex[:8]}-"
    reader = psycopg2.connect(_dsn())
    with reader.cursor() as cur:
        cur.execute("SET ROLE umrs_audit_reader")
    reader.commit()
    yield owner, reader, tag
    reader.close()
    _purge(owner, tag)
    owner.close()


def _audit_rows(conn: Any, rows: list[tuple[str, str, float, dict[str, Any]]]) -> None:
    with conn.cursor() as cur:
        for event, actor, ts, details in rows:
            cur.execute(
                "INSERT INTO audit_events (event_id, ts, event, actor, outcome, details) "
                "VALUES (%s, %s, %s, %s, 'denied', %s)",
                (uuid.uuid4().hex, ts, event, actor, json.dumps(details)),
            )
    conn.commit()


def _edge_rows(conn: Any, client: str, count: int, ts: float) -> None:
    with conn.cursor() as cur:
        for _ in range(count):
            cur.execute(
                "INSERT INTO gateway_audit_log (timestamp, client_ip, method, path, action) "
                "VALUES (to_timestamp(%s), %s, 'GET', '/umrs/jobs', 'auth.failed')",
                (ts, client),
            )
    conn.commit()


def _mine(hits: Any, tag: str) -> list[Any]:
    return [hit for hit in hits if hit.group.startswith(tag)]


def test_auth_failure_burst_fires(pg: Any) -> None:
    owner, reader, tag = pg
    noisy, quiet = f"{tag}noisy", f"{tag}quiet"
    rows = [("auth.failed", noisy, BASE_TS - i, {"reason": "bad_token"}) for i in range(20)]
    rows += [("auth.failed", quiet, BASE_TS - i, {}) for i in range(19)]
    rows += [("auth.failed", quiet, BASE_TS - 3600, {})]  # outside the 300 s window
    _audit_rows(owner, rows)
    rule = load_rule_pack().get("auth-failure-burst-actor")
    assert rule is not None
    hits = _mine(run_rule(reader, rule, now=BASE_TS + 1), tag)
    assert [(h.group, h.count, h.severity) for h in hits] == [(noisy, 20, "medium")]
    assert len(hits[0].evidence_ids) == 20 and hits[0].last_ts == BASE_TS


def test_edge_burst_fires_on_the_client_fingerprint(pg: Any) -> None:
    owner, reader, tag = pg
    _edge_rows(owner, f"{tag}ipfp", 30, BASE_TS)
    rule = load_rule_pack().get("edge-auth-failure-burst-client")
    assert rule is not None
    hits = _mine(run_rule(reader, rule, now=BASE_TS + 1), tag)
    assert [(h.group, h.count) for h in hits] == [(f"{tag}ipfp", 30)]


def test_where_narrows_a_match_rule(pg: Any) -> None:
    owner, reader, tag = pg
    _audit_rows(
        owner,
        [
            ("export.requested", f"{tag}raw", BASE_TS, {"include_raw": True, "job_id": "j1"}),
            ("export.requested", f"{tag}plain", BASE_TS, {"include_raw": False, "job_id": "j2"}),
        ],
    )
    rule = load_rule_pack().get("raw-knowledge-export")
    assert rule is not None
    assert [h.group for h in _mine(run_rule(reader, rule, now=BASE_TS + 1), tag)] == [f"{tag}raw"]


def test_pack_refuses_a_writer_role_and_skips_by_id(pg: Any) -> None:
    owner, reader, _tag = pg
    pack = load_rule_pack()
    with pytest.raises(WriterRoleRefused):
        run_pack(pack, {"audit_events": owner}, now=BASE_TS)
    result = run_pack(pack, {"audit_events": reader}, now=BASE_TS)
    assert result.skipped == ("edge-auth-failure-burst-client",)
    assert result.rules_version == pack.rules_version


class _RecordingConn:
    autocommit = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    def cursor(self) -> Any:
        conn = self

        class _Cur:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def execute(self, sql: str, params: Any = None) -> None:
                conn.calls.append(sql.split(" ", 2)[0] if params else sql)

            def fetchall(self) -> list[Any]:
                return []

        return _Cur()

    def rollback(self) -> None:
        self.calls.append("ROLLBACK")


def test_detection_runs_read_only_and_rolls_back() -> None:
    conn = _RecordingConn()
    rule = load_rule_pack().get("auth-failure-burst-actor")
    assert rule is not None
    assert run_rule(conn, rule, now=BASE_TS) == []
    assert conn.calls == ["SET TRANSACTION READ ONLY", "SELECT", "ROLLBACK"]
