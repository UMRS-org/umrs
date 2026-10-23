"""ADR-099 D-4/D-5 (RS-1 lane Q): usage rows and quotas at the inference proxy.

End to end through ``create_app`` with the proxy tests' fakes (fixed lease,
MockTransport upstream) plus a recording pool:

- every relayed call writes one ``llm_usage_log`` row, ``origin='infer_proxy'``
  (before: none; the proxy recorded no token counts);
- a reached daily token cap is a 429 with the exact contract body and
  headers, audited, the provider never called (before: relayed);
- the per-user rpm window refuses the call over the limit and frees up once
  the oldest call leaves the minute;
- no limit set means no usage query at all;
- unreadable limits or usage, while a limit may apply, fail closed (503).
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone

import httpx
import pytest
from test_infer_proxy import (
    _FakeBroker,
    _FakeStore,
    _lease,
    _make_fetch,
    _request,
    _stream_response,
)
from test_infer_proxy_site import _registry
from umrs.infer_proxy.app import create_app
from umrs.quotas.enforcer import QuotaEnforcer
from umrs.security.seal import generate_recipient_keypair
from umrs.shared.site_source import SITE_SOURCE_ID

NOW = datetime(2026, 10, 23, 9, 0, tzinfo=timezone.utc).timestamp()
CHAT = "/s/sess-1/v1/chat/completions"
USAGE_BODY = json.dumps(
    {"model": "gpt-x", "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4}}
).encode()


class _Cursor:
    def __init__(self, pool):
        self.pool = pool
        self._row = None

    def execute(self, sql, params=None):
        self.pool.sql.append((sql, params))
        if "SUM(total_tokens)" in sql:
            if self.pool.sum_error:
                raise RuntimeError("ledger down")
            column = "project" if "project_id = %s" in sql else "user"
            self._row = (self.pool.used.get(column, 0),)

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


class _Conn:
    def __init__(self, pool):
        self.pool = pool

    def cursor(self):
        return _Cursor(self.pool)

    def commit(self):
        pass

    def rollback(self):
        pass


class _Pool:
    """Records every statement; answers the quota SUM with ``used``."""

    def __init__(self, used=None, sum_error=False):
        self.sql: list = []
        self.used = used or {}
        self.sum_error = sum_error

    def getconn(self):
        return _Conn(self)

    def putconn(self, _conn):
        pass

    def inserts(self):
        return [p for s, p in self.sql if "INSERT INTO llm_usage_log" in s]

    def sums(self):
        return [s for s, _p in self.sql if "SUM(total_tokens)" in s]


class _Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def _limits(**budget):
    full = {"tokens_per_day_user": None, "tokens_per_day_project": None, "rpm_user": None}
    full.update(budget)
    return lambda _venue, _user, _project: {"budget": full, "sessions": {}}


def _app(
    monkeypatch,
    *,
    limits_fn=None,
    pool=None,
    lease=None,
    clock=None,
    body=USAGE_BODY,
    content_type="application/json",
):
    lease = lease or _lease(owner_id="alice", project="opossum", venue="rccs")
    priv, pub = generate_recipient_keypair()
    fetch = _make_fetch((pub, priv))
    monkeypatch.setattr("umrs.infer_proxy.app.lease_by_token", lambda _pool, _token: lease)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, _port, _family, _socktype: [(2, 1, 6, "", ("104.18.2.161", 0))],
    )
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.headers))
        return _stream_response(body=body, headers={"content-type": content_type})

    pool = pool if pool is not None else _Pool()
    broker = _FakeBroker()
    enforcer = None
    if limits_fn is not None:
        enforcer = QuotaEnforcer(pool, limits_fn, clock=clock or _Clock())
    app = create_app(
        umrs_db_pool=pool,
        unseal_privkey=priv,
        sealed_store=_FakeStore(fetch),
        broker=broker,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        quota_enforcer=enforcer,
    )
    return app, pool, broker, calls


def _post(app, url=CHAT, model="gpt-x"):
    return _request(
        app,
        "POST",
        url,
        json={"model": model, "messages": []},
        headers={"authorization": "Bearer tok-session", "accept-encoding": "gzip"},
    )


class TestUsageRows:
    def test_one_usage_row_per_proxied_call(self, monkeypatch):
        app, pool, _broker, calls = _app(monkeypatch)
        assert _post(app).status_code == 200
        assert _post(app).status_code == 200
        rows = pool.inserts()
        assert len(rows) == 2
        row = rows[0]
        assert row[1:3] == ("alice", "opossum")
        assert row[3] == "gpt-x"  # the answer's model
        assert (row[7], row[8], row[9]) == (11, 4, 15)
        assert row[-3:] == ("rccs", "sess-1", "infer_proxy")
        # The tap reads plain bytes: the upstream is asked for identity.
        assert calls[0]["accept-encoding"] == "identity"

    def test_a_streamed_answer_is_counted_from_its_last_chunk(self, monkeypatch):
        sse = (
            b'data: {"model": "gpt-x", "choices": [{"delta": {"content": "hi"}}]}\n\n'
            b'data: {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 9}}\n\n'
            b"data: [DONE]\n\n"
        )
        app, pool, _broker, _calls = _app(monkeypatch, body=sse, content_type="text/event-stream")
        assert _post(app).status_code == 200
        row = pool.inserts()[0]
        assert (row[9], row[16]) == (29, True)  # total tokens, is_streaming

    def test_no_enforcer_means_no_quota_query(self, monkeypatch):
        app, pool, _broker, _calls = _app(monkeypatch)
        assert _post(app).status_code == 200
        assert pool.sums() == []


class TestDailyTokens:
    def test_cap_reached_is_the_exact_typed_429(self, monkeypatch):
        pool = _Pool(used={"user": 2000310})
        app, pool, broker, calls = _app(
            monkeypatch, limits_fn=_limits(tokens_per_day_user=2000000), pool=pool
        )
        res = _post(app)
        assert res.status_code == 429
        assert res.headers["x-platform-error-code"] == "quota.tokens_per_day_exceeded"
        assert res.headers["retry-after"] == str(15 * 3600)
        assert res.headers["content-type"] == "application/json"
        assert res.json() == {
            "error": "quota.tokens_per_day_exceeded",
            "message": "daily token limit reached for user alice at rccs: "
            "2000310 of 2000000 tokens; resets 2026-10-24T00:00:00Z",
            "scope": "user",
            "subject": "alice",
            "limit": 2000000,
            "used": 2000310,
            "window": "day",
            "resets_at": "2026-10-24T00:00:00Z",
        }
        assert calls == []  # the provider was never asked
        assert ("deny", "quota.tokens_per_day_exceeded") in broker.ops
        assert pool.inserts() == []

    def test_below_the_cap_relays_and_counts_today_only(self, monkeypatch):
        pool = _Pool(used={"user": 10})
        app, pool, _broker, calls = _app(
            monkeypatch, limits_fn=_limits(tokens_per_day_user=2000000), pool=pool
        )
        assert _post(app).status_code == 200 and len(calls) == 1
        sql, params = next((s, p) for s, p in pool.sql if "SUM(total_tokens)" in s)
        assert "origin = 'infer_proxy'" in sql and "user_id = %s" in sql
        assert params[:2] == ("rccs", "alice")
        assert params[2] == datetime(2026, 10, 23, tzinfo=timezone.utc)

    def test_project_cap_names_the_project(self, monkeypatch):
        pool = _Pool(used={"user": 0, "project": 500})
        app, _pool, _broker, _calls = _app(
            monkeypatch, limits_fn=_limits(tokens_per_day_project=500), pool=pool
        )
        body = _post(app).json()
        assert (body["scope"], body["subject"], body["limit"]) == ("project", "opossum", 500)

    def test_day_follows_the_configured_zone(self):
        # 09:00 UTC is 18:00 in Tokyo: today ends at 15:00 UTC.
        pool = _Pool(used={"user": 5})
        lease = _lease(owner_id="alice", project="opossum", venue="rccs")
        enforcer = QuotaEnforcer(
            pool, _limits(tokens_per_day_user=5), day_tz="Asia/Tokyo", clock=_Clock()
        )
        denial = enforcer.check(lease)
        assert denial.resets_at == "2026-10-23T15:00:00Z"
        assert denial.retry_after_s == 6 * 3600
        assert pool.sql[0][1][2] == datetime(2026, 10, 22, 15, tzinfo=timezone.utc)

    def test_models_listing_is_not_metered(self, monkeypatch):
        pool = _Pool(used={"user": 9})
        app, pool, _broker, calls = _app(
            monkeypatch, limits_fn=_limits(tokens_per_day_user=1), pool=pool
        )
        res = _request(
            app, "GET", "/s/sess-1/v1/models", headers={"authorization": "Bearer tok-session"}
        )
        assert res.status_code == 200 and len(calls) == 1
        assert pool.sums() == []


class TestRequestsPerMinute:
    @pytest.fixture
    def burst(self, monkeypatch):
        """Two calls 10 s apart under ``rpm_user=2``, then a third 5 s later."""
        clock = _Clock()
        app, pool, broker, calls = _app(monkeypatch, limits_fn=_limits(rpm_user=2), clock=clock)
        statuses = [_post(app).status_code]
        clock.now += 10
        statuses.append(_post(app).status_code)
        clock.now += 5
        return app, pool, broker, calls, clock, statuses, _post(app)

    def test_the_call_over_the_window_is_refused(self, burst):
        _app_, _pool, broker, calls, _clock, statuses, res = burst
        assert statuses == [200, 200] and len(calls) == 2
        assert res.status_code == 429
        assert res.headers["x-platform-error-code"] == "quota.rpm_exceeded"
        assert res.headers["retry-after"] == "45"  # the first call leaves at +60
        body = res.json()
        assert (body["window"], body["limit"], body["used"]) == ("minute", 2, 2)
        assert ("deny", "quota.rpm_exceeded") in broker.ops

    def test_the_window_frees_and_never_reads_the_ledger(self, burst):
        app, pool, _broker, _calls, clock, _statuses, _res = burst
        clock.now += 46
        assert _post(app).status_code == 200
        assert pool.sums() == []  # rpm alone never reads the ledger


class TestCapUnset:
    def test_no_limit_set_makes_no_sum_query(self, monkeypatch):
        app, pool, _broker, calls = _app(monkeypatch, limits_fn=_limits())
        assert _post(app).status_code == 200 and len(calls) == 1
        assert pool.sums() == []


class TestFailClosed:
    def test_limits_unreadable_is_503(self, monkeypatch):
        def broken(*_a):
            raise RuntimeError("rules table unreadable")

        app, _pool, broker, calls = _app(monkeypatch, limits_fn=broken)
        res = _post(app)
        assert res.status_code == 503
        assert res.headers["x-platform-error-code"] == "quota.unavailable"
        assert res.json()["error"] == "quota.unavailable"
        assert calls == [] and ("deny", "quota.unavailable") in broker.ops

    def test_usage_unreadable_with_a_cap_is_503(self, monkeypatch):
        pool = _Pool(sum_error=True)
        app, _pool, _broker, calls = _app(
            monkeypatch, limits_fn=_limits(tokens_per_day_user=100), pool=pool
        )
        res = _post(app)
        assert res.status_code == 503 and res.json()["error"] == "quota.unavailable"
        assert calls == []


class TestSiteSource:
    @pytest.fixture
    def site_app(self, monkeypatch):
        lease = _lease(
            source_ids=("src-a", SITE_SOURCE_ID),
            project="proj-a",
            owner_id="alice",
            venue="rccs",
        )
        monkeypatch.setattr("umrs.infer_proxy.app.lease_by_token", lambda _p, _t: lease)
        monkeypatch.setenv("GATEWAY_LITELLM_SERVICE_KEY", "site-service-key")

        def build(pool, limits_fn):
            calls: list = []

            def handler(request):
                calls.append(request)
                return _stream_response(
                    body=USAGE_BODY, headers={"content-type": "application/json"}
                )

            broker = _FakeBroker()
            app = create_app(
                umrs_db_pool=pool,
                unseal_privkey="unused",
                sealed_store=_FakeStore(None),
                broker=broker,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                site_registry=_registry(),
                quota_enforcer=QuotaEnforcer(pool, limits_fn, clock=_Clock()),
            )
            return app, calls

        return build

    def test_site_source_is_capped_and_recorded(self, site_app):
        pool = _Pool(used={"user": 100})
        app, calls = site_app(pool, _limits(tokens_per_day_user=100))
        res = _post(app, url="/s/sess-1/site/v1/chat/completions", model="qwen3-8b")
        assert res.status_code == 429 and calls == []

        pool = _Pool(used={"user": 1})
        app, calls = site_app(pool, _limits(tokens_per_day_user=100))
        res = _post(app, url="/s/sess-1/site/v1/chat/completions", model="qwen3-8b")
        assert res.status_code == 200 and len(calls) == 1
        row = pool.inserts()[0]
        assert (row[3], row[4], row[5], row[6]) == ("qwen3-8b", "qwen3-8b", "ep-1", "site")
        assert row[-3:] == ("rccs", "sess-1", "infer_proxy")
