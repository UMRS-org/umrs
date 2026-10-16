"""ADR-062 amendment (2026-10-15): the third provider class, institutional.

An inference service run by the user's own institution outside UMRS (Rikyu
first: api.rikyu.r-ccs.riken.jp) is admitted for ``data_class=sensitive``
projects at every enforcement seam (the venue-start gate and the infer-proxy
lease gate); commercial and other public providers stay refused.

Membership comes from an allowlist the site keeps — never a user label:
``UMRS_INSTITUTIONAL_PROVIDER_HOSTS`` (comma/space separated) replaces the
packaged default; an empty value means no institutional hosts. connect-info
advertises the effective list so r4 classifies the same way.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from umrs.shared.project_data_class import (
    DEFAULT_INSTITUTIONAL_PROVIDER_HOSTS,
    institutional_provider_hosts,
    is_institutional_url,
    url_host,
)

RIKYU = "api.rikyu.r-ccs.riken.jp"


class TestAllowlist:
    def test_default_is_rikyu(self, monkeypatch):
        monkeypatch.delenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", raising=False)
        hosts = institutional_provider_hosts()
        assert hosts == DEFAULT_INSTITUTIONAL_PROVIDER_HOSTS
        assert RIKYU in hosts

    def test_env_replaces_the_default(self, monkeypatch):
        monkeypatch.setenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", "llm.example.jp, other.example.jp")
        hosts = institutional_provider_hosts()
        assert "llm.example.jp" in hosts
        assert "other.example.jp" in hosts
        assert RIKYU not in hosts  # replaced, not extended

    def test_env_empty_means_no_institutional(self, monkeypatch):
        monkeypatch.setenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", " ")
        assert institutional_provider_hosts() == frozenset()


class TestAllowlistEntryNormalization:
    """F6 (wave-2 review, 2026-10-16): an operator who writes an entry with
    a scheme, a port, a trailing dot or IPv6 brackets means the bare host.
    Before, such entries were stored verbatim and silently never matched."""

    @pytest.fixture(autouse=True)
    def _fresh_parse_cache(self):
        # The parsed list is cached per raw value (one warning per value).
        from umrs.shared import project_data_class

        parse = getattr(project_data_class, "_parse_allowlist", None)
        cache_clear = getattr(parse, "cache_clear", None)
        if cache_clear:
            cache_clear()
        yield

    @pytest.mark.parametrize(
        "entry",
        [
            "https://api.rikyu.r-ccs.riken.jp",
            "https://api.rikyu.r-ccs.riken.jp/v1",  # a pasted base URL
            "HTTP://API.Rikyu.R-CCS.riken.jp:8443",
            "api.rikyu.r-ccs.riken.jp:443",
            "api.rikyu.r-ccs.riken.jp.",  # fully-qualified trailing dot
            "API.RIKYU.R-CCS.RIKEN.JP.",
        ],
    )
    def test_entry_shapes_normalize_to_the_bare_host(self, monkeypatch, entry):
        monkeypatch.setenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", entry)
        assert institutional_provider_hosts() == frozenset({RIKYU})
        assert is_institutional_url("https://api.rikyu.r-ccs.riken.jp/v1")

    @pytest.mark.parametrize(
        "entry,host",
        [
            ("[fd00::1]", "fd00::1"),
            ("[FD00::1]:8443", "fd00::1"),
            ("fd00:0:0::1", "fd00::1"),  # bare v6, canonical form
            ("10.0.0.5:8000", "10.0.0.5"),
        ],
    )
    def test_ip_entries_normalize(self, monkeypatch, entry, host):
        monkeypatch.setenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", entry)
        assert institutional_provider_hosts() == frozenset({host})

    @pytest.mark.parametrize(
        ("entry", "shown", "why"),
        [
            # wildcards are not supported (exact match only)
            ("*.riken.jp", "*.riken.jp", "wildcard"),
            ("user@api.rikyu.r-ccs.riken.jp", "***@api.rikyu.r-ccs.riken.jp", "user name"),
            (
                "https://u:p@api.rikyu.r-ccs.riken.jp",
                "https://***@api.rikyu.r-ccs.riken.jp",
                "user name",
            ),
            # not an http(s) service
            ("ftp://api.rikyu.r-ccs.riken.jp", "ftp://api.rikyu.r-ccs.riken.jp", "http(s)"),
            ("api.rikyu.r-ccs.riken.jp:notaport", "api.rikyu.r-ccs.riken.jp:notaport", "host"),
            (
                "https://api.rikyu.r-ccs.riken.jp/?q=1",
                "https://api.rikyu.r-ccs.riken.jp/***",
                "query",
            ),
            ("https://", "https://", "host"),
            ("bad..host.jp", "bad..host.jp", "host"),
        ],
    )
    def test_unparseable_entries_are_ignored_with_a_warning(
        self, monkeypatch, caplog, entry, shown, why
    ):
        monkeypatch.setenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", f"llm.example.jp,{entry}")
        with caplog.at_level(logging.WARNING, logger="umrs.shared.project_data_class"):
            hosts = institutional_provider_hosts()
        assert hosts == frozenset({"llm.example.jp"})  # the good entry stays
        messages = [rec.getMessage() for rec in caplog.records]
        assert any(f"'{shown}'" in m and why in m for m in messages), caplog.text

    @pytest.mark.parametrize(
        ("entry", "secret", "host"),
        [
            ("https://svc:DUMMY-PASSWORD@llm.example.ac.jp", "DUMMY-PASSWORD", "llm.example.ac.jp"),
            ("svc:DUMMY-PASSWORD@llm.example.ac.jp", "DUMMY-PASSWORD", "llm.example.ac.jp"),
            ("https://svc:DUMMY/PASS@llm.example.ac.jp", "DUMMY", "llm.example.ac.jp"),
            ("https://svc:DUMMY?PASS@llm.example.ac.jp", "DUMMY", "llm.example.ac.jp"),
            ("https://svc:DUMMY@PASS@llm.example.ac.jp", "DUMMY", "llm.example.ac.jp"),
            ("ftp://svc:DUMMY-PASSWORD@llm.example.ac.jp", "DUMMY-PASSWORD", "llm.example.ac.jp"),
            ("https://llm2.example.ac.jp/v1?api_key=DUMMY-KEY", "DUMMY-KEY", "llm2.example.ac.jp"),
            ("https://llm2.example.ac.jp?api_key=DUMMY-KEY", "DUMMY-KEY", "llm2.example.ac.jp"),
            (
                "https://llm3.example.ac.jp/v1#token=DUMMY-TOKEN",
                "DUMMY-TOKEN",
                "llm3.example.ac.jp",
            ),
            ("ftp://llm4.example.ac.jp/DUMMY-PATH-TOKEN", "DUMMY-PATH-TOKEN", "llm4.example.ac.jp"),
        ],
    )
    def test_warning_never_echoes_a_credential(self, monkeypatch, caplog, entry, secret, host):
        # A rejected entry may be a URL with a password, an api_key query or
        # a token path: the warning names the host so the operator can find
        # the entry, and never writes the credential into the server logs.
        monkeypatch.setenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", f"ok.example.jp,{entry}")
        with caplog.at_level(logging.WARNING, logger="umrs.shared.project_data_class"):
            hosts = institutional_provider_hosts()
        assert hosts == frozenset({"ok.example.jp"})
        assert caplog.records, "the rejected entry must still be reported"
        assert secret not in caplog.text
        assert host in caplog.text

    def test_warning_is_logged_once_per_value_not_per_request(self, monkeypatch, caplog):
        # The lease gate reads the list on every proxied request.
        monkeypatch.setenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", "*.once.example.jp")
        with caplog.at_level(logging.WARNING, logger="umrs.shared.project_data_class"):
            for _ in range(5):
                institutional_provider_hosts()
        hits = [r for r in caplog.records if "*.once.example.jp" in r.getMessage()]
        assert len(hits) == 1


class TestUrlHost:
    @pytest.mark.parametrize(
        "url,host",
        [
            ("https://api.rikyu.r-ccs.riken.jp/v1", RIKYU),
            ("http://API.RIKYU.R-CCS.RIKEN.JP:8443/x", RIKYU),
            ("api.rikyu.r-ccs.riken.jp", RIKYU),  # scheme-less allow_host
            ("https://api.rikyu.r-ccs.riken.jp./v1", RIKYU),  # trailing dot
            ("[::1]:8080/v1", "::1"),
            ("", ""),
        ],
    )
    def test_host_shapes(self, url, host):
        assert url_host(url) == host

    @pytest.mark.parametrize(
        "url",
        [
            # F7: userinfo can make a URL parser and a connect parser
            # disagree on the host — fail closed, never a host.
            "https://user:pw@api.Example.com:443/v1",
            "https://evil.example.com:x@api.rikyu.r-ccs.riken.jp",
            "https://api.rikyu.r-ccs.riken.jp@evil.example.com",
            "api.rikyu.r-ccs.riken.jp@evil.example.com",
            # not an http(s) service
            "ftp://api.rikyu.r-ccs.riken.jp",
        ],
    )
    def test_userinfo_or_foreign_scheme_has_no_host(self, url):
        assert url_host(url) == ""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://api.rikyu.r-ccs.riken.jp/v1", True),
            ("http://api.rikyu.r-ccs.riken.jp:8080", True),
            ("api.rikyu.r-ccs.riken.jp", True),
            ("https://not-rikyu.r-ccs.riken.jp", False),  # no suffix games
            ("https://api.openai.com", False),
            ("", False),
            ("127.0.0.1:11434", False),  # loopback is local, not institutional
            ("https://api.rikyu.r-ccs.riken.jp.", True),  # trailing dot, same host
            ("https://api.rikyu.r-ccs.riken.jp.evil.example.com", False),
            ("https://evil.example.com:x@api.rikyu.r-ccs.riken.jp", False),  # userinfo
            ("https://api.rikyu.r-ccs.riken.jp@evil.example.com", False),
        ],
    )
    def test_classification(self, monkeypatch, url, expected):
        monkeypatch.delenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", raising=False)
        assert is_institutional_url(url) is expected


# ── the lease gate (infer proxy) ────────────────────────────────────────────


class _FakeCur:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, _sql, params):
        self._row = (self._rows.get(params[0]),) if params[0] in self._rows else None

    def fetchone(self):
        return self._row

    _row = None


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCur(self._rows)


class _FakePool:
    """{project: data_class} psycopg2-shaped pool."""

    def __init__(self, rows):
        self._rows = rows

    def getconn(self):
        return _FakeConn(self._rows)

    def putconn(self, _conn):
        pass


def _full(allow_host: str, provider_kind: str = "openai"):
    return SimpleNamespace(provider_kind=provider_kind, allow_host=allow_host)


class TestLeaseGateInstitutional:
    """``_source_class_refused``: public-kind stays refused; institutional is
    admitted for a sensitive project (ADR-062 amendment seam 2)."""

    def _refused(self, monkeypatch, allow_host, project="proj-s", env=None):
        from umrs.infer_proxy.app import _source_class_refused

        if env is None:
            monkeypatch.delenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", raising=False)
        else:
            monkeypatch.setenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", env)
        lease = SimpleNamespace(project=project)
        return _source_class_refused(_FakePool({project: "sensitive"}), lease, _full(allow_host))

    def test_institutional_admitted_for_sensitive(self, monkeypatch):
        assert self._refused(monkeypatch, "https://api.rikyu.r-ccs.riken.jp") is False

    def test_commercial_still_refused_for_sensitive(self, monkeypatch):
        assert self._refused(monkeypatch, "https://api.openai.com") is True

    def test_env_replacement_can_refuse_rikyu(self, monkeypatch):
        # A site that runs no institutional service empties the list.
        assert self._refused(monkeypatch, "https://api.rikyu.r-ccs.riken.jp", env="") is True


# ── connect-info advertises the list ────────────────────────────────────────


class TestConnectInfoAdvertisesAllowlist:
    def test_default_list_emitted(self, monkeypatch):
        from umrs.api import ops

        monkeypatch.delenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", raising=False)
        result = ops.ops_connect_info(x_user_id=None)
        assert result["institutional_provider_hosts"] == [RIKYU]

    def test_env_list_emitted(self, monkeypatch):
        from umrs.api import ops

        monkeypatch.setenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", "llm.example.jp")
        result = ops.ops_connect_info(x_user_id=None)
        assert result["institutional_provider_hosts"] == ["llm.example.jp"]

    def test_response_model_keeps_the_field(self, monkeypatch):
        """The route runs under response_model=ConnectInfoResponse: a key the
        model does not declare is stripped before it reaches the client."""
        from umrs.api import ops

        monkeypatch.delenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", raising=False)
        result = ops.ops_connect_info(x_user_id=None)
        parsed = ops.ConnectInfoResponse(**result)
        assert parsed.institutional_provider_hosts == [RIKYU]


# ── module texts and shape (F4/F8 of the wave-2 review) ─────────────────────


class TestGateModuleTexts:
    def test_venue_gate_module_has_no_import_after_a_definition(self):
        """F4: the gate helper sat inside the import block (E402)."""
        import ast
        import inspect

        import umrs.venues.frontend_service as mod

        body = ast.parse(inspect.getsource(mod)).body
        first_def = next(
            i for i, node in enumerate(body) if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        )
        late = [
            node.lineno
            for node in body[first_def:]
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert late == []

    def test_proxy_docstring_names_the_institutional_exemption(self):
        """F8: the request-flow summary (step 2) predated the amendment."""
        import umrs.infer_proxy.app as app

        step2 = app.__doc__.split("  3.")[0].split("  2.")[1]
        assert "institutional" in step2
