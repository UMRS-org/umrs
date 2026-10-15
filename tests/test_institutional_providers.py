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
        monkeypatch.setenv(
            "UMRS_INSTITUTIONAL_PROVIDER_HOSTS", "llm.example.jp, other.example.jp"
        )
        hosts = institutional_provider_hosts()
        assert "llm.example.jp" in hosts
        assert "other.example.jp" in hosts
        assert RIKYU not in hosts  # replaced, not extended

    def test_env_empty_means_no_institutional(self, monkeypatch):
        monkeypatch.setenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", " ")
        assert institutional_provider_hosts() == frozenset()


class TestUrlHost:
    @pytest.mark.parametrize(
        "url,host",
        [
            ("https://api.rikyu.r-ccs.riken.jp/v1", RIKYU),
            ("http://API.RIKYU.R-CCS.RIKEN.JP:8443/x", RIKYU),
            ("https://user:pw@api.Example.com:443/v1", "api.example.com"),
            ("api.rikyu.r-ccs.riken.jp", RIKYU),  # scheme-less allow_host
            ("[::1]:8080/v1", "::1"),
            ("", ""),
        ],
    )
    def test_host_shapes(self, url, host):
        assert url_host(url) == host

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
        return _source_class_refused(
            _FakePool({project: "sensitive"}), lease, _full(allow_host)
        )

    def test_institutional_admitted_for_sensitive(self, monkeypatch):
        assert self._refused(monkeypatch, "https://api.rikyu.r-ccs.riken.jp") is False

    def test_commercial_still_refused_for_sensitive(self, monkeypatch):
        assert self._refused(monkeypatch, "https://api.openai.com") is True

    def test_env_replacement_can_refuse_rikyu(self, monkeypatch):
        # A site that runs no institutional service empties the list.
        assert (
            self._refused(monkeypatch, "https://api.rikyu.r-ccs.riken.jp", env="") is True
        )


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
