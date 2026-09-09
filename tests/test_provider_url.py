"""Server-side provider base_url validation (R0-3, §3.1.3 vectors)."""

from __future__ import annotations

import socket

import pytest
from umrs.security.provider_url import ProviderUrlError, validate_provider_base_url

# ── Scheme / shape rules ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url,reason",
    [
        ("http://api.example.com/v1", "scheme_not_https"),
        ("https://user:pass@api.example.com/v1", "userinfo_forbidden"),
        ("https://api.example.com/v1#x", "fragment_forbidden"),
        ("https://api.example.com/v1?key=abc", "credential_query_key"),
        ("https://api.example.com/v1?api-key=abc", "credential_query_key"),
        ("", "url_empty"),
        ("https://" + "a" * 2041 + ".com", "url_too_long"),
    ],
)
def test_shape_rules(url: str, reason: str) -> None:
    with pytest.raises(ProviderUrlError, match=reason):
        validate_provider_base_url(url)


def test_trusted_http_host_allowed(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(["104.18.2.161"]))
    result = validate_provider_base_url(
        "http://litellm.localnet:4000/v1", trusted_http_hosts=("litellm.localnet",)
    )
    assert result.allow_host == "http://litellm.localnet:4000"


def test_untrusted_http_host_still_refused() -> None:
    with pytest.raises(ProviderUrlError, match="scheme_not_https"):
        validate_provider_base_url("http://litellm.localnet:4000/v1")


# ── Host/address rules ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url,reason",
    [
        ("https://localhost/v1", "host_infrastructure"),
        ("https://foo.localhost/v1", "host_infrastructure"),
        ("https://foo.internal/v1", "host_infrastructure"),
        ("https://foo.infra/v1", "host_infrastructure"),
        ("https://host.containers.internal/v1", "host_infrastructure"),
        ("https://127.0.0.1/v1", "address_not_global"),
        ("https://127.1/v1", "host_numeric_literal"),
        ("https://0x7f000001/v1", "host_numeric_literal"),
        ("https://2130706433/v1", "host_numeric_literal"),
        ("https://10.0.0.5/v1", "address_not_global"),
        ("https://172.16.0.1/v1", "address_not_global"),
        ("https://192.168.1.1/v1", "address_not_global"),
        ("https://169.254.169.254/v1", "address_not_global"),
        ("https://[::1]/v1", "address_not_global"),
        ("https://[fe80::1]/v1", "address_not_global"),
        ("https://[::ffff:10.0.0.1]/v1", "address_not_global"),
    ],
)
def test_infra_and_numeric_rejects(url: str, reason: str) -> None:
    with pytest.raises(ProviderUrlError, match=reason):
        validate_provider_base_url(url)


# ── DNS resolution behavior (stubbed getaddrinfo) ───────────────────────


def _fake_getaddrinfo(records):
    def fake(host, port, family, socktype):
        return [(2, 1, 6, "", (addr, 0)) for addr in records]

    return fake


def test_dns_name_passes_when_all_addresses_public(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(["104.18.2.161", "104.18.3.161"]))
    result = validate_provider_base_url("https://api.example-provider.com/v1")
    assert result.allow_host == "https://api.example-provider.com"


def test_dns_name_rejected_when_any_address_private(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(["104.18.2.161", "10.0.0.5"]))
    with pytest.raises(ProviderUrlError, match="address_not_global"):
        validate_provider_base_url("https://sneaky.example.com")


def test_dns_rejected_on_link_local(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(["169.254.169.254"]))
    with pytest.raises(ProviderUrlError, match="address_not_global"):
        validate_provider_base_url("https://imds.example.internal.cloud")


def test_unresolvable_name_rejected(monkeypatch):
    def boom(host, port, family, socktype):
        raise socket.gaierror()

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(ProviderUrlError, match="host_unresolvable"):
        validate_provider_base_url("https://no-such-host.invalid-zzz")


# ── Normalization ───────────────────────────────────────────────────────


def test_normalization_strips_trailing_slash_and_keeps_prefix(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(["93.184.216.34"]))
    result = validate_provider_base_url("https://api.example-provider.com/v1/")
    assert result.base_url == "https://api.example-provider.com/v1"
    assert result.allow_host == "https://api.example-provider.com"


def test_normalization_keeps_nondefault_port(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(["93.184.216.34"]))
    result = validate_provider_base_url("https://api.example-provider.com:8443/v1")
    assert result.allow_host == "https://api.example-provider.com:8443"
