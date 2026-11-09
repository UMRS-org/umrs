"""``scheduler.hpc_access`` is parsed explicitly and validated fail-closed.

ADR-112 D-1 (wave A, 2026-11-08). The generic loader drops unknown keys, so
before this a ``hpc_access:`` block in a site file silently vanished, and a
misspelt ``allowed_projects`` would have opened a lane. Every invalid value
now refuses to load with a ``ValueError`` that names the key.
"""

from __future__ import annotations

import copy
import logging
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml
from umrs.orchestrator.site_config import (
    HpcAccessConfig,
    ServiceAccountAccess,
    UserAccountAccess,
    _parse_raw,
    load_site_config,
)

_BASE: dict[str, Any] = {
    "name": "site-x",
    "scheduler": {
        "kind": "slurm-remote",
        "hpc_access": {
            "operator": "Centre X",
            "modes_allowed": ["service_account"],
            "default_mode": "service_account",
            "service_account": {
                "login": "op1",
                "personal": True,
                "consent_ref": "",
                "allowed_projects": ["proj-a", "proj-b"],
            },
        },
    },
    "extra": {"ssh_user": "op1"},
}


def _raw(**hpc_overrides: Any) -> dict[str, Any]:
    raw = copy.deepcopy(_BASE)
    raw["scheduler"]["hpc_access"].update(hpc_overrides)
    return raw


def _service(**overrides: Any) -> dict[str, Any]:
    service = copy.deepcopy(_BASE["scheduler"]["hpc_access"]["service_account"])
    service.update(overrides)
    return service


def test_block_is_parsed_into_typed_dataclasses() -> None:
    hpc = _parse_raw(_raw()).scheduler.hpc_access
    assert hpc == HpcAccessConfig(
        operator="Centre X",
        modes_allowed=["service_account"],
        default_mode="service_account",
        service_account=ServiceAccountAccess(
            login="op1", personal=True, consent_ref="", allowed_projects=["proj-a", "proj-b"]
        ),
        user_account=None,
    )


def test_block_survives_load_from_a_file(tmp_path: Path) -> None:
    # The old loader dropped the key: hpc_access read as absent.
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(_raw()))
    hpc = load_site_config(str(path)).scheduler.hpc_access
    assert hpc is not None and hpc.service_account is not None
    assert hpc.service_account.allowed_projects == ["proj-a", "proj-b"]


_USER_ACCOUNT: dict[str, Any] = {
    "methods": ["gated_key"],
    "signing_domain": "platform.example.org",
}


def _user_raw(**user_overrides: Any) -> dict[str, Any]:
    return _raw(
        modes_allowed=["service_account", "user_account"],
        user_account={**_USER_ACCOUNT, **user_overrides},
    )


def test_user_account_block_is_parsed() -> None:
    hpc = _parse_raw(_user_raw()).scheduler.hpc_access
    assert hpc is not None
    assert hpc.user_account == UserAccountAccess(
        methods=["gated_key"], signing_domain="platform.example.org"
    )


def test_user_account_wire_keys_are_parsed() -> None:
    # Plan §4 B1.5a: the four keys the gate, the server and r4 agree on.
    raw = _user_raw(
        from_address="10.0.0.0/8,front.example.org", uv_required=True, gate_min_version=2
    )
    ua = _parse_raw(raw).scheduler.hpc_access.user_account  # type: ignore[union-attr]
    assert ua is not None
    assert (ua.from_address, ua.uv_required, ua.gate_min_version) == (
        "10.0.0.0/8,front.example.org",
        True,
        2,
    )


def test_user_account_defaults_are_the_contract_defaults() -> None:
    ua = _parse_raw(_user_raw()).scheduler.hpc_access.user_account  # type: ignore[union-attr]
    assert ua is not None
    assert (ua.from_address, ua.uv_required, ua.gate_min_version) == ("", False, 1)


def test_signing_domain_is_not_required_while_user_account_is_not_allowed() -> None:
    raw = _raw(user_account={"methods": ["gated_key"]})
    hpc = _parse_raw(raw).scheduler.hpc_access
    assert hpc is not None and hpc.user_account is not None
    assert hpc.user_account.signing_domain == ""


@pytest.mark.parametrize(
    "user_account",
    [
        None,  # user_account allowed but no block at all
        {"methods": ["gated_key"]},
        {**_USER_ACCOUNT, "signing_domain": ""},
        {**_USER_ACCOUNT, "signing_domain": "<platform DNS name>"},
        {**_USER_ACCOUNT, "signing_domain": "localhost"},
        {**_USER_ACCOUNT, "signing_domain": "Platform.Example.org"},
        {**_USER_ACCOUNT, "signing_domain": "a..b"},
        {**_USER_ACCOUNT, "signing_domain": "-bad.example.org"},
    ],
)
def test_user_account_without_a_dns_signing_domain_refuses_to_load(user_account) -> None:
    raw = _raw(modes_allowed=["service_account", "user_account"], user_account=user_account)
    with pytest.raises(ValueError, match="signing_domain"):
        _parse_raw(raw)


@pytest.mark.parametrize(
    ("overrides", "names"),
    [
        ({"uv_required": "yes"}, "uv_required"),
        ({"gate_min_version": 0}, "gate_min_version"),
        ({"gate_min_version": "1"}, "gate_min_version"),
        ({"gate_min_version": True}, "gate_min_version"),
        ({"from_address": 'a" ,command="x'}, "from_address"),
        ({"from_address": "a b"}, "from_address"),
        # 2026-11-08 integration: the old pattern accepted these, which the
        # gate's own authorized_keys builder then refused at `r4 hpc add`.
        ({"from_address": "10.0.0.1,,10.0.0.2"}, "from_address"),
        ({"from_address": "10.0.0.1,"}, "from_address"),
        ({"from_address": "[::1]"}, "from_address"),
        ({"from_address": "10.0.0.!1"}, "from_address"),
        ({"signing_key": "x"}, "signing_key"),
    ],
)
def test_invalid_user_account_values_refuse_to_load(overrides, names) -> None:
    with pytest.raises(ValueError, match=names):
        _parse_raw(_user_raw(**overrides))


def test_empty_consent_with_empty_list_is_valid_and_means_admins_only() -> None:
    raw = _raw(service_account=_service(allowed_projects=[]))
    hpc = _parse_raw(raw).scheduler.hpc_access
    assert hpc is not None and hpc.service_account is not None
    assert hpc.service_account.allowed_projects == []


def test_absent_block_is_undeclared() -> None:
    raw = copy.deepcopy(_BASE)
    del raw["scheduler"]["hpc_access"]
    assert _parse_raw(raw).scheduler.hpc_access is None


def test_login_check_is_skipped_without_an_ssh_user() -> None:
    raw = _raw()
    raw["extra"] = {}
    assert _parse_raw(raw).scheduler.hpc_access is not None


@pytest.mark.parametrize(
    ("overrides", "names"),
    [
        ({"operater": "typo"}, "operater"),
        ({"service_account": _service(allowed_project=["p"])}, "allowed_project"),
        ({"user_account": {"method": ["gated_key"]}}, "method"),
        ({"modes_allowed": ["service_account", "per_user_ssh"]}, "modes_allowed"),
        ({"modes_allowed": "service_account"}, "modes_allowed"),
        ({"default_mode": "user_account"}, "default_mode"),
        ({"modes_allowed": [], "default_mode": ""}, "default_mode"),
        ({"service_account": None}, "service_account"),
        ({"service_account": _service(login="someone-else")}, "service_account.login"),
        ({"service_account": _service(login="")}, "service_account.login"),
        ({"service_account": _service(allowed_projects="proj-a")}, "allowed_projects"),
        ({"service_account": _service(personal="yes")}, "personal"),
        ({"service_account": ["op1"]}, "service_account"),
    ],
)
def test_invalid_values_refuse_to_load_naming_the_key(overrides, names) -> None:
    with pytest.raises(ValueError, match=names):
        _parse_raw(_raw(**overrides))


def test_a_non_mapping_block_is_refused() -> None:
    raw = copy.deepcopy(_BASE)
    raw["scheduler"]["hpc_access"] = ["service_account"]
    with pytest.raises(ValueError, match="scheduler.hpc_access"):
        _parse_raw(raw)


def test_deprecated_keys_still_load_and_are_kept_readable() -> None:
    raw = _raw()
    raw["scheduler"].update(
        slurm_delegation="per_user_ssh",
        service_pubkey_path="/x/umrs-service.pub",
        slurm_proxy_path="/x/proxy",
    )
    scheduler = _parse_raw(raw).scheduler
    assert scheduler.slurm_delegation == "per_user_ssh"
    assert scheduler.hpc_access is not None


def test_per_user_ssh_without_hpc_access_logs_one_error(caplog) -> None:
    site = textwrap.dedent(
        """
        name: legacy
        scheduler:
          kind: slurm-remote
          slurm_delegation: per_user_ssh
        """
    )
    with caplog.at_level(logging.ERROR, logger="umrs.orchestrator.site_config"):
        cfg = _parse_raw(yaml.safe_load(site))
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "hpc_access" in errors[0].getMessage()
    assert cfg.scheduler.hpc_access is None


def test_per_user_ssh_with_hpc_access_still_logs_one_error(caplog) -> None:
    """Plan §3 A.1: a site file that still says ``per_user_ssh`` logs one
    ERROR naming ``hpc_access`` even when it declares the block (review
    2026-11-08: the old code logged only when the block was absent)."""
    raw = _raw()
    raw["scheduler"]["slurm_delegation"] = "per_user_ssh"
    with caplog.at_level(logging.ERROR, logger="umrs.orchestrator.site_config"):
        cfg = _parse_raw(raw)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "hpc_access" in errors[0].getMessage()
    assert "ignored" in errors[0].getMessage()
    assert cfg.scheduler.hpc_access is not None


def test_a_service_account_site_logs_no_retirement_error(caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="umrs.orchestrator.site_config"):
        _parse_raw(_raw())
    assert not [r for r in caplog.records if r.levelno == logging.ERROR]


_SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "umrs"
    / "orchestrator"
    / "site-config.schema.json"
)


def test_the_schema_accepts_the_wire_keys_and_requires_the_domain() -> None:
    import json

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(_SCHEMA.read_text())
    jsonschema.validate(_user_raw(from_address="", uv_required=False, gate_min_version=1), schema)
    missing = _raw(
        modes_allowed=["service_account", "user_account"], user_account={"methods": ["gated_key"]}
    )
    with pytest.raises(jsonschema.ValidationError, match="signing_domain"):
        jsonschema.validate(missing, schema)
