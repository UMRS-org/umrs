"""One resolver for every lane that reaches a cluster (ADR-112 D-2, wave A).

Platform lanes always run under the site's platform login and never under a
user's identity (I-2). User-work lanes resolve per verified project: on a
``service_account`` site with no consent on record, only admins and the
listed projects may use the personal login (G-8). Refusals carry a typed
``hpcaccess.*`` code and are never retried under another identity (I-1).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from umrs.orchestrator.hpc_identity import (
    REFUSAL_REASONS,
    HpcAccessRefused,
    HpcIdentity,
    Lane,
    platform_error,
    refusal_message,
    resolve_hpc_identity,
)
from umrs.orchestrator.site_config import (
    HpcAccessConfig,
    ServiceAccountAccess,
    SiteConfig,
    UserAccountAccess,
    load_site_config,
)

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
LISTED = ["proj-a", "proj-b"]


def _site(
    *,
    mode: str | None = "service_account",
    listed: list[str] | None = None,
    consent_ref: str = "",
    personal: bool = True,
) -> SiteConfig:
    site = SiteConfig(name="site-x", extra={"ssh_user": "op1"})
    site.scheduler.account = "master-acct"
    if mode is None:
        return site
    site.scheduler.hpc_access = HpcAccessConfig(
        operator="Centre X",
        modes_allowed=["service_account", "user_account"],
        default_mode=mode,
        service_account=ServiceAccountAccess(
            login="op1",
            personal=personal,
            consent_ref=consent_ref,
            allowed_projects=list(LISTED if listed is None else listed),
        ),
        user_account=UserAccountAccess(methods=["gated_key"]),
    )
    return site


def _user_work(site: SiteConfig, project: str | None, *, admin: bool = False) -> HpcIdentity:
    return resolve_hpc_identity(
        site, lane=Lane.USER_WORK, user_id="alice", project_id=project, is_admin=admin
    )


def _refusal(site: SiteConfig, project: str | None, *, admin: bool = False) -> HpcAccessRefused:
    with pytest.raises(HpcAccessRefused) as info:
        _user_work(site, project, admin=admin)
    return info.value


# --- platform lanes ---------------------------------------------------------


@pytest.mark.parametrize("mode", ["service_account", "user_account", None])
@pytest.mark.parametrize("admin", [True, False])
def test_platform_lane_is_always_the_platform_login(mode, admin) -> None:
    ident = resolve_hpc_identity(
        _site(mode=mode, listed=[]),
        lane=Lane.PLATFORM,
        user_id="mallory",
        project_id="not-listed",
        is_admin=admin,
    )
    assert (ident.lane, ident.mode, ident.login) == (Lane.PLATFORM, "service_account", "op1")
    assert ident.declared is (mode is not None)


def test_platform_lane_accepts_the_string_name() -> None:
    ident = resolve_hpc_identity(
        _site(), lane="platform", user_id=None, project_id=None, is_admin=False
    )
    assert ident.lane is Lane.PLATFORM and ident.personal is True


@pytest.mark.parametrize("front", ["rccs-front", "ai4s-front", "bdrt-front", "octopus-front"])
def test_platform_lane_on_every_front_is_its_ssh_user(front) -> None:
    site = load_site_config(str(DEPLOY / front / "site.yaml"))
    ident = resolve_hpc_identity(
        site, lane=Lane.PLATFORM, user_id="u", project_id=None, is_admin=False
    )
    assert (ident.mode, ident.login) == ("service_account", site.extra["ssh_user"])
    assert ident.declared and ident.personal


# --- user work: service_account ---------------------------------------------


def test_admin_runs_user_work_on_any_project() -> None:
    ident = _user_work(_site(listed=[]), "anything", admin=True)
    assert (ident.lane, ident.mode, ident.login, ident.declared) == (
        Lane.USER_WORK,
        "service_account",
        "op1",
        True,
    )


def test_listed_project_runs_user_work() -> None:
    ident = _user_work(_site(), "proj-b")
    assert (ident.mode, ident.login, ident.personal) == ("service_account", "op1", True)


def test_unlisted_project_is_refused_naming_the_site_and_the_personal_login() -> None:
    exc = _refusal(_site(), "proj-z")
    assert (exc.reason, exc.code, exc.site) == (
        "project_not_allowed",
        "hpcaccess.project_not_allowed",
        "site-x",
    )
    assert "site-x" in exc.detail and "personal" in exc.detail and "proj-z" in exc.detail
    assert "op1" not in exc.detail  # never whose login it is


def test_empty_list_without_consent_is_admins_only() -> None:
    exc = _refusal(_site(listed=[]), "proj-a")
    assert exc.reason == "project_not_allowed"
    assert "only administrators" in exc.detail


@pytest.mark.parametrize("project", [None, "", "default"])
def test_unverified_project_never_matches_even_when_listed(project) -> None:
    site = _site(listed=["default", ""])
    assert _refusal(site, project).reason == "project_not_allowed"


def test_a_claimed_project_only_counts_when_the_caller_passes_it_as_verified() -> None:
    # The resolver sees exactly what it is given: a None verified project is
    # refused although the caller may have *claimed* a listed one.
    assert _user_work(_site(), "proj-a").login == "op1"
    assert _refusal(_site(), None).reason == "project_not_allowed"


def test_consent_with_a_list_still_enforces_the_list() -> None:
    site = _site(consent_ref="LETTER-1")
    assert _user_work(site, "proj-a").declared
    assert _refusal(site, "proj-z").reason == "project_not_allowed"


def test_consent_with_an_empty_list_allows_every_verified_project() -> None:
    site = _site(listed=[], consent_ref="LETTER-1")
    assert _user_work(site, "proj-z").mode == "service_account"
    assert _refusal(site, "default").reason == "project_not_allowed"


def test_service_login_that_is_not_personal_says_so() -> None:
    exc = _refusal(_site(personal=False), "proj-z")
    assert "personal" not in exc.detail and "service account" in exc.detail


# --- user work: undeclared and user_account ---------------------------------


def test_undeclared_site_refuses_non_admins() -> None:
    exc = _refusal(_site(mode=None), "proj-a")
    assert (exc.reason, exc.code) == ("mode_undeclared", "hpcaccess.mode_undeclared")


def test_undeclared_site_serves_an_admin_on_the_platform_login_undeclared() -> None:
    ident = _user_work(_site(mode=None), "proj-a", admin=True)
    assert (ident.mode, ident.login, ident.declared, ident.personal) == (
        "service_account",
        "op1",
        False,
        False,
    )


@pytest.mark.parametrize("admin", [True, False])
def test_user_account_is_not_enrolled_until_wave_b_even_for_admins(admin) -> None:
    exc = _refusal(_site(mode="user_account"), "proj-a", admin=admin)
    assert (exc.reason, exc.code) == ("not_enrolled", "hpcaccess.not_enrolled")


def test_user_work_never_resolves_to_user_account() -> None:
    for mode in ("service_account", None):
        ident = _user_work(_site(mode=mode), "proj-a", admin=True)
        assert ident.mode == "service_account"


# --- record, error type, messages -------------------------------------------


def test_identity_records_mode_login_account_and_lane() -> None:
    ident = resolve_hpc_identity(
        _site(),
        lane=Lane.USER_WORK,
        user_id="alice",
        project_id="proj-a",
        is_admin=False,
        slurm_account="grp-a",
    )
    assert ident.to_metrics() == {
        "site": "site-x",
        "lane": "user_work",
        "access_mode": "service_account",
        "site_login": "op1",
        "slurm_account": "grp-a",
        "personal": True,
        "declared": True,
    }
    assert _user_work(_site(), "proj-a").slurm_account is None


def test_refusal_reasons_are_the_typed_wave_a_codes() -> None:
    from r4_contracts.errors import HPCACCESS_ERRORS

    assert REFUSAL_REASONS == ("mode_undeclared", "project_not_allowed", "not_enrolled")
    assert all(f"hpcaccess.{r}" in HPCACCESS_ERRORS for r in REFUSAL_REASONS)
    with pytest.raises(ValueError):
        HpcAccessRefused("made_up", "site-x", "d")


def test_refusal_message_and_platform_error_carry_the_code() -> None:
    exc = _refusal(_site(), "proj-z")
    assert refusal_message(exc).endswith("(hpcaccess.project_not_allowed)")
    err = platform_error(exc)
    assert (err.code, err.http_status, err.detail) == (
        "hpcaccess.project_not_allowed",
        403,
        {"site": "site-x"},
    )
    assert err.message == refusal_message(exc)


def test_unknown_lane_fails_closed() -> None:
    with pytest.raises(ValueError):
        resolve_hpc_identity(_site(), lane="bogus", user_id="u", project_id="p", is_admin=True)
