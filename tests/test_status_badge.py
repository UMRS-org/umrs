"""Public service-model badge (ADR-106 amendment, 2026-11-07).

``GET /umrs/status/service-badge`` is anonymous, so it must answer exactly the
shields.io endpoint keys, from integers and fixed words, and count only the
replicas the service tier owns. These tests pin the kept-up / on-demand split,
READY vs DEGRADED, the empty / unset / unreadable roster answers, service-tier
ownership (a user's endpoint of the same model never counts) and the leak
contract (exact keys, no roster or registry string in the body).
"""

from __future__ import annotations

import inspect
import json
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import status_badge
from umrs.orchestrator import endpoint_registry as er
from umrs.orchestrator import provision_roster, service_badge
from umrs.orchestrator.service_controller import DIGEST_KEY

READY = er.EndpointStatus.READY
DEGRADED = er.EndpointStatus.DEGRADED
STARTING = er.EndpointStatus.STARTING

# Two kept-up roles (a service replica and the keep_alive relay), one service
# role with replicas: 0 and one alias with no lifecycle block (on demand).
ROSTER = """\
pool: test
endpoints:
  - name: fast
    model: fast-model
    aliases: [svc-fast]
    service: {project_id: svc-proj, user: svc-user, replicas: 1}
  - name: nimble
    model: decision-model
    aliases: [svc-decision]
    keep_alive: {project_id: svc-proj, user: svc-user}
  - name: embed
    model: embed-model
    aliases: [svc-embed]
    service: {project_id: svc-proj, user: svc-user, replicas: 0}
  - name: chat
    model: chat-model
    aliases: [svc-chat]
"""

ON_DEMAND_ONLY = """\
pool: test
endpoints:
  - name: embed
    model: embed-model
    aliases: [svc-embed]
    service: {project_id: svc-proj, user: svc-user, replicas: 0}
"""

NO_ROLES = """\
pool: test
endpoints:
  - name: plain
    model: plain-model
"""

CO_SERVED = """\
pool: test
endpoints:
  - name: vision
    model: vision-model
    aliases: [svc-vision]
    service: {project_id: svc-proj, user: svc-user, replicas: 0}
    co_serve:
      - {name: ocr, model: ocr-model, aliases: [svc-ocr], model_type: vision, adapter: sam3}
"""

SERVICE_META = {DIGEST_KEY: "d1g3st", "tier": "system", "owner_id": "svc-user"}
RELAY_META = {"tier": "system", "owner_type": "system", "owner_id": "svc-user"}

BADGE_KEYS = {"schemaVersion", "label", "message", "color", "cacheSeconds"}
MESSAGE = re.compile(r"^(\d+/\d+ up|\d+ up|idle|none configured|unknown|down)$")


@pytest.fixture
def roster(tmp_path, monkeypatch):
    """Publish ``text`` as this site's roster."""

    def publish(text: str) -> None:
        path = tmp_path / "pool.yaml"
        path.write_text(text)
        monkeypatch.setenv(provision_roster.ROSTER_ENV_VAR, str(path))
        provision_roster._cache.clear()

    yield publish
    provision_roster._cache.clear()


class _Rows:
    """A registry stand-in; counts reads so a test can pin "one read"."""

    def __init__(self) -> None:
        self.registry = er.EndpointRegistry()
        self.reads = 0
        self._n = 0

    def add(self, model: str, status, *, project: str = "svc-proj", meta=None) -> str:
        self._n += 1
        job = f"job-{self._n}"
        self.registry.register(
            project_id=project,
            model=model,
            job_id=job,
            host="node-7",
            port=8123,
            backend="slurm",
            metadata=dict(meta or {}),
        )
        self.registry.update_status(job, status)
        return job

    def __call__(self):
        self.reads += 1
        return self.registry.list_all()


def _badge(rows: _Rows) -> dict:
    return service_badge.service_badge(rows)


# ── kept-up / on-demand split ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("entry", "kept"),
    [
        ({"service": {"replicas": 1}}, True),
        ({"service": {}}, True),  # replicas default to 1
        ({"service": {"replicas": 0}}, False),  # ADR-109 D-4 on demand
        ({"keep_alive": {"project_id": "p"}}, True),
        ({"service": {"replicas": 0}, "keep_alive": {"project_id": "p"}}, False),
        ({}, False),
    ],
)
def test_kept_up_is_a_service_with_replicas_or_keep_alive(entry, kept) -> None:
    assert service_badge.keeps_up(entry) is kept


def test_all_kept_up_roles_ready_is_green(roster) -> None:
    roster(ROSTER)
    rows = _Rows()
    rows.add("fast-model", READY, meta=SERVICE_META)
    rows.add("decision-model", READY, meta=RELAY_META)
    assert _badge(rows) == service_badge.badge("2/2 up", "brightgreen")
    assert rows.reads == 1


def test_on_demand_roles_do_not_count_toward_kept_up(roster) -> None:
    roster(ROSTER)
    rows = _Rows()
    rows.add("fast-model", READY, meta=SERVICE_META)
    rows.add("decision-model", READY, meta=RELAY_META)
    rows.add("embed-model", READY, meta=SERVICE_META)  # on demand, live
    assert _badge(rows)["message"] == "2/2 up"


def test_one_kept_up_role_missing_is_yellow(roster) -> None:
    roster(ROSTER)
    rows = _Rows()
    rows.add("fast-model", READY, meta=SERVICE_META)
    rows.add("decision-model", STARTING, meta=RELAY_META)
    assert _badge(rows) == service_badge.badge("1/2 up", "yellow")


def test_degraded_only_is_yellow_not_up(roster) -> None:
    roster(ROSTER)
    rows = _Rows()
    rows.add("fast-model", DEGRADED, meta=SERVICE_META)
    rows.add("decision-model", DEGRADED, meta=RELAY_META)
    assert _badge(rows) == service_badge.badge("0/2 up", "yellow")


def test_kept_up_roles_none_live_is_red(roster) -> None:
    roster(ROSTER)
    rows = _Rows()
    rows.add("embed-model", READY, meta=SERVICE_META)  # on demand does not rescue it
    assert _badge(rows) == service_badge.badge("0/2 up", "red")


def test_only_on_demand_roles_live_counts_them(roster) -> None:
    roster(ON_DEMAND_ONLY)
    rows = _Rows()
    rows.add("embed-model", READY, meta=SERVICE_META)
    assert _badge(rows) == service_badge.badge("1 up", "brightgreen")


def test_only_on_demand_roles_nothing_live_is_idle(roster) -> None:
    roster(ON_DEMAND_ONLY)
    rows = _Rows()
    rows.add("embed-model", DEGRADED, meta=SERVICE_META)
    assert _badge(rows) == service_badge.badge("idle", "lightgrey")


def test_co_served_role_counts_through_its_owned_primary(roster) -> None:
    roster(CO_SERVED)
    rows = _Rows()
    primary = rows.add("vision-model", READY, meta=SERVICE_META)
    rows.add("ocr-model", READY, meta={"co_served_by": primary})
    # a co-served row of a user's job does not count
    user_job = rows.add("vision-model", READY, project="alice-proj")
    rows.add("ocr-model", READY, project="alice-proj", meta={"co_served_by": user_job})
    assert _badge(rows) == service_badge.badge("2 up", "brightgreen")


# ── empty / unset / unreadable roster ──────────────────────────────────────


def test_no_roster_is_none_configured_without_a_registry_read(monkeypatch) -> None:
    monkeypatch.delenv(provision_roster.ROSTER_ENV_VAR, raising=False)
    rows = _Rows()
    assert _badge(rows) == service_badge.badge("none configured", "lightgrey")
    assert rows.reads == 0


def test_roster_without_roles_is_none_configured(roster) -> None:
    roster(NO_ROLES)
    assert _badge(_Rows())["message"] == "none configured"


@pytest.mark.parametrize("text", ["endpoints: [unclosed", "- just\n- a list\n"])
def test_unreadable_roster_is_unknown_not_none_configured(roster, text) -> None:
    roster(text)
    rows = _Rows()
    assert _badge(rows) == service_badge.badge("unknown", "lightgrey")
    assert rows.reads == 0


def test_missing_roster_file_is_unknown(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(provision_roster.ROSTER_ENV_VAR, str(tmp_path / "gone.yaml"))
    provision_roster._cache.clear()
    assert _badge(_Rows())["message"] == "unknown"


def test_roster_state_tells_unset_unreadable_and_ok_apart(roster, monkeypatch) -> None:
    monkeypatch.delenv(provision_roster.ROSTER_ENV_VAR, raising=False)
    assert provision_roster.roster_state() == provision_roster.ROSTER_UNSET
    roster("endpoints: [unclosed")
    assert provision_roster.roster_state() == provision_roster.ROSTER_UNREADABLE
    roster("")  # an empty file is a readable, empty roster
    assert provision_roster.roster_state() == provision_roster.ROSTER_OK


# ── service-tier ownership ────────────────────────────────────────────────


def test_users_private_endpoint_of_the_same_model_does_not_count(roster) -> None:
    roster(ROSTER)
    rows = _Rows()
    rows.add("fast-model", READY, project="alice-proj", meta={"tier": "user", "owner_id": "alice"})
    rows.add(
        "decision-model", READY, project="alice-proj", meta={"tier": "user", "owner_id": "alice"}
    )
    assert _badge(rows) == service_badge.badge("0/2 up", "red")


def test_digestless_system_tier_service_replica_counts(roster) -> None:
    """A ``service`` replica started before the job-id fix (401fa3a8) has no
    digest; the controller owns it through the relay lineage, so it counts."""
    roster(ROSTER)
    rows = _Rows()
    rows.add("fast-model", READY, meta={"tier": "system", "owner_id": "svc-user"})
    rows.add("decision-model", READY, meta=RELAY_META)
    assert _badge(rows) == service_badge.badge("2/2 up", "brightgreen")


@pytest.mark.parametrize(
    "meta",
    [
        {"tier": "user", "owner_id": "svc-user"},  # a user's row in the service project
        {"tier": "system", "owner_id": "someone-else"},  # not the policy's submitter
        {},  # no tier, no digest
    ],
)
def test_digestless_service_row_outside_the_lineage_does_not_count(roster, meta) -> None:
    roster(ROSTER)
    rows = _Rows()
    rows.add("fast-model", READY, meta=meta)
    rows.add("decision-model", READY, meta=RELAY_META)
    assert _badge(rows) == service_badge.badge("1/2 up", "yellow")


@pytest.mark.parametrize(
    "meta",
    [
        {"tier": "user", "owner_id": "svc-user"},  # not the relay's system tier
        {"tier": "system", "owner_id": "someone-else"},  # not the policy's submitter
    ],
)
def test_keep_alive_counts_only_the_relay_lineage(roster, meta) -> None:
    roster(ROSTER)
    rows = _Rows()
    rows.add("fast-model", READY, meta=SERVICE_META)
    rows.add("decision-model", READY, meta=meta)
    assert _badge(rows) == service_badge.badge("1/2 up", "yellow")


def test_draining_replica_is_not_up(roster) -> None:
    roster(ROSTER)
    rows = _Rows()
    job = rows.add("fast-model", READY, meta=SERVICE_META)
    rows.registry.mark_draining(job)
    rows.add("decision-model", READY, meta=RELAY_META)
    assert _badge(rows)["message"] == "1/2 up"


# ── the route: exact keys, no leak, fixed failure ──────────────────────────

SECRETS = (
    "svc-proj",
    "svc-user",
    "alice",
    "fast-model",
    "decision-model",
    "embed-model",
    "chat-model",
    "node-7",
    "8123",
    "d1g3st",
    "job-",
    "slurm",
    "pool.yaml",
)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(status_badge.router)
    return TestClient(app)


def _live_site(roster, monkeypatch) -> _Rows:
    roster(ROSTER)
    rows = _Rows()
    rows.add("fast-model", READY, meta=SERVICE_META)
    rows.add("decision-model", DEGRADED, meta=RELAY_META)
    rows.add("fast-model", READY, project="alice-proj", meta={"tier": "user", "owner_id": "alice"})
    monkeypatch.setattr(er, "get_registry", lambda: rows.registry)
    return rows


def test_route_answers_exactly_the_badge_keys_and_no_private_string(
    client, roster, monkeypatch
) -> None:
    _live_site(roster, monkeypatch)
    response = client.get("/umrs/status/service-badge")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == BADGE_KEYS
    assert body == service_badge.badge("1/2 up", "yellow")
    text = response.text
    assert not [s for s in SECRETS if s in text]
    assert "cache-control" not in response.headers  # nginx owns caching


def test_route_ignores_identity_headers_and_query_args(client, roster, monkeypatch) -> None:
    _live_site(roster, monkeypatch)
    plain = client.get("/umrs/status/service-badge").json()
    hostile = client.get(
        "/umrs/status/service-badge?project_id=alice-proj&model=fast-model",
        headers={
            "X-User-ID": "alice",
            "X-Verified-Project-ID": "alice-proj",
            "X-Verified-Is-Admin": "true",
            "Authorization": "Bearer x",
        },
    ).json()
    assert hostile == plain


def test_route_takes_no_parameters() -> None:
    assert not inspect.signature(status_badge.service_badge_route).parameters


def test_registry_outage_is_the_fixed_red_badge(client, roster, monkeypatch) -> None:
    roster(ROSTER)

    def boom():
        raise RuntimeError("postgres at db-host:5432 refused user svc-user")

    monkeypatch.setattr(er, "get_registry", boom)
    response = client.get("/umrs/status/service-badge")
    assert response.status_code == 503
    assert response.json() == service_badge.badge("down", "red")
    assert response.headers["cache-control"] == "no-store"
    assert "svc-user" not in response.text and "db-host" not in response.text


def test_roster_surprise_after_the_read_is_unknown(roster, monkeypatch) -> None:
    roster(ROSTER)

    def broken(_rows):
        raise KeyError("svc-proj")

    monkeypatch.setattr(service_badge, "role_states", broken)
    assert _badge(_Rows())["message"] == "unknown"


@pytest.mark.parametrize(
    "states",
    [
        [],
        [service_badge.RoleState(kept_up=True, ready=True, degraded=False)] * 12,
        [service_badge.RoleState(kept_up=False, ready=False, degraded=True)],
        [service_badge.RoleState(kept_up=True, ready=False, degraded=False)] * 3,
    ],
)
def test_every_message_is_integers_and_fixed_words(states) -> None:
    body = service_badge.badge_for(states)
    assert set(body) == BADGE_KEYS
    assert MESSAGE.fullmatch(body["message"])
    assert body["color"] in service_badge.COLORS
    assert json.loads(json.dumps(body)) == body


def test_badge_refuses_a_colour_outside_the_fixed_set() -> None:
    with pytest.raises(ValueError):
        service_badge.badge("1 up", "blue")
