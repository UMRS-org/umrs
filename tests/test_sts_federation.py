"""WP-11 round 2 — per-attempt STS federation pins (fail-without-change)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api.jobs import get_caller_identity, get_job_store, router
from umrs.jobs.attempt_credentials import mint_attempt_credentials
from umrs.jobs.sts_federation import (
    STSFederationError,
    assume_role_for_attempt,
    attempt_policy_document,
)


def _record(ttl: int = 900):
    return mint_attempt_credentials(
        "job-1", "attempt-1", 1,
        ttl_sec=ttl,
        extra_prefixes=("shared/dataset-x/",),
    )


class TestAttemptPolicyDocument:
    def test_policy_scopes_to_recorded_prefixes_only(self):
        policy = attempt_policy_document(_record(), bucket="umrs-files")
        object_stmts = [s for s in policy["Statement"] if s["Sid"].startswith("AttemptScope") and "Resource" in s and s["Action"] != ["s3:ListBucket"]]
        assert len(object_stmts) == 2
        resources = {r for s in object_stmts for r in s["Resource"]}
        assert "arn:aws:s3:::umrs-files/jobs/job-1/attempt_1/*" in resources
        assert "arn:aws:s3:::umrs-files/shared/dataset-x/*" in resources
        assert all("umrs-files/*" not in r for r in resources)  # no host-wildcard
        list_stmt = [s for s in policy["Statement"] if s["Sid"] == "AttemptScopeList"]
        assert list_stmt[0]["Resource"] == ["arn:aws:s3:::umrs-files"]
        prefixes = list_stmt[0]["Condition"]["StringLike"]["s3:prefix"]
        assert "jobs/job-1/attempt_1/*" in prefixes


class TestAssumeRoleForAttempt:
    def _fake_client(self, creds):
        client = MagicMock()
        client.assume_role.return_value = {"Credentials": creds}
        return client

    def test_mints_and_forwards_identity_without_persisting_policy(self):
        record = _record()
        from datetime import datetime, timezone

        expiry = datetime(2026, 9, 11, tzinfo=timezone.utc)
        client = self._fake_client(
            {
                "AccessKeyId": "AKIA-attempt",
                "SecretAccessKey": "sekrit",
                "SessionToken": "tok",
                "Expiration": expiry,
            }
        )
        out = assume_role_for_attempt(
            record,
            bucket="b",
            endpoint_url="http://minio",
            access_key="a",
            secret_key="s",
            sts_client=client,
        )
        assert out["access_key"] == "AKIA-attempt"
        assert out["session_token"] == "tok"
        kwargs = client.assume_role.call_args.kwargs
        assert kwargs["DurationSeconds"] == 900
        assert "umrs-job-1-1" in kwargs["RoleSessionName"]
        assert '"jobs/job-1/attempt_1/*"' in kwargs["Policy"]

    def test_ttl_capped_at_3600(self):
        client = self._fake_client(
            {"AccessKeyId": "x", "SecretAccessKey": "y", "SessionToken": "z"}
        )
        assume_role_for_attempt(
            _record(ttl=9000), bucket="b", endpoint_url="e",
            access_key="a", secret_key="s", sts_client=client,
        )
        assert client.assume_role.call_args.kwargs["DurationSeconds"] == 3600

    def test_venue_failure_fails_closed_never_ambient(self):
        client = MagicMock()
        client.assume_role.side_effect = RuntimeError("sts down")
        with pytest.raises(STSFederationError):
            assume_role_for_attempt(
                _record(), bucket="b", endpoint_url="e",
                access_key="a", secret_key="s", sts_client=client,
            )

    def test_malformed_response_fails_closed(self):
        client = self._fake_client({"AccessKeyId": "x"})  # missing fields
        with pytest.raises(STSFederationError):
            assume_role_for_attempt(
                _record(), bucket="b", endpoint_url="e",
                access_key="a", secret_key="s", sts_client=client,
            )


class TestSTSRoutes:
    def _app(self, store):
        app = FastAPI()
        app.include_router(router)  # router self-prefixes /umrs/jobs
        app.dependency_overrides[get_job_store] = lambda: store
        app.dependency_overrides[get_caller_identity] = lambda: _OWNER
        return TestClient(app)

    def _store(self, with_credentials=True):
        store = MagicMock()
        job = MagicMock(spec=None)
        job.spec = {}
        job.project_id = "proj-1"
        store.get_job.return_value = job
        attempt = MagicMock()
        attempt.job_id = "job-1"
        attempt.attempt_id = "attempt-1"
        attempt.attempt_no = 1
        store.get_attempt.return_value = attempt
        store.get_attempts_for_job.return_value = [attempt]
        creds = _record().to_dict() if with_credentials else None
        store.get_attempt_credentials.return_value = creds
        return store

    def test_sts_route_federates_and_returns_token_only_to_caller(self):
        store = self._store()
        fake_creds = {
            "access_key": "AK",
            "secret_key": "SK",
            "session_token": "T",
            "expiration": "2026-09-11T00:00:00+00:00",
        }
        with (
            patch("umrs.jobs.sts_federation.import", create=True) as _,
            patch(
                "umrs.api.files.get_s3_config",
                return_value=MagicMock(
                    bucket="umrs-files",
                    endpoint_url="http://minio:9000",
                    access_key="a",
                    secret_key="s",
                ),
            ),
            patch(
                "umrs.jobs.sts_federation.assume_role_for_attempt",
                return_value=fake_creds,
            ),
        ):
            response = self._app(store).post(
                "/umrs/jobs/job-1/attempts/attempt-1/credentials/sts"
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "federated"
        assert body["credentials"]["session_token"] == "T"
        # nothing secret persisted
        assert not any(
            "SK" in str(call) or '"T"' in str(call)
            for call in store.record_attempt_credentials.call_args_list
        )

    def test_sts_route_503_on_venue_failure_no_fallback(self):
        store = self._store()
        with patch(
            "umrs.jobs.sts_federation.assume_role_for_attempt",
            side_effect=STSFederationError("AssumeRole failed: boom"),
        ):
            response = self._app(store).post(
                "/umrs/jobs/job-1/attempts/attempt-1/credentials/sts"
            )
        assert response.status_code == 503
        assert "ambient" not in response.text.lower()

    def test_sts_route_409_without_mint_record(self):
        store = self._store(with_credentials=False)
        response = self._app(store).post(
            "/umrs/jobs/job-1/attempts/attempt-1/credentials/sts"
        )
        assert response.status_code == 409, f"body: {response.text}"


OWNER_CALLER = MagicMock(project_id="proj-1", roles=set())
_OWNER = OWNER_CALLER
