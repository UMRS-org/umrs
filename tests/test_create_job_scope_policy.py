"""Stage-model scope gates on the durable-jobs path (M10 HIGH).

Pre-fix: POST /umrs/jobs with MODEL_DOWNLOAD + spec.scope='system' succeeded
for any caller that could present plain verified headers — bypassing
stage_model's admin/membership rules and auto-poisoning the tenant-wide
model registry at SYSTEM scope (ranked first in provision resolution).
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from r4_contracts.handshake_hmac import sign_claims


def _mod():
    from umrs.api import jobs

    return jobs


class TestScopePolicy:
    def _call(self, job_type, spec, user="alice", project="proj-a", admin=False):
        jobs = _mod()
        return jobs._enforce_job_scope_policy(job_type, spec, user, project, admin)

    def _md_type(self):
        from umrs.api.jobs import JobType

        return JobType.MODEL_DOWNLOAD

    def test_system_scope_requires_admin(self):
        with pytest.raises(HTTPException) as exc:
            self._call(self._md_type(), {"scope": "system"}, admin=False)
        assert exc.value.status_code == 403

    def test_system_scope_with_admin_allowed(self):
        self._call(self._md_type(), {"scope": "system"}, admin=True)

    def test_project_scope_into_other_project_denied(self):
        with pytest.raises(HTTPException) as exc:
            self._call(
                self._md_type(),
                {"scope": "project", "metadata": {"project_id": "proj-b"}},
                admin=False,
            )
        assert exc.value.status_code == 403

    def test_project_scope_matching_allowed(self):
        self._call(
            self._md_type(),
            {"scope": "project", "metadata": {"project_id": "proj-a"}},
            admin=False,
        )

    def test_user_scope_owner_mismatch_denied(self):
        with pytest.raises(HTTPException) as exc:
            self._call(
                self._md_type(),
                {"scope": "user", "metadata": {"owner_id": "victim"}},
                admin=False,
            )
        assert exc.value.status_code == 403

    def test_non_model_download_job_types_not_gated(self):
        from umrs.api.jobs import JobType

        self._call(JobType.PIPELINE_RUN, {"scope": "system"}, admin=False)


class TestPrimitiveResolution:
    def test_strict_rejects_unsigned_headers(self, monkeypatch):
        from fastapi import HTTPException as HE

        jobs = _mod()
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "s")
        with pytest.raises(HE) as exc:
            jobs._resolve_job_create_identity_primitive(
                "attacker", "proj-a", None, "proj-a", None, None
            )
        assert exc.value.status_code == 401

    def test_strict_accepts_signed_headers(self, monkeypatch):
        jobs = _mod()
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "s")
        mac = sign_claims("s", "gateway-user", "proj-a", False)
        project_id, user_id, is_admin = jobs._resolve_job_create_identity_primitive(
            "gateway-user", None, None, "proj-a", None, mac
        )
        assert (project_id, user_id, is_admin) == ("proj-a", "gateway-user", False)

    def test_strict_admin_signed_as_admin(self, monkeypatch):
        jobs = _mod()
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "s")
        mac = sign_claims("s", "gateway-mnagaso", "proj-a", True)
        _p, _u, is_admin = jobs._resolve_job_create_identity_primitive(
            "gateway-mnagaso", None, None, "proj-a", "true", mac
        )
        assert is_admin is True

    def test_strict_admin_claim_without_admin_signature_rejected(self, monkeypatch):
        from fastapi import HTTPException as HE

        jobs = _mod()
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "s")
        mac = sign_claims("s", "alice", "proj-a", False)  # signed non-admin
        with pytest.raises(HE):
            jobs._resolve_job_create_identity_primitive("alice", None, None, "proj-a", "true", mac)

    def test_mvp_body_fallback_preserved(self, monkeypatch):
        jobs = _mod()
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        monkeypatch.delenv("UMRS_HANDSHAKE_HMAC_SECRET", raising=False)
        pid, uid, admin = jobs._resolve_job_create_identity_primitive(
            None, "proj-x", "bob-x", None, None, None
        )
        assert (pid, uid, admin) == ("proj-x", "bob-x", False)
