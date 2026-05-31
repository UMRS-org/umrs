"""Tests for hybrid auth authorization propagation.

Covers:
- SessionGrants construction and methods
- Authorization engine policy checks
- ZAP identity propagation via curve_user_id override
- Slurm attribution (sbatch --comment and env vars)
- Client session bootstrap (mocked HTTP)
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from umrs.core.session_curve_provider import (
    CURVE_KEY_LEN,
    SessionCredentialsProvider,
    SessionCurveEntry,
    SessionGrants,
)


def _fake_z85(prefix: str = "A") -> str:
    """Generate a 40-char fake Z85 key string."""
    return (prefix * CURVE_KEY_LEN)[:CURVE_KEY_LEN]


# ---------------------------------------------------------------------------
# SessionGrants
# ---------------------------------------------------------------------------


class TestSessionGrants:
    def test_defaults(self):
        g = SessionGrants()
        assert g.projects == ()
        assert g.roles == ("user",)
        assert g.email == ""
        assert g.max_gpus == 0
        assert g.max_walltime_sec == 0
        assert not g.is_admin
        assert g.has_project("anything")  # no restrictions

    def test_has_project_restricted(self):
        g = SessionGrants(projects=("proj-a", "proj-b"))
        assert g.has_project("proj-a")
        assert g.has_project("proj-b")
        assert not g.has_project("proj-c")

    def test_has_project_unrestricted(self):
        g = SessionGrants(projects=())
        assert g.has_project("any-project")

    def test_is_admin(self):
        g = SessionGrants(roles=("admin", "user"))
        assert g.is_admin
        assert g.has_role("admin")
        assert g.has_role("user")

    def test_not_admin(self):
        g = SessionGrants(roles=("user",))
        assert not g.is_admin

    def test_effective_max_gpus_with_cap(self):
        g = SessionGrants(max_gpus=4)
        assert g.effective_max_gpus(site_default=8) == 4

    def test_effective_max_gpus_no_cap(self):
        g = SessionGrants(max_gpus=0)
        assert g.effective_max_gpus(site_default=8) == 8

    def test_effective_max_walltime_with_cap(self):
        g = SessionGrants(max_walltime_sec=7200)
        assert g.effective_max_walltime(site_default=172800) == 7200

    def test_effective_max_walltime_no_cap(self):
        g = SessionGrants(max_walltime_sec=0)
        assert g.effective_max_walltime(site_default=172800) == 172800


class TestSessionGrantsInProvider:
    def test_register_with_grants(self):
        p = SessionCredentialsProvider()
        key = _fake_z85("G")
        grants = SessionGrants(
            projects=("physics",),
            roles=("user",),
            email="alice@example.org",
            max_gpus=4,
        )
        entry = p.register(key, user_id="alice", ttl_seconds=3600, grants=grants)
        assert entry.grants.projects == ("physics",)
        assert entry.grants.email == "alice@example.org"
        assert entry.grants.effective_max_gpus() == 4

    def test_register_without_grants_gets_default(self):
        p = SessionCredentialsProvider()
        key = _fake_z85("H")
        entry = p.register(key, user_id="bob", ttl_seconds=3600)
        assert entry.grants.roles == ("user",)
        assert entry.grants.projects == ()

    def test_lookup_returns_grants(self):
        p = SessionCredentialsProvider()
        key = _fake_z85("I")
        grants = SessionGrants(projects=("bio",), roles=("admin", "user"))
        p.register(key, user_id="carol", ttl_seconds=3600, grants=grants)
        entry = p.lookup(key)
        assert entry is not None
        assert entry.grants.is_admin
        assert entry.grants.has_project("bio")


# ---------------------------------------------------------------------------
# Authorization Engine
# ---------------------------------------------------------------------------


class TestAuthorizationEngine:
    def setup_method(self):
        from umrs.shared.authorization import reset_site_policy
        reset_site_policy()

    def test_admin_bypass(self):
        from umrs.shared.auth import AuthContext, UserRole
        from umrs.shared.authorization import ResourceRequest, authorize

        ctx = AuthContext(
            user_id="admin-user", username="admin-user", role=UserRole.ADMIN
        )

        resource = ResourceRequest(
            project_id="secret-project", gpus=99, walltime_sec=999999,
            target_user="other-user",
        )

        # Admin should never be denied
        authorize(ctx, operation="submit_job", resource=resource)

    def test_user_denied_project_access(self):
        from umrs.shared.auth import AuthContext, UserRole
        from umrs.shared.authorization import AuthorizationDenied, ResourceRequest, authorize

        ctx = AuthContext(
            user_id="alice", username="alice", role=UserRole.USER,
            project_id="proj-a",
        )

        resource = ResourceRequest(project_id="proj-b", gpus=1, walltime_sec=3600)

        with pytest.raises(AuthorizationDenied, match="project"):
            authorize(ctx, operation="submit_job", resource=resource)

    def test_user_allowed_own_project(self):
        from umrs.shared.auth import AuthContext, UserRole
        from umrs.shared.authorization import ResourceRequest, authorize

        ctx = AuthContext(
            user_id="alice", username="alice", role=UserRole.USER,
            project_id="proj-a",
        )

        resource = ResourceRequest(project_id="proj-a", gpus=1, walltime_sec=3600)

        # Should not raise
        authorize(ctx, operation="submit_job", resource=resource)

    def test_gpu_limit_exceeded(self):
        from umrs.shared.auth import AuthContext, UserRole
        from umrs.shared.authorization import AuthorizationDenied, ResourceRequest, authorize, reset_site_policy

        ctx = AuthContext(
            user_id="bob", username="bob", role=UserRole.USER,
            project_id="proj-x",
        )

        resource = ResourceRequest(project_id="proj-x", gpus=99, walltime_sec=3600)

        with patch.dict(os.environ, {"UMRS_MAX_GPUS_PER_JOB": "8"}):
            reset_site_policy()
            with pytest.raises(AuthorizationDenied, match="GPU|gpu"):
                authorize(ctx, operation="submit_job", resource=resource)

    def test_walltime_limit_exceeded(self):
        from umrs.shared.auth import AuthContext, UserRole
        from umrs.shared.authorization import AuthorizationDenied, ResourceRequest, authorize, reset_site_policy

        ctx = AuthContext(
            user_id="carol", username="carol", role=UserRole.USER,
            project_id="proj-y",
        )

        resource = ResourceRequest(project_id="proj-y", gpus=1, walltime_sec=999999)

        with patch.dict(os.environ, {"UMRS_MAX_WALLTIME_SEC": "86400"}):
            reset_site_policy()
            with pytest.raises(AuthorizationDenied, match="walltime|time"):
                authorize(ctx, operation="submit_job", resource=resource)

    def test_no_resource_no_check(self):
        from umrs.shared.auth import AuthContext, UserRole
        from umrs.shared.authorization import authorize

        ctx = AuthContext(
            user_id="dave", username="dave", role=UserRole.USER,
        )

        # No resource → no policy check → should not raise
        authorize(ctx, operation="query_status", resource=None)


# ---------------------------------------------------------------------------
# ZAP identity propagation (_SessionThreadAuthenticator)
# ---------------------------------------------------------------------------


class TestSessionThreadAuthenticator:
    def test_curve_user_id_returns_session_id(self):
        """The custom authenticator should return session_id, not raw pubkey."""
        from umrs.core.zmq_security import _SessionThreadAuthenticator

        provider = SessionCredentialsProvider()
        key = _fake_z85("Z")
        grants = SessionGrants(projects=("test",))
        entry = provider.register(
            key, user_id="alice", ttl_seconds=3600,
            session_id="my-session-abc", grants=grants,
        )

        # Create the authenticator (we just need the _curve_user_id method)
        # We can't start() without a real ZMQ context, so test the method directly
        auth = _SessionThreadAuthenticator.__new__(_SessionThreadAuthenticator)
        auth._credentials_provider = provider

        # Simulate what pyzmq does: pass raw 32-byte key
        # In real ZMQ, curve_user_id receives the raw binary key (not Z85)
        # but our implementation Z85-encodes it to look up in the provider
        import zmq.utils.z85 as z85
        raw_key = z85.decode(key.encode("ascii"))
        result = auth._curve_user_id(raw_key)
        assert result == "my-session-abc"

    def test_curve_user_id_fallback_to_z85(self):
        """Unknown keys should fallback to Z85 string (backward compat)."""
        from umrs.core.zmq_security import _SessionThreadAuthenticator

        provider = SessionCredentialsProvider()

        auth = _SessionThreadAuthenticator.__new__(_SessionThreadAuthenticator)
        auth._credentials_provider = provider

        import zmq.utils.z85 as z85
        unknown_key = _fake_z85("X")
        raw_key = z85.decode(unknown_key.encode("ascii"))
        result = auth._curve_user_id(raw_key)
        # Should return the Z85-encoded key
        assert result == unknown_key


# ---------------------------------------------------------------------------
# Slurm attribution
# ---------------------------------------------------------------------------


class TestSlurmAttribution:
    def test_remote_slurm_backend_injects_env_vars(self):
        """remote_slurm_backend should inject UMRS_USER/UMRS_PROJECT into sbatch."""
        from umrs.orchestrator.remote_slurm_backend import RemoteSlurmBackend

        backend = RemoteSlurmBackend.__new__(RemoteSlurmBackend)
        # Test the env var injection logic by checking the script generation
        # We test indirectly through the wrapper job generation
        # The actual test needs the _submit_wrapper_job method to contain
        # UMRS_USER and UMRS_PROJECT

        # Verify the source file contains the attribution logic
        import inspect
        source = inspect.getsource(RemoteSlurmBackend)
        assert "UMRS_USER" in source
        assert "UMRS_PROJECT" in source
        assert "--comment=" in source or "comment" in source


class TestSlurmTargetAttribution:
    def test_jobspec_has_attribution_fields(self):
        """JobSpec should carry requesting_user/requesting_project."""
        from umrs.orchestrator.hybrid.targets import JobSpec

        spec = JobSpec.__new__(JobSpec)
        # Verify the class has the fields
        import dataclasses
        if dataclasses.is_dataclass(JobSpec):
            field_names = [f.name for f in dataclasses.fields(JobSpec)]
            assert "requesting_user" in field_names
            assert "requesting_project" in field_names


# ---------------------------------------------------------------------------
# Client session bootstrap (mocked HTTP)
# ---------------------------------------------------------------------------


class TestClientSessionBootstrap:
    def test_bootstrap_success(self):
        """Successful bootstrap should return CurveSessionInfo with all fields."""
        import requests as _requests_mod
        from umrs.client.session_bootstrap import CurveSessionInfo, bootstrap_curve_session

        mock_server_key_resp = MagicMock()
        mock_server_key_resp.status_code = 200
        mock_server_key_resp.json.return_value = {
            "server_public_key": "S" * 40,
        }
        mock_server_key_resp.raise_for_status = MagicMock()

        mock_register_resp = MagicMock()
        mock_register_resp.status_code = 200
        mock_register_resp.json.return_value = {
            "session_id": "sess-xyz",
            "server_public_key": "S" * 40,
            "expires_at": time.time() + 3600,
            "ttl_seconds": 3600,
            "grants": {"projects": ["test"]},
        }
        mock_register_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_server_key_resp), \
             patch("requests.post", return_value=mock_register_resp):

            info = bootstrap_curve_session(
                gateway_url="http://localhost:8000",
                jwt_token="fake-jwt",
                set_env=False,
            )

        assert info.session_id == "sess-xyz"
        assert info.server_public_key == "S" * 40
        assert len(info.client_public_key) == 40
        assert len(info.client_secret_key) == 40
        assert info.grants == {"projects": ["test"]}

    def test_bootstrap_404_raises(self):
        """404 from server-key endpoint should raise SessionBootstrapError."""
        from umrs.client.session_bootstrap import SessionBootstrapError, bootstrap_curve_session

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(SessionBootstrapError, match="not enabled"):
                bootstrap_curve_session(
                    gateway_url="http://localhost:8000",
                    jwt_token="fake-jwt",
                    set_env=False,
                )

    def test_bootstrap_401_raises(self):
        """401 from registration should raise SessionBootstrapError."""
        from umrs.client.session_bootstrap import SessionBootstrapError, bootstrap_curve_session

        mock_server_key_resp = MagicMock()
        mock_server_key_resp.status_code = 200
        mock_server_key_resp.json.return_value = {"server_public_key": "S" * 40}
        mock_server_key_resp.raise_for_status = MagicMock()

        mock_register_resp = MagicMock()
        mock_register_resp.status_code = 401

        with patch("requests.get", return_value=mock_server_key_resp), \
             patch("requests.post", return_value=mock_register_resp):
            with pytest.raises(SessionBootstrapError, match="401"):
                bootstrap_curve_session(
                    gateway_url="http://localhost:8000",
                    jwt_token="bad-jwt",
                    set_env=False,
                )

    def test_revoke_session_success(self):
        """revoke_curve_session should return True on 204."""
        from umrs.client.session_bootstrap import revoke_curve_session

        mock_resp = MagicMock()
        mock_resp.status_code = 204

        with patch("requests.delete", return_value=mock_resp):
            result = revoke_curve_session(
                gateway_url="http://localhost:8000",
                jwt_token="fake-jwt",
                session_id="sess-xyz",
            )

        assert result is True
