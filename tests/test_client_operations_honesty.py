"""Pins for the OperationSDK honest-refusal fix (TODO sweep 2026-09-19).

The trio `run_workflow` / `run_experience` / `cancel` previously returned
`ok=True` + blank detail on behalf of implementations that don't exist —
the exact success fabrication the honesty rules forbid. Those calls now
either delegate to an explicit injected executor or raise a named error naming
the missing fabric, never a quiet fake.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from umrs.client.operations import OperationSDK


class TestHonestRefusal:
    def test_run_workflow_without_submitter_refuses_loudly(self):
        sdk = OperationSDK()
        with pytest.raises(RuntimeError, match="run_workflow is not wired"):
            sdk.run_workflow(name="wf-1", bucket="b", params={})

    def test_run_experience_without_submitter_refuses_loudly(self):
        sdk = OperationSDK()
        with pytest.raises(RuntimeError, match="run_experience is not wired"):
            sdk.run_experience(name="exp-1", bucket="b")

    def test_cancel_without_canceller_refuses_loudly(self):
        sdk = OperationSDK()
        with pytest.raises(RuntimeError, match="cancel is not wired"):
            sdk.cancel("op-1")


class TestDelegationPaths:
    def test_run_workflow_delegates_and_wraps_response(self):
        submitter = MagicMock(
            return_value={
                "operation_id": "wf-run-42",
                "detail": "submitted remotely",
                "artifacts": ["s3://out/a.json"],
            }
        )
        sdk = OperationSDK(workflow_submitter=submitter)
        result = sdk.run_workflow(name="wf-1", bucket="b", params={"seed": 4})
        assert result.ok is True
        assert result.operation_id == "wf-run-42"
        assert result.artifacts == ["s3://out/a.json"]
        submitter.assert_called_once_with(name="wf-1", params={"seed": 4})

    def test_run_experience_delegates_overrides(self):
        submitter = MagicMock(return_value={"operation_id": "exp-9"})
        sdk = OperationSDK(experience_submitter=submitter)
        result = sdk.run_experience(name="exp-1", bucket="b", overrides={"x": 1})
        assert result.ok is True
        submitter.assert_called_once_with(name="exp-1", bucket="b", x=1)

    def test_cancel_delegates_to_canceller(self):
        canceller = MagicMock(return_value={"cancelled": "op-1"})
        sdk = OperationSDK(job_canceller=canceller)
        assert sdk.cancel("op-1") is True
        canceller.assert_called_once_with(job_id="op-1")

    def test_backend_error_maps_to_ok_false_not_raise_generic(self):
        submitter = MagicMock(side_effect=RuntimeError("backend exploded"))
        sdk = OperationSDK(workflow_submitter=submitter)
        result = sdk.run_workflow(name="wf-1", bucket="b", params={})
        assert result.ok is False
        assert "backend exploded" in result.detail


# Concrete runner class for the test surface — PluginRunner is abstract by
# design; the harness constructs a minimal subclass.
from umrs.plugins.runner import PluginRunner as _AbstractRunner


class _ConcreteRunner(_AbstractRunner):
    async def _execute_backend(self, *args, **kwargs):  # pragma: no cover
        return None

    async def cancel(self, *args, **kwargs):  # pragma: no cover
        return True

    async def get_status(self, *args, **kwargs):  # pragma: no cover
        return None


# ── Plugin runner source-accessibility pins (TODO sweep sibling leg) ────────

class TestRunnerSourceChecks:
    @pytest.mark.asyncio
    async def test_git_ls_remote_failure_throws_named_valueerror(self, monkeypatch):
        from umrs.plugins.runner import PluginRunner

        class FakeProc:
            returncode = 2

        async def _communicate():
            return (b"", b"fatal: repository not found")

        async def _make_exec(*args, **kwargs):
            class _P:
                returncode = 2

                async def communicate(self):
                    return (b"", b"fatal: repository not found")

            return _P()

        monkeypatch.setattr("asyncio.create_subprocess_exec", _make_exec)

        runner = _ConcreteRunner()
        with pytest.raises(ValueError, match="repository not found|git ls-remote failed"):
            await runner._require_git_ref("https://github.com/org/gone.git", "abc123")

    @pytest.mark.asyncio
    async def test_git_ref_prefix_match_via_ls_remote(self, monkeypatch):
        from umrs.plugins.runner import PluginRunner

        async def _make_exec(*args, **kwargs):
            class _P:
                returncode = 0

                async def communicate(self):
                    return (
                        b"abc12345deadbeef\trefs/heads/main\n"
                        b"99999999deadbeef\trefs/tags/v1.2.0\n",
                        b"",
                    )

            return _P()

        monkeypatch.setattr("asyncio.create_subprocess_exec", _make_exec)

        runner = _ConcreteRunner()
        await runner._require_git_ref("https://github.com/org/repo.git", "abc123")

    @pytest.mark.asyncio
    async def test_git_ref_missing_raises_named(self, monkeypatch):
        from umrs.plugins.runner import PluginRunner

        async def _make_exec(*args, **kwargs):
            class _P:
                returncode = 0

                async def communicate(self):
                    return (b"fffffffdeadbeefbad\trefs/heads/main\n", b"")

            return _P()

        monkeypatch.setattr("asyncio.create_subprocess_exec", _make_exec)

        runner = _ConcreteRunner()
        with pytest.raises(ValueError, match="'gone123' not found"):
            await runner._require_git_ref("https://github.com/org/repo.git", "gone123")

    @pytest.mark.asyncio
    async def test_s3_head_failure_raises_named(self, monkeypatch):
        from umrs.plugins.runner import PluginRunner

        from botocore.exceptions import ClientError

        class _FakeClient:
            def head_object(self, payload):
                raise ClientError({"Error": {"Code": "404"}}, "head_object")

        monkeypatch.setattr(
            "umrs.shared.storage.s3_config.get_s3_config",
            lambda: MagicMock(get_client=lambda: _FakeClient()),
        )

        runner = _ConcreteRunner()
        with pytest.raises(ValueError, match="s3 object not found"):
            await runner._require_s3_object("s3://gone/missing.whl")

    @pytest.mark.asyncio
    async def test_s3_malformed_uri_raises_value(self):
        from umrs.plugins.runner import PluginRunner

        runner = _ConcreteRunner()
        with pytest.raises(ValueError, match="invalid s3 uri"):
            await runner._require_s3_object("s3://")


class TestK2ReadinessCoverage:
    def test_verified_count_only_counts_real_verified_metadata(self):
        from umrs.api.knowledge_v2 import _verified_artifact_count
        from types import SimpleNamespace

        rows = [
            SimpleNamespace(metadata={"status": "verified", "owner": "a"}),
            SimpleNamespace(metadata={"status": "ready"}),
            SimpleNamespace(metadata={"verified": True}),
            SimpleNamespace(metadata={}),
            SimpleNamespace(metadata="not a dict"),
            SimpleNamespace(status="active", metadata={"status": "registered"}),
        ]
        assert _verified_artifact_count(rows) == 2

    def test_verified_count_empty_rows_is_zero(self):
        from umrs.api.knowledge_v2 import _verified_artifact_count

        assert _verified_artifact_count([]) == 0

    def test_verified_count_metadata_alias_metadata_underscore(self):
        from umrs.api.knowledge_v2 import _verified_artifact_count
        from types import SimpleNamespace

        rows = [
            SimpleNamespace(metadata_={"status": "verified"}),
        ]
        assert _verified_artifact_count(rows) == 1
