"""Tests for UMRS Doctor module."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def ctx():
    return MagicMock(executable="umrs", profile_name="test")


class TestDoctorModule:
    """Tests for the Doctor diagnostic system."""

    def test_check_all_runs_all_checks(self, ctx):
        from umrs.client.doctor import Doctor

        doctor = Doctor(ctx)
        report = doctor.check_all()
        assert len(report.checks) == 8  # server, auth, storage, models, rag, workflows, jobs, demos
        check_names = [c.name for c in report.checks]
        assert "Server" in check_names
        assert "Authentication" in check_names
        assert "Storage" in check_names
        assert "Models" in check_names
        assert "RAG" in check_names
        assert "Workflows" in check_names
        assert "Jobs" in check_names
        assert "Demos" in check_names

    def test_check_server_reachable(self, ctx):
        from umrs.client.doctor import Doctor

        doctor = Doctor(ctx)
        with patch.object(doctor, "_run") as mock_run:
            import json

            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = json.dumps({"status": "ok"})
            results = doctor._check_server()
            assert len(results) == 1
            assert results[0].status == "ok"

    def test_check_server_unreachable(self, ctx):
        from umrs.client.doctor import Doctor

        doctor = Doctor(ctx)
        with patch.object(doctor, "_run") as mock_run:
            mock_run.return_value.returncode = 1
            results = doctor._check_server()
            assert results[0].status == "error"
            assert "unreachable" in results[0].message

    def test_check_auth_profile_exists(self, ctx):
        from umrs.client.doctor import Doctor

        doctor = Doctor(ctx)
        with patch("os.path.exists", return_value=True):
            results = doctor._check_auth()
            assert results[0].status == "ok"

    def test_check_auth_no_profile(self, ctx):
        from umrs.client.doctor import Doctor

        ctx.profile_name = None
        doctor = Doctor(ctx)
        results = doctor._check_auth()
        assert results[0].status == "warning"
        assert "No profile" in results[0].message

    def test_check_storage_with_buckets(self, ctx):
        from umrs.client.doctor import Doctor

        doctor = Doctor(ctx)
        import json

        data = {"buckets": [{"name": "bucket1"}, {"name": "bucket2"}]}
        with patch.object(doctor, "_run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = json.dumps(data)
            results = doctor._check_storage()
            assert results[0].status == "ok"
            assert "2 buckets" in results[0].message

    def test_check_models(self, ctx):
        from umrs.client.doctor import Doctor

        doctor = Doctor(ctx)
        import json

        models = [
            {"name": "m1", "type": "embedding"},
            {"name": "m2", "type": "text"},
        ]
        with patch.object(doctor, "_run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = json.dumps({"models": models})
            results = doctor._check_models()
            assert results[0].status == "ok"

    def test_check_rag_reads_the_component_registry(self, ctx):
        """`r4 rag ingest` is retired (2026-10-18, H-10.1 follow-up): the
        check reads the live component registry instead of pipelines."""
        from umrs.client.doctor import Doctor

        doctor = Doctor(ctx)
        with patch.object(doctor, "_run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "[]"
            results = doctor._check_rag()
        argv = mock_run.call_args.args
        assert "ingest" not in argv
        assert argv[:4] == ("r4", "rag", "retriever", "components")
        assert results[0].status == "ok"

    def test_check_rag_failure_suggests_no_retired_command(self, ctx):
        from umrs.client.doctor import Doctor

        doctor = Doctor(ctx)
        with patch.object(doctor, "_run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            results = doctor._check_rag()
        assert results[0].status == "warning"
        assert "ingest" not in (results[0].suggestion or "")

    def test_check_workflows(self, ctx):
        from umrs.client.doctor import Doctor

        doctor = Doctor(ctx)
        with patch.object(doctor, "_run") as mock_run:
            mock_run.return_value.returncode = 0
            results = doctor._check_workflows()
            assert results[0].status == "ok"

    def test_check_demos(self, ctx):
        from umrs.client.doctor import Doctor

        doctor = Doctor(ctx)
        with patch.object(doctor, "_run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = '{"buckets": []}'
            # Patch the Path.exists check to avoid demo dir issues in test
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.iterdir", return_value=[]):
                    results = doctor._check_demos()
                    assert len(results) == 1
                    assert (
                        "ready" in results[0].message.lower()
                        or "missing" in results[0].message.lower()
                        or "ok" in results[0].message.lower()
                    )


class TestDoctorReport:
    """Tests for DoctorReport output formatting."""

    def test_report_summary(self):
        from umrs.client.doctor import CheckResult, DoctorReport

        report = DoctorReport()
        report.checks.extend(
            [
                CheckResult(name="A", status="ok", message="OK"),
                CheckResult(name="B", status="warning", message="Warn"),
                CheckResult(name="C", status="error", message="Error"),
            ]
        )
        summary = report.summary()
        assert "OK: 1" in summary
        assert "Warnings: 1" in summary
        assert "Errors: 1" in summary
        assert "✓ A:" in summary
        assert "✗ C:" in summary

    def test_report_ok_property(self):
        from umrs.client.doctor import CheckResult, DoctorReport

        report = DoctorReport()
        report.checks.append(CheckResult(name="A", status="ok", message="OK"))
        assert report.ok is True

    def test_report_not_ok_on_errors(self):
        from umrs.client.doctor import CheckResult, DoctorReport

        report = DoctorReport()
        report.checks.append(CheckResult(name="A", status="error", message="Error"))
        assert report.ok is False

    def test_report_ok_on_warnings(self):
        from umrs.client.doctor import CheckResult, DoctorReport

        report = DoctorReport()
        report.checks.append(CheckResult(name="A", status="warning", message="Warn"))
        assert report.ok is True


class TestDoctorRun:
    """Tests for the run_doctor entry point."""

    def test_run_doctor_all(self, ctx):
        import io

        from umrs.client.doctor import run_doctor

        output = io.StringIO()
        args = MagicMock(doctor_subcommand="all")
        with patch("sys.stdout", output):
            run_doctor(ctx, args)
        result = output.getvalue()
        assert "UMRS Doctor Report" in result

    def test_run_doctor_rag_only(self, ctx):
        import io

        from umrs.client.doctor import run_doctor

        output = io.StringIO()
        args = MagicMock(doctor_subcommand="rag")
        with patch("sys.stdout", output):
            run_doctor(ctx, args)
        result = output.getvalue()
        assert "RAG" in result

    def test_run_doctor_unknown(self, ctx):
        from umrs.client.doctor import run_doctor

        args = MagicMock(doctor_subcommand="invalid")
        result = run_doctor(ctx, args)
        assert result == 1
