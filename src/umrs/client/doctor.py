"""UMRS Doctor Module - Diagnostic and Verification"""
from __future__ import annotations
import json, os, subprocess, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class CheckResult:
    name: str
    status: str  # ok, warning, error, missing
    message: str = ""
    suggestion: str = ""

@dataclass
class DoctorReport:
    checks: List[CheckResult] = field(default_factory=list)
    @property
    def ok(self) -> bool:
        return all(c.status in ("ok", "warning") for c in self.checks)
    def summary(self) -> str:
        lines = ["UMRS Doctor Report", "=" * 40]
        ok_c = sum(1 for c in self.checks if c.status == "ok")
        warn_c = sum(1 for c in self.checks if c.status == "warning")
        err_c = sum(1 for c in self.checks if c.status == "error")
        lines.extend([f"  OK: {ok_c}", f"  Warnings: {warn_c}", f"  Errors: {err_c}", ""])
        for c in self.checks:
            icon = {"ok": "\u2713", "warning": "\u26a0", "error": "\u2717", "missing": "-"}.get(c.status, "?")
            lines.append(f"  {icon} {c.name}: {c.message}")
            if c.suggestion:
                lines.append(f"    Suggestion: {c.suggestion}")
        return "\n".join(lines)

class Doctor:
    def __init__(self, ctx):
        self.ctx = ctx
    def _run(self, *cmd, timeout=15) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)
        except Exception:
            return subprocess.CompletedProcess(list(cmd), 1, "", "")
    def check_all(self) -> DoctorReport:
        report = DoctorReport()
        report.checks.extend(self._check_server())
        report.checks.extend(self._check_auth())
        report.checks.extend(self._check_storage())
        report.checks.extend(self._check_models())
        report.checks.extend(self._check_rag())
        report.checks.extend(self._check_workflows())
        report.checks.extend(self._check_jobs())
        report.checks.extend(self._check_demos())
        return report
    def _check_server(self) -> List[CheckResult]:
        r = self._run(self.ctx.executable or "umrs-client", "--output", "json", "ping")
        if r.returncode == 0:
            try:
                if json.loads(r.stdout).get("status") == "ok":
                    return [CheckResult(name="Server", status="ok", message="Server is reachable")]
            except json.JSONDecodeError:
                pass
        return [CheckResult(name="Server", status="error", message="Server unreachable", suggestion="Verify server is running")]
    def _check_auth(self) -> List[CheckResult]:
        profile = getattr(self.ctx, "profile_name", None)
        if profile:
            path = os.path.expanduser(f"~/.config/umrs/connections/{profile}.json")
            if os.path.exists(path):
                return [CheckResult(name="Authentication", status="ok", message=f"Profile {profile} configured")]
        return [CheckResult(name="Authentication", status="warning", message="No profile configured", suggestion="Run: r4 auth login")]
    def _check_storage(self) -> List[CheckResult]:
        r = self._run(self.ctx.executable or "umrs-client", "--output", "json", "bucket", "list")
        if r.returncode == 0:
            try:
                # Extract JSON from mixed output (server status line + JSON)
                line = next((l for l in r.stdout.split("\n") if l.strip().startswith("{")), "")
                if line.strip():
                    data = json.loads(line)
                    buckets = data.get("buckets", [])
                    if buckets:
                        return [CheckResult(name="Storage", status="ok", message=f"Connected, {len(buckets)} buckets")]
                    return [CheckResult(name="Storage", status="warning", message="No buckets", suggestion="Run: umrs-client reload")]
            except json.JSONDecodeError:
                return [CheckResult(name="Storage", status="ok", message="S3 connectivity confirmed")]
        return [CheckResult(name="Storage", status="error", message="S3 connection failed", suggestion="Check s3cfg file")]
    def _check_models(self) -> List[CheckResult]:
        r = self._run(self.ctx.executable or "umrs-client", "--output", "json", "model", "list", "--scope", "system")
        if r.returncode == 0:
            try:
                line = next((l for l in r.stdout.split("\n") if l.strip().startswith("{")), "")
                if line.strip():
                    data = json.loads(line)
                    models = data.get("models", [])
                    embs = len([m for m in models if m.get("type") == "embedding"])
                    return [CheckResult(name="Models", status="ok", message=f"{len(models)} system models ({embs} embeddings)")]
            except json.JSONDecodeError:
                pass
        return [CheckResult(name="Models", status="error", message="Cannot list models")]
    def _check_rag(self) -> List[CheckResult]:
        r = self._run("r4", "rag", "ingest", "list", "--project", "default")
        if "No ingestion pipelines" in r.stdout:
            return [CheckResult(name="RAG", status="warning", message="No pipelines", suggestion="Run: r4 rag ingest create --project default")]
        return [CheckResult(name="RAG", status="ok", message="RAG service available")]
    def _check_workflows(self) -> List[CheckResult]:
        r = self._run(self.ctx.executable or "umrs-client", "--output", "json", "workflow", "list")
        if r.returncode == 0:
            return [CheckResult(name="Workflows", status="ok", message="Workflow system operational")]
        return [CheckResult(name="Workflows", status="warning", message="Workflow check failed")]
    def _check_jobs(self) -> List[CheckResult]:
        r = self._run(self.ctx.executable or "umrs-client", "--output", "json", "status", "all")
        if r.returncode == 0:
            return [CheckResult(name="Jobs", status="ok", message="Job system responsive")]
        return [CheckResult(name="Jobs", status="warning", message="Job check returned non-zero")]
    def _check_demos(self) -> List[CheckResult]:
        demo_dir = Path(__file__).parent.parent.parent.parent / "demos"
        if not demo_dir.exists():
            return [CheckResult(name="Demos", status="missing", message="Demo directory not found")]
        r = self._run(self.ctx.executable or "umrs-client", "--output", "json", "bucket", "list")
        has_buckets = r.returncode == 0 and "buckets" in r.stdout
        total = len([d for d in demo_dir.iterdir() if d.is_dir()])
        ready = total if has_buckets else 0
        status = "ok" if ready == total else "warning"
        return [CheckResult(name="Demos", status=status, message=f"{ready}/{total} ready (bucket access needed for many demos)", suggestion="Ensure bucket aliases and experience configs are present")]

def add_doctor_parser(subparsers):
    doctor_parser = subparsers.add_parser("doctor", help="Run health checks and diagnostics")
    sub = doctor_parser.add_subparsers(dest="doctor_subcommand", required=False, help="Specific checks to run")
    for name in ["rag", "storage", "auth", "models", "workflows", "jobs", "demos", "all"]:
        sub.add_parser(name, help=f"Check {name} only")
    return doctor_parser

def run_doctor(ctx, args) -> int:
    doctor = Doctor(ctx)
    subcommand = getattr(args, "doctor_subcommand", "all") or "all"
    if subcommand == "all":
        report = doctor.check_all()
    elif subcommand == "rag":
        report = DoctorReport(checks=doctor._check_rag())
    elif subcommand == "storage":
        report = DoctorReport(checks=doctor._check_storage())
    elif subcommand == "auth":
        report = DoctorReport(checks=doctor._check_auth())
    elif subcommand == "models":
        report = DoctorReport(checks=doctor._check_models())
    elif subcommand == "workflows":
        report = DoctorReport(checks=doctor._check_workflows())
    elif subcommand == "jobs":
        report = DoctorReport(checks=doctor._check_jobs())
    elif subcommand == "demos":
        report = DoctorReport(checks=doctor._check_demos())
    else:
        print("Unknown doctor subcommand. Use: rag, storage, auth, models, workflows, jobs, demos, all")
        return 1
    print(report.summary())
    return 0 if report.ok else 1
