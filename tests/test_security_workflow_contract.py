"""Structural contract of .github/workflows/security.yml (ADR-117 §7.2, U4).

The repository is public and runs fork code, so the security workflow must use
GitHub-hosted runners only, read no secret, and hold write permission only
where CodeQL uploads its results. Its secret scan stays on the HEAD tree or a
PR's own commits (the SEC-MU7 history fails a full-history scan), and it is
never folded into the one required check, ``ci-success`` (branch protection is
the repository owner's setting, not this file's).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
SECURITY = WORKFLOWS / "security.yml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert SECURITY.is_file(), f"missing {SECURITY.relative_to(ROOT)}"
    return _load(SECURITY)


@pytest.fixture(scope="module")
def text() -> str:
    """The workflow without its comment lines (the header names what it avoids)."""
    lines = SECURITY.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def _steps(job: dict) -> list[dict]:
    return job.get("steps", [])


def _run_text(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in _steps(job))


def test_triggers_are_weekly_pr_and_manual(workflow: dict) -> None:
    on = workflow.get("on", workflow.get(True))  # YAML 1.1 reads `on` as True
    assert {"pull_request", "schedule", "workflow_dispatch"} <= set(on)
    assert on["schedule"], "weekly cron missing"


def test_top_level_permissions_are_read_only(workflow: dict) -> None:
    assert workflow["permissions"] == {"contents": "read"}


def test_every_job_runs_on_a_hosted_runner(workflow: dict) -> None:
    for name, job in workflow["jobs"].items():
        assert job["runs-on"] == "ubuntu-latest", name
        assert "timeout-minutes" in job, name


def test_only_codeql_may_write_and_only_security_events(workflow: dict) -> None:
    for name, job in workflow["jobs"].items():
        perms = job.get("permissions", {})
        writes = {k for k, v in perms.items() if v == "write"}
        expected = {"security-events"} if name == "codeql" else set()
        assert writes == expected, name


def test_no_secret_is_read(text: str) -> None:
    assert "secrets." not in text
    assert "self-hosted" not in text
    assert "pull_request_target" not in text


def test_actions_are_pinned_to_a_release_or_sha(workflow: dict) -> None:
    for job in workflow["jobs"].values():
        for step in _steps(job):
            ref = step.get("uses")
            if ref:
                assert re.search(r"@(v\d+(\.\d+)*|[0-9a-f]{40})$", ref), ref


def test_pip_audit_reads_the_committed_lock(workflow: dict) -> None:
    run = _run_text(workflow["jobs"]["pip-audit"])
    assert re.search(r"uv export --project src --frozen", run)
    assert re.search(r"pip-audit .*--require-hashes", run)
    assert re.search(r"pip install .*pip-audit==\d", run)


def test_secret_scan_is_head_tree_or_pr_commits_only(workflow: dict) -> None:
    run = _run_text(workflow["jobs"]["secret-scan"])
    scans = [line for line in run.splitlines() if re.match(r"\s*gitleaks (dir|git) ", line)]
    assert [line.split()[1] for line in scans] == ["dir", "git"]
    for line in scans:
        assert "--config .gitleaks.toml --redact" in line
    assert '--log-opts="--no-merges ${BASE}..${HEAD}"' in scans[1]


def test_gitleaks_binary_is_pinned_and_checksummed(workflow: dict) -> None:
    env = workflow["jobs"]["secret-scan"]["env"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", env["GITLEAKS_VERSION"])
    assert re.fullmatch(r"[0-9a-f]{64}", env["GITLEAKS_SHA256"])
    assert "sha256sum -c -" in _run_text(workflow["jobs"]["secret-scan"])


def test_gitleaks_config_extends_the_default_rules() -> None:
    # Text checks, not tomllib: umrs still supports Python 3.10.
    config = (ROOT / ".gitleaks.toml").read_text(encoding="utf-8")
    assert re.search(r"^\[extend\]\nuseDefault = true$", config, re.M)
    assert re.search(r'^regexTarget = "line"$', config, re.M)
    entries = re.findall(r"^\s*'''(.+)''',$", config, re.M)
    assert entries, "allowlist regexes missing"
    for regex in entries:
        assert regex.startswith(("^", "(?m)^")) and regex.endswith("$"), regex
        re.compile(regex)


def test_codeql_scans_python_without_a_build(workflow: dict) -> None:
    steps = _steps(workflow["jobs"]["codeql"])
    init = next(s for s in steps if s.get("uses", "").startswith("github/codeql-action/init@"))
    assert init["with"]["languages"] == "python"
    assert init["with"]["build-mode"] == "none"
    assert any(s.get("uses", "").startswith("github/codeql-action/analyze@") for s in steps)


def test_not_a_required_check(workflow: dict) -> None:
    assert "ci-success" not in workflow["jobs"]
    needs = _load(WORKFLOWS / "ci.yml")["jobs"]["ci-success"]["needs"]
    assert not set(needs) & set(workflow["jobs"])
