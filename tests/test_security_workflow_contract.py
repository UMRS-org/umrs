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


def _gitleaks_tables() -> tuple[str, list[str]]:
    """The config's header and its ``[[allowlists]]`` tables, as text.

    Text, not tomllib: umrs still supports Python 3.10.
    """
    config = (ROOT / ".gitleaks.toml").read_text(encoding="utf-8")
    head, *tables = re.split(r"^\[\[allowlists\]\]\n", config, flags=re.M)
    return head, tables


def _table_regexes(table: str) -> list[str]:
    return re.findall(r"^\s*'''(.+)''',?$", table, re.M)


def test_gitleaks_config_extends_the_default_rules() -> None:
    head, _ = _gitleaks_tables()
    assert re.search(r"^\[extend\]\nuseDefault = true$", head, re.M)


def test_gitleaks_config_uses_only_allowlists_tables() -> None:
    # The pinned 8.30.1 refuses "[allowlist]" next to "[[allowlists]]", and a
    # per-rule (targetRules) entry exists only in the "[[allowlists]]" form.
    head, tables = _gitleaks_tables()
    assert not re.search(r"^\[allowlist\]$", head, re.M)
    assert len(tables) == 2, "one line table and one PEM table"


def test_gitleaks_line_entries_are_single_line_anchored() -> None:
    # 8.30.1 joins a table's regexes into one, so an inline flag such as (?m)
    # applies to all of them; and the "line" of a finding on an unterminated
    # last line spans the file. Together they let any anchored entry match
    # anywhere in that file and hide a real secret (2026-11-24 probe: an
    # sk-proj value under VLLM_TOKENIZER at EOF went unreported).
    _, tables = _gitleaks_tables()
    line_table = tables[0]
    assert re.search(r'^regexTarget = "line"$', line_table, re.M)
    assert "targetRules" not in line_table and "paths" not in line_table
    entries = _table_regexes(line_table)
    assert entries, "allowlist regexes missing"
    for regex in entries:
        assert regex.startswith("^") and regex.endswith("$"), regex
        assert "(?" not in regex, regex
        re.compile(regex)


# Every file the pinned gitleaks 8.30.1 flags on the HEAD tree (2026-11-24),
# apart from the PEM fixture, which the second table covers.
_TRIAGED_FILES = (
    "deploy/bdrt/pool-bdrt-3models.yaml",
    "deploy/octopus/pool-octopus-3models.yaml",
    "deploy/rccs/dsv41/extract.py",
    "tests/test_c2_llm_bad_body.py",
    "tests/test_h10b_followups.py",
    "tests/test_h1_scope_guard.py",
    "tests/test_h1r_skills_grants_identity.py",
    "tests/test_k1_contributions_pg.py",
    "tests/test_k1b_review.py",
    "tests/test_secret_broker_durable.py",
    "tests/test_workflow_plans.py",
)


def test_gitleaks_line_entries_each_cover_a_triaged_line() -> None:
    _, tables = _gitleaks_tables()
    lines = [
        line
        for path in _TRIAGED_FILES
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
    ]
    for regex in _table_regexes(tables[0]):
        assert any(re.search(regex, line) for line in lines), f"stale entry: {regex}"


def test_gitleaks_pem_entry_is_scoped_to_the_intake_fixture() -> None:
    _, tables = _gitleaks_tables()
    pem = tables[1]
    assert re.search(r'^targetRules = \["private-key"\]$', pem, re.M)
    assert re.search(r'^condition = "AND"$', pem, re.M)
    assert re.search(r'^regexTarget = "match"$', pem, re.M)
    assert re.search(r"^paths = \['''\^tests/test_intake_gate\\.py\$'''\]$", pem, re.M)
    (regex,) = _table_regexes(pem)
    assert regex.startswith("\\A") and regex.endswith("\\z"), regex


def _pem_match_regex() -> re.Pattern[str]:
    """The PEM entry in Python syntax (Go's end-of-text \\z is Python's \\Z)."""
    _, tables = _gitleaks_tables()
    (regex,) = _table_regexes(tables[1])
    return re.compile(regex[: -len("\\z")] + "\\Z")


def _intake_fixture_span() -> str:
    """What the private-key rule matches in test_intake_gate.py: the "XYZ"
    header, lazily up to the next "KEY-----" (the "secret.pem" fixture)."""
    source = (ROOT / "tests" / "test_intake_gate.py").read_text(encoding="utf-8")
    start = source.index("-----BEGIN PRIVATE KEY-----\\nXYZ")
    end = source.index("KEY-----", start + 64) + len("KEY-----")
    return source[start:end]


def test_gitleaks_pem_entry_covers_the_fixture_and_nothing_else() -> None:
    pattern = _pem_match_regex()
    span = _intake_fixture_span()
    assert pattern.search(span), "the triaged fixture is no longer allowlisted"
    real = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----"
    cut = span.index("\n", span.index("assert dec.action"))
    # A real key between the two fixtures ends the lazy match at its own header.
    hidden = span[:cut] + "\n" + real.split("\n")[0]
    assert not pattern.search(hidden)
    assert not pattern.search(real)
    assert not pattern.search(span + "\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC")


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
