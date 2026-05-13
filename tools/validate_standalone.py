#!/usr/bin/env python3
"""Standalone UMRS validation script.

Runs the minimal validation suite required for UMRS-only CI without
requiring R4, Gateway, or other upper-layer components.

Usage:
    python umrs/tools/validate_standalone.py [--full]

Exit codes:
    0  All checks passed
    1  One or more checks failed
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UMRS_ROOT = REPO_ROOT / "umrs"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

# Use venv python if available, else system python
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def _run(cmd: list[str], label: str) -> bool:
    """Run a command and return True on success."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode == 0:
        print(f"  ✓ {label} PASSED")
    else:
        print(f"  ✗ {label} FAILED (exit {result.returncode})")
    return result.returncode == 0


def main() -> int:
    full_mode = "--full" in sys.argv
    results: list[tuple[str, bool]] = []

    # --- Boundary validators (always run) ---
    results.append((
        "Import boundary validator",
        _run([PYTHON, "tools/import_boundary_validator.py"], "Import boundary validator"),
    ))
    results.append((
        "Contracts fence",
        _run([PYTHON, "tools/contracts_fence.py"], "Contracts fence"),
    ))
    results.append((
        "Schema version tracker",
        _run([PYTHON, "tools/schema_version_tracker.py"], "Schema version tracker"),
    ))
    results.append((
        "Stale name validator",
        _run([PYTHON, "tools/stale_name_validator.py"], "Stale name validator"),
    ))

    # --- Slurm script syntax check ---
    slurm_script = UMRS_ROOT / "src" / "umrs" / "mini-rivault-llm" / "serve-model.slurm"
    if slurm_script.exists():
        results.append((
            "Slurm script syntax",
            _run(["bash", "-n", str(slurm_script)], "Slurm script syntax (bash -n)"),
        ))

    # --- UMRS domain tests ---
    domain_tests = [
        "umrs/tests/test_orchestrator.py",
        "umrs/tests/test_site_config.py",
        "umrs/tests/test_site_profiles.py",
        "umrs/tests/test_slurm_script_profiles.py",
        "umrs/tests/test_mini_rivault_definition.py",
        "umrs/tests/test_endpoints_provision_api.py",
        "umrs/tests/test_mini_rivault_egress.py",
        "umrs/tests/test_scheduler.py",
        "umrs/tests/test_dag_scheduler.py",
    ]
    existing_tests = [t for t in domain_tests if (REPO_ROOT / t).exists()]
    if existing_tests:
        results.append((
            "UMRS domain tests",
            _run([PYTHON, "-m", "pytest"] + existing_tests + ["-q"], "UMRS domain tests"),
        ))

    # --- Full mode: additional enforcement tests ---
    if full_mode:
        enforcement_tests = [
            "tests/test_import_boundary_validator.py",
            "tests/test_contracts_fence.py",
            "tests/test_dto_equivalence.py",
            "tests/test_schema_version_tracker.py",
            "tests/test_stale_name_validator.py",
            "tests/test_exemption_governance.py",
            "tests/test_orphan_module_detector.py",
            "tests/test_drift_detectors.py",
        ]
        existing_enforcement = [t for t in enforcement_tests if (REPO_ROOT / t).exists()]
        if existing_enforcement:
            results.append((
                "Enforcement tests",
                _run(
                    [PYTHON, "-m", "pytest"] + existing_enforcement + ["-q"],
                    "Enforcement tests (61 tests)",
                ),
            ))

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    for label, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {label}")
    print(f"\n  Total: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
