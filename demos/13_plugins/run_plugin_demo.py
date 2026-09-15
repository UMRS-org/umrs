#!/usr/bin/env python3
"""
Demo 13 — Plugin System Demo
==============================

Demonstrates the UMRS plugin system workflow:
  1. Define a PluginSpec (what to run, resource limits, outputs)
  2. Validate the spec
  3. Execute locally (simulates Slurm/K8s sandbox)
  4. Collect results
  5. Write audit log

Uses local execution — no Slurm or K8s required.
For HPC: use SlurmPluginRunner (Singularity containers)
For cloud: use K8sPluginRunner (Kubernetes Jobs)

Usage:
    python3 run_plugin_demo.py [--input-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Plugin SDK imports
from umrs.plugins import (
    PluginAuditEntry,
    PluginAuditLog,
    PluginResult,
    PluginSourceType,
    PluginSpec,
    PluginStatus,
)


def create_sample_spec(plugin_script: str) -> PluginSpec:
    """Create a PluginSpec for the word-count plugin."""
    return PluginSpec(
        source_type=PluginSourceType.LOCAL_PATH,
        source_uri=plugin_script,
        entrypoint="word_count_plugin:main",
        requirements=[],  # No extra pip packages
        env_vars={},
        cpu_limit="1",
        memory_limit="512Mi",
        gpu_count=0,
        timeout_seconds=60,
        output_destinations=["umrs://plugin-outputs/"],
        project_id="demo-project",
        user_id="demo-user",
        metadata={"description": "Word frequency analysis plugin"},
    )


def validate_spec(spec: PluginSpec) -> bool:
    """Run the built-in spec validation."""
    try:
        spec._validate()
        print("[OK] Spec validation passed")
        print(f"     Spec hash: {spec.spec_hash()}")
        return True
    except ValueError as e:
        print(f"[FAIL] Spec validation failed: {e}")
        return False


def execute_locally(spec: PluginSpec, input_dir: str, work_dir: str) -> PluginResult:
    """Execute the plugin locally, simulating what SlurmPluginRunner does.

    In production, SlurmPluginRunner would:
      1. Generate an sbatch script
      2. Submit it to Slurm (sbatch --parsable)
      3. Run inside a Singularity container with --containall
      4. Bind-mount /workspace with input/output dirs
      5. Collect results from /workspace/output/
    """
    plugin_id = f"plugin-demo-{os.getpid()}"
    started_at = datetime.now(timezone.utc)

    # Set up workspace (simulates Singularity bind mounts)
    workspace = Path(work_dir) / "workspace"
    input_dest = workspace / "input"
    output_dest = workspace / "output"
    input_dest.mkdir(parents=True, exist_ok=True)
    output_dest.mkdir(parents=True, exist_ok=True)

    # Copy input files to workspace
    for f in Path(input_dir).glob("*.txt"):
        shutil.copy2(f, input_dest / f.name)

    # Execute the plugin script
    env = os.environ.copy()
    env["PLUGIN_WORKSPACE"] = str(workspace)
    env["PLUGIN_ENTRYPOINT"] = spec.entrypoint
    env["PLUGIN_SOURCE_TYPE"] = spec.source_type.value
    env["PLUGIN_SOURCE_URI"] = spec.source_uri
    env["PLUGIN_PROJECT_ID"] = spec.project_id or ""
    env["PLUGIN_USER_ID"] = spec.user_id or ""

    result = subprocess.run(
        [sys.executable, spec.source_uri],
        capture_output=True,
        text=True,
        env=env,
        timeout=spec.timeout_seconds,
        cwd=work_dir,
    )

    ended_at = datetime.now(timezone.utc)

    # Collect output files
    output_files = [str(p.relative_to(workspace)) for p in output_dest.rglob("*") if p.is_file()]

    # Read metrics if available
    metrics = {}
    metrics_file = workspace / "metrics.txt"
    if metrics_file.exists():
        for line in metrics_file.read_text().strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                metrics[k.strip()] = v.strip()

    status = PluginStatus.SUCCEEDED if result.returncode == 0 else PluginStatus.FAILED

    return PluginResult(
        plugin_id=plugin_id,
        status=status,
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr or None,
        output_files=output_files,
        metrics=metrics,
        started_at=started_at,
        ended_at=ended_at,
        error_message=result.stderr if result.returncode != 0 else None,
    )


def print_result(result: PluginResult) -> None:
    """Print a human-readable result summary."""
    print(f"\n{'─' * 60}")
    print(f"Plugin ID:     {result.plugin_id}")
    print(f"Status:        {result.status.value}")
    print(f"Exit code:     {result.exit_code}")
    if result.duration_seconds is not None:
        print(f"Duration:      {result.duration_seconds:.2f}s")
    print(f"Output files:  {result.output_files}")
    print(f"Metrics:       {result.metrics}")
    print(f"{'─' * 60}")

    if result.stdout:
        print("\n── Plugin stdout ──")
        print(result.stdout.rstrip())

    if result.stderr:
        print("\n── Plugin stderr ──")
        print(result.stderr.rstrip())


def print_audit_trail(spec: PluginSpec, result: PluginResult) -> None:
    """Show what the audit log would contain."""
    events = [
        ("submitted", {"spec_hash": spec.spec_hash(), "source_type": spec.source_type.value}),
        ("started", {"workspace": "/workspace"}),
    ]
    if result.output_files:
        events.append(
            (
                "output_written",
                {
                    "files": result.output_files,
                    "destinations": spec.output_destinations,
                },
            )
        )
    final = "succeeded" if result.status == PluginStatus.SUCCEEDED else "failed"
    events.append(
        (
            final,
            {
                "exit_code": result.exit_code,
                "metrics": result.metrics,
            },
        )
    )

    print("\n── Audit Trail ──")
    for event_type, details in events:
        entry = PluginAuditEntry(
            plugin_id=result.plugin_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            user_id=spec.user_id,
            project_id=spec.project_id,
            details=details,
        )
        print(
            f"  [{entry.event_type:>16}]  plugin={entry.plugin_id}  "
            f"user={entry.user_id}  project={entry.project_id}"
        )
        for k, v in entry.details.items():
            print(f"                      {k}: {v}")


def show_sbatch_preview(spec: PluginSpec) -> None:
    """Show what the generated sbatch script would look like for Slurm."""
    print("\n── Slurm sbatch preview (what SlurmPluginRunner generates) ──")
    print(f"""#!/bin/bash
#SBATCH --job-name=plugin-{spec.spec_hash()[:8]}
#SBATCH --partition=gpu
#SBATCH --time=00:{spec.timeout_seconds // 60:02d}:00
#SBATCH --cpus-per-task={spec.cpu_limit}
#SBATCH --mem={spec.memory_limit}
{"#SBATCH --gres=gpu:" + str(spec.gpu_count) if spec.gpu_count > 0 else "# (no GPU requested)"}
#SBATCH --output=/workspace/slurm-%j.out
#SBATCH --error=/workspace/slurm-%j.err

export PLUGIN_ENTRYPOINT="{spec.entrypoint}"
export PLUGIN_SOURCE_TYPE="{spec.source_type.value}"
export PLUGIN_SOURCE_URI="{spec.source_uri}"

singularity exec --containall \\
    --bind /workspace:/workspace \\
    --bind /etc/minio:/etc/minio:ro \\
    ghcr.io/rivault/plugin-runner:latest \\
    python -m plugin_runner.entrypoint""")


def main():
    parser = argparse.ArgumentParser(
        description="Plugin System Demo — local execution workflow",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing .txt files to process (default: ../../data/file_of_texts/)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_dir = args.input_dir or str(script_dir / "../../data/file_of_texts")
    plugin_script = str(script_dir / "word_count_plugin.py")

    if not Path(input_dir).is_dir():
        print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Phase 1: Define the plugin
    print("=" * 60)
    print("PHASE 1: Define PluginSpec")
    print("=" * 60)
    spec = create_sample_spec(plugin_script)
    print(f"  Source type:     {spec.source_type.value}")
    print(f"  Source URI:      {spec.source_uri}")
    print(f"  Entrypoint:     {spec.entrypoint}")
    print(f"  CPU limit:      {spec.cpu_limit}")
    print(f"  Memory limit:   {spec.memory_limit}")
    print(f"  GPU count:      {spec.gpu_count}")
    print(f"  Timeout:        {spec.timeout_seconds}s")
    print(f"  Output dest:    {spec.output_destinations}")

    # Phase 2: Validate
    print(f"\n{'=' * 60}")
    print("PHASE 2: Validate PluginSpec")
    print("=" * 60)
    if not validate_spec(spec):
        sys.exit(1)

    # Phase 3: Show what Slurm would generate
    print(f"\n{'=' * 60}")
    print("PHASE 3: Preview Slurm submission")
    print("=" * 60)
    show_sbatch_preview(spec)

    # Phase 4: Execute locally
    print(f"\n\n{'=' * 60}")
    print("PHASE 4: Execute plugin (local mode)")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="umrs-plugin-demo-") as work_dir:
        result = execute_locally(spec, input_dir, work_dir)
        print_result(result)

        # Show output file contents
        workspace = Path(work_dir) / "workspace"
        output_json = workspace / "output" / "word_frequencies.json"
        if output_json.exists():
            print("\n── Output: word_frequencies.json ──")
            data = json.loads(output_json.read_text())
            print(json.dumps(data, indent=2, default=str))

    # Phase 5: Audit trail
    print(f"\n{'=' * 60}")
    print("PHASE 5: Audit Trail")
    print("=" * 60)
    print_audit_trail(spec, result)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  Plugin status: {result.status.value}")
    print(f"  Files processed: {result.metrics.get('files_processed', '?')}")
    print(f"  Total words: {result.metrics.get('total_words', '?')}")
    print()
    print("For HPC deployment, use SlurmPluginRunner:")
    print("  from umrs.plugins import SlurmPluginRunner")
    print("  runner = SlurmPluginRunner(partition='gpu')")
    print("  result = await runner.execute(spec)")

    sys.exit(0 if result.status == PluginStatus.SUCCEEDED else 1)


if __name__ == "__main__":
    main()
