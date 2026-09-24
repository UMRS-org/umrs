"""ShellStep — run a shell command over materialized inputs (D-08 leg).

The enabling primitive for workflow importers (ROADMAP 4.4): Snakemake /
Nextflow / CWL rules are shell commands over files; this step materializes
S3 inputs into a work directory, runs the command, and uploads the files
the command declares as outputs.

Config (workflow.json ``steps[].config``):

- ``command`` (required): shell command. Run with ``cwd`` = the work dir
  and ``UMRS_INPUTS_DIR`` / ``UMRS_OUTPUTS_DIR`` in the environment.
- ``output_globs`` (required): glob patterns (relative to the work dir)
  naming the files the command produces — only matches upload.
- ``timeout_seconds`` (default 3600).

Trust: a workflow bundle already runs arbitrary Python via its
``entrypoint`` classes — this step adds no new trust boundary; it runs
wherever the workflow engine runs the bundle (sandbox/venue profile).
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import tempfile
from typing import Any

from ..step import Step, StepContext, StepResult

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 3600


def _clean_globs(raw: Any) -> list[str]:
    return [str(g).strip() for g in (raw or []) if str(g).strip()]


def _clean_aliases(raw: Any) -> dict[str, str]:
    return {
        str(k): str(v)
        for k, v in (raw or {}).items()
        if str(k).strip() and str(v).strip()
    }


class ShellStep(Step):
    """Run one shell command; upload its declared outputs."""

    name = "shell"

    def setup(self, config: dict[str, Any]) -> None:
        self.command = str(config.get("command") or "").strip()
        self.output_globs: list[str] = _clean_globs(config.get("output_globs"))
        self.timeout = int(config.get("timeout_seconds") or _DEFAULT_TIMEOUT)
        self.input_aliases: dict[str, str] = _clean_aliases(config.get("input_aliases"))
        if not self.command:
            raise ValueError("ShellStep config requires a non-empty 'command'")
        if not self.output_globs:
            raise ValueError(
                "ShellStep config requires at least one 'output_globs' entry"
            )

    def run(self, ctx: StepContext) -> StepResult:
        out_keys: list[str] = []
        metadata_list: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="umrs-shell-") as workdir:
            inputs_dir = os.path.join(workdir, "inputs")
            outputs_dir = os.path.join(workdir, "outputs")
            os.makedirs(inputs_dir, exist_ok=True)
            os.makedirs(outputs_dir, exist_ok=True)

            # Inputs mirror their S3 key paths under UMRS_INPUTS_DIR —
            # commands reference $UMRS_INPUTS_DIR/<key> (the converter's
            # placeholder expansion relies on this contract).
            for key in ctx.input_keys:
                data = self._download(ctx, key)
                dest = os.path.join(inputs_dir, key)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as handle:
                    handle.write(data)

            self._apply_aliases_step(inputs_dir)

            # Pre-create the declared output directories — a redirected
            # write into outputs/<sub>/... must not fail on a missing dir.
            for pattern in self.output_globs:
                target = os.path.join(workdir, pattern)
                parent = os.path.dirname(target)
                if parent:
                    os.makedirs(parent, exist_ok=True)

            env = dict(os.environ)
            env["UMRS_INPUTS_DIR"] = inputs_dir
            env["UMRS_OUTPUTS_DIR"] = outputs_dir
            proc = subprocess.run(  # noqa: PLW1510 — the command IS the workflow's payload; exit codes handled below
                self.command,
                shell=True,
                cwd=workdir,
                env=env,
                timeout=self.timeout,
                capture_output=True,
                text=True,
            )
            if proc.stdout:
                logger.info("[run=%s] shell stdout: %s", ctx.run_id, proc.stdout[:4000])
            if proc.returncode != 0:
                raise RuntimeError(
                    f"shell command failed (exit {proc.returncode}): "
                    f"{proc.stderr[-2000:]}"
                )

            for pattern in self.output_globs:
                for path in sorted(glob.glob(os.path.join(workdir, pattern))):
                    rel = os.path.relpath(path, workdir)
                    # globs are outputs/-rooted (UMRS_OUTPUTS_DIR layer);
                    # the S3 key drops the redundant layer.
                    rel = rel.removeprefix("outputs/")
                    if not os.path.isfile(path):
                        continue
                    with open(path, "rb") as handle:
                        data = handle.read()
                    key = f"shell/{ctx.run_id}/{rel}"
                    ctx.upload_bytes(key, data)
                    out_keys.append(key)
                    metadata_list.append(
                        {
                            "file": rel,
                            "size_bytes": len(data),
                            "step": "shell",
                        }
                    )
        return StepResult(
            output_keys=out_keys,
            metadata=metadata_list,
            extra={"step": "shell", "command": self.command},
        )

    def _apply_aliases_step(self, inputs_dir: str) -> None:
        """Materialize stable input names (importers' DAG contract).

        Upstream ShellStep keys are run-scoped (``shell/<run>/<name>``);
        a child step's command needs a stable path. Each alias (name →
        glob under inputs_dir) copies the first match to ``<name>`` —
        zero matches refuse by name, never a silent miss.
        """
        import glob as _glob
        import shutil as _shutil

        for alias, pattern in self.input_aliases.items():
            dest = os.path.join(inputs_dir, alias)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            matches = sorted(
                _glob.glob(os.path.join(inputs_dir, pattern), recursive=True)
            )
            if not matches:
                raise RuntimeError(
                    f"input alias {alias!r} matched nothing (pattern {pattern!r})"
                )
            _shutil.copyfile(matches[0], dest)

    @staticmethod
    def _download(ctx: StepContext, key: str) -> bytes:
        """Root steps read the source bucket; children read parent artifacts."""
        if ctx.parent_output_bucket:
            return ctx.download_parent_bytes(key)
        return ctx.download_bytes(key)
