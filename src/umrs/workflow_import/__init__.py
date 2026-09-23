"""Workflow importers (ROADMAP 4.4, D-08).

Converters from external workflow languages into the bundle
``workflow.json`` shape the workflow engine already executes (DAG via
``depends_on``; shell commands run through
``umrs.pipeline_sdk.templates.shell_step:ShellStep``).
Snakemake (declarative subset), CWL (CommandLineTool + Workflow), and
Nextflow (deterministic subset) are live.

Converter honesty contract (the design's never-guess rule): a converter
that cannot map a construct **refuses by name** — it never approximates.
"""

from . import cwl, nextflow, snakemake

__all__ = ["cwl", "nextflow", "snakemake"]
