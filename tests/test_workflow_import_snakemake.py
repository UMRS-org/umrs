"""D-08 pins: Snakemake converter (declarative subset, named refusals)."""

from __future__ import annotations

import pytest
from umrs.workflow_import.snakemake import SnakefileError, convert_snakemake

_SNAKE = """
rule map_reads:
    input: "reads/a.fastq", "reads/b.fastq"
    output: "mapped/a.bam"
    threads: 2
    shell: "minimap2 -t {threads} {input} > {output}"

rule call_vars:
    input: "mapped/a.bam"
    output: "calls/a.vcf"
    params: cutoff = "30"
    shell: "bcftools call -Q {params.cutoff} {input} > {output}"
"""


def test_two_rule_pipeline_maps_with_dag():
    wf = convert_snakemake(_SNAKE, workflow_name="bio")
    assert wf["name"] == "bio"
    steps = {s["name"]: s for s in wf["steps"]}
    assert steps["call_vars"]["depends_on"] == ["map_reads"]
    assert steps["map_reads"]["depends_on"] == []


def test_two_rule_pipeline_commands_and_entrypoint():
    steps = {s["name"]: s for s in convert_snakemake(_SNAKE, workflow_name="bio")["steps"]}
    cmd = steps["map_reads"]["config"]["command"]
    assert "-t 2" in cmd
    assert "$UMRS_INPUTS_DIR/reads/a.fastq" in cmd
    assert "> $UMRS_OUTPUTS_DIR/mapped/a.bam" in cmd
    assert steps["call_vars"]["config"]["command"].startswith("bcftools call -Q 30")
    assert steps["map_reads"]["entrypoint"].endswith("shell_step:ShellStep")


def test_shell_command_payload_is_not_refused_for_include_words():
    """Structure-only refusal scan: a command may legitimately contain 'include'."""
    wf = convert_snakemake(
        "rule copy:\n"
        '    input: "a.txt"\n'
        '    output: "b.txt"\n'
        '    shell: "rsync --include=*.txt {input} {output}"\n'
    )
    assert "--include=*.txt" in wf["steps"][0]["config"]["command"]


@pytest.mark.parametrize(
    "bad,reason",
    [
        ('include "other.snake"', "multi-file includes"),
        ('configfile: "x.yaml"', "configfile"),
        ('rule w:\n    output: "{sample}.bam"\n    shell: "touch {output}"', "wildcard"),
        ('rule t:\n    output: temp("a.bam")\n    shell: "touch {output}"', r"temp\("),
        (
            'rule c:\n    container: "docker://x"\n    output: "a"\n    shell: "touch {output}"',
            "container",
        ),
        ('rule r:\n    run:\n        print("x")\n', "'run:' Python bodies"),
    ],
)
def test_unmappable_constructs_refuse_by_name(bad, reason):
    with pytest.raises(SnakefileError, match=reason):
        convert_snakemake(bad)


def test_rule_without_shell_refuses():
    with pytest.raises(SnakefileError, match="no shell command"):
        convert_snakemake('rule x:\n    input: "a"\n    output: "b"\n')


def test_child_steps_carry_input_aliases():
    """DAG contract: upstream inputs get stable-name aliases."""
    wf = convert_snakemake(_SNAKE)
    steps = {s["name"]: s for s in wf["steps"]}
    assert steps["map_reads"]["config"].get("input_aliases") == {}
    assert steps["call_vars"]["config"]["input_aliases"] == {"mapped/a.bam": "**/mapped/a.bam"}
