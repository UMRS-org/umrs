"""D-08 leg 3 pins: Nextflow converter (deterministic subset, named refusals)."""

from __future__ import annotations

import pytest
from umrs.workflow_import.nextflow import NextflowError, convert_nextflow

_NF = '''
process align {
    input:
    path reads
    output:
    path bam
    script:
    """
    minimap2 ${reads} > ${bam}
    """
}

process call_vcf {
    input:
    path bam
    output:
    path vcf
    script:
    """
    bcftools call ${bam} > ${vcf}
    """
}

workflow {
    reads_ch = Channel.fromPath("reads/a.fastq")
    align(reads_ch)
    call_vcf(align.out)
}
'''


def test_two_process_pipeline_maps_with_dag_and_aliases():
    wf = convert_nextflow(_NF, workflow_name="bio")
    steps = {s["name"]: s for s in wf["steps"]}
    assert steps["call_vcf"]["depends_on"] == ["align"]
    # root channel literal wires the scalar input to a stable alias
    assert steps["align"]["config"]["input_aliases"] == {"reads": "**/reads/a.fastq"}
    # upstream output feeds the child input by name
    assert steps["call_vcf"]["config"]["input_aliases"] == {"bam": "**/bam"}
    assert steps["align"]["config"]["command"] == (
        "minimap2 $UMRS_INPUTS_DIR/reads > $UMRS_OUTPUTS_DIR/bam"
    )


def test_script_interpolations_must_reference_declared_names():
    bad = (
        "process p {\n"
        "    input:\n"
        "    path x\n"
        "    output:\n"
        "    path y\n"
        '    script:\n    """\n    tool ${ghost} ${x} > ${y}\n    """\n}\n'
    )
    with pytest.raises(NextflowError, match="undeclared name 'ghost'"):
        convert_nextflow(bad)


@pytest.mark.parametrize(
    "bad,reason",
    [
        (
            "process p {\n    input:\n    tuple path a, path b\n    output:\n    path y\n}\n",
            "tuple",
        ),
        (
            "process p {\n    input:\n    each path x\n    output:\n    path y\n}\n",
            "each",
        ),
        (
            "process p {\n    conda: 'bio'\n    input:\n    path x\n    output:\n    path y\n}\n",
            "conda",
        ),
        (
            "process p {\n    container: 'docker://x'\n    input:\n    path x\n    output:\n    path y\n}\n",
            "container",
        ),
        (
            "process p {\n    output:\n    path ${meta.id}.txt\n}\n",
            "dynamic-output",
        ),
        (
            "workflow {\n    ch = Channel.fromSRA('x')\n}\n",
            "channel",
        ),
        (
            "process p {\n    publishDir 'out'\n    input:\n    path x\n    output:\n    path y\n}\n",
            "publishDir",
        ),
    ],
)
def test_unmappable_constructs_refuse_by_name(bad, reason):
    with pytest.raises(NextflowError, match=reason):
        convert_nextflow(bad)
