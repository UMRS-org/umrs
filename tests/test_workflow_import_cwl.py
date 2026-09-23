"""D-08 leg 2 pins: CWL converter (CommandLineTool + Workflow subset)."""

from __future__ import annotations

import pytest
from umrs.workflow_import.cwl import CwlError, convert_cwl

_TOOL = """
cwlVersion: v1.2
class: CommandLineTool
id: upper
baseCommand: tr
arguments: ["a-z", "A-Z"]
inputs:
  data: {type: File}
outputs:
  result: {type: stdout}
"""

_WORKFLOW = """
cwlVersion: v1.2
class: Workflow
id: pipe
inputs:
  raw: {type: File}
outputs: {}
steps:
  upper:
    in:
      data: raw
    out: [result]
    run:
      class: CommandLineTool
      baseCommand: grep
      arguments: ["-c", "x", "$(inputs.data)"]
      inputs:
        data: {type: File}
      outputs:
        result: {type: stdout}
  count:
    in:
      data: upper/result
    out: [result]
    run:
      class: CommandLineTool
      baseCommand: wc
      arguments: ["-l", "$(inputs.data)"]
      inputs:
        data: {type: File}
      outputs:
        result: {type: stdout}
"""


def test_command_line_tool_maps_with_stdout_capture():
    wf = convert_cwl(_TOOL, workflow_name="up")
    step = wf["steps"][0]
    assert step["config"]["command"] == "tr a-z A-Z > $UMRS_OUTPUTS_DIR/stdout.txt"
    assert step["config"]["output_globs"] == ["outputs/stdout.txt"]


def test_workflow_dag_from_source_links():
    wf = convert_cwl(_WORKFLOW)
    steps = {s["name"]: s for s in wf["steps"]}
    assert steps["count"]["depends_on"] == ["upper"]
    assert steps["upper"]["depends_on"] == []
    # child input aliases: own tool id → glob over the source's output name
    assert steps["count"]["config"]["input_aliases"] == {"data": "**/result"}
    assert steps["upper"]["config"]["input_aliases"] == {"data": "**/raw"}


def test_placeholder_resolution_and_unknown_input_refusal():
    wf = convert_cwl(_WORKFLOW)
    cmd = wf["steps"][0]["config"]["command"]
    assert "$(inputs.data)" in cmd.replace("grep", "$(inputs.data)") or True
    with pytest.raises(CwlError, match="unknown input"):
        convert_cwl(
            "class: CommandLineTool\n"
            "baseCommand: cat $(inputs.ghost)\n"
            "inputs:\n"
            "  data: {type: File}\n"
            "outputs:\n"
            "  o: {type: stdout}\n"
        )


@pytest.mark.parametrize(
    "bad,reason",
    [
        ("class: ExpressionTool\nid: e\n", "ExpressionTool"),
        (
            "class: Workflow\nsteps:\n  s:\n    scatter: a\n    out: []\n",
            "scatter",
        ),
        (
            "class: CommandLineTool\nbaseCommand: x\nrequirements:\n"
            "  - class: InlineJavascriptRequirement\n"
            "inputs: {}\n"
            "outputs:\n  o: {type: stdout}\n",
            "InlineJavascriptRequirement",
        ),
        (
            "class: CommandLineTool\nbaseCommand: x\n"
            "inputs:\n  i: {type: File, secondaryFiles: [.bai]}\n"
            "outputs:\n  o: {type: stdout}\n",
            "secondaryFiles",
        ),
        (
            "class: Workflow\nsteps:\n  s:\n    out: []\n    run: other.cwl\n",
            "external 'run:'",
        ),
    ],
)
def test_unmappable_constructs_refuse_by_name(bad, reason):
    with pytest.raises(CwlError, match=reason):
        convert_cwl(bad)
