"""CWL → UMRS workflow converter (D-08 leg 2, declarative subset).

Maps CWL ``CommandLineTool`` and ``Workflow`` documents (YAML) into a
bundle ``workflow.json`` — the same ShellStep target the Snakemake
converter emits.

Mappable subset:

- ``CommandLineTool``: ``baseCommand``, ``arguments`` (literal strings),
  ``inputs`` (File/string params referenced as ``$(inputs.<id>)``),
  one ``stdout`` capture; ``inputs.default`` allowed (literal values).
- ``Workflow``: ``steps[].in`` (source: ``<step>/<out>`` or ``<input>``),
  ``steps[].out``, ``run:`` embedded CommandLineTools only; explicit
  source/sink DAG — CWL's links are already concrete (no channel magic).

Anything else **refuses by name** (never approximated): ``ExpressionTool``,
``scatter``, ``$(...)`` in outputs (JS expressions), ``requirements`` other
than the tool-shape ones, ``secondaryFiles``, ``glob`` outputs, multiple
``stdout`` captures, external ``run:`` references, missing fields.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

SHELL_STEP_ENTRYPOINT = "umrs.pipeline_sdk.templates.shell_step:ShellStep"

_ALLOWED_REQUIREMENTS: set[str] = set()  # none carry semantics we can map
_PLACEHOLDER_RE = re.compile(r"\$\(inputs\.([A-Za-z0-9_]+)\)")


class CwlError(ValueError):
    """Named refusal: the construct is outside the mappable subset."""


def convert_cwl(source: str, *, workflow_name: str = "imported-cwl") -> dict[str, Any]:
    """Convert a CWL document (declarative subset) to bundle workflow.json."""
    try:
        doc = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise CwlError(f"unparseable YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise CwlError("CWL document must be a mapping")
    cwl_class = str(doc.get("class") or "")
    if cwl_class == "CommandLineTool":
        return _tool_to_workflow(doc, workflow_name)
    if cwl_class == "Workflow":
        return _workflow_doc(doc, workflow_name)
    raise CwlError(
        f"class {cwl_class!r} is not mappable (CommandLineTool/Workflow only)"
    )


def _tool_to_workflow(tool: dict[str, Any], workflow_name: str) -> dict[str, Any]:
    """One CommandLineTool → one ShellStep workflow."""
    tool = _check_tool(tool, context="tool")
    step = _tool_step(tool, name=str(tool.get("id") or "tool"), deps=[])
    return _wrap(
        workflow_name, [step], description="Imported from a CWL CommandLineTool"
    )


def _check_tool(tool: dict[str, Any], *, context: str) -> dict[str, Any]:
    """Refuse everything outside the mappable tool subset."""
    if tool.get("class") != "CommandLineTool":
        raise CwlError(f"{context}: only embedded CommandLineTools are mappable")
    for req in tool.get("requirements") or []:
        name = req.get("class") if isinstance(req, dict) else req
        if name not in _ALLOWED_REQUIREMENTS:
            raise CwlError(f"{context}: requirement {name!r} is not mappable")
    if tool.get("hints"):
        raise CwlError(f"{context}: hints are not mappable (carry runtime semantics)")
    if not tool.get("baseCommand"):
        raise CwlError(f"{context}: baseCommand is required")
    _check_tool_outputs(tool, context)
    _check_tool_inputs(tool, context)
    return tool


def _check_tool_outputs(tool: dict[str, Any], context: str) -> None:
    outputs = tool.get("outputs") or {}
    stdouts = [
        out
        for out in outputs.values()
        if isinstance(out, dict) and out.get("type") == "stdout"
    ]
    if len(stdouts) > 1:
        raise CwlError(f"{context}: multiple stdout captures are not mappable")
    for oid, out in outputs.items():
        _check_one_output(oid, out, context)


def _check_one_output(oid: str, out: Any, context: str) -> None:
    if isinstance(out, dict) and out.get("glob"):
        raise CwlError(f"{context}: glob output {oid!r} is not mappable")
    if isinstance(out, dict) and "$(" in str(out.get("outputBinding") or ""):
        raise CwlError(f"{context}: JS expression in output {oid!r} is not mappable")


def _check_tool_inputs(tool: dict[str, Any], context: str) -> None:
    for iid, inp in (tool.get("inputs") or {}).items():
        if isinstance(inp, dict) and inp.get("secondaryFiles"):
            raise CwlError(
                f"{context}: secondaryFiles on input {iid!r} are not mappable"
            )


def _tool_step(tool: dict[str, Any], *, name: str, deps: list[str]) -> dict[str, Any]:
    """Build the ShellStep config for one validated CommandLineTool."""
    base = tool["baseCommand"]
    if isinstance(base, list):
        command = " ".join(str(part) for part in base)
    else:
        command = str(base)
    for arg in tool.get("arguments") or []:
        command += " " + _literal_argument(arg)
    if "$(" in command and not _PLACEHOLDER_RE.search(command):
        raise CwlError(
            f"{name!r}: command expression is not a plain $(inputs.X) reference"
        )

    command = _PLACEHOLDER_RE.sub(
        lambda m: _input_anchor(tool, m.group(1), name), command
    )
    stdout_out = _stdout_capture(tool)
    output_globs = []
    if stdout_out:
        command += f" > $UMRS_OUTPUTS_DIR/{stdout_out}"
        output_globs.append(f"outputs/{stdout_out}")
    if not output_globs:
        raise CwlError(
            f"{name!r}: no stdout capture — a step must declare at least one output"
        )
    return {
        "name": name,
        "entrypoint": SHELL_STEP_ENTRYPOINT,
        "depends_on": deps,
        "config": {"command": command, "output_globs": output_globs},
    }


def _literal_argument(arg: Any) -> str:
    """CWL argument: literal string (or {valueFrom: literal}); refuse the rest."""
    if isinstance(arg, str):
        if "$(" in arg:
            return _PLACEHOLDER_RE.sub(lambda m: f"$UMRS_INPUTS_DIR/{m.group(1)}", arg)
        return arg
    if isinstance(arg, dict) and "valueFrom" in arg:
        return _literal_argument(arg["valueFrom"])
    raise CwlError("non-literal argument values are not mappable")


def _input_anchor(tool: dict[str, Any], input_id: str, name: str) -> str:
    """$(inputs.<id>) → $UMRS_INPUTS_DIR/<id> (File inputs materialize there)."""
    if input_id not in (tool.get("inputs") or {}):
        raise CwlError(f"{name!r}: command references unknown input {input_id!r}")
    return f"$UMRS_INPUTS_DIR/{input_id}"


def _stdout_capture(tool: dict[str, Any]) -> str:
    for out in (tool.get("outputs") or {}).values():
        if isinstance(out, dict) and out.get("type") == "stdout":
            return str(
                out.get("id")
                or out.get("outputBinding", {}).get("glob")
                or "stdout.txt"
            )
    return ""


def _workflow_doc(doc: dict[str, Any], workflow_name: str) -> dict[str, Any]:
    """CWL Workflow → steps with the explicit source/sink DAG."""
    steps_src = doc.get("steps") or {}
    if not isinstance(steps_src, dict) or not steps_src:
        raise CwlError("workflow has no steps")
    if doc.get("scatter") or any(
        isinstance(s, dict) and s.get("scatter") for s in steps_src.values()
    ):
        raise CwlError("scatter is not mappable")

    tool_steps: list[dict[str, Any]] = []
    for step_id, step in steps_src.items():
        tool_steps.append(_workflow_step(step_id, step))
    return _wrap(
        workflow_name,
        tool_steps,
        description=f"Imported from a CWL Workflow; {len(tool_steps)} steps",
    )


def _workflow_step(step_id: str, step: Any) -> dict[str, Any]:
    """One CWL step → ShellStep config with its in-map aliases + deps."""
    if not isinstance(step, dict):
        raise CwlError(f"step {step_id!r} is not a mapping")
    run = step.get("run")
    if not isinstance(run, dict):
        raise CwlError(
            f"step {step_id!r}: external 'run:' references are not mappable — embed the CommandLineTool"
        )
    tool = _check_tool(run, context=f"step {step_id!r}")
    in_map = step.get("in") or {}
    deps = sorted(
        {str(src).split("/", 1)[0] for src in _sources(in_map) if "/" in str(src)}
    )
    step_obj = _tool_step(tool, name=str(step_id), deps=deps)
    step_obj["config"]["input_aliases"] = _input_aliases(in_map)
    return step_obj


def _sources(in_map: Any) -> list[str]:
    """Collect 'source' links (str shorthand, {source: x}, or arrays)."""
    out: list[str] = []
    if isinstance(in_map, dict):
        for binding in in_map.values():
            if isinstance(binding, str):
                out.append(binding)
            elif isinstance(binding, dict):
                src = binding.get("source")
                out.extend([str(src)] if isinstance(src, str) else list(src or []))
    return out


def _input_aliases(in_map: Any) -> dict[str, str]:
    """Tool input id → glob for the source's declared output name.

    CWL 'in' maps own-id → source (``<step>/<out>`` or a workflow input);
    the child command references ``$UMRS_INPUTS_DIR/<own_id>`` — the alias
    copies the run-scoped upstream key to that stable name.
    """
    if not isinstance(in_map, dict):
        return {}
    aliases: dict[str, str] = {}
    for own_id, binding in in_map.items():
        source = binding.get("source") if isinstance(binding, dict) else binding
        if isinstance(source, list):
            raise CwlError(f"input {own_id!r}: array sources are not mappable")
        if not source:
            continue
        out_id = str(source).split("/", 1)[-1]
        aliases[str(own_id)] = f"**/{out_id}"
    return aliases


def _wrap(
    workflow_name: str, steps: list[dict[str, Any]], *, description: str
) -> dict[str, Any]:
    return {
        "name": workflow_name,
        "version": "1.0",
        "description": description,
        "steps": steps,
        "source": {"type": "imported", "format": "cwl"},
    }
