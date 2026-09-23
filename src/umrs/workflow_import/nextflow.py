"""Nextflow → UMRS workflow converter (D-08 leg 3, declarative subset).

Maps the DETERMINISTIC subset of Nextflow into a bundle
``workflow.json`` (same ShellStep target as the Snakemake/CWL lanes):

- ``process`` blocks with ``input:`` / ``output:`` / ``script:`` —
  scalar ``path``/``val`` inputs, scalar ``path`` outputs, triple-quoted
  scripts whose ``${name}`` interpolations reference declared
  inputs/outputs (mapped to the ShellStep stable-name contract);
- ``workflow`` blocks with plain invocation expressions
  (``next_proc(prev.out)``) — the DAG comes from those explicit
  invocations; root inputs come from ``Channel.fromPath("literal")`` /
  ``Channel.value("literal")``.

Nextflow's essence is **dynamic dataflow** — most channel operators
(map/filter/combine/flatten/…), tuples, ``each:``, closures, and
dynamic output names exist precisely to construct channels at runtime.
None of that is mappable without approximating, so everything outside
the subset **refuses by name** (the strictest refusal set of the three
converters): ``tuple``/``each``/``env:``/``conda:``/``container:``
(running without the env is a silent semantic change), channel
operators other than fromPath/value, closures in invocations,
``shell:``/``exec:`` bodies, ``publishDir``, dynamic ``${...}`` output
names, ``include``.
"""

from __future__ import annotations

import re
from typing import Any

SHELL_STEP_ENTRYPOINT = "umrs.pipeline_sdk.templates.shell_step:ShellStep"

_PROCESS_RE = re.compile(r"^process\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{?\s*$")
_INVOCATION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$")
_SCRIPT_INTERP_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_CHANNEL_LIT_RE = re.compile(
    r"^(?:[A-Za-z_]\w*\s*=\s*)?Channel\.(fromPath|value)\(\s*(['\"])(.*?)\2\s*\)$"
)

_REFUSALS: dict[str, str] = {
    "tuple": "tuple inputs/outputs are not mappable (dynamic channel shapes)",
    "each": "each: scatter-style inputs are not mappable",
    "env:": "env: directives are not mappable",
    "conda": "conda: directives are not mappable — the command would run without the env",
    "container": "container: directives are not mappable — the command would run without the container",
    "publishDir": "publishDir is not mappable",
    "shell:": "shell: bodies are not mappable — use script:",
    "exec:": "exec: bodies are not mappable — use script:",
    "include": "module includes are not mappable",
    "closure": "closure arguments in invocations are not mappable",
}

_ALLOWED_CHANNEL_OPS = {"fromPath", "value"}

_KNOWN_PROCESS_DIRECTIVES = {"input", "output", "script"}


class NextflowError(ValueError):
    """Named refusal: the construct is outside the mappable subset."""


def _refuse(construct: str, line_no: int) -> None:
    reason = _REFUSALS.get(construct, f"unsupported construct: {construct}")
    raise NextflowError(f"line {line_no}: {reason}")


def convert_nextflow(
    source: str, *, workflow_name: str = "imported-nextflow"
) -> dict[str, Any]:
    """Convert a Nextflow script (deterministic subset) to bundle workflow.json."""
    processes, invocations, root_channels = _parse(source)
    if not processes:
        raise NextflowError(
            "no processes found — the script is empty or outside the mappable subset"
        )
    steps = _build_steps(processes, invocations, root_channels)
    return {
        "name": workflow_name,
        "version": "1.0",
        "description": (
            f"Imported from Nextflow (deterministic subset); {len(steps)} processes"
        ),
        "steps": steps,
        "source": {"type": "imported", "format": "nextflow"},
    }


def _parse(
    source: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """One pass: process blocks + workflow invocations + root channel literals."""
    processes: dict[str, dict[str, Any]] = {}
    invocations: list[dict[str, Any]] = []
    root_inputs: list[str] = []
    root_channels: dict[str, str] = {}
    state = _ParserState()

    for line_no, raw in enumerate(source.splitlines(), start=1):
        stripped = raw.strip()
        if _skippable(stripped):
            continue

        if state.in_script:
            state.script_line(stripped, processes)
            continue

        if _scan_channel_line(stripped, line_no, root_inputs, root_channels):
            continue

        if state.try_open_process(stripped, processes):
            continue
        if state.workflow_fence(stripped):
            continue
        if state.process_transition(stripped):
            continue
        if state.in_workflow_body():
            _workflow_body_line(stripped, invocations, line_no)
            continue

        state.directive = _process_line(
            processes, state.current, state.directive, stripped, line_no
        )

    return processes, invocations, root_channels


def _skippable(stripped: str) -> bool:
    """Blank or comment lines."""
    return not stripped or stripped.startswith("//")


class _ParserState:
    """Line-parser state: the process/workflow/script block we are inside."""

    def __init__(self) -> None:
        self.current: str | None = None
        self.directive: str | None = None
        self.in_workflow = False
        self.in_script = False

    def try_open_process(
        self, stripped: str, processes: dict[str, dict[str, Any]]
    ) -> bool:
        """A 'process NAME {' line opens the next block."""
        pm = _PROCESS_RE.match(stripped)
        if not pm:
            return False
        self.current = pm.group(1)
        self.directive = None
        processes[self.current] = {"inputs": [], "outputs": [], "script": ""}
        return True

    def in_workflow_body(self) -> bool:
        return self.in_workflow and self.current is None

    def workflow_fence(self, stripped: str) -> bool:
        """Workflow block open/close fences; True when the line was consumed."""
        if stripped == "workflow {":
            self.in_workflow = True
            return True
        if self.in_workflow and stripped == "}":
            self.in_workflow = False
            return True
        return False

    def process_transition(self, stripped: str) -> bool:
        """Script fences + process close; True when the line was consumed."""
        if (
            self.directive == "script"
            and self.current is not None
            and stripped == '"""'
        ):
            # Body lines are raw command text (they may end with '}' via
            # ${...} interpolations; the structure handling must not see them).
            self.in_script = True
            return True
        if self.current is not None and stripped.endswith("}") and "${" not in stripped:
            self.current = None
            self.directive = None
            return True
        return False

    def script_line(self, stripped: str, processes: dict[str, dict[str, Any]]) -> None:
        """One script-body line (or the closing fence)."""
        if '"""' in stripped:
            self.in_script = False
            return
        if self.current:
            proc = processes[self.current]
            proc["script"] += (" " if proc["script"] else "") + stripped


def _scan_channel_line(
    stripped: str,
    line_no: int,
    root_inputs: list[str],
    root_channels: dict[str, str],
) -> bool:
    """Refusal scan + literal-channel capture; True when the line was handled."""
    for construct in _REFUSALS:
        if construct in stripped:
            _refuse(construct, line_no)
    m = _CHANNEL_LIT_RE.match(stripped)
    if m:
        if m.group(1) not in _ALLOWED_CHANNEL_OPS:
            _refuse(f"channel:{m.group(1)}", line_no)
        root_inputs.append(m.group(3))
        var = re.match(r"^([A-Za-z_]\w*)\s*=", stripped)
        if var:
            root_channels[var.group(1)] = m.group(3)
        return True
    if re.search(r"Channel\.\w+", stripped):
        _refuse("channel-operator", line_no)
    return False


def _process_line(
    processes: dict[str, dict[str, Any]],
    current: str | None,
    directive: str | None,
    stripped: str,
    line_no: int,
) -> str | None:
    """Handle one line inside a process block; returns the directive state."""
    if current is None:
        _refuse("top-level-groovy", line_no)
    m = re.match(r"^([a-z_]+)\s*:\s*$", stripped)
    if m:
        if m.group(1) not in _KNOWN_PROCESS_DIRECTIVES:
            _refuse(f"directive:{m.group(1)}", line_no)
        return m.group(1)
    if directive is None:
        _refuse(f"process-line:{stripped[:40]}", line_no)
    if directive == "input":
        _record_scalar(
            processes[current]["inputs"], stripped, line_no, ("path", "val", "file")
        )
    elif directive == "output":
        _record_scalar(
            processes[current]["outputs"], stripped, line_no, ("path", "file")
        )
    elif directive == "script":
        processes[current]["script"] += (
            " " if processes[current]["script"] else ""
        ) + stripped
    return directive


def _record_scalar(
    target: list[str], stripped: str, line_no: int, kinds: tuple[str, ...]
) -> None:
    """One '<kind> <name>' scalar line; anything richer refuses by name."""
    parts = stripped.split()
    if len(parts) != 2 or parts[0] not in kinds:
        _refuse(f"scalar-shape:{stripped[:40]}", line_no)
    if "${" in stripped:
        _refuse("dynamic-output", line_no)
    target.append(parts[1])


def _build_steps(
    processes: dict[str, dict[str, Any]],
    invocations: list[dict[str, Any]],
    root_channels: dict[str, str],
) -> list[dict[str, Any]]:
    """ShellStep per process; DAG from explicit invocation arguments."""
    deps_of, root_aliases = _invocation_edges(processes, invocations, root_channels)

    steps: list[dict[str, Any]] = []
    for name, proc in processes.items():
        command = _resolve_script(proc, name)
        deps = deps_of.get(name, [])
        aliases = _upstream_aliases(processes, name, deps)
        aliases.update(root_aliases.get(name, {}))
        if not proc["outputs"]:
            raise NextflowError(f"process {name!r} declares no outputs — not mappable")
        steps.append(
            {
                "name": name,
                "entrypoint": SHELL_STEP_ENTRYPOINT,
                "depends_on": sorted(set(deps)),
                "config": {
                    "command": command,
                    "output_globs": [f"outputs/{o}" for o in proc["outputs"]],
                    "input_aliases": aliases,
                },
            }
        )
    return steps


def _workflow_body_line(
    stripped: str, invocations: list[dict[str, Any]], line_no: int
) -> None:
    """One line inside the workflow block: invocation or refusal."""
    inv = _INVOCATION_RE.match(stripped.replace(";", ""))
    if inv:
        invocations.append({"proc": inv.group(1), "args": inv.group(2).strip()})
        return
    _refuse(f"workflow-line:{stripped[:40]}", line_no)


def _invocation_edges(
    processes: dict[str, dict[str, Any]],
    invocations: list[dict[str, Any]],
    root_channels: dict[str, str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, str]]]:
    """DAG edges + root-channel aliases from the workflow invocations."""
    deps_of: dict[str, list[str]] = {}
    root_aliases: dict[str, dict[str, str]] = {}
    for inv in invocations:
        deps_of[inv["proc"]] = _invocation_deps(
            processes, inv, root_channels, root_aliases
        )
    return deps_of, root_aliases


def _invocation_deps(
    processes: dict[str, dict[str, Any]],
    inv: dict[str, Any],
    root_channels: dict[str, str],
    root_aliases: dict[str, dict[str, str]],
) -> list[str]:
    """One invocation: upstream deps + root-channel aliases; refusals named."""
    proc, args = inv["proc"], inv["args"]
    if "{" in args or "}" in args or "->" in args:
        _refuse("closure", 0)
    deps: list[str] = []
    for arg in (a.strip() for a in args.split(",")):
        if not arg:
            continue
        _apply_invocation_arg(processes, proc, arg, root_channels, root_aliases, deps)
    return deps


def _apply_invocation_arg(
    processes: dict[str, dict[str, Any]],
    proc: str,
    arg: str,
    root_channels: dict[str, str],
    root_aliases: dict[str, dict[str, str]],
    deps: list[str],
) -> None:
    """Classify + apply one invocation argument (refusals named)."""
    kind, value = _classify_arg(arg)
    if kind == "upstream":
        if value not in processes:
            _refuse(f"unknown-source:{arg}", 0)
        deps.append(value)
    elif kind == "channel":
        if value not in root_channels:
            _refuse(f"unknown-channel:{value}", 0)
        _wire_root_channel(processes, proc, value, root_channels, root_aliases)
    else:
        _refuse(f"invocation-arg:{arg}", 0)


def _classify_arg(arg: str) -> tuple[str, str]:
    """'x.out' → upstream process; bare identifier → channel name."""
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\.out$", arg)
    if m:
        return "upstream", m.group(1)
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", arg):
        return "channel", arg
    return "bad", arg


def _wire_root_channel(
    processes: dict[str, dict[str, Any]],
    proc: str,
    channel_var: str,
    root_channels: dict[str, str],
    root_aliases: dict[str, dict[str, str]],
) -> None:
    """A root channel literal feeds the process's single scalar input."""
    inputs = processes[proc]["inputs"]
    if len(inputs) != 1:
        _refuse(f"root-channel-arity:{proc}", 0)
    literal = root_channels[channel_var]
    root_aliases.setdefault(proc, {})[inputs[0]] = f"**/{literal}"


def _upstream_aliases(
    processes: dict[str, dict[str, Any]], name: str, deps: list[str]
) -> dict[str, str]:
    """Inputs produced by upstream processes alias the upstream output name."""
    aliases: dict[str, str] = {}
    for inp in processes[name]["inputs"]:
        if any(inp in processes[d]["outputs"] for d in deps):
            aliases[inp] = f"**/{inp}"
    return aliases


def _output_name_of(
    processes: dict[str, dict[str, Any]], dep_names: list[str], input_name: str
) -> str:
    for dep in dep_names:
        if input_name in processes[dep]["outputs"]:
            return input_name
    return input_name


def _resolve_script(proc: dict[str, Any], name: str) -> str:
    """${x} → $UMRS_INPUTS_DIR/<x> (input) or $UMRS_OUTPUTS_DIR/<x> (output)."""
    command = proc["script"]
    for interp in _SCRIPT_INTERP_RE.findall(command):
        if interp in proc["inputs"]:
            continue
        if interp in proc["outputs"]:
            continue
        raise NextflowError(
            f"process {name!r}: script references undeclared name {interp!r}"
        )
    command = _SCRIPT_INTERP_RE.sub(_interp_value(proc), command)
    return command


def _interp_value(proc: dict[str, Any]) -> Any:
    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in proc["inputs"]:
            return f"$UMRS_INPUTS_DIR/{name}"
        return f"$UMRS_OUTPUTS_DIR/{name}"

    return repl
