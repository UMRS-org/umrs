"""Snakemake → UMRS workflow converter (D-08, declarative subset).

Maps the DECLARATIVE subset of Snakefiles into a bundle ``workflow.json``:

- ``rule <name>:`` with ``input:`` / ``output:`` / ``shell:`` /
  ``params:`` / ``threads:`` blocks (literal file names, no wildcards);
- DAG edges derive from concrete file flow: when rule B's input file is
  rule A's output, B depends on A;
- the shell command becomes a :class:`ShellStep` config with
  ``{input}``/``{output}``/``{params.X}``/``{threads}`` placeholders
  resolved to the concrete file lists (inputs materialize under
  ``UMRS_INPUTS_DIR``).

Anything outside the subset **refuses by name** (never approximated):
wildcards (``{sample}``), ``checkpoint``, ``dynamic``, ``temp()``,
``directory()``, ``run:``/``script:``/``notebook:`` bodies, ``include``,
``configfile``/``config:`` indirection, ``wildcard_constraints``, and
unknown rule directives.
"""

from __future__ import annotations

import re
from typing import Any

SHELL_STEP_ENTRYPOINT = "umrs.pipeline_sdk.templates.shell_step:ShellStep"

_REFUSALS: dict[str, str] = {
    "wildcard": "wildcard rules ({sample}) are not mappable — provide concrete file lists",
    "checkpoint": "checkpoints are not mappable",
    "dynamic": "dynamic files are not mappable",
    "temp(": "temp() files are not mappable",
    "directory(": "directory() outputs are not mappable",
    "run:": "'run:' Python bodies are not mappable — use 'shell:'",
    "script:": "'script:' bodies are not mappable — use 'shell:'",
    "notebook:": "'notebook:' bodies are not mappable — use 'shell:'",
    "include": "multi-file includes are not mappable — convert one self-contained Snakefile",
    "configfile": "configfile indirection is not mappable — inline literal values",
    "wildcard_constraints": "wildcard_constraints require wildcards, which are refused",
    "container:": "container directives are not mappable — the command would run without the container (silent approximation refused)",
    "conda:": "conda directives are not mappable — the command would run without the env (silent approximation refused)",
}

_KNOWN_DIRECTIVES = {
    "input",
    "output",
    "shell",
    "params",
    "threads",
    "resources",
    "benchmark",
    "log",
    "message",
}

_RULE_RE = re.compile(r"^rule\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$")
_BRACE_RE = re.compile(r"\{([^}:|]+)\}")


class SnakefileError(ValueError):
    """Named refusal: the construct is outside the mappable subset."""


def _refuse(construct: str, line_no: int) -> None:
    reason = _REFUSALS.get(construct, f"unsupported construct: {construct}")
    raise SnakefileError(f"line {line_no}: {reason}")


def _refuse_function_values(text: str, line_no: int) -> None:
    """temp()/directory() wrappers inside file lists refuse by name."""
    for construct in ("temp(", "directory(", "ancient(", "touch("):
        if construct in text:
            _refuse(construct, line_no)


def _parse_items(text: str) -> list[str]:
    """Comma/whitespace-separated literal file names."""
    items = [t.strip(" '\"") for t in text.split(",")]
    return [t for t in items if t]


def convert_snakemake(
    source: str, *, workflow_name: str = "imported-snakemake"
) -> dict[str, Any]:
    """Convert a Snakefile (declarative subset) to bundle workflow.json."""
    rules = _parse_rules(source)
    _validate_rules(rules)
    return _build_workflow(rules, workflow_name)


def _parse_rules(source: str) -> dict[str, dict[str, Any]]:
    """One pass over the Snakefile: rule blocks with their directives."""
    rules: dict[str, dict[str, Any]] = {}
    current: str | None = None
    directive: str | None = None

    for line_no, raw in enumerate(source.splitlines(), start=1):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        _scan_structure(stripped, indent, line_no)

        rule_match = _RULE_RE.match(line)
        if rule_match:
            current = rule_match.group(1)
            rules[current] = {
                "input": [],
                "output": [],
                "shell": "",
                "params": {},
                "threads": 1,
            }
            directive = None
            continue
        if current is None:
            _refuse("top-level-python", line_no)  # only rule blocks are mappable

        directive = _direct_value(rules, current, directive, stripped, indent, line_no)
    return rules


def _scan_structure(stripped: str, indent: int, line_no: int) -> None:
    """Refusal scan on STRUCTURE lines only — command payloads may
    legitimately contain words like 'include'."""
    if indent > 4:
        return
    structure = stripped.split(":", 1)[0].strip()
    for construct in _REFUSALS:
        if structure == construct.rstrip(":") or (
            indent == 0 and construct in stripped.split()
        ):
            _refuse(construct, line_no)


def _direct_value(
    rules: dict[str, dict[str, Any]],
    current: str,
    directive: str | None,
    stripped: str,
    indent: int,
    line_no: int,
) -> str | None:
    """Handle one non-rule line: set/record the directive, return its state."""
    if indent == 4:
        directive, stripped = _inline_directive(stripped, directive, line_no)
        if stripped is None:
            return directive
    if directive is None:
        _refuse(f"line-shape:{stripped[:40]}", line_no)
    _record_value(rules, current, directive, stripped, line_no)
    return directive


def _inline_directive(
    stripped: str, directive: str | None, line_no: int
) -> tuple[str | None, str | None]:
    """Parse a 4-indent line; returns (directive, inline_value|None)."""
    m = re.match(r"^([a-z_]+)\s*:\s*(.*)$", stripped)
    if m and m.group(1) not in _KNOWN_DIRECTIVES:
        _refuse(f"directive:{m.group(1)}", line_no)
    if m:
        value = m.group(2).strip()
        return m.group(1), (value or None)
    if not stripped.endswith(":"):
        _refuse(f"line-shape:{stripped[:40]}", line_no)
    return stripped[:-1].rstrip(":"), None


def _record_value(
    rules: dict[str, dict[str, Any]],
    current: str,
    directive: str | None,
    stripped: str,
    line_no: int,
) -> None:
    """Record one directive value into the current rule."""
    if directive in ("input", "output"):
        _refuse_function_values(stripped, line_no)
        rules[current][directive].extend(_parse_items(stripped))
        return
    _record_scalar(rules, current, directive, stripped, line_no)


def _record_scalar(
    rules: dict[str, dict[str, Any]],
    current: str,
    directive: str | None,
    stripped: str,
    line_no: int,
) -> None:
    """params / threads / shell — the scalar directives."""
    if directive == "params":
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", stripped)
        if not m:
            _refuse("params-shape", line_no)
        rules[current]["params"][m.group(1)] = m.group(2).strip(" \"'")
    elif directive == "threads":
        try:
            rules[current]["threads"] = int(stripped)
        except ValueError:
            _refuse("threads-shape", line_no)
    elif directive == "shell":
        cmd = stripped.strip(" \"'")
        if cmd.startswith(('r"', "r'")):
            _refuse("raw-shell-string", line_no)
        rules[current]["shell"] += (" " if rules[current]["shell"] else "") + cmd


def _validate_rules(rules: dict[str, dict[str, Any]]) -> None:
    """Refuse by name anything outside the mappable subset."""
    if not rules:
        raise SnakefileError(
            "no rules found — the Snakefile is empty or outside the mappable subset"
        )
    for name, rule in rules.items():
        if not rule["shell"]:
            raise SnakefileError(
                f"rule {name!r} has no shell command — only 'shell:' bodies are mappable"
            )
        for brace in _BRACE_RE.findall(rule["shell"]):
            base = brace.split(".")[0].strip()
            if base not in ("input", "output", "params", "threads"):
                _refuse(f"shell-placeholder:{{{brace}}}", 0)
        for item in rule["input"] + rule["output"]:
            if "{" in item or "}" in item:
                _refuse("wildcard", 0)


def _build_workflow(
    rules: dict[str, dict[str, Any]], workflow_name: str
) -> dict[str, Any]:
    """DAG edges from concrete file flow; ShellStep config per rule."""
    producer: dict[str, str] = {}
    for name, rule in rules.items():
        for out in rule["output"]:
            producer[out] = name

    steps: list[dict[str, Any]] = []
    for name, rule in rules.items():
        deps = sorted({producer[f] for f in rule["input"] if f in producer})
        command = _resolve_placeholders(rule)
        steps.append(
            {
                "name": name,
                "entrypoint": SHELL_STEP_ENTRYPOINT,
                "depends_on": deps,
                "config": {
                    "command": command,
                    "output_globs": [
                        f if f.startswith("outputs/") else f"outputs/{f}"
                        for f in rule["output"]
                    ],
                },
            }
        )

    return {
        "name": workflow_name,
        "version": "1.0",
        "description": f"Imported from Snakemake (declarative subset); {len(steps)} rules",
        "steps": steps,
        "source": {"type": "imported", "format": "snakemake"},
    }


def _resolve_placeholders(rule: dict[str, Any]) -> str:
    """Expand {input}/{output}/{params.X}/{threads} to concrete values."""
    command = rule["shell"]
    inputs = " ".join(f"$UMRS_INPUTS_DIR/{f}" for f in rule["input"])
    outputs = " ".join(f"$UMRS_OUTPUTS_DIR/{f}" for f in rule["output"])
    command = command.replace("{input}", inputs)
    command = command.replace("{output}", outputs)
    command = command.replace("{threads}", str(rule["threads"]))
    for key, value in rule["params"].items():
        command = command.replace(f"{{params.{key}}}", value)
    return command
