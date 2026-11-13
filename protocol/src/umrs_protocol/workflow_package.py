"""What a workflow.json must declare to be uploaded (2026-10-02).

UMRS refuses an upload that breaks a rule here; ``r4 workflow upload`` and
``r4 workflow lint`` run the same checks first, so an author sees every
problem at once, before a round trip. Mandatory:

- ``description``: a non-blank string saying what the workflow does, what
  it needs and what it produces. It is what users and their agents read
  first (catalog rows, the console's definition card, workflow_describe).
- The flow graph: every step lists ``depends_on`` explicitly (the steps it
  waits for; ``[]`` for a step that starts the run), naming known steps,
  never itself, with no cycle. The runner still reads the older shapes (no
  ``depends_on`` anywhere = a declaration-order chain; the
  ``execution.parallel`` shorthand), so workflows registered before this
  rule keep running. An upload spells the graph out.

The structural keys an upload has always needed (``name``, ``source``
with a lowercase ``type``, a known ``execution_mode``, ``steps``, a step's
``entrypoint`` unless it is a ``gate``) are reported here in the same
words, instead of as a bare parser error.

Warnings never refuse an upload; they come back as advice:

- ``{{ params.x }}`` used in an LLMStep prompt while no
  ``parameter_schema`` declares ``x`` (users and agents cannot see what to
  trigger the workflow with);
- a ``{{ ... }}`` placeholder in a built-in step's config that nothing
  fills in at run time. LLMStep renders only ``{{ params.* }}`` and
  ``{{ steps.* }}`` inside ``prompt_template``, and the router fills
  ``model`` only when the config leaves it out.

Pure stdlib (the r4-contracts rule): UMRS and r4 both import it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ERROR = "error"
WARNING = "warning"

#: Keys UMRS reads a parameter declaration from (first found wins).
SCHEMA_KEYS = ("parameter_schema", "parameters_schema", "parameters")

#: Run values the platform fills in itself (never typed by the user).
AUTO_PARAMS = frozenset({"project_id", "input_files"})

#: umrs.pipeline_sdk.workflow.SourceType / ExecutionMode values (pinned
#: against the enums by umrs/tests/test_upload_workflow_contract.py).
SOURCE_TYPES = ("local_path", "s3", "git", "uploaded")
EXECUTION_MODES = ("in_process", "subprocess", "slurm", "k8s")

_TEMPLATES = "umrs.pipeline_sdk.templates."
_LLM_STEP = "umrs.pipeline_sdk.templates.llm_step:LLMStep"
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
_PARAM_REF_RE = re.compile(r"\{\{\s*params\.([A-Za-z_]\w*)")


@dataclass(frozen=True)
class PackageProblem:
    """One finding: a stable ``code``, where it is, and how to fix it."""

    code: str
    where: str
    message: str
    severity: str = ERROR

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


def _problem(code: str, where: str, message: str, severity: str = ERROR) -> PackageProblem:
    return PackageProblem(code=code, where=where, message=message, severity=severity)


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _name_problems(data: Mapping[str, Any]) -> list[PackageProblem]:
    if _is_text(data.get("name")):
        return []
    return [_problem("missing_name", "name", "missing — the workflow's name")]


def _description_problems(data: Mapping[str, Any]) -> list[PackageProblem]:
    if _is_text(data.get("description")):
        return []
    return [
        _problem(
            "missing_description",
            "description",
            "missing — say in a sentence or two what the workflow does, "
            "what it needs (inputs, parameters) and what it produces",
        )
    ]


def _source_problems(data: Mapping[str, Any]) -> list[PackageProblem]:
    source = data.get("source")
    if not isinstance(source, Mapping) or not source.get("type") or not source.get("path"):
        return [
            _problem(
                "missing_source",
                "source",
                'missing — for code shipped in the bundle use {"type": "local_path", "path": "."}',
            )
        ]
    if source["type"] in SOURCE_TYPES:
        return []
    return [
        _problem(
            "bad_source_type",
            "source.type",
            f'"{source["type"]}" is not one of {", ".join(SOURCE_TYPES)} (lowercase)',
        )
    ]


def _mode_problems(data: Mapping[str, Any]) -> list[PackageProblem]:
    mode = data.get("execution_mode")
    if mode is None or mode in EXECUTION_MODES:
        return []
    return [
        _problem(
            "bad_execution_mode",
            "execution_mode",
            f'"{mode}" is not one of {", ".join(EXECUTION_MODES)} (lowercase)',
        )
    ]


def _steps_list_problems(data: Mapping[str, Any]) -> list[PackageProblem]:
    steps = data.get("steps")
    if isinstance(steps, list) and steps:
        return []
    return [_problem("no_steps", "steps", "missing — a list of at least one step")]


_STRUCTURE_CHECKS = (
    _name_problems,
    _description_problems,
    _source_problems,
    _mode_problems,
    _steps_list_problems,
)


def _step_name(step: Any) -> str | None:
    name = step.get("name") if isinstance(step, Mapping) else None
    return name if isinstance(name, str) and name else None


def _step_label(step: Any, index: int) -> str:
    name = _step_name(step)
    return f'steps[{index}] "{name}"' if name else f"steps[{index}]"


def _dependency_hint(steps: Sequence[Any], index: int) -> str:
    previous_name = _step_name(steps[index - 1]) if index > 0 else None
    if previous_name:
        return (
            f'e.g. ["{previous_name}"] if it waits for {previous_name}, or [] if it starts the run'
        )
    return "[] if it starts the run"


def _entrypoint_problems(step: Mapping[str, Any], label: str) -> list[PackageProblem]:
    if step.get("gate") or step.get("entrypoint"):
        return []
    return [
        _problem(
            "missing_entrypoint",
            f"{label}.entrypoint",
            'missing — "module:Class" of the step\'s Python class, '
            'or "gate": true for a manual approval step',
        )
    ]


def _edge_problems(name: str, deps: list[str], label: str, known: set[str]) -> list[PackageProblem]:
    problems: list[PackageProblem] = []
    for dep in deps:
        if dep == name:
            problems.append(
                _problem("self_dependency", f"{label}.depends_on", "a step cannot wait for itself")
            )
        elif dep not in known:
            choices = ", ".join(sorted(known - {name})) or "none"
            problems.append(
                _problem(
                    "unknown_dependency",
                    f"{label}.depends_on",
                    f'"{dep}" is not a step of this workflow (steps: {choices})',
                )
            )
    return problems


def _depends_on_problems(
    steps: Sequence[Any], index: int, known: set[str]
) -> tuple[list[PackageProblem], list[str]]:
    """Problems with one step's depends_on, and its valid edges."""
    step = steps[index]
    label = _step_label(step, index)
    deps = step.get("depends_on")
    if deps is None:
        hint = _dependency_hint(steps, index)
        return [
            _problem(
                "missing_depends_on",
                f"{label}.depends_on",
                f"missing — list the steps it waits for, {hint}",
            )
        ], []
    if not isinstance(deps, list) or not all(isinstance(d, str) and d for d in deps):
        return [
            _problem("bad_depends_on", f"{label}.depends_on", "must be a list of step names")
        ], []
    name = step["name"]
    return _edge_problems(name, deps, label, known), [d for d in deps if d in known and d != name]


def _step_problems(
    steps: Sequence[Any], known: set[str]
) -> tuple[list[PackageProblem], dict[str, list[str]]]:
    """Per-step entrypoint + depends_on checks; returns the valid edges too."""
    problems: list[PackageProblem] = []
    edges: dict[str, list[str]] = {}
    for index, step in enumerate(steps):
        label = _step_label(step, index)
        if _step_name(step) is None:
            problems.append(_problem("bad_step", label, 'each step is an object with a "name"'))
            continue
        problems += _entrypoint_problems(step, label)
        step_problems, edges[step["name"]] = _depends_on_problems(steps, index, known)
        problems += step_problems
    return problems, edges


def _find_cycle(edges: Mapping[str, Sequence[str]]) -> list[str] | None:
    """A dependency cycle as ``[a, b, …, a]``, or None."""
    state: dict[str, int] = {}  # 1 = on the path, 2 = done
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        path.append(node)
        for dep in edges.get(node, ()):
            if state.get(dep) == 1:
                return path[path.index(dep) :] + [dep]
            found = visit(dep) if state.get(dep) is None else None
            if found:
                return found
        path.pop()
        state[node] = 2
        return None

    for node in edges:
        found = visit(node) if state.get(node) is None else None
        if found:
            return found
    return None


def _cycle_problems(edges: Mapping[str, Sequence[str]]) -> list[PackageProblem]:
    cycle = _find_cycle(edges)
    if not cycle:
        return []
    return [
        _problem(
            "dependency_cycle",
            "steps",
            f"cycle {' → '.join(cycle)} — a step cannot wait for itself through others",
        )
    ]


def _legacy_parallel_problems(
    data: Mapping[str, Any], steps: Sequence[Any]
) -> list[PackageProblem]:
    execution = data.get("execution")
    parallel = execution.get("parallel") if isinstance(execution, Mapping) else None
    if not parallel or any(
        isinstance(s, Mapping) and s.get("depends_on") is not None for s in steps
    ):
        return []
    return [
        _problem(
            "legacy_parallel",
            "execution.parallel",
            "a shorthand the runner still reads for registered workflows; an upload spells the "
            "graph out — the parallel steps list the step before the group in depends_on, the "
            "step after lists the whole group",
            WARNING,
        )
    ]


def _declared_params(data: Mapping[str, Any]) -> set[str] | None:
    """Names a parameter declaration covers; None when there is none."""
    for key in SCHEMA_KEYS:
        candidate = data.get(key)
        if isinstance(candidate, Mapping) and isinstance(candidate.get("properties"), Mapping):
            return set(candidate["properties"])
        if isinstance(candidate, list):
            return {item.get("name") for item in candidate if isinstance(item, Mapping)} - {None}
    return None


def _strings(value: Any, path: str):
    """(path, text) for every string inside a config value."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _strings(item, f"{path}[{i}]")


def _template_config(step: Any) -> tuple[str, Mapping[str, Any]] | None:
    """(entrypoint, config) of a built-in template step; None for custom ones."""
    if not isinstance(step, Mapping):
        return None
    entrypoint = str(step.get("entrypoint") or "")
    config = step.get("config")
    if not entrypoint.startswith(_TEMPLATES) or not isinstance(config, Mapping):
        return None  # a custom class may render its own placeholders
    return entrypoint, config


def _placeholder_problem(path: str, expr: str) -> PackageProblem:
    root = expr.split(".", 1)[0].strip()
    fix = (
        " — leave model out to let the platform route the step, or name a served model"
        if root == "model"
        else " — only {{ params.* }} and {{ steps.* }} inside an LLMStep prompt_template are"
    )
    return _problem(
        "unfilled_placeholder", path, f"{{{{ {expr} }}}} is not filled in at run time{fix}", WARNING
    )


def _rendered(entrypoint: str, path: str, expr: str) -> bool:
    """LLMStep renders params.* / steps.* inside prompt_template — nothing else."""
    root = expr.split(".", 1)[0].strip()
    return (
        entrypoint == _LLM_STEP
        and path.endswith(".config.prompt_template")
        and root in ("params", "steps")
    )


def _placeholder_problems(steps: Sequence[Any]) -> list[PackageProblem]:
    problems: list[PackageProblem] = []
    for index, step in enumerate(steps):
        template = _template_config(step)
        if template is None:
            continue
        entrypoint, config = template
        for path, text in _strings(config, f"{_step_label(step, index)}.config"):
            problems += [
                _placeholder_problem(path, expr)
                for expr in _PLACEHOLDER_RE.findall(text)
                if not _rendered(entrypoint, path, expr)
            ]
    return problems


def _prompt_params(steps: Sequence[Any]) -> set[str]:
    """Run parameters LLMStep prompts read (keys their own config sets excluded)."""
    used: set[str] = set()
    for step in steps:
        template = _template_config(step)
        if template is None or template[0] != _LLM_STEP:
            continue
        config = template[1]
        prompt = config.get("prompt_template")
        if isinstance(prompt, str):
            used |= {name for name in _PARAM_REF_RE.findall(prompt) if name not in config}
    return used


def _schema_problems(data: Mapping[str, Any], steps: Sequence[Any]) -> list[PackageProblem]:
    missing = sorted(_prompt_params(steps) - (_declared_params(data) or set()) - AUTO_PARAMS)
    if not missing:
        return []
    return [
        _problem(
            "undeclared_params",
            "parameter_schema",
            f"the prompts use {', '.join(missing)} but no parameter_schema declares "
            "them — declare each (type, description, required) so users and agents "
            "see what to trigger the workflow with",
            WARNING,
        )
    ]


def _graph_problems(data: Mapping[str, Any], steps: Sequence[Any]) -> list[PackageProblem]:
    known = {name for name in map(_step_name, steps) if name}
    step_problems, edges = _step_problems(steps, known)
    return step_problems + _cycle_problems(edges) + _legacy_parallel_problems(data, steps)


def _errors_first(problems: list[PackageProblem]) -> list[PackageProblem]:
    return errors_of(problems) + warnings_of(problems)


def workflow_package_problems(data: Any) -> list[PackageProblem]:
    """Every rule ``data`` (a parsed workflow.json) breaks; errors first."""
    if not isinstance(data, Mapping):
        return [_problem("not_an_object", "workflow.json", "must be a JSON object")]
    problems = [p for check in _STRUCTURE_CHECKS for p in check(data)]
    steps = data["steps"] if isinstance(data.get("steps"), list) else []
    problems += _graph_problems(data, steps)
    problems += _placeholder_problems(steps) + _schema_problems(data, steps)
    return _errors_first(problems)


def errors_of(problems: Sequence[PackageProblem]) -> list[PackageProblem]:
    return [p for p in problems if p.severity == ERROR]


def warnings_of(problems: Sequence[PackageProblem]) -> list[PackageProblem]:
    return [p for p in problems if p.severity != ERROR]


def format_problems(problems: Sequence[PackageProblem]) -> str:
    """One line per problem, numbered — what an upload refusal says."""
    return "; ".join(f"({i}) {p}" for i, p in enumerate(problems, 1))
