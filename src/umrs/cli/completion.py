"""Small argparse-based shell completion helpers."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path


_NO_VALUE_ACTION_TYPES = (
    argparse._StoreTrueAction,
    argparse._StoreFalseAction,
    argparse._StoreConstAction,
    argparse._AppendConstAction,
    argparse._CountAction,
    argparse._HelpAction,
    argparse._VersionAction,
)


def _action_takes_value(action: argparse.Action) -> bool:
    if isinstance(action, _NO_VALUE_ACTION_TYPES):
        return False
    return action.nargs != 0


def _subparser_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _option_action(
    parser: argparse.ArgumentParser,
    token: str,
) -> argparse.Action | None:
    option = token.split("=", 1)[0]
    return parser._option_string_actions.get(option)


def _walk_parser(
    parser: argparse.ArgumentParser,
    tokens: Sequence[str],
) -> argparse.ArgumentParser:
    active = parser
    skip_value = False
    for token in tokens:
        if skip_value:
            skip_value = False
            continue
        if token == "--":
            break
        option = _option_action(active, token)
        if option is not None:
            if "=" not in token and _action_takes_value(option):
                skip_value = True
            continue
        subparsers = _subparser_action(active)
        if subparsers is not None and token in subparsers.choices:
            active = subparsers.choices[token]
    return active


def _choice_candidates(action: argparse.Action, prefix: str) -> list[str]:
    if not action.choices:
        return []
    return sorted(str(choice) for choice in action.choices if str(choice).startswith(prefix))


def _option_candidates(parser: argparse.ArgumentParser, prefix: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for action in parser._actions:
        if action.help == argparse.SUPPRESS:
            continue
        for option in action.option_strings:
            if option.startswith(prefix) and option not in seen:
                values.append(option)
                seen.add(option)
    return sorted(values)


def _subcommand_candidates(parser: argparse.ArgumentParser, prefix: str) -> list[str]:
    subparsers = _subparser_action(parser)
    if subparsers is None:
        return []
    return sorted(name for name in subparsers.choices if name.startswith(prefix))


def _positional_choice_candidates(parser: argparse.ArgumentParser, prefix: str) -> list[str]:
    for action in parser._actions:
        if action.option_strings or isinstance(action, argparse._SubParsersAction):
            continue
        candidates = _choice_candidates(action, prefix)
        if candidates:
            return candidates
    return []


def completion_candidates(
    parser: argparse.ArgumentParser,
    words: Sequence[str],
    cword: int,
) -> list[str]:
    """Return completion candidates for normalized shell words.

    ``words`` must not include the program name. ``cword`` is the index of
    the current word inside ``words``.
    """
    cword = max(cword, 0)
    prefix = words[cword] if cword < len(words) else ""
    tokens = list(words[:cword])
    active = _walk_parser(parser, tokens)

    previous = tokens[-1] if tokens else ""
    previous_action = _option_action(active, previous) if previous else None
    if previous_action is not None and _action_takes_value(previous_action):
        candidates = _choice_candidates(previous_action, prefix)
        if candidates:
            return candidates
        return []

    if prefix.startswith("-"):
        return _option_candidates(active, prefix)

    candidates = _subcommand_candidates(active, prefix)
    if candidates:
        return candidates
    return _positional_choice_candidates(active, prefix)


def normalize_completion_words(
    argv: Sequence[str],
    cword: int,
    command_name: str,
) -> tuple[list[str], int]:
    """Drop the shell command word from a completion request."""
    words = list(argv)
    adjusted_cword = cword
    if words and Path(words[0]).name == command_name:
        words = words[1:]
        adjusted_cword -= 1
    return words, max(adjusted_cword, 0)


def print_completion_candidates(
    parser: argparse.ArgumentParser,
    words: Sequence[str],
    cword: int,
) -> None:
    for candidate in completion_candidates(parser, words, cword):
        print(candidate)


def emit_completion_if_requested(
    parser: argparse.ArgumentParser,
    command_name: str,
    env_var: str,
) -> bool:
    if not os.environ.get(env_var):
        return False
    try:
        cword = int(os.environ.get("COMP_CWORD", "0"))
    except ValueError:
        cword = 0
    words, adjusted_cword = normalize_completion_words(sys.argv[1:], cword, command_name)
    print_completion_candidates(parser, words, adjusted_cword)
    return True


def completion_script(command_name: str, env_var: str, shell: str) -> str:
    function_name = f"_{command_name.replace('-', '_')}_complete"
    zsh_setup = ""
    if shell == "zsh":
        zsh_setup = "autoload -Uz bashcompinit\nbashcompinit\n\n"
    return (
        f"# {command_name} {shell} completion\n"
        f"# Install with: eval \"$({command_name} completion {shell})\"\n"
        f"{zsh_setup}"
        f"{function_name}() {{\n"
        "    local IFS=$'\\n'\n"
        f"    COMPREPLY=( $( {env_var}=1 COMP_CWORD=\"$COMP_CWORD\" "
        f"{command_name} \"${{COMP_WORDS[@]}}\" ) )\n"
        "}\n"
        f"complete -o default -o bashdefault -F {function_name} {command_name}\n"
    )


def print_completion_script(command_name: str, env_var: str, shell: str) -> None:
    print(completion_script(command_name, env_var, shell), end="")


# ---------------------------------------------------------------------------
# Shell RC file install / uninstall
# ---------------------------------------------------------------------------

_SHELL_RC: dict[str, str] = {
    "bash": os.path.expanduser("~/.bashrc"),
    "zsh": os.path.expanduser("~/.zshrc"),
}

_MARKER_RE = r'^\s*eval\s+"\$\({command}\s+completion\s+(bash|zsh)\)"'


def _detect_shell() -> str:
    """Return 'zsh' or 'bash' based on the current $SHELL."""
    shell_path = os.environ.get("SHELL", "")
    if "zsh" in shell_path:
        return "zsh"
    return "bash"


def _completion_line(command_name: str, shell: str) -> str:
    return f'eval "$({command_name} completion {shell})"'


def _already_installed(command_name: str, rc_path: str) -> bool:
    """Check whether the RC file already sources completion for *command_name*."""
    pattern = re.compile(
        _MARKER_RE.format(command=re.escape(command_name)),
    )
    try:
        with open(rc_path) as fh:
            return any(pattern.match(line) for line in fh)
    except FileNotFoundError:
        return False


def install_completion(command_name: str, shell: str | None = None) -> str:
    """Append the ``eval "$(cmd completion <shell>)"`` line to the shell RC.

    Returns a human-readable status message.
    """
    shell = shell or _detect_shell()
    rc_path = _SHELL_RC.get(shell)
    if rc_path is None:
        return f"Unsupported shell: {shell}"

    if _already_installed(command_name, rc_path):
        return f"{command_name} completion already installed in {rc_path}"

    line = _completion_line(command_name, shell)
    with open(rc_path, "a") as fh:
        fh.write(f"\n# {command_name} tab completion\n{line}\n")

    return f"Installed {command_name} completion in {rc_path}"


def uninstall_completion(command_name: str, shell: str | None = None) -> str:
    """Remove the completion line from the shell RC.

    Returns a human-readable status message.
    """
    shell = shell or _detect_shell()
    rc_path = _SHELL_RC.get(shell)
    if rc_path is None:
        return f"Unsupported shell: {shell}"

    try:
        with open(rc_path) as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return f"{rc_path} not found"

    pattern = re.compile(
        _MARKER_RE.format(command=re.escape(command_name)),
    )
    comment_pattern = re.compile(
        rf"^#\s*{re.escape(command_name)}\s+tab completion\s*$",
    )

    new_lines: list[str] = []
    removed = False
    skip_next_blank = False
    for line in lines:
        if comment_pattern.match(line):
            removed = True
            skip_next_blank = True
            continue
        if pattern.match(line):
            removed = True
            skip_next_blank = True
            continue
        if skip_next_blank and line.strip() == "":
            skip_next_blank = False
            continue
        skip_next_blank = False
        new_lines.append(line)

    if not removed:
        return f"No {command_name} completion found in {rc_path}"

    with open(rc_path, "w") as fh:
        fh.writelines(new_lines)

    return f"Removed {command_name} completion from {rc_path}"