"""Code parser — extracts structure from Python, C++, CUDA source files.

Code is a first-class scientific artifact. This parser extracts:
  - Modules (file-level module name)
  - APIs (functions, classes, methods)
  - Algorithms (heuristic: functions with docstrings containing "algorithm")
  - Dependencies (import statements)
  - Parameters (function signatures)
  - Kernels (CUDA kernel functions)
  - Execution flow (main block, entry points)

The parser is stdlib-only for Python (uses ``ast`` module) and falls back
to regex-based extraction for C++/CUDA. This avoids hard dependencies on
tree-sitter or similar while still providing useful structure.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """Information about a function or method."""

    name: str
    signature: str
    docstring: str = ""
    line_start: int = 0
    line_end: int = 0
    is_method: bool = False
    class_name: str = ""
    is_kernel: bool = False  # CUDA kernel
    parameters: list[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Information about a class."""

    name: str
    docstring: str = ""
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionInfo] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0


@dataclass
class CodeParseResult:
    """Structured result of parsing a code file."""

    filename: str
    language: str = "python"  # python | cpp | cuda | unknown
    # Extracted structure
    modules: list[str] = field(default_factory=list)
    apis: list[str] = field(default_factory=list)  # function/class names
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    algorithms: list[str] = field(default_factory=list)  # heuristic
    dependencies: list[str] = field(default_factory=list)  # imports
    parameters: list[str] = field(default_factory=list)  # function params
    kernels: list[FunctionInfo] = field(default_factory=list)  # CUDA kernels
    execution_flow: str = ""  # main block / entry points
    # Extracted text (full source code)
    extracted_text: str = ""
    # Summary
    summary: str = ""
    # Tags
    tags: list[str] = field(default_factory=list)
    # Error
    error: str | None = None


def parse_code(filename: str, content: bytes) -> CodeParseResult:
    """Parse a code file (Python, C++, CUDA).

    Args:
        filename: The filename
        content: The raw bytes of the code file

    Returns:
        A ``CodeParseResult`` with structured code content.
    """
    result = CodeParseResult(filename=filename)
    result.language = _detect_language(filename)
    source, decode_error = _decode_source(content)
    if decode_error:
        result.error = decode_error
        return result

    result.extracted_text = source
    result.modules = _module_name_from_filename(filename)

    parser = _LANGUAGE_PARSERS.get(result.language)
    if parser is None:
        # Unknown language — just use the raw source
        result.summary = f"Unknown code file {filename}"
        result.tags = [result.language]
        return result
    parser(source, result)

    # Build summary
    result.summary = _build_summary(result)
    result.tags = _extract_tags(result)

    return result


def _detect_language(filename: str) -> str:
    """Detect the supported language from a filename."""
    lower = filename.lower()
    if lower.endswith(".py"):
        return "python"
    if lower.endswith((".cu", ".cuh")):
        return "cuda"
    if lower.endswith((".cpp", ".cc", ".cxx", ".hpp", ".h", ".c")):
        return "cpp"
    return "unknown"


def _decode_source(content: bytes) -> tuple[str, str | None]:
    """Decode source bytes using the parser's historical fallback order."""
    try:
        return content.decode("utf-8"), None
    except UnicodeDecodeError:
        try:
            return content.decode("latin-1"), None
        except UnicodeDecodeError as exc:
            return "", f"Failed to decode source: {exc}"


def _parse_python(source: str, result: CodeParseResult) -> None:
    """Parse Python source code using the ``ast`` module."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result.error = f"Python syntax error: {exc}"
        return

    _collect_python_dependencies(tree, result)
    for node in ast.iter_child_nodes(tree):
        handler = _PYTHON_NODE_HANDLERS.get(type(node))
        if handler:
            handler(node, source, result)
    _set_python_execution_flow(tree, result)


def _collect_python_dependencies(
    tree: ast.Module,
    result: CodeParseResult,
) -> None:
    """Collect imports from the complete Python syntax tree."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.dependencies.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result.dependencies.append(node.module)


def _add_python_function(
    node: ast.AST,
    source: str,
    result: CodeParseResult,
) -> None:
    """Add a top-level sync or async function to the result."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return
    func = _extract_function_info(node, source, is_method=False)
    result.functions.append(func)
    result.apis.append(func.name)
    result.parameters.extend(func.parameters)
    _record_algorithm(func, result)


def _add_python_class(
    node: ast.AST,
    source: str,
    result: CodeParseResult,
) -> None:
    """Add a top-level class and its method metadata to the result."""
    if not isinstance(node, ast.ClassDef):
        return
    cls = _extract_class_info(node, source)
    result.classes.append(cls)
    result.apis.append(cls.name)
    for method in cls.methods:
        result.apis.append(f"{cls.name}.{method.name}")
        result.parameters.extend(method.parameters)
        _record_algorithm(method, result, prefix=f"{cls.name}.")


_PYTHON_NODE_HANDLERS = {
    ast.FunctionDef: _add_python_function,
    ast.AsyncFunctionDef: _add_python_function,
    ast.ClassDef: _add_python_class,
}


def _record_algorithm(
    func: FunctionInfo,
    result: CodeParseResult,
    prefix: str = "",
) -> None:
    """Record functions whose documentation identifies an algorithm."""
    if func.docstring and "algorithm" in func.docstring.lower():
        result.algorithms.append(f"{prefix}{func.name}")


def _set_python_execution_flow(
    tree: ast.Module,
    result: CodeParseResult,
) -> None:
    """Classify a Python file as a main program or library."""
    for node in ast.iter_child_nodes(tree):
        if _is_python_main_guard(node):
            result.execution_flow = "has_main_block"
            break
    else:
        if result.functions or result.classes:
            result.execution_flow = "library"


def _is_python_main_guard(node: ast.AST) -> bool:
    """Recognize the parser's intentionally permissive main-block shape."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
    )


def _extract_function_info(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
    is_method: bool = False,
    class_name: str = "",
) -> FunctionInfo:
    """Extract function information from an AST node."""
    # Build signature
    args = [arg.arg for arg in node.args.args]
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    signature = f"def {node.name}({', '.join(args)})"

    return FunctionInfo(
        name=node.name,
        signature=signature,
        docstring=_docstring_of(node),
        line_start=node.lineno,
        line_end=node.end_lineno or node.lineno,
        is_method=is_method,
        class_name=class_name,
        parameters=args,
    )


def _extract_class_info(node: ast.ClassDef, source: str) -> ClassInfo:
    """Extract class information from an AST node."""
    return ClassInfo(
        name=node.name,
        docstring=_docstring_of(node),
        bases=_class_bases(node),
        methods=_class_methods(node, source),
        line_start=node.lineno,
        line_end=node.end_lineno or node.lineno,
    )


def _docstring_of(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> str:
    """Return a node's literal leading docstring without normalization."""
    if not node.body or not isinstance(node.body[0], ast.Expr):
        return ""
    value = node.body[0].value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return ""


def _class_bases(node: ast.ClassDef) -> list[str]:
    """Return supported class base names in source order."""
    bases: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(_get_attribute_name(base))
    return bases


def _class_methods(node: ast.ClassDef, source: str) -> list[FunctionInfo]:
    """Return direct sync and async methods in source order."""
    return [
        _extract_function_info(
            child,
            source,
            is_method=True,
            class_name=node.name,
        )
        for child in ast.iter_child_nodes(node)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _get_attribute_name(node: ast.Attribute) -> str:
    """Get the full name of an attribute (e.g., 'module.Class')."""
    parts = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _parse_cpp_cuda(source: str, result: CodeParseResult) -> None:
    """Parse C++/CUDA source code using regex-based extraction."""
    _collect_cuda_kernels(source, result)
    _collect_cpp_functions(source, result)
    _collect_cpp_classes(source, result)
    _collect_cpp_dependencies(source, result)
    _set_cpp_execution_flow(source, result)


def _collect_cuda_kernels(source: str, result: CodeParseResult) -> None:
    """Collect CUDA ``__global__`` functions."""
    kernel_pattern = re.compile(
        r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)",
        re.MULTILINE,
    )
    for match in kernel_pattern.finditer(source):
        name = match.group(1)
        params = match.group(2)
        param_list = _split_cpp_parameters(params)
        result.kernels.append(
            FunctionInfo(
                name=name,
                signature=f"__global__ void {name}({params})",
                line_start=_line_number(source, match.start()),
                line_end=_line_number(source, match.end()),
                is_kernel=True,
                parameters=param_list,
            )
        )
        result.apis.append(name)
        result.parameters.extend(param_list)


def _collect_cpp_functions(source: str, result: CodeParseResult) -> None:
    """Collect regex-recognized C/C++ function definitions."""
    func_pattern = re.compile(
        r"^(?:static\s+|inline\s+|extern\s+)*"
        r"(?:[\w:*&<>]+\s+)+"
        r"(\w+)\s*\(([^)]*)\)\s*(?:\{|\n\s*\{)",
        re.MULTILINE,
    )
    for match in func_pattern.finditer(source):
        name = match.group(1)
        if name in ("if", "while", "for", "switch", "return", "sizeof"):
            continue
        params = match.group(2)
        param_list = _split_cpp_parameters(params)
        result.functions.append(
            FunctionInfo(
                name=name,
                signature=f"{name}({params})",
                line_start=_line_number(source, match.start()),
                line_end=_line_number(source, match.end()),
                parameters=param_list,
            )
        )
        result.apis.append(name)
        result.parameters.extend(param_list)


def _collect_cpp_classes(source: str, result: CodeParseResult) -> None:
    """Collect regex-recognized C++ class definitions."""
    class_pattern = re.compile(
        r"class\s+(\w+)\s*(?::\s*[^{]+)?\s*\{",
        re.MULTILINE,
    )
    for match in class_pattern.finditer(source):
        name = match.group(1)
        result.classes.append(
            ClassInfo(
                name=name,
                line_start=_line_number(source, match.start()),
                line_end=_line_number(source, match.end()),
            )
        )
        result.apis.append(name)


def _collect_cpp_dependencies(source: str, result: CodeParseResult) -> None:
    """Collect C/C++ include targets."""
    include_pattern = re.compile(
        r'#include\s+[<"]([^>"]+)[>"]',
        re.MULTILINE,
    )
    for match in include_pattern.finditer(source):
        result.dependencies.append(match.group(1))


def _set_cpp_execution_flow(source: str, result: CodeParseResult) -> None:
    """Classify C/C++ code as a main program or library."""
    main_pattern = re.compile(
        r"int\s+main\s*\(",
        re.MULTILINE,
    )
    if main_pattern.search(source):
        result.execution_flow = "has_main"
    else:
        if result.functions or result.classes:
            result.execution_flow = "library"


def _split_cpp_parameters(params: str) -> list[str]:
    """Split the simple parameter syntax supported by the regex parser."""
    return [param.strip() for param in params.split(",") if param.strip()]


def _line_number(source: str, offset: int) -> int:
    """Return the one-based line number for a source offset."""
    return source.count("\n", 0, offset) + 1


_LANGUAGE_PARSERS = {
    "python": _parse_python,
    "cpp": _parse_cpp_cuda,
    "cuda": _parse_cpp_cuda,
}


def _module_name_from_filename(filename: str) -> list[str]:
    """Extract module name from filename."""
    # Strip path and extension
    import os

    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    return [name] if name else []


def _build_summary(result: CodeParseResult) -> str:
    """Build a summary of the code file."""
    parts = [
        f"{result.language} file: {result.filename}",
        f"  Functions: {len(result.functions)}",
        f"  Classes: {len(result.classes)}",
    ]
    optional_lines = [
        (result.kernels, f"  CUDA kernels: {len(result.kernels)}"),
        (result.dependencies, f"  Imports: {len(result.dependencies)}"),
        (result.execution_flow, f"  Flow: {result.execution_flow}"),
    ]
    parts.extend(line for value, line in optional_lines if value)
    _append_top_names(parts, "functions", [item.name for item in result.functions[:5]])
    _append_top_names(parts, "classes", [item.name for item in result.classes[:5]])
    _append_top_names(parts, "kernels", [item.name for item in result.kernels[:5]])

    return "\n".join(parts)


def _append_top_names(parts: list[str], label: str, names: list[str]) -> None:
    """Append one non-empty top-entity summary line."""
    if names:
        parts.append(f"  Top {label}: {', '.join(names)}")


def _extract_tags(result: CodeParseResult) -> list[str]:
    """Extract tags from the code parse result."""
    tags: set[str] = set()

    tags.add(result.language)
    tags.add(f"functions:{len(result.functions)}")
    tags.add(f"classes:{len(result.classes)}")
    if result.kernels:
        tags.add(f"kernels:{len(result.kernels)}")
    if result.dependencies:
        tags.add(f"imports:{len(result.dependencies)}")
    if result.execution_flow:
        tags.add(result.execution_flow)

    # Add function/class names as tags
    for func in result.functions[:10]:
        tags.add(f"fn:{func.name}")
    for cls in result.classes[:5]:
        tags.add(f"class:{cls.name}")
    for kernel in result.kernels[:5]:
        tags.add(f"kernel:{kernel.name}")

    return sorted(tags)
