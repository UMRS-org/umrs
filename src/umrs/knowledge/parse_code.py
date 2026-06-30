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
from typing import Any

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

    # Detect language from extension
    lower = filename.lower()
    if lower.endswith(".py"):
        result.language = "python"
    elif lower.endswith((".cu", ".cuh")):
        result.language = "cuda"
    elif lower.endswith((".cpp", ".cc", ".cxx", ".hpp", ".h", ".c")):
        result.language = "cpp"
    else:
        result.language = "unknown"

    # Decode content
    try:
        source = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            source = content.decode("latin-1")
        except UnicodeDecodeError as exc:
            result.error = f"Failed to decode source: {exc}"
            return result

    result.extracted_text = source
    result.modules = [_module_name_from_filename(filename)]

    # Parse based on language
    if result.language == "python":
        _parse_python(source, result)
    elif result.language in ("cpp", "cuda"):
        _parse_cpp_cuda(source, result)
    else:
        # Unknown language — just use the raw source
        result.summary = f"Unknown code file {filename}"
        result.tags = [result.language]
        return result

    # Build summary
    result.summary = _build_summary(result)
    result.tags = _extract_tags(result)

    return result


def _parse_python(source: str, result: CodeParseResult) -> None:
    """Parse Python source code using the ``ast`` module."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result.error = f"Python syntax error: {exc}"
        return

    # Extract imports (dependencies)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.dependencies.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result.dependencies.append(node.module)

    # Extract functions, classes, methods
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            func = _extract_function_info(node, source, is_method=False)
            result.functions.append(func)
            result.apis.append(func.name)
            result.parameters.extend(func.parameters)
            # Heuristic: function with "algorithm" in docstring
            if func.docstring and "algorithm" in func.docstring.lower():
                result.algorithms.append(func.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            func = _extract_function_info(node, source, is_method=False)
            result.functions.append(func)
            result.apis.append(func.name)
            result.parameters.extend(func.parameters)
            if func.docstring and "algorithm" in func.docstring.lower():
                result.algorithms.append(func.name)
        elif isinstance(node, ast.ClassDef):
            cls = _extract_class_info(node, source)
            result.classes.append(cls)
            result.apis.append(cls.name)
            for method in cls.methods:
                result.parameters.extend(method.parameters)
                # Heuristic: method with "algorithm" in docstring
                if method.docstring and "algorithm" in method.docstring.lower():
                    result.algorithms.append(f"{cls.name}.{method.name}")

    # Extract execution flow (main block)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.If):
            # Check for `if __name__ == "__main__":`
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                result.execution_flow = "has_main_block"
                break
    else:
        if result.functions or result.classes:
            result.execution_flow = "library"


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

    # Extract docstring
    docstring = ""
    if (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
        docstring = node.body[0].value.value

    return FunctionInfo(
        name=node.name,
        signature=signature,
        docstring=docstring,
        line_start=node.lineno,
        line_end=node.end_lineno or node.lineno,
        is_method=is_method,
        class_name=class_name,
        parameters=args,
    )


def _extract_class_info(node: ast.ClassDef, source: str) -> ClassInfo:
    """Extract class information from an AST node."""
    # Extract bases
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(_get_attribute_name(base))

    # Extract methods
    methods = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method = _extract_function_info(
                child,
                source,
                is_method=True,
                class_name=node.name,
            )
            methods.append(method)

    # Extract docstring
    docstring = ""
    if (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
        docstring = node.body[0].value.value

    return ClassInfo(
        name=node.name,
        docstring=docstring,
        bases=bases,
        methods=methods,
        line_start=node.lineno,
        line_end=node.end_lineno or node.lineno,
    )


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
    # Extract CUDA kernel functions (__global__ functions)
    kernel_pattern = re.compile(
        r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)",
        re.MULTILINE,
    )
    for match in kernel_pattern.finditer(source):
        name = match.group(1)
        params = match.group(2)
        param_list = [p.strip() for p in params.split(",") if p.strip()]
        result.kernels.append(
            FunctionInfo(
                name=name,
                signature=f"__global__ void {name}({params})",
                line_start=source.count("\n", 0, match.start()) + 1,
                line_end=source.count("\n", 0, match.end()) + 1,
                is_kernel=True,
                parameters=param_list,
            )
        )
        result.apis.append(name)
        result.parameters.extend(param_list)

    # Extract function definitions
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
        param_list = [p.strip() for p in params.split(",") if p.strip()]
        result.functions.append(
            FunctionInfo(
                name=name,
                signature=f"{name}({params})",
                line_start=source.count("\n", 0, match.start()) + 1,
                line_end=source.count("\n", 0, match.end()) + 1,
                parameters=param_list,
            )
        )
        result.apis.append(name)
        result.parameters.extend(param_list)

    # Extract class definitions
    class_pattern = re.compile(
        r"class\s+(\w+)\s*(?::\s*[^{]+)?\s*\{",
        re.MULTILINE,
    )
    for match in class_pattern.finditer(source):
        name = match.group(1)
        result.classes.append(
            ClassInfo(
                name=name,
                line_start=source.count("\n", 0, match.start()) + 1,
                line_end=source.count("\n", 0, match.end()) + 1,
            )
        )
        result.apis.append(name)

    # Extract includes (dependencies)
    include_pattern = re.compile(
        r'#include\s+[<"]([^>"]+)[>"]',
        re.MULTILINE,
    )
    for match in include_pattern.finditer(source):
        result.dependencies.append(match.group(1))

    # Extract execution flow (main function)
    main_pattern = re.compile(
        r"int\s+main\s*\(",
        re.MULTILINE,
    )
    if main_pattern.search(source):
        result.execution_flow = "has_main"
    else:
        if result.functions or result.classes:
            result.execution_flow = "library"


def _module_name_from_filename(filename: str) -> list[str]:
    """Extract module name from filename."""
    # Strip path and extension
    import os

    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    return [name] if name else []


def _build_summary(result: CodeParseResult) -> str:
    """Build a summary of the code file."""
    parts: list[str] = []
    parts.append(f"{result.language} file: {result.filename}")
    parts.append(f"  Functions: {len(result.functions)}")
    parts.append(f"  Classes: {len(result.classes)}")
    if result.kernels:
        parts.append(f"  CUDA kernels: {len(result.kernels)}")
    if result.dependencies:
        parts.append(f"  Imports: {len(result.dependencies)}")
    if result.execution_flow:
        parts.append(f"  Flow: {result.execution_flow}")

    # List top functions/classes
    if result.functions:
        names = [f.name for f in result.functions[:5]]
        parts.append(f"  Top functions: {', '.join(names)}")
    if result.classes:
        names = [c.name for c in result.classes[:5]]
        parts.append(f"  Top classes: {', '.join(names)}")
    if result.kernels:
        names = [k.name for k in result.kernels[:5]]
        parts.append(f"  Top kernels: {', '.join(names)}")

    return "\n".join(parts)


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
