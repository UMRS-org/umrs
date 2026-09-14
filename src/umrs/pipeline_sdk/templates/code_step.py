"""``CodeStep`` — built-in step that parses source code files.

The step reads a code file from S3, parses it using the appropriate
language parser (Python AST, JavaScript tree-sitter, etc.), and
extracts functions, classes, and top-level statements as separate
chunks.

Supported languages:
    - Python (``.py``) — uses :mod:`ast` module
    - JavaScript/TypeScript (``.js``, ``.ts``) — regex-based extraction
    - Generic (any extension) — line-based chunking

Example ``workflow.json`` snippet::

    {
      "name": "code-parse",
      "entrypoint": "umrs.pipeline_sdk.templates.code_step:CodeStep",
      "config": {"language": "python", "chunk_size": 100}
    }
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

from umrs.pipeline_sdk.step import Step, StepContext, StepResult

logger = logging.getLogger(__name__)

# Language detection from file extension
_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".sh": "shell",
    ".bash": "shell",
    ".r": "r",
    ".R": "r",
    ".m": "matlab",
    ".jl": "julia",
    ".rb": "ruby",
    ".sql": "sql",
    ".scala": "scala",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".dart": "dart",
    ".lua": "lua",
    ".pl": "perl",
    ".pm": "perl",
    ".t": "perl",
    ".ps1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".com": "batch",
    ".fish": "shell",
    ".zsh": "shell",
    ".php": "php",
    ".phtml": "php",
    ".php3": "php",
    ".php4": "php",
    ".php5": "php",
    ".php7": "php",
    ".phps": "php",
    ".pht": "php",
    ".thtml": "php",
    ".pyp": "python",
    ".pyc": "python",
    ".pyo": "python",
    ".pyd": "python",
    ".pyw": "python",
    ".pyx": "python",
    ".pxi": "python",
    ".pxd": "python",
    ".pyz": "python",
    ".pywz": "python",
    ".egg": "python",
    ".whl": "python",
    ".zip": "archive",
    ".tar": "archive",
    ".gz": "archive",
    ".bz2": "archive",
    ".7z": "archive",
    ".rar": "archive",
    ".iso": "archive",
    ".dmg": "archive",
    ".pkg": "archive",
    ".deb": "archive",
    ".rpm": "archive",
    ".msi": "archive",
    ".apk": "archive",
    ".jar": "java",
    ".war": "java",
    ".ear": "java",
    ".class": "java",
}


class CodeStep(Step):
    """Parse source code files into semantic chunks."""

    name = "code"

    def setup(self, config: dict[str, Any]) -> None:
        self.language: str | None = config.get("language")
        self.chunk_size: int = int(config.get("chunk_size", 100))
        self.include_imports: bool = bool(config.get("include_imports", True))
        self.include_docstrings: bool = bool(config.get("include_docstrings", True))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, ctx: StepContext) -> StepResult:
        out_keys: list[str] = []
        metadata_list: list[dict[str, Any]] = []

        for key in ctx.input_keys:
            try:
                raw = ctx.download_bytes(key)
                source = raw.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning("CodeStep: not UTF-8: %s", key)
                continue
            except Exception as exc:
                logger.warning("CodeStep: failed to read %s: %s", key, exc)
                continue

            language = self._detect_language(key)
            chunks = self._extract_chunks(source, language, key)

            if not chunks:
                continue

            out_key = self._write_chunks(chunks, key, ctx)
            out_keys.append(out_key)
            metadata_list.append(
                {
                    "source_key": key,
                    "language": language,
                    "num_chunks": len(chunks),
                }
            )

        return StepResult(
            output_keys=out_keys,
            metadata=metadata_list,
            extra={"step": "code"},
        )

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        import os

        if self.language:
            return self.language

        ext = os.path.splitext(file_path)[1].lower()
        return _LANGUAGE_MAP.get(ext, "generic")

    # ------------------------------------------------------------------
    # Chunk extraction
    # ------------------------------------------------------------------

    def _extract_chunks(
        self,
        source: str,
        language: str,
        source_key: str,
    ) -> list[dict[str, Any]]:
        """Extract semantic chunks from source code."""
        if language == "python":
            return self._extract_python(source, source_key)
        if language in ("javascript", "typescript"):
            return self._extract_javascript(source, language, source_key)
        return self._extract_generic(source, source_key)

    def _extract_python(
        self,
        source: str,
        source_key: str,
    ) -> list[dict[str, Any]]:
        """Extract chunks from Python source using AST."""
        chunks: list[dict[str, Any]] = []
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            logger.warning("CodeStep: Python parse error in %s: %s", source_key, exc)
            return self._extract_generic(source, source_key)

        lines = source.splitlines(keepends=True)

        # Extract imports
        if self.include_imports:
            import_nodes = [
                n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
            ]
            for node in import_nodes:
                start = node.lineno - 1
                end = getattr(node, "end_lineno", node.lineno)
                text = "".join(lines[start:end])
                chunks.append(
                    {
                        "type": "import",
                        "name": ast.get_source_segment(source, node) or text,
                        "text": text,
                        "start_line": start + 1,
                        "end_line": end,
                        "source_key": source_key,
                    }
                )

        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._make_python_chunk(
                    node, "function", source, lines, source_key
                )
                chunks.append(chunk)
            elif isinstance(node, ast.ClassDef):
                chunk = self._make_python_chunk(
                    node, "class", source, lines, source_key
                )
                chunks.append(chunk)

        return chunks

    def _make_python_chunk(
        self,
        node: ast.AST,
        chunk_type: str,
        source: str,
        lines: list[str],
        source_key: str,
    ) -> dict[str, Any]:
        """Create a chunk dict from a Python AST node."""
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)
        text = "".join(lines[start:end])

        docstring = None
        if self.include_docstrings and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            ds = ast.get_docstring(node)
            if ds:
                docstring = ds

        return {
            "type": chunk_type,
            "name": getattr(node, "name", "<anonymous>"),
            "text": text,
            "docstring": docstring,
            "start_line": start + 1,
            "end_line": end,
            "source_key": source_key,
        }

    def _extract_javascript(
        self,
        source: str,
        language: str,
        source_key: str,
    ) -> list[dict[str, Any]]:
        """Extract chunks from JavaScript/TypeScript using regex."""
        chunks: list[dict[str, Any]] = []

        # Match function declarations, arrow functions, class declarations
        patterns = [
            (r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", "function"),
            (
                r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(",
                "function",
            ),
            (r"(?:export\s+)?class\s+(\w+)", "class"),
            (r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=", "variable"),
        ]

        lines = source.splitlines()
        for pattern, chunk_type in patterns:
            for match in re.finditer(pattern, source):
                name = match.group(1) if match.groups() else "<anonymous>"
                # Find the line number
                pos = match.start()
                line_num = source[:pos].count("\n") + 1

                # Try to find the end of the block
                start_idx = line_num - 1
                end_idx = self._find_block_end(lines, start_idx)
                text = "\n".join(lines[start_idx:end_idx])

                chunks.append(
                    {
                        "type": chunk_type,
                        "name": name,
                        "text": text,
                        "start_line": start_idx + 1,
                        "end_line": end_idx,
                        "source_key": source_key,
                    }
                )

        return chunks

    def _find_block_end(self, lines: list[str], start: int) -> int:
        """Find the end line of a code block (matching braces)."""
        depth = 0
        found_open = False
        for i in range(start, len(lines)):
            for char in lines[i]:
                if char == "{":
                    depth += 1
                    found_open = True
                elif char == "}":
                    depth -= 1
                    if found_open and depth == 0:
                        return i + 1
        return len(lines)

    def _extract_generic(
        self,
        source: str,
        source_key: str,
    ) -> list[dict[str, Any]]:
        """Extract chunks from any source using line-based chunking."""
        lines = source.splitlines(keepends=True)
        chunks: list[dict[str, Any]] = []

        for i in range(0, len(lines), self.chunk_size):
            chunk_lines = lines[i : i + self.chunk_size]
            text = "".join(chunk_lines)
            chunks.append(
                {
                    "type": "lines",
                    "name": f"lines_{i}_{i + len(chunk_lines)}",
                    "text": text,
                    "start_line": i + 1,
                    "end_line": i + len(chunk_lines),
                    "source_key": source_key,
                }
            )

        return chunks

    def _write_chunks(
        self,
        chunks: list[dict[str, Any]],
        source_key: str,
        ctx: StepContext,
    ) -> str:
        """Write chunks as JSON to S3."""
        import json
        import os

        base = os.path.splitext(source_key)[0]
        out_key = f"code/{base}.json"
        data = json.dumps(
            {
                "source": source_key,
                "chunks": chunks,
            },
            indent=2,
        ).encode("utf-8")
        ctx.upload_bytes(out_key, data, content_type="application/json")
        return out_key
