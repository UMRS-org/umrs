"""IngestSource resolution for knowledge ingestion.

Resolves IngestSourceSpec into concrete paths / URIs.

Supported kinds:
  - ``local``      -> auto-detects ``local_dir`` or ``local_file``
  - ``local_dir``  -> directory path (glob-expands children)
  - ``local_file`` -> single file path
  - ``s3_uri``     -> ``s3://bucket/key`` reference
  - ``http_url``   -> remote URL (requires egress grant)
  - ``git_repo``   -> remote git URL (requires egress grant)

Remote types (``http_url``, ``git_repo``) raise ``EgressDisabledError``
unless the site allowlist (``extra.knowledge_ingest.allowed_domains``),
narrowed by any caller ``allow_domains``, is non-empty (K1.1, 2026-10-24:
the caller's list used to replace the site list).  ``s3_uri`` is treated as a reference -- no egress needed.

Kind auto-detection via ``detect_kind(uri)``.  Caller may override.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from umrs.orchestrator.site_config import SiteConfig

logger = logging.getLogger(__name__)

# Directories skipped during local_dir glob walk.
_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "_data_catalog",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".ruff_cache",
        ".idea",
        ".vscode",
        "build",
        "dist",
        ".eggs",
    }
)

# File suffixes skipped during local_dir glob walk.
_SKIP_FILE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dll",
        ".dylib",
        ".egg",
        ".whl",
    }
)


class EgressDisabledError(RuntimeError):
    """Raised when a remote ingest source is blocked by egress policy."""

    def __init__(self, source_kind: str, reason: str = "egress disabled") -> None:
        self.source_kind = source_kind  # type: ignore[assignment]
        self.reason = reason  # type: ignore[assignment]
        super().__init__(
            f"Egress disabled for {source_kind} source: {reason}. "
            f"Set 'knowledge_ingest.allowed_domains' in site config to enable."
        )


@dataclass(frozen=True)
class ResolvedFile:
    """A single file discovered by glob expansion of a local_dir source."""

    relative_path: str  # Path relative to the resolved directory
    absolute_path: str  # Absolute filesystem path


@dataclass(frozen=True)
class ResolvedSource:
    """A resolved ingest source -- path, files, or config after resolution."""

    kind: str  # local_dir, local_file, s3_uri, http_url, git_repo
    path: str | None = None  # local filesystem path (for local sources)
    uri: str | None = None  # remote URI (for http_url, git_repo, s3_uri)
    allow_egress: bool = False  # Whether egress is permitted
    allow_domains: list[str] = field(
        default_factory=list
    )  # EgressGuard domain allowlist
    resolved: bool = False  # Whether resolution succeeded

    # --- local_dir expansion ---
    files: list[ResolvedFile] = field(default_factory=list)  # Expanded files

    # --- helpers ---

    @property
    def is_local(self) -> bool:
        """Check if this is a local source (filesystem path)."""
        return self.kind in ("local_dir", "local_file")

    @property
    def is_remote(self) -> bool:
        """Check if this is a remote source (requires egress)."""
        return self.kind in ("http_url", "git_repo")

    def __repr__(self) -> str:
        parts = [f"kind={self.kind!r}", f"resolved={self.resolved}"]
        if self.path is not None:
            parts.append(f"path={self.path!r}")
        if self.uri is not None:
            parts.append(f"uri={self.uri!r}")
        if self.files:
            parts.append(f"files={len(self.files)}")
        return f"ResolvedSource({', '.join(parts)})"


class SourceResolver:
    """Resolves ingest source specifications into ResolvedSource.

    Supports local_dir, local_file, s3_uri natively.
    Remote types (http_url, git_repo) require egress permit.
    """

    def __init__(
        self,
        site_config: SiteConfig | None = None,  # type: ignore[name-defined]
    ) -> None:
        self._config = site_config

    # --- public API ---

    def resolve(
        self,
        source_kind: str,
        uri: str,
        allow_domains: list[str] | None = None,
    ) -> ResolvedSource:
        """Resolve an ingest source specification.

        Args:
            source_kind: One of 'local', 'http_url', 'git_repo', 's3_uri',
                         'local_dir', 'local_file'.
            uri: The source URI or path.
            allow_domains: Optional caller allowlist; it narrows the site
                allowlist and never widens or replaces it.

        Returns:
            ResolvedSource with resolved path/uri/files.

        Raises:
            EgressDisabledError: If remote source is blocked by egress policy.
        """
        kind = self._normalize_kind(source_kind)

        # --- remote types: egress check ---
        # K1.1(d): the site allowlist always applies. A caller's list only
        # narrows it (entries the site list does not admit are dropped); it
        # never replaces it, and no caller list means the site list.
        if kind in ("http_url", "git_repo"):
            from .egress_fetch import intersect_allowlists

            domains = intersect_allowlists(allow_domains, self._get_allowed_domains())
            if not domains:
                raise EgressDisabledError(
                    kind, "no allowlist configured (site list empty or caller list outside it)"
                )

            return ResolvedSource(
                kind=kind,
                uri=uri,
                allow_egress=True,
                allow_domains=domains,
                resolved=True,
            )

        # --- local_dir: expand files ---
        if kind == "local_dir":
            return self._resolve_local_dir(uri)

        # --- local_file: single file ---
        if kind == "local_file":
            return self._resolve_local_file(uri)

        # --- s3_uri: validate format ---
        if kind == "s3_uri":
            if self._is_valid_s3_uri(uri):
                return ResolvedSource(
                    kind="s3_uri",
                    uri=uri,
                    allow_egress=True,
                    resolved=True,
                )
            logger.warning("Invalid S3 URI: %s", uri)
            return ResolvedSource(kind="s3_uri", uri=uri, resolved=False)

        # --- unknown kind ---
        logger.warning("Unknown source kind: %s", kind)
        raise EgressDisabledError(kind, "unknown kind not permitted")

    def detect_kind(self, uri: str) -> str:
        """Auto-detect source kind from URI / path.

        Detection rules (in priority order):

        1. ``s3://...``            -> ``s3_uri``
        2. ``http://`` / ``https://`` -> ``http_url``
        3. ``git+`` / ``git://`` / ends with ``.git`` -> ``git_repo``
        4. Exists & is directory -> ``local_dir``
        5. Exists & is file      -> ``local_file``
        6. Otherwise             -> ``local_dir`` (default)
        """
        detected = self._detect_uri_kind(uri)
        if detected is not None:
            return detected
        return self._detect_path_kind(uri)

    @staticmethod
    def _detect_uri_kind(uri: str) -> str | None:
        """Detect kind from URI scheme/pattern.  Returns None for local paths."""
        if uri.startswith("s3://"):
            return "s3_uri"
        if uri.endswith(".git"):
            return "git_repo"
        if uri.startswith("http://") or uri.startswith("https://"):
            return "http_url"
        if uri.startswith("git+") or uri.startswith("git://"):
            return "git_repo"
        return None

    @staticmethod
    def _detect_path_kind(uri: str) -> str:
        """Detect kind from filesystem path.  Default local_dir."""
        path = Path(uri)
        if path.exists():
            if path.is_dir():
                return "local_dir"
            if path.is_file():
                return "local_file"
        return "local_dir"

    # --- private helpers ---

    @staticmethod
    def _normalize_kind(source_kind: str) -> str:
        """Map user-facing kind to canonical kind."""
        if source_kind == "local":
            return "local_dir"
        return source_kind

    @staticmethod
    def _is_valid_s3_uri(uri: str) -> bool:
        """Validate an S3 URI: ``s3://bucket/key``."""
        if not uri.startswith("s3://"):
            return False
        remainder = uri[5:]
        bucket = remainder.split("/", 1)[0]
        return bool(bucket)

    def _get_allowed_domains(self) -> list[str]:
        """Read the site-config knowledge ingest allowlist.

        Returns empty list when config is absent or allowlist is empty.
        """
        if self._config is None:
            return []

        try:
            # SiteConfig stores extra fields in self.extra dict.
            extra = getattr(self._config, "extra", {})
            ki = extra.get("knowledge_ingest", {})
            domains = ki.get("allowed_domains", None)
            if isinstance(domains, list):
                return list(domains)
        except Exception:
            logger.debug("Failed to read knowledge_ingest allowlist", exc_info=True)

        return []

    # --- local resolution ---

    def _resolve_local_dir(
        self,
        uri: str,
    ) -> ResolvedSource:
        """Resolve a local directory source -- expands matching files."""
        root = Path(uri)

        if not root.is_dir():
            logger.warning("Local directory not found: %s", uri)
            return ResolvedSource(kind="local_dir", path=str(uri), resolved=False)

        files = self._glob_files(root)
        return ResolvedSource(
            kind="local_dir",
            path=str(root.resolve()),
            files=files,
            resolved=True,
        )

    def _resolve_local_file(
        self,
        uri: str,
    ) -> ResolvedSource:
        """Resolve a single local file source."""
        path = Path(uri)

        if not path.is_file():
            logger.warning("Local file not found: %s", uri)
            return ResolvedSource(kind="local_file", path=str(uri), resolved=False)

        return ResolvedSource(
            kind="local_file",
            path=str(path.resolve()),
            resolved=True,
        )

    @staticmethod
    def _glob_files(root: Path) -> list[ResolvedFile]:
        """Recursively glob files under *root*, skipping dot-dir and binary suffixes."""
        results: list[ResolvedFile] = []

        for dirpath, dirnames, filenames in os.walk(root):
            SourceResolver._prune_dirs(dirnames)

            for fname in filenames:
                if not SourceResolver._accept_file(fname):
                    continue

                full = Path(dirpath) / fname
                rel = SourceResolver._relative(full, root)
                if rel is None:
                    continue

                results.append(
                    ResolvedFile(
                        relative_path=str(rel),
                        absolute_path=str(full.resolve()),
                    )
                )

        results.sort(key=lambda f: f.relative_path)
        return results

    @staticmethod
    def _prune_dirs(dirnames: list[str]) -> None:
        """Prune ignored dirs from *dirnames* in-place."""
        skip_dirs = set(_SKIP_DIR_NAMES)
        dirnames[:] = [
            d
            for d in dirnames
            if d not in skip_dirs
            and not any(fnmatch.fnmatch(d, pat) for pat in skip_dirs)
        ]
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

    @staticmethod
    def _accept_file(fname: str) -> bool:
        """Return True if file should be included."""
        if fname.startswith("."):
            return False
        return not any(fname.endswith(suf) for suf in _SKIP_FILE_SUFFIXES)

    @staticmethod
    def _relative(full: Path, root: Path) -> Path | None:
        """Return relative path or None if outside root."""
        try:
            return full.relative_to(root)
        except ValueError:
            return None


# --- module-level convenience ---


def resolve_ingest_source(
    source_kind: str,
    uri: str,
    site_config: SiteConfig | None = None,
    allow_domains: list[str] | None = None,
) -> ResolvedSource:
    """Resolve an ingest source specification (convenience wrapper).

    Args:
        source_kind: Source kind string.
        uri: Source URI or path.
        site_config: Optional site config for egress policy.
        allow_domains: Optional caller allowlist (narrows the site list).

    Returns:
        ResolvedSource.

    Raises:
        EgressDisabledError: If remote source is blocked.
    """
    resolver = SourceResolver(site_config=site_config)
    return resolver.resolve(source_kind, uri, allow_domains=allow_domains)
