"""Tests for umrs/knowledge/sources.py — IngestSource resolution."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from umrs.knowledge.sources import (
    EgressDisabledError,
    ResolvedFile,
    ResolvedSource,
    SourceResolver,
    resolve_ingest_source,
)

if TYPE_CHECKING:
    pass


class TestDetectKind:
    """detect_kind — auto-detection table unit tests."""

    def test_s3_uri(self) -> None:
        assert SourceResolver().detect_kind("s3://my-bucket/data") == "s3_uri"

    def test_s3_uri_with_key(self) -> None:
        assert SourceResolver().detect_kind("s3://bucket/path/to/file.parquet") == "s3_uri"

    def test_http_url(self) -> None:
        assert SourceResolver().detect_kind("http://example.com/data.csv") == "http_url"

    def test_https_url(self) -> None:
        assert SourceResolver().detect_kind("https://example.com/data.csv") == "http_url"

    def test_git_plus_url(self) -> None:
        assert SourceResolver().detect_kind("git+https://github.com/org/repo") == "git_repo"

    def test_git_protocol_url(self) -> None:
        assert SourceResolver().detect_kind("git://github.com/org/repo.git") == "git_repo"

    def test_git_ssh_url(self) -> None:
        assert SourceResolver().detect_kind("git@github.com:org/repo.git") == "git_repo"

    def test_dot_git_extension(self) -> None:
        assert SourceResolver().detect_kind("https://github.com/org/repo.git") == "git_repo"

    def test_local_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "data"
        d.mkdir()
        assert SourceResolver().detect_kind(str(d)) == "local_dir"

    def test_local_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("hello")
        assert SourceResolver().detect_kind(str(f)) == "local_file"

    def test_local_dir_default_for_absent_path(self) -> None:
        assert SourceResolver().detect_kind("/nonexistent/path") == "local_dir"

    def test_local_dir_default_for_nonexistent_relative(self) -> None:
        assert SourceResolver().detect_kind("some/unknown/dir") == "local_dir"


class TestResolveLocalDir:
    """local_dir resolution with glob expansion."""

    def test_existing_dir_resolves(self, tmp_path: Path) -> None:
        resolver = SourceResolver()
        d = tmp_path / "data"
        d.mkdir()
        (d / "file.txt").write_text("data")
        result = resolver.resolve("local_dir", str(d))
        assert result.resolved is True
        assert result.kind == "local_dir"
        assert result.path == str(d.resolve())

    def test_glob_expands_files(self, tmp_path: Path) -> None:
        """Manifest expansion: local_dir discovers files recursively."""
        resolver = SourceResolver()
        d = tmp_path / "data"
        d.mkdir()
        (d / "readme.md").write_text("# README")
        (d / "notes.txt").write_text("notes")
        sub = d / "subdir"
        sub.mkdir()
        (sub / "code.py").write_text("print(1)")
        (sub / ".hidden").write_text("hidden")

        result = resolver.resolve("local_dir", str(d))
        assert result.resolved is True
        assert len(result.files) == 3

        paths = {f.relative_path for f in result.files}
        assert "readme.md" in paths
        assert "notes.txt" in paths
        assert "subdir/code.py" in paths

    def test_skips_dot_dirs(self, tmp_path: Path) -> None:
        """Hidden directories are skipped during glob."""
        resolver = SourceResolver()
        d = tmp_path / "data"
        d.mkdir()
        (d / "visible.txt").write_text("visible")
        hidden = d / ".hidden_dir"
        hidden.mkdir()
        (hidden / "secret.txt").write_text("secret")
        (d / ".gitkeep").write_text("")  # hidden file also skipped

        result = resolver.resolve("local_dir", str(d))
        paths = {f.relative_path for f in result.files}
        assert "visible.txt" in paths
        assert "secret.txt" not in paths  # inside .hidden_dir
        assert ".gitkeep" not in paths  # starts with .

    def test_skips_cached_dirs(self, tmp_path: Path) -> None:
        """Standard cache dirs are skipped."""
        resolver = SourceResolver()
        d = tmp_path / "data"
        d.mkdir()
        (d / "main.py").write_text("code")

        pycache = d / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-312.pyc").write_bytes(b"bytecode")

        dot_venv = d / ".venv"
        dot_venv.mkdir()
        (dot_venv / "py").write_bytes(b"venv")

        result = resolver.resolve("local_dir", str(d))
        paths = {f.relative_path for f in result.files}
        assert "main.py" in paths
        assert "__pycache__/main.cpython-312.pyc" not in paths
        assert ".venv/py" not in paths

    def test_skips_binary_suffixes(self, tmp_path: Path) -> None:
        """Binary file suffixes are excluded from expansion."""
        resolver = SourceResolver()
        d = tmp_path / "data"
        d.mkdir()
        (d / "main.py").write_text("code")
        (d / "main.pyc").write_bytes(b"bytecode")
        (d / "lib.so").write_bytes(b"binary")

        result = resolver.resolve("local_dir", str(d))
        paths = {f.relative_path for f in result.files}
        assert "main.py" in paths
        assert "main.pyc" not in paths
        assert "lib.so" not in paths

    def test_nonexistent_dir_not_resolved(self, tmp_path: Path) -> None:
        resolver = SourceResolver()
        result = resolver.resolve("local_dir", str(tmp_path / "nonexistent"))
        assert result.resolved is False
        assert result.kind == "local_dir"

    def test_local_alias_map_to_local_dir(self, tmp_path: Path) -> None:
        """Kind 'local' normalizes to 'local_dir'."""
        d = tmp_path / "data"
        d.mkdir()
        result = SourceResolver().resolve("local", str(d))
        assert result.kind == "local_dir"
        assert result.resolved is True


class TestResolveLocalFile:
    """local_file resolution."""

    def test_existing_file_resolves(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("content")
        result = SourceResolver().resolve("local_file", str(f))
        assert result.resolved is True
        assert result.kind == "local_file"
        assert result.path == str(f.resolve())

    def test_nonexistent_file_not_resolved(self, tmp_path: Path) -> None:
        result = SourceResolver().resolve("local_file", str(tmp_path / "missing.txt"))
        assert result.resolved is False
        assert result.kind == "local_file"


class TestResolveS3Uri:
    """s3_uri resolution."""

    def test_valid_s3_uri_resolves(self) -> None:
        result = SourceResolver().resolve("s3_uri", "s3://bucket/key")
        assert result.resolved is True
        assert result.kind == "s3_uri"
        assert result.uri == "s3://bucket/key"
        assert result.allow_egress is True

    def test_s3_uri_with_path(self) -> None:
        result = SourceResolver().resolve("s3_uri", "s3://bucket/path/to/file.parquet")
        assert result.resolved is True
        assert result.uri == "s3://bucket/path/to/file.parquet"


class TestResolveRemoteEgressFailClose:
    """Remote types MUST fail closed — raise EgressDisabledError when no allowlist."""

    def test_http_url_no_allowlist(self) -> None:
        resolver = SourceResolver()
        with pytest.raises(EgressDisabledError) as exc_info:
            resolver.resolve("http_url", "https://example.com/data.json")
        err = exc_info.value
        assert err.source_kind == "http_url"
        assert "no allowlist configured" in err.reason
        assert "egress disabled" in str(err).lower()

    def test_http_url_with_allowlist_from_config(self) -> None:
        config = type(
            "Config", (), {"extra": {"knowledge_ingest": {"allowed_domains": ["example.com"]}}}
        )()
        resolver = SourceResolver(site_config=config)
        result = resolver.resolve("http_url", "https://example.com/data.json")
        assert result.resolved is True
        assert result.kind == "http_url"
        assert result.allow_egress is True
        assert "example.com" in result.allow_domains

    def test_http_url_with_inline_allow_domains(self) -> None:
        resolver = SourceResolver()
        result = resolver.resolve(
            "http_url",
            "https://example.com/data.json",
            allow_domains=["override.com", "api.test.org"],
        )
        assert result.allow_egress is True
        assert result.allow_domains == ["override.com", "api.test.org"]

    def test_http_url_inline_overrides_config(self) -> None:
        config = type(
            "Config", (), {"extra": {"knowledge_ingest": {"allowed_domains": ["config.com"]}}}
        )()
        resolver = SourceResolver(site_config=config)
        result = resolver.resolve(
            "http_url",
            "https://override.com/data.json",
            allow_domains=["override.com"],
        )
        assert result.allow_domains == ["override.com"]

    def test_git_repo_no_allowlist(self) -> None:
        resolver = SourceResolver()
        with pytest.raises(EgressDisabledError):
            resolver.resolve("git_repo", "https://github.com/org/repo.git")

    def test_git_repo_with_allow_domains(self) -> None:
        config = type(
            "Config", (), {"extra": {"knowledge_ingest": {"allowed_domains": ["github.com"]}}}
        )()
        resolver = SourceResolver(site_config=config)
        result = resolver.resolve("git_repo", "https://github.com/org/repo.git")
        assert result.resolved is True
        assert result.kind == "git_repo"
        assert result.allow_egress is True

    def test_unknown_kind_raises_egress_disabled(self) -> None:
        resolver = SourceResolver()
        with pytest.raises(EgressDisabledError) as exc_info:
            resolver.resolve("magic_protocol", "fake://uri")
        assert "unknown kind" in str(exc_info.value).lower()


class TestResolvedSourceHelpers:
    """ResolvedSource convenience properties."""

    def test_is_local_for_local_dir(self) -> None:
        src = ResolvedSource(kind="local_dir", path="/tmp/data")
        assert src.is_local is True
        assert src.is_remote is False

    def test_is_local_for_local_file(self) -> None:
        src = ResolvedSource(kind="local_file", path="/tmp/f.txt")
        assert src.is_local is True
        assert src.is_remote is False

    def test_is_remote_for_http(self) -> None:
        src = ResolvedSource(kind="http_url", uri="https://example.com")
        assert src.is_local is False
        assert src.is_remote is True

    def test_is_remote_for_git(self) -> None:
        src = ResolvedSource(kind="git_repo", uri="https://github.com/org/repo.git")
        assert src.is_local is False
        assert src.is_remote is True

    def test_s3_is_neither_local_nor_remote(self) -> None:
        src = ResolvedSource(kind="s3_uri", uri="s3://bucket/key")
        assert src.is_local is False
        assert src.is_remote is False

    def test_repr_contains_kind_and_resolved(self) -> None:
        src = ResolvedSource(kind="local_dir", path="/tmp", resolved=True)
        r = repr(src)
        assert "kind='local_dir'" in r
        assert "resolved=True" in r

    def test_repr_with_uri(self) -> None:
        src = ResolvedSource(kind="http_url", uri="https://example.com", resolved=True)
        r = repr(src)
        assert "uri='https://example.com'" in r


class TestManifestExpansion:
    """Manifest expansion: local_dir expands to sorted file list."""

    def test_expansion_produces_files(self, tmp_path: Path) -> None:
        d = tmp_path / "data"
        d.mkdir()
        (d / "readme.md").write_text("# README")
        sub = d / "sub"
        sub.mkdir()
        (sub / "code.py").write_text("print(1)")
        result = SourceResolver().resolve("local_dir", str(d))
        assert len(result.files) >= 2

    def test_expansion_sorted(self, tmp_path: Path) -> None:
        d = tmp_path / "data"
        d.mkdir()
        (d / "z.txt").write_text("z")
        (d / "a.txt").write_text("a")
        (d / "m.txt").write_text("m")
        result = SourceResolver().resolve("local_dir", str(d))
        rel_paths = [f.relative_path for f in result.files]
        assert rel_paths == sorted(rel_paths)

    def test_expansion_files_are_valid(self, tmp_path: Path) -> None:
        d = tmp_path / "data"
        d.mkdir()
        (d / "file.txt").write_text("content")
        result = SourceResolver().resolve("local_dir", str(d))
        for f in result.files:
            assert isinstance(f, ResolvedFile)
            assert Path(f.absolute_path).exists()
            assert not Path(f.absolute_path).is_dir()
            assert f.relative_path != ""


class TestEgressDisabledError:
    """EgressDisabledError attributes and message."""

    def test_attributes(self) -> None:
        err = EgressDisabledError("http_url", "no allowlist configured")
        assert err.source_kind == "http_url"
        assert err.reason == "no allowlist configured"
        assert "http_url" in str(err)
        assert "egress disabled" in str(err).lower()

    def test_default_reason(self) -> None:
        err = EgressDisabledError("git_repo")
        assert err.reason == "egress disabled"

    def test_invalid_s3_uri_not_resolved(self) -> None:
        result = SourceResolver().resolve("s3_uri", "https://example.com/file")
        assert result.resolved is False
        assert result.kind == "s3_uri"

    def test_malformed_s3_uri_no_bucket(self) -> None:
        result = SourceResolver().resolve("s3_uri", "s3:///")
        assert result.resolved is False


class TestResolvedSourceProperties:
    """ResolvedSource helper properties and repr."""

    def test_is_local_true(self) -> None:
        rs = ResolvedSource(kind="local_dir", path="/tmp", resolved=True)
        assert rs.is_local is True

    def test_is_local_false_for_remote(self) -> None:
        rs = ResolvedSource(kind="http_url", uri="https://x.com", resolved=True, allow_egress=True)
        assert rs.is_local is False

    def test_is_remote_true(self) -> None:
        rs = ResolvedSource(kind="http_url", uri="https://x.com", resolved=True, allow_egress=True)
        assert rs.is_remote is True

    def test_is_remote_false_for_local(self) -> None:
        rs = ResolvedSource(kind="local_file", path="/tmp/f", resolved=True)
        assert rs.is_remote is False

    def test_repr_local(self) -> None:
        rs = ResolvedSource(kind="local_dir", path="/data", resolved=True)
        r = repr(rs)
        assert "kind='local_dir'" in r
        assert "resolved=True" in r

    def test_repr_remote(self) -> None:
        rs = ResolvedSource(kind="http_url", uri="https://x.com", allow_egress=True, resolved=True)
        r = repr(rs)
        assert "kind='http_url'" in r
        assert "uri='https://x.com'" in r

    def test_repr_with_files(self) -> None:
        rs = ResolvedSource(
            kind="local_dir",
            path="/data",
            resolved=True,
            files=[
                ResolvedFile(relative_path="a.txt", absolute_path="/data/a.txt"),
                ResolvedFile(relative_path="b.txt", absolute_path="/data/b.txt"),
            ],
        )
        r = repr(rs)
        assert "files=2" in r


class TestSiteConfigEgress:
    """Egress allowlist from site config extra dict."""

    def test_no_site_config_returns_empty(self) -> None:
        resolver = SourceResolver(site_config=None)
        assert resolver._get_allowed_domains() == []

    def test_site_config_with_allow_domains(self) -> None:
        """Egress permit flows from site config extra dict."""
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.extra = {
            "knowledge_ingest": {
                "allowed_domains": ["example.com", "data.org"],
            }
        }

        resolver = SourceResolver(site_config=mock_config)  # type: ignore[arg-type]
        domains = resolver._get_allowed_domains()
        assert domains == ["example.com", "data.org"]

        result = resolver.resolve("http_url", "https://example.com/data")
        assert result.resolved is True
        assert result.allow_egress is True
        assert result.allow_domains == ["example.com", "data.org"]

    def test_site_config_empty_allowlist_fails(self) -> None:
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.extra = {"knowledge_ingest": {"allowed_domains": []}}

        resolver = SourceResolver(site_config=mock_config)  # type: ignore[arg-type]
        assert resolver._get_allowed_domains() == []

        with pytest.raises(EgressDisabledError):
            resolver.resolve("http_url", "https://example.com/data")

    def test_site_config_missing_ki_section_fails(self) -> None:
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.extra = {}

        resolver = SourceResolver(site_config=mock_config)  # type: ignore[arg-type]
        assert resolver._get_allowed_domains() == []

        with pytest.raises(EgressDisabledError):
            resolver.resolve("http_url", "https://example.com/data")

    def test_allow_domains_override_takes_precedence(self) -> None:
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.extra = {"knowledge_ingest": {"allowed_domains": ["config.com"]}}

        resolver = SourceResolver(site_config=mock_config)  # type: ignore[arg-type]
        result = resolver.resolve(
            "http_url",
            "https://example.com/data",
            allow_domains=["override.com"],
        )
        assert result.allow_domains == ["override.com"]


class TestResolveIngestSourceConvenience:
    """Module-level convenience function."""

    def test_local_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "data"
        d.mkdir()
        result = resolve_ingest_source("local_dir", str(d))
        assert result.resolved is True

    def test_http_url_with_domains(self) -> None:
        result = resolve_ingest_source(
            "http_url",
            "https://example.com/data",
            allow_domains=["example.com"],
        )
        assert result.resolved is True

    def test_http_url_without_domains_raises(self) -> None:
        with pytest.raises(EgressDisabledError):
            resolve_ingest_source("http_url", "https://example.com/data")

    def test_s3_uri(self) -> None:
        result = resolve_ingest_source("s3_uri", "s3://bucket/key")
        assert result.resolved is True


class TestResolvedFile:
    """ResolvedFile immutable dataclass."""

    def test_creation(self) -> None:
        rf = ResolvedFile(relative_path="a/b.txt", absolute_path="/data/a/b.txt")
        assert rf.relative_path == "a/b.txt"
        assert rf.absolute_path == "/data/a/b.txt"

    def test_frozen(self) -> None:
        rf = ResolvedFile(relative_path="x", absolute_path="/x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            rf.relative_path = "y"
