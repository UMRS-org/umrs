"""GitHub connector — indexes repositories via the GitHub REST API.

Fetches repository content (README, source files, issues, PRs) and
yields documents for the UMRS RAG pipeline. Uses GitHub's REST API
with token auth.

No dependency on r4.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from . import (
    Connector,
    ConnectorConfig,
    ConnectorType,
    Document,
    SyncCheckpoint,
    register_connector,
)

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"
_TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".rb",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".xml",
    ".html",
    ".css",
    ".sql",
}
_MAX_FILE_SIZE = 1_000_000  # 1 MB


class GitHubConnector(Connector):
    """Fetch documents from a GitHub repository."""

    @property
    def connector_type(self) -> ConnectorType:
        return ConnectorType.GITHUB

    @property
    def name(self) -> str:
        return "GitHub"

    def validate_config(self, config: ConnectorConfig) -> list[str]:
        errors: list[str] = []
        if not config.settings.get("repo"):
            errors.append("repo is required (format: owner/repo)")
        if not config.credentials.get("token"):
            errors.append("credentials.token is required (GitHub PAT)")
        return errors

    def _request(
        self,
        path: str,
        token: str,
        params: dict[str, str] | None = None,
    ) -> Any:
        url = f"{_GITHUB_API}{path}"
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{qs}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "umrs-connector/1.0",
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())

    def _fetch_tree(
        self,
        repo: str,
        token: str,
        branch: str = "main",
    ) -> list[dict[str, Any]]:
        """Fetch the full file tree for a repo."""
        try:
            data = self._request(
                f"/repos/{repo}/git/trees/{branch}",
                token,
                {"recursive": "1"},
            )
            return data.get("tree", [])
        except Exception as exc:
            logger.warning("Failed to fetch tree for %s: %s", repo, exc)
            return []

    def _fetch_file_content(self, repo: str, path: str, token: str) -> str | None:
        """Fetch raw file content."""
        try:
            url = f"{_GITHUB_API}/repos/{repo}/contents/{path}"
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github.v3.raw",
                    "User-Agent": "umrs-connector/1.0",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            logger.debug("Cannot fetch %s/%s: %s", repo, path, exc)
            return None

    def fetch(
        self,
        config: ConnectorConfig,
        checkpoint: SyncCheckpoint | None = None,
    ) -> Iterator[Document]:
        repo = config.settings["repo"]
        token = config.credentials["token"]
        branch = config.settings.get("branch", "main")
        include_issues = config.settings.get("include_issues", False)

        # Fetch file tree
        tree = self._fetch_tree(repo, token, branch)
        for item in tree:
            if item.get("type") != "blob":
                continue
            path = item.get("path", "")
            size = item.get("size", 0)

            # Filter by extension and size
            ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
            if ext.lower() not in _TEXT_EXTENSIONS:
                continue
            if size > _MAX_FILE_SIZE:
                continue

            content = self._fetch_file_content(repo, path, token)
            if content is None:
                continue

            yield Document(
                source_type=ConnectorType.GITHUB,
                source_id=f"{repo}/{path}",
                title=path,
                content=content,
                content_type="text/plain",
                metadata={
                    "repo": repo,
                    "branch": branch,
                    "path": path,
                    "sha": item.get("sha", ""),
                },
                url=f"https://github.com/{repo}/blob/{branch}/{path}",
                size_bytes=size,
            )

        # Optionally fetch issues
        if include_issues:
            yield from self._fetch_issues(repo, token, checkpoint)

    def _fetch_issues(
        self,
        repo: str,
        token: str,
        checkpoint: SyncCheckpoint | None = None,
    ) -> Iterator[Document]:
        """Fetch open issues as documents."""
        params: dict[str, str] = {
            "state": "open",
            "per_page": "100",
            "sort": "updated",
        }
        if checkpoint and checkpoint.last_sync:
            params["since"] = checkpoint.last_sync.isoformat()

        try:
            issues = self._request(f"/repos/{repo}/issues", token, params)
        except Exception as exc:
            logger.warning("Failed to fetch issues for %s: %s", repo, exc)
            return

        for issue in issues:
            if issue.get("pull_request"):
                continue  # Skip PRs in issue list
            body = issue.get("body") or ""
            title = issue.get("title", "")
            number = issue.get("number", 0)
            yield Document(
                source_type=ConnectorType.GITHUB,
                source_id=f"{repo}/issues/{number}",
                title=f"Issue #{number}: {title}",
                content=f"# {title}\n\n{body}",
                content_type="text/markdown",
                metadata={
                    "repo": repo,
                    "issue_number": number,
                    "labels": [l.get("name", "") for l in issue.get("labels", [])],
                    "state": issue.get("state", "open"),
                },
                url=issue.get("html_url", ""),
            )

    def health_check(self, config: ConnectorConfig) -> bool:
        token = config.credentials.get("token", "")
        if not token:
            return False
        try:
            self._request("/user", token)
            return True
        except Exception:
            return False


register_connector(GitHubConnector())
