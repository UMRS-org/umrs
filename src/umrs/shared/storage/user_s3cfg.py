"""Per-user s3cfg rendering and the ``UMRS_CONFIG/user_files`` mirror.

Backend-neutral: the s3cfg format (s3cmd's) is what every client of the
platform consumes, whichever object store issued the credentials. Moved out
of ``storage/minio.py`` for ADR-055 so the SeaweedFS adapter shares one
writer; ``storage/minio.py`` keeps its historical names as aliases.
"""

from __future__ import annotations

import os
import textwrap


def render_user_s3cfg(
    *,
    host_base: str,
    bucket_location: str,
    use_https: bool,
    verify_ssl: bool,
    access_key: str,
    secret_key: str,
    alias: str,
) -> str:
    """Render the s3cfg text delivered to a user (same shape MinIO used)."""
    return textwrap.dedent(
        f"""\
        [default]
        host_base = {host_base}
        host_bucket = {host_base}
        bucket_location = {bucket_location}
        use_https = {use_https}
        check_ssl_certificate = {verify_ssl}
        access_key = {access_key}
        secret_key = {secret_key}
        alias = {alias}
        """
    )


def write_user_s3cfg(path: str, content: str) -> str:
    """Write *content* to *path* (0600, parent dirs created); returns the path."""
    expanded = os.path.expanduser(path)
    os.makedirs(os.path.dirname(expanded) or ".", exist_ok=True)
    fd = os.open(expanded, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    return expanded


def user_files_dir() -> str | None:
    """Return ``UMRS_CONFIG/user_files`` when it exists, else None."""
    umrs_config = os.environ.get("UMRS_CONFIG")
    if not umrs_config:
        return None
    path = os.path.join(os.path.expanduser(umrs_config), "user_files")
    return path if os.path.isdir(path) else None


def mirror_to_user_files(username: str, s3cfg_content: str) -> None:
    """Keep ``user_files/s3cfg_<username>`` in sync with the authoritative copy."""
    target_dir = user_files_dir()
    if not target_dir:
        return
    try:
        with open(
            os.path.join(target_dir, f"s3cfg_{username}"), "w", encoding="utf-8"
        ) as fh:
            fh.write(s3cfg_content)
    except OSError:
        pass


def prune_stale_user_files(username: str, s3cfg_content: str) -> None:
    """Drop a ``user_files`` mirror copy that no longer matches *s3cfg_content*."""
    target_dir = user_files_dir()
    if not target_dir:
        return
    target_path = os.path.join(target_dir, f"s3cfg_{username}")
    if not os.path.isfile(target_path):
        return
    try:
        with open(target_path, encoding="utf-8") as fh:
            existing = fh.read()
        if existing.strip() != s3cfg_content.strip():
            os.remove(target_path)
    except OSError:
        pass
