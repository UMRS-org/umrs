"""Client side of KNN: the query image's reserved key, its upload and removal.

``umrs knn`` and ``bench_knn`` share these (review U1): the query goes up
under ``[<subdir>/].umrs-knn/<uuid>-<basename>``, never under its bare
basename, which replaced any object of that name (a collaborator's in a
shared bucket included). The server never returns or counts keys under the
reserved directory; the object is removed after the reply.
"""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from umrs.shared.storage.s3_upload import upload_local_file

# Equal to umrs.server.knn_worker.QUERY_UPLOAD_DIR; the client does not
# import the server module.
QUERY_UPLOAD_DIR = ".umrs-knn"


def query_key(file_path: str, subdir: str | None) -> str:
    """A fresh key under ``[<subdir>/].umrs-knn/`` for the query image."""
    name = f"{QUERY_UPLOAD_DIR}/{uuid.uuid4().hex}-{os.path.basename(file_path)}"
    return f"{subdir}/{name}" if subdir else name


def query_name(file_path: str, subdir: str | None) -> str:
    """The name the reply shows for the query: ``[<subdir>/]<basename>``."""
    base = os.path.basename(file_path)
    return f"{subdir}/{base}" if subdir else base


def upload_query(s3, bucket: str, file_path: str, subdir: str | None) -> str:
    """Upload *file_path* to a fresh reserved key; returns the key.

    The key is unique, so there is nothing to compare: no skip-unchanged HEAD.
    """
    key = query_key(file_path, subdir)
    upload_local_file(s3, bucket, key, file_path, skip_unchanged=False)
    return key


def remove_query(s3, bucket: str, key: str) -> None:
    """Delete the uploaded query object; a failure only warns."""
    try:
        s3.delete_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001 — the search result stands
        print(f"Warning: could not remove the KNN query object {key}: {exc}",
              file=sys.stderr)


@contextmanager
def uploaded_query(s3, bucket: str, file_path: str, subdir: str | None) -> Iterator[str]:
    """The query's key for the block; the object is removed on exit."""
    key = upload_query(s3, bucket, file_path, subdir)
    try:
        yield key
    finally:
        remove_query(s3, bucket, key)
