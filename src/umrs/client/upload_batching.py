"""Batched PROCESS_OBJECTS notifications for ``upload_dir`` (plan U1 A3).

Port of upstream 73ab20a0 and d8b0b364. Each uploaded object's URI goes down
the PUSH/PULL data queue, but the server only reads that queue after a
control-channel ``PROCESS_OBJECTS <n>``. Notifying once at the end let a large
upload fill both high-water marks (client SNDHWM + server RCVHWM, 10 240
each by default) and stall until the send timed out. ``BatchedNotifier`` counts
every URI the data queue accepted and notifies every ``NOTIFY_BATCH_SIZE``
objects; when the upload aborts it still notifies the URIs already queued, so
those objects are processed instead of left unindexed.

Unlike upstream, the count is taken per accepted ``notify_s3`` call rather than
per finished file, so a text file whose lines fail half-way is counted exactly.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Below one high-water mark (DEFAULT_NB_MSG = 10 240) so a batch always fits.
NOTIFY_BATCH_SIZE = 8000


class BatchedNotifier:
    """Data-queue proxy that tells the server about queued URIs in batches.

    *notify(count, blocking)* sends ``PROCESS_OBJECTS count`` and returns False
    when the server rejected it. The proxy exposes ``notify_s3`` so it can be
    handed to the ``s3_store_*`` helpers in place of the data queue.
    """

    def __init__(
        self,
        data_mq,
        notify: Callable[[int, bool], bool],
        batch_size: int | None = None,
    ):
        self._data_mq = data_mq
        self._notify = notify
        size = NOTIFY_BATCH_SIZE if batch_size is None else batch_size
        self.batch_size = max(1, int(size))
        self.pending = 0
        self.total = 0
        self.ok = True

    def notify_s3(self, s3_object) -> None:
        self._data_mq.notify_s3(s3_object)
        self.pending += 1
        self.total += 1
        if self.pending >= self.batch_size:
            self.flush(blocking=False)

    def flush(self, blocking: bool = False) -> bool:
        """Notify the server of every pending URI; False once any was rejected."""
        if self.pending > 0:
            count, self.pending = self.pending, 0
            if not self._notify(count, blocking):
                self.ok = False
        return self.ok

    def _flush_after_abort(self) -> None:
        if self.pending <= 0:
            return
        print(
            f"Upload aborted; notifying the server of {self.pending} object(s) "
            "already queued.",
            file=sys.stderr,
        )
        try:
            self.flush(blocking=False)
        except Exception:  # the abort's own error must surface, not this one
            logger.warning("notify after aborted upload failed", exc_info=True)
            self.ok = False

    @contextmanager
    def flushing(self, blocking: bool = False) -> Iterator[BatchedNotifier]:
        """Flush on exit; on an abort flush (non-blocking) and re-raise."""
        try:
            yield self
        except BaseException:
            self._flush_after_abort()
            raise
        self.flush(blocking=blocking)


def _file_subdir(rel_dir: str, subdir: str | None) -> str | None:
    if rel_dir in (".", ""):
        return subdir
    if subdir:
        return f"{subdir}/{rel_dir}"
    return rel_dir


def collect_upload_files(path: str, subdir: str | None):
    """Walk *path* recursively; return ``(entries, nb_seen, broken_symlinks)``.

    *entries* lists ``(file_path, file_subdir)`` in a stable order. Each file's
    relative directory becomes part of its S3 key prefix, which keeps nested
    hierarchies (for example camera/Schedule/date) intact.
    """
    norm_path = os.path.abspath(path)
    entries: list[tuple[str, str | None]] = []
    broken_symlinks: list[str] = []
    nb_seen = 0
    for root, dirs, files in os.walk(norm_path):
        dirs.sort()
        file_subdir = _file_subdir(os.path.relpath(root, norm_path), subdir)
        for fname in sorted(files):
            nb_seen += 1
            file_path = os.path.join(root, fname)
            if os.path.isfile(file_path):
                entries.append((file_path, file_subdir))
            elif os.path.islink(file_path):
                broken_symlinks.append(f"{file_path} -> {os.readlink(file_path)}")
    return entries, nb_seen, broken_symlinks


def record_uploaded(s3_key, nb_sent, file_path, s3_key_list, file_list) -> None:
    """Append what one ``process_file`` call stored to the result lists."""
    if not s3_key:
        return
    if isinstance(s3_key, list):
        s3_key_list.extend(s3_key)
        file_list.extend([file_path] * nb_sent)
    else:
        s3_key_list.append(s3_key)
        file_list.append(file_path)


def report_nothing_uploaded(path, modality, nb_seen, broken_symlinks) -> None:
    """Explain why a directory upload stored no object."""
    if nb_seen == 0:
        print(f"Error: directory '{path}' contains no files to upload.")
    elif broken_symlinks:
        print(
            f"Error: none of the {nb_seen} entries in '{path}' "
            "could be uploaded because they are broken symlinks "
            "(their targets do not exist):"
        )
        for link in broken_symlinks:
            print(f"  - {link}")
        print(
            "Re-stage the input files so the symlinks point at existing "
            "targets, then retry."
        )
    else:
        print(
            f"Error: found {nb_seen} file(s) in '{path}' but none "
            f"were accepted for modality '{modality}'. Check that the "
            "files match the expected type for this modality."
        )
