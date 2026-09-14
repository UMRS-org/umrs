from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import Operation, OperationQueue, OperationStatus

log = logging.getLogger(__name__)


class OperationAPI:
    """Background operations that modify UMRS state (upload, download, run).

    Wraps OperationSDK — no duplicated transfer logic.  Keeps OperationQueue
    for progress bookkeeping and background task management.
    """

    def __init__(
        self,
        control_api: Any,
        control_mq: Any,
        client_id: str,
        operation_queue: OperationQueue,
        config_path: str | None = None,
    ) -> None:
        self._api = control_api
        self._mq = control_mq
        self._client_id = client_id
        self._op_queue = operation_queue
        self._config_path = config_path

        # Bucket config loaded lazily
        self._buckets_cfg: dict[str, Any] | None = None

        # OperationSDK instance (shared across all operations)
        self._sdk: Any = None

    @property
    def sdk(self) -> Any:
        """Lazy OperationSDK instance."""
        if self._sdk is None:
            from ..client.operations import OperationSDK

            self._sdk = OperationSDK(
                gateway_url=self._api.base_url
                if hasattr(self._api, "base_url")
                else None,
            )
        return self._sdk

    # ------------------------------------------------------------------
    # Upload operations
    # ------------------------------------------------------------------

    def upload_file(
        self,
        file_path: str,
        bucket_alias: str,
        subdir: str = "",
        progress_fn: Callable[[Operation], None] | None = None,
    ) -> None:
        """Upload a single file to S3 via s3cmd. Updates op.progress."""
        op = Operation(
            operation_type="upload_file",
            title=file_path,
        )
        self._op_queue.add(op)

        def _coro(op: Operation) -> asyncio.Future:
            async def _run() -> None:
                try:
                    await self._do_upload_file(
                        op, file_path, bucket_alias, subdir, progress_fn
                    )
                    op.mark_done(OperationStatus.SUCCEEDED)
                except Exception as e:
                    op.mark_done(OperationStatus.FAILED, error=str(e))
                    log.exception("Upload failed: %s", file_path)

            loop = asyncio.get_event_loop()
            return loop.create_task(_run())

        self._op_queue.start_background(op, _coro)

    async def _do_upload_file(
        self,
        op: Operation,
        file_path: str,
        bucket_alias: str,
        subdir: str,
        progress_fn: Callable[[Operation], None] | None,
    ) -> None:
        s3cmd = shutil.which("s3cmd") or "s3cmd"
        # Resolve bucket name from alias via the config file
        bucket_name = await self._resolve_bucket_name(bucket_alias)
        if not bucket_name:
            raise ValueError(f"Bucket alias '{bucket_alias}' not found in config")

        cmd = [s3cmd, "put"]
        if subdir:
            cmd.extend(["--prefix", f"{subdir}/"])
        cmd.extend([file_path, f"s3://{bucket_name}/"])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"s3cmd upload failed: {stdout.decode()}")

        # s3cmd prints file size in the output; set 100%
        op.mark_progress(percent=100.0, message="Done", current=1, total=1)

    def upload_dir(
        self,
        dir_path: str,
        bucket_alias: str,
        max_images: int = 0,
    ) -> None:
        """Upload a directory to an S3 bucket."""
        op = Operation(
            operation_type="upload_dir",
            title=dir_path,
        )
        self._op_queue.add(op)

        async def _coro(op: Operation) -> None:
            try:
                await self._do_upload_dir(op, dir_path, bucket_alias, max_images)
                op.mark_done(OperationStatus.SUCCEEDED)
            except Exception as e:
                op.mark_done(OperationStatus.FAILED, error=str(e))

        self._op_queue.start_background(op, lambda o: asyncio.create_task(_coro(o)))

    async def _do_upload_dir(
        self,
        op: Operation,
        dir_path: str,
        bucket_alias: str,
        max_images: int,
    ) -> None:
        """Recursively upload a directory to S3, respecting max_images limit."""
        dir_root = Path(dir_path).resolve()
        files = self._collect_files(dir_root, max_images)
        await self._sync_to_bucket(op, files, dir_root, bucket_alias)

    def _collect_files(self, dir_root: Path, max_images: int) -> list[str]:
        """Collect files from directory, respecting max_images limit."""
        files = [str(p) for p in sorted(dir_root.rglob("*")) if p.is_file()]
        if max_images > 0:
            files = files[:max_images]
        return files

    async def _sync_to_bucket(
        self,
        op: Operation,
        files: list[str],
        dir_root: Path,
        bucket_alias: str,
    ) -> None:
        """Run s3cmd sync and stream progress."""
        s3cmd = shutil.which("s3cmd") or "s3cmd"
        bucket_name = await self._resolve_bucket_name(bucket_alias)
        if not bucket_name:
            raise ValueError(f"Bucket alias '{bucket_alias}' not found in config")

        total = len(files)
        if total == 0:
            op.mark_progress(
                percent=100.0, message="No files found", current=0, total=0
            )
            op.mark_done(OperationStatus.SUCCEEDED)
            return

        op.mark_progress(
            percent=0.0,
            message=f"Uploading {total} files to {bucket_name}...",
            current=0,
            total=total,
        )

        cmd = [s3cmd, "sync", str(dir_root), f"s3://{bucket_name}/"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await self._stream_progress(op, proc, total, files)

    async def _stream_progress(
        self,
        op: Operation,
        proc,
        total: int,
        files: list[str],
    ) -> None:
        """Stream output from a subprocess and update operation progress."""
        import re

        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            line_str = line.decode().strip()
            if "%" in line_str:
                pct_part = line_str.split("(")[-1].rstrip(")")
                try:
                    pct = float(pct_part.rstrip("%")) / 100.0
                    op.mark_progress(
                        percent=pct * 100.0,
                        message=line_str,
                        current=0,
                        total=total,
                    )
                except (ValueError, IndexError):
                    pass
                if "100.0" in pct_part:
                    op.mark_progress(
                        percent=100.0, message="Done", current=total, total=total
                    )
                    break
            elif "uploading" in line_str.lower():
                count_match = re.search(r"uploaded\s+(\d+)", line_str)
                if count_match:
                    count = int(count_match.group(1))
                    op.mark_progress(
                        percent=(count / total) * 100.0,
                        message=f"Uploaded {count}/{total}",
                        current=count,
                        total=total,
                    )

        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            raise RuntimeError(f"s3cmd sync failed: {stderr.decode()}")

        op.mark_progress(percent=100.0, message="Done", current=total, total=total)

    def download_from_bucket(
        self,
        bucket_alias: str,
        s3_path: str,
        local_path: str,
        progress_fn: Callable[[Operation], None] | None = None,
    ) -> None:
        """Download a file or directory from an S3 bucket."""
        op = Operation(
            operation_type="download_from_bucket",
            title=f"{bucket_alias}:{s3_path} → {local_path}",
        )
        self._op_queue.add(op)

        async def _coro(op: Operation) -> None:
            try:
                await self._do_download_from_bucket(
                    op, bucket_alias, s3_path, local_path, progress_fn
                )
                op.mark_done(OperationStatus.SUCCEEDED)
            except Exception as e:
                op.mark_done(OperationStatus.FAILED, error=str(e))

        self._op_queue.start_background(op, lambda o: asyncio.create_task(_coro(o)))

    async def _do_download_from_bucket(
        self,
        op: Operation,
        bucket_alias: str,
        s3_path: str,
        local_path: str,
        progress_fn: Callable[[Operation], None] | None,
    ) -> None:
        s3cmd = shutil.which("s3cmd") or "s3cmd"
        bucket_name = await self._resolve_bucket_name(bucket_alias)
        if not bucket_name:
            raise ValueError(f"Bucket alias '{bucket_alias}' not found in config")

        cmd = [s3cmd, "get", f"s3://{bucket_name}/{s3_path}", local_path]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"s3cmd download failed: {stdout.decode()}")

        op.mark_progress(percent=100.0, message="Downloaded", current=1, total=1)

    def upload_workflow(
        self,
        workflow_path: str,
        workflow_name: str,
        bucket_alias: str,
        progress_fn: Callable[[Operation], None] | None = None,
    ) -> None:
        """Upload a workflow directory to an S3 bucket."""
        op = Operation(
            operation_type="upload_workflow",
            title=workflow_name,
        )
        self._op_queue.add(op)

        async def _coro(op: Operation) -> None:
            try:
                await self._do_upload_workflow(
                    op, workflow_path, workflow_name, bucket_alias, progress_fn
                )
                op.mark_done(OperationStatus.SUCCEEDED)
            except Exception as e:
                op.mark_done(OperationStatus.FAILED, error=str(e))

        self._op_queue.start_background(op, lambda o: asyncio.create_task(_coro(o)))

    async def _do_upload_workflow(
        self,
        op: Operation,
        workflow_path: str,
        workflow_name: str,
        bucket_alias: str,
        progress_fn: Callable[[Operation], None] | None,
    ) -> None:
        s3cmd = shutil.which("s3cmd") or "s3cmd"
        bucket_name = await self._resolve_bucket_name(bucket_alias)
        if not bucket_name:
            raise ValueError(f"Bucket alias '{bucket_alias}' not found in config")

        cmd = [s3cmd, "sync", workflow_path, f"s3://{bucket_name}/{workflow_name}/"]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"s3cmd workflow upload failed: {stdout.decode()}")

        op.mark_progress(percent=100.0, message="Done", current=1, total=1)

    def run_workflow(
        self,
        workflow_name: str,
        bucket_alias: str,
        params: dict[str, Any] | None = None,
        progress_fn: Callable[[Operation], None] | None = None,
    ) -> None:
        """Execute a registered workflow. Delegates to OperationSDK."""
        op = Operation(
            operation_type="run_workflow",
            title=workflow_name,
        )
        self._op_queue.add(op)

        async def _coro(op: Operation) -> None:
            try:
                result = self.sdk.run_workflow(
                    name=workflow_name,
                    bucket=bucket_alias,
                    params=params or {},
                    progress_cb=lambda pct, msg: (
                        op.mark_progress(percent=pct, message=msg, current=0, total=100)
                        if hasattr(op, "mark_progress")
                        else None
                    ),
                )
                op.mark_done(OperationStatus.SUCCEEDED, detail=result.detail)
            except Exception as e:
                op.mark_done(OperationStatus.FAILED, error=str(e))

        self._op_queue.start_background(op, lambda o: asyncio.create_task(_coro(o)))

    def run_experience(
        self,
        experience_name: str,
        bucket_alias: str,
        overrides: dict[str, Any] | None = None,
        progress_fn: Callable[[Operation], None] | None = None,
    ) -> None:
        """Run an experience (HMAE, MAE, etc.). Delegates to OperationSDK."""
        op = Operation(
            operation_type="run_experience",
            title=experience_name,
        )
        self._op_queue.add(op)

        async def _coro(op: Operation) -> None:
            try:
                result = self.sdk.run_experience(
                    name=experience_name,
                    bucket=bucket_alias,
                    overrides=overrides or {},
                    progress_cb=lambda pct, msg: (
                        op.mark_progress(percent=pct, message=msg, current=0, total=100)
                        if hasattr(op, "mark_progress")
                        else None
                    ),
                )
                op.mark_done(OperationStatus.SUCCEEDED, detail=result.detail)
            except Exception as e:
                op.mark_done(OperationStatus.FAILED, error=str(e))

        self._op_queue.start_background(op, lambda o: asyncio.create_task(_coro(o)))

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    async def _resolve_bucket_name(self, bucket_alias: str) -> str | None:
        """Resolve bucket alias to actual bucket name from config."""
        import json

        if self._buckets_cfg is not None:
            return self._buckets_cfg.get(bucket_alias)

        self._buckets_cfg = {}
        if self._config_path:
            try:
                with open(self._config_path) as f:
                    cfg = json.load(f)
                    buckets = cfg.get("buckets", {})
                    self._buckets_cfg.update(
                        {k: v.get("name") for k, v in buckets.items()}
                    )
            except (OSError, json.JSONDecodeError):
                pass
        return self._buckets_cfg.get(bucket_alias)
