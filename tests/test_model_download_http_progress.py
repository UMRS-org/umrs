"""HTTP model-download progress behavior."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from umrs.jobs.model_download import (
    DownloadSource,
    DownloadTaskContext,
    DownloadTaskRegistry,
    ModelDownloadBackend,
)


class InterruptingContent:
    """Async response content that fails after two chunks."""

    async def iter_chunked(self, _chunk_size):
        yield b"abcd"
        yield b"efg"
        raise RuntimeError("connection dropped")


class FakeResponse:
    """Minimal aiohttp response context manager."""

    status = 200
    reason = "OK"
    content_length = 10
    content = InterruptingContent()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeSession:
    """Minimal aiohttp session context manager."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def get(self, _url):
        return FakeResponse()


@pytest.mark.asyncio
async def test_interrupted_http_download_publishes_partial_progress(tmp_path):
    """Interrupted HTTP transfers expose partial byte counts before failing."""
    import aiohttp

    validator = MagicMock()
    validator.is_allowed.return_value = True
    backend = ModelDownloadBackend(
        model_store=MagicMock(),
        path_validator=validator,
        download_registry=DownloadTaskRegistry(),
    )
    ctx = DownloadTaskContext(
        job_id="job-http",
        attempt_id="att-http",
        source=DownloadSource.HTTP,
        source_id="https://example.invalid/model.bin",
        target_path=str(tmp_path),
        model_name="model-bin",
        scope="user",
        owner_id="user-test",
    )
    published = []

    with patch.object(aiohttp, "ClientSession", return_value=FakeSession()):
        progress = await backend._download_from_http(
            ctx,
            progress_callback=lambda p: published.append(p.to_dict()),
        )

    assert progress.status == "failed"
    assert progress.downloaded_bytes == 7
    assert progress.total_bytes == 10
    assert "connection dropped" in progress.error_message
    assert (tmp_path / "model.bin").read_bytes() == b"abcdefg"
    assert [item["downloaded_bytes"] for item in published] == [0, 4, 7, 7]
    assert published[-1]["status"] == "failed"


async def _start_interrupting_server(total_bytes: int, chunk_bytes: int):
    """Start a loopback server that closes before Content-Length is satisfied."""
    from aiohttp import web

    async def handler(request):
        response = web.StreamResponse(
            status=200,
            headers={"Content-Length": str(total_bytes)},
        )
        await response.prepare(request)
        await response.write(b"a" * chunk_bytes)
        await response.write(b"b" * chunk_bytes)
        await asyncio.sleep(0)
        transport = request.transport
        if transport:
            transport.close()
        return response

    app = web.Application()
    app.router.add_get("/model.bin", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets
    assert sockets
    return runner, sockets[0].getsockname()[1]


@pytest.mark.asyncio
async def test_runtime_interrupted_http_download_publishes_partial_progress(tmp_path):
    """A real loopback HTTP interruption leaves partial progress metrics."""
    validator = MagicMock()
    validator.is_allowed.return_value = True
    backend = ModelDownloadBackend(
        model_store=MagicMock(),
        path_validator=validator,
        download_registry=DownloadTaskRegistry(),
    )
    runner, port = await _start_interrupting_server(
        total_bytes=1024 * 1024,
        chunk_bytes=128 * 1024,
    )
    published = []
    try:
        ctx = DownloadTaskContext(
            job_id="job-http-runtime",
            attempt_id="att-http-runtime",
            source=DownloadSource.HTTP,
            source_id=f"http://127.0.0.1:{port}/model.bin",
            target_path=str(tmp_path),
            model_name="model-bin",
            scope="user",
            owner_id="user-test",
        )
        progress = await backend._download_from_http(
            ctx,
            progress_callback=lambda p: published.append(p.to_dict()),
        )
    finally:
        await runner.cleanup()

    target = tmp_path / "model.bin"
    assert progress.status == "failed"
    assert 0 < progress.downloaded_bytes < progress.total_bytes
    assert target.stat().st_size == progress.downloaded_bytes
    assert any(item["downloaded_bytes"] > 0 for item in published)
    assert published[-1]["status"] == "failed"