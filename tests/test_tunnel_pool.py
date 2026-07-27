"""Tests for TunnelPool liveness handling.

Regression: a ControlMaster restart silently drops master-owned forwards
(``TunnelSpec.process is None``). The pool must detect the dead local
listener and re-establish the tunnel instead of returning the stale port
forever (symptom: gateway 502 upstream.bad_gateway on a READY endpoint).
"""

import asyncio
from unittest import mock

import pytest
from umrs.orchestrator.tunnel_pool import TunnelPool, TunnelPoolConfig, TunnelSpec


def _make_pool(**overrides) -> TunnelPool:
    cfg = TunnelPoolConfig(
        ssh_target="login-node",
        control_path="/run/umrs-ssh/test-%p",
        **overrides,
    )
    return TunnelPool(cfg)


class _FakeProc:
    """Stand-in for asyncio subprocess that exited 0 (multiplexed slave)."""

    returncode = 0
    stderr = None


@pytest.mark.asyncio
async def test_reuse_when_listener_alive():
    """Existing master-owned tunnel with live listener is reused as-is."""
    pool = _make_pool()

    server = await asyncio.start_server(lambda _r, w: w.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        pool._tunnels["node:8001"] = TunnelSpec(
            remote_host="node", remote_port=8001, local_port=port, process=None
        )
        with mock.patch("asyncio.create_subprocess_exec", side_effect=AssertionError("no re-open")):
            result = await pool.open_tunnel("node", 8001, local_port=port)
        assert result == port
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_reopen_when_listener_dead():
    """Master-owned tunnel with dead listener is torn down and reopened."""
    pool = _make_pool()

    # Find a port with nothing listening on it.
    server = await asyncio.start_server(lambda _r, w: w.close(), "127.0.0.1", 0)
    dead_port = server.sockets[0].getsockname()[1]
    server.close()
    await server.wait_closed()

    pool._tunnels["node:8001"] = TunnelSpec(
        remote_host="node", remote_port=8001, local_port=dead_port, process=None
    )

    with (
        mock.patch(
            "asyncio.create_subprocess_exec",
            new=mock.AsyncMock(return_value=_FakeProc()),
        ) as spawn,
        mock.patch.object(pool, "_cancel_master_forward", new=mock.AsyncMock()) as cancel,
        mock.patch("asyncio.sleep", new=mock.AsyncMock()),
    ):
        result = await pool.open_tunnel("node", 8001, local_port=dead_port)

    assert result == dead_port
    cancel.assert_awaited_once()  # stale master forward cancelled
    assert spawn.await_count == 1  # tunnel re-established
    spec = pool._tunnels["node:8001"]
    assert spec.process is None  # rc==0 -> master-owned again


@pytest.mark.asyncio
async def test_listener_up_true_and_false():
    pool = _make_pool()

    server = await asyncio.start_server(lambda _r, w: w.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        assert await pool._listener_up(port) is True
    finally:
        server.close()
        await server.wait_closed()

    assert await pool._listener_up(port) is False
