"""
Tests for MessageQueue tunnel connection mode.

Tunnel mode allows ZeroMQ communication over SSH tunnels, which is
required for HPC environments (like HUCC↔RCCS) where:
- Compute nodes have both inbound AND outbound TCP blocked
- Only SSH tunnels through gateways are allowed
- Both client and server use localhost (127.0.0.1)
- SSH tunnel provides the actual transport layer

Example SSH tunnel command:
    ssh -L 5555:localhost:5555 -L 5556:localhost:5556 user@rccs-server -N
"""

import pytest
from unittest.mock import MagicMock, patch
import os
import json
import logging

import sys

# splice removed (ADR-053): umrs installed editable in .venv


class TestTunnelModeBasics:
    """Tests for basic tunnel mode functionality."""

    def test_tunnel_mode_default_false(self):
        """tunnel_mode should default to False."""
        from umrs.core.MessageQueue import MessageQueue

        with patch("zmq.Context"):
            mq = MessageQueue(role="server", queue_type="control")
            assert mq.tunnel_mode is False
            mq.close()

    def test_tunnel_mode_stored_true(self):
        """tunnel_mode=True should be stored."""
        from umrs.core.MessageQueue import MessageQueue

        with patch("zmq.Context"):
            mq = MessageQueue(role="server", queue_type="control", tunnel_mode=True)
            assert mq.tunnel_mode is True
            mq.close()

    def test_tunnel_mode_from_env_var(self):
        """tunnel_mode should be set from UMRS_TUNNEL_MODE env var."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {"UMRS_TUNNEL_MODE": "true"}):
            with patch("zmq.Context"):
                mq = MessageQueue(role="server", queue_type="control")
                assert mq.tunnel_mode is True
                mq.close()

    def test_tunnel_mode_env_var_false(self):
        """UMRS_TUNNEL_MODE=false should disable tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {"UMRS_TUNNEL_MODE": "false"}):
            with patch("zmq.Context"):
                mq = MessageQueue(role="server", queue_type="control")
                assert mq.tunnel_mode is False
                mq.close()

    def test_tunnel_mode_explicit_overrides_env(self):
        """Explicit tunnel_mode parameter should override env var."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {"UMRS_TUNNEL_MODE": "true"}):
            with patch("zmq.Context"):
                mq = MessageQueue(role="server", queue_type="control", tunnel_mode=False)
                assert mq.tunnel_mode is False
                mq.close()


class TestControlMessageLogging:
    """Tests for control-plane log redaction."""

    def test_send_control_request_uses_multipart_auth_frame(self):
        """Admin tokens should not be embedded in colon-delimited commands."""
        from umrs.core.MessageQueue import MessageQueue
        from umrs.core.TaskCmdStatus import TaskCmd

        token = "secret:token:with:colons"
        mq = object.__new__(MessageQueue)
        mq.role = "client"
        mq.type = "control"
        mq.debug = False
        mq.socket = MagicMock()

        mq.send_control_request(
            [TaskCmd.CREATE_DATA_QUEUE, "alice"],
            admin_token=token,
        )

        frames = mq.socket.send_multipart.call_args.args[0]
        assert frames[0] == b"CREATE_DATA_QUEUE:alice"
        assert token.encode("utf-8") not in frames[0]
        assert json.loads(frames[1].decode("utf-8")) == {"admin_token": token}

    def test_recv_string_decodes_multipart_auth_frame(self):
        """Server control receive should keep auth out of the command body."""
        from umrs.core.MessageQueue import MessageQueue

        token = "secret:token:with:colons"
        mq = object.__new__(MessageQueue)
        mq.role = "server"
        mq.type = "control"
        mq.socket = MagicMock()
        mq.zmq_security = MagicMock(is_session_mode=False)
        mq._zap_user_ids = {}
        mq.socket.recv_multipart.return_value = [
            b"client-1",
            b"ENROLL_USER:alice:1",
            json.dumps({"admin_token": token}).encode("utf-8"),
        ]

        client_id, message = mq.recv_string()

        assert client_id == b"client-1"
        assert message == "ENROLL_USER:alice:1"
        assert mq._last_control_auth == {"admin_token": token}

    def test_control_auth_token_uses_multipart_frame_only(self):
        """Admin tokens must come from the multipart auth frame only."""
        from umrs.core.MessageQueue import MessageQueue

        mq = object.__new__(MessageQueue)
        mq._last_control_auth = {"admin_token": "new-token"}

        assert mq._control_auth_token() == "new-token"
        mq._last_control_auth = {}
        assert mq._control_auth_token() is None

    def test_redact_control_message_hides_admin_token(self):
        """Admin tokens in legacy colon-delimited messages must not be logged."""
        from umrs.core.MessageQueue import MessageQueue

        token = "umrs-admin-secret-token"
        summary = MessageQueue._redact_control_message(f"CREATE_DATA_QUEUE:alice:{token}")

        assert token not in summary
        assert "CREATE_DATA_QUEUE" in summary
        assert "client_id=alice" in summary
        assert "has_auth=true" in summary

    def test_redact_control_message_hides_json_payload(self):
        """JSON payloads can contain secrets and should be summarized only."""
        from umrs.core.MessageQueue import MessageQueue

        payload = '{"api_key":"secret-value","model":"demo"}'
        summary = MessageQueue._redact_control_message(f"MODEL_REGISTER:alice:{payload}")

        assert "secret-value" not in summary
        assert "api_key" not in summary
        assert "MODEL_REGISTER" in summary
        assert "client_id=alice" in summary

    def test_wait_for_request_does_not_print_or_log_control_payload(
        self,
        caplog,
        capsys,
    ):
        """Full control request handling must not leak payloads to logs or stdio."""
        from umrs.core.MessageQueue import MessageQueue
        from umrs.core.TaskCmdStatus import TaskCmd

        secret = "secret-token-value"
        mq = object.__new__(MessageQueue)
        mq.recv_string = MagicMock(
            return_value=(b"client-1", f"WAIT_OUTPUT:alice:bucket;0:{secret}")
        )

        caplog.set_level(logging.INFO)
        mq.role = "server"

        result_cmd, payload = mq.server_control_queue_loop()

        captured = capsys.readouterr()
        assert result_cmd == TaskCmd.WAIT_OUTPUT
        assert payload[0] == "alice"
        assert secret not in captured.out
        assert secret not in captured.err
        assert secret not in caplog.text
        assert "Received control request" in caplog.text
        assert "cmd=WAIT_OUTPUT" in caplog.text


class TestTunnelModeControlQueue:
    """Tests for control queue with tunnel mode."""

    def test_server_control_queue_binds_localhost_in_tunnel_mode(self):
        """Server control queue should bind to 127.0.0.1 in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch("zmq.Context") as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role="server",
                queue_type="control",
                tunnel_mode=True,
            )

            # Server should bind to localhost in tunnel mode
            mock_socket.bind.assert_called_once()
            bind_call = str(mock_socket.bind.call_args)
            assert "127.0.0.1" in bind_call
            mq.close()

    def test_server_control_queue_binds_wildcard_in_normal_mode(self):
        """Server control queue should bind to tcp://* in normal mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch("zmq.Context") as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role="server",
                queue_type="control",
                tunnel_mode=False,
            )

            # Server should bind to wildcard in normal mode
            mock_socket.bind.assert_called_once()
            bind_call = str(mock_socket.bind.call_args)
            assert "tcp://*:" in bind_call
            mq.close()

    def test_client_control_queue_connects_localhost_in_tunnel_mode(self):
        """Client control queue should connect to 127.0.0.1 in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch("zmq.Context") as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role="client",
                queue_type="control",
                server_ip="192.168.1.100",  # Should be ignored in tunnel mode
                tunnel_mode=True,
            )

            # Client should connect to localhost in tunnel mode
            mock_socket.connect.assert_called_once()
            connect_call = str(mock_socket.connect.call_args)
            assert "127.0.0.1" in connect_call
            assert "192.168.1.100" not in connect_call
            mq.close()

    def test_client_control_queue_connects_server_ip_in_normal_mode(self):
        """Client control queue should connect to server_ip in normal mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {"UMRS_SERVER_HOST": ""}, clear=False):
            with patch("zmq.Context") as mock_context:
                mock_socket = MagicMock()
                mock_context.return_value.socket.return_value = mock_socket

                mq = MessageQueue(
                    role="client",
                    queue_type="control",
                    server_ip="192.168.1.100",
                    tunnel_mode=False,
                )

                # Client should connect to server_ip in normal mode
                mock_socket.connect.assert_called_once()
                connect_call = str(mock_socket.connect.call_args)
                assert "192.168.1.100" in connect_call
                mq.close()


class TestTunnelModeDataQueue:
    """Tests for data queue with tunnel mode."""

    def test_server_data_queue_binds_localhost_in_tunnel_mode(self):
        """Server data queue should bind to 127.0.0.1 in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch("zmq.Context") as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role="server",
                queue_type="data",
                port=5556,
                tunnel_mode=True,
            )

            # Server should bind to localhost in tunnel mode
            mock_socket.bind.assert_called_once()
            bind_call = str(mock_socket.bind.call_args)
            assert "127.0.0.1" in bind_call
            mq.close()

    def test_client_data_queue_connects_localhost_in_tunnel_mode(self):
        """Client data queue should connect to 127.0.0.1 in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch("zmq.Context") as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role="client",
                queue_type="data",
                server_ip="192.168.1.100",  # Should be ignored in tunnel mode
                port=5556,
                tunnel_mode=True,
            )

            # Client should connect to localhost in tunnel mode
            mock_socket.connect.assert_called_once()
            connect_call = str(mock_socket.connect.call_args)
            assert "127.0.0.1" in connect_call
            assert "192.168.1.100" not in connect_call
            mq.close()


class TestTunnelModePortConfig:
    """Tests for port configuration in tunnel mode."""

    def test_control_port_from_env_var(self):
        """Control port should be configurable via UMRS_CONTROL_PORT."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {"UMRS_CONTROL_PORT": "6666", "UMRS_TUNNEL_MODE": "true"}):
            with patch("zmq.Context") as mock_context:
                mock_socket = MagicMock()
                mock_context.return_value.socket.return_value = mock_socket

                mq = MessageQueue(
                    role="server",
                    queue_type="control",
                )

                bind_call = str(mock_socket.bind.call_args)
                assert "6666" in bind_call
                mq.close()

    def test_data_port_from_env_var(self):
        """Data port should be configurable via UMRS_DATA_PORT."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {"UMRS_DATA_PORT": "7777", "UMRS_TUNNEL_MODE": "true"}):
            with patch("zmq.Context") as mock_context:
                mock_socket = MagicMock()
                mock_context.return_value.socket.return_value = mock_socket

                mq = MessageQueue(
                    role="server",
                    queue_type="data",
                )

                bind_call = str(mock_socket.bind.call_args)
                assert "7777" in bind_call
                mq.close()

    def test_data_port_range_from_env_var_uses_range_start(self):
        """Data queues should use the start of UMRS_DATA_PORT_RANGE by default."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(
            os.environ,
            {"UMRS_DATA_PORT_RANGE": "7780-7784", "UMRS_TUNNEL_MODE": "true"},
            clear=False,
        ):
            with patch("zmq.Context") as mock_context:
                mock_socket = MagicMock()
                mock_context.return_value.socket.return_value = mock_socket

                mq = MessageQueue(
                    role="server",
                    queue_type="data",
                )

                bind_call = str(mock_socket.bind.call_args)
                assert "7780" in bind_call
                mq.close()

    def test_ensure_client_data_queue_allocates_within_configured_range(self):
        """Server allocation should stay inside UMRS_DATA_PORT_RANGE and skip reserved ports."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {"UMRS_DATA_PORT_RANGE": "7800-7802"}, clear=False):
            with patch("zmq.Context") as mock_context:
                mock_socket = MagicMock()
                mock_context.return_value.socket.return_value = mock_socket

                mq = MessageQueue(role="server", queue_type="control", tunnel_mode=True)
                mq.client_dictionary["other-client"] = {"port": 7800}

                port, port_range = mq._ensure_client_data_queue("new-client")

                assert port == 7801
                assert port_range.label == "7800-7802"
                assert mq.client_dictionary["new-client"]["port"] == 7801
                mq.close()

    def test_ensure_client_data_queue_reports_exhausted_range(self):
        """Server allocation should fail cleanly when no data ports remain."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {"UMRS_DATA_PORT_RANGE": "7810-7811"}, clear=False):
            with patch("zmq.Context") as mock_context:
                mock_socket = MagicMock()
                mock_context.return_value.socket.return_value = mock_socket

                mq = MessageQueue(role="server", queue_type="control", tunnel_mode=True)
                mq.client_dictionary["client-a"] = {"port": 7810}
                mq.client_dictionary["client-b"] = {"port": 7811}

                port, port_range = mq._ensure_client_data_queue("client-c")

                assert port is None
                assert port_range.label == "7810-7811"
                mq.close()


class TestGetDataQueueTunnelMode:
    """Tests for get_data_queue with tunnel mode."""

    def test_get_data_queue_inherits_tunnel_mode(self):
        """get_data_queue should inherit tunnel_mode from control queue."""
        from umrs.core.MessageQueue import MessageQueue

        with patch("zmq.Context") as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            # Simulate server response
            mock_socket.recv_string.return_value = "tcp://192.168.1.100:5556"

            control_mq = MessageQueue(
                role="client",
                queue_type="control",
                tunnel_mode=True,
            )

            data_mq = control_mq.get_data_queue(client_id="test-client")

            # Data queue should inherit tunnel_mode
            assert data_mq.tunnel_mode is True
            control_mq.close()
            data_mq.close()

    def test_get_data_queue_explicit_tunnel_mode_override(self):
        """get_data_queue tunnel_mode parameter should override control queue setting."""
        from umrs.core.MessageQueue import MessageQueue

        with patch("zmq.Context") as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            # Simulate server response
            mock_socket.recv_string.return_value = "tcp://192.168.1.100:5556"

            control_mq = MessageQueue(
                role="client",
                queue_type="control",
                tunnel_mode=True,
            )

            # Explicitly disable tunnel mode for data queue
            data_mq = control_mq.get_data_queue(client_id="test-client", tunnel_mode=False)

            assert data_mq.tunnel_mode is False
            control_mq.close()
            data_mq.close()

    def test_get_data_queue_uses_localhost_in_tunnel_mode(self):
        """get_data_queue should connect to localhost regardless of server response in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch("zmq.Context") as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            # Server responds with its actual IP
            mock_socket.recv_string.return_value = "tcp://192.168.1.100:5556"

            control_mq = MessageQueue(
                role="client",
                queue_type="control",
                tunnel_mode=True,
            )

            data_mq = control_mq.get_data_queue(client_id="test-client")

            # Data queue should use localhost despite server responding with different IP
            assert data_mq.server_ip == "127.0.0.1"
            control_mq.close()
            data_mq.close()

    def test_get_data_queue_reports_range_exhaustion(self):
        """get_data_queue should raise a clear error when the server exhausts the configured range."""
        from umrs.core.MessageQueue import MessageQueue

        with patch("zmq.Context") as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket
            mock_socket.recv_multipart.return_value = [b"NO_DATA_PORT_AVAILABLE:7820-7821"]

            control_mq = MessageQueue(
                role="client",
                queue_type="control",
                tunnel_mode=True,
            )

            with pytest.raises(RuntimeError, match="7820-7821"):
                control_mq.get_data_queue(client_id="test-client")

            control_mq.close()


class TestResetControlQueueTunnelMode:
    """Tests for reset_control_queue with tunnel mode."""

    def test_reset_control_queue_connects_localhost_in_tunnel_mode(self):
        """reset_control_queue should connect to localhost in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch("zmq.Context") as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role="client",
                queue_type="control",
                server_ip="192.168.1.100",
                tunnel_mode=True,
            )

            mock_socket.reset_mock()
            mq.reset_control_queue()

            # Should connect to localhost after reset
            mock_socket.connect.assert_called()
            connect_call = str(mock_socket.connect.call_args)
            assert "127.0.0.1" in connect_call
            mq.close()

    def test_reset_control_queue_connects_server_ip_in_normal_mode(self):
        """reset_control_queue should connect to server_ip in normal mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {"UMRS_SERVER_HOST": ""}, clear=False):
            with patch("zmq.Context") as mock_context:
                mock_socket = MagicMock()
                mock_context.return_value.socket.return_value = mock_socket

                mq = MessageQueue(
                    role="client",
                    queue_type="control",
                    server_ip="192.168.1.100",
                    tunnel_mode=False,
                )

                mock_socket.reset_mock()
                mq.reset_control_queue()

                # Should connect to server_ip after reset
                mock_socket.connect.assert_called()
                connect_call = str(mock_socket.connect.call_args)
                assert "192.168.1.100" in connect_call
                mq.close()


class TestDocumentation:
    """Tests to verify documentation is present."""

    def test_init_docstring_mentions_tunnel_mode(self):
        """__init__ docstring should document tunnel_mode."""
        from umrs.core.MessageQueue import MessageQueue

        docstring = MessageQueue.__init__.__doc__
        assert "tunnel_mode" in docstring
        assert "SSH" in docstring or "localhost" in docstring.lower()

    def test_get_data_queue_docstring(self):
        """get_data_queue docstring should document tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        docstring = MessageQueue.get_data_queue.__doc__
        assert "tunnel_mode" in docstring


class TestSSHTunnelScenario:
    """End-to-end scenario tests for SSH tunnel usage."""

    def test_full_tunnel_scenario_both_use_localhost(self):
        """Both server and client should use localhost in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch("zmq.Context") as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            # Server side (on RCCS)
            server_mq = MessageQueue(
                role="server",
                queue_type="control",
                tunnel_mode=True,
            )
            server_bind_call = str(mock_socket.bind.call_args)
            assert "127.0.0.1" in server_bind_call
            server_mq.close()

            mock_socket.reset_mock()

            # Client side (on HUCC, through SSH tunnel)
            client_mq = MessageQueue(
                role="client",
                queue_type="control",
                server_ip="rccs-server.example.com",  # Ignored in tunnel mode
                tunnel_mode=True,
            )
            client_connect_call = str(mock_socket.connect.call_args)
            assert "127.0.0.1" in client_connect_call
            assert "rccs-server.example.com" not in client_connect_call
            client_mq.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
