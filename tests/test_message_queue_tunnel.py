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

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestTunnelModeBasics:
    """Tests for basic tunnel mode functionality."""

    def test_tunnel_mode_default_false(self):
        """tunnel_mode should default to False."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context'):
            mq = MessageQueue(role='server', queue_type='control')
            assert mq.tunnel_mode is False
            mq.close()

    def test_tunnel_mode_stored_true(self):
        """tunnel_mode=True should be stored."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context'):
            mq = MessageQueue(role='server', queue_type='control', tunnel_mode=True)
            assert mq.tunnel_mode is True
            mq.close()

    def test_tunnel_mode_from_env_var(self):
        """tunnel_mode should be set from UMRS_TUNNEL_MODE env var."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {'UMRS_TUNNEL_MODE': 'true'}):
            with patch('zmq.Context'):
                mq = MessageQueue(role='server', queue_type='control')
                assert mq.tunnel_mode is True
                mq.close()

    def test_tunnel_mode_env_var_false(self):
        """UMRS_TUNNEL_MODE=false should disable tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {'UMRS_TUNNEL_MODE': 'false'}):
            with patch('zmq.Context'):
                mq = MessageQueue(role='server', queue_type='control')
                assert mq.tunnel_mode is False
                mq.close()

    def test_tunnel_mode_explicit_overrides_env(self):
        """Explicit tunnel_mode parameter should override env var."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {'UMRS_TUNNEL_MODE': 'true'}):
            with patch('zmq.Context'):
                mq = MessageQueue(role='server', queue_type='control', tunnel_mode=False)
                assert mq.tunnel_mode is False
                mq.close()


class TestTunnelModeControlQueue:
    """Tests for control queue with tunnel mode."""

    def test_server_control_queue_binds_localhost_in_tunnel_mode(self):
        """Server control queue should bind to 127.0.0.1 in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context') as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role='server',
                queue_type='control',
                tunnel_mode=True,
            )

            # Server should bind to localhost in tunnel mode
            mock_socket.bind.assert_called_once()
            bind_call = str(mock_socket.bind.call_args)
            assert '127.0.0.1' in bind_call
            mq.close()

    def test_server_control_queue_binds_wildcard_in_normal_mode(self):
        """Server control queue should bind to tcp://* in normal mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context') as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role='server',
                queue_type='control',
                tunnel_mode=False,
            )

            # Server should bind to wildcard in normal mode
            mock_socket.bind.assert_called_once()
            bind_call = str(mock_socket.bind.call_args)
            assert 'tcp://*:' in bind_call
            mq.close()

    def test_client_control_queue_connects_localhost_in_tunnel_mode(self):
        """Client control queue should connect to 127.0.0.1 in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context') as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role='client',
                queue_type='control',
                server_ip='192.168.1.100',  # Should be ignored in tunnel mode
                tunnel_mode=True,
            )

            # Client should connect to localhost in tunnel mode
            mock_socket.connect.assert_called_once()
            connect_call = str(mock_socket.connect.call_args)
            assert '127.0.0.1' in connect_call
            assert '192.168.1.100' not in connect_call
            mq.close()

    def test_client_control_queue_connects_server_ip_in_normal_mode(self):
        """Client control queue should connect to server_ip in normal mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context') as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role='client',
                queue_type='control',
                server_ip='192.168.1.100',
                tunnel_mode=False,
            )

            # Client should connect to server_ip in normal mode
            mock_socket.connect.assert_called_once()
            connect_call = str(mock_socket.connect.call_args)
            assert '192.168.1.100' in connect_call
            mq.close()


class TestTunnelModeDataQueue:
    """Tests for data queue with tunnel mode."""

    def test_server_data_queue_binds_localhost_in_tunnel_mode(self):
        """Server data queue should bind to 127.0.0.1 in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context') as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role='server',
                queue_type='data',
                port=5556,
                tunnel_mode=True,
            )

            # Server should bind to localhost in tunnel mode
            mock_socket.bind.assert_called_once()
            bind_call = str(mock_socket.bind.call_args)
            assert '127.0.0.1' in bind_call
            mq.close()

    def test_client_data_queue_connects_localhost_in_tunnel_mode(self):
        """Client data queue should connect to 127.0.0.1 in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context') as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role='client',
                queue_type='data',
                server_ip='192.168.1.100',  # Should be ignored in tunnel mode
                port=5556,
                tunnel_mode=True,
            )

            # Client should connect to localhost in tunnel mode
            mock_socket.connect.assert_called_once()
            connect_call = str(mock_socket.connect.call_args)
            assert '127.0.0.1' in connect_call
            assert '192.168.1.100' not in connect_call
            mq.close()


class TestTunnelModePortConfig:
    """Tests for port configuration in tunnel mode."""

    def test_control_port_from_env_var(self):
        """Control port should be configurable via UMRS_CONTROL_PORT."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {'UMRS_CONTROL_PORT': '6666', 'UMRS_TUNNEL_MODE': 'true'}):
            with patch('zmq.Context') as mock_context:
                mock_socket = MagicMock()
                mock_context.return_value.socket.return_value = mock_socket

                mq = MessageQueue(
                    role='server',
                    queue_type='control',
                )

                bind_call = str(mock_socket.bind.call_args)
                assert '6666' in bind_call
                mq.close()

    def test_data_port_from_env_var(self):
        """Data port should be configurable via UMRS_DATA_PORT."""
        from umrs.core.MessageQueue import MessageQueue

        with patch.dict(os.environ, {'UMRS_DATA_PORT': '7777', 'UMRS_TUNNEL_MODE': 'true'}):
            with patch('zmq.Context') as mock_context:
                mock_socket = MagicMock()
                mock_context.return_value.socket.return_value = mock_socket

                mq = MessageQueue(
                    role='server',
                    queue_type='data',
                )

                bind_call = str(mock_socket.bind.call_args)
                assert '7777' in bind_call
                mq.close()


class TestGetDataQueueTunnelMode:
    """Tests for get_data_queue with tunnel mode."""

    def test_get_data_queue_inherits_tunnel_mode(self):
        """get_data_queue should inherit tunnel_mode from control queue."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context') as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            # Simulate server response
            mock_socket.recv_string.return_value = "tcp://192.168.1.100:5556"

            control_mq = MessageQueue(
                role='client',
                queue_type='control',
                tunnel_mode=True,
            )

            data_mq = control_mq.get_data_queue(client_id='test-client')

            # Data queue should inherit tunnel_mode
            assert data_mq.tunnel_mode is True
            control_mq.close()
            data_mq.close()

    def test_get_data_queue_explicit_tunnel_mode_override(self):
        """get_data_queue tunnel_mode parameter should override control queue setting."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context') as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            # Simulate server response
            mock_socket.recv_string.return_value = "tcp://192.168.1.100:5556"

            control_mq = MessageQueue(
                role='client',
                queue_type='control',
                tunnel_mode=True,
            )

            # Explicitly disable tunnel mode for data queue
            data_mq = control_mq.get_data_queue(client_id='test-client', tunnel_mode=False)

            assert data_mq.tunnel_mode is False
            control_mq.close()
            data_mq.close()

    def test_get_data_queue_uses_localhost_in_tunnel_mode(self):
        """get_data_queue should connect to localhost regardless of server response in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context') as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            # Server responds with its actual IP
            mock_socket.recv_string.return_value = "tcp://192.168.1.100:5556"

            control_mq = MessageQueue(
                role='client',
                queue_type='control',
                tunnel_mode=True,
            )

            data_mq = control_mq.get_data_queue(client_id='test-client')

            # Data queue should use localhost despite server responding with different IP
            assert data_mq.server_ip == "127.0.0.1"
            control_mq.close()
            data_mq.close()


class TestResetControlQueueTunnelMode:
    """Tests for reset_control_queue with tunnel mode."""

    def test_reset_control_queue_connects_localhost_in_tunnel_mode(self):
        """reset_control_queue should connect to localhost in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context') as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role='client',
                queue_type='control',
                server_ip='192.168.1.100',
                tunnel_mode=True,
            )

            mock_socket.reset_mock()
            mq.reset_control_queue()

            # Should connect to localhost after reset
            mock_socket.connect.assert_called()
            connect_call = str(mock_socket.connect.call_args)
            assert '127.0.0.1' in connect_call
            mq.close()

    def test_reset_control_queue_connects_server_ip_in_normal_mode(self):
        """reset_control_queue should connect to server_ip in normal mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context') as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            mq = MessageQueue(
                role='client',
                queue_type='control',
                server_ip='192.168.1.100',
                tunnel_mode=False,
            )

            mock_socket.reset_mock()
            mq.reset_control_queue()

            # Should connect to server_ip after reset
            mock_socket.connect.assert_called()
            connect_call = str(mock_socket.connect.call_args)
            assert '192.168.1.100' in connect_call
            mq.close()


class TestDocumentation:
    """Tests to verify documentation is present."""

    def test_init_docstring_mentions_tunnel_mode(self):
        """__init__ docstring should document tunnel_mode."""
        from umrs.core.MessageQueue import MessageQueue

        docstring = MessageQueue.__init__.__doc__
        assert 'tunnel_mode' in docstring
        assert 'SSH' in docstring or 'localhost' in docstring.lower()

    def test_get_data_queue_docstring(self):
        """get_data_queue docstring should document tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        docstring = MessageQueue.get_data_queue.__doc__
        assert 'tunnel_mode' in docstring


class TestSSHTunnelScenario:
    """End-to-end scenario tests for SSH tunnel usage."""

    def test_full_tunnel_scenario_both_use_localhost(self):
        """Both server and client should use localhost in tunnel mode."""
        from umrs.core.MessageQueue import MessageQueue

        with patch('zmq.Context') as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket

            # Server side (on RCCS)
            server_mq = MessageQueue(
                role='server',
                queue_type='control',
                tunnel_mode=True,
            )
            server_bind_call = str(mock_socket.bind.call_args)
            assert '127.0.0.1' in server_bind_call
            server_mq.close()

            mock_socket.reset_mock()

            # Client side (on HUCC, through SSH tunnel)
            client_mq = MessageQueue(
                role='client',
                queue_type='control',
                server_ip='rccs-server.example.com',  # Ignored in tunnel mode
                tunnel_mode=True,
            )
            client_connect_call = str(mock_socket.connect.call_args)
            assert '127.0.0.1' in client_connect_call
            assert 'rccs-server.example.com' not in client_connect_call
            client_mq.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
