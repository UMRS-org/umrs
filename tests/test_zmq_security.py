import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


KEY_A = "A" * 40
KEY_B = "B" * 40
KEY_C = "C" * 40


class TestZMQSecurityConfig:
    def test_default_auth_type_is_none(self):
        from umrs.core.zmq_security import ZMQSecurityConfig

        config = ZMQSecurityConfig.from_env({})

        assert config.auth_type == "none"
        assert config.enabled is False

    def test_curve_config_from_env(self):
        from umrs.core.zmq_security import ZMQSecurityConfig

        config = ZMQSecurityConfig.from_env({
            "UMRS_ZMQ_AUTH_TYPE": "curve",
            "UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY": KEY_A,
            "UMRS_ZMQ_CURVE_SERVER_SECRET_KEY": KEY_B,
            "UMRS_ZMQ_CURVE_CLIENT_PUBLIC_KEY": KEY_C,
            "UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY": KEY_A,
            "UMRS_ZMQ_CURVE_ALLOW_ANY_CLIENT_KEY": "true",
        })

        assert config.enabled is True
        assert config.auth_type == "curve"
        assert config.server_public_key == KEY_A
        assert config.allow_any_client_key is True

    def test_client_socket_options_are_applied(self):
        from umrs.core.zmq_security import ZMQSecurityConfig

        config = ZMQSecurityConfig(
            auth_type="curve",
            server_public_key=KEY_A,
            client_public_key=KEY_B,
            client_secret_key=KEY_C,
        )
        socket = MagicMock()

        config.apply_to_socket(socket, server=False)

        assert socket.curve_serverkey == KEY_A.encode("ascii")
        assert socket.curve_publickey == KEY_B.encode("ascii")
        assert socket.curve_secretkey == KEY_C.encode("ascii")

    def test_server_requires_trusted_client_keys_or_explicit_allow_any(self):
        from umrs.core.zmq_security import ZMQSecurityConfig

        config = ZMQSecurityConfig(
            auth_type="curve",
            server_public_key=KEY_A,
            server_secret_key=KEY_B,
        )

        with pytest.raises(ValueError, match="UMRS_ZMQ_CURVE_CLIENT_KEYS_DIR"):
            config.apply_to_socket(MagicMock(), server=True, context=MagicMock())

    def test_server_socket_options_and_authenticator_are_applied(self):
        from umrs.core.zmq_security import ZMQSecurityConfig

        config = ZMQSecurityConfig(
            auth_type="curve",
            server_public_key=KEY_A,
            server_secret_key=KEY_B,
            allow_any_client_key=True,
        )
        socket = MagicMock()
        auth = MagicMock()

        with patch("zmq.auth.thread.ThreadAuthenticator", return_value=auth):
            returned = config.apply_to_socket(socket, server=True, context=MagicMock())

        assert returned is auth
        assert socket.curve_publickey == KEY_A.encode("ascii")
        assert socket.curve_secretkey == KEY_B.encode("ascii")
        assert socket.curve_server is True
        auth.start.assert_called_once()
        auth.configure_curve.assert_called_once()

    def test_invalid_curve_key_length_is_rejected(self):
        from umrs.core.zmq_security import ZMQSecurityConfig

        config = ZMQSecurityConfig(
            auth_type="curve",
            server_public_key="short",
            client_public_key=KEY_B,
            client_secret_key=KEY_C,
        )

        with pytest.raises(ValueError, match="40 characters"):
            config.apply_to_socket(MagicMock(), server=False)


class TestMessageQueueZMQSecurity:
    def test_client_control_queue_applies_curve_before_connect(self):
        from umrs.core.MessageQueue import MessageQueue

        env = {
            "UMRS_ZMQ_AUTH_TYPE": "curve",
            "UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY": KEY_A,
            "UMRS_ZMQ_CURVE_CLIENT_PUBLIC_KEY": KEY_B,
            "UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY": KEY_C,
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("zmq.Context") as mock_context:
                mock_socket = MagicMock()
                mock_context.return_value.socket.return_value = mock_socket

                mq = MessageQueue(role="client", queue_type="control")

                assert mock_socket.curve_serverkey == KEY_A.encode("ascii")
                assert mock_socket.curve_publickey == KEY_B.encode("ascii")
                assert mock_socket.curve_secretkey == KEY_C.encode("ascii")
                mock_socket.connect.assert_called_once()
                mq.close()

    def test_server_control_queue_applies_curve_and_stops_authenticator(self):
        from umrs.core.MessageQueue import MessageQueue

        env = {
            "UMRS_ZMQ_AUTH_TYPE": "curve",
            "UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY": KEY_A,
            "UMRS_ZMQ_CURVE_SERVER_SECRET_KEY": KEY_B,
            "UMRS_ZMQ_CURVE_ALLOW_ANY_CLIENT_KEY": "true",
        }
        auth = MagicMock()
        with patch.dict(os.environ, env, clear=False):
            with patch("zmq.Context") as mock_context:
                with patch("zmq.auth.thread.ThreadAuthenticator", return_value=auth):
                    mock_socket = MagicMock()
                    mock_context.return_value.socket.return_value = mock_socket

                    mq = MessageQueue(role="server", queue_type="control")

                    assert mock_socket.curve_publickey == KEY_A.encode("ascii")
                    assert mock_socket.curve_secretkey == KEY_B.encode("ascii")
                    assert mock_socket.curve_server is True
                    mock_socket.bind.assert_called_once()
                    mq.close()

        auth.stop.assert_called_once()


class TestWorkerZMQSecurity:
    def test_remote_worker_applies_curve_to_req_socket(self):
        from umrs.server.step_executor import _RemoteWorker

        env = {
            "UMRS_ZMQ_AUTH_TYPE": "curve",
            "UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY": KEY_A,
            "UMRS_ZMQ_CURVE_CLIENT_PUBLIC_KEY": KEY_B,
            "UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY": KEY_C,
        }
        socket = MagicMock()
        context = MagicMock()
        context.socket.return_value = socket

        with patch.dict(os.environ, env, clear=False):
            worker = _RemoteWorker(
                address="tcp://127.0.0.1:59999",
                zmq_context=context,
                timeout=1.0,
            )

        assert socket.curve_serverkey == KEY_A.encode("ascii")
        assert socket.curve_publickey == KEY_B.encode("ascii")
        assert socket.curve_secretkey == KEY_C.encode("ascii")
        socket.connect.assert_called_once_with("tcp://127.0.0.1:59999")


class TestCurveKeyMap:
    def test_empty_directory_produces_empty_map(self, tmp_path):
        from umrs.core.zmq_security import CurveKeyMap

        km = CurveKeyMap.from_directory(str(tmp_path))
        assert len(km) == 0
        assert km.username_for_key("anything") is None

    def test_none_directory_produces_empty_map(self):
        from umrs.core.zmq_security import CurveKeyMap

        km = CurveKeyMap.from_directory(None)
        assert len(km) == 0

    def test_loads_keys_and_maps_to_username(self, tmp_path):
        from umrs.core.zmq_security import CurveKeyMap, generate_curve_keypair

        generate_curve_keypair(str(tmp_path), "alice")
        generate_curve_keypair(str(tmp_path), "bob")

        km = CurveKeyMap.from_directory(str(tmp_path))

        assert len(km) == 2
        # Read back the actual public key from the generated file
        from zmq.auth import load_certificate

        alice_pub, _ = load_certificate(str(tmp_path / "alice.key"))
        alice_z85 = alice_pub.decode("ascii")
        assert km.username_for_key(alice_z85) == "alice"

    def test_reload_picks_up_new_keys(self, tmp_path):
        from umrs.core.zmq_security import CurveKeyMap, generate_curve_keypair

        km = CurveKeyMap.from_directory(str(tmp_path))
        assert len(km) == 0

        generate_curve_keypair(str(tmp_path), "carol")
        km.reload()
        assert len(km) == 1

    def test_nonexistent_directory_is_safe(self):
        from umrs.core.zmq_security import CurveKeyMap

        km = CurveKeyMap.from_directory("/nonexistent/path/abc123")
        assert len(km) == 0


class TestGenerateCurveKeypair:
    def test_generates_key_files(self, tmp_path):
        from umrs.core.zmq_security import generate_curve_keypair

        pub_path, sec_path = generate_curve_keypair(str(tmp_path), "testuser")

        assert os.path.exists(pub_path)
        assert os.path.exists(sec_path)
        assert pub_path.endswith("testuser.key")
        assert sec_path.endswith("testuser.key_secret")

        # Secret file should be mode 600
        import stat
        mode = os.stat(sec_path).st_mode & 0o777
        assert mode == 0o600

    def test_generated_keys_are_loadable(self, tmp_path):
        from umrs.core.zmq_security import generate_curve_keypair

        pub_path, sec_path = generate_curve_keypair(str(tmp_path), "loadtest")

        from zmq.auth import load_certificate

        pub_key, _ = load_certificate(pub_path)
        assert len(pub_key) == 40  # Z85 encoded

        pub2, sec_key = load_certificate(sec_path)
        assert pub2 == pub_key
        assert len(sec_key) == 40

    def test_creates_directory_if_missing(self, tmp_path):
        from umrs.core.zmq_security import generate_curve_keypair

        nested = str(tmp_path / "a" / "b" / "c")
        pub_path, _ = generate_curve_keypair(nested, "nested")
        assert os.path.exists(pub_path)


class TestCurveIntegration:
    """Live ZMQ CURVE handshake test — no mocks."""

    def test_curve_authenticated_handshake_succeeds(self, tmp_path):
        """Server and client with matching keys can exchange messages."""
        import zmq
        from umrs.core.zmq_security import ZMQSecurityConfig, generate_curve_keypair

        # Generate server and client keys
        server_keys_dir = str(tmp_path / "server")
        client_keys_dir = str(tmp_path / "clients")
        generate_curve_keypair(server_keys_dir, "server")
        generate_curve_keypair(client_keys_dir, "alice")

        from zmq.auth import load_certificate

        _, server_secret = load_certificate(str(tmp_path / "server" / "server.key_secret"))
        server_pub, _ = load_certificate(str(tmp_path / "server" / "server.key"))
        client_pub, _ = load_certificate(str(tmp_path / "clients" / "alice.key_secret"))
        _, client_secret = load_certificate(str(tmp_path / "clients" / "alice.key_secret"))

        # Copy the client's .key to a trusted-clients dir for the server
        trusted_dir = str(tmp_path / "trusted")
        os.makedirs(trusted_dir)
        import shutil
        shutil.copy(str(tmp_path / "clients" / "alice.key"), trusted_dir)

        server_config = ZMQSecurityConfig(
            auth_type="curve",
            server_public_key=server_pub.decode("ascii"),
            server_secret_key=server_secret.decode("ascii"),
            client_keys_dir=trusted_dir,
        )
        client_config = ZMQSecurityConfig(
            auth_type="curve",
            server_public_key=server_pub.decode("ascii"),
            client_public_key=client_pub.decode("ascii"),
            client_secret_key=client_secret.decode("ascii"),
        )

        ctx = zmq.Context()
        try:
            # Server: REP socket
            server_sock = ctx.socket(zmq.REP)
            auth = server_config.apply_to_socket(server_sock, server=True, context=ctx)
            server_sock.bind("tcp://127.0.0.1:*")
            endpoint = server_sock.getsockopt_string(zmq.LAST_ENDPOINT)

            # Client: REQ socket
            client_sock = ctx.socket(zmq.REQ)
            client_config.apply_to_socket(client_sock, server=False)
            client_sock.connect(endpoint)

            # Exchange a message
            client_sock.send_string("PING")
            msg = server_sock.recv_string()
            assert msg == "PING"

            server_sock.send_string("PONG")
            reply = client_sock.recv_string()
            assert reply == "PONG"
        finally:
            client_sock.close(linger=0)
            server_sock.close(linger=0)
            if auth:
                auth.stop()
            ctx.term()

    def test_curve_rejects_unknown_client_key(self, tmp_path):
        """A client with an untrusted key is rejected by the server."""
        import zmq
        from umrs.core.zmq_security import ZMQSecurityConfig, generate_curve_keypair

        # Generate server keys and a trusted client
        server_keys_dir = str(tmp_path / "server")
        trusted_dir = str(tmp_path / "trusted")
        generate_curve_keypair(server_keys_dir, "server")
        os.makedirs(trusted_dir)
        # trusted_dir is empty — no clients are trusted

        # Generate an untrusted client
        untrusted_dir = str(tmp_path / "untrusted")
        generate_curve_keypair(untrusted_dir, "mallory")

        from zmq.auth import load_certificate

        _, server_secret = load_certificate(str(tmp_path / "server" / "server.key_secret"))
        server_pub, _ = load_certificate(str(tmp_path / "server" / "server.key"))
        client_pub, _ = load_certificate(str(tmp_path / "untrusted" / "mallory.key_secret"))
        _, client_secret = load_certificate(str(tmp_path / "untrusted" / "mallory.key_secret"))

        server_config = ZMQSecurityConfig(
            auth_type="curve",
            server_public_key=server_pub.decode("ascii"),
            server_secret_key=server_secret.decode("ascii"),
            client_keys_dir=trusted_dir,
        )
        client_config = ZMQSecurityConfig(
            auth_type="curve",
            server_public_key=server_pub.decode("ascii"),
            client_public_key=client_pub.decode("ascii"),
            client_secret_key=client_secret.decode("ascii"),
        )

        ctx = zmq.Context()
        try:
            server_sock = ctx.socket(zmq.REP)
            auth = server_config.apply_to_socket(server_sock, server=True, context=ctx)
            server_sock.bind("tcp://127.0.0.1:*")
            endpoint = server_sock.getsockopt_string(zmq.LAST_ENDPOINT)

            client_sock = ctx.socket(zmq.REQ)
            client_config.apply_to_socket(client_sock, server=False)
            client_sock.connect(endpoint)

            # The send should succeed (ZMQ buffers), but the server should
            # never receive it because ZAP rejects the client key.
            client_sock.send_string("SHOULD_FAIL", zmq.NOBLOCK)
            import time
            time.sleep(0.3)  # Give ZAP time to reject

            # Server should NOT have received anything
            events = server_sock.poll(timeout=500)
            assert events == 0, "Server received a message from an untrusted client"
        finally:
            client_sock.close(linger=0)
            server_sock.close(linger=0)
            if auth:
                auth.stop()
            ctx.term()


class TestAdminZmqKeygen:
    """Test the CLI keygen command."""

    def test_keygen_generates_files(self, tmp_path):
        from umrs.cli.admin.main import main as admin_main

        rc = admin_main(["zmq", "keygen", "testuser", "-o", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "testuser.key").exists()
        assert (tmp_path / "testuser.key_secret").exists()

    def test_keygen_server_mode(self, tmp_path):
        from umrs.cli.admin.main import main as admin_main

        rc = admin_main(["zmq", "keygen", "server", "--server", "-o", str(tmp_path), "--json"])
        assert rc == 0
        assert (tmp_path / "server.key").exists()
        assert (tmp_path / "server.key_secret").exists()

    def test_list_keys_empty(self, tmp_path):
        from umrs.cli.admin.main import main as admin_main

        rc = admin_main(["zmq", "list-keys", "--keys-dir", str(tmp_path)])
        assert rc == 0

    def test_list_keys_after_keygen(self, tmp_path):
        from umrs.cli.admin.main import main as admin_main

        admin_main(["zmq", "keygen", "alice", "-o", str(tmp_path)])
        admin_main(["zmq", "keygen", "bob", "-o", str(tmp_path)])

        rc = admin_main(["zmq", "list-keys", "--keys-dir", str(tmp_path), "--json"])
        assert rc == 0