import collections
import contextlib
import json
import logging
import os
import random
import shutil
import socket
import tempfile
import time

from umrs.core.TaskCmdStatus import TaskCmd
from umrs.shared.auth import AuthContext, UserRole, create_auth_context_from_request
from umrs.shared.connection_profiles import DataPortRange, parse_data_port_range

try:
    import zmq
except ImportError as e:
    raise ImportError("Cannot find zeroMQ module") from e

logger = logging.getLogger(__name__)

BASE_PORT = 5555


def _validate_data_port_range(port_range: DataPortRange) -> DataPortRange:
    if port_range.contains(BASE_PORT):
        raise ValueError(
            f"Data port range {port_range.label} must not include BASE_PORT ({BASE_PORT})"
        )
    return port_range


def _resolve_default_data_port_range() -> DataPortRange:
    port_range = (
        parse_data_port_range(os.environ.get("UMRS_DATA_PORT_RANGE"))
        or parse_data_port_range(os.environ.get("UMRS_DATA_PORT"))
        or parse_data_port_range(os.environ.get("UMRS_DATA_BASE_PORT"))
        or DataPortRange(BASE_PORT + 1, BASE_PORT + 10)
    )
    return _validate_data_port_range(port_range)


DATA_PORT_RANGE = _resolve_default_data_port_range()
DATA_BASE_PORT = DATA_PORT_RANGE.base_port
DEFAULT_NB_MSG = 10240

# Sentinel for the control-loop dispatcher: continue the loop (never a parser result).
_CONTINUE = object()

# Control commands that carry no client_id field (readiness pings, the global
# status view, and the JWT-authenticated bootstrap).
_IDENTITY_FREE_COMMANDS = frozenset(
    {
        TaskCmd.HANDSHAKE,
        TaskCmd.HAND_CHECK,
        TaskCmd.STATUS_ALL,
        TaskCmd.GET_BOOTSTRAP_CONTEXT,
    }
)


def _frame_bytes(frame) -> bytes:
    """A received frame's payload (a pyzmq Frame, or bytes from a test double)."""
    return frame.bytes if hasattr(frame, "bytes") else frame


def _frame_session_id(frame) -> str | None:
    """The ZAP user-id of the connection a frame arrived on (the session id
    in session mode); ``None`` without CURVE/ZAP or for plain bytes."""
    get = getattr(frame, "get", None)
    if get is None or isinstance(frame, (bytes, bytearray)):
        return None
    try:
        value = get("User-Id")
    except zmq.ZMQError:
        return None
    return value or None


def _wire_client_id(user_id: str) -> str:
    """A user id as clients put it in the colon-split wire format."""
    return (user_id or "").replace(":", "_")


#: Bound on the per-session and per-connection identity caches.
_IDENTITY_CACHE_MAX = 4096


def _remember(cache, key, value) -> None:
    """Insert into a bounded LRU ``OrderedDict`` (oldest evicted)."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _IDENTITY_CACHE_MAX:
        cache.popitem(last=False)


class MessageQueue:
    # ZMQ identity binding state (session mode; see _bind_session_identity).
    # Class defaults keep instances built without __init__ (test doubles) sound.
    _last_control_session: str | None = None
    _identity_provider_cache: tuple[object | None] | None = None
    _session_entries: "collections.OrderedDict | None" = None
    _sender_entries: "collections.OrderedDict | None" = None
    _current_session_entry: object | None = None
    owner_client_id: str | None = None  # data queues: the user they belong to

    _AUTH_FIELD_INDEX = {
        TaskCmd.CREATE_DATA_QUEUE: 2,
        TaskCmd.ENROLL_USER: 3,
        TaskCmd.RECOMPUTE: 3,
        TaskCmd.UPDATE: 3,
        TaskCmd.CHECK: 3,
        TaskCmd.RELOAD: 3,
    }

    @staticmethod
    def _env_flag(name, default=False):
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _redact_control_message(cls, message):
        """Return a log-safe summary for a control-plane message."""
        if message is None:
            return "<empty>"

        parts = str(message).split(":")
        if not parts or not parts[0]:
            return "cmd=<empty> fields=0"

        cmd = parts[0]
        client_id = parts[1] if len(parts) > 1 else "<missing>"
        auth_index = cls._AUTH_FIELD_INDEX.get(cmd)
        has_auth = bool(
            auth_index is not None and len(parts) > auth_index and parts[auth_index]
        )
        return (
            f"cmd={cmd} client_id={client_id} "
            f"fields={max(len(parts) - 1, 0)} has_auth={str(has_auth).lower()}"
        )

    @staticmethod
    def _resolve_port(queue_type, requested_port):
        if requested_port != BASE_PORT:
            return requested_port

        if queue_type == "control":
            control_port = os.environ.get("UMRS_CONTROL_PORT")
            return int(control_port) if control_port is not None else BASE_PORT

        if queue_type == "data":
            data_port = os.environ.get("UMRS_DATA_PORT")
            if data_port is not None:
                return int(data_port)

            data_base_port = os.environ.get("UMRS_DATA_BASE_PORT")
            if data_base_port is not None:
                return int(data_base_port)

            data_port_range = os.environ.get("UMRS_DATA_PORT_RANGE")
            if data_port_range is not None:
                return parse_data_port_range(data_port_range).base_port

            return _resolve_default_data_port_range().base_port

        return requested_port

    @staticmethod
    def _data_bind_host(tunnel_mode):
        return "127.0.0.1" if tunnel_mode else "0.0.0.0"

    def _configured_data_port_range(self):
        return _resolve_default_data_port_range()

    def _is_data_port_available(self, port):
        bind_host = self._data_bind_host(self.tunnel_mode)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((bind_host, int(port)))
        except OSError:
            return False
        return True

    def _allocate_data_port(self, client_id, port_range):
        reserved_ports = {
            int(entry["port"])
            for other_client_id, entry in self.client_dictionary.items()
            if other_client_id != client_id and "port" in entry
        }

        for port in range(port_range.start, port_range.end + 1):
            if port == BASE_PORT or port in reserved_ports:
                continue
            if self._is_data_port_available(port):
                return port
        return None

    def _ensure_client_data_queue(self, client_id):
        port_range = self._configured_data_port_range()
        entry = self.client_dictionary.setdefault(client_id, {})
        existing_port = entry.get("port")
        existing_queue = entry.get("data_queue")

        if (
            existing_port is not None
            and port_range.contains(existing_port)
            and existing_queue is not None
        ):
            return int(existing_port), port_range

        if (
            existing_port is not None
            and port_range.contains(existing_port)
            and self._is_data_port_available(existing_port)
        ):
            selected_port = int(existing_port)
        else:
            selected_port = self._allocate_data_port(client_id, port_range)
            if selected_port is None:
                return None, port_range

        if existing_queue is not None and existing_port != selected_port:
            try:
                existing_queue.close()
            except Exception as exc:
                logger.warning(
                    "Failed to close stale data queue for %s: %s", client_id, exc
                )

        if existing_queue is None or existing_port != selected_port:
            entry["data_queue"] = MessageQueue(
                role="server",
                queue_type="data",
                server_ip=self.server_ip,
                port=selected_port,
                tunnel_mode=self.tunnel_mode,
                # Share the control queue's ZMQ context so the ZAP
                # authenticator covers data-queue sockets too (#64).
                context=self.context,
            )

        entry["data_queue"].owner_client_id = client_id
        entry["port"] = selected_port
        entry["last_activity"] = time.monotonic()
        return selected_port, port_range

    def _queue_is_reapable(self, client_id, entry, now: float, idle_ttl_s: float) -> bool:
        """Idle-TTL + uncompleted-task guard for one client entry (never reap work)."""
        port = entry.get("port")
        data_queue = entry.get("data_queue")
        if port is None or data_queue is None:
            return False
        last_activity = getattr(
            data_queue, "last_activity_monotonic", entry.get("last_activity", now)
        )
        if now - last_activity < idle_ttl_s:
            return False
        if self.task_manager is None:
            return True
        try:
            uncompleted = self.task_manager.tasks_status.all_uncompleted_task(
                client_id, "all"
            )
        except Exception as exc:  # noqa: BLE001 — doubt means keep the queue
            logger.warning(
                "Data-queue reaper: status lookup failed for %s (%s); keeping",
                client_id,
                exc,
            )
            return False
        return not uncompleted

    def reap_idle_data_queues(self, idle_ttl_s: float | None = None) -> list:
        """Release ports held by idle client data queues (server role).

        A port stays reserved in ``client_dictionary`` until the client
        disconnects — under bursty multi-user load (PROCESS_OBJECTS_BATCH
        lanes) the pool exhausts and ``_allocate_data_port`` starts
        returning ``None``, so ``CREATE_DATA_QUEUE`` would hard-fail with
        NO_DATA_PORT_AVAILABLE.  This reaper closes queues whose last
        data-plane activity is older than the TTL and hands their ports
        back to the pool.

        A queue is NEVER reaped while its client has uncompleted tasks
        (any command): cancelling an in-flight PROCESS_OBJECTS burst
        would drop S3 events mid-upload.

        TTL: ``idle_ttl_s`` argument, else ``UMRS_DATA_QUEUE_IDLE_TTL_S``
        (default 90 s).  Returns the list of reaped client ids.
        """
        if self.role != "server":
            raise PermissionError("Only the server reaps data queues")
        if idle_ttl_s is None:
            idle_ttl_s = float(os.environ.get("UMRS_DATA_QUEUE_IDLE_TTL_S", "90"))
        now = time.monotonic()
        reaped = []
        for client_id, entry in list(self.client_dictionary.items()):
            if not self._queue_is_reapable(client_id, entry, now, idle_ttl_s):
                continue
            with contextlib.suppress(Exception):
                entry["data_queue"].close()
            entry.pop("port", None)
            entry.pop("data_queue", None)
            reaped.append(client_id)
        if reaped:
            logger.info(
                "Reaped %d idle client data queue(s) (ttl=%ss): %s",
                len(reaped),
                idle_ttl_s,
                reaped,
            )
        return reaped

    def __init__(
        self,
        role="server",
        queue_type="control",
        server_ip="127.0.0.1",
        port=BASE_PORT,
        max_nb_messages=DEFAULT_NB_MSG,
        timeout_ms=8000,
        tmp_dir=None,
        task_manager=None,
        status_store=None,
        tunnel_mode=None,
        prefer_env_server_host=True,
        context=None,
        use_status_all_cache=True,
        status_all_cache_ttl=0.2,
    ):
        """
        Initializes the MessageQueue.

        Args:
            queue_type (str): 'control' or 'data'.
            role (str): 'server' or 'client'.
            server_ip (str): IP address of server.
            port (int): Port number. Defaults can be overridden at runtime via
                        UMRS_CONTROL_PORT, UMRS_DATA_PORT, UMRS_DATA_BASE_PORT,
                        or UMRS_DATA_PORT_RANGE.
            tunnel_mode (bool | None): When true, bind/connect to localhost so the
                        transport can run through SSH tunnels. When None, read
                        UMRS_TUNNEL_MODE from the environment.
            context (zmq.Context | None): Shared ZMQ context. When provided, the
                        data queue reuses the control queue's context so that a
                        single ZAP authenticator can cover both control and data
                        sockets (see GitHub issue #64).
            target_nb_messages (int): Max number of messages in queue (for POSIX/SYSV).
            target_message_size (int): Max size per message (for POSIX/SYSV).
            status_store: a dictionnary to store event related to task in server.
                          Shared with UMRSTasks (ZeroMQ reads theis event while UMRSTasks writes it)
        """
        self.role = role.lower()
        self.type = queue_type.lower()
        self.tunnel_mode = (
            self._env_flag("UMRS_TUNNEL_MODE", False)
            if tunnel_mode is None
            else bool(tunnel_mode)
        )
        self.server_ip = self._select_server_ip(server_ip, prefer_env_server_host)
        self.context = context if context is not None else zmq.Context()
        self.client_dictionary = {}
        self.timeout = timeout_ms
        self.max_msg = max_nb_messages if max_nb_messages > 0 else DEFAULT_NB_MSG
        self.tmp_dir = tmp_dir
        self.task_manager = task_manager
        self.status_store = status_store
        self.use_status_all_cache = self._resolve_status_all_cache_flag(use_status_all_cache)
        self.status_all_cache_ttl = status_all_cache_ttl
        # (timestamp, payload) — None until first build
        self._status_all_cache: tuple[float, str] | None = None
        # Note: the upstream per-instance _send_lock was reverted by 2ad7ee04
        # (STATUS threads removed); sends stay synchronous on the event loop.
        self.debug = False  # disable noisy send/recv logs by default
        # Liveness stamp for the server-side idle data-queue reaper
        # (reap_idle_data_queues): set at construction and refreshed on
        # every data-plane receive.
        self.last_activity_monotonic = time.monotonic()
        self._auth_sessions: dict[str, AuthContext] = {}  # client_id -> AuthContext
        self._last_control_auth: dict[str, str] = {}
        # Session mode: the ZAP session id of the last control message, and
        # each session's entry as first seen (ZAP authenticates a connection
        # once, so a session that expires mid-connection keeps its identity).
        self._last_control_session = None
        self._session_entries = collections.OrderedDict()
        self._sender_entries = collections.OrderedDict()
        self._identity_provider_cache = None
        # Started ZAP authenticators for CURVE/session sockets (server role).
        # Retained so close() can stop their background threads. Empty unless
        # UMRS_ZMQ_AUTH_TYPE is curve/session.
        self._zmq_authenticators: list = []

        # Task 11: ZMQ file-transfer /tmp space guard
        self._check_tmp_disk_space()

        # Task 13: queue depth instrumentation
        try:
            from umrs.metrics import METRICS

            self._metrics = METRICS
        except Exception:
            self._metrics = None  # no-op when prometheus unavailable
        self._message_count = 0  # rough in-flight message counter

        # Initialize the selected queue type
        self._init_queue_transport(port)

    def _select_server_ip(self, server_ip, prefer_env_server_host: bool) -> str:
        """Server binding/connect address: env var wins, tunnel forces localhost."""
        env_server_host = (
            os.environ.get("UMRS_SERVER_HOST") if prefer_env_server_host else None
        )
        remote_server_ip = env_server_host if env_server_host else server_ip
        return "127.0.0.1" if self.tunnel_mode else remote_server_ip

    @staticmethod
    def _resolve_status_all_cache_flag(use_status_all_cache: bool) -> bool:
        """UMRS_STATUS_ALL_CACHE=0/false/no disables the cache without code changes."""
        env_cache = os.environ.get("UMRS_STATUS_ALL_CACHE")
        if env_cache is not None:
            use_status_all_cache = env_cache.strip() not in ("0", "false", "no")
        return use_status_all_cache

    def _init_queue_transport(self, port) -> None:
        """Resolve the port and initialize the selected queue type."""
        self.port = self._resolve_port(self.type, port)
        if self.type == "control":
            self._initialize_control_queue()
            return
        if self.type == "data":
            if self.port == BASE_PORT:
                raise ValueError(f"Cannot use port {BASE_PORT} for data channel")
            self._initialize_data_queue()
            return
        raise ValueError(f"Unsupported queue type: {self.type}")

    # ------------------------ Initializers ------------------------

    def _apply_socket_security(self, socket, *, server: bool) -> None:
        """Apply ZMQ CURVE/session security to *socket* before bind/connect.

        No-op when ``UMRS_ZMQ_AUTH_TYPE`` is unset or ``none`` (the default),
        which preserves the existing plaintext transport behavior exactly.

        In ``curve``/``session`` modes the socket is configured for ZMTP CURVE.
        Server sockets additionally start a ZAP authenticator; in ``session``
        mode that authenticator is backed by the process-wide
        :class:`SessionCredentialsProvider` shared with the HTTP session API.
        """
        from umrs.core.zmq_security import ZMQSecurityConfig

        cfg = ZMQSecurityConfig.from_env()
        if cfg.auth_type == "none":
            return

        if server and cfg.is_session_mode:
            from dataclasses import replace as _dc_replace

            from umrs.core.session_curve_provider import get_shared_session_provider

            provider = get_shared_session_provider()
            if provider is None:
                raise RuntimeError(
                    "UMRS_ZMQ_AUTH_TYPE=session but no SessionCredentialsProvider "
                    "is registered; server startup must call "
                    "set_shared_session_provider() before binding ZMQ sockets"
                )
            if cfg.credentials_provider is not provider:
                cfg = _dc_replace(cfg, credentials_provider=provider)

        authenticator = cfg.apply_to_socket(socket, server=server, context=self.context)
        if authenticator is not None:
            self._zmq_authenticators.append(authenticator)

    def _initialize_data_queue(self):
        if self.role == "server":
            # Server uses PULL socket to receive pushed data
            self.socket = self.context.socket(zmq.PULL)
            self._apply_socket_security(self.socket, server=True)
            bind_host = "127.0.0.1" if self.tunnel_mode else "*"
            self.socket.bind(f"tcp://{bind_host}:{self.port}")  # Data channel bind
            self.socket.setsockopt(zmq.RCVHWM, self.max_msg)  # e.g. max queued messages
            self.set_timeouts(self.timeout, self.timeout)
        else:  # client
            # Client uses PUSH socket to send data
            self.socket = self.context.socket(zmq.PUSH)
            self.socket.setsockopt(zmq.LINGER, 0)  # Don't block on close/term
            self._apply_socket_security(self.socket, server=False)
            self.socket.connect(f"tcp://{self.server_ip}:{self.port}")
            self.socket.setsockopt(zmq.SNDHWM, self.max_msg)  # e.g. max queued messages
            self.set_timeouts(self.timeout, self.timeout)

    def _initialize_control_queue(self):
        if self.role == "server":
            self.socket = self.context.socket(zmq.ROUTER)
            self._apply_socket_security(self.socket, server=True)
            bind_host = "127.0.0.1" if self.tunnel_mode else "*"
            try:
                self.socket.bind(f"tcp://{bind_host}:{self.port}")
            except zmq.error.ZMQError as e:
                if "Address already in use" in str(e):
                    raise zmq.error.ZMQError(
                        e.errno,
                        f"Port {self.port} is already in use — "
                        f"is another UMRS server already running? "
                        f"(addr='tcp://{bind_host}:{self.port}')",
                    ) from None
                raise
            self.set_timeouts(-1, self.timeout)
        else:  # client
            self.socket = self.context.socket(zmq.DEALER)
            self.socket.setsockopt(zmq.LINGER, 0)  # Don't block on close/term
            self._apply_socket_security(self.socket, server=False)
            self.socket.connect(f"tcp://{self.server_ip}:{self.port}")
            self.set_timeouts(self.timeout, self.timeout)

    # ------------------------ Core Methods ------------------------

    @staticmethod
    def _encode_control_auth_frame(admin_token=None):
        """Encode optional auth data for a control multipart request."""
        if not admin_token:
            return None
        return json.dumps({"admin_token": admin_token}).encode("utf-8")

    @staticmethod
    def _decode_control_auth_frame(frame):
        """Decode optional auth data from a control multipart request."""
        if not frame:
            return {}
        try:
            payload = json.loads(frame.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("Ignoring malformed control auth frame")
            return {}
        return payload if isinstance(payload, dict) else {}

    def _control_auth_token(self):
        """Return admin token from the multipart auth frame only."""
        return self._last_control_auth.get("admin_token")

    def send_control_request(self, fields, admin_token=None):
        """Send a control request with auth in a separate multipart frame."""
        if self.role != "client" or self.type != "control":
            raise PermissionError("Only control clients can send control requests")

        request = ":".join(str(field) for field in fields)
        frames = [request.encode("utf-8")]
        auth_frame = self._encode_control_auth_frame(admin_token)
        if auth_frame is not None:
            frames.append(auth_frame)

        if self.debug:
            logger.debug(
                "Sending message role=%s type=%s client_id=None %s",
                self.role,
                self.type,
                self._redact_control_message(request),
            )

        try:
            self.socket.send_multipart(frames)
        except zmq.Again:
            raise RuntimeError("Timeout while sending message") from None
        except zmq.ZMQError as e:
            raise RuntimeError(f"ZMQ error while sending: {e}") from e

    def send_string(self, string, client_id=None):
        if self.debug:
            if self.type == "control":
                message = self._redact_control_message(string)
            else:
                message = f"payload_bytes={len(str(string).encode('utf-8'))}"
            logger.debug(
                "Sending message role=%s type=%s client_id=%s %s",
                self.role,
                self.type,
                client_id,
                message,
            )

        try:
            # ---------- CONTROL PLANE ----------
            if self.type == "control":
                if self.role == "server":
                    # ROUTER socket - always send [client_id][empty][payload]
                    # This format works for both REQ and DEALER clients
                    if client_id is None:
                        raise RuntimeError(
                            "Server ROUTER socket requires client_id to send a message"
                        )
                    self.socket.send_multipart(
                        [
                            client_id,
                            b"",  # Empty delimiter for REQ compatibility
                            string.encode("utf-8"),
                        ]
                    )
                else:
                    # DEALER socket
                    self.socket.send_string(string)

            # ---------- DATA PLANE ----------
            elif self.type == "data":
                if self.role == "server":
                    raise RuntimeError("Server cannot send message on data plane")

                # PUSH socket (server or client, no difference)
                self.socket.send_string(string)

            else:
                raise RuntimeError(f"Unknown queue type: {self.type}")

        except zmq.Again:
            raise RuntimeError("Timeout while sending message") from None

        except zmq.ZMQError as e:
            raise RuntimeError(f"ZMQ error while sending: {e}") from e

        except Exception as e:
            raise RuntimeError(
                f"Unexpected error while sending '{string}' from {self.role} to {self.type}: {e}"
            ) from e

    def recv_string(self):
        """
        Receive a string from the socket.

        Returns:
            (client_id, message)
            - client_id is bytes for ROUTER sockets
            - client_id is None otherwise
        """
        try:
            if self.type == "control" and self.role == "server":
                return self._recv_control_request()

            elif self.type == "control" and self.role == "client":
                # DEALER receives: [empty][payload] or [payload]
                # when ROUTER sends with empty delimiter
                frames = self.socket.recv_multipart()
                if len(frames) == 2:
                    # [empty][payload] format
                    _empty, msg = frames
                elif len(frames) == 1:
                    # [payload] format
                    msg = frames[0]
                else:
                    raise ValueError(f"Unexpected frame count: {len(frames)}")
                return None, msg.decode()

            else:
                # PULL socket: single frame (the queue owner's, session mode)
                if self._owner_check_active():
                    msg = self._recv_owned_frame().decode()
                else:
                    msg = self.socket.recv_string()
                # Server-side data queues: data-plane traffic is liveness —
                # the idle reaper (reap_idle_data_queues) never harvests a
                # queue that is actively receiving S3 events.
                self.last_activity_monotonic = time.monotonic()
                return None, msg

        except zmq.Again:
            return None, None

    def _recv_control_request(self):
        """One ROUTER request → ``(sender, message)``.

        ROUTER receives ``[sender][empty][payload]`` from REQ,
        ``[sender][payload]`` from DEALER, or ``[sender][payload][auth]``
        (DEALER multipart auth). The connection's verified session (the ZAP
        user-id) rides on every frame; the payload's own client_id field is
        only a claim (:meth:`_bind_session_identity`).
        """
        self._last_control_auth = {}
        raw = self.socket.recv_multipart(copy=False)
        self._last_control_session = _frame_session_id(raw[-1]) if raw else None
        frames = [_frame_bytes(frame) for frame in raw]
        if len(frames) == 3:
            client_id, second, third = frames
            if second == b"":
                msg = third
            else:
                msg = second
                self._last_control_auth = self._decode_control_auth_frame(third)
        elif len(frames) == 2:
            client_id, msg = frames
        else:
            raise ValueError(f"Unexpected frame count: {len(frames)}")
        return client_id, msg.decode()

    def set_timeouts(self, recv_timeout_ms, send_timeout_ms):
        old_recv = self.socket.RCVTIMEO
        old_send = self.socket.SNDTIMEO
        self.socket.RCVTIMEO = recv_timeout_ms
        self.socket.SNDTIMEO = send_timeout_ms
        return old_recv, old_send

    def send_file(self, send_file_path, chunk_size=4096):
        if self.role != "client" or self.type != "data":
            raise PermissionError(
                "Sending file can only be done by the client using the data queue"
            )
        if not os.path.isfile(send_file_path):
            raise FileNotFoundError(f"File '{send_file_path}' not found")

        with open(send_file_path, "rb") as f:
            # Send file in chunks
            try:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    self.socket.send(chunk)
                # Send end signal (can be empty frame)
                self.socket.send(b"__END_OF_FILE__")
            except zmq.ZMQError as e:
                raise RuntimeError(f"Failed to send file: {e}") from e

    def _data_frame_receiver(self):
        """The data-plane frame receiver: owner-checked in session mode."""
        return self._recv_owned_frame if self._owner_check_active() else self.socket.recv

    def _owner_check_active(self) -> bool:
        """Session mode on a data queue that knows its owner."""
        return self.owner_client_id is not None and self._identity_provider() is not None

    def _recv_owned_frame(self) -> bytes:
        """The next data-plane frame sent by the queue owner's session.

        Every registered session passes the shared ZAP authenticator and data
        ports come from a known range, so another user could push S3 event
        notices or file chunks into this queue (review 2026-10-04). Frames
        from any other session (an admin's excepted) are dropped.
        """
        while True:
            frame = self.socket.recv(copy=False)
            session_id = _frame_session_id(frame)
            provider = self._identity_provider()
            entry = self._session_entry(provider, session_id) if session_id else None
            if entry is not None and (
                _wire_client_id(entry.user_id) == self.owner_client_id
                or entry.grants.is_admin
            ):
                return _frame_bytes(frame)
            logger.warning(
                "data queue of %s: dropped a frame from session %r",
                self.owner_client_id,
                session_id,
            )

    def recv_file(self, out_dir="/tmp"):
        if self.role != "server" or self.type != "data":
            raise PermissionError(
                "Receiving file can only be done by the server using the data queue"
            )

        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)  # Ensure directory exists

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b", dir=out_dir, delete=False
            ) as tmp_file:
                tmp_path = os.path.abspath(tmp_file.name)
                recv_chunk = self._data_frame_receiver()
                while True:
                    chunk = recv_chunk()
                    if chunk == b"__END_OF_FILE__":
                        break
                    tmp_file.write(chunk)
                tmp_file.flush()
            return tmp_path
        except Exception:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # ------------------------ Close ------------------------

    def close(self):
        # Close data sockets for clients (if any)
        for entry in self.client_dictionary.values():
            data_queue = entry.get("data_queue")
            if data_queue:
                try:
                    data_queue.socket.close()
                except Exception as e:
                    logger.warning("Failed to close data socket: %s", e)

        # Close main socket
        try:
            self.socket.close()
        except Exception as e:
            logger.warning("Failed to close main socket: %s", e)

        # Stop any started ZAP authenticators (CURVE/session server sockets)
        for authenticator in self._zmq_authenticators:
            try:
                authenticator.stop()
            except Exception as e:
                logger.warning("Failed to stop ZMQ authenticator: %s", e)
        self._zmq_authenticators = []

    # ------------------------ Server methods ------------------------
    # ------------------------------------------------------------------
    # ZMQ control-loop command parsers
    # Each parser takes (self, splited_msg, zmq_sender_id) and either:
    #   - returns a (cmd, payload) tuple  → caller (server_event_loop) dispatches
    #   - returns None                    → already handled (reply sent), loop continues
    #   - raises _CmdError                → sends CMD_ERROR and continues
    # ------------------------------------------------------------------

    class _CmdError(Exception):
        """Signals a malformed command; server_control_queue_loop sends CMD_ERROR."""

    def _parse_handshake(self, splited_msg, zmq_sender_id):
        # D-04: liveness reply carries the wire protocol version since 1.0
        # ("READY!:v<major.minor>"); pre-1.0 clients comparing the bare
        # prefix keep working (client tolerates both, contract pins both).
        from r4_contracts.protocol import handshake_reply

        with contextlib.suppress(RuntimeError):
            self.send_string(handshake_reply(), zmq_sender_id)

    def _parse_create_data_queue(self, splited_msg, zmq_sender_id):
        if len(splited_msg) < 2:
            raise self._CmdError("CREATE_DATA_QUEUE missing client_id")
        client_id = splited_msg[1]
        token = self._control_auth_token()
        try:
            auth_ctx = create_auth_context_from_request(client_id, token=token)
        except Exception as e:
            logger.warning("Failed to create auth context for %s: %s", client_id, e)
            auth_ctx = AuthContext(
                user_id=client_id, username=client_id, role=UserRole.USER
            )
        self._store_auth(client_id, auth_ctx)
        logger.info(
            "Auth session created: user=%s role=%s", client_id, auth_ctx.role.value
        )
        port, port_range = self._ensure_client_data_queue(client_id)
        if port is None:
            # Contention-point self-healing: a busy reply is only honest
            # after idle queues have been reaped once — batch lanes
            # (PROCESS_OBJECTS_BATCH) drive bursty CREATE_DATA_QUEUE traffic
            # that would otherwise hard-fail past the port-pool size.
            self.reap_idle_data_queues()
            port, port_range = self._ensure_client_data_queue(client_id)
        if port is None:
            with contextlib.suppress(RuntimeError):
                self.send_string(
                    f"NO_DATA_PORT_AVAILABLE:{port_range.label}", zmq_sender_id
                )
            return
        # Advertise a client-reachable host, never the bind address.
        # UMRS_SERVER_HOST is typically "0.0.0.0" inside containers, which is
        # meaningless to a remote client (it would connect to its own
        # localhost and every data message would be silently dropped).
        advertise_host = os.environ.get("UMRS_ADVERTISE_HOST") or self.server_ip
        with contextlib.suppress(RuntimeError):
            self.send_string(f"tcp://{advertise_host}:{port}", zmq_sender_id)
        return

    def _parse_status(self, splited_msg, zmq_sender_id):
        client_id = splited_msg[1] if len(splited_msg) > 1 else None
        status_key = splited_msg[2] if len(splited_msg) > 2 else "all"
        # Final upstream form (92720901): synchronous — the short-lived
        # STATUS_ALL cache carries the concurrency load; per-call threads and
        # the coarse send lock were introduced then reverted upstream.
        payload = self._get_status_payload(client_id, status_key)
        with contextlib.suppress(RuntimeError):
            self.send_string(payload, zmq_sender_id)
        return None

    def _parse_status_all(self, splited_msg, zmq_sender_id):
        payload = self._get_status_all_payload()
        with contextlib.suppress(RuntimeError):
            self.send_string(payload, zmq_sender_id)
        return None

    def _parse_count_cmd(self, splited_msg, zmq_sender_id):
        """Parser for FILE, PROCESS_OBJECTS, ENROLL_USER: field[2] is an integer count."""
        cmd = splited_msg[0]
        if len(splited_msg) < 3:
            raise self._CmdError(f"{cmd} missing fields")
        client_id = splited_msg[1]
        try:
            count = int(splited_msg[2])
        except ValueError:
            raise self._CmdError(f"{cmd} invalid count: {splited_msg[2]}")
        if cmd == TaskCmd.PROCESS_OBJECTS:
            return self._process_object_cmd(client_id, count, zmq_sender_id)
        if cmd == TaskCmd.ENROLL_USER:
            self._refresh_auth_session(client_id, cmd)
        return self._process_file_cmd(
            client_id,
            count,
            zmq_sender_id,
            result_cmd=cmd if cmd != TaskCmd.FILE else None,
        )

    def _parse_upload_workflow(self, splited_msg, zmq_sender_id):
        if len(splited_msg) < 3:
            raise self._CmdError("UPLOAD_WORKFLOW missing fields")
        client_id, workflow_name = splited_msg[1], splited_msg[2]
        return self._process_file_cmd(
            client_id,
            1,
            zmq_sender_id,
            result_cmd=TaskCmd.UPLOAD_WORKFLOW,
            extra_payload=workflow_name,
        )

    def _parse_wait_output(self, splited_msg, zmq_sender_id):
        client_id = splited_msg[1] if len(splited_msg) > 1 else None
        args = splited_msg[2] if len(splited_msg) > 2 else None
        if client_id is None or args is None:
            raise self._CmdError("WAIT_OUTPUT missing client_id or args")
        parts = args.split(";")
        bucket_name = parts[0]
        timeout = int(parts[1]) if len(parts) > 1 else None
        return TaskCmd.WAIT_OUTPUT, (client_id, bucket_name, timeout, zmq_sender_id)

    def _parse_simple_payload(self, splited_msg, zmq_sender_id):
        """Parser for EXPORT_EMBEDDINGS: field[2] is a string payload."""
        cmd = splited_msg[0]
        client_id = splited_msg[1] if len(splited_msg) > 1 else None
        extra = splited_msg[2] if len(splited_msg) > 2 else None
        if client_id is None or extra is None:
            raise self._CmdError(f"{cmd} missing client_id or payload")
        return cmd, (client_id, extra, zmq_sender_id)

    def _parse_maintenance(self, splited_msg, zmq_sender_id):
        """Parser for RECOMPUTE, UPDATE, CHECK, RELOAD: defer ACK until after auth."""
        cmd = splited_msg[0]
        client_id = splited_msg[1] if len(splited_msg) > 1 else None
        self._refresh_auth_session(
            client_id, cmd, token_field=3, splited_msg=splited_msg
        )
        return cmd, (client_id, None, zmq_sender_id)

    def _parse_json_payload(self, splited_msg, zmq_sender_id):
        """Parser for MODEL_*, JOB_*, LIST_WORKFLOWS, RUN_WORKFLOW: rest of message is JSON."""
        cmd = splited_msg[0]
        client_id = splited_msg[1] if len(splited_msg) > 1 else None
        payload_str = ":".join(splited_msg[2:]) if len(splited_msg) > 2 else ""
        # Extract correlation_id from payload if present, log it.
        # This bridges HTTP-style tracing to ZMQ transport (G-001).
        if payload_str:
            try:
                payload_data = json.loads(payload_str)
                if isinstance(payload_data, dict):
                    from .zmq_correlation import extract_correlation_id

                    cid = extract_correlation_id(payload_data)
                    if cid:
                        logger.info("ZMQ %s correlation_id=%s", cmd, cid)
            except (json.JSONDecodeError, ValueError):
                pass  # Not JSON — nothing to extract
        return cmd, (client_id, payload_str, zmq_sender_id)

    def _parse_bootstrap_context(self, splited_msg, zmq_sender_id):
        """Parser for GET_BOOTSTRAP_CONTEXT: requires JWT auth, returns auth context."""
        jwt_token = self._control_auth_jwt()
        if not jwt_token:
            with contextlib.suppress(RuntimeError):
                self.send_string(
                    "AUTH_REQUIRED:JWT required for bootstrap", zmq_sender_id
                )
            return None
        try:
            from umrs.shared.auth import authenticate_request

            auth_ctx = authenticate_request(jwt_token)
        except Exception as exc:
            with contextlib.suppress(RuntimeError):
                self.send_string(f"AUTH_ERROR:{exc}", zmq_sender_id)
            return None
        return TaskCmd.GET_BOOTSTRAP_CONTEXT, (auth_ctx, zmq_sender_id)

    def _refresh_auth_session(self, client_id, cmd, token_field=None, splited_msg=None):
        """Refresh _auth_sessions[client_id] from the admin token embedded in the message."""
        token = None
        if token_field is not None and splited_msg is not None:
            token = splited_msg[token_field] if len(splited_msg) > token_field else None
        else:
            token = self._control_auth_token()
        if client_id and token:
            try:
                auth_ctx = create_auth_context_from_request(client_id, token=token)
                self._store_auth(client_id, auth_ctx)
                logger.info(
                    "Auth session refreshed: user=%s role=%s cmd=%s",
                    client_id,
                    auth_ctx.role.value,
                    cmd,
                )
            except Exception as e:
                logger.warning(
                    "Failed to refresh auth context for %s %s: %s", cmd, client_id, e
                )

    # Maps each TaskCmd string to the name of the parser method that handles it.
    # Parser methods are defined above; strings are used so the dict can live
    # alongside them in the class body without forward-reference issues.
    _CONTROL_PARSERS: "dict[str, str]" = {
        # HANDSHAKE and HAND_CHECK are equivalent readiness pings (both reply
        # "READY!"). HANDSHAKE comes from the in-repo client; HAND_CHECK from the
        # r4 SDK transport. Both aliases are kept so either client can probe the
        # server regardless of which package version is deployed.
        TaskCmd.HANDSHAKE: "_parse_handshake",
        TaskCmd.HAND_CHECK: "_parse_handshake",
        TaskCmd.CREATE_DATA_QUEUE: "_parse_create_data_queue",
        TaskCmd.STATUS: "_parse_status",
        TaskCmd.STATUS_ALL: "_parse_status_all",
        TaskCmd.PROCESS_OBJECTS: "_parse_count_cmd",
        TaskCmd.FILE: "_parse_count_cmd",
        TaskCmd.ENROLL_USER: "_parse_count_cmd",
        TaskCmd.UPLOAD_WORKFLOW: "_parse_upload_workflow",
        TaskCmd.WAIT_OUTPUT: "_parse_wait_output",
        TaskCmd.EXPORT_EMBEDDINGS: "_parse_simple_payload",
        TaskCmd.RECOMPUTE: "_parse_maintenance",
        TaskCmd.UPDATE: "_parse_maintenance",
        TaskCmd.CHECK: "_parse_maintenance",
        TaskCmd.RELOAD: "_parse_maintenance",
        TaskCmd.MODEL_REGISTER: "_parse_json_payload",
        TaskCmd.MODEL_LIST: "_parse_json_payload",
        TaskCmd.MODEL_DOWNLOAD: "_parse_json_payload",
        TaskCmd.LIST_WORKFLOWS: "_parse_json_payload",
        TaskCmd.RUN_WORKFLOW: "_parse_json_payload",
        TaskCmd.WORKFLOW_STATUS: "_parse_json_payload",
        TaskCmd.WORKFLOW_RESULTS: "_parse_json_payload",
        TaskCmd.WORKFLOW_CANCEL: "_parse_json_payload",
        TaskCmd.WORKFLOW_ERRORS: "_parse_json_payload",
        TaskCmd.WORKFLOW_EXECUTIONS: "_parse_json_payload",
        TaskCmd.JOB_SUBMIT: "_parse_json_payload",
        TaskCmd.JOB_CANCEL: "_parse_json_payload",
        TaskCmd.JOB_LIST: "_parse_json_payload",
        TaskCmd.PURGE_BUCKETS: "_parse_json_payload",
        TaskCmd.GET_BOOTSTRAP_CONTEXT: "_parse_bootstrap_context",
    }

    def server_control_queue_loop(self):
        if self.role != "server":
            raise PermissionError("Only server can call 'server_control_queue_loop'.")

        # Lookahead deque: non-PROCESS_OBJECTS messages captured during batch drain
        if not hasattr(self, "_lookahead"):
            self._lookahead = collections.deque()

        while True:
            zmq_sender_id, msg, session_id = self._next_control_message()
            logger.info(
                "Received control request: %s", self._redact_control_message(msg)
            )
            splited_msg = msg.split(":")
            cmd = splited_msg[0]
            splited_msg = self._bind_session_identity(
                cmd, splited_msg, zmq_sender_id, session_id
            )
            if splited_msg is None:
                continue

            result = self._dispatch_control_parser(cmd, splited_msg, zmq_sender_id)
            if result is _CONTINUE:
                continue
            if result is None:
                # Command handled inline, loop continues
                continue

            # Got a PROCESS_OBJECTS — always wrap in BATCH (N=1 is handled uniformly)
            if result[0] == TaskCmd.PROCESS_OBJECTS:
                return self._assemble_process_batch(result)

            return result

    def _next_control_message(self):
        """Lookahead deque first, then blocking ZMQ receive (ejeannot batch lane).

        Returns ``(sender, message, session_id)``; ``session_id`` is the
        connection's ZAP user-id (``None`` outside session mode).
        """
        if self._lookahead:
            zmq_sender_id, msg, session_id, auth = self._lookahead.popleft()
            # Its own auth frame, never the one of the message that drained it.
            self._last_control_auth = auth
            return zmq_sender_id, msg, session_id
        zmq_sender_id, msg = self.recv_string()
        return zmq_sender_id, msg, self._last_control_session

    def _identity_provider(self):
        """The session provider when ZMQ identity binding applies, else None.

        Binding applies in ``UMRS_ZMQ_AUTH_TYPE=session`` unless
        ``UMRS_ZMQ_SESSION_IDENTITY=off`` (an escape hatch, logged loudly).
        """
        if self._identity_provider_cache is None:
            provider = None
            from umrs.core.zmq_security import ZMQSecurityConfig

            if self.role == "server" and ZMQSecurityConfig.from_env().is_session_mode:
                mode = (
                    os.environ.get("UMRS_ZMQ_SESSION_IDENTITY", "bind").strip().lower()
                )
                if mode == "off":
                    logger.warning(
                        "UMRS_ZMQ_SESSION_IDENTITY=off: ZMQ commands act as whatever "
                        "client_id they claim (any session can act as any user)"
                    )
                else:
                    from umrs.core.session_curve_provider import (
                        get_shared_session_provider,
                    )

                    provider = get_shared_session_provider()
            self._identity_provider_cache = (provider,)
        return self._identity_provider_cache[0]

    def _session_entry(self, provider, session_id):
        """The session's live entry; a session that merely expired while its
        connection stayed open keeps the identity it was first seen with.

        A session gone before its expiry (revoked, account disabled, evicted
        by the per-user cap) is refused: ZAP authenticated the connection
        once, so this re-check is what ends an open revoked connection.
        """
        if self._session_entries is None:
            self._session_entries = collections.OrderedDict()
        live = provider.lookup_session(session_id)
        if live is not None:
            _remember(self._session_entries, session_id, live)
            return live
        cached = self._session_entries.get(session_id)
        if cached is not None and cached.expires_at > time.time():
            self._session_entries.pop(session_id, None)
            return None
        return cached

    def verified_session(self, zmq_sender_id):
        """The session entry of the connection *zmq_sender_id* (a ROUTER
        routing id) as bound by its last command, else None."""
        return (self._sender_entries or {}).get(zmq_sender_id)

    def _auth_refusal(self, cmd) -> str:
        """The refusal reply in the shape the command's clients parse."""
        reason = "no authenticated ZMQ session"
        if self._CONTROL_PARSERS.get(cmd) == "_parse_json_payload":
            return json.dumps({"status": "error", "message": reason})
        return f"AUTH_ERROR:{reason}"

    def _carries_client_id(self, cmd, splited_msg) -> bool:
        """A known command with a client_id field (others pass untouched: the
        dispatcher answers UNKNOWN_CMD / CMD_ERROR for them)."""
        return (
            cmd not in _IDENTITY_FREE_COMMANDS
            and cmd in self._CONTROL_PARSERS
            and len(splited_msg) >= 2
        )

    def _bind_session_identity(self, cmd, splited_msg, zmq_sender_id, session_id):
        """Session mode: the client_id field becomes the connection's user.

        The field is a claim; the CURVE session the connection authenticated
        with is the identity. A claim that differs is rebound (a client that
        sent ``default`` or a project id now acts as its own user); only an
        admin session may act as another client_id. A connection without a
        known session is refused. Returns the (possibly rebound) message
        fields, or ``None`` when refused (reply already sent).
        """
        self._current_session_entry = None
        provider = self._identity_provider()
        if provider is None or not self._carries_client_id(cmd, splited_msg):
            return splited_msg
        entry = self._session_entry(provider, session_id) if session_id else None
        if entry is None:
            logger.warning(
                "ZMQ %s refused: no authenticated session (%r)", cmd, session_id
            )
            with contextlib.suppress(RuntimeError):
                self.send_string(self._auth_refusal(cmd), zmq_sender_id)
            return None
        self._current_session_entry = entry
        if self._sender_entries is None:
            self._sender_entries = collections.OrderedDict()
        _remember(self._sender_entries, zmq_sender_id, entry)
        user = _wire_client_id(entry.user_id)
        claimed = splited_msg[1]
        if claimed == user:
            return splited_msg
        if entry.grants.is_admin:
            logger.info(
                "ZMQ %s: admin session %s acts as client_id %r", cmd, user, claimed
            )
            return splited_msg
        logger.warning(
            "ZMQ %s: client_id %r rebound to the session's user %r", cmd, claimed, user
        )
        return [splited_msg[0], user, *splited_msg[2:]]

    def _dispatch_control_parser(self, cmd, splited_msg, zmq_sender_id):
        """Command → parser result; _CONTINUE on unknown/malformed/parser errors."""
        parser_name = self._CONTROL_PARSERS.get(cmd)
        if parser_name is None:
            logger.warning("Unknown ZMQ command: %s", cmd)
            with contextlib.suppress(RuntimeError):
                self.send_string("UNKNOWN_CMD", zmq_sender_id)
            return _CONTINUE
        try:
            return getattr(self, parser_name)(splited_msg, zmq_sender_id)
        except self._CmdError as e:
            logger.error("Malformed command %s: %s", cmd, e)
            with contextlib.suppress(RuntimeError):
                self.send_string("CMD_ERROR", zmq_sender_id)
        except Exception as e:
            logger.error("Unexpected error in parser for %s: %s", cmd, e)
            with contextlib.suppress(RuntimeError):
                self.send_string("SERVER_ERROR", zmq_sender_id)
        return _CONTINUE

    def _assemble_process_batch(self, result):
        """PROCESS_OBJECTS result + inline drain → a BATCH tuple for the caller."""
        batch = [result[1]]
        batch.extend(self._drain_process_objects())
        logger.info(
            "[PROCESS] Batch assembled: %d client request(s) — %s",
            len(batch),
            [(cid, nb) for cid, nb in batch],
        )
        return TaskCmd.PROCESS_OBJECTS_BATCH, batch

    def _drain_process_objects(self):
        """Non-blocking drain of the ZMQ control socket.

        Collects additional PROCESS_OBJECTS payloads that are already queued.
        Non-PROCESS_OBJECTS messages are pushed to the lookahead deque for the
        next iteration of server_control_queue_loop.

        Returns a list of additional PROCESS_OBJECTS payloads (may be empty).
        """
        extra = []
        while True:
            try:
                raw = self.socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
            except zmq.Again:
                break

            drained = self._read_drained_message(raw)
            if drained is None:
                continue
            zmq_sender_id, msg, session_id, auth = drained

            splited_msg = msg.split(":")
            cmd = splited_msg[0]

            if cmd != TaskCmd.PROCESS_OBJECTS:
                self._lookahead.append((zmq_sender_id, msg, session_id, auth))
                continue

            splited_msg = self._bind_session_identity(
                cmd, splited_msg, zmq_sender_id, session_id
            )
            if splited_msg is None:
                continue
            self._ack_process_object_drain(splited_msg, cmd, zmq_sender_id, extra)

        return extra

    def _read_drained_message(self, raw):
        """Drained ROUTER frames → ``(sender, message, session_id, auth)``;
        None for unexpected frame counts. The message keeps its own session
        and auth frame for when the lookahead deque hands it on."""
        session_id = _frame_session_id(raw[-1]) if raw else None
        frames = [_frame_bytes(frame) for frame in raw]
        parsed = self._parse_drain_frames(frames)
        if parsed is None:
            return None
        has_auth = len(frames) == 3 and frames[1] != b""
        auth = self._decode_control_auth_frame(frames[2]) if has_auth else {}
        return (*parsed, session_id, auth)

    @staticmethod
    def _parse_drain_frames(frames):
        """ROUTER frame → (sender_id, msg); None for unexpected frame counts."""
        if len(frames) == 3 and frames[1] == b"":
            zmq_sender_id, msg_bytes = frames[0], frames[2]
        elif len(frames) == 3:
            zmq_sender_id, msg_bytes = frames[0], frames[1]
        elif len(frames) == 2:
            zmq_sender_id, msg_bytes = frames[0], frames[1]
        else:
            logger.warning(
                "_drain_process_objects: unexpected frame count %d", len(frames)
            )
            return None
        return zmq_sender_id, msg_bytes.decode()

    def _ack_process_object_drain(self, splited_msg, cmd, zmq_sender_id, extra) -> None:
        """Parse and ACK one drained PROCESS_OBJECTS inline (errors → SERVER_ERROR)."""
        try:
            result = self._parse_count_cmd(splited_msg, zmq_sender_id)
        except Exception as e:
            logger.error("_drain_process_objects: parse error for %s: %s", cmd, e)
            with contextlib.suppress(RuntimeError):
                self.send_string("SERVER_ERROR", zmq_sender_id)
            return
        if result is not None:
            extra.append(result[1])

    def wait_for_s3_events(self, nb_objects, debug=False, timeout_s=60):
        """Block until new messages arrive, then read all available messages.

        Args:
            nb_objects: Number of S3 event messages to collect.
            debug: Enable debug printing.
            timeout_s: Maximum wall-clock seconds to wait before returning
                       with whatever messages have been collected so far.
                       Default 60s.  Set to 0 or None for no limit.

        Returns:
            List of received message strings (may be shorter than nb_objects
            if the timeout fires).
        """
        if self.role != "server":
            raise PermissionError("Only server can call 'wait_for_s3_events'.")

        messages = []
        logger.debug(
            "Listening for %s new messages (timeout=%s)",
            nb_objects,
            timeout_s,
        )
        nb_received = 0
        start = time.time()
        while nb_received < nb_objects:
            if timeout_s and (time.time() - start) > timeout_s:
                logger.warning(
                    "wait_for_s3_events timed out after %s seconds (%s/%s received)",
                    timeout_s,
                    nb_received,
                    nb_objects,
                )
                break
            _, message = self.recv_string()
            if message:
                logger.debug(
                    "Received S3 event payload_bytes=%s", len(message.encode("utf-8"))
                )
                messages.append(message)
                nb_received += 1

        return messages

    def _check_tmp_disk_space(self) -> None:
        """Guard against ZMQ file-transfer space exhaustion.

        When ``self.tmp_dir`` is set, checks that at least 500 MiB of
        free space is available.  Raises on critical shortage so that
        ZMQ socket creation fails fast instead of silently corrupting
        messages or filling the disk.
        """
        if not self.tmp_dir:
            return
        try:
            stat = shutil.disk_usage(self.tmp_dir)
            free_mb = stat.free / (1024 * 1024)
            threshold_mb = 500  # minimum free space required
            if free_mb < threshold_mb:
                raise OSError(
                    f"ZMQ tmp_dir {self.tmp_dir} has only {free_mb:.0f} MiB free "
                    f"(need >= {threshold_mb} MiB) — refusing to create ZMQ files."
                )
            logger.debug(
                "ZMQ disk guard OK: %.0f MiB free on %s",
                free_mb,
                self.tmp_dir,
            )
        except OSError as exc:
            logger.warning(
                "ZMQ disk guard check failed (%s) — continuing: %s",
                self.tmp_dir,
                exc,
            )

    def _process_file_cmd(
        self, client_id, nb_files, zmq_sender_id, result_cmd=None, extra_payload=None
    ):
        if result_cmd is None:
            result_cmd = TaskCmd.FILE
        # Verify client has a data socket ready
        if (
            client_id in self.client_dictionary
            and "data_queue" in self.client_dictionary[client_id]
        ):
            data_queue = self.client_dictionary[client_id]["data_queue"]
            port = self.client_dictionary[client_id]["port"]
            logger.info("Listening for data on port %s", port)

            path_list = []
            if not self.tmp_dir or not os.path.exists(self.tmp_dir):
                logger.warning(
                    "No temporary receiving directory configured or accessible; using %s",
                    os.devnull,
                )
                recv_dir = os.devnull
            else:
                recv_dir = self.tmp_dir

            # For each expected file:
            for _ in range(nb_files):
                # Receive file data into it
                file_path = data_queue.recv_file(out_dir=recv_dir)
                # Keep track of temp files
                path_list.append(file_path)

            # Send acknowledgment to client after receiving all files
            try:
                self.send_string("ACKNOWLEDGE", zmq_sender_id)
            except RuntimeError as e:
                logger.error("Error acknowledging file client: %s", e)
            # Return client_id along with received file paths so server can enforce identity
            if extra_payload is not None:
                return result_cmd, (client_id, path_list, extra_payload)
            return result_cmd, (client_id, path_list)
        else:  # cannot find the client in the client_dictionnary
            try:
                self.send_string("NO_DATA_QUEUE", zmq_sender_id)
            except RuntimeError as e:
                logger.error("Error sending NO_DATA_QUEUE to client: %s", e)

    def _process_object_cmd(self, client_id, nb_objects, zmq_sender_id):
        if (
            client_id in self.client_dictionary
            and "data_queue" in self.client_dictionary[client_id]
        ):
            # ACK immediately so the client doesn't timeout while we collect S3 events
            try:
                self.send_string("ACKNOWLEDGE", zmq_sender_id)
            except RuntimeError as e:
                logger.error("Error acknowledging object client: %s", e)
            # Return immediately with the count of expected objects.
            # The ProcessObjectsHandler collects S3 events asynchronously
            # so the control queue loop is not blocked.
            return TaskCmd.PROCESS_OBJECTS, (client_id, nb_objects)
        else:
            logger.warning("No data queue for client=%s", client_id)
            try:
                self.send_string("NO_DATA_QUEUE", zmq_sender_id)
            except RuntimeError as e:
                logger.error("Error sending NO_DATA_QUEUE to client: %s", e)
        return None

    # ------------------------ client methods ------------------------

    def reset_control_queue(self):
        if self.role != "client" or self.type != "control":
            raise PermissionError(
                "reset_control_plane can only be called by client control queue"
            )
        # Close existing socket if it exists
        if hasattr(self, "socket"):
            try:
                self.socket.close()
            except Exception:
                # Best-effort cleanup: ignore errors while closing old socket
                pass

        # Create a new DEALER socket and connect
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't block on close/term
        self._apply_socket_security(self.socket, server=False)
        self.socket.connect(f"tcp://{self.server_ip}:{self.port}")

    def _is_ready_reply(self, answer) -> bool:
        """Tolerate both the legacy ``READY!`` and the versioned ``READY!:v1.x``
        liveness replies (D-04).  ``None`` (recv timeout swallowed by
        recv_string) is simply not ready, never an error.  Falls back to the
        bare comparison when the contracts package is absent."""
        if answer is None:
            return False
        try:
            from r4_contracts.protocol import parse_handshake_reply
        except ImportError:
            return answer == "READY!"
        ready, _version = parse_handshake_reply(answer)
        return ready

    def is_server_ready(self, timeout_ms=5000):
        if self.role != "client":
            raise PermissionError("Only client can call 'get_data_queue'")

        if self.type != "control":
            raise PermissionError("Only control message queue can check server")

        logger.debug("Checking for server availability...")
        timeout_ms = max(timeout_ms, 1500)

        self.set_timeouts(timeout_ms, timeout_ms)
        try:
            try:
                self.send_string(TaskCmd.HANDSHAKE)
                _, answer = self.recv_string()
                if self._is_ready_reply(answer):
                    return True
            except zmq.Again:
                # Retry once on timeout
                try:
                    # but first need to reset the socket to avoid error.
                    self.reset_control_queue()
                    self.set_timeouts(timeout_ms, timeout_ms)
                    self.send_string("READY?")
                    _, answer = self.recv_string()
                    if self._is_ready_reply(answer):
                        return True
                except zmq.Again:
                    return False
            return False
        finally:
            self.set_timeouts(
                self.timeout, self.timeout
            )  # always reset timeout before exiting

    def _await_data_queue_reply(self, client_id, admin_token, busy_deadline):
        """CREATE_DATA_QUEUE with capped exponential backoff on pool exhaustion.

        Returns ``(reply, attempts)``. NO_DATA_PORT_AVAILABLE is transient:
        bursty batch lanes free their ports as they finish — retry until the
        deadline with capped backoff + jitter.
        """
        attempt = 0
        while True:
            attempt += 1
            self.send_control_request(
                [TaskCmd.CREATE_DATA_QUEUE, client_id],
                admin_token=admin_token,
            )
            try:
                _, reply = self.recv_string()
            except zmq.Again:
                raise RuntimeError(
                    "Timeout waiting for server response in get_data_queue"
                ) from None
            except (AttributeError, TypeError, ValueError):
                reply = self.socket.recv_string()

            if (
                reply is None
                or not reply.startswith("NO_DATA_PORT_AVAILABLE")
                or time.monotonic() >= busy_deadline
            ):
                return reply, attempt

            sleep_s = min(15.0, 0.5 * (1.7 ** (attempt - 1)))
            sleep_s += random.uniform(0.0, 0.5)
            logger.info(
                "Data port pool exhausted for %s; retrying in %.1fs (attempt %d)",
                client_id,
                sleep_s,
                attempt,
            )
            time.sleep(sleep_s)

    def get_data_queue(self, client_id, tunnel_mode=None, admin_token=None):
        """Request a client data queue.

        Args:
            client_id (str): Logical client identity used by the server.
            tunnel_mode (bool | None): Override the control queue tunnel_mode for
                the returned data queue. In tunnel mode, localhost is used so the
                data channel can traverse an SSH tunnel.
            admin_token (str | None): Optional admin token. When provided the
                server grants ``UserRole.ADMIN`` for this session.
        """
        if self.role != "client":
            raise PermissionError("Only client can call 'get_data_queue'")

        effective_tunnel_mode = (
            self.tunnel_mode if tunnel_mode is None else bool(tunnel_mode)
        )

        # NO_DATA_PORT_AVAILABLE is a TRANSIENT condition, not a hard error:
        # the server reaps idle data queues at the contention point and
        # bursty batch lanes free their ports as they finish.  Retry with
        # capped exponential backoff + jitter until
        # UMRS_DATA_QUEUE_BUSY_TIMEOUT_S (default 600 s) elapses; past the
        # deadline the same RuntimeError as before is raised.
        busy_deadline = time.monotonic() + float(
            os.environ.get("UMRS_DATA_QUEUE_BUSY_TIMEOUT_S", "600")
        )
        reply, busy_attempt = self._await_data_queue_reply(
            client_id, admin_token, busy_deadline
        )

        # ``recv_string`` swallows ``zmq.Again`` and returns ``(None, None)`` on a
        # receive timeout, so the ``except zmq.Again`` above never fires for that
        # case. Guard against a ``None`` reply here instead of letting the
        # ``reply.startswith(...)`` call fail with the opaque
        # ``'NoneType' object has no attribute 'startswith'`` error.
        if reply is None:
            self._raise_data_queue_timeout(client_id)

        if reply.startswith("tcp://"):
            return self._build_data_queue_from_advert(reply, effective_tunnel_mode)
        self._raise_data_queue_verdict(reply, busy_attempt)
    def _build_data_queue_from_advert(self, reply: str, effective_tunnel_mode):
        """Parse ``tcp://ip:port`` and build the data ``MessageQueue``.

        A server that binds wildcard may advertise it verbatim; connecting a
        PUSH socket there would target the client's own localhost and silently
        drop S3 events — fall back to the control channel's host instead.
        """
        endpoint = reply[len("tcp://") :]
        server_ip, port_str = endpoint.split(":")
        port = int(port_str)
        if server_ip in ("0.0.0.0", "::", ""):
            logger.warning(
                "Server advertised non-routable data endpoint %s; "
                "falling back to control host %s",
                reply,
                self.server_ip,
            )
            server_ip = self.server_ip
        return MessageQueue(
            role="client",
            queue_type="data",
            server_ip=server_ip,
            port=port,
            tunnel_mode=effective_tunnel_mode,
            # Share the control queue's ZMQ context so the ZAP
            # authenticator covers data-queue sockets too (#64).
            context=self.context,
        )

    def _raise_data_queue_timeout(self, client_id) -> None:
        raise RuntimeError(
            f"Timeout waiting for server to allocate a data queue for client '{client_id}'. "
            "The control channel reached the server (CURVE session is valid) but "
            "the server did not return a data-queue endpoint in time. This "
            "usually means the server failed while allocating the data queue "
            "(check the server logs for a CREATE_DATA_QUEUE error) or the "
            "configured data port range (UMRS_DATA_PORT_RANGE / "
            "UMRS_DATA_BASE_PORT) is exhausted or unreachable."
        )

    @staticmethod
    def _raise_data_queue_verdict(reply: str, busy_attempt: int) -> None:
        """Non-tcp, non-timeout reply: range exhaustion or an unexpected reply."""
        if reply.startswith("NO_DATA_PORT_AVAILABLE"):
            _, _, port_range = reply.partition(":")
            suffix = f" in configured range {port_range}" if port_range else ""
            raise RuntimeError(
                f"No data ports available{suffix} "
                f"(still exhausted after {busy_attempt} attempt(s))"
            )
        raise RuntimeError(f"Unexpected reply from server: {reply}")

    def notify_server(
        self,
        client_id,
        notification_type,
        nb_objects,
        verbose=False,
        admin_token=None,
        blocking=False,
    ):
        """Notify the server that objects have been uploaded to S3.

        The data queue (PUSH/PULL) carries the actual S3 object URIs; this
        control-channel notification only signals how many objects to expect.
        CURVE authentication is applied to the data queue sockets when
        ``UMRS_ZMQ_AUTH_TYPE`` is set to ``curve`` or ``session`` (see #64).
        """
        if self.role != "client":
            raise PermissionError("Only client can call 'notify_server'")

        self.send_control_request(
            [notification_type, client_id, nb_objects],
            admin_token=admin_token,
        )
        reply = self._await_notify_reply(blocking)
        return self._notify_reply_verdict(
            reply, notification_type, client_id, verbose
        )

    def _await_notify_reply(self, blocking: bool):
        """Receive the server's reply, honoring the blocking-timeout swap."""
        try:
            if blocking:
                prev_recv_timeout, prev_send_timeout = self.set_timeouts(-1, -1)
            try:
                _, reply = self.recv_string()
            finally:
                if blocking:
                    self.set_timeouts(prev_recv_timeout, prev_send_timeout)
        except zmq.Again:
            raise TimeoutError(
                "Timeout waiting for server response in notify_server"
            ) from None
        return reply

    def _notify_reply_verdict(self, reply, notification_type, client_id, verbose):
        """Map the server's reply to the notify outcome (ACK / named errors)."""
        if reply is None:
            raise TimeoutError(
                f"Timeout waiting for server response to {notification_type} for user {client_id}. "
                "The server may be blocked waiting for data-queue messages that did not arrive. "
                "Check that the SSH tunnel forwards the configured data port range "
                "(see UMRS_DATA_PORT_RANGE / UMRS_DATA_BASE_PORT)."
            )
        elif reply == "ACKNOWLEDGE":
            if verbose:
                logger.info("Server acknowledged reception of the objects")
            return True
        elif reply == "NO_DATA_QUEUE":
            raise RuntimeError(
                f"Error: the server cannot find the data queue for user {client_id}."
            )
        elif reply.startswith("UNAUTHORIZED:"):
            logger.warning(
                "Server denied %s for user %s: %s", notification_type, client_id, reply
            )
            return False
        else:
            raise RuntimeError(
                f"Unexpected error on server side for user {client_id}: {reply}."
            )

    def request_status(self, client_id, status_key="all"):
        if self.role != "client":
            raise PermissionError("Only client can call 'request_status'")

        safe_client_id = (client_id or "").replace(":", "_")
        safe_status_key = (status_key or "all").replace(":", "_")
        request_str = f"{TaskCmd.STATUS}:{safe_client_id}:{safe_status_key}"
        self.send_string(request_str)

        try:
            _, reply = self.recv_string()
        except zmq.Again:
            raise TimeoutError(
                "Timeout waiting for server response in request_status"
            ) from None

        return reply

    def request_export_embeddings(
        self, client_id, path_to_bucket, export_format="csv", export_profile="minimal"
    ):
        """Send EXPORT_EMBEDDINGS request and receive the serialised result."""
        if self.role != "client":
            raise PermissionError("Only client can call 'request_export_embeddings'")

        safe_client_id = (client_id or "").replace(":", "_")
        safe_args = f"{path_to_bucket};{export_format};{export_profile}".replace(
            ":", "_"
        )
        request_str = f"{TaskCmd.EXPORT_EMBEDDINGS}:{safe_client_id}:{safe_args}"

        # Allow a generous receive timeout for large tables
        old_rcv = self.socket.getsockopt(zmq.RCVTIMEO)
        self.socket.setsockopt(zmq.RCVTIMEO, 120_000)

        self.send_string(request_str)
        try:
            _, reply = self.recv_string()
        except zmq.Again:
            raise TimeoutError(
                "Timeout waiting for server response in request_export_embeddings"
            ) from None
        finally:
            self.socket.setsockopt(zmq.RCVTIMEO, old_rcv)

        return reply

    def wait_for_new_data(self, client_id, full_bucket_name, timeout=60):
        if self.role != "client":
            raise PermissionError("Only client can call 'request_status'")

        timeout_ms = 10000 * (timeout * 2)

        self.set_timeouts(timeout_ms, self.timeout)

        safe_client_id = (client_id or "").replace(":", "_")
        safe_args = f"{full_bucket_name};{timeout}".replace(":", "_")
        request_str = f"{TaskCmd.WAIT_OUTPUT}:{safe_client_id}:{safe_args}"
        self.send_string(request_str)

        try:
            _, reply = self.recv_string()
        except zmq.Again:
            raise TimeoutError(
                "Timeout waiting for server response in request_status"
            ) from None

        finally:
            self.set_timeouts(self.timeout, self.timeout)

        return reply

    def get_status_store(self):
        return self.status_store

    def set_debug(self, enabled: bool):
        self.debug = bool(enabled)

    def _auth_key(self, client_id: str) -> str:
        """Where a token-derived AuthContext lives: per CURVE session while a
        bound command is being handled (session mode), else per client_id.

        Per client_id in session mode let an admin acting as carol leave an
        ADMIN context that carol's own session then read (review 2026-10-04).
        """
        entry = self._current_session_entry
        if entry is not None:
            return f"session:{entry.session_id}"
        return client_id

    def _store_auth(self, client_id: str, auth_ctx: AuthContext) -> None:
        sessions = self._auth_sessions
        sessions[self._auth_key(client_id)] = auth_ctx
        while len(sessions) > _IDENTITY_CACHE_MAX:
            del sessions[next(iter(sessions))]

    def get_auth_context(self, client_id: str) -> AuthContext:
        """Return the AuthContext for *client_id*.

        Session mode: the role of the CURVE session that sent the command
        being handled (its JWT grants, or an admin token it presented itself),
        never a context another session left under the same client_id.
        If the client has not yet authenticated via CREATE_DATA_QUEUE the
        default role is ``UserRole.USER`` (backward compatible).
        """
        ctx = self._auth_sessions.get(self._auth_key(client_id))
        entry = self._current_session_entry
        if entry is not None and entry.grants.is_admin and not (ctx and ctx.is_admin):
            return AuthContext(user_id=client_id, username=client_id, role=UserRole.ADMIN)
        if ctx is not None:
            return ctx
        # Create a default USER context for clients that connected before
        # auth enforcement was deployed (backward compatibility).
        return AuthContext(
            user_id=client_id or "",
            username=client_id or "",
            role=UserRole.USER,
        )

    def _collect_status_extras(self, client_id, status_key):
        """(uncompleted_key_list, pool_metrics) — both optional, failure-tolerant."""
        uncompleted_key_list = None
        pool_metrics = None
        if self.task_manager is not None:
            try:
                uncompleted_key_list = (
                    self.task_manager.tasks_status.all_uncompleted_task(
                        client_id, status_key
                    )
                )
            except Exception as e:
                logger.warning(
                    "Could not fetch uncompleted tasks for STATUS (pool exhausted?): %s",
                    e,
                )
            with contextlib.suppress(Exception):
                pool_metrics = self.task_manager.active_threads.pool.metrics().to_dict()
        return uncompleted_key_list, pool_metrics

    def _get_status_payload(self, client_id, status_key):
        store = self.status_store or {}
        key = status_key or TaskCmd.ALL

        if not client_id:
            payload = {"status": "error", "message": "Missing client identifier"}
            return json.dumps(payload, ensure_ascii=False)

        uncompleted_key_list, pool_metrics = self._collect_status_extras(
            client_id, status_key
        )

        client_store = store.get(client_id, {})
        if key in (TaskCmd.ALL, "*"):
            if not client_store:
                payload = {
                    "status": "error",
                    "message": "No status entries",
                    "client": client_id,
                    "uncompleted_key_list": uncompleted_key_list,
                }
            else:
                payload = {
                    "status": "ok",
                    "client": client_id,
                    "summaries": client_store,
                    "uncompleted_key_list": uncompleted_key_list,
                }
            if pool_metrics is not None:
                payload["pool"] = pool_metrics
            return json.dumps(payload, ensure_ascii=False)

        entry = client_store.get(key)
        if not entry:
            payload = {
                "status": "error",
                "message": f"No status for '{key}'",
                "client": client_id,
                "uncompleted_key_list": uncompleted_key_list,
            }
        else:
            payload = {
                "status": "ok",
                "client": client_id,
                "summary": entry,
                "uncompleted_key_list": uncompleted_key_list,
            }
        if pool_metrics is not None:
            payload["pool"] = pool_metrics
        return json.dumps(payload, ensure_ascii=False)

    def _build_status_all_payload(self) -> str:
        """Build and return the raw STATUS_ALL JSON string (no caching)."""
        try:
            store = self.status_store or {}
            generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            db_by_client: dict = {}
            if self.task_manager is not None:
                with contextlib.suppress(Exception):
                    db_by_client = (
                        self.task_manager.tasks_status.all_tasks_summary_by_client()
                    )

            # diskcache.Cache has no .keys() method — use list(store) (__iter__)
            # which works for both plain dict and diskcache.Cache.
            store_keys = list(store)
            all_ids = set(store_keys) | set(db_by_client.keys())
            clients = {
                cid: {
                    "summaries": store.get(cid, {}),
                    "uncompleted_key_list": db_by_client.get(cid, []),
                }
                for cid in all_ids
            }
            return json.dumps(
                {"status": "ok", "generated_at": generated_at, "clients": clients},
                ensure_ascii=False,
                default=str,  # diskcache values may contain datetime or custom types
            )
        except Exception as exc:
            logger.exception("STATUS_ALL payload build failed: %s", exc)
            return json.dumps(
                {"status": "error", "message": f"STATUS_ALL build error: {exc}"},
                ensure_ascii=False,
            )

    def _get_status_all_payload(self) -> str:
        """Return the STATUS_ALL JSON, using a short-lived cache when enabled.

        When ``use_status_all_cache`` is True (default), repeated calls within
        ``status_all_cache_ttl`` seconds (default 0.2 s) return the same
        pre-built payload without a DB round-trip.  This collapses N concurrent
        status polls into a single query.

        Set ``use_status_all_cache=False`` on the MessageQueue constructor to
        disable caching (useful for benchmarking or when sub-TTL freshness is
        required).
        """
        if self.use_status_all_cache:
            # Single unlocked check-and-rebuild (upstream 92720901 final):
            # the ZMQ loop is single-threaded; a lock around the rebuild was
            # introduced then reverted upstream.
            now = time.monotonic()
            entry = self._status_all_cache
            if entry is not None and now - entry[0] < self.status_all_cache_ttl:
                return entry[1]
            payload = self._build_status_all_payload()
            self._status_all_cache = (time.monotonic(), payload)
            return payload
        return self._build_status_all_payload()

    def request_status_all(self) -> str:
        """Send STATUS_ALL and return the raw JSON reply (client role only)."""
        if self.role != "client":
            raise PermissionError("Only client can call 'request_status_all'")
        self.send_string(TaskCmd.STATUS_ALL)
        try:
            _, reply = self.recv_string()
        except zmq.Again:
            raise TimeoutError(
                "Timeout waiting for server response in request_status_all"
            ) from None
        if reply is None:
            raise TimeoutError("No response from server for STATUS_ALL")
        if not reply.startswith("{"):
            raise RuntimeError(f"Unexpected server reply for STATUS_ALL: {reply!r}")
        return reply

    def notify_s3(self, s3_object):
        """Notify the server about a new S3 event."""
        if self.role != "client":
            raise PermissionError("notify_s3 can only be used by the client.")

        # print(f"sending data to port: {self.port}")
        self.send_string(s3_object)

    def request_model_command(
        self, client_id, model_cmd, payload_json="", timeout_ms=300_000
    ):
        """Send a model registry command and receive a JSON response.

        Args:
            client_id: Username or client identifier
            model_cmd: TaskCmd constant (e.g. MODEL_REGISTER, MODEL_LIST, JOB_SUBMIT, JOB_CANCEL)
            payload_json: JSON-encoded payload string
            timeout_ms: Receive timeout in milliseconds. Defaults to 300s to
                accommodate slow model downloads; control-plane queries should
                pass a shorter value so they fail fast when the server is
                unresponsive or does not support the command.
        Returns:
            JSON string response from server

        Raises:
            TimeoutError: if the server does not respond within ``timeout_ms``.
        """
        if self.role != "client":
            raise PermissionError("Only client can call 'request_model_command'")

        safe_client_id = (client_id or "").replace(":", "_")
        request_str = f"{model_cmd}:{safe_client_id}:{payload_json}"

        old_rcv = self.socket.getsockopt(zmq.RCVTIMEO)
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))

        self.send_string(request_str)
        try:
            _, reply = self.recv_string()
        finally:
            self.socket.setsockopt(zmq.RCVTIMEO, old_rcv)

        # recv_string() swallows zmq.Again and returns (None, None) on timeout,
        # so a ``None`` reply means the server never answered within the window.
        # Surface a clear error instead of letting callers crash on ``None``.
        if reply is None:
            raise TimeoutError(
                f"No response from server for '{model_cmd}' within "
                f"{timeout_ms / 1000:.0f}s. The server may be unreachable, "
                f"overloaded, or running a version that does not support this "
                f"command."
            )

        return reply

    def request_job_list(self, client_id, filters=None):
        """Send a JOB_LIST command and receive a JSON response.

        Args:
            client_id: Username or client identifier
            filters: Optional dict with keys: status, job_type, project_id, limit
        Returns:
            Parsed dict with 'status' and 'jobs' keys
        """
        payload = json.dumps(filters or {})
        raw = self.request_model_command(client_id, TaskCmd.JOB_LIST, payload)
        return json.loads(raw)
