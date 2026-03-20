import os
import tempfile
import json
from umrs.TaskCmdStatus import TaskCmd
try:
    import zmq
except ImportError as e:
    raise ImportError("Cannot find zeroMQ module") from e

BASE_PORT = 5555
DEFAULT_NB_MSG = 10240
class MessageQueue:
    def __init__(self, role='server', queue_type='control', server_ip="127.0.0.1", port=BASE_PORT,
                 max_nb_messages=DEFAULT_NB_MSG, timeout_ms=8000, tmp_dir = None, task_manager = None, status_store=None):
        """
        Initializes the MessageQueue.

        Args:
            queue_type (str): 'control' or 'data'.
            role (str): 'server' or 'client'.
            server_ip (str): IP address of server.
            port (int): Port number.
            target_nb_messages (int): Max number of messages in queue (for POSIX/SYSV).
            target_message_size (int): Max size per message (for POSIX/SYSV).
            status_store: a dictionnary to store event related to task in server.
                          Shared with UMRSTasks (ZeroMQ reads theis event while UMRSTasks writes it)
        """
        self.role = role.lower()
        self.type = queue_type.lower()
        # Determine server_ip: prefer explicit env var so other containers/hosts can connect
        env_server_host = os.environ.get("UMRS_SERVER_HOST")
        if env_server_host:
            self.server_ip = env_server_host
        else:
            self.server_ip = server_ip
        self.context = zmq.Context()
        self.client_dictionary={}
        self.timeout = timeout_ms
        self.max_msg = max_nb_messages if max_nb_messages > 0 else DEFAULT_NB_MSG
        self.tmp_dir = tmp_dir
        self.task_manager = task_manager
        self.status_store = status_store
        self.debug = False  # disable noisy send/recv logs by default

        # Initialize the selected queue type
        if self.type == "control":
            # Use provided port (allows non-standard ports for Docker/HPC)
            self.port = port
            self._initialize_control_queue()
        elif self.type == "data":
            if port == BASE_PORT:
                raise ValueError(f"Cannot use port {BASE_PORT} for data channel")
            self.port = port
            self._initialize_data_queue()
        else:
            raise ValueError(f"Unsupported queue type: {self.type}")

    # ------------------------ Initializers ------------------------


    def _initialize_data_queue(self):
        if self.role == 'server':
            # Server uses PULL socket to receive pushed data
            self.socket = self.context.socket(zmq.PULL)
            self.socket.bind(f"tcp://*:{self.port}")  # Data channel bind
            self.socket.setsockopt(zmq.RCVHWM, self.max_msg)  # e.g. max queued messages
            self.set_timeouts(self.timeout, self.timeout)
        else:  # client
            # Client uses PUSH socket to send data
            self.socket = self.context.socket(zmq.PUSH)
            self.socket.connect(f"tcp://{self.server_ip}:{self.port}")
            self.socket.setsockopt(zmq.SNDHWM, self.max_msg)  # e.g. max queued messages
            self.set_timeouts(self.timeout, self.timeout)


    def _initialize_control_queue(self):
        if self.role == 'server':
            self.socket = self.context.socket(zmq.ROUTER)
            self.socket.bind(f"tcp://*:{self.port}")
            self.set_timeouts(-1, self.timeout)
        else:  # client
            self.socket = self.context.socket(zmq.DEALER)
            self.socket.connect(f"tcp://{self.server_ip}:{self.port}")
            self.set_timeouts(self.timeout, self.timeout)

    # ------------------------ Core Methods ------------------------

    def send_string(self, string, client_id=None):
        if self.debug:
            print(
                f"[SEND] role={self.role}, type={self.type}, "
                f"client_id={client_id}, msg='{string}'"
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
                    self.socket.send_multipart([
                        client_id,
                        b"",  # Empty delimiter for REQ compatibility
                        string.encode("utf-8")
                    ])
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
            raise RuntimeError("Timeout while sending message")

        except zmq.ZMQError as e:
            raise RuntimeError(f"ZMQ error while sending: {e}")

        except Exception as e:
            raise RuntimeError(f"Unexpected error while sending '{string}' from {self.role} to {self.type}: {e}")

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
                # ROUTER receives: [client_id][empty_delimiter][payload] from REQ
                # or [client_id][payload] from DEALER
                frames = self.socket.recv_multipart()
                if len(frames) == 3:
                    # REQ socket: [client_id][empty][payload]
                    client_id, _empty, msg = frames
                elif len(frames) == 2:
                    # DEALER socket: [client_id][payload]
                    client_id, msg = frames
                else:
                    raise ValueError(f"Unexpected frame count: {len(frames)}")
                return client_id, msg.decode()

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
                # PULL socket: single frame
                msg = self.socket.recv_string()
                return None, msg

        except zmq.Again:
            return None, None

    def set_timeouts(self, recv_timeout_ms, send_timeout_ms):
        self.socket.RCVTIMEO = recv_timeout_ms
        self.socket.SNDTIMEO = send_timeout_ms



    def send_file(self, send_file_path, chunk_size=4096):
        if self.role != 'client' or self.type != 'data':
            raise PermissionError("Sending file can only be done by the client using the data queue")
        if not os.path.isfile(send_file_path):
            raise FileNotFoundError(f"File '{send_file_path}' not found")

        with open(send_file_path, "rb") as f:
            # Send file in chunks
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                self.socket.send(chunk)
            # Send end signal (can be empty frame)
            self.socket.send(b'__END_OF_FILE__')


    def recv_file(self, out_dir="/tmp"):
        if self.role != 'server' or self.type != 'data':
            raise PermissionError("Receiving file can only be done by the server using the data queue")

        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)  # Ensure directory exists

        with tempfile.NamedTemporaryFile(mode='w+b', dir=out_dir, delete=False) as tmp_file:
            while True:
                chunk = self.socket.recv()
                if chunk == b'__END_OF_FILE__':
                    break
                tmp_file.write(chunk)
            tmp_file.flush()
            tmp_path = os.path.abspath(tmp_file.name)
        return tmp_path



    # ------------------------ Close ------------------------

    def close(self):
        # Close data sockets for clients (if any)
        for entry in self.client_dictionary.values():
            data_queue = entry.get("data_queue")
            if data_queue:
                try:
                    data_queue.socket.close()
                except Exception as e:
                    print(f"Warning: Failed to close data socket: {e}")

        # Close main socket
        try:
            self.socket.close()
        except Exception as e:
            print(f"Warning: Failed to close main socket: {e}")


    # ------------------------ Server methods ------------------------
    def server_control_queue_loop(self):
        if self.role != "server":
            raise PermissionError("Only server can call 'server_control_queue_loop'.")

        while True:
            zmq_sender_id, msg = self.recv_string() # Blocking call
            print(f"Received request: {msg}", flush=True)
            # messgae structure should be "{command}:{username}:{other information (e.g file/object count)}"
            splited_msg = msg.split(":")

            cmd = splited_msg[0]
            if cmd == TaskCmd.HAND_CHECK:  # handshake
                try:
                    self.send_string("READY!", zmq_sender_id)
                except RuntimeError as e:
                    print(f"Error in client handshake: {e}")

            elif cmd == TaskCmd.CREATE_DATA_QUEUE:  # prepare PUB-SUB request
                client_id = splited_msg[1]
                if client_id in self.client_dictionary and "port" in self.client_dictionary[client_id]:
                    port = self.client_dictionary[client_id]["port"]
                else:
                    max_port = max((entry.get("port", BASE_PORT) for entry in self.client_dictionary.values()), default=BASE_PORT)
                    port = max_port + 1
                    self.client_dictionary.setdefault(client_id, {})["port"] = port
                    data_queue = MessageQueue(role='server', queue_type='data', server_ip=self.server_ip, port=port)
                    self.client_dictionary[client_id]["data_queue"] = data_queue

                reply = f"tcp://{self.server_ip}:{port}"  # Replace with server IP in real usage
                try:
                    self.send_string(reply, zmq_sender_id)
                except RuntimeError as e:
                    print(f"Error in sending back port to client: {e}")

            elif cmd == TaskCmd.PROCESS_OBJECTS:  # prepare PUB-SUB request
                client_id = splited_msg[1]
                nb_objects = int(splited_msg[2])
                return self._process_object_cmd(client_id, nb_objects, zmq_sender_id)

            elif cmd == TaskCmd.FILE:  # prepare PUB-SUB request
                client_id = splited_msg[1]
                nb_files = int(splited_msg[2])
                # include client_id in the payload for server-side identity enforcement
                return self._process_file_cmd(client_id, nb_files, zmq_sender_id)
            elif cmd ==  TaskCmd.STATUS:
                client_id = splited_msg[1] if len(splited_msg) > 1 else None
                status_key = splited_msg[2] if len(splited_msg) > 2 else "all"
                payload = self._get_status_payload(client_id, status_key)
                try:
                    self.send_string(payload, zmq_sender_id)
                except RuntimeError as e:
                    print(f"Error sending status payload: {e}")
                continue
            elif cmd == TaskCmd.WAIT_OUTPUT :
                client_id = splited_msg[1] if len(splited_msg) > 1 else None
                args = splited_msg[2] if len(splited_msg) > 2 else None
                if client_id is None or args is None:
                    try:
                        self.send_string("CMD_ERROR", zmq_sender_id)
                    except RuntimeError as e:
                        print(f"Error sending status payload: {e}")
                    continue
                splited_args = args.split(";")
                bucket_name = splited_args[0]
                timeout = int(splited_args[1]) if len(splited_args) > 1 else None
                return TaskCmd.WAIT_OUTPUT, (client_id, bucket_name, timeout, zmq_sender_id)

            elif cmd == TaskCmd.EXPORT_EMBEDDINGS:
                client_id = splited_msg[1] if len(splited_msg) > 1 else None
                export_args = splited_msg[2] if len(splited_msg) > 2 else None
                if client_id is None or export_args is None:
                    try:
                        self.send_string("CMD_ERROR", zmq_sender_id)
                    except RuntimeError as e:
                        print(f"Error sending export error: {e}")
                    continue
                return TaskCmd.EXPORT_EMBEDDINGS, (client_id, export_args, zmq_sender_id)

            elif cmd in (TaskCmd.RECOMPUTE, TaskCmd.UPDATE, TaskCmd.CHECK, TaskCmd.RELOAD):
                # Acknowledge and let the server main loop perform the operation
                try:
                    self.send_string("ACKNOWLEDGE", zmq_sender_id)
                except RuntimeError as e:
                    print(f"Error acknowledging {cmd} request: {e}")
                client_id = splited_msg[1] if len(splited_msg) > 1 else None
                return cmd, client_id

            elif cmd in (TaskCmd.MODEL_REGISTER, TaskCmd.MODEL_LIST, TaskCmd.MODEL_DOWNLOAD):
                # Model registry commands: immediate request/reply with JSON payload
                client_id = splited_msg[1] if len(splited_msg) > 1 else None
                # The rest of the message (after "CMD:client_id:") is the JSON payload
                payload_str = ":".join(splited_msg[2:]) if len(splited_msg) > 2 else ""
                return cmd, (client_id, payload_str, zmq_sender_id)

            elif cmd in (TaskCmd.JOB_SUBMIT, TaskCmd.JOB_CANCEL):
                # Job tracking commands: immediate request/reply with JSON payload
                client_id = splited_msg[1] if len(splited_msg) > 1 else None
                payload_str = ":".join(splited_msg[2:]) if len(splited_msg) > 2 else ""
                return cmd, (client_id, payload_str, zmq_sender_id)

            else:
                print(f"WARNING: Ignoring unknown command from unexpected client: {cmd[:80]}")
                try:
                    self.send_string("CMD_ERROR", zmq_sender_id)
                except RuntimeError:
                    pass
                continue

    def wait_for_s3_events(self, nb_objects, debug=False):
        """Block until new messages arrive, then read all available messages."""
        if self.role != 'server':
            raise PermissionError("Only server can call 'wait_for_s3_events'.")

        messages = []
        print(f"Listening for {nb_objects} new messages...")
        nb_received = 0
        while nb_received < nb_objects:
            _, message = self.recv_string()
            if message:
                print("Received :", message)
                messages.append(message)
                nb_received += 1

        return messages

    def _process_file_cmd(self, client_id, nb_files, zmq_sender_id):
        # Verify client has a data socket ready
        if client_id in self.client_dictionary and "data_queue" in self.client_dictionary[client_id]:
            data_queue = self.client_dictionary[client_id]["data_queue"]
            port = self.client_dictionary[client_id]["port"]
            print(f"Listening data to port {port}")

            path_list = []
            if not self.tmp_dir or not os.path.exists(self.tmp_dir) :
                print(f"Warning no directory set for temporary receiving files or not accessible, using {os.devnull}")
                recv_dir = os.devnull
            else :
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
                print(f"Error in sending acknowleding client: {e}")
            # Return client_id along with received file paths so server can enforce identity
            return TaskCmd.FILE, (client_id, path_list)
        else: # cannot find the client in the client_dictionnary
            try:
                self.send_string("NO_DATA_QUEUE", zmq_sender_id)
            except RuntimeError as e:
                print(f"Error in sending 'no available data queue' to client: {e}")

    def _process_object_cmd(self, client_id, nb_objects, zmq_sender_id):
        if client_id in self.client_dictionary and "data_queue" in self.client_dictionary[client_id]:
            data_queue = self.client_dictionary[client_id]["data_queue"]
            port = self.client_dictionary[client_id]["port"]
            objects =  data_queue.wait_for_s3_events(nb_objects)
            try:
                self.send_string("ACKNOWLEDGE", zmq_sender_id)
            except RuntimeError as e:
                print(f"Error in sending acknowleding client: {e}")
            return TaskCmd.PROCESS_OBJECTS, (client_id, objects)
        else:
            print(f"Warning: no data queue for client={client_id}")
            try:
                self.send_string("NO_DATA_QUEUE", zmq_sender_id)
            except RuntimeError as e:
                print(f"Error in sending 'no available data queue' to client: {e}")
        return None

    # ------------------------ client methods ------------------------

    def reset_control_queue(self):
        if self.role != 'client' or self.type != 'control':
            raise PermissionError("reset_control_plane can only be called by client control queue")

        # Close existing socket if it exists
        if hasattr(self, 'socket'):
            try:
                self.socket.close()
            except Exception:
                # Best-effort cleanup: ignore errors while closing old socket
                pass

        # Create a new REQ socket and connect
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.server_ip}:{self.port}")

    def is_server_ready(self, timeout_ms=5000):
        if self.role != "client":
            raise PermissionError(f"Only client can call 'get_data_queue'")

        if self.type != "control":
            raise PermissionError(f"Only control message queue can check server")

        print("Checking for server availability...")
        if timeout_ms < 1500:
            timeout_ms = 1500

        self.set_timeouts(timeout_ms, timeout_ms)
        try:
            try:
                self.send_string(TaskCmd.HAND_CHECK)
                _, answer = self.recv_string()
                if answer == "READY!":
                    return True
            except zmq.Again:
                # Retry once on timeout
                try:
                    # but first need to reset the socket to avoid error.
                    self.reset_control_queue()
                    self.set_timeouts(timeout_ms, timeout_ms)
                    self.send_string("READY?")
                    _, answer = self.recv_string()
                    if answer == "READY!":
                        return True
                except zmq.Again:
                    return False
            return False
        finally:
            self.set_timeouts(self.timeout, self.timeout) # always reset timeout before exiting

    def get_data_queue(self, client_id):
        if self.role != "client":
            raise PermissionError(f"Only client can call 'get_data_queue'")

        request_str = f"CREATE_DATA_QUEUE:{client_id}"
        self.send_string(request_str)

        try:
            _, reply = self.recv_string()
        except zmq.Again:
            raise RuntimeError("Timeout waiting for server response in get_data_queue")

        if reply.startswith("tcp://"):
            # Remove protocol prefix
            endpoint = reply[len("tcp://"):]
            # Split IP and port
            server_ip, port_str = endpoint.split(":")
            port = int(port_str)

            # Create MessageQueue with parsed IP and port
            data_mq = MessageQueue(
                role='client',
                queue_type='data',
                server_ip=server_ip,
                port=port
            )
            return data_mq
        else:
            raise RuntimeError(f"Unexpected reply from server: {reply}")

    def notify_server(self, client_id, notification_type, nb_objects, verbose=False):
        if self.role != "client":
            raise PermissionError(f"Only client can call 'notify_server'")

        request_str = f"{notification_type}:{client_id}:{nb_objects}"
        self.send_string(request_str)

        try:
            _, reply = self.recv_string()
        except zmq.Again:
            raise RuntimeError("Timeout waiting for server response in notify_server")

        if reply == "ACKNOWLEDGE":
            if verbose:
                print("Server acknowledges reception of the objects")
            return True
        elif reply == "NO_DATA_QUEUE":
            raise RuntimeError(f"Error: the server cannot find the data queue for user {client_id}.")
        else:
            raise RuntimeError(f"Unexpected error on server side for user {client_id}: {reply}.")

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
            raise RuntimeError("Timeout waiting for server response in request_status")

        return reply

    def request_export_embeddings(self, client_id, path_to_bucket, export_format="csv"):
        """Send EXPORT_EMBEDDINGS request and receive the serialised result."""
        if self.role != "client":
            raise PermissionError("Only client can call 'request_export_embeddings'")

        safe_client_id = (client_id or "").replace(":", "_")
        safe_args = f"{path_to_bucket};{export_format}".replace(":", "_")
        request_str = f"{TaskCmd.EXPORT_EMBEDDINGS}:{safe_client_id}:{safe_args}"

        # Allow a generous receive timeout for large tables
        old_rcv = self.socket.getsockopt(zmq.RCVTIMEO)
        self.socket.setsockopt(zmq.RCVTIMEO, 120_000)

        self.send_string(request_str)
        try:
            _, reply = self.recv_string()
        except zmq.Again:
            raise RuntimeError("Timeout waiting for server response in request_export_embeddings")
        finally:
            self.socket.setsockopt(zmq.RCVTIMEO, old_rcv)

        return reply

    def wait_for_new_data(self, client_id, full_bucket_name, timeout=60):
        if self.role != "client":
            raise PermissionError("Only client can call 'request_status'")


        timeout_ms = 10000*(timeout*2)

        self.set_timeouts(timeout_ms,self.timeout)


        safe_client_id = (client_id or "").replace(":", "_")
        safe_args = f"{full_bucket_name};{timeout}".replace(":", "_")
        request_str = f"{TaskCmd.WAIT_OUTPUT}:{safe_client_id}:{safe_args}"
        self.send_string(request_str)

        try:
            _, reply = self.recv_string()
        except zmq.Again:
            raise RuntimeError("Timeout waiting for server response in request_status")

        finally:
            self.set_timeouts(self.timeout,self.timeout)

        return reply


    def get_status_store(self):
        return self.status_store

    def set_debug(self, enabled: bool):
        self.debug = bool(enabled)

    def _get_status_payload(self, client_id, status_key):
        store = self.status_store or {}
        key = (status_key or "all").lower()

        if not client_id:
            payload = {"status": "error", "message": "Missing client identifier"}
            return json.dumps(payload, ensure_ascii=False)

        uncompleted_key_list = None
        if self.task_manager is not None:
            uncompleted_key_list = self.task_manager.tasks_status.all_uncompleted_task(client_id, status_key)

        client_store = store.get(client_id, {})
        if key in ("all", "*"):
            if not client_store:
                payload = {"status": "error", "message": "No status entries", "client": client_id, "uncompleted_key_list": uncompleted_key_list}
            else:
                payload = {"status": "ok", "client": client_id, "summaries": client_store, "uncompleted_key_list": uncompleted_key_list}
            return json.dumps(payload, ensure_ascii=False)

        entry = client_store.get(key)
        if not entry:
            payload = {"status": "error", "message": f"No status for '{key}'", "client": client_id, "uncompleted_key_list": uncompleted_key_list}
        else:
            payload = {"status": "ok", "client": client_id, "summary": entry, "uncompleted_key_list": uncompleted_key_list}
        return json.dumps(payload, ensure_ascii=False)

    def notify_s3(self, s3_object):
        """Notify the server about a new S3 event."""
        if self.role != 'client':
            raise PermissionError("notify_s3 can only be used by the client.")

        # print(f"sending data to port: {self.port}")
        self.send_string(s3_object)

    def request_model_command(self, client_id, model_cmd, payload_json=""):
        """Send a model registry command and receive a JSON response.

        Args:
            client_id: Username or client identifier
            model_cmd: TaskCmd constant (e.g. MODEL_REGISTER, MODEL_LIST, JOB_SUBMIT, JOB_CANCEL)
            payload_json: JSON-encoded payload string
        Returns:
            JSON string response from server
        """
        if self.role != "client":
            raise PermissionError("Only client can call 'request_model_command'")

        safe_client_id = (client_id or "").replace(":", "_")
        request_str = f"{model_cmd}:{safe_client_id}:{payload_json}"

        # Allow generous timeout for model downloads
        old_rcv = self.socket.getsockopt(zmq.RCVTIMEO)
        self.socket.setsockopt(zmq.RCVTIMEO, 300_000)

        self.send_string(request_str)
        try:
            _, reply = self.recv_string()
        except zmq.Again:
            raise RuntimeError("Timeout waiting for server response in request_model_command")
        finally:
            self.socket.setsockopt(zmq.RCVTIMEO, old_rcv)

        return reply





