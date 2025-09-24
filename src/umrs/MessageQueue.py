import os
import tempfile
try:
    import zmq
except ImportError as e:
    raise ImportError("Cannot find zeroMQ module") from e

BASE_PORT = 5555
DEFAULT_NB_MSG = 10240
class MessageQueue:
    def __init__(self, role='server', queue_type='control', server_ip="127.0.0.1", port=BASE_PORT, max_nb_messages=DEFAULT_NB_MSG, timeout_ms=8000):
        """
        Initializes the MessageQueue.

        Args:
            queue_type (str): 'control' or 'data'.
            role (str): 'server' or 'client'.
            server_ip (str): IP address of server.
            port (int): Port number.
            target_nb_messages (int): Max number of messages in queue (for POSIX/SYSV).
            target_message_size (int): Max size per message (for POSIX/SYSV).
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
        
        # Initialize the selected queue type
        if self.type == "control":
            self.port = BASE_PORT
            self._initialize_control_queue()
        elif self.type == "data":
            if port == BASE_PORT:
                raise ValueError(f"Cannot use port {BASE_PORT} for data channel")
            self.port = port
            self._initialize_data_queue()
        else:
            raise ValueError(f"Unsupported queue type: {self.type}")

    # ------------------------ Initializers ------------------------

    def _initialize_control_queue(self):
        if self.role == 'server':
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{self.port}")  # Control channel
            self.set_timeouts(-1, self.timeout) #receiving is always blocking. 
        else:  # client
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{self.server_ip}:{self.port}")
            self.set_timeouts(self.timeout, self.timeout)


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


    # ------------------------ Core Methods ------------------------

    def send_string(self, string):
        print(f"Sending '{string}' to {self.type}")
        try:
            self.socket.send_string(string)
        except zmq.Again:
            raise RuntimeError(f"Timeout: sending '{string}' from {self.role} to {self.type} failed (socket not ready or high-water mark reached)")
        except zmq.ZMQError as e:
            raise RuntimeError(f"ZeroMQ error while sending '{string}' from {self.role} to {self.type}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while sending '{string}' from {self.role} to {self.type}: {e}")
    

    def recv_string(self):
        return self.socket.recv_string()

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
            msg = self.recv_string() # Blocking call
            print(f"Received request: {msg}")
            # messgae structure should be "{command}:{username}:{other information (e.g file/object count)}"
            splited_msg = msg.split(":")
            
            cmd = splited_msg[0]
            if cmd == "READY?":  # handshake
                try: 
                    self.send_string("READY!")
                except RuntimeError as e:
                    print(f"Error in client handshake: {e}")
                    
            elif cmd == "CREATE_DATA_QUEUE":  # prepare PUB-SUB request
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
                    self.send_string(reply)
                except RuntimeError as e:
                    print(f"Error in sending back port to client: {e}")

            elif cmd == "OBJECT":  # prepare PUB-SUB request
                client_id = splited_msg[1]
                nb_objects = int(splited_msg[2])
                return self._process_object_cmd(client_id, nb_objects)

            elif cmd == "FILE":  # prepare PUB-SUB request
                client_id = splited_msg[1]
                nb_files = int(splited_msg[2])
                return self._process_file_cmd(client_id, nb_files)
            elif cmd in ("RECOMPUTE", "UPDATE", "CHECK", "RELOAD"):
                # Acknowledge and let the server main loop perform the operation
                try:
                    self.send_string("ACKNOWLEDGE")
                except RuntimeError as e:
                    print(f"Error acknowledging {cmd} request: {e}")
                return cmd, None
            else:
                raise RuntimeError(f"Unknown command: {cmd}")
                    
    def wait_for_s3_events(self, nb_objects, debug=False):
        """Block until new messages arrive, then read all available messages."""
        if self.role != 'server':
            raise PermissionError("Only server can call 'wait_for_s3_events'.")
        
        messages = []
        print(f"Listening for {nb_objects} new messages...")
        nb_received = 0
        while nb_received < nb_objects:
            message = self.recv_string()
            if message:
                print("Received :", message)
                messages.append(message)
                nb_received += 1

        return messages

    def _process_file_cmd(self, client_id, nb_files):
        # Verify client has a data socket ready
        if client_id in self.client_dictionary and "data_queue" in self.client_dictionary[client_id]:
            data_queue = self.client_dictionary[client_id]["data_queue"]
            port = self.client_dictionary[client_id]["port"]
            print(f"Listening data to port {port}")
            
            path_list = []
            # For each expected file:
            for _ in range(nb_files):
                # Receive file data into it
                file_path = data_queue.recv_file(out_dir='./tmp')
                # Keep track of temp files
                path_list.append(file_path)

            # Send acknowledgment to client after receiving all files
            try: 
                self.send_string("ACKNOWLEDGE")
            except RuntimeError as e:
                print(f"Error in sending acknowleding client: {e}")
            return "FILE", path_list
        else: # cannot find the client in the client_dictionnary         
            try: 
                self.send_string("NO_DATA_QUEUE")
            except RuntimeError as e:
                print(f"Error in sending 'no available data queue' to client: {e}")

    def _process_object_cmd(self, client_id, nb_objects):
        if client_id in self.client_dictionary and "data_queue" in self.client_dictionary[client_id]:
            data_queue = self.client_dictionary[client_id]["data_queue"]
            port = self.client_dictionary[client_id]["port"]
            print(f"Listening data to port {port}")
            objects =  data_queue.wait_for_s3_events(nb_objects)
            try: 
                self.send_string("ACKNOWLEDGE")
            except RuntimeError as e:
                print(f"Error in sending acknowleding client: {e}")
            return "OBJECT", objects
        else:
            try: 
                self.send_string("NO_DATA_QUEUE")
            except RuntimeError as e:
                print(f"Error in sending 'no available data queue' to client: {e}")

    # ------------------------ client methods ------------------------
    
    def reset_control_queue(self):
        if self.role != 'client' or self.type != 'control':
            raise PermissionError("reset_control_plane can only be called by client control queue")

        # Close existing socket if it exists
        if hasattr(self, 'socket'):
            try:
                self.socket.close()
            except Exception:
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
                self.send_string("READY?")
                answer = self.recv_string()
                if answer == "READY!":
                    return True
            except zmq.Again:
                # Retry once on timeout
                try:
                    # but first need to reset the socket to avoid error. 
                    self.reset_control_queue()
                    self.set_timeouts(timeout_ms, timeout_ms)
                    self.send_string("READY?")
                    answer = self.recv_string()
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
            reply = self.recv_string()
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
            reply = self.recv_string()
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

    def notify_s3(self, s3_object):
        """Notify the server about a new S3 event."""
        if self.role != 'client':
            raise PermissionError("notify_s3 can only be used by the client.")

        print(f"sending data to port: {self.port}")
        self.send_string(s3_object)
        
        print(f"Notifying: {s3_object}")

        


                
