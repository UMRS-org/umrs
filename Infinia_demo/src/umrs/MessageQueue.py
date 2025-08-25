import platform
import sys
import os
import hashlib
import time

try:
    import zmq
except ImportError:
    zmq = None

if not (zmq):
    raise ImportError("Cannot find zeroMQ module")
import zmq

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

    # ------------------------ Close ------------------------
    
    def close(self):
        # Close data sockets for clients (if any)
        for entry in self.client_dictionary.values():
            data_socket = entry.get("data_socket")
            if data_socket:
                try:
                    data_socket.socket.close()
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
            raise PermissionError(f"Only server can call '{server_control_queue_loop}'.")

        while True:
            msg = self.recv_string()
            print(f"Received request: {msg}")
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
                    data_socket = MessageQueue(role='server', queue_type='data', server_ip=self.server_ip, port=port)
                    self.client_dictionary[client_id]["data_socket"] = data_socket
                    
                reply = f"tcp://{self.server_ip}:{port}"  # Replace with server IP in real usage
                try:
                    self.send_string(reply)
                except RuntimeError as e:
                    print(f"Error in sending back port to client: {e}")

            elif cmd == "OBJECT":  # prepare PUB-SUB request
                client_id = splited_msg[1]
                nb_objects = int(splited_msg[2])
                if client_id in self.client_dictionary and "data_socket" in self.client_dictionary[client_id]:
                    data_socket = self.client_dictionary[client_id]["data_socket"]
                    port = self.client_dictionary[client_id]["port"]
                    print(f"Listening data to port {port}")
                    objects =  data_socket.wait_for_s3_events(nb_objects)
                    try: 
                        self.send_string("ACKNOWLEDGE")
                    except RuntimeError as e:
                        print(f"Error in sending acknowleding client: {e}")
                    return objects
                else:
                    try: 
                        self.send_string("NO_DATA_SOCKET")
                    except RuntimeError as e:
                        print(f"Error in sending 'no available port' to client: {e}")

            else:
                raise RuntimeError("Unknown command: {cmd}")
                    
    def wait_for_s3_events(self, nb_objects, debug=False):
        """Block until new messages arrive, then read all available messages."""
        if self.role != 'server':
            raise PermissionError(f"Only server can call '{wait_for_s3_events}'.")
        
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

    def notify_server(self, client_id, nb_objects, verbose=False):
        if self.role != "client":
            raise PermissionError(f"Only client can call 'notify_server'")

        request_str = f"OBJECT:{client_id}:{nb_objects}"
        self.send_string(request_str)

        try:
            reply = self.recv_string()
        except zmq.Again:
            raise RuntimeError("Timeout waiting for server response in notify_server")

        if reply == "ACKNOWLEDGE":
            if verbose:
                print("Server acknowledges reception of the objects")
            return True
        elif reply == "NO_DATA_SOCKET":
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

        


                
