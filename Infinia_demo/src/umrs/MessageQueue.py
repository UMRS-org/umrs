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

class MessageQueue:
    def __init__(self, role='server', queue_type='control', server_ip="127.0.0.1", port=BASE_PORT, target_nb_messages=100, target_message_size=4096):
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
        self.target_nb_messages = target_nb_messages
        self.target_message_size = target_message_size
        self.context = zmq.Context()
        self.client_dictionary={}

        # Initialize the selected queue type
        if self.type == "control":
            self.port = BASE_PORT
            self._initialize_control_queue()
        elif self.type == "data":
            self._initialize_data_queue()
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
        else:  # client
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{self.server_ip}:{self.port}")

    def _initialize_data_queue(self):
        if self.role == 'server':
            self.socket = self.context.socket(zmq.SUB)
            self.socket.bind(f"tcp://*:{self.port}")  # Data channel
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics
        else:  # client
            self.socket = self.context.socket(zmq.PUB)
            self.socket.connect(f"tcp://{self.server_ip}:{self.port}")

    # ------------------------ Core Methods ------------------------

    def send_string(self, string):
        # print(f"Sending '{string}' to '{self.queue_type}")
        self.socket.send_string(string)

    def recv_string(self):
        # print(f"Sending '{string}' to '{self.queue_type}")
        self.socket.recv_string(string)

    def set_timeout(self, timeout_ms):
        self.socket.RCVTIMEO = timeout_ms  

    # ------------------------ Close ------------------------
    
    def close(self):
        # Close data sockets for clients (if any)
        for entry in self.client_dictionary.values():
            data_socket = entry.get("data_socket")
            if data_socket:
                try:
                    data_socket.close()
                except Exception as e:
                    print(f"Warning: Failed to close data socket: {e}")

        # Close main socket
        try:
            self.socket.close()
        except Exception as e:
            print(f"Warning: Failed to close main socket: {e}")

        print("All Queue(s) closed successfully.")
    
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
                self._send_string("READY!")

            elif cmd = "CREATE_DATA_QUEUE":  # prepare PUB-SUB request
                client_id = splited_msg[1]
                if client_id in self.client_dictionary and "port" in self.client_dictionary[client_id]:
                    port = self.client_dictionary[client_id]["port"]
                else:
                    max_port_entry = max(self.client_dictionary.values(), key=lambda x: x.get("port", BASE_PORT))
                    port = max_port_entry.get("port", BASE_PORT) + 1
                    self.client_dictionary.setdefault(client_id, {})["port"] = port
                    data_socket = MessageQueue(role='server', queue_type='data', server_ip=self.server_ip, port=port)
                    self.client_dictionary[client_id]["data_socket"] = data_socket
                    
                reply = f"tcp://{self.server_ip}:{port}"  # Replace with server IP in real usage
                self._send_string(reply)

            elif cmd = "OBJECT:":
                client_id = splited_msg[1]
                nb_objects = int(splited_msg[2])
                if client_id in self.client_dictionary and "data_socket" in self.client_dictionary[client_id]:
                    data_socket = self.client_dictionary[client_id]["data_socket"]
                    objects =  data_socket.wait_for_s3_events(nb_objects)
                    self.send_string("ACKNOWLEDGE")
                    return objects
                else:
                    self._send_string("PORT ERROR")
                    
    def wait_for_s3_events(self, nb_objects, debug=False):
        """Block until new messages arrive, then read all available messages."""
        if self.role != 'server':
            raise PermissionError(f"Only server can call '{wait_for_s3_events}'.")
        
        messages = []
        print("Listening for new messages...")
        while len(messages) < nb_objects:
            message = self._receive_message(blocking=True)
            if message:
                print("Received :", message)
                messages.append(message)

        return messages

    # ------------------------ client methods ------------------------
    
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
        elif reply == "PORT ERROR":
            raise RuntimeError(f"Error: the server cannot find the data queue for user {client_id}.")
        else:
            raise RuntimeError(f"Unexpected error on server side for user {client_id}: {reply}.")

    def notify_s3(self, s3_object):
        """Notify the server about a new S3 event."""
        if self.role != 'client':
            raise PermissionError("notify_s3 can only be used by the client.")

        self.send_string(s3_object)
        
        print(f"Notifying: {s3_object}")

        
    # ------------------------ Private Methods ------------------------


    def _receive_message(self, blocking=False):
        """Receive message from ZeroMQ queue."""
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        timeout = None if blocking else 0

        socks = dict(poller.poll(timeout))
        if self.socket in socks:
            return self.socket.recv_string()
        return None



                
