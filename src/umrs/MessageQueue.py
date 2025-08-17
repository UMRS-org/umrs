import platform
import sys
import os
import hashlib
import time

# Conditional imports based on queue type and OS
try:
    import posix_ipc
except ImportError:
    posix_ipc = None

try:
    import sysv_ipc
except ImportError:
    sysv_ipc = None

try:
    import zmq
except ImportError:
    zmq = None

if not (zmq or sysv_ipc or posix_ipc):
    raise ImportError("No usefull message queue module found (Posix IPC or SYSV or zeroMQ.)")


DEFAULT_QUEUE_NAME = '/tmp/MQ_Queue'

class MessageQueue:
    def __init__(self, queue_type='zeromq', role='server', queue_name=DEFAULT_QUEUE_NAME, target_nb_messages=100, target_message_size=4096):
        """
        Initializes the MessageQueue.

        Args:
            queue_type (str): 'zeromq', 'posix_ipc', or 'sysv_ipc'.
            role (str): 'server' or 'client'.
            queue_name (str): Unique queue identifier.
            target_nb_messages (int): Max number of messages in queue (for POSIX/SYSV).
            target_message_size (int): Max size per message (for POSIX/SYSV).
        """
        self.queue_type = queue_type.lower()
        self.role = role.lower()
        self.queue_name = queue_name
        self.target_nb_messages = target_nb_messages
        self.target_message_size = target_message_size

        # Check for macOS limitations
        if platform.system() == 'Darwin' and self.queue_type == 'posix_ipc':
            raise EnvironmentError("POSIX IPC is not supported on macOS. Use sysv_ipc or zeromq instead.")

        # Initialize the selected queue type
        if self.queue_type == 'zeromq':
            self._init_zeromq()
        elif self.queue_type == 'posix_ipc':
            self._init_posix_ipc()
        elif self.queue_type == 'sysv_ipc':
            self._init_sysv_ipc()
        else:
            raise ValueError(f"Unsupported queue type: {self.queue_type}")

    # ------------------------ Initializers ------------------------

    def _init_zeromq(self):
        if zmq is None:
            raise ImportError("zeroMQ module not found.")

        """Initialize ZeroMQ context and sockets."""

        ## intitialize port based on queue_name
        ZMQ_PORT_MAP = {
            DEFAULT_QUEUE_NAME: 5555,
            "/tmp/response": 5556,
        }
        port = ZMQ_PORT_MAP.get(self.queue_name, 5600)
        if port == 5557:
            print(f"[ZeroMQ] Warning: Unknown queue name '{self.queue_name}'. Using special port {port}")


        self.context = zmq.Context()
        if self.role == 'server':
            self.socket = self.context.socket(zmq.SUB)
            self.socket.bind(f"tcp://127.0.0.1:{port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")            
        else:  # Client
            self.socket = self.context.socket(zmq.PUB)
            self.socket.connect(f"tcp://127.0.0.1:{port}")

    def _init_posix_ipc(self):
        """Initialize POSIX IPC message queue."""
        if posix_ipc is None:
            raise ImportError("posix_ipc module not found.")
        
        flags = posix_ipc.O_CREAT if self.role == 'server' else 0
        try :
            self.mq = posix_ipc.MessageQueue(
                self.queue_name,
                flags,
                max_messages=self.target_nb_messages,
                max_message_size=self.target_message_size
            )
        except posix_ipc.ExistentialError:
            sys.exit(f"Queue '{self.queue_name}' not found. Please ensure the server is running.")

    def _init_sysv_ipc(self):
        """Initialize SYSV IPC message queue."""
        if sysv_ipc is None:
            raise ImportError("sysv_ipc module not found.")

        # Hash the queue_name to create a unique integer key
        key = self._hash_queue_name(self.queue_name)

        try:
            self.mq = sysv_ipc.MessageQueue(key, sysv_ipc.IPC_CREX)  # Create new queue

        except sysv_ipc.ExistentialError:
            self.mq = sysv_ipc.MessageQueue(key)  # Connect if it already exists

        except sysv_ipc.ExistentialError:
            if self.role == 'server':
                sys.exit(f"SYSV IPC queue '{self.queue_name}' already exists. Stop the existing server or use a different queue.")
            else:
                sys.exit(f"SYSV IPC queue '{self.queue_name}' not found. Please ensure the server is running.")
                
        except Exception as e:
            sys.exit(f"Unexpected error during SYSV IPC setup: {e}")

    def _hash_queue_name(self, queue_name):
        """Hash the queue_name into a 32-bit integer for SYSV IPC."""
        # Use SHA-256 and take first 4 bytes for a 32-bit key
        hash_object = hashlib.sha256(queue_name.encode())
        hash_digest = hash_object.digest()
        unique_key = int.from_bytes(hash_digest[:4], byteorder='big')
    
        # Ensure the key is within SYSV IPC limits
        return unique_key % sysv_ipc.KEY_MAX
            

    # ------------------------ Core Methods ------------------------

    def wait_for_s3_events(self, debug=False):
        """Block until new messages arrive, then read all available messages."""
        if self.role != 'server':
            raise PermissionError("wait_for_s3_events can only be used by the server.")
        
        messages = []
        print("Listening for new messages...")

        # BLOCKING: Wait for the first message
        message = self._receive_message(blocking=True)
        if message:
            if message == "READY?":
                if debug:
                    print("Received handshake: 'READY?'")
                return ([], True)
            print("Received (blocking):", message)
            messages.append(message)

        # NON-BLOCKING: Check for more messages
        while True:
            message = self._receive_message(blocking=False)
            if message:
                messages.append(message)
                if debug:
                    print("Received (non-blocking):", message)
            else:
                if debug:
                    print("No more messages in queue.")
                break

        return (messages, False)

    
    def notify_s3(self, s3_object):
        """Notify the server about a new S3 event."""
        if self.role != 'client':
            raise PermissionError("notify_s3 can only be used by the client.")

        self._send(s3_object)
        
        print(f"Notifying: {s3_object}")

    def close(self):
        """Close the queue (server only)."""
        if self.role != 'server':
            raise PermissionError("close can only be used by the server.")

        if self.queue_type == 'zeromq':
            self.socket.close()
            self.context.term()
        elif self.queue_type == 'posix_ipc':
            self.mq.close()
            self.mq.unlink()
        elif self.queue_type == 'sysv_ipc':
            self.mq.remove()
        print("Queue closed successfully.")


    def handshake(self, msg):
        if msg in ["READY?", "READY!"]:
            self._send(msg)
        else:
            print("Error. Not allow to send message '{msg}' for handshake...")

    def busy_wait(self, timeout_ms):
        start_time = time.time()
        duration_ms = 0
        sleep_time_ms = 1
        max_sleep_time_ms = 100
        while duration_ms < timeout_ms:
            msg = self._receive_message(blocking=False)
            if msg :
                return msg                
            duration_ms = (time.time() - start_time) * 1000
            time.sleep(sleep_time_ms / 1000)
            sleep_time_ms = max (sleep_time_ms *2, max_sleep_time_ms)
        return "NOT READY..."
        
    # ------------------------ Private Methods ------------------------

    def _send(self, string):
        # print(f"Sending '{string}' to '{self.queue_type}")
        if self.queue_type == 'zeromq':
            self.socket.send_string(string)
        elif self.queue_type == 'posix_ipc' or self.queue_type == 'sysv_ipc':
            self.mq.send(string.encode())
        else:
            raise ValueError("Unsupported queue type.")
        
    def _receive_message(self, blocking=True):
        """Handles message reception based on the queue type and blocking mode."""
        try:
            if self.queue_type == 'zeromq':
                return self._receive_zeromq(blocking)
            elif self.queue_type == 'posix_ipc':
                return self._receive_posix_ipc(blocking)
            elif self.queue_type == 'sysv_ipc':
                return self._receive_sysv_ipc(blocking)
            else:
                raise ValueError(f"Unsupported queue type: {self.queue_type}")
        except Exception as e:
            if blocking:
                print(f"Error during blocking receive: {e}")
            return None

    def _receive_zeromq(self, blocking):
        """Receive message from ZeroMQ queue."""
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        timeout = None if blocking else 0

        socks = dict(poller.poll(timeout))
        if self.socket in socks:
            return self.socket.recv_string()
        return None

    def _receive_posix_ipc(self, blocking):
        """Receive message from POSIX IPC queue."""
        try:
            timeout = None if blocking else 0
            message, _ = self.mq.receive(timeout=timeout)
            return message.decode()
        except posix_ipc.BusyError:
            return None

    def _receive_sysv_ipc(self, blocking):
        """Receive message from SYSV IPC queue."""
        try:
            message, _ = self.mq.receive(block=blocking)
            return message.decode()
        except sysv_ipc.BusyError:
            return None
