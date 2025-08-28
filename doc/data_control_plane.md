## UMRS Communication Architecture (ZeroMQ-only)
### Overview
Removed POSIX and SYSV queue mechanisms entirely to unify communication via **ZeroMQ**.
Two distinct communication planes are used to separate control and data flows.
***
### 1. Control Plane
- Uses **REQ-REP** ZeroMQ sockets for reliable bidirectional requests.
- **Clients connect to the server’s control socket** to:
  - Check if the server is online and ready (handshake with timeout).
  - Request creation of **dedicated data queues** for sending object path notifications.
  - Notify the server when new object paths have been sent on the data plane.
- The server maintains a **cache (dictionary)** mapping client IDs to their corresponding data message queues and ports.
- Ensures synchronization and coordination without blocking long-running tasks.
- Control messages implement the following protocol:
  - **Handshake:** Client queries server readiness and waits for a timely reply.
  - **Create Data Socket:** Client requests a port; server allocates and caches a data queue for signaling object paths.
  - **Send Object Notification:** Client signals server via control plane that new object paths are available on the data queue.
***
### 2. Data Plane
- Built on **PUSH-PULL** ZeroMQ sockets for asynchronous signaling.
- The **client PUSHes object paths** (strings referencing data locations on the S3 server) to the server’s **PULL socket**.
- The server **maintains internal dictionary** mapping clients to their dedicated data queues.
- Clients obtain their data queue endpoint by asking the server via the control plane.
- Data queue ports are dynamically assigned and managed by the server.
### Improvements To Do:
- Increase **asynchrony** in control plane handling so server does not block on long requests.
- Consider **multi-threading** or async event loops to separate

### File Transfer in UMRS (e.g. for configuring the server)

There are two low-level `MessageQueue` methods that provide chunked file transfer functionality using ZeroMQ PUSH/PULL sockets:

- **`send_file(send_file_path, chunk_size=4096)`**  
  Enables a client role using the data queue to send a file by reading it in binary mode and sending fixed-size chunks over the PUSH socket. After all chunks are sent, a special end-of-file marker (`b'__END_OF_FILE__'`) signals completion. The method enforces role and queue type permissions and verifies the file exists before sending.

- **`recv_file(dir="/tmp")`**  
  Enables a server role using the data queue to receive file chunks from the PULL socket. It ensures the specified directory exists, creates a temporary file there, and writes incoming chunks until the end-of-file marker is received. The method flushes and closes the file, returning its absolute path.

---

### Integration for File Send/Receive Workflow

#### Client Side

- **Function:** `send_files_to_server(username, file_path_list, control_mq, data_mq)`  
  Sends each file in `file_path_list` via the data channel using `data_mq.send_file()`. After all files are sent, it notifies the server through the control channel with `control_mq.notify_server()`, indicating the number of files sent. It then waits for acknowledgment from the server before confirming success.

#### Server Side

- **Method:** `server_control_queue_loop(self)`  
  Continuously listens for control messages. On receiving the `"FILE"` command, it extracts client ID and file count, then calls:

- **Method:** `_process_file_cmd(self, client_id, nb_files)`  
  Checks the client’s data queue availability. For each expected file, calls `data_queue.recv_file(dir)` to receive and save the file as a temporary file. Collects all file paths, sends an acknowledgment via `send_string("ACKNOWLEDGE")`, and returns the list of file paths.

---

#### Extensibility Note

To add a new client command for server configuration requests, rename the `"FILE"` command to a clear, descriptive name, then update client and server logic accordingly. This allows sending the appropriate files or data and handling them on the server based on the new command, leveraging the existing messaging framework for organized protocol extension.
