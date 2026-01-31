# SSH Tunnel Setup for UMRS

This document describes how to configure SSH tunnels for ZeroMQ-based file transfer between HPC environments (e.g., HUCC) and RCCS.

## Problem

HPC compute nodes typically have strict network restrictions:
- Inbound TCP connections are blocked
- Outbound TCP connections are blocked  
- Only SSH access through a gateway is allowed

This prevents direct ZeroMQ socket connections between UMRS clients (on HUCC) and servers (on RCCS).

## Solution: SSH Tunnel Mode

UMRS supports `tunnel_mode` which works with SSH tunnels:

1. **Both sides use localhost (127.0.0.1)**
   - Server binds to `127.0.0.1:5555` (control) and `127.0.0.1:5556` (data)
   - Client connects to `127.0.0.1:5555` and `127.0.0.1:5556`

2. **SSH tunnel bridges the localhost endpoints**
   - Forwards local port → remote localhost port
   - All traffic flows through the SSH connection

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            HUCC (Client Side)                            │
│                                                                          │
│  ┌─────────────────┐          ┌─────────────────┐                        │
│  │  UMRS Client    │          │   SSH Client    │                        │
│  │                 │          │                 │                        │
│  │ connect to      │──────────│ -L 5555:...     │                        │
│  │ 127.0.0.1:5555  │          │ -L 5556:...     │                        │
│  │ 127.0.0.1:5556  │          │                 │                        │
│  └─────────────────┘          └────────┬────────┘                        │
│                                        │                                 │
└────────────────────────────────────────┼─────────────────────────────────┘
                                         │ SSH Tunnel
                                         │ (encrypted)
┌────────────────────────────────────────┼─────────────────────────────────┐
│                            RCCS (Server Side)                            │
│                                        │                                 │
│                               ┌────────▼────────┐                        │
│                               │   SSH Server    │                        │
│                               │                 │                        │
│  ┌─────────────────┐          │ forwards to     │                        │
│  │  UMRS Server    │◄─────────│ localhost:5555  │                        │
│  │                 │          │ localhost:5556  │                        │
│  │ bind to         │          │                 │                        │
│  │ 127.0.0.1:5555  │          └─────────────────┘                        │
│  │ 127.0.0.1:5556  │                                                     │
│  └─────────────────┘                                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UMRS_TUNNEL_MODE` | `false` | Enable tunnel mode (`true`, `1`, `yes`) |
| `UMRS_CONTROL_PORT` | `5555` | Control queue port |
| `UMRS_DATA_PORT` | `5556` | Data queue port |

### Server Side (RCCS)

```bash
# Set environment variables
export UMRS_TUNNEL_MODE=true
export UMRS_CONTROL_PORT=5555
export UMRS_DATA_PORT=5556

# Start UMRS server
python -m umrs.server_umrs
```

Or in Python:
```python
from umrs.MessageQueue import MessageQueue

# Server control queue with tunnel mode
server_mq = MessageQueue(
    role='server',
    queue_type='control',
    tunnel_mode=True,  # Binds to 127.0.0.1
)
```

### Client Side (HUCC)

**Step 1: Establish SSH tunnel**

```bash
# Basic tunnel (run in background)
ssh -L 5555:localhost:5555 -L 5556:localhost:5556 user@rccs-gateway -N &

# With keep-alive and compression
ssh -L 5555:localhost:5555 -L 5556:localhost:5556 \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -C \
    user@rccs-gateway -N &
```

**Step 2: Start UMRS client**

```bash
# Set environment variables
export UMRS_TUNNEL_MODE=true
export UMRS_CONTROL_PORT=5555
export UMRS_DATA_PORT=5556

# Run client
python -m umrs.client_umrs
```

Or in Python:
```python
from umrs.MessageQueue import MessageQueue

# Client connects through tunnel
client_mq = MessageQueue(
    role='client',
    queue_type='control',
    server_ip='rccs-server.example.com',  # Ignored in tunnel mode
    tunnel_mode=True,  # Connects to 127.0.0.1
)

# Get data queue (also uses tunnel)
data_mq = client_mq.get_data_queue(client_id='my-client')
# data_mq.tunnel_mode is automatically True (inherited)
```

## SSH Configuration Tips

### ~/.ssh/config

```ssh-config
Host rccs-tunnel
    HostName rccs-gateway.riken.jp
    User your-username
    LocalForward 5555 localhost:5555
    LocalForward 5556 localhost:5556
    ServerAliveInterval 60
    ServerAliveCountMax 3
    Compression yes
    # Keep tunnel alive
    ExitOnForwardFailure yes
```

Then simply:
```bash
ssh rccs-tunnel -N &
```

### Persistent Tunnel with autossh

```bash
# Install autossh if not available
# apt-get install autossh  # Debian/Ubuntu
# yum install autossh      # RHEL/CentOS

# Start persistent tunnel
autossh -M 0 -f -N \
    -L 5555:localhost:5555 \
    -L 5556:localhost:5556 \
    -o "ServerAliveInterval 60" \
    -o "ServerAliveCountMax 3" \
    user@rccs-gateway
```

### systemd Service for Tunnel

Create `/etc/systemd/system/umrs-tunnel.service`:

```ini
[Unit]
Description=UMRS SSH Tunnel to RCCS
After=network.target

[Service]
Type=simple
User=your-username
ExecStart=/usr/bin/ssh -N \
    -L 5555:localhost:5555 \
    -L 5556:localhost:5556 \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -i /home/your-username/.ssh/id_rsa \
    your-username@rccs-gateway
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable umrs-tunnel
sudo systemctl start umrs-tunnel
```

## Slurm Integration

For HPC jobs, establish the tunnel before the job runs:

```bash
#!/bin/bash
#SBATCH --job-name=umrs-client
#SBATCH --time=01:00:00

# Establish SSH tunnel
ssh -L 5555:localhost:5555 -L 5556:localhost:5556 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i ~/.ssh/rccs_key \
    user@rccs-gateway -N &
TUNNEL_PID=$!

# Wait for tunnel to establish
sleep 3

# Set tunnel mode
export UMRS_TUNNEL_MODE=true

# Run UMRS client
python /path/to/my_umrs_job.py

# Cleanup
kill $TUNNEL_PID 2>/dev/null
```

## Troubleshooting

### Tunnel Not Working

1. **Check tunnel is running:**
   ```bash
   ps aux | grep "ssh.*-L.*5555"
   ```

2. **Test port forwarding:**
   ```bash
   nc -zv localhost 5555
   ```

3. **Check server is listening:**
   On RCCS:
   ```bash
   ss -tlnp | grep 5555
   ```

### Connection Refused

- Ensure server started with `tunnel_mode=True`
- Ensure tunnel established before client connects
- Check firewall on RCCS allows localhost connections

### Tunnel Drops

- Use `autossh` for automatic reconnection
- Increase `ServerAliveInterval` and `ServerAliveCountMax`
- Check gateway timeout settings

### Multiple Clients

Each client needs its own SSH tunnel. The data port (5556) is shared, but the control queue handles client multiplexing via ZeroMQ ROUTER/DEALER pattern.

## Security Considerations

- SSH tunnel provides encryption for all ZeroMQ traffic
- Use SSH keys instead of passwords
- Consider SSH agent forwarding for nested connections
- Restrict SSH access to specific users/hosts

## Related Documentation

- [UMRS Overview](umrs_overview.md)
- [ZMQ API Contract](zmq_api_contract.md)
- [HPC Deployment Guide](../../.github/instructions/hpc-deployment.instructions.md)
