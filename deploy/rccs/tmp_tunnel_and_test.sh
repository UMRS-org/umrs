#!/bin/bash
# Kill stale SSH tunnels to rccs-cloud
pkill -f "ssh.*-L.*5555.*rccs-cloud" 2>/dev/null
pkill -f "ssh.*-fN.*rccs-cloud" 2>/dev/null
sleep 2

# Start fresh SSH tunnel
ssh -fN \
  -L 5555:localhost:5555 \
  -L 5556:localhost:5556 \
  -L 9000:localhost:9000 \
  -L 8081:localhost:8081 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  rccs-cloud

if [ $? -eq 0 ]; then
  echo "TUNNEL_STARTED"
else
  echo "TUNNEL_FAILED"
  exit 1
fi

# Quick test
sleep 2
nc -z localhost 5555 && echo "LOCAL_ZMQ:OK" || echo "LOCAL_ZMQ:FAIL"
nc -z localhost 9000 && echo "LOCAL_MINIO:OK" || echo "LOCAL_MINIO:FAIL"
echo "SCRIPT_DONE"
