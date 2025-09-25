#!/usr/bin/env bash
set -euo pipefail

retry() {
  local max_attempts="$1"; shift
  local sleep_secs="$1"; shift
  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Command failed after $attempt attempts: $*" >&2
      return 1
    fi
    attempt=$((attempt+1))
    sleep "$sleep_secs"
  done
}

# In legacy mode, we run the UMRS server and client in the same container.
# The server exposes a ZeroMQ control plane; the client communicates directly over it.

export UMRS_SERVER_HOST=${UMRS_SERVER_HOST:-127.0.0.1}

echo "Starting UMRS server (legacy, in-container)..."
python /app/src/server_umrs.py -c /root/conf_umrs.json &
SERVER_PID=$!

cleanup() {
  echo "Stopping UMRS server (pid=${SERVER_PID})..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

CLIENT_CONF=/root/user_files/user1.json

echo "Waiting for UMRS server control plane to be ready at ${UMRS_SERVER_HOST}..."
retry 60 2 python /app/src/client_umrs.py ping

# Hot-reload configuration by pushing user1.json to the server and issuing RELOAD
retry 5 2 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text reload

# List buckets for user1 (text/image bucket aliases) # TODO this should show only the buckets of user1 (not all buckets)
# TODO: STD of listing buckets etc. should be returned to the client side as well.
retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text list_buckets

# Upload a sample text to a text bucket for user1 (xp2/text => bucket-2)
retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:test upload_text "hello from test" bucket-2

# Upload a small batch of inline texts to the same text bucket (upload_text_batch)
retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:batch upload_text_batch bucket-2 --texts "alpha one" "beta two" "gamma three"

# Upload a text file object to the 'text file' bucket (bucket-5)
if [ -f /root/data/texts/beast_in_the_jungle.txt ]; then
  retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:beast upload_textfile /root/data/texts/beast_in_the_jungle.txt bucket-5
fi

# Upload a sample image to an image bucket for user1 if present
if [ -f /root/data/img_test/cat.jpeg ]; then
  retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:test upload_image /root/data/img_test/cat.jpeg bucket-1
  # Also exercise the other image bucket (bucket-3) if desired
  python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:test2 -k author:demo -k x_resolution:256 -k y_resolution:512 upload_image /root/data/img_test/cat.jpeg bucket-3 || true
fi

# Trigger server-side embedding operations (no config required)
retry 10 3 python /app/src/client_umrs.py --output text check
retry 10 3 python /app/src/client_umrs.py --output text update
retry 10 3 python /app/src/client_umrs.py --output text recompute

echo "UMRS legacy test completed. Keeping server alive for a short grace period..."
sleep 30
