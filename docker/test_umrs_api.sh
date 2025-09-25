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

# Simple integration test for ZeroMQ API using client_umrs.py
# Requires server to be reachable; use $UMRS_SERVER_HOST or $UMRS_SERVER_IP
# Defaults to docker-compose service name "umrs-server"

# Prefer UMRS_SERVER_HOST (used by MessageQueue internally), fallback to UMRS_SERVER_IP, else default
export UMRS_SERVER_HOST="${UMRS_SERVER_HOST:-${UMRS_SERVER_IP:-umrs-server}}"

CLIENT_CONF=/app/user_files/user1.json

# Wait for server readiness on ZeroMQ via client ping (retry until ready)
echo "Waiting for UMRS server control plane to be ready at ${UMRS_SERVER_HOST}..."
retry 60 2 python /app/src/client_umrs.py ping

# Ask server to reload configuration (hot load) before tests; use -c to push full user1.json so server replaces user1_init.json and updates mappings
retry 5 2 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text reload

# List buckets for user1 (text/image bucket aliases) # TODO this should show only the buckets of user1 (not all buckets)
# TODO: STD of listing buckets etc. should be returned to the client side as well.
retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text list_buckets

# Upload a sample text to a text bucket for user1 (xp2/text => bucket-2)
retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:test upload_text "hello from test" bucket-2

# Upload a small batch of inline texts to the same text bucket (upload_text_batch)
retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:batch upload_text_batch bucket-2 --texts "alpha one" "beta two" "gamma three"

# Upload a text file object to the 'text file' bucket (bucket-5)
if [ -f /app/data/texts/beast_in_the_jungle.txt ]; then
  retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:beast upload_textfile /app/data/texts/beast_in_the_jungle.txt bucket-5
fi

# Upload a sample image to an image bucket for user1 if present
if [ -f /app/data/img_test/cat.jpeg ]; then
  retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:test upload_image /app/data/img_test/cat.jpeg bucket-1
  # Also exercise the other image bucket (bucket-3) if desired
  python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:test2 -k author:demo -k x_resolution:256 -k y_resolution:512 upload_image /app/data/img_test/cat.jpeg bucket-3 || true
fi

# Trigger server-side embedding operations (no config required)
retry 10 3 python /app/src/client_umrs.py --output text check
retry 10 3 python /app/src/client_umrs.py --output text update
retry 10 3 python /app/src/client_umrs.py --output text recompute

echo "UMRS API test completed."
