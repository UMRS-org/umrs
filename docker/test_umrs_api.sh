#!/usr/bin/env bash
# Use the Bash interpreter found in the environment
set -euo pipefail
# -e: Exit immediately if a command exits with a non-zero status
# -u: Treat unset variables as an error and exit immediately
# -o pipefail: Return the exit status of the last command in the pipeline that failed

retry() {
  # retry MAX_ATTEMPTS SLEEP_SECS <command...>
  # Runs a command until it succeeds or the number of attempts is exhausted.
  local max_attempts="$1"; shift
  # Number of seconds to sleep between attempts
  local sleep_secs="$1"; shift
  # Attempt counter starts at 1
  local attempt=1
  while true; do
    # Execute the command provided as arguments; if it succeeds, return 0
    if "$@"; then
      return 0
    fi
    # If we've reached the maximum number of attempts, print an error and fail
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Command failed after $attempt attempts: $*" >&2
      return 1
    fi
    # Otherwise, increment attempt counter and sleep before retrying
    attempt=$((attempt+1))
    sleep "$sleep_secs"
  done
}

# Simple integration test for ZeroMQ API using client_umrs.py
# Requires server to be reachable; use $UMRS_SERVER_HOST or $UMRS_SERVER_IP
# Defaults to docker-compose service name "umrs-server"

# Prefer UMRS_SERVER_HOST (used by MessageQueue internally), fallback to UMRS_SERVER_IP, else default
export UMRS_SERVER_HOST="${UMRS_SERVER_HOST:-${UMRS_SERVER_IP:-umrs-server}}"  # Resolve the host the server is reachable at
export UMRS_CLIENT_ID="${UMRS_CLIENT_ID:-user1}"  # Client identity propagated to the server for scoping status

CLIENT_CONF=/root/user_files/user1.json  # Path to client configuration for user1

# Wait for server readiness on ZeroMQ via client ping (retry until ready)
echo "Waiting for UMRS server control plane to be ready at ${UMRS_SERVER_HOST}..."
retry 60 2 python /app/src/client_umrs.py ping  # Keep pinging until the control plane replies

# Ask server to reload configuration (hot load) before tests; use -c to push full user1.json so server replaces user1_init.json and updates mappings
retry 5 2 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text reload  # Hot-reload server config from the provided client config

# Status: reload — fetches the server-captured summary of the latest configuration reload
# (effective mappings, buckets, models), scoped to UMRS_CLIENT_ID
retry 5 2 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text status reload  # Show last reload summary for this client

# List buckets for user1 (text/image bucket aliases) # TODO this should show only the buckets of user1 (not all buckets)
# TODO: STD of listing buckets etc. should be returned to the client side as well.
retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text list_buckets  # Verify bucket visibility for user1

# Skipping inline text-sending via CLI; not supported anymore. Use upload_file with actual files or upload_dir.

# Upload a text file object to the 'text file' bucket (bucket-5)
if [ -f /root/data/texts/beast_in_the_jungle.txt ]; then
  # Attach metadata (-k) and upload a sample text into the designated bucket for user1
  retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:beast upload_file /root/data/texts/beast_in_the_jungle.txt bucket-5
fi

# Upload a sample image to an image bucket for user1 if present
if [ -f /root/data/img_test/cat.jpeg ]; then
  # Upload a sample image to the first image bucket
  retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:test upload_file /root/data/img_test/cat.jpeg bucket-1
  # Also exercise the other image bucket (bucket-3) if desired
  python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:test2 -k author:demo -k x_resolution:256 -k y_resolution:512 upload_file /root/data/img_test/cat.jpeg bucket-3 || true  # Don't fail the whole script if this optional upload fails
fi

# Also exercise upload_dir for images if directory is present (aligns with demo.md)
if [ -d /root/data/img ]; then
  # Upload all images in the directory with demo metadata to bucket-1
  retry 10 3 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text -k keywords:demo upload_dir /root/data/img bucket-1
fi

# Trigger server-side embedding operations (no config required)
retry 10 3 python /app/src/client_umrs.py --output text check  # Ask server to check which items need updates/recompute
# Status: check — retrieves the last check-operation summary (items requiring update/recompute)
# for this client (UMRS_CLIENT_ID)
retry 5 2 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text status check  # Show last check summary

retry 10 3 python /app/src/client_umrs.py --output text update  # Perform incremental updates (index new or changed items)
# Status: update — retrieves the last update-operation summary (what was indexed/updated)
# for this client (UMRS_CLIENT_ID)
retry 5 2 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text status update  # Show last update summary

retry 10 3 python /app/src/client_umrs.py --output text recompute  # Recompute embeddings as needed or requested
# Status: recompute — retrieves the last recompute-operation summary (embeddings recomputed)
# for this client (UMRS_CLIENT_ID)
retry 5 2 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text status recompute  # Show last recompute summary
# Status: all — consolidated view of all stored summaries (config + maintenance)
# for this client (UMRS_CLIENT_ID)
retry 5 2 python /app/src/client_umrs.py -c "$CLIENT_CONF" --output text status all  # Show consolidated status for this client

echo "UMRS API test completed."  # Final message indicating the integration run finished
