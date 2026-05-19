#!/usr/bin/env bash
# Stop UMRS services on umrs-front
set -euo pipefail
cd "$(dirname "$0")"

echo "Stopping UMRS servers..."
for pidfile in server-*.pid; do
    [[ -f "$pidfile" ]] || continue
    pid=$(cat "$pidfile")
    name="${pidfile%.pid}"
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Stopping $name (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
done

case "${1:-}" in
    --all|--clean)
        echo "Stopping infrastructure containers..."
        podman stop umrs-postgres umrs-minio 2>/dev/null || true
        if [[ "${1:-}" == "--clean" ]]; then
            echo "Removing containers and volumes..."
            podman rm -f umrs-postgres umrs-minio 2>/dev/null || true
            podman volume rm umrs_pg_data umrs_minio_data 2>/dev/null || true
        fi
        ;;
    "")
        echo "Infrastructure containers left running (use --all to stop them too)"
        ;;
esac

echo "Done."
