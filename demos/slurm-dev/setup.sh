#!/bin/bash
# =============================================================================
# Setup Script for Slurm Development Environment
# =============================================================================
#
# Usage:
#   ./setup.sh [start|stop|status|test|clean]
#
# Commands:
#   start   - Start the Slurm cluster
#   stop    - Stop the Slurm cluster
#   status  - Show cluster status
#   test    - Submit a test job
#   clean   - Remove all containers and volumes
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yaml"

# Load environment if exists
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

cmd_start() {
    log_info "Starting Slurm development cluster..."
    
    # Check for .env file
    if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
        log_warn ".env file not found. Using defaults."
        log_info "Copy env.template to .env and customize if needed."
    fi
    
    # Build and start containers
    docker compose -f "$COMPOSE_FILE" up -d --build
    
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Check status
    cmd_status
    
    echo ""
    log_info "Slurm cluster is ready!"
    echo ""
    echo "Quick commands:"
    echo "  docker exec slurmctld sinfo          # Show nodes"
    echo "  docker exec slurmctld squeue         # Show queue"
    echo "  docker exec slurmctld sbatch /etc/slurm/mini-rivault-job.sh  # Submit job"
    echo ""
}

cmd_stop() {
    log_info "Stopping Slurm development cluster..."
    docker compose -f "$COMPOSE_FILE" down
    log_info "Cluster stopped."
}

cmd_status() {
    log_info "Cluster status:"
    echo ""
    
    # Container status
    echo "Containers:"
    docker compose -f "$COMPOSE_FILE" ps
    echo ""
    
    # Slurm status
    if docker exec slurmctld sinfo -N 2>/dev/null; then
        echo ""
        echo "Job queue:"
        docker exec slurmctld squeue 2>/dev/null || true
    else
        log_warn "slurmctld not ready yet"
    fi
}

cmd_test() {
    log_info "Submitting test job..."
    
    # Set environment for the job
    export MINI_RIVAULT_MODE=internal  # Use mock server
    
    # Submit job
    JOB_ID=$(docker exec -e MINI_RIVAULT_MODE=internal slurmctld \
        sbatch --parsable /etc/slurm/mini-rivault-job.sh 2>/dev/null)
    
    if [[ -n "$JOB_ID" ]]; then
        log_info "Job submitted: $JOB_ID"
        
        # Wait a bit and check status
        sleep 3
        echo ""
        echo "Job queue:"
        docker exec slurmctld squeue
        
        echo ""
        log_info "To check job info:"
        echo "  docker exec slurmctld cat /var/run/rivault/rivault-job-${JOB_ID}.json"
    else
        log_error "Failed to submit job"
        exit 1
    fi
}

cmd_clean() {
    log_warn "This will remove all containers and volumes!"
    read -p "Are you sure? [y/N] " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleaning up..."
        docker compose -f "$COMPOSE_FILE" down -v --remove-orphans
        log_info "Cleanup complete."
    else
        log_info "Cancelled."
    fi
}

cmd_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start   - Start the Slurm cluster"
    echo "  stop    - Stop the Slurm cluster"
    echo "  status  - Show cluster status"
    echo "  test    - Submit a test job"
    echo "  clean   - Remove all containers and volumes"
    echo "  help    - Show this help"
    echo ""
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

case "${1:-help}" in
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    test)   cmd_test ;;
    clean)  cmd_clean ;;
    help)   cmd_help ;;
    *)
        log_error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
