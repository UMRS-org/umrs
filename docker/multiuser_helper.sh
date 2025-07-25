#!/bin/bash

# Helper script for UMRS multi-user setup and testing
# Usage: ./multiuser_helper.sh [command]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    echo -e "${GREEN}UMRS Multi-User Helper Script${NC}"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup     - Set up the multi-user environment"
    echo "  test      - Run the multi-user test"
    echo "  clean     - Clean up containers and data"
    echo "  logs      - Show logs from all services"
    echo "  shell     - Open shell in client tester container"
    echo "  user1     - Switch to user1 in client tester"
    echo "  user2     - Switch to user2 in client tester"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup    # Set up and start all services"
    echo "  $0 test     # Run the multi-user test"
    echo "  $0 user1    # Get shell as user1"
}

setup_environment() {
    echo -e "${BLUE}Setting up UMRS multi-user environment...${NC}"

    # Build and start all services
    docker compose up -d --build

    echo -e "${GREEN}Environment setup complete!${NC}"
    echo -e "${YELLOW}Services running:${NC}"
    docker compose ps
}

run_test() {
    echo -e "${BLUE}Running multi-user test...${NC}"

    # Run the test script in the client tester container
    docker compose exec umrs-client-tester /client/test_multiuser.sh

    echo -e "${GREEN}Multi-user test completed!${NC}"
}

clean_environment() {
    echo -e "${BLUE}Cleaning up UMRS environment...${NC}"

    # Stop and remove containers
    docker compose down

    # Remove volumes (uncomment if you want to remove data)
    # docker-compose down -v

    echo -e "${GREEN}Environment cleaned up!${NC}"
}

show_logs() {
    echo -e "${BLUE}Showing logs from all services...${NC}"
    docker compose logs -f
}

open_shell() {
    echo -e "${BLUE}Opening shell in client tester container...${NC}"
    docker compose exec umrs-client-tester /bin/bash
}

switch_to_user1() {
    echo -e "${BLUE}Switching to user1 in client tester container...${NC}"
    docker compose exec umrs-client-tester su - user1
}

switch_to_user2() {
    echo -e "${BLUE}Switching to user2 in client tester container...${NC}"
    docker compose exec umrs-client-tester su - user2
}

# Main execution
case "${1:-help}" in
    setup)
        setup_environment
        ;;
    test)
        run_test
        ;;
    clean)
        clean_environment
        ;;
    logs)
        show_logs
        ;;
    shell)
        open_shell
        ;;
    user1)
        switch_to_user1
        ;;
    user2)
        switch_to_user2
        ;;
    help|*)
        show_help
        ;;
esac
