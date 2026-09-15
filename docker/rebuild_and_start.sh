#!/bin/bash

set -e

# Colors for output
green='\033[0;32m'
yellow='\033[1;33m'
red='\033[0;31m'
nc='\033[0m'

echo -e "${yellow}Stopping and removing containers, networks, and volumes...${nc}"
docker compose down -v --remove-orphans

echo -e "${yellow}Erasing all unused Docker volumes...${nc}"
docker volume prune -f

echo -e "${yellow}Removing dangling images...${nc}"
docker image prune -f

echo -e "${yellow}Building images...${nc}"
docker compose build --no-cache

echo -e "${green}Starting containers...${nc}"
docker compose up
