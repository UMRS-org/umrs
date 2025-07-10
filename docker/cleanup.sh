#!/bin/bash
set -e

echo "=== Docker Compose Cleanup Script ==="
echo "This will remove all containers, images, volumes, and networks for this project"
echo

# Ask for confirmation
read -p "Are you sure you want to proceed? This will delete all data! (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo "Starting cleanup..."

# Stop and remove all containers, networks, and volumes
echo "Stopping and removing containers, networks, and volumes..."
docker compose down -v --remove-orphans || echo "No containers to stop"

# Remove all containers that might still be using the images
echo "Removing any remaining containers using project images..."
CONTAINERS=$(docker ps -a --filter "ancestor=umrs-server" --filter "ancestor=docker-umrs-client-tester" -q 2>/dev/null || true)
if [ ! -z "$CONTAINERS" ]; then
    echo "Found containers to remove: $CONTAINERS"
    echo $CONTAINERS | xargs docker rm -f || echo "Some containers couldn't be removed"
else
    echo "No additional containers found"
fi

# Remove all images built by this compose file
echo "Removing images..."
docker compose down --rmi all || echo "No images to remove"

# Force remove project images if they still exist
echo "Force removing project images..."
docker rmi umrs-server:latest docker-umrs-client-tester:latest 2>/dev/null || echo "Some images already removed or don't exist"

# Remove any dangling volumes that might be left
echo "Removing any remaining volumes..."
VOLUMES=$(docker volume ls -q | grep -E "(docker_|umrs)" 2>/dev/null || true)
if [ ! -z "$VOLUMES" ]; then
    echo "Found volumes to remove: $VOLUMES"
    echo $VOLUMES | xargs docker volume rm || echo "Some volumes couldn't be removed"
else
    echo "No additional volumes found"
fi

# Remove any dangling images
echo "Removing dangling images..."
DANGLING_IMAGES=$(docker images -f "dangling=true" -q)
if [ ! -z "$DANGLING_IMAGES" ]; then
    echo "Removing dangling images..."
    echo $DANGLING_IMAGES | xargs docker rmi || echo "Some images couldn't be removed"
else
    echo "No dangling images found"
fi

# Optional: Remove all unused containers, networks, images
echo
read -p "Do you want to run 'docker system prune' to clean up all unused Docker resources? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running docker system prune..."
    docker system prune -f
fi

echo
echo "=== Cleanup completed! ==="
echo "You can now run 'docker compose up' to start fresh."
