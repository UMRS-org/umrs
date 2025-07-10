#!/bin/bash
set -e

# test data directory
DATA_DIR="${PWD}/../data"
if [ ! -d "$DATA_DIR" ]; then
  echo "Data directory does not exist: $DATA_DIR"
  exit 1
fi

DOCKER_IMAGE="docker-umrs-server"




echo "=== Adding test data to buckets ==="

# add text string to user*-bucket-2
docker run --rm -it \
  -v "$PWD:/workdir" \
  -v "$DATA_DIR:/data" \
  --network="docker_default" \
  $DOCKER_IMAGE \
  python src/client_umrs.py -c /workdir/conf_umrs_user1.json /data/texts/ user1-bucket-5

docker run --rm -it \
  -v "$PWD:/workdir" \
  -v "$DATA_DIR:/data" \
  --network="docker_default" \
  $DOCKER_IMAGE \
  python src/client_umrs.py -c /workdir/conf_umrs_user2.json /data/texts/ user2-bucket-5


# add image data from data directory to user*-bucket-1
echo "Adding image data to user1-bucket-1..."
docker run --rm -it \
  -v "$PWD:/workdir" \
  -v "$DATA_DIR:/data" \
  --network="docker_default" \
  $DOCKER_IMAGE \
  python src/client_umrs.py -c /workdir/conf_umrs_user1.json /data/img_test/ user1-bucket-3

echo "Adding image data to user2-bucket-1..."
docker run --rm -it \
  -v "$PWD:/workdir" \
  -v "$DATA_DIR:/data" \
  --network="docker_default" \
  $DOCKER_IMAGE \
  python src/client_umrs.py -c /workdir/conf_umrs_user2.json /data/img_test/ user2-bucket-3

