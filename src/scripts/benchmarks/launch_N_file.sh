#!/bin/bash

N=$1

if [ -z "$N" ]; then
    echo "Usage: $0 <N>"
    exit 1
fi

for i in $(seq 1 "$N"); do
    ./client_umrs.py \
        -c ~/.config/umrs/n_users/users/user1_${i}.json \
        --output text \
        upload_file ../data/img_test/cat.jpeg bucket-3 &
done

wait
