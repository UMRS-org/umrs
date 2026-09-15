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
        -k keywords:cat \
        -k author:demo \
        -k x_resolution:256 \
        -k y_resolution:512 \
        upload_file ../data/img_test/cat.jpeg bucket-3 &
done

wait
