#!/usr/bin/sh 

access_key=$(awk '/s3_key/ {print $2}' ~/infinia_cred.yml)
secret_key=$(awk '/s3_secret/ {print $2}' ~/infinia_cred.yml)


vhost="ejeannot-s3-vm.infinia.lab.local"
port="8111"


echo "Put"
# PUT
warp put --access-key=$access_key \
    --secret-key=$secret_key \
    --host $vhost:$port \
    --tls=true \
    --obj.size 4MiB \
    --insecure \
    --autoterm \
    --concurrent 120

echo "Get"
# GET
warp get --access-key=$access_key \
    --secret-key=$secret_key \
    --host $vhost:$port \
    --tls=true \
    --obj.size 4MiB \
    --insecure \
    --autoterm \
    --concurrent 120
