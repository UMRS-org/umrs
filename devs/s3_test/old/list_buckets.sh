#!/usr/bin/sh 

access_key=$(awk '/s3_key/ {print $2}' ~/infinia_cred.yml)
secret_key=$(awk '/s3_secret/ {print $2}' ~/infinia_cred.yml)


vhost="ejeannot-s3-vm.infinia.lab.local"
port="8111"


# List bucket
s3cmd --host=$vhost:$port \
    --host-bucket=$vhost:$port \
    --access_key=$access_key \
    --secret_key=$secret_key \
    --bucket-location=ap-northeast-1 \
    --ssl --no-check-certificate \
    ls
