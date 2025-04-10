# Infinia Internal User Guide

This guide provides instructions on creating an S3 service on Infinia and performing simple tests using s3cmd and warp. Your user is registered as a sub-tenant administrator. Sub-tenant administrators are primarily granted the authority to manage data services and S3 access keys.

- [Infinia Internal User Guide](#infinia-internal-user-guide)
  - [Version, Revision, LUD](#version-revision-lud)
  - [Intended Audience for This Guide](#intended-audience-for-this-guide)
  - [Required Information](#required-information)
  - [Log In to the VM](#log-in-to-the-vm)
  - [Log In to Infninia by redcli](#log-in-to-infninia-by-redcli)
  - [Create an S3 Service and Credentials](#create-an-s3-service-and-credentials)
  - [Test s3cmd](#test-s3cmd)
  - [s3cmd configuration file ~/.s3cfg](#s3cmd-configuration-file-s3cfg)
  - [Run warp benchmark](#run-warp-benchmark)
  - [Access UI](#access-ui)
  - [Access S3 and UI via Reverse Proxy](#access-s3-and-ui-via-reverse-proxy)


## Version, Revision, LUD

|Item               |Value         |
|-------------------|--------------|
|Infinia Version    |1.3.39        |
|Document Revision  |0.1.4         |
|Latest Update Date |2025-01-22    |

## Intended Audience for This Guide

This guide is intended for members of DDN Japan. It is not designed to be shared with customers. (That said, if the content of this document evolves to a sufficient level, there is a possibility it may serve that purpose in the future. This will likely be determined based on future directions and progress.)

## Required Information

- User account and password for the VM
- IP address of the VM
- Infinia user account and password
- Tenant name with access rights
- Sub-tenant name

## Log In to the VM

Required packages, redcli, s3cmd, and warp are installed on the VM. Start by logging in to the VM.

``` shell
# SSH login
ssh [user]@[ip address]

# Example
ssh ktanaka@172.16.47.177
```

## Log In to Infninia by redcli

You can log in to Infinia using the following command. Note that logging in doesn’t mean moving to Infinia like SSH; it simply enables you to use Infinia commands from the VM.

``` shell
# Log in by redcli
redcli user login [user] -t [tenant]

# Example
redcli user login ktanaka -t ten9
```

Please confirm that the list of tenants and subtenants is displayed using the following commands.

``` shell
redcli tenant list
redcli subtenant list
```

## Create an S3 Service and Credentials

The S3 service on Infinia is of the type “file-and-object.” To create the service, specify the type, tenant name, sub-tenant name, and your desired vhost. The vhost domain should follow the format:
[custom_vhost_name].infinia.lab.local.

``` shell
# Creatre S3 service
redcli service create [service] -T file-and-object -t [tenant] -s [subtenant] --vhost [custom_vhost_name].infinia.lab.local

# Example
redcli service create s3 -T file-and-object -t ten9 -s subten_ktanaka --vhost kts3.infinia.lab.local
```

``` shell
# Check the Scope from the Service List
redcli service list
```

``` shell
# Create credential
redcli s3 access add [user] --scope [tenant]/[subtenant]/[service] -e 1y -o yaml | tee ~/infinia_cred.yml

# Example
redcli s3 access add ktanaka --scope ten9/subten_ktanaka/s3 -e 1y -o yaml | tee ~/infinia_cred.yml
```

## Test s3cmd

``` shell
access_key=$(awk '/s3_key/ {print $2}' ~/infinia_cred.yml)
secret_key=$(awk '/s3_secret/ {print $2}' ~/infinia_cred.yml)

vhost="[custom_vhost_name].infinia.lab.local"
# Example
vhost="kts3.infinia.lab.local"

# Cteate a bucket
s3cmd --host=$vhost:8111 \
    --host-bucket=$vhost:8111 \
    --access_key=$access_key \
    --secret_key=$secret_key \
    --bucket-location=us-east-1 \
    --ssl --no-check-certificate \
    mb s3://bucket1/

# List bucket
s3cmd --host=$vhost:8111 \
    --host-bucket=$vhost:8111 \
    --access_key=$access_key \
    --secret_key=$secret_key \
    --bucket-location=us-east-1 \
    --ssl --no-check-certificate \
    ls
```

## s3cmd configuration file ~/.s3cfg

Below is a sample `~/.s3cfg` file for your reference. By configuring this file, you won’t need to specify the endpoint, access key, or secret key in command-line arguments. This simplifies operations like creating a bucket and listing buckets, allowing you to use commands such as `s3cmd mb s3://bucket1` and `s3cmd ls`.

``` text
[default]
host_base = [custom_vhost_name].infinia.lab.local:8111
host_bucket = [custom_vhost_name].infinia.lab.local:8111
bucket_location = us-east-1
use_https = True
check_ssl_certificate = False
access_key = "Check out ~/infinia_cred.yml"
secret_key = "Check out ~/infinia_cred.yml"
```

## Run warp benchmark

``` shell
# PUT
warp get --access-key=$access_key \
    --secret-key=$secret_key \
    --host $vhost:8111 \
    --tls=true \
    --obj.size 4MiB \
    --insecure \
    --autoterm \
    --concurrent 120

# GET
warp get --access-key=$access_key \
    --secret-key=$secret_key \
    --host $vhost:8111 \
    --tls=true \
    --obj.size 4MiB \
    --insecure \
    --autoterm \
    --concurrent 120
```

## Access UI

UI's URL: https://172.16.40.83

You can log in to the UI using a web browser. By default, the login screen shows “Realm” selected, but make sure to click on “Tenant” instead. Then, enter the tenant name, username, and password, and click "Login."

This document does not cover the UI in detail, but you can use it to check the status of data services.

## Access S3 and UI via Reverse Proxy

The Infinia S3 service deployed in the lab is available on Infinia’s 100GbE network. To access Infinia from outside this network, you need to go through a reverse proxy. The VM used for running tools like redcli works as this reverse proxy.

For macOS or Linux users, you can register the VM’s IP address with the vhost in the /etc/hosts file. This allows you to use the S3 service from outside the dedicated network. Here’s an example:

``` text
# Example entry in /etc/hosts 
172.16.47.177  kts3.infinia.lab.local
```

By registering the vhost in your /etc/hosts file as explained earlier, you can also access the UI through your web browser. Simply open the vhost on port 443 in your browser. For example:

```
https://kts3.infinia.lab.local
```

This allows you to access Infinia's UI and manage services or check their status directly from your browser.
