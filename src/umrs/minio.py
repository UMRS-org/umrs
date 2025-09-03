import subprocess
import sys
import os
import json
import configparser
import textwrap


def minio_check_create_alias(alias, url, access_key, secret_key, insecure=True):
    """Ensure the mc alias exists. If not, create it."""
    print(f"Checking alias: {alias}")
    try:
        result = subprocess.run(["mc", "alias", "list"], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if line.strip().startswith(alias):
                print(f"Alias '{alias}' already exists.")
                return True
    except subprocess.CalledProcessError as e:
        print(f"Error running 'mc alias list': {e.stderr}")
        return False

    cmd = ["mc", "alias", "set", alias, url, access_key, secret_key]
    if insecure:
        cmd.append("--insecure")
    try:
        subprocess.run(cmd, check=True)
        print(f"Alias '{alias}' created.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating alias '{alias}': {e.stderr}")
        return False


def minio_create_user_with_policy(user, password, target, policy_name):
    """Creates a MinIO user (if not present) and attaches a policy."""
    try:
        result = subprocess.run(["mc", "admin", "user", "list", target, "--insecure"], capture_output=True, text=True, check=True)
        if user in result.stdout:
            print(f"User '{user}' already exists.")
            try:
                result = subprocess.run(["mc", "admin", "policy", "list", target, "--insecure"], capture_output=True, text=True, check=True)
                policies = [line.strip().split()[0] for line in result.stdout.splitlines() if line.strip()]
                if policy_name in policies:
                    print(f"Policy '{policy_name}' already exists on '{target}' for '{user}'.")
                    return
            except subprocess.CalledProcessError as e:
                sys.exit(f"Error listing policies: {e.stderr}")
        else:
            subprocess.run(["mc", "admin", "user", "add", target, user, password, "--insecure"], check=True)
            print(f"User '{user}' created.")
    except subprocess.CalledProcessError as e:
        print(f"Error checking or creating user: {e.stderr}")
        return

    try:
        subprocess.run(["mc", "admin", "policy", "attach", target, policy_name, "--user", user, "--insecure"], check=True)
        print(f"Policy '{policy_name}' attached to user '{user}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error attaching policy to user: {e.stderr}")


def minio_check_create_list_only_policy(target):
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetBucketLocation"],
                "Resource": ["arn:aws:s3:::*"]
            }
        ]
    }
    policy_json = json.dumps(policy, indent=2)
    policy_name = "listonly"
    policy_file = "/tmp/policy.json"

    try:
        result = subprocess.run(["mc", "admin", "policy", "list", target, "--insecure"], capture_output=True, text=True, check=True)
        policies = [line.strip().split()[0] for line in result.stdout.splitlines() if line.strip()]
        if policy_name in policies:
            print(f"Policy '{policy_name}' already exists on '{target}'.")
            return policy_name
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error listing policies: {e.stderr}")

    with open(policy_file, 'w') as f:
        f.write(policy_json)

    try:
        subprocess.run(["mc", "admin", "policy", "create", target, policy_name, policy_file, "--insecure"], check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error creating policy: {e}")

    return policy_name


def minio_grant_user_access(user, bucket, target):
    policy_name = f"allow-{bucket}"
    try:
        results = subprocess.run(["mc", "admin", "policy", "entities", target, "--user", user, "--insecure"], capture_output=True, text=True, check=True)
        if policy_name in results.stdout:
            print(f"==> Policy '{policy_name}' already attached to user '{user}'.")
            return
    except subprocess.CalledProcessError as e:
        print(f"Error checking policy: {e.stderr or e}")
        return

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:DeleteObject", "s3:ListAllMyBuckets", "s3:ListBucket", "s3:PutObject", "s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket}/*"]
            }
        ]
    }

    policy_json = json.dumps(policy, indent=2)
    policy_file = "/tmp/policy.json"
    with open(policy_file, 'w') as f:
        f.write(policy_json)

    try:
        subprocess.run(["mc", "admin", "policy", "create", target, policy_name, policy_file, "--insecure"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating policy: {e}")
        return

    try:
        print(f"Attaching allow policy to bucket '{bucket}'...")
        subprocess.run(["mc", "admin", "policy", "attach", target, policy_name, "--user", user, "--insecure"], check=True)
        print(f"==> Policy '{policy_name}' attached to user '{user}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error attaching policy to user: {e}")


def minio_s3_create_user(username, s3_connection, user_s3_configuration_file, admin_s3_configuration_file):
    from umrs.s3 import s3_strong_password

    config_path = os.path.expanduser(admin_s3_configuration_file)
    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read(config_path)
    alias = config.get("default", "alias", fallback="myminio")

    if not os.path.exists(os.path.expanduser(user_s3_configuration_file)):
        s3_host = config.get("default", "host_base")
        use_https = config.getboolean("default", "use_https", fallback=False)
        verify_ssl = config.getboolean("default", "check_ssl_certificate", fallback=True)
        bucket_location = config.get("default", "bucket_location", fallback="ap-northeast-1")
        protocol = "https" if use_https else "http"
        endpoint_url = f"{protocol}://{s3_host}"

        user_passwd = s3_strong_password()
        user_cfg = f"""[default]
host_base = {s3_host}
host_bucket = {s3_host}
bucket_location = {bucket_location}
use_https = {use_https}
check_ssl_certificate = {verify_ssl}
access_key = {username}
secret_key = {user_passwd}
"""

        with open(os.path.expanduser(user_s3_configuration_file), "w") as file:
            file.write(textwrap.dedent(user_cfg))

        print(f"==> Configuration file written to: {user_s3_configuration_file}")
    else:
        print(f"==> S3 configuration file '{user_s3_configuration_file}' already exists for user '{username}' and alias '{alias}'.")
        print(f"==> Delete this file manually if you want to reconfigure the user.")

        config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        config.read(os.path.expanduser(user_s3_configuration_file))
        s3_host = config.get("default", "host_base")
        use_https = config.getboolean("default", "use_https", fallback=False)
        verify_ssl = config.getboolean("default", "check_ssl_certificate", fallback=True)
        access_key = config.get("default", "access_key")
        secret_key = config.get("default", "secret_key")
        bucket_location = config.get("default", "bucket_location", fallback="ap-northeast-1")
        alias = config.get("default", "alias", fallback="myminio")

        if access_key != username:
            sys.exit(
                f"Critical error. The access key in '{user_s3_configuration_file}' should be '{username}', but is '{access_key}'.\n"
                "Cannot continue. Either update the JSON user config file or change/remove the S3 user config file."
            )

        protocol = "https" if use_https else "http"
        endpoint_url = f"{protocol}://{s3_host}"
        minio_check_create_alias(alias, endpoint_url, access_key, secret_key, insecure=True)
        list_only_policy = minio_check_create_list_only_policy(alias)
        minio_create_user_with_policy(access_key, secret_key, alias, list_only_policy)

    return alias
