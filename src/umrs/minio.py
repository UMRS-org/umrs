import subprocess
import sys, os
import json
import secrets
import string
import configparser
import textwrap
import subprocess


def minio_check_create_alias(alias, url, access_key, secret_key, insecure=True):
    """
    Ensure the mc alias exists. If not, create it.

    Args:
        alias (str): Alias name (e.g., 'myminio')
        url (str): Endpoint URL (e.g., 'https://127.0.0.1:9000')
        access_key (str): MinIO access key
        secret_key (str): MinIO secret key
        insecure (bool): Set True to skip TLS verification

    Returns:
        bool: True if alias exists or was created, False on error
    """

    print(f"Checking alias: {alias}")

    # Check if alias exists
    try:
        result = subprocess.run(
            ["mc", "alias", "list"],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.splitlines():
            if line.strip().startswith(alias):
                print(f"Alias '{alias}' already exists.")
                return True
    except subprocess.CalledProcessError as e:
        print(f"Error running 'mc alias list': {e.stderr}")
        return False

    # If not found, create alias
    cmd = ["mc", "alias", "set", alias, url, access_key, secret_key]
    if insecure:
        cmd.append('--insecure')
    try:
        subprocess.run(cmd, check=True)
        print(f"Alias '{alias}' created.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating alias '{alias}': {e.stderr}")
        return False



def minio_create_user_with_policy(user, password, target, policy_name):
    """
    Creates a MinIO user (if not present) and attaches a policy.

    Args:
        user (str): Username (access key)
        password (str): User's password (secret key)
        target (str): mc alias for MinIO server
        policy_name (str): Policy to attach (must already exist)
    """

    # First, check if the user already exists
    try:
        result = subprocess.run(
            ["mc", "admin", "user", "list", target, "--insecure"],
            capture_output=True, text=True, check=True
        )
        if user in result.stdout:
            print(f"User '{user}' already exists.")

            # Check if the listonly  policy already exists for this user
            try:
                result = subprocess.run(
                    ["mc", "admin", "policy", "list", target, "--insecure"],
                    capture_output=True, text=True, check=True
                )
                policies = [line.strip().split()[0] for line in result.stdout.splitlines() if line.strip()]
                if policy_name in policies:
                    print(f"Policy '{policy_name}' already exists on '{target}' for '{user}'.")
                    return # user and policy exist
            except subprocess.CalledProcessError as e:
                sys.exit(f"Error listing policies: {e.stderr}")
        else:
            # Create the user
            subprocess.run(
                ["mc", "admin", "user", "add", target, user, password, "--insecure"],
                check=True
            )
            print(f"User '{user}' created.")
    except subprocess.CalledProcessError as e:
        print(f"Error checking or creating user: {e.stderr}")
        return

    # Attach the policy
    try:
        subprocess.run(
            ["mc", "admin", "policy", "attach", target, policy_name, "--user", user, "--insecure"],
            check=True
        )
        print(f"Policy '{policy_name}' attached to user '{user}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error attaching policy to user: {e.stderr}")


def minio_check_create_list_only_policy(target):
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetBucketLocation"
                ],
                "Resource": [
                    "arn:aws:s3:::*"
                ]
            }
        ]
    }

    policy_json = json.dumps(policy, indent=2)
    policy_name = f"listonly"
    policy_file = "/tmp/policy.json"



    # Check if the policy already exists
    try:
        result = subprocess.run(
            ["mc", "admin", "policy", "list", target, "--insecure"],
            capture_output=True, text=True, check=True
        )
        policies = [line.strip().split()[0] for line in result.stdout.splitlines() if line.strip()]
        if policy_name in policies:
            print(f"Policy '{policy_name}' already exists on '{target}'.")
            return policy_name
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error listing policies: {e.stderr}")

    # Save policy JSON to file
    with open(policy_file, 'w') as f:
        f.write(policy_json)

    # Create policy
    try:
        subprocess.run(
            ["mc", "admin", "policy", "create", target, policy_name, policy_file, "--insecure"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error creating policy: {e}")

    return policy_name


def minio_grant_user_access(user, bucket, target):
    """
    Create and attach a deny-write policy for the specified bucket to a MinIO user.

    Args:
        user (str): MinIO username (access key).
        bucket (str): Bucket name to restrict.
    """
    policy_name = f"allow-{bucket}"
    #  mc admin policy entities myminio --user user1 --insecure
    try:
        results = subprocess.run(
            ["mc", "admin", "policy", "entities", target, "--user", user, "--insecure"],
            capture_output=True, text=True, check=True
        )

        # print(f"results: {results.stdout}\n{policy_name}")
        if policy_name in results.stdout:
            print(f"==> Policy '{policy_name}' allready attached to user '{user}'.")
            return
    except subprocess.CalledProcessError as e:
        print(f"Error checking policy: {e.stderr or e}")
        return


    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:DeleteObject",
                    "s3:ListAllMyBuckets",
                    "s3:ListBucket",
                    "s3:PutObject",
                    "s3:GetObject"
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket}/*"
                ]
            }
        ]
    }

    policy_json = json.dumps(policy, indent=2)

    policy_file = "/tmp/policy.json"
    # Save policy JSON to file
    with open(policy_file, 'w') as f:
        f.write(policy_json)


    # Create policy
    try:
        subprocess.run(
            ["mc", "admin", "policy", "create", target, policy_name, policy_file, "--insecure"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error creating policy: {e}")
        return

    # Attach policy to user
    try:
        print(f"Attaching allow policy to bucket '{bucket}'...")
        subprocess.run(
            ["mc", "admin", "policy", "attach", target, policy_name, "--user", user, "--insecure"],
            check=True
        )
        print(f"==> Policy '{policy_name}' attached to user '{user}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error attaching policy to user: {e}")


def minio_s3_create_user(username, s3_connection, user_s3_configuration_file, admin_s3_configuration_file):
    from umrs.s3 import s3_strong_password

    config = {}

    config_path = os.path.expanduser(admin_s3_configuration_file)

    # Initialize parser
    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read(config_path)
    alias = config.get("default", "alias", fallback="myminio")


    if not os.path.exists(os.path.expanduser(user_s3_configuration_file)):
        # Extract values with defaults
        s3_host = config.get("default", "host_base")
        use_https = config.getboolean("default", "use_https", fallback=False)
        verify_ssl = config.getboolean("default", "check_ssl_certificate", fallback=True)
        access_key = config.get("default", "access_key")
        secret_key = config.get("default", "secret_key")
        bucket_location = config.get("default", "bucket_location", fallback="ap-northeast-1")
        # Ensure correct protocol (http or https)
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
        alias = myminio
        """

        with open(os.path.expanduser(user_s3_configuration_file), "w") as file:
            file.write(textwrap.dedent(user_cfg))

        print(f"==> Configuration file written to: {user_s3_configuration_file}")

        # Set variables for MinIO user creation (NEW USER PATH)
        access_key = username
        secret_key = user_passwd
        alias = "myminio"

        # For new users, create the MinIO user immediately
        print(f"==> Creating MinIO user '{username}' (new user)...")
        minio_check_create_alias(alias, endpoint_url, access_key, secret_key, insecure=True)
        list_only_policy = minio_check_create_list_only_policy(alias)
        minio_create_user_with_policy(access_key, secret_key, alias, list_only_policy)
        print(f"==> MinIO user '{username}' created successfully.")

    else:
        print(f"==> S3 configuration file '{user_s3_configuration_file}' allready exists for user '{username}' and alias '{alias}'.")
        print(f"==> Delete this file manually if you want to reconfigure the user.")

        config = {}

        config_path = os.path.expanduser(user_s3_configuration_file)

        # Initialize parser
        config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        config.read(config_path)
        s3_host = config.get("default", "host_base")
        use_https = config.getboolean("default", "use_https", fallback=False)
        verify_ssl = config.getboolean("default", "check_ssl_certificate", fallback=True)
        access_key = config.get("default", "access_key")
        secret_key = config.get("default", "secret_key")
        bucket_location = config.get("default", "bucket_location", fallback="ap-northeast-1")
        alias = config.get("default", "alias", fallback="myminio")
        # Ensure correct protocol (http or https)
        protocol = "https" if use_https else "http"
        endpoint_url = f"{protocol}://{s3_host}"

        # For existing users, verify and potentially recreate the MinIO user
        print(f"==> Verifying MinIO user '{username}' (existing user)...")
        minio_check_create_alias(alias, endpoint_url, access_key, secret_key, insecure=True)
        list_only_policy = minio_check_create_list_only_policy(alias)
        minio_create_user_with_policy(access_key, secret_key, alias, list_only_policy)
        print(f"==> MinIO user '{username}' verified/created successfully.")

    # Common validation (for both new and existing users)
    if access_key != username:
        sys.exit(
            f"Critical error. The access key in '{user_s3_configuration_file}' should be '{username}', but is '{access_key}'.\n"
            "Cannot continue. Either update the JSON user config file or change/remove the S3 user config file."
        )

    return alias


def minio_remove_user_bucket_access(username, bucket_name, alias):
    """
    Remove user access policies for a specific bucket.

    Args:
        username (str): Username
        bucket_name (str): Name of the bucket to remove access from
        alias (str): MinIO alias
    """
    policy_name = f"{username}-{bucket_name}-policy"
    target = alias

    try:
        # Remove policy from user
        subprocess.run(
            ["mc", "admin", "policy", "detach", target, policy_name, "--user", username, "--insecure"],
            check=True
        )
        print(f"==> Policy '{policy_name}' detached from user '{username}'.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to detach policy from user: {e}")

    try:
        # Remove the policy itself
        subprocess.run(
            ["mc", "admin", "policy", "remove", target, policy_name, "--insecure"],
            check=True
        )
        print(f"==> Policy '{policy_name}' removed.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to remove policy: {e}")


def minio_admin_delete_user_bucket(username, bucket_name, alias):
    """
    Admin function to delete a user's bucket and all associated policies.
    This should only be called by admin users.

    Args:
        username (str): Username who owns the bucket
        bucket_name (str): Name of the bucket to delete
        alias (str): MinIO alias
    """
    target = alias

    print(f"==> Admin: Deleting bucket '{bucket_name}' for user '{username}'...")

    # First remove user access policies
    minio_remove_user_bucket_access(username, bucket_name, alias)

    # Remove any allow policies for this bucket
    allow_policy_name = f"allow-{bucket_name}"
    try:
        subprocess.run(
            ["mc", "admin", "policy", "detach", target, allow_policy_name, "--user", username, "--insecure"],
            check=True
        )
        print(f"==> Policy '{allow_policy_name}' detached from user '{username}'.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to detach allow policy: {e}")

    try:
        subprocess.run(
            ["mc", "admin", "policy", "remove", target, allow_policy_name, "--insecure"],
            check=True
        )
        print(f"==> Policy '{allow_policy_name}' removed.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to remove allow policy: {e}")

    # Delete the bucket itself (this will fail if bucket is not empty)
    try:
        subprocess.run(
            ["mc", "rb", f"{target}/{bucket_name}", "--insecure"],
            check=True
        )
        print(f"==> Bucket '{bucket_name}' deleted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to delete bucket '{bucket_name}': {e}")
        print(f"Note: Bucket must be empty before deletion. Use 'mc rb --force' to delete non-empty bucket.")


def minio_admin_delete_user(username, alias):
    """
    Admin function to completely delete a user and all their associated data.
    This should only be called by admin users.

    Args:
        username (str): Username to delete
        alias (str): MinIO alias
    """
    target = alias

    print(f"==> Admin: Deleting user '{username}' and all associated data...")

    # List all buckets owned by this user (buckets starting with username.)
    try:
        result = subprocess.run(
            ["mc", "ls", target, "--insecure"],
            capture_output=True, text=True, check=True
        )
        user_buckets = []
        for line in result.stdout.splitlines():
            if line.strip():
                # Extract bucket name from ls output
                parts = line.strip().split()
                if len(parts) >= 5:
                    bucket_name = parts[4]  # Bucket name is typically the last part
                    if bucket_name.startswith(f"{username}."):
                        user_buckets.append(bucket_name)

        # Delete all user's buckets
        for bucket_name in user_buckets:
            minio_admin_delete_user_bucket(username, bucket_name, alias)

    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to list buckets for cleanup: {e}")

    # Remove user from MinIO
    try:
        subprocess.run(
            ["mc", "admin", "user", "remove", target, username, "--insecure"],
            check=True
        )
        print(f"==> User '{username}' removed from MinIO.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to remove user '{username}': {e}")

    # Clean up user's S3 config file
    user_s3_config = f"/tmp/{username}.s3cfg"
    try:
        if os.path.exists(user_s3_config):
            os.remove(user_s3_config)
            print(f"==> User S3 config file '{user_s3_config}' removed.")
    except Exception as e:
        print(f"Warning: Failed to remove user S3 config: {e}")

    print(f"==> User '{username}' deletion completed.")
