import subprocess
import sys, os
import json
import secrets
import string
import configparser
import textwrap
import subprocess
from umrs.utils import normalize_json

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
        user (str): username (access key)
        password (str): user's password (secret key)
        target (str): mc alias for MinIO server
        policy_name (str): policy to attach (must already exist)
    """

    # First, ensure the user exists and the password matches the provided one
    try:
        result = subprocess.run(
            ["mc", "admin", "user", "list", target, "--insecure"],
            capture_output=True, text=True, check=True
        )
        user_exists = user in result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error listing users: {e.stderr}")
        return

    try:
        # mc admin user add sets/updates the user's secret when invoked
        subprocess.run(
            ["mc", "admin", "user", "add", target, user, password, "--insecure"],
            check=True,
            capture_output=True,
            text=True,
        )
        if user_exists:
            print(f"User '{user}' exists; secret updated.")
        else:
            print(f"User '{user}' created.")
    except subprocess.CalledProcessError as e:
        # If add fails (unexpected), log and try to continue to attach policies
        stderr = e.stderr or str(e)
        print(f"Warning: Unable to add/update user '{user}': {stderr}")

    # Ensure the policy exists on the cluster
    try:
        result = subprocess.run(
            ["mc", "admin", "policy", "list", target, "--insecure"],
            capture_output=True, text=True, check=True
        )
        policies = [line.strip().split()[0] for line in result.stdout.splitlines() if line.strip()]
        if policy_name not in policies:
            print(f"Policy '{policy_name}' not found on '{target}', creating it first may be required.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error listing policies: {e.stderr}")

    # Attach the policy to the user (idempotent)
    try:
        subprocess.run(
            ["mc", "admin", "policy", "attach", target, policy_name, "--user", user, "--insecure"],
            check=True
        )
        print(f"Policy '{policy_name}' attached to user '{user}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error attaching policy to user: {e}")


def check_statement_policy(target, policy_name, policy_json):
    """
    Check if a policy exists on MinIO and whether its 'Statement' matches policy_json
    """
    # Load input policy and extract Statement
    input_policy = json.loads(policy_json)
    input_statement = normalize_json(input_policy.get("Statement", []))
    
    try:
        # List policies on the MinIO server
        result = subprocess.run(
            ["mc", "admin", "policy", "list", target, "--insecure"],
            capture_output=True, text=True, check=True
        )
        policies = [line.strip().split()[0] for line in result.stdout.splitlines() if line.strip()]

        if policy_name not in policies:
            print(f"Policy '{policy_name}' does not exist on '{target}'.")
            return False

        # Get the actual policy content
        result = subprocess.run(
            ["mc", "admin", "policy", "info", target, policy_name, "--insecure"],
            capture_output=True, text=True, check=True
        )
        actual_policy = json.loads(result.stdout)
        actual_statement = normalize_json(actual_policy.get("Policy", {}).get("Statement", []))
     
        if input_statement == actual_statement:
            print(f"Policy '{policy_name}' exists and Statement matches.")
            return True
        else:
            print(f"Policy '{policy_name}' exists but Statement does NOT match.")
            return False

    except subprocess.CalledProcessError as e:
        print("Error accessing MinIO:", e)
        return False
    
def minio_check_create_list_only_policy(target):
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucket",
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
       if check_statement_policy(target, policy_name, policy_json):
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

    # Load admin S3 config (for mc admin operations)
    admin_cfg_path = os.path.expanduser(admin_s3_configuration_file)
    admin_cfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    admin_cfg.read(admin_cfg_path)
    alias = admin_cfg.get("default", "alias", fallback="myminio")
    s3_host = admin_cfg.get("default", "host_base")
    use_https = admin_cfg.getboolean("default", "use_https", fallback=False)
    verify_ssl = admin_cfg.getboolean("default", "check_ssl_certificate", fallback=True)
    admin_access_key = admin_cfg.get("default", "access_key")
    admin_secret_key = admin_cfg.get("default", "secret_key")
    bucket_location = admin_cfg.get("default", "bucket_location", fallback="us-east-1")
    protocol = "https" if use_https else "http"
    endpoint_url = f"{protocol}://{s3_host}"

    
    if not os.path.exists(os.path.expanduser(user_s3_configuration_file)):
        # First-run: create mc alias (admin), ensure policy, create user with random password, then write user config
        user_passwd = s3_strong_password()

        # Ensure mc alias exists using admin credentials
        minio_check_create_alias(alias, endpoint_url, admin_access_key, admin_secret_key, insecure=not use_https or not verify_ssl)
        list_only_policy = minio_check_create_list_only_policy(alias)
        minio_create_user_with_policy(username, user_passwd, alias, list_only_policy)

        user_cfg = f"""[default]
        host_base = {s3_host}
        host_bucket = {s3_host}
        bucket_location = {bucket_location}
        use_https = {use_https}
        check_ssl_certificate = {verify_ssl}
        access_key = {username}
        secret_key = {user_passwd}
        alias = {alias}
        """

        with open(os.path.expanduser(user_s3_configuration_file), "w") as file:
            file.write(textwrap.dedent(user_cfg))

        print(f"==> Configuration file written to: {user_s3_configuration_file}")
    else:
        print(f"==> S3 configuration file '{user_s3_configuration_file}' allready exists for user '{username}' and alias '{alias}'.")
        print(f"==> Delete this file manually if you want to reconfigure the user.")
        # Read existing user config
        user_cfg_path = os.path.expanduser(user_s3_configuration_file)
        ucfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        ucfg.read(user_cfg_path)
        u_access_key = ucfg.get("default", "access_key")
        u_secret_key = ucfg.get("default", "secret_key")

        if u_access_key != username:
            sys.exit(
                f"Critical error. The access key in '{user_s3_configuration_file}' should be '{username}', but is '{u_access_key}'.\n"
                "Cannot continue. Either update the JSON user config file or change/remove the S3 user config file."
            )
        # Ensure mc alias exists using admin credentials (needed for admin commands)
        minio_check_create_alias(alias, endpoint_url, admin_access_key, admin_secret_key, insecure=not use_https or not verify_ssl)
        list_only_policy = minio_check_create_list_only_policy(alias)
        # Ensure user exists with the password declared in the user's config
        minio_create_user_with_policy(username, u_secret_key, alias, list_only_policy)

        
    return alias
