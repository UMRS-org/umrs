import configparser
import json
import os
import subprocess
import textwrap

from umrs.shared.utils import normalize_json

MC_TIMEOUT = 30  # seconds, timeout for mc CLI subprocess calls

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
            capture_output=True, text=True, check=True,
            timeout=MC_TIMEOUT
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
        subprocess.run(cmd, check=True, timeout=MC_TIMEOUT)
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

    if not password or password.startswith("-"):
        raise ValueError(
            "MinIO user secrets passed to mc must be non-empty and must not start with '-'"
        )

    # First, ensure the user exists and the password matches the provided one
    try:
        result = subprocess.run(
            ["mc", "admin", "user", "list", target, "--insecure"],
            capture_output=True, text=True, check=True,
            timeout=MC_TIMEOUT
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
            timeout=MC_TIMEOUT,
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
            capture_output=True, text=True, check=True,
            timeout=MC_TIMEOUT
        )
        policies = [line.strip().split()[0] for line in result.stdout.splitlines() if line.strip()]
        if policy_name not in policies:
            print(f"Policy '{policy_name}' not found on '{target}', creating it first may be required.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error listing policies: {e.stderr}") from e

    # Attach the policy to the user (idempotent)
    try:
        subprocess.run(
            ["mc", "admin", "policy", "attach", target, policy_name, "--user", user, "--insecure"],
            check=True,
            timeout=MC_TIMEOUT
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
            capture_output=True, text=True, check=True,
            timeout=MC_TIMEOUT
        )
        policies = [line.strip().split()[0] for line in result.stdout.splitlines() if line.strip()]

        if policy_name not in policies:
            print(f"Policy '{policy_name}' does not exist on '{target}'.")
            return False

        # Get the actual policy content
        result = subprocess.run(
            ["mc", "admin", "policy", "info", target, policy_name, "--insecure"],
            capture_output=True, text=True, check=True,
            timeout=MC_TIMEOUT
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
                    "s3:ListAllMyBuckets",
                    "s3:GetBucketLocation"
                ],
                "Resource": [
                    "arn:aws:s3:::*"
                ]
            }
        ]
    }

    policy_json = json.dumps(policy, indent=2)
    policy_name = "listonly"
    policy_file = "/tmp/policy.json"

    # Check if the policy already exists
    try:
       if check_statement_policy(target, policy_name, policy_json):
            print(f"Policy '{policy_name}' already exists on '{target}'.")
            return policy_name
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error listing policies: {e.stderr}") from e

    # Save policy JSON to file
    with open(policy_file, 'w') as f:
        f.write(policy_json)

    # Create policy
    try:
        subprocess.run(
            ["mc", "admin", "policy", "create", target, policy_name, policy_file, "--insecure"],
            check=True,
            timeout=MC_TIMEOUT
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error creating policy: {e}") from e

    return policy_name


def minio_check_create_user_list_policy(target, username):
    """Create a per-user list policy.

    Uses two statements because ``ListAllMyBuckets`` is a service-level
    action (must apply to ``arn:aws:s3:::*``), while ``ListBucket``,
    ``CreateBucket``, and ``GetBucketLocation`` are scoped to the user's
    own bucket prefix.

    Note: ``ListAllMyBuckets`` on ``*`` exposes all bucket *names*
    (MinIO limitation). Per-bucket ``allow-*`` policies still protect
    data access — users can only read/write their own buckets.
    """
    policy_name = f"list-{username}"
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:ListAllMyBuckets",
                ],
                "Resource": [
                    "arn:aws:s3:::*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucket",
                    "s3:CreateBucket",
                    "s3:GetBucketLocation",
                ],
                "Resource": [
                    f"arn:aws:s3:::{username}.*",
                ],
            },
        ],
    }

    policy_json = json.dumps(policy, indent=2)
    policy_file = "/tmp/policy.json"

    # Check if the policy already exists with correct content
    try:
        if check_statement_policy(target, policy_name, policy_json):
            print(f"Policy '{policy_name}' already exists on '{target}'.")
            return policy_name
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error listing policies: {e.stderr}") from e

    # Save policy JSON to file
    with open(policy_file, "w") as f:
        f.write(policy_json)

    # Create policy
    try:
        subprocess.run(
            ["mc", "admin", "policy", "create", target, policy_name, policy_file, "--insecure"],
            check=True,
            timeout=MC_TIMEOUT,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error creating policy: {e}") from e

    return policy_name


def minio_migrate_user_list_policy(username, target):
    """Migrate a user from the shared ``listonly`` policy to per-user ``list-{username}``.

    Idempotent: safe to call multiple times.
    """
    # Create per-user policy (idempotent)
    user_policy = minio_check_create_user_list_policy(target, username)

    # Attach per-user policy
    try:
        subprocess.run(
            ["mc", "admin", "policy", "attach", target, user_policy, "--user", username, "--insecure"],
            check=True,
            timeout=MC_TIMEOUT,
        )
        print(f"==> Policy '{user_policy}' attached to user '{username}'.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: failed to attach '{user_policy}' to '{username}': {e}")

    # Detach shared listonly (best-effort)
    try:
        subprocess.run(
            ["mc", "admin", "policy", "detach", target, "listonly", "--user", username, "--insecure"],
            capture_output=True, text=True, check=True,
            timeout=MC_TIMEOUT,
        )
        print(f"==> Detached 'listonly' from user '{username}'.")
    except subprocess.CalledProcessError:
        pass  # Not attached or doesn't exist — fine


def minio_grant_user_access(user, bucket, target):
    """
    Create and attach a per-bucket access policy for a MinIO user.

    Grants:
    - Bucket-level: ListBucket + GetBucketLocation on the specific bucket
    - Object-level: Get/Put/DeleteObject on objects within the bucket

    Args:
        user (str): MinIO username (access key).
        bucket (str): Bucket name to grant access to.
        target (str): mc alias for MinIO server.
    """
    policy_name = f"allow-{bucket}"

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
                    f"arn:aws:s3:::{bucket}"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject"
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket}/*"
                ]
            }
        ]
    }

    policy_json = json.dumps(policy, indent=2)
    policy_file = "/tmp/policy.json"

    # Check if policy already exists with correct content AND is attached
    try:
        results = subprocess.run(
            ["mc", "admin", "policy", "entities", target, "--user", user, "--insecure"],
            capture_output=True, text=True, check=True,
            timeout=MC_TIMEOUT
        )
        already_attached = policy_name in results.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error checking policy: {e.stderr or e}")
        already_attached = False

    # Re-create policy if content changed (idempotent), even if attached
    policy_matches = check_statement_policy(target, policy_name, policy_json)
    if already_attached and policy_matches:
        print(f"==> Policy '{policy_name}' already attached to user '{user}' with correct scope.")
        return

    # Save policy JSON to file
    with open(policy_file, 'w') as f:
        f.write(policy_json)

    # Create/update policy
    try:
        subprocess.run(
            ["mc", "admin", "policy", "create", target, policy_name, policy_file, "--insecure"],
            check=True,
            timeout=MC_TIMEOUT
        )
        if not policy_matches:
            print(f"==> Policy '{policy_name}' updated with corrected scope.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating policy: {e}")
        return

    # Attach policy to user
    try:
        print(f"Attaching allow policy to bucket '{bucket}'...")
        subprocess.run(
            ["mc", "admin", "policy", "attach", target, policy_name, "--user", user, "--insecure"],
            check=True,
            timeout=MC_TIMEOUT
        )
        print(f"==> Policy '{policy_name}' attached to user '{user}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error attaching policy to user: {e}")


def minio_s3_create_user(username, s3_connection, user_s3_configuration_file, admin_s3_configuration_file):
    from umrs.shared.storage.s3 import s3_strong_password

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
        user_list_policy = minio_check_create_user_list_policy(alias, username)
        minio_create_user_with_policy(username, user_passwd, alias, user_list_policy)

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

        s3cfg_content = textwrap.dedent(user_cfg)
        expanded_path = os.path.expanduser(user_s3_configuration_file)
        os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
        with open(expanded_path, "w") as file:
            file.write(s3cfg_content)
        print(f"==> Configuration file written to: {user_s3_configuration_file}")

        # Mirror to UMRS_CONFIG/user_files/ so _read_user_s3cfg's priority-1
        # path stays in sync with the authoritative ~/.config/umrs copy.
        _mirror_s3cfg_to_user_files(username, s3cfg_content)

    else:
        print(f"==> S3 configuration file '{user_s3_configuration_file}' allready exists for user '{username}' and alias '{alias}'.")
        print("==> Delete this file manually if you want to reconfigure the user.")
        # Read existing user config
        user_cfg_path = os.path.expanduser(user_s3_configuration_file)
        ucfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        ucfg.read(user_cfg_path)
        u_access_key = ucfg.get("default", "access_key")
        u_secret_key = ucfg.get("default", "secret_key")

        if u_access_key != username:
            raise RuntimeError(
                f"Critical error. The access key in '{user_s3_configuration_file}' should be '{username}', but is '{u_access_key}'.\n"
                "Cannot continue. Either update the JSON user config file or change/remove the S3 user config file."
            )
        if not u_secret_key or u_secret_key.startswith("-"):
            # mc parses a leading '-' in positional args as another flag, so
            # regenerate an existing unsafe secret before reconciling the user.
            u_secret_key = s3_strong_password()
            ucfg.set("default", "secret_key", u_secret_key)
            with open(user_cfg_path, "w") as file:
                ucfg.write(file)
            print(f"==> Updated unsafe S3 secret in: {user_s3_configuration_file}")
        # Ensure mc alias exists using admin credentials (needed for admin commands)
        minio_check_create_alias(alias, endpoint_url, admin_access_key, admin_secret_key, insecure=not use_https or not verify_ssl)
        user_list_policy = minio_check_create_user_list_policy(alias, username)
        # Ensure user exists with the password declared in the user's config
        minio_create_user_with_policy(username, u_secret_key, alias, user_list_policy)

        # Read existing s3cfg content for delivery to remote client
        with open(user_cfg_path, encoding="utf-8") as f:
            s3cfg_content = f.read()

        # Mirror after reading to keep user_files in sync
        _prune_stale_user_files_s3cfg(username, s3cfg_content)


    return alias, s3cfg_content


def _umrs_user_files_dir() -> str | None:
    """Return the ``UMRS_CONFIG/user_files`` directory path if it exists."""
    umrs_config = os.environ.get("UMRS_CONFIG")
    if not umrs_config:
        return None
    path = os.path.join(os.path.expanduser(umrs_config), "user_files")
    return path if os.path.isdir(path) else None


def _mirror_s3cfg_to_user_files(username: str, s3cfg_content: str) -> None:
    """Write *s3cfg_content* to ``UMRS_CONFIG/user_files/s3cfg_{username}``.

    This keeps the priority-1 path in ``_read_user_s3cfg`` in sync with the
    authoritative ``~/.config/umrs/s3cfg_{username}`` copy, preventing stale
    credentials from being served after a MinIO user is re-provisioned.
    """
    target_dir = _umrs_user_files_dir()
    if not target_dir:
        return
    target_path = os.path.join(target_dir, f"s3cfg_{username}")
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(s3cfg_content)
    except OSError:
        pass


def _prune_stale_user_files_s3cfg(username: str, s3cfg_content: str) -> None:
    """Remove ``user_files/s3cfg_{username}`` if it differs from *s3cfg_content*.

    When a user is re-provisioned, the stale copy in ``UMRS_CONFIG/user_files``
    otherwise shadows the updated credentials in ``~/.config/umrs``.
    """
    target_dir = _umrs_user_files_dir()
    if not target_dir:
        return
    target_path = os.path.join(target_dir, f"s3cfg_{username}")
    if not os.path.isfile(target_path):
        return
    try:
        with open(target_path, encoding="utf-8") as f:
            existing = f.read()
        if existing.strip() != s3cfg_content.strip():
            os.remove(target_path)
    except OSError:
        pass


def minio_delete_user(username, target):
    """Remove a MinIO IAM user and all attached policies.

    Args:
        username: MinIO username (access key) to remove.
        target: mc alias (e.g. 'myminio').
    """
    # Detach all policies first.
    minio_detach_all_policies(username, target)

    try:
        subprocess.run(
            ["mc", "admin", "user", "remove", target, username, "--insecure"],
            check=True,
            timeout=MC_TIMEOUT,
        )
        print(f"==> MinIO user '{username}' removed from '{target}'.")
    except subprocess.CalledProcessError as e:
        # User may not exist — treat as idempotent.
        stderr = getattr(e, "stderr", "") or ""
        if "does not exist" in stderr.lower() or "not found" in stderr.lower():
            print(f"==> MinIO user '{username}' does not exist on '{target}', skipping.")
        else:
            print(f"Error removing MinIO user '{username}': {e}")


def minio_detach_all_policies(username, target):
    """Detach all IAM policies from a MinIO user.

    Args:
        username: MinIO username (access key).
        target: mc alias (e.g. 'myminio').
    """
    try:
        result = subprocess.run(
            ["mc", "admin", "policy", "entities", target, "--user", username, "--insecure"],
            capture_output=True, text=True, check=True,
            timeout=MC_TIMEOUT,
        )
        # Parse attached policy names from output.
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("Policy") or line.startswith("-"):
                continue
            policy_name = line.split()[0] if line.split() else None
            if policy_name:
                try:
                    subprocess.run(
                        ["mc", "admin", "policy", "detach", target, policy_name,
                         "--user", username, "--insecure"],
                        check=True, timeout=MC_TIMEOUT,
                    )
                    print(f"  Detached policy '{policy_name}' from user '{username}'.")
                except subprocess.CalledProcessError:
                    print(f"  Warning: failed to detach policy '{policy_name}' from '{username}'.")
    except subprocess.CalledProcessError:
        # User may not exist or have no policies — safe to ignore.
        pass


def minio_remove_bucket(bucket, target):
    """Remove an S3 bucket and all its contents via mc.

    Args:
        bucket: Bucket name (e.g. 'user1.text-embedding.user1-texts').
        target: mc alias (e.g. 'myminio').
    """
    try:
        subprocess.run(
            ["mc", "rb", "--force", f"{target}/{bucket}", "--insecure"],
            check=True,
            timeout=MC_TIMEOUT * 10,  # Large buckets may take longer
        )
        print(f"==> Bucket '{bucket}' removed from '{target}'.")
    except subprocess.CalledProcessError as e:
        stderr = getattr(e, "stderr", "") or ""
        if "does not exist" in stderr.lower() or "not found" in stderr.lower():
            print(f"==> Bucket '{bucket}' does not exist on '{target}', skipping.")
        else:
            print(f"Error removing bucket '{bucket}': {e}")
