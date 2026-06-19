"""Server-side file operations: user config persistence, workflow upload."""

import json
import logging
import os

from umrs.shared.auth import sanitize_username, validate_config_identity
from umrs.shared.config import (
    get_default_tmp_subdir,
    get_users_default_cfg_subdir,
)
from umrs.shared.json_config import load_client_cfg_file

logger = logging.getLogger(__name__)


def ensure_user_file_in_conf(conf_path, username, canonical_user_path):
    """Ensure the server conf JSON lists exactly one config entry for *username*."""
    canonical_user_path = os.path.expanduser(canonical_user_path)

    try:
        with open(conf_path, encoding="utf-8") as f:
            conf = json.load(f)
    except Exception as e:
        logger.info(f"Warning: failed to read conf '{conf_path}': {e}")
        return False

    users = conf.get("users", [])

    # Build a new list that keeps only one entry for this username (the canonical path)
    new_users = []
    removed = []
    for p in users:
        p = os.path.expanduser(p)
        if p == canonical_user_path:
            # keep canonical path if already present
            if p not in new_users:
                new_users.append(p)
            continue
        # Try to read file and inspect username; if mismatch or file missing, keep as-is
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            u = data.get("username")
        except Exception:
            u = None
        if u == username:
            # Another config for the same user; drop it
            removed.append(p)
            continue
        # Different user or unreadable -> keep entry
        if p not in new_users:
            new_users.append(p)

    # Ensure canonical path present
    if canonical_user_path not in new_users:
        new_users.append(canonical_user_path)

    if removed:
        logger.info(
            f"Updated server conf: removed {len(removed)} stale user config(s) "
            f"for '{username}': {removed}"
        )

    # change absolute path to '~' if possible
    home = os.path.expanduser("~")
    new_users = [path.replace(home, "~", 1) for path in new_users]

    conf["users"] = new_users
    try:
        with open(conf_path, "w", encoding="utf-8") as f:
            json.dump(conf, f, indent=2)
        logger.info(
            f"Updated server conf: ensured unique config for "
            f"'{username}' -> {canonical_user_path}"
        )
    except Exception as e:
        logger.info(f"Warning: failed to update conf '{conf_path}': {e}")
        return False

    # Best-effort cleanup: remove pruned files under user_cfg_path only
    user_cfg_path = os.path.join(
        os.path.dirname(conf_path), get_users_default_cfg_subdir()
    )
    for stale in removed:
        try:
            if (
                stale.startswith(user_cfg_path)
                and stale != canonical_user_path
                and os.path.isfile(stale)
            ):
                os.remove(stale)
                logger.info(f"Deleted stale user config file: {stale}")
        except Exception as e:
            logger.info(f"Warning: failed to delete stale user config '{stale}': {e}")
    return True


def handle_upload_workflow(client_id, zip_path, workflow_name):
    """Validate and extract an uploaded workflow zip to the user_workflows directory.

    The zip must contain a ``workflow.json`` at its root (or one level deep)
    with a parseable :class:`WorkflowDefinition`.

    Returns ``(success: bool, dest_dir: str | None, error: str | None)``.
    """
    import zipfile

    from umrs.pipeline_sdk.workflow import WorkflowDefinition

    if not os.path.isfile(zip_path):
        return False, None, f"Uploaded file not found: {zip_path}"

    # Basic size sanity check (reject > 500 MB)
    max_size = 500 * 1024 * 1024
    if os.path.getsize(zip_path) > max_size:
        return (
            False,
            None,
            f"Zip file too large ({os.path.getsize(zip_path)} bytes, max {max_size})",
        )

    if not zipfile.is_zipfile(zip_path):
        return False, None, f"Uploaded file is not a valid zip archive: {zip_path}"

    # Determine destination — use an ABSOLUTE path so that the
    # stored ``source_path`` resolves correctly regardless of the
    # working directory of the process that later reads it (e.g.
    # the durable-job backend running in a different cwd than the
    # UMRS server).
    workflows_root = os.path.join(
        get_default_tmp_subdir(), "user_workflows", client_id
    )
    dest_dir = os.path.abspath(
        os.path.join(workflows_root, workflow_name)
    )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Security: reject paths that escape the extraction directory
            for member in zf.namelist():
                resolved = os.path.realpath(os.path.join(dest_dir, member))
                if not resolved.startswith(os.path.realpath(dest_dir)):
                    return False, None, f"Zip contains unsafe path: {member}"

            # Look for workflow.json at root or one level deep
            candidates = [
                n for n in zf.namelist() if os.path.basename(n) == "workflow.json"
            ]
            if not candidates:
                return False, None, "Zip archive does not contain a workflow.json"

            # Validate workflow.json contents before extracting
            wf_entry = candidates[0]
            try:
                wf_data = json.loads(zf.read(wf_entry))
                WorkflowDefinition.from_dict(wf_data)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                return False, None, f"Invalid workflow.json: {e}"

            # Extract
            os.makedirs(dest_dir, exist_ok=True)
            zf.extractall(dest_dir)

        # If workflow.json was nested one level (e.g., myworkflow/workflow.json),
        # move contents up so dest_dir directly contains workflow.json
        actual_wf = os.path.join(dest_dir, "workflow.json")
        if not os.path.isfile(actual_wf):
            # Try to find it one level deep
            for name in os.listdir(dest_dir):
                candidate = os.path.join(dest_dir, name, "workflow.json")
                if os.path.isfile(candidate):
                    # Move everything from the subdirectory up
                    sub_dir = os.path.join(dest_dir, name)
                    import shutil

                    for item in os.listdir(sub_dir):
                        shutil.move(os.path.join(sub_dir, item), dest_dir)
                    os.rmdir(sub_dir)
                    break

        # Rewrite workflow.json source to UPLOADED type
        actual_wf = os.path.join(dest_dir, "workflow.json")
        if os.path.isfile(actual_wf):
            with open(actual_wf, encoding="utf-8") as fh:
                wf_data = json.load(fh)

            source_cfg = wf_data.get("source") or {}
            source_path = source_cfg.get("path") or "."
            if os.path.isabs(source_path):
                resolved_source = os.path.realpath(source_path)
            else:
                resolved_source = os.path.realpath(os.path.join(dest_dir, source_path))

            dest_root = os.path.realpath(dest_dir)
            if not resolved_source.startswith(dest_root):
                return (
                    False,
                    None,
                    f"Workflow source path escapes extracted directory: {source_path}",
                )
            if not os.path.isdir(resolved_source):
                return (
                    False,
                    None,
                    f"Workflow source directory missing after extraction: "
                    f"{resolved_source}",
                )

            wf_data["source"] = {"type": "uploaded", "path": resolved_source}
            with open(actual_wf, "w", encoding="utf-8") as fh:
                json.dump(wf_data, fh, indent=2)
    except zipfile.BadZipFile:
        return False, None, f"Corrupted zip archive: {zip_path}"
    except Exception as e:
        return False, None, f"Failed to extract workflow: {e}"
    finally:
        # Clean up the uploaded temp zip
        try:
            os.remove(zip_path)
        except Exception as e:
            logger.info(f"Warning: failed to clean up zip file '{zip_path}': {e}")

    logger.info(f"Workflow '{workflow_name}' for user '{client_id}' extracted to {dest_dir}")
    return True, dest_dir, None


def persist_uploaded_user_config(
    tmp_path,
    configure_file,
    user_config_map,
    expected_username=None,
    *,
    allow_create=False,
):
    """Persist an uploaded user config JSON.

    When *allow_create* is ``False`` (the default) the function refuses to
    create entries for usernames that are not already present in
    *user_config_map*.  Only the admin-only ``ENROLL_USER`` path should
    pass ``allow_create=True``.

    Returns ``(success: bool, username: str | None, dest_path: str | None)``.
    """
    try:
        # Validate against client schema (strict) and obtain data
        data = load_client_cfg_file(tmp_path)
        username = data.get("username")
        if not username:
            logger.info(
                f"Warning: uploaded file '{tmp_path}' missing 'username' field; "
                f"ignoring"
            )
            return False, None, None

        # Sanitize username to prevent path-traversal attacks
        safe_username = sanitize_username(username)
        if safe_username != username:
            logger.warning(
                "Rejecting unsafe username: %r (sanitized: %r)",
                username,
                safe_username,
            )
            logger.info(
                f"Warning: uploaded file '{tmp_path}' contains unsafe "
                f"username '{username}'; rejecting"
            )
            return False, None, None

        # Rich identity validation (checks username match + flags admin claims)
        if expected_username:
            id_result = validate_config_identity(data, expected_username)
            if not id_result.valid:
                logger.info(
                    f"Warning: uploaded config identity validation failed: "
                    f"{'; '.join(id_result.errors)}; ignoring file"
                )
                return False, None, None
            for w in id_result.warnings:
                logger.warning("Config identity warning: %s", w)
    except SystemExit:
        # load_client_cfg_file may sys.exit() on validation error
        logger.info(f"Warning: uploaded file '{tmp_path}' failed validation; ignoring")
        return False, None, None
    except Exception as e:
        logger.info(f"Warning: cannot parse uploaded config '{tmp_path}': {e}")
        return False, None, None

    # look up existing users configuration path
    dest_path = user_config_map.get(username, None)

    if not dest_path:
        if not allow_create:
            logger.info(
                f"Rejected: user '{username}' is not enrolled on this server. "
                f"An admin must run 'admin user add' first."
            )
            return False, username, None

        # Destination path (canonical) — only reached when allow_create=True
        umrs_dir = os.path.dirname(configure_file)
        user_dir = os.path.join(umrs_dir, get_users_default_cfg_subdir())
        try:
            os.makedirs(user_dir, exist_ok=True)
        except Exception as e:
            logger.info(f"Warning: cannot ensure user files directory '{user_dir}': {e}")
            return False, username, None
        dest_path = os.path.join(user_dir, f"{username}.json")

    dest_path = os.path.expanduser(dest_path)

    # Atomically replace the destination file
    try:
        with open(tmp_path, encoding="utf-8") as src:
            content = src.read()
        tmp_out = dest_path + ".tmp"
        with open(tmp_out, "w", encoding="utf-8") as out:
            out.write(content)
        os.replace(tmp_out, dest_path)
        # Remove original temp upload
        try:
            os.remove(tmp_path)
        except Exception as e:
            logger.info(f"Warning: failed to clean up temp file '{tmp_path}': {e}")
        logger.info(f"Stored updated user config for '{username}' at '{dest_path}'")
    except Exception as e:
        logger.info(f"Warning: failed to persist uploaded config to '{dest_path}': {e}")
        return False, username, None

    # Ensure conf contains this user file path
    conf_path = os.path.abspath(configure_file)
    ensure_user_file_in_conf(conf_path, username, dest_path)
    return True, username, dest_path
