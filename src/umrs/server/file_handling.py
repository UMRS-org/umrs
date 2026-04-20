"""Server-side file operations: user config persistence, workflow upload."""

import json
import os
import logging

from umrs.shared.config import (
    ensure_private_directory,
    get_default_tmp_subdir,
    get_users_default_cfg_subdir,
    resolve_config_path,
    resolve_server_user_files_dir,
)
from umrs.server.json_config import load_server_cfg_json
from umrs.shared.json_config import load_client_cfg_file

logger = logging.getLogger(__name__)


def ensure_user_file_in_conf(conf_path, username, canonical_user_path):
    """Ensure the server conf JSON lists exactly one config entry for *username*."""
    canonical_user_path = os.path.expanduser(canonical_user_path)

    try:
        with open(conf_path, "r", encoding="utf-8") as f:
            conf = json.load(f)
    except Exception as e:
        print(f"Warning: failed to read conf '{conf_path}': {e}")
        return False

    users = conf.get("users", [])

    # Build a new list that keeps only one entry for this username (the canonical path)
    new_users = []
    removed = []
    for p in users:
        p = resolve_config_path(p, config_file=conf_path)
        if p == canonical_user_path:
            # keep canonical path if already present
            if p not in new_users:
                new_users.append(p)
            continue
        # Try to read file and inspect username; if mismatch or file missing, keep as-is
        try:
            with open(p, "r", encoding="utf-8") as f:
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
        print(
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
        print(
            f"Updated server conf: ensured unique config for "
            f"'{username}' -> {canonical_user_path}"
        )
    except Exception as e:
        print(f"Warning: failed to update conf '{conf_path}': {e}")
        return False

    # Best-effort cleanup: remove pruned files under user_cfg_path only
    user_cfg_path = os.path.dirname(canonical_user_path)
    for stale in removed:
        try:
            if (
                stale.startswith(user_cfg_path)
                and stale != canonical_user_path
                and os.path.isfile(stale)
            ):
                os.remove(stale)
                print(f"Deleted stale user config file: {stale}")
        except Exception as e:
            print(f"Warning: failed to delete stale user config '{stale}': {e}")
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

    # Determine destination
    workflows_root = os.path.join(
        get_default_tmp_subdir(), "user_workflows", client_id
    )
    dest_dir = os.path.join(workflows_root, workflow_name)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Security: reject paths that escape the extraction directory
            for member in zf.namelist():
                resolved = os.path.realpath(os.path.join(dest_dir, member))
                if not resolved.startswith(os.path.realpath(dest_dir)):
                    return False, None, f"Zip contains unsafe path: {member}"

            # Look for workflow.json at root or one level deep
            candidates = [
                n
                for n in zf.namelist()
                if os.path.basename(n) == "workflow.json"
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
            with open(actual_wf, "r", encoding="utf-8") as fh:
                wf_data = json.load(fh)

            source_cfg = wf_data.get("source") or {}
            source_path = source_cfg.get("path") or "."
            if os.path.isabs(source_path):
                resolved_source = os.path.realpath(source_path)
            else:
                resolved_source = os.path.realpath(
                    os.path.join(dest_dir, source_path)
                )

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
            print(f"Warning: failed to clean up zip file '{zip_path}': {e}")

    print(
        f"Workflow '{workflow_name}' for user '{client_id}' extracted to {dest_dir}"
    )
    return True, dest_dir, None


def persist_uploaded_user_config(
    tmp_path, configure_file, user_config_map, expected_username=None
):
    """Persist an uploaded user config JSON.

    Returns ``(success: bool, username: str | None, dest_path: str | None)``.
    """
    try:
        # Validate against client schema (strict) and obtain data
        data = load_client_cfg_file(tmp_path)
        username = data.get("username")
        if not username:
            print(
                f"Warning: uploaded file '{tmp_path}' missing 'username' field; "
                f"ignoring"
            )
            return False, None, None
        if expected_username and username != expected_username:
            print(
                f"Warning: uploaded config username '{username}' does not match "
                f"client identity '{expected_username}'; ignoring file"
            )
            return False, None, None
    except SystemExit:
        # load_client_cfg_file may sys.exit() on validation error
        print(f"Warning: uploaded file '{tmp_path}' failed validation; ignoring")
        return False, None, None
    except Exception as e:
        print(f"Warning: cannot parse uploaded config '{tmp_path}': {e}")
        return False, None, None

    # look up existing users configuration path
    dest_path = user_config_map.get(username, None)

    if not dest_path:
        try:
            server_cfg_json = load_server_cfg_json(configure_file)
            user_dir = ensure_private_directory(
                resolve_server_user_files_dir(
                    config_file=configure_file,
                    server_cfg=server_cfg_json,
                    user_config_map=user_config_map,
                )
            )
        except (OSError, SystemExit) as e:
            print(
                f"Warning: cannot ensure user files directory for '{configure_file}': {e}"
            )
            return False, username, None
        dest_path = os.path.join(user_dir, f"{username}.json")

    dest_path = os.path.expanduser(dest_path)

    # Atomically replace the destination file
    try:
        with open(tmp_path, "r", encoding="utf-8") as src:
            content = src.read()
        tmp_out = dest_path + ".tmp"
        with open(tmp_out, "w", encoding="utf-8") as out:
            out.write(content)
        os.replace(tmp_out, dest_path)
        # Remove original temp upload
        try:
            os.remove(tmp_path)
        except Exception as e:
            print(f"Warning: failed to clean up temp file '{tmp_path}': {e}")
        print(f"Stored updated user config for '{username}' at '{dest_path}'")
    except Exception as e:
        print(
            f"Warning: failed to persist uploaded config to '{dest_path}': {e}"
        )
        return False, username, None

    # Ensure conf contains this user file path
    conf_path = os.path.abspath(configure_file)
    ensure_user_file_in_conf(conf_path, username, dest_path)
    return True, username, dest_path
