import sys

import pandas as pd
from tabulate import tabulate

from umrs.shared.utils import utc_to_local


def _thread_sort_key(thread_status, key):
    thread_id = thread_status[key].get("thread_id")
    if isinstance(thread_id, int):
        return (0, thread_id)
    return (1, str(thread_id))


def _render_thread_id(thread_id):
    if thread_id in (None, ""):
        return "N/A"
    return thread_id


def _render_last_update(info):
    return utc_to_local(info.get("updated_at", "-"))


def _build_thread_status_row(work, thread_id, status, updated_at, running_only):
    row = [
        thread_id,
        work.get("work_type", "N/A"),
        work.get("user", "N/A"),
        work.get("nb_objects", "N/A"),
        work.get("modality_name", "N/A"),
        work.get("bucket", "N/A"),
        f"{work.get('data_base', 'N/A')}/{work.get('table_name', 'N/A')}",
    ]
    if not running_only:
        row.append(status)
    row.append(updated_at)
    return row


def _thread_status_headers(running_only):
    headers = [
        "Thread ID",
        "Work Type",
        "Username",
        "Nb Objects",
        "Modality",
        "Bucket",
        "Database/Table",
    ]
    if not running_only:
        headers.append("Status")
    headers.append("Last Update")
    return headers


def pretty_print_thread_status(thread_status, running_only=True, print_output=True):
    rows = []
    for key in sorted(
        thread_status, key=lambda current: _thread_sort_key(thread_status, current)
    ):
        info = thread_status[key]
        thread_id = _render_thread_id(info.get("thread_id", "N/A"))
        status = info.get("status", "N/A")
        updated_at = _render_last_update(info)
        if status != "Running" and running_only:
            continue
        for work in info.get("work", []):
            rows.append(
                _build_thread_status_row(
                    work, thread_id, status, updated_at, running_only
                )
            )

    if rows:
        headers = _thread_status_headers(running_only)
        table = tabulate(rows, headers, tablefmt="grid")
        message = "Server status:\n" + table
    else:
        message = "Server is idle"
    if print_output:
        print(message)
    return message


def pretty_print_cfg(cfg_dict, print_output=True):
    # Create an empty DataFrame with predefined columns
    df = pd.DataFrame(
        columns=[
            "Bucket alias",
            "S3 user",
            "S3 path",
            "Experience",
            "Modality",
            "Database",
            "SQL table",
            "SQL user",
            "Model",
            "Embeds?",
            "Metadata",
        ]
    )
    buckets_cfg = cfg_dict.buckets_cfg

    def _truncate(val: str, max_len: int = 40) -> str:
        s = str(val)
        if len(s) > max_len:
            return s[:max_len - 3] + "..."
        return s

    for key, entry in buckets_cfg.items():
        bucket_alias = entry.bucket_alias
        s3_username = entry.s3_username
        sql_username = entry.sql_username
        model_name = entry.model.model_name
        store_embeddings_flag = entry.store_embeddings_flag
        modality = entry.modality_name
        table_name = entry.table_name
        xp = entry.xp_name
        db_name = entry.db_name
        metadata = entry.metadata
        metadata_str = ""
        for mentry in metadata.values():
            name = mentry["name"]
            value_type = mentry["type"]
            if metadata_str != "":
                metadata_str += " ; "
            metadata_str += f"{name} ({value_type})"

        df.loc[len(df)] = [
            _truncate(bucket_alias, 20),
            _truncate(s3_username, 12),
            _truncate(key, 50),
            _truncate(xp, 15),
            _truncate(modality, 10),
            _truncate(db_name, 30),
            _truncate(table_name, 12),
            _truncate(sql_username, 12),
            _truncate(model_name, 20),
            store_embeddings_flag,
            metadata_str,
        ]

    df.sort_values(by=["S3 user", "Bucket alias"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    message = (
        "\nServer configuration:"
        + "\n"
        + tabulate(df, headers="keys", tablefmt="grid")
    )
    if print_output:
        print(message)
    return message


def _model_name_for_bucket(entry) -> str:
    """Return a display name for the model attached to a bucket entry."""
    if entry.model is None:
        return "(unknown)"
    return getattr(entry.model, "model_name", "(unknown)")


def pretty_print_cfg_for_user(cfg_dict, username, print_output=False):
    """Return a configuration summary filtered to a single user's buckets."""
    df = pd.DataFrame(
        columns=[
            "Bucket alias",
            "Experience",
            "Modality",
            "Database",
            "SQL table",
            "Model",
            "Embeds?",
            "Metadata",
        ]
    )

    def _truncate(val: str, max_len: int = 40) -> str:
        s = str(val)
        if len(s) > max_len:
            return s[:max_len - 3] + "..."
        return s

    # Accept dict, BucketsCfg (has .items()), or object with .buckets_cfg attr
    if isinstance(cfg_dict, dict):
        buckets_cfg = cfg_dict
    elif hasattr(cfg_dict, "items"):
        buckets_cfg = cfg_dict
    else:
        buckets_cfg = cfg_dict.buckets_cfg
    for key, entry in buckets_cfg.items():
        if entry.s3_username != username:
            continue
        metadata_str = ""
        for mentry in entry.metadata.values():
            name = mentry["name"]
            value_type = mentry["type"]
            if metadata_str:
                metadata_str += " ; "
            metadata_str += f"{name} ({value_type})"
        df.loc[len(df)] = [
            _truncate(entry.bucket_alias, 20),
            _truncate(entry.xp_name, 15),
            _truncate(entry.modality_name, 10),
            _truncate(entry.db_name, 30),
            _truncate(entry.table_name, 12),
            _truncate(_model_name_for_bucket(entry), 20),
            entry.store_embeddings_flag,
            metadata_str,
        ]

    df.sort_values(by=["Bucket alias"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if df.empty:
        message = f"\nNo configuration found for user '{username}'."
    else:
        message = f"\nConfiguration for '{username}':\n" + tabulate(
            df, headers="keys", tablefmt="simple"
        )
    if print_output:
        print(message)
    return message


def pretty_print_user_cfg(cfg_dict, print_output=True):

    df = pd.DataFrame(
        columns=[
            "Modality",
            "Bucket alias",
            "Experience",
            "Bucket path",
            "DB name",
            "SQL user",
        ]
    )
    username = None
    for key, entry in cfg_dict.items():
        bucket_alias = key
        bucket_path = entry.path_to_bucket
        xp_name = entry.xp_name
        modality = entry.modality_name
        sql_username = entry.sql_username
        db_name = entry.db_name

        user = entry.username
        if not username:
            username = user
        elif user != username:
            sys.exit(f"user mismatch between {username} and {user}")

        df.loc[len(df)] = [
            modality,
            bucket_alias,
            xp_name,
            bucket_path,
            db_name,
            sql_username,
        ]
    df.sort_values(by=["Bucket alias"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    message = f"Configuration for S3 user '{username}':\n" + tabulate(
        df, headers="keys", tablefmt="grid"
    )
    if print_output:
        print(message)
    return message


def pretty_print_recompute(recompute_dict, update=False, print_output=True):
    objects_col = "Synced/Total" if update else "Objects"
    df = pd.DataFrame(
        columns=[
            "Bucket alias",
            "S3 user",
            "Bucket path",
            "Modality",
            "Model",
            objects_col,
            "Size (MB)",
            "Duration (s)",
        ]
    )

    for path_to_bucket, entry in recompute_dict.items():
        bucket_alias = entry["bucket_alias"]
        modality = entry["modality_name"]
        duration = entry["duration"]
        num_objects = entry["nb_objects"]
        s3_user = entry["s3_username"]
        model = entry["model"].model_name
        size = entry["total_size_MB"]

        df.loc[len(df)] = [
            bucket_alias,
            s3_user,
            path_to_bucket,
            modality,
            model,
            num_objects,
            size,
            duration,
        ]

    df.sort_values(by=["S3 user", "Bucket alias"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    title = "\nUpdate Summary:" if update else "\nRecompute Summary:"

    # If no rows to display, avoid tabulate alignment issues and print a friendly message
    if df.empty:
        message = f"{title}\n(no buckets to process)"
        if print_output:
            print(message)
        return message

    colalign = ("right", "left", "left", "left", "left", "right", "right", "right")
    table = tabulate(df, headers="keys", tablefmt="grid", colalign=colalign)
    message = f"{title}\n{table}"
    if print_output:
        print(message)
    return message


def pretty_print_check(check_dict, print_output=True):
    df = pd.DataFrame(
        columns=[
            "Bucket alias",
            "S3 user",
            "Bucket path",
            "Modality",
            "Model",
            "Objects",
            "Size (MB)",
            "Matches",
            "Mismatches",
            "Not Found",
            "Errors",
        ]
    )

    for path_to_bucket, entry in check_dict.items():
        bucket_alias = entry["bucket_alias"]
        s3_user = entry["s3_username"]
        modality = entry["modality_name"]
        model_name = (
            entry["model"].model_name
            if hasattr(entry["model"], "model_name")
            else str(entry["model"])
        )
        num_objects = entry["nb_objects"]
        size_mb = entry["total_size_MB"]
        nb_match = entry["matches"]
        nb_mismatch = entry["Mismatches"]
        nb_not_found = entry["Not_found"]
        nb_error = entry["Errors"]

        df.loc[len(df)] = [
            bucket_alias,
            s3_user,
            path_to_bucket,
            modality,
            model_name,
            num_objects,
            size_mb,
            nb_match,
            nb_mismatch,
            nb_not_found,
            nb_error,
        ]

    df.sort_values(by=["S3 user", "Bucket alias"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    message = (
        "\nObject Check Summary:"
        + "\n"
        + tabulate(df, headers="keys", tablefmt="grid")
    )
    if print_output:
        print(message)
    return message
