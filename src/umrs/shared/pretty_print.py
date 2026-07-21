"""Plain-text table rendering for server summaries.

All tables share the aligned-column style used by ``umrs bucket list``::

    [Title]
      Header1  Header2
      -----------------
      value1   value2
"""

import sys

from umrs.shared.utils import utc_to_local


def format_table(title, headers, rows, empty_message="(no entries)"):
    """Render rows as an aligned-column table (``umrs bucket list`` style)."""
    lines = [f"[{title}]"] if title else []
    if not rows:
        lines.append(f"  {empty_message}")
        return "\n".join(lines)
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(header), *(len(row[i]) for row in str_rows))
        for i, header in enumerate(headers)
    ]
    lines.append(
        "  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()
    )
    lines.append("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in str_rows:
        lines.append(
            "  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip()
        )
    return "\n".join(lines)


def _truncate(value, max_len=40):
    s = str(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _metadata_str(metadata):
    return " ; ".join(f"{m['name']} ({m['type']})" for m in metadata.values())


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
        message = format_table("Server status", headers, rows)
    else:
        message = "Server is idle"
    if print_output:
        print(message)
    return message


def pretty_print_cfg(cfg_dict, print_output=True):
    headers = [
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

    rows = []
    for key, entry in cfg_dict.buckets_cfg.items():
        rows.append(
            [
                _truncate(entry.bucket_alias, 20),
                _truncate(entry.s3_username, 12),
                _truncate(key, 50),
                _truncate(entry.xp_name, 15),
                _truncate(entry.modality_name, 10),
                _truncate(entry.db_name, 30),
                _truncate(entry.table_name, 12),
                _truncate(entry.sql_username, 12),
                _truncate(entry.model.model_name, 20),
                entry.store_embeddings_flag,
                _metadata_str(entry.metadata),
            ]
        )

    rows.sort(key=lambda r: (r[1], r[0]))
    message = "\n" + format_table("Server configuration", headers, rows)
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
    headers = [
        "Bucket alias",
        "Experience",
        "Modality",
        "Database",
        "SQL table",
        "Model",
        "Embeds?",
        "Metadata",
    ]

    # Accept dict, BucketsCfg (has .items()), or object with .buckets_cfg attr
    if isinstance(cfg_dict, dict):
        buckets_cfg = cfg_dict
    elif hasattr(cfg_dict, "items"):
        buckets_cfg = cfg_dict
    else:
        buckets_cfg = cfg_dict.buckets_cfg

    rows = []
    for _key, entry in buckets_cfg.items():
        if entry.s3_username != username:
            continue
        rows.append(
            [
                _truncate(entry.bucket_alias, 20),
                _truncate(entry.xp_name, 15),
                _truncate(entry.modality_name, 10),
                _truncate(entry.db_name, 30),
                _truncate(entry.table_name, 12),
                _truncate(_model_name_for_bucket(entry), 20),
                entry.store_embeddings_flag,
                _metadata_str(entry.metadata),
            ]
        )

    rows.sort(key=lambda r: r[0])
    if not rows:
        message = f"\nNo configuration found for user '{username}'."
    else:
        message = "\n" + format_table(
            f"Configuration for {username}", headers, rows
        )
    if print_output:
        print(message)
    return message


def pretty_print_user_cfg(cfg_dict, print_output=True):
    headers = [
        "Modality",
        "Bucket alias",
        "Experience",
        "Bucket path",
        "DB name",
        "SQL user",
    ]
    username = None
    rows = []
    for key, entry in cfg_dict.items():
        user = entry.username
        if not username:
            username = user
        elif user != username:
            sys.exit(f"user mismatch between {username} and {user}")

        rows.append(
            [
                entry.modality_name,
                key,
                entry.xp_name,
                entry.path_to_bucket,
                entry.db_name,
                entry.sql_username,
            ]
        )

    rows.sort(key=lambda r: r[1])
    message = format_table(f"Configuration for S3 user {username}", headers, rows)
    if print_output:
        print(message)
    return message


def pretty_print_recompute(recompute_dict, update=False, print_output=True):
    objects_col = "Synced/Total" if update else "Objects"
    headers = [
        "Bucket alias",
        "S3 user",
        "Bucket path",
        "Modality",
        "Model",
        objects_col,
        "Size (MB)",
        "Duration (s)",
    ]

    rows = []
    for path_to_bucket, entry in recompute_dict.items():
        rows.append(
            [
                entry["bucket_alias"],
                entry["s3_username"],
                path_to_bucket,
                entry["modality_name"],
                entry["model"].model_name,
                entry["nb_objects"],
                entry["total_size_MB"],
                entry["duration"],
            ]
        )

    rows.sort(key=lambda r: (r[1], r[0]))
    title = "Update Summary" if update else "Recompute Summary"
    message = "\n" + format_table(
        title, headers, rows, empty_message="(no buckets to process)"
    )
    if print_output:
        print(message)
    return message


def pretty_print_check(check_dict, print_output=True):
    headers = [
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

    rows = []
    for path_to_bucket, entry in check_dict.items():
        model_name = (
            entry["model"].model_name
            if hasattr(entry["model"], "model_name")
            else str(entry["model"])
        )
        rows.append(
            [
                entry["bucket_alias"],
                entry["s3_username"],
                path_to_bucket,
                entry["modality_name"],
                model_name,
                entry["nb_objects"],
                entry["total_size_MB"],
                entry["matches"],
                entry["Mismatches"],
                entry["Not_found"],
                entry["Errors"],
            ]
        )

    rows.sort(key=lambda r: (r[1], r[0]))
    message = "\n" + format_table("Object Check Summary", headers, rows)
    if print_output:
        print(message)
    return message
