"""Client-safe helpers for UMRS database naming and SQL metadata shape."""


def create_db_name(sql_username, xp_name, username=None):
    """Build the per-experience Postgres database name.

    The platform ``username`` is appended so two different users that pick the
    same ``(xp_name, sql_username)`` pair get separate, isolated databases —
    mirroring how S3 buckets are already namespaced per user. ``username`` is
    optional: when omitted the legacy (un-namespaced) name is returned, which
    callers use to locate and migrate pre-existing databases.
    """
    db_name = xp_name + "_" + sql_username
    if username:
        db_name = db_name + "_" + username
    db_name = db_name.replace(" ", "_").lower()
    return db_name


def metadata_dict(sql_cfg):
    metadata = {}
    for i, column in enumerate(sql_cfg["meta_data"]):
        metadata[f"key{i}"] = column

    return metadata
