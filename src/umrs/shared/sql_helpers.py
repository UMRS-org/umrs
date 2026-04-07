"""Client-safe helpers for UMRS database naming and SQL metadata shape."""


def create_db_name(sql_username, xp_name):
    db_name = xp_name + "_" + sql_username
    db_name = db_name.replace(" ", "_").lower()
    return db_name


def metadata_dict(sql_cfg):
    metadata = {}
    for i, column in enumerate(sql_cfg["meta_data"]):
        metadata[f"key{i}"] = column

    return metadata